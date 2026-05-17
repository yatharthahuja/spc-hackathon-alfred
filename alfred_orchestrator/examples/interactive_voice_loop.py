"""Alfred interactive voice loop, phone-accessible.

This script serves a browser UI that uses a phone or laptop microphone for
voice input while the Python server drives the host-side wrist camera. Desk
inspection requests run the real Alfred pipeline: move the robot to the
overlook pose, capture a wrist-camera image, send it to the VLM, and return
the model output in the debug panel.

The server listens over HTTPS on ``0.0.0.0`` by default, with an auto-generated
self-signed certificate, so the page can be opened from a phone on the LAN and
the **phone's microphone** will be the audio source — browsers refuse
``getUserMedia`` on non-secure origins, which is the only reason the page
couldn't be used from a phone before.

The TTS response is also streamed back as bytes inside the JSON response so
the phone plays it through the phone's speakers; this sidesteps the Ubuntu
host audio-playback path (``aplay`` cannot decode MP3) without modifying any
other file in the repo.

Legacy macOS keystroke / Tk flows are preserved behind ``--terminal-only``
and ``--tk-gui``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from collections import OrderedDict

from app.config import Settings
from app.hardware.resources import HardwareContext
from app.interfaces.camera import detect_cameras
from app.orchestrator.schemas import SkillCall, UserRequest
from app.pipeline import AlfredRuntime
from test_elevenlabs_api import load_env_value, synthesize_speech, transcribe_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────
# TTS cache — served via /api/tts/<request_id>.mp3 so the browser's
# <audio> element can use a normal URL (cleaner than base64-in-JSON,
# and it lets the user tap the controls to replay).
# ──────────────────────────────────────────────────────────────────────

_TTS_LOCK = threading.Lock()
_TTS_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_TTS_MAX_ENTRIES = 32


def _store_tts(request_id: str, mp3: bytes) -> None:
    with _TTS_LOCK:
        _TTS_CACHE[request_id] = mp3
        _TTS_CACHE.move_to_end(request_id)
        while len(_TTS_CACHE) > _TTS_MAX_ENTRIES:
            _TTS_CACHE.popitem(last=False)


def _get_tts(request_id: str) -> Optional[bytes]:
    with _TTS_LOCK:
        return _TTS_CACHE.get(request_id)


# ──────────────────────────────────────────────────────────────────────
# Browser UI — same 3-column layout as browser_interactive_demo.py.
# Changes vs. the original:
#   * <meta viewport> + tiny mobile tweaks for phone layout.
#   * Camera dropdown is populated from the SERVER (host's cameras via
#     cv2.VideoCapture), not from the browser's navigator.enumerateDevices.
#     Microphone dropdown still lists the BROWSER's audio inputs (so the
#     phone's mic is used).
#   * The live camera panel is an <img> bound to an MJPEG stream from the
#     host (/api/stream?camera_id=N) — works on iOS Safari without any
#     browser camera permission.
#   * getUserMedia only requests {audio: true} since the camera is no
#     longer sourced from the browser.
#   * A hidden <audio> element + JS to auto-play TTS bytes returned by
#     the server (so playback works on Ubuntu, where `aplay` can't
#     decode MP3 — the phone speaker plays it instead).
#   * A small banner shown when getUserMedia is blocked (insecure origin),
#     pointing the user at /cert.
# ──────────────────────────────────────────────────────────────────────


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <title>Alfred Orchestrator Browser Demo</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #202124; color: #e8eaed; }
    header { padding: 14px 18px; background: #303134; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    button, select { font-size: 15px; padding: 8px 12px; border-radius: 6px; border: 1px solid #5f6368; }
    button { background: #1a73e8; color: white; cursor: pointer; }
    button:disabled { background: #5f6368; cursor: not-allowed; }
    #stopBtn { background: #d93025; }
    label { display: flex; align-items: center; gap: 6px; }
    main { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 76px); box-sizing: border-box; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; height: auto; }
      header { gap: 8px; }
    }
    section { background: #303134; border: 1px solid #5f6368; border-radius: 10px; padding: 10px; min-width: 0; display: flex; flex-direction: column; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    img { width: 100%; background: #111; border-radius: 8px; object-fit: contain; flex: 1; max-height: 70vh; }
    pre { flex: 1; margin: 0; white-space: pre-wrap; overflow: auto; background: #111; color: #e8eaed; padding: 10px; border-radius: 8px; }
    .status { color: #fbbc04; }
    .banner { display: none; background: rgba(217,48,37,0.12); border: 1px solid rgba(217,48,37,0.4); color: #ffb4ad; padding: 10px 14px; border-radius: 8px; margin: 0 12px; font-size: 14px; line-height: 1.5; }
    .banner.show { display: block; }
    .banner code { background: rgba(0,0,0,0.4); padding: 1px 5px; border-radius: 4px; }
    .banner a { color: #ffd0cb; }
    .tts-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .tts-row audio { flex: 1; height: 36px; }
    .tts-row .tts-label { font-size: 12px; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.08em; }
  </style>
</head>
<body>
  <header>
    <button id="startBtn">Start Recording</button>
    <button id="stopBtn" disabled>Stop Recording + Process</button>
    <label><input id="speakToggle" type="checkbox" checked /> Speak response</label>
    <label>Camera <select id="cameraSelect"></select></label>
    <label>Microphone <select id="micSelect"></select></label>
    <span id="status" class="status">Ready</span>
  </header>
  <div id="insecureBanner" class="banner"></div>
  <main>
    <section>
      <h2>Live Wrist Camera</h2>
      <img id="livePreview" alt="Live host camera will appear here" />
    </section>
    <section>
      <h2>Captured Wrist Image Sent To VLM</h2>
      <img id="capturedImage" alt="Captured frame will appear here" />
    </section>
    <section>
      <h2>Alfred Output / Debug</h2>
      <div class="tts-row">
        <span class="tts-label">Alfred speech</span>
        <audio id="ttsPlayer" controls preload="auto"></audio>
      </div>
      <pre id="log"></pre>
    </section>
  </main>

  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const speakToggle = document.getElementById("speakToggle");
    const cameraSelect = document.getElementById("cameraSelect");
    const micSelect = document.getElementById("micSelect");
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const livePreview = document.getElementById("livePreview");
    const capturedImage = document.getElementById("capturedImage");
    const ttsPlayer = document.getElementById("ttsPlayer");
    const insecureBanner = document.getElementById("insecureBanner");

    let mediaStream = null;
    let recorder = null;
    let chunks = [];
    let recordingStartedAt = 0;

    function log(message) {
      console.log(message);
      logEl.textContent += message + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setStatus(message) {
      statusEl.textContent = message;
      log(message);
    }

    function isMicAvailable() {
      // Browsers expose getUserMedia only on secure origins (https://, or
      // http://localhost). Phones on the LAN therefore need https://.
      return !!(window.isSecureContext
                && navigator.mediaDevices
                && navigator.mediaDevices.getUserMedia
                && window.MediaRecorder);
    }

    function explainInsecureContext() {
      const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
                    || (navigator.userAgent.includes("Mac") && "ontouchend" in document);
      const onHttp = location.protocol === "http:";
      let html;
      if (onHttp) {
        html = "You're on <code>http://</code>, so the browser refuses to expose the microphone. "
             + "Restart the server without <code>--http</code> and reload over <code>https://</code>.";
      } else if (isIOS) {
        html = "iPhone Safari blocks the microphone on untrusted-certificate origins. "
             + "Install + trust the self-signed cert: "
             + "<a href='/cert' download='alfred-bridge.crt'>tap to download</a>, then "
             + "<em>Settings → General → VPN &amp; Device Management → Install profile</em>, then "
             + "<em>Settings → General → About → Certificate Trust Settings</em> and enable full trust "
             + "for &ldquo;alfred-bridge&rdquo;. Or use <code>ngrok http " + location.port + "</code>.";
      } else {
        html = "The browser blocks the microphone on this origin. Accept the certificate "
             + "(<a href='/cert' download='alfred-bridge.crt'>download cert</a>) or use a tunnel "
             + "like <code>ngrok http " + location.port + "</code> for a real HTTPS URL.";
      }
      insecureBanner.innerHTML = html;
      insecureBanner.classList.add("show");
    }

    async function populateHostCameras() {
      cameraSelect.innerHTML = "";
      try {
        const res = await fetch("/api/cameras");
        const data = await res.json();
        const cams = data.cameras || [];
        if (!cams.length) {
          const opt = document.createElement("option");
          opt.value = ""; opt.textContent = "(no host cameras found)";
          cameraSelect.appendChild(opt);
          cameraSelect.disabled = true;
          livePreview.removeAttribute("src");
          log("Host has no available cameras.");
          return;
        }
        for (const cam of cams) {
          const opt = document.createElement("option");
          opt.value = String(cam.camera_id);
          opt.textContent = `Host camera ${cam.camera_id} (${cam.width}x${cam.height})`;
          cameraSelect.appendChild(opt);
        }
        cameraSelect.disabled = false;
        startHostPreview();
      } catch (e) {
        log("Failed to list host cameras: " + (e.message || e));
      }
    }

    function startHostPreview() {
      const id = cameraSelect.value;
      if (id === "" || id === undefined) {
        livePreview.removeAttribute("src");
        return;
      }
      // Cache-bust so changing camera_id always forces a fresh MJPEG stream.
      livePreview.src = `/api/stream?camera_id=${encodeURIComponent(id)}&_=${Date.now()}`;
    }

    function stopHostPreview() {
      livePreview.removeAttribute("src");
    }

    async function populateMics() {
      micSelect.innerHTML = "";
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        let count = 0;
        for (const device of devices) {
          if (device.kind !== "audioinput") continue;
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = device.label || `microphone ${device.deviceId.slice(0, 6)}`;
          micSelect.appendChild(option);
          count++;
        }
        if (!count) {
          const opt = document.createElement("option");
          opt.value = ""; opt.textContent = "(default microphone)";
          micSelect.appendChild(opt);
        }
      } catch (e) {
        log("Could not enumerate microphones: " + (e.message || e));
      }
    }

    async function ensureMicStream() {
      if (mediaStream && mediaStream.getAudioTracks().length) return mediaStream;
      if (!isMicAvailable()) {
        explainInsecureContext();
        throw new Error("microphone blocked by browser security policy");
      }
      const constraints = {
        audio: micSelect.value ? { deviceId: { exact: micSelect.value } } : true,
      };
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      return mediaStream;
    }

    async function initDevices() {
      await populateHostCameras();
      if (!isMicAvailable()) {
        explainInsecureContext();
        return;
      }
      try {
        // Prompt for mic permission up front so device labels are visible
        // in the dropdown and the user only sees the permission dialog once.
        await ensureMicStream();
        await populateMics();
        setStatus("Ready. Choose camera / mic if needed, then click Start Recording.");
      } catch (e) {
        log("Microphone permission denied or unavailable: " + (e.message || e));
        setStatus("Microphone permission failed.");
      }
    }

    function blobToDataUrl(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    function pickMimeType() {
      const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        ""
      ];
      for (const t of candidates) {
        if (!t || (window.MediaRecorder && MediaRecorder.isTypeSupported(t))) return t;
      }
      return "";
    }

    // iOS Safari only allows <audio>.play() when the call originates from a
    // user gesture. After an `await fetch(...)`, that gesture is gone, so
    // autoplay of the TTS response would silently fail. We "unlock" the
    // audio element inside the synchronous part of a click handler by
    // calling .load() and a no-op play()/pause() — once unlocked, future
    // play() calls (after fetches resolve) are allowed.
    let audioUnlocked = false;
    function unlockAudioElement() {
      if (audioUnlocked) return;
      try {
        ttsPlayer.muted = true;
        const p = ttsPlayer.play();
        if (p && typeof p.then === "function") {
          p.then(() => { ttsPlayer.pause(); ttsPlayer.muted = false; audioUnlocked = true; })
           .catch(() => { ttsPlayer.muted = false; });
        } else {
          ttsPlayer.pause();
          ttsPlayer.muted = false;
          audioUnlocked = true;
        }
      } catch (_) { ttsPlayer.muted = false; }
    }

    startBtn.onclick = async () => {
      try {
        unlockAudioElement();
        chunks = [];
        const stream = await ensureMicStream();
        const audioOnly = new MediaStream(stream.getAudioTracks());
        const mime = pickMimeType();
        recorder = mime ? new MediaRecorder(audioOnly, { mimeType: mime })
                        : new MediaRecorder(audioOnly);
        recorder.ondataavailable = event => {
          if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        recorder.start();
        recordingStartedAt = Date.now();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        setStatus("Recording... speak your request, then click Stop Recording + Process.");
      } catch (e) {
        log("Could not start recording: " + (e.message || e));
        setStatus("Failed to start.");
      }
    };

    stopBtn.onclick = async () => {
      stopBtn.disabled = true;
      setStatus("Stopping recording...");
      unlockAudioElement();
      recorder.onstop = async () => {
        try {
          const audioBlob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          const recordedSeconds = recordingStartedAt ? ((Date.now() - recordingStartedAt) / 1000) : 0;
          log(`Recorded audio: ${audioBlob.size} bytes over ${recordedSeconds.toFixed(1)} seconds.`);
          if (audioBlob.size < 4096) {
            throw new Error(
              "Browser microphone recorded almost no audio. Refresh the page, choose the working mic, "
              + "then record for a few seconds while speaking clearly."
            );
          }
          const audioDataUrl = await blobToDataUrl(audioBlob);
          const cameraId = cameraSelect.value === "" ? null : Number(cameraSelect.value);
          stopHostPreview();
          setStatus("Sending audio to Alfred server...");
          const response = await fetch("/api/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audio_data_url: audioDataUrl,
              camera_id: cameraId,
              speak: speakToggle.checked,
              require_robot_move: true,
            }),
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || response.statusText);
          log("");
          log("Transcript: " + payload.transcript);
          log("Intent: " + payload.intent);
          log("Task complete: " + payload.task_complete);
          log("Alfred: " + payload.answer_text);
          log("Run artifacts: " + payload.run_dir);
          if (payload.robot_move) {
            log("Robot move: " + JSON.stringify(payload.robot_move, null, 2));
          }
          if (payload.vlm_output) {
            log("VLM output: " + JSON.stringify(payload.vlm_output, null, 2));
          }
          if (payload.captured_image_data_url) {
            capturedImage.src = payload.captured_image_data_url;
          }
          if (payload.tts_audio_url) {
            // Cache-bust so the <audio> always refetches a fresh response.
            ttsPlayer.src = payload.tts_audio_url + "?_=" + Date.now();
            ttsPlayer.load();
            log("Alfred speech: " + payload.tts_audio_url
                + " (" + (payload.tts_audio_bytes || "?") + " bytes)");
            try {
              await ttsPlayer.play();
              log("Alfred speech: playing.");
            } catch (e) {
              log("Autoplay blocked — tap the play control above to hear Alfred. ("
                  + (e.message || e) + ")");
            }
          } else if (payload.tts_error) {
            log("TTS failed on server: " + payload.tts_error);
          } else if (payload.speak === false) {
            log("Speak toggle was off — no TTS produced.");
          }
          setStatus("Done. Click Start Recording to run again.");
        } catch (error) {
          log("Error: " + error.message);
          setStatus("Failed. See debug output.");
        } finally {
          startHostPreview();
          startBtn.disabled = false;
        }
      };
      recorder.stop();
    };

    cameraSelect.onchange = startHostPreview;
    micSelect.onchange = async () => {
      // Drop existing audio tracks so the next ensureMicStream picks the new device.
      if (mediaStream) {
        mediaStream.getAudioTracks().forEach(t => t.stop());
        mediaStream = null;
      }
    };

    initDevices().catch(error => {
      log("Init failed: " + error.message);
      setStatus("Initialization failed.");
    });
  </script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# LAN discovery + self-signed cert (so phones can use getUserMedia).
# Adapted from spc-hackathon-alfred/interactive/elevenlabs.py
# ──────────────────────────────────────────────────────────────────────


def _get_lan_ips() -> List[str]:
    """Best-effort detection of this machine's LAN IPv4 addresses."""
    ips: set[str] = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def _ensure_self_signed_cert(extra_ips: List[str]) -> Tuple[Path, Path]:
    """Return (cert.pem, key.pem), generating a fresh self-signed pair if needed.

    Cached under ``$TMPDIR/alfred_bridge_cert/`` so reruns share a fingerprint
    (the phone only has to accept the warning once).
    """
    cache_dir = Path(tempfile.gettempdir()) / "alfred_bridge_cert"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cert = cache_dir / "cert.pem"
    key = cache_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key

    if not shutil.which("openssl"):
        raise RuntimeError(
            "openssl not found in PATH; cannot auto-generate a self-signed TLS cert. "
            "Install openssl, or pass --cert/--key with your own."
        )

    san_entries = ["DNS:localhost", "IP:127.0.0.1"]
    for ip in extra_ips:
        if ip and ip != "127.0.0.1":
            san_entries.append(f"IP:{ip}")
    san = ",".join(san_entries)

    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert),
        "-days", "3650",
        "-subj", "/CN=alfred-bridge",
        "-addext", f"subjectAltName={san}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(cert),
                "-days", "3650",
                "-subj", "/CN=alfred-bridge",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    return cert, key


# ──────────────────────────────────────────────────────────────────────
# Host camera streaming (so the browser can list + preview the host's
# cameras instead of the client's). Camera access delegates to the shared
# HardwareContext so preview and skill capture use one VideoCapture owner.
# ──────────────────────────────────────────────────────────────────────


_HARDWARE_CONTEXT: Optional[HardwareContext] = None


def get_hardware_context() -> HardwareContext:
    global _HARDWARE_CONTEXT
    if _HARDWARE_CONTEXT is None:
        settings = Settings.load()
        _HARDWARE_CONTEXT = HardwareContext.from_settings(settings)
        _HARDWARE_CONTEXT.connect()
    return _HARDWARE_CONTEXT


class CameraStreamer:
    """Compatibility wrapper around the server-lifetime camera resource."""

    def grab_jpeg(self, camera_id: int, quality: int = 80) -> bytes:
        return get_hardware_context().camera.grab_jpeg(camera_id, quality=quality)

    def capture_to_file(self, camera_id: int, path: Path, quality: int = 92) -> Path:
        jpeg = self.grab_jpeg(camera_id, quality=quality)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return path

    def release(self) -> None:
        get_hardware_context().camera.close()


CAMERAS = CameraStreamer()
_DETECTED_CAMERAS_LOCK = threading.Lock()
_DETECTED_CAMERAS_CACHE: Optional[List[Dict[str, Any]]] = None


def list_host_cameras(refresh: bool = False) -> List[Dict[str, Any]]:
    """Probe a few /dev/video* indices for working cameras (cached).

    Detection is slow because each probe opens + reads from a V4L2 device,
    so the result is memoised. The streamer is released before probing
    because cv2.VideoCapture often refuses to share devices with the
    detection helper.
    """
    global _DETECTED_CAMERAS_CACHE
    with _DETECTED_CAMERAS_LOCK:
        if _DETECTED_CAMERAS_CACHE is not None and not refresh:
            return _DETECTED_CAMERAS_CACHE
        CAMERAS.release()
        cams = detect_cameras(max_index=6)
        _DETECTED_CAMERAS_CACHE = [
            {"camera_id": c.camera_id, "width": c.width, "height": c.height, "fps": c.fps}
            for c in cams
        ]
        return _DETECTED_CAMERAS_CACHE


# ──────────────────────────────────────────────────────────────────────
# HTTP handler — adds /api/cameras and /api/stream for host-side camera.
# ──────────────────────────────────────────────────────────────────────


class BrowserDemoHandler(BaseHTTPRequestHandler):
    server_version = "AlfredBrowserDemo/0.3"
    cert_path: Optional[Path] = None  # set by run_server when HTTPS is on

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self._send_html(HTML)
            return
        if path == "/cert" and self.cert_path is not None and self.cert_path.exists():
            body = self.cert_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-x509-ca-cert")
            self.send_header("Content-Disposition", 'attachment; filename="alfred-bridge.crt"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/cameras":
            refresh = query.get("refresh", ["0"])[0] in {"1", "true", "yes"}
            try:
                cams = list_host_cameras(refresh=refresh)
                self._send_json({"cameras": cams})
            except Exception as exc:
                self._send_json({"error": str(exc), "cameras": []}, status=500)
            return
        if path == "/api/stream":
            camera_id = self._int_query(query, "camera_id", 0)
            self._stream_mjpeg(camera_id)
            return
        if path.startswith("/api/tts/") and path.endswith(".mp3"):
            request_id = path[len("/api/tts/"):-len(".mp3")]
            mp3 = _get_tts(request_id)
            if mp3 is None:
                self.send_error(404, "TTS expired")
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(mp3)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(mp3)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/process":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = process_request(payload)
            self._send_json(response)
        except Exception as exc:
            print(f"Browser demo request failed: {exc}", flush=True)
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        # MJPEG streams produce one log line per frame at high FPS — silence them.
        if "/api/stream" in (self.requestline or ""):
            return
        print(f"[browser-demo] {self.address_string()} - {format % args}", flush=True)

    @staticmethod
    def _int_query(query: Dict[str, List[str]], key: str, default: int) -> int:
        try:
            return int(query.get(key, [str(default)])[0])
        except (TypeError, ValueError):
            return default

    def _stream_mjpeg(self, camera_id: int) -> None:
        """Multipart MJPEG stream — works in <img src=...> on every browser."""
        boundary = "alfredframe"
        try:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.end_headers()
        except Exception:
            return
        try:
            while True:
                try:
                    jpeg = CAMERAS.grab_jpeg(camera_id)
                except Exception as exc:
                    print(f"[stream] camera {camera_id} error: {exc}", flush=True)
                    return
                try:
                    self.wfile.write(b"--" + boundary.encode("ascii") + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                # ~10 fps preview; light enough for a phone-over-LAN connection.
                time.sleep(0.1)
        except Exception as exc:
            print(f"[stream] unexpected error on camera {camera_id}: {exc}", flush=True)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ──────────────────────────────────────────────────────────────────────
# Server-side pipeline — same shape as browser_interactive_demo.process_request,
# but the TTS step is inlined so we can stream the MP3 back to the phone
# instead of relying on the host's audio stack (broken on Ubuntu w/ only aplay).
# ──────────────────────────────────────────────────────────────────────


def process_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    load_env_into_process()
    api_key = load_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")

    settings = Settings.load()
    request_id = str(uuid4())
    run_dir = settings.new_run_dir("browser_demo")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_camera_id = payload.get("camera_id")
    if raw_camera_id is None:
        camera_id = settings.camera_id
    else:
        try:
            camera_id = int(raw_camera_id)
        except (TypeError, ValueError):
            camera_id = settings.camera_id
    raw_require_robot_move = payload.get("require_robot_move", True)
    require_robot_move = (
        raw_require_robot_move
        if isinstance(raw_require_robot_move, bool)
        else str(raw_require_robot_move).strip().lower() not in {"0", "false", "no", "off"}
    )

    print_header("Browser Demo Request")
    print(f"Request id: {request_id}")
    print(f"Camera id: {camera_id}")
    print(f"Speak requested: {bool(payload.get('speak', True))}")
    print(f"Require robot move: {require_robot_move}")
    print(f"Payload keys: {sorted(payload.keys())}")
    print(f"Uploaded audio data URL chars: {len(str(payload.get('audio_data_url', '')))}")

    print_step("1", "Saving browser audio")
    audio_path = save_data_url(payload["audio_data_url"], run_dir / "browser_audio")
    audio_size = audio_path.stat().st_size
    print(f"Voice received and saved: {audio_path} ({audio_size} bytes)")
    if audio_size < 4096:
        raise RuntimeError(
            "Browser microphone upload was too small to contain usable speech "
            f"({audio_size} bytes). Refresh the page, choose the working microphone, "
            "and record for a few seconds while speaking clearly."
        )
    hardware = get_hardware_context()
    hardware.camera.set_camera_id(camera_id)
    print("Using shared hardware context for robot motion and wrist-camera capture.")

    print_step("2", "Sending audio to ElevenLabs speech-to-text")
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or settings.elevenlabs_stt_model
    print(f"Voice sent to STT model: {stt_model}")
    print(f"STT audio path: {audio_path}")
    transcript_payload = transcribe_audio(
        api_key=api_key,
        model_id=stt_model,
        audio_path=audio_path,
        timeout=120,
    )
    print_json_block("STT raw output", transcript_payload)
    transcript = str(transcript_payload.get("text") or "").strip()
    print(f"Transcript: {transcript}")
    if not transcript:
        raise RuntimeError(
            "STT returned no transcript. The uploaded browser audio may be silent or from "
            f"the wrong microphone. Saved audio: {audio_path} ({audio_size} bytes). "
            f"ElevenLabs payload: {transcript_payload}"
        )

    print_step("3", "Creating Alfred runtime")
    runtime = AlfredRuntime(settings, run_dir=run_dir, hardware_context=hardware)
    runtime.request_id = request_id
    runtime.logger.request_id = request_id
    runtime.orchestrator.camera_id = camera_id
    print(f"Run artifacts directory: {runtime.run_dir}")
    print(f"Registered skills: {runtime.router.names()}")

    print_step("4", "Classifying intent")
    request = UserRequest(request_id=request_id, input_type="voice", raw_text=transcript)
    runtime.logger.write_text("transcript.txt", request.raw_text)
    runtime.logger.log("user_request", "success", output_data=request.model_dump(mode="json"))
    intent_result = runtime.orchestrator.classify_intent(request.raw_text)
    print(f"Intent: {intent_result.intent.value}")
    print(f"Confidence: {intent_result.confidence}")
    print(f"Reason: {intent_result.reason}")

    print_step("5", "Creating skill plan")
    plan = runtime.orchestrator.create_plan(request, intent_result)
    print(f"Goal: {plan.goal}")
    print_json_block("Full plan", plan.model_dump(mode="json"))
    for index, call in enumerate(plan.skill_calls, start=1):
        print(f"Selected skill {index}: {call.skill_name}")
        print_json_block("Selected skill arguments", call.arguments)

    print_step("6", "Executing skills")
    skill_results = []
    outputs_by_skill: Dict[str, Dict[str, Any]] = {}
    for call in plan.skill_calls:
        call_arguments = dict(call.arguments)
        if call.skill_name == "capture_wrist_camera_image":
            call_arguments["move_to_overlook"] = True
            call_arguments["require_robot_move"] = require_robot_move
        resolved_call = SkillCall(
            skill_name=call.skill_name,
            arguments=runtime.executor._resolve_arguments(call_arguments, outputs_by_skill),
        )
        print(f"Executing skill: {resolved_call.skill_name}")
        print_json_block("Resolved skill arguments", resolved_call.arguments)
        result = runtime.executor.execute(resolved_call)
        skill_results.append(result)
        outputs_by_skill[result.skill_name] = result.output
        print(f"Skill: {result.skill_name}")
        print(f"  Status: {result.status}")
        if result.error:
            print(f"  Error: {result.error}")
        if result.output:
            print(f"  Output: {result.output}")
        if result.status == "error":
            break

    capture_result = next(
        (result for result in skill_results if result.skill_name == "capture_wrist_camera_image"),
        None,
    )
    raw_image_path = (
        capture_result.output.get("image_path")
        if capture_result and capture_result.status == "success"
        else None
    )
    image_path = Path(str(raw_image_path)) if raw_image_path else None
    captured_image_data_url = None
    if image_path is not None and image_path.exists():
        captured_image_data_url = (
            "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
        )
    robot_move = None
    if capture_result is not None:
        robot_move = {
            key: capture_result.output.get(key)
            for key in (
                "robot_pose_name",
                "robot_move_attempted",
                "robot_moved",
                "robot_unavailable",
                "robot_error",
            )
            if key in capture_result.output
        }
    vlm_result = next(
        (result for result in skill_results if result.skill_name == "describe_image_with_vlm"),
        None,
    )
    vlm_output = (
        vlm_result.output
        if vlm_result is not None and vlm_result.status == "success"
        else None
    )

    print_step("7", "Checking completion")
    completion = runtime.orchestrator.evaluate_completion(request, skill_results)
    print(f"Task complete: {completion.task_complete}")
    print(f"Reason: {completion.reason}")
    print(f"Next action: {completion.next_action}")

    print_step("8", "Generating final response")
    final_response = runtime.orchestrator.generate_final_answer(request, skill_results)
    print_json_block("Final response object", final_response.model_dump(mode="json"))
    print(f"Alfred answer: {final_response.answer_text}")
    print(f"Response confidence: {final_response.confidence}")
    runtime.record_task_history(request, plan, skill_results, completion, final_response)

    tts_audio_url: Optional[str] = None
    tts_audio_bytes: Optional[int] = None
    tts_error: Optional[str] = None
    speak_requested = bool(payload.get("speak", True))
    if speak_requested and final_response.answer_text.strip():
        # Inline TTS so we can stream the MP3 back to the browser (the phone
        # will play it). We do NOT route through the `speak` skill because
        # that goes through app/interfaces/audio.py → aplay on Ubuntu, which
        # cannot decode MP3 and would either silently fail or play noise.
        print_step("9", "Synthesizing TTS for phone playback")
        try:
            mp3_path = run_dir / "alfred_response.mp3"
            print(f"Text sent to TTS: {final_response.answer_text}")
            print(f"TTS voice id: {settings.elevenlabs_voice_id}")
            print(f"TTS model: {settings.elevenlabs_tts_model}")
            print(f"TTS output format: {settings.elevenlabs_output_format}")
            synthesize_speech(
                api_key=api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id=settings.elevenlabs_tts_model,
                output_format=settings.elevenlabs_output_format,
                text=final_response.answer_text,
                output_path=mp3_path,
                timeout=120,
            )
            mp3_bytes = mp3_path.read_bytes()
            _store_tts(request_id, mp3_bytes)
            tts_audio_url = f"/api/tts/{request_id}.mp3"
            tts_audio_bytes = len(mp3_bytes)
            print(f"TTS bytes: {tts_audio_bytes} → served at {tts_audio_url}")
            print(f"TTS audio file: {mp3_path}")
        except Exception as exc:
            tts_error = str(exc)
            print(f"TTS failed: {exc}")
    else:
        print_step("9", "Skipping text-to-speech")

    print_step("Done", "Browser demo request complete")
    return {
        "request_id": request_id,
        "transcript": transcript,
        "intent": intent_result.intent.value,
        "task_complete": completion.task_complete,
        "answer_text": final_response.answer_text,
        "confidence": final_response.confidence,
        "image_path": str(image_path) if image_path is not None else None,
        "camera_id": camera_id,
        "require_robot_move": require_robot_move,
        "robot_move": robot_move,
        "vlm_output": vlm_output,
        "skill_results": [result.model_dump(mode="json") for result in skill_results],
        "captured_image_data_url": captured_image_data_url,
        "run_dir": str(run_dir),
        "speak": speak_requested,
        "tts_audio_url": tts_audio_url,
        "tts_audio_bytes": tts_audio_bytes,
        "tts_error": tts_error,
    }


def save_data_url(data_url: str, path_without_suffix: Path) -> Path:
    header, encoded = data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    suffix = mime_to_suffix(mime)
    path = path_without_suffix.with_suffix(suffix)
    path.write_bytes(base64.b64decode(encoded))
    return path


def mime_to_suffix(mime: str) -> str:
    if mime == "image/jpeg":
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime in {"audio/webm", "video/webm"}:
        return ".webm"
    if mime in {"audio/mp4", "video/mp4"}:
        return ".mp4"
    if mime in {"audio/ogg"}:
        return ".ogg"
    if mime in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    return ".bin"


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────


def load_env_into_process() -> None:
    for env_path in (PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if env_path.name == ".env":
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)


def print_header(title: str) -> None:
    print()
    print("=" * len(title), flush=True)
    print(title, flush=True)
    print("=" * len(title), flush=True)


def print_step(label: str, title: str) -> None:
    print()
    print(f"[{label}] {title}", flush=True)
    print("-" * (len(str(label)) + len(title) + 3), flush=True)


def print_json_block(title: str, payload: Any) -> None:
    print(f"{title}:")
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    except TypeError:
        print(repr(payload), flush=True)


# ──────────────────────────────────────────────────────────────────────
# Server entry point — HTTPS on 0.0.0.0 so phones can join.
# ──────────────────────────────────────────────────────────────────────


def run_server(
    host: str,
    port: int,
    use_https: bool,
    cert: Optional[Path],
    key: Optional[Path],
    open_browser: bool,
) -> int:
    global _HARDWARE_CONTEXT
    load_env_into_process()
    settings = Settings.load()
    _HARDWARE_CONTEXT = HardwareContext.from_settings(settings)
    _HARDWARE_CONTEXT.connect()

    server = ThreadingHTTPServer((host, port), BrowserDemoHandler)
    scheme = "http"
    cert_path: Optional[Path] = None
    if use_https:
        if cert and key:
            cert_path = cert
            key_path = key
        else:
            cert_path, key_path = _ensure_self_signed_cert(_get_lan_ips())
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    BrowserDemoHandler.cert_path = cert_path

    lan_ips = _get_lan_ips()
    print(f"[browser-demo] scheme = {scheme}, bound to {host}:{port}")
    print(f"[browser-demo] open on this machine:  {scheme}://localhost:{port}/")
    if host in {"0.0.0.0", "::"}:
        for ip in lan_ips:
            print(f"[browser-demo] open on a phone on LAN: {scheme}://{ip}:{port}/")
    if scheme == "https" and cert_path is not None:
        print(
            "[browser-demo] iPhone? Open /cert in Safari to download the cert, install it under "
            "Settings → General → VPN & Device Management, then enable trust under "
            "Settings → General → About → Certificate Trust Settings."
        )
        print(f"[browser-demo] cert file: {cert_path}")
    if scheme == "http" and host not in {"127.0.0.1", "localhost"}:
        print(
            "[browser-demo] WARNING: serving plain HTTP on the LAN — phones will refuse "
            "camera/mic access. Drop --http or pass --cert/--key for TLS."
        )

    print("The browser handles live camera/mic preview; Python handles STT, VLM, logs, and TTS.")

    if open_browser:
        url = f"{scheme}://localhost:{port}/"
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[browser-demo] shutting down…")
    finally:
        server.server_close()
        if _HARDWARE_CONTEXT is not None:
            _HARDWARE_CONTEXT.close()
            _HARDWARE_CONTEXT = None
    return 0


# ──────────────────────────────────────────────────────────────────────
# Legacy macOS S/E keystroke flow — preserved behind --terminal-only.
# ──────────────────────────────────────────────────────────────────────


def run_terminal_only(args: argparse.Namespace) -> int:
    import termios
    import time
    import tty
    from typing import Iterable

    load_env_into_process()
    api_key = load_env_value("ELEVENLABS_API_KEY")
    settings = Settings.load()
    settings.print_model_configuration()
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or settings.elevenlabs_stt_model
    if not api_key:
        print("ELEVENLABS_API_KEY was not found in the environment, .env, or .env.example.")
        return 1
    if args.camera_id is not None:
        os.environ["ALFRED_CAMERA_ID"] = str(args.camera_id)

    def wait_for_key(valid_keys: "Iterable[str]", prompt: str) -> str:
        valid = {k.lower() for k in valid_keys}
        print(prompt)
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                key = sys.stdin.read(1).lower()
                if key in valid:
                    print(key.upper())
                    return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def require_ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH.")
        return ffmpeg

    def start_recording(device: str, output_path: Path) -> "subprocess.Popen[bytes]":
        ffmpeg = require_ffmpeg()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", device,
            "-ac", "1", "-ar", "16000", str(output_path),
        ]
        return subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def stop_recording(process: "subprocess.Popen[bytes]") -> None:
        if process.stdin:
            try:
                process.stdin.write(b"q")
                process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            _, stderr = process.communicate(timeout=5)
        if process.returncode not in (0, 255):
            raise RuntimeError(
                f"ffmpeg recording failed: {stderr.decode('utf-8', errors='replace').strip()}"
            )

    print_header("Alfred Interactive Voice Loop (terminal mode)")
    print("Controls: S start, E stop, Q quit before recording.")
    print(f"Audio device: {args.device}")
    print(f"Camera id: {os.getenv('ALFRED_CAMERA_ID', '0')}")
    print(f"Audio output: {args.audio_output}")

    key = wait_for_key({"s", "q"}, prompt="Waiting for S to start recording...")
    if key == "q":
        print("Cancelled.")
        return 0

    print_step("1", "Recording started")
    process = start_recording(args.device, args.audio_output)
    started_at = time.monotonic()
    wait_for_key({"e"}, prompt="Recording... press E to stop.")
    stop_recording(process)
    print(f"Recording stopped after {time.monotonic() - started_at:.1f}s.")

    print_step("2", "Speech-to-text")
    transcript_payload = transcribe_audio(
        api_key=api_key, model_id=stt_model, audio_path=args.audio_output, timeout=args.timeout,
    )
    transcript = str(transcript_payload.get("text") or "").strip()
    print(f"Transcript: {transcript}")
    if not transcript:
        return 1

    with AlfredRuntime(settings) as runtime:
        request = UserRequest(request_id=runtime.request_id, input_type="voice", raw_text=transcript)
        runtime.logger.write_text("transcript.txt", request.raw_text)
        runtime.logger.log("user_request", "success", output_data=request.model_dump(mode="json"))
        intent_result = runtime.orchestrator.classify_intent(request.raw_text)
        plan = runtime.orchestrator.create_plan(request, intent_result)
        skill_results = runtime.executor.execute_plan(plan)
        completion = runtime.orchestrator.evaluate_completion(request, skill_results)
        response = runtime.orchestrator.generate_final_answer(request, skill_results)
        runtime.record_task_history(request, plan, skill_results, completion, response)
        print(f"Alfred answer: {response.answer_text}")

        if not args.no_speak:
            runtime.executor.execute(
                SkillCall(skill_name="speak", arguments={"text": response.answer_text})
            )
        return 0 if response.task_complete else 1


def list_devices() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg was not found on PATH.")
        return 1
    result = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        text=True, capture_output=True, check=False,
    )
    print(result.stderr or result.stdout)
    return 0


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Alfred interactive voice loop. By default serves the existing 3-column browser "
            "UI over HTTPS on the LAN, so a phone can use its microphone + camera to drive Alfred."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host (default 0.0.0.0 for LAN access).")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000).")
    parser.add_argument("--http", action="store_true", help="Disable HTTPS. Phones will be unable to use the mic.")
    parser.add_argument("--cert", type=Path, help="Path to TLS cert PEM (instead of auto-generated self-signed).")
    parser.add_argument("--key", type=Path, help="Path to TLS key PEM (instead of auto-generated self-signed).")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open localhost in a browser.")

    parser.add_argument("--camera-id", type=int, help="Override wrist camera index used by the Alfred pipeline.")

    parser.add_argument("--terminal-only", action="store_true", help="Legacy macOS S/E keystroke + ffmpeg flow.")
    parser.add_argument("--tk-gui", action="store_true", help="Legacy Tk GUI flow.")
    parser.add_argument("--list-devices", action="store_true", help="List AVFoundation audio devices (macOS).")

    # Legacy terminal-only flags.
    parser.add_argument("--device", default=":2", help='AVFoundation audio device for --terminal-only, e.g. ":2".')
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "interactive" / "voice_command.wav",
        help="Where --terminal-only writes its raw recording.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-speak", action="store_true", help="Skip TTS in --terminal-only mode.")

    args = parser.parse_args()

    if args.list_devices:
        return list_devices()

    if args.camera_id is not None:
        os.environ["ALFRED_CAMERA_ID"] = str(args.camera_id)

    if args.terminal_only:
        return run_terminal_only(args)

    if args.tk_gui:
        from gui_interactive_demo import run_gui_demo

        return run_gui_demo(args)

    if bool(args.cert) ^ bool(args.key):
        print("--cert and --key must be provided together.", file=sys.stderr)
        return 2

    return run_server(
        host=args.host,
        port=args.port,
        use_https=not args.http,
        cert=args.cert,
        key=args.key,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
