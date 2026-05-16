from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import unquote
from uuid import uuid4

from app.config import Settings
from app.orchestrator.schemas import SkillCall, SkillResult, UserRequest
from app.pipeline import AlfredRuntime
from test_elevenlabs_api import load_env_value, transcribe_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Alfred Orchestrator Browser Demo</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #202124; color: #e8eaed; }
    header { padding: 14px 18px; background: #303134; display: flex; align-items: center; gap: 12px; }
    button, select { font-size: 15px; padding: 8px 12px; border-radius: 6px; border: 1px solid #5f6368; }
    button { background: #1a73e8; color: white; cursor: pointer; }
    button:disabled { background: #5f6368; cursor: not-allowed; }
    #stopBtn { background: #d93025; }
    label { display: flex; align-items: center; gap: 6px; }
    main { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 76px); box-sizing: border-box; }
    section { background: #303134; border: 1px solid #5f6368; border-radius: 10px; padding: 10px; min-width: 0; display: flex; flex-direction: column; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    video, img, canvas { width: 100%; background: #111; border-radius: 8px; object-fit: contain; }
    video, img { flex: 1; max-height: 70vh; }
    canvas { display: none; }
    pre { flex: 1; margin: 0; white-space: pre-wrap; overflow: auto; background: #111; color: #e8eaed; padding: 10px; border-radius: 8px; }
    .status { color: #fbbc04; }
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
  <main>
    <section>
      <h2>Live Browser Camera</h2>
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas"></canvas>
    </section>
    <section>
      <h2>Captured Image Sent To VLM</h2>
      <img id="capturedImage" alt="Captured frame will appear here" />
    </section>
    <section>
      <h2>Alfred Output / Debug</h2>
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
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const capturedImage = document.getElementById("capturedImage");

    let mediaStream = null;
    let recorder = null;
    let chunks = [];

    function log(message) {
      console.log(message);
      logEl.textContent += message + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setStatus(message) {
      statusEl.textContent = message;
      log(message);
    }

    async function initDevices() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not support getUserMedia. Try Chrome on macOS.");
      }
      if (!window.MediaRecorder) {
        throw new Error("This browser does not support MediaRecorder. Try Chrome on macOS.");
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      await populateDeviceLists();
      await startPreview();
      setStatus("Ready. Choose devices if needed, then click Start Recording.");
    }

    async function populateDeviceLists() {
      const devices = await navigator.mediaDevices.enumerateDevices();
      cameraSelect.innerHTML = "";
      micSelect.innerHTML = "";
      for (const device of devices) {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `${device.kind} ${device.deviceId.slice(0, 6)}`;
        if (device.kind === "videoinput") cameraSelect.appendChild(option);
        if (device.kind === "audioinput") micSelect.appendChild(option);
      }
    }

    async function startPreview() {
      if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
      const constraints = {
        video: cameraSelect.value ? { deviceId: { exact: cameraSelect.value } } : true,
        audio: micSelect.value ? { deviceId: { exact: micSelect.value } } : true,
      };
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = mediaStream;
    }

    function captureFrame() {
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, width, height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      capturedImage.src = dataUrl;
      log(`Captured browser frame for VLM: ${width}x${height}`);
      return dataUrl;
    }

    function blobToDataUrl(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    startBtn.onclick = async () => {
      chunks = [];
      await startPreview();
      const audioTracks = mediaStream.getAudioTracks();
      const audioOnly = new MediaStream(audioTracks);
      recorder = new MediaRecorder(audioOnly);
      recorder.ondataavailable = event => {
        if (event.data && event.data.size > 0) chunks.push(event.data);
      };
      recorder.start();
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus("Recording... speak your request, then click Stop Recording + Process.");
    };

    stopBtn.onclick = async () => {
      stopBtn.disabled = true;
      setStatus("Stopping recording...");
      recorder.stop();
      recorder.onstop = async () => {
        try {
          const audioBlob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          const audioDataUrl = await blobToDataUrl(audioBlob);
          const imageDataUrl = captureFrame();
          setStatus("Sending audio + image to Alfred server...");
          const response = await fetch("/api/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audio_data_url: audioDataUrl,
              image_data_url: imageDataUrl,
              speak: speakToggle.checked,
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
          setStatus("Done. Click Start Recording to run again.");
        } catch (error) {
          log("Error: " + error.message);
          setStatus("Failed. See debug output.");
        } finally {
          startBtn.disabled = false;
        }
      };
    };

    cameraSelect.onchange = startPreview;
    micSelect.onchange = startPreview;

    initDevices().catch(error => {
      log("Browser media initialization failed: " + error.message);
      setStatus("Camera/mic permission failed.");
    });
  </script>
</body>
</html>
"""


class BrowserDemoHandler(BaseHTTPRequestHandler):
    server_version = "AlfredBrowserDemo/0.1"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(HTML)
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
        print(f"[browser-demo] {self.address_string()} - {format % args}", flush=True)

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


def process_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    load_env_into_process()
    api_key = load_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")

    request_id = str(uuid4())
    run_dir = Settings.load().new_run_dir("browser_demo")
    run_dir.mkdir(parents=True, exist_ok=True)

    print_header("Browser Demo Request")
    print_step("1", "Saving browser audio and image")
    audio_path = save_data_url(payload["audio_data_url"], run_dir / "browser_audio")
    image_path = save_data_url(payload["image_data_url"], run_dir / "browser_capture")
    print(f"Saved audio: {audio_path}")
    print(f"Saved captured image sent to VLM: {image_path}")

    print_step("2", "Sending audio to ElevenLabs speech-to-text")
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or "scribe_v1"
    transcript_payload = transcribe_audio(
        api_key=api_key,
        model_id=stt_model,
        audio_path=audio_path,
        timeout=120,
    )
    transcript = str(transcript_payload.get("text") or "").strip()
    print(f"Transcript: {transcript}")
    if not transcript:
        raise RuntimeError(f"STT returned no transcript: {transcript_payload}")

    print_step("3", "Creating Alfred runtime")
    runtime = AlfredRuntime(Settings.load(), run_dir=run_dir)
    runtime.request_id = request_id
    runtime.logger.request_id = request_id
    print(f"Run artifacts directory: {runtime.run_dir}")

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
    print("  1. browser_capture_image", {"image_path": str(image_path)})
    print("  2. describe_image_with_vlm", {"image_path": str(image_path)})

    print_step("6", "Executing VLM skill")
    capture_result = SkillResult(
        skill_name="capture_wrist_camera_image",
        status="success",
        output={"image_path": str(image_path), "source": "browser"},
    )
    vlm_result = runtime.executor.execute(
        SkillCall(
            skill_name="describe_image_with_vlm",
            arguments={
                "image_path": str(image_path),
                "question": "What objects are visible on the desk?",
                "user_text": transcript,
            },
        )
    )
    skill_results = [capture_result, vlm_result]
    for result in skill_results:
        print(f"Skill: {result.skill_name}")
        print(f"  Status: {result.status}")
        if result.error:
            print(f"  Error: {result.error}")
        if result.output:
            print(f"  Output: {result.output}")

    print_step("7", "Checking completion")
    completion = runtime.orchestrator.evaluate_completion(request, skill_results)
    print(f"Task complete: {completion.task_complete}")
    print(f"Reason: {completion.reason}")
    print(f"Next action: {completion.next_action}")

    print_step("8", "Generating final response")
    final_response = runtime.orchestrator.generate_final_answer(request, skill_results)
    print(f"Alfred answer: {final_response.answer_text}")
    print(f"Response confidence: {final_response.confidence}")

    if bool(payload.get("speak", True)):
        print_step("9", "Speaking answer with ElevenLabs TTS")
        speak_result = runtime.executor.execute(
            SkillCall(skill_name="speak", arguments={"text": final_response.answer_text})
        )
        print(f"TTS status: {speak_result.status}")
        if speak_result.error:
            print(f"TTS error: {speak_result.error}")
        if speak_result.output:
            print(f"TTS output: {speak_result.output}")
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
        "image_path": str(image_path),
        "run_dir": str(run_dir),
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
    if mime in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    return ".bin"


def load_env_into_process() -> None:
    for env_path in (PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def print_header(title: str) -> None:
    print()
    print("=" * len(title), flush=True)
    print(title, flush=True)
    print("=" * len(title), flush=True)


def print_step(label: str, title: str) -> None:
    print()
    print(f"[{label}] {title}", flush=True)
    print("-" * (len(str(label)) + len(title) + 3), flush=True)


def run_server(host: str, port: int, open_browser: bool) -> None:
    load_env_into_process()
    server = ThreadingHTTPServer((host, port), BrowserDemoHandler)
    url = f"http://{host}:{port}"
    print(f"Alfred browser demo running at {url}")
    print("The browser handles live camera/mic preview; Python handles STT, VLM, logs, and TTS.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser-based Alfred interactive demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
