"""Voice → orchestrator bridge powered by ElevenLabs Speech-to-Text.

This module exposes a single class, :class:`ElevenLabsInteraction`, that
spins up an async ``aiohttp`` HTTP server, serves a mobile-friendly
press-and-hold microphone page, transcribes incoming audio with
ElevenLabs Scribe, and queues the transcripts for downstream processing.

Two operating modes
-------------------

**Standalone** (this file run directly, e.g. ``python -m interactive.elevenlabs``)
    The bridge owns the orchestrator. Each recording is transcribed,
    printed to stdout, then dispatched to :func:`orchestrator._handle_prompt`
    via an internal worker. Prompts execute strictly one at a time.

**Library** (``ElevenLabsInteraction(...)`` instantiated by another module
such as ``orchestrator.py``)
    The bridge only handles audio capture + transcription. It then
    *notifies the consumer* that a new task is available — either by
    pushing the transcript onto an ``asyncio.Queue`` that the consumer
    drains via :meth:`get_next_prompt` / :meth:`transcripts`, or by
    invoking an ``on_transcript`` callback. The consumer (e.g. the
    orchestrator) is responsible for feeding the prompt to its LLM.

Usage
-----
    # Standalone:
    python -m interactive.elevenlabs              # dry-run
    python interactive/elevenlabs.py --execute    # drive the real arm

    # Library (called from orchestrator.py):
    bridge = ElevenLabsInteraction(autorun_orchestrator=False)
    async for prompt in bridge.transcripts():
        ...  # feed prompt to the LLM / executor

Environment
-----------
    ELEVENLABS_API_KEY   required for transcription. Without it, the
                         server still runs but only accepts ``/text``
                         (typed) prompts as a graceful fallback.
    OPENAI_API_KEY       optional. If set and ELEVENLABS_API_KEY is
                         missing, Whisper-1 is used as a fallback STT.

HTTPS / mobile
--------------
Browsers refuse ``getUserMedia()`` on non-secure origins **except for
localhost**. To use this from a phone on the LAN, expose the port via a
tunnel (e.g. ``ngrok http 8000`` or ``cloudflared``) or pass ``--cert``
and ``--key`` PEM paths to enable HTTPS directly.

NOTE: this file is named ``elevenlabs.py`` for clarity, but it does NOT
``import elevenlabs`` — it talks to the REST API directly through aiohttp,
so there is no shadow-import conflict with the official SDK.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import ssl
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

import aiohttp
from aiohttp import web

TranscriptCallback = Callable[[str], Union[Awaitable[None], None]]

# Make the sibling orchestrator module importable when running either as
# ``python interactive/elevenlabs.py`` or ``python -m interactive.elevenlabs``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator import (  # noqa: E402  (path-injection above)
    DEFAULT_POSES_YAML,
    DEFAULT_SKILLS_YAML,
    ExecContext,
    _handle_prompt,
    load_poses,
    load_skills,
)


ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_DEFAULT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")


# ──────────────────────────────────────────────────────────────────────
# HTML page served at "/"
# ──────────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
<title>Robot Voice Console</title>
<style>
  :root {
    --bg: #0e0f12;
    --panel: #16181d;
    --text: #e7e9ee;
    --muted: #8b91a0;
    --accent: #4f7cff;
    --accent-hot: #ff4f6b;
    --ok: #3ddc97;
    --warn: #ffb454;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-tap-highlight-color: transparent;
    overscroll-behavior: none;
    user-select: none;
  }
  .wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: space-between;
    height: 100%;
    padding: env(safe-area-inset-top, 16px) 16px env(safe-area-inset-bottom, 16px);
    max-width: 520px; margin: 0 auto;
  }
  header {
    width: 100%;
    padding: 8px 4px 4px;
    text-align: center;
  }
  header h1 {
    margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.2px;
  }
  header p {
    margin: 4px 0 0; font-size: 13px; color: var(--muted);
  }
  .status {
    width: 100%; background: var(--panel);
    border: 1px solid #23262d; border-radius: 12px;
    padding: 12px 14px;
    font-size: 14px; line-height: 1.4;
    min-height: 44px;
    overflow-wrap: anywhere;
  }
  .status .label { color: var(--muted); font-size: 12px; }
  .row { display: flex; align-items: center; gap: 8px; }
  .pill {
    display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
    background: #23262d; color: var(--muted);
  }
  .pill.ok    { background: rgba(61,220,151,0.15);   color: var(--ok); }
  .pill.warn  { background: rgba(255,180,84,0.15);   color: var(--warn); }
  .pill.busy  { background: rgba(79,124,255,0.18);   color: var(--accent); }
  .pill.err   { background: rgba(255,79,107,0.18);   color: var(--accent-hot); }

  .mic {
    width: 200px; height: 200px; border-radius: 50%;
    border: 4px solid #2a2f3a; background: linear-gradient(180deg, #1a1d24, #11131a);
    color: var(--text); font-size: 16px; font-weight: 600;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; touch-action: none;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45),
                inset 0 0 0 1px rgba(255,255,255,0.03);
    transition: transform 90ms ease, border-color 120ms ease, background 120ms ease;
  }
  .mic .glyph {
    width: 56px; height: 56px;
    background: var(--accent);
    -webkit-mask: var(--mic-mask) center/contain no-repeat;
            mask: var(--mic-mask) center/contain no-repeat;
    transition: background 120ms ease;
  }
  .mic.active {
    border-color: var(--accent-hot);
    background: radial-gradient(circle at 50% 40%,
                rgba(255,79,107,0.22), #11131a 60%);
    transform: scale(0.97);
  }
  .mic.active .glyph { background: var(--accent-hot); }
  .mic.disabled { opacity: 0.45; pointer-events: none; }

  .hint { font-size: 12px; color: var(--muted); margin-top: 12px; }
  .err  { color: var(--accent-hot); font-size: 13px; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Robot Voice Console</h1>
      <p>Hold the mic to talk. Release to send.</p>
    </header>

    <div class="status" id="status">
      <div class="row">
        <span class="label">Server:</span>
        <span id="server-pill" class="pill">…</span>
        <span id="queue-pill" class="pill" style="margin-left:auto"></span>
      </div>
      <div style="margin-top: 10px">
        <div class="label">Last transcript</div>
        <div id="transcript">—</div>
      </div>
      <div id="error" class="err" style="margin-top: 8px"></div>
    </div>

    <button id="mic" class="mic" aria-label="Push to talk">
      <span class="glyph"></span>
    </button>

    <div class="hint">
      ElevenLabs STT → orchestrator. One task runs at a time.
    </div>
  </div>

<script>
  // SVG mic glyph as a CSS mask (keeps the icon recolorable).
  document.documentElement.style.setProperty(
    '--mic-mask',
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='black' d='M12 14.5a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5.5a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.58A7 7 0 0 0 19 11.5h-2Z'/></svg>\")"
  );

  const micBtn = document.getElementById('mic');
  const transcriptEl = document.getElementById('transcript');
  const errEl = document.getElementById('error');
  const serverPill = document.getElementById('server-pill');
  const queuePill = document.getElementById('queue-pill');

  let mediaRecorder = null;
  let chunks = [];
  let stream = null;
  let recording = false;

  function setError(msg) {
    errEl.textContent = msg || '';
  }

  function setMicState(state /* idle | recording | uploading | disabled */) {
    micBtn.classList.toggle('active', state === 'recording');
    micBtn.classList.toggle('disabled', state === 'disabled' || state === 'uploading');
  }

  async function ensureStream() {
    if (stream) return stream;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('getUserMedia not available. Use HTTPS or localhost.');
    }
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return stream;
  }

  function pickMimeType() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      ''
    ];
    for (const t of candidates) {
      if (!t || (window.MediaRecorder && MediaRecorder.isTypeSupported(t))) return t;
    }
    return '';
  }

  async function startRecording() {
    if (recording) return;
    setError('');
    try {
      const s = await ensureStream();
      const mime = pickMimeType();
      mediaRecorder = mime ? new MediaRecorder(s, { mimeType: mime })
                           : new MediaRecorder(s);
      chunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = onStop;
      mediaRecorder.start();
      recording = true;
      setMicState('recording');
    } catch (e) {
      setError(String(e.message || e));
      setMicState('idle');
    }
  }

  function stopRecording() {
    if (!recording || !mediaRecorder) return;
    recording = false;
    setMicState('uploading');
    try { mediaRecorder.stop(); } catch (_) {}
  }

  async function onStop() {
    const type = (mediaRecorder && mediaRecorder.mimeType) || 'audio/webm';
    const blob = new Blob(chunks, { type });
    chunks = [];
    if (blob.size < 200) {
      setError('Recording too short.');
      setMicState('idle');
      return;
    }
    try {
      const fd = new FormData();
      const ext = type.includes('mp4') ? 'mp4' :
                  type.includes('ogg') ? 'ogg' : 'webm';
      fd.append('audio', blob, 'clip.' + ext);
      const res = await fetch('/audio', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || ('HTTP ' + res.status));
      } else {
        transcriptEl.textContent = data.transcript || '(empty)';
      }
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setMicState('idle');
      pollStatus();
    }
  }

  // Press-and-hold semantics across mouse + touch + pointer.
  function bind(el) {
    const down = (ev) => { ev.preventDefault(); startRecording(); };
    const up   = (ev) => { ev.preventDefault(); stopRecording(); };
    if ('PointerEvent' in window) {
      el.addEventListener('pointerdown', down);
      el.addEventListener('pointerup', up);
      el.addEventListener('pointerleave', up);
      el.addEventListener('pointercancel', up);
    } else {
      el.addEventListener('mousedown', down);
      el.addEventListener('mouseup', up);
      el.addEventListener('mouseleave', up);
      el.addEventListener('touchstart', down, { passive: false });
      el.addEventListener('touchend', up,   { passive: false });
      el.addEventListener('touchcancel', up);
    }
    // Also support spacebar push-to-talk on desktop.
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !e.repeat) { e.preventDefault(); startRecording(); }
    });
    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') { e.preventDefault(); stopRecording(); }
    });
  }
  bind(micBtn);

  async function pollStatus() {
    try {
      const r = await fetch('/status');
      const s = await r.json();
      if (s.stt_ready) {
        serverPill.className = 'pill ok';
        serverPill.textContent = s.stt_provider || 'ready';
      } else {
        serverPill.className = 'pill warn';
        serverPill.textContent = 'no STT key';
      }
      const q = (s.queued || 0) + (s.running ? 1 : 0);
      if (s.running) {
        queuePill.className = 'pill busy';
        queuePill.textContent = q > 1 ? ('busy · +' + (q - 1) + ' queued') : 'busy';
      } else if (q > 0) {
        queuePill.className = 'pill busy';
        queuePill.textContent = q + ' queued';
      } else {
        queuePill.className = 'pill ok';
        queuePill.textContent = 'idle';
      }
    } catch (_) {
      serverPill.className = 'pill err';
      serverPill.textContent = 'offline';
    }
  }
  pollStatus();
  setInterval(pollStatus, 1500);
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────


class ElevenLabsInteraction:
    """Async voice-driven front door for the orchestrator.

    Parameters
    ----------
    host, port :
        Where to bind the HTTP server.
    api_key :
        ElevenLabs API key. Falls back to ``$ELEVENLABS_API_KEY`` env var.
    skills_path, poses_path :
        Forwarded to the orchestrator loaders. Only used when
        ``autorun_orchestrator`` is True (i.e. the bridge runs the
        orchestrator itself).
    dry_run :
        If True, the orchestrator only prints planned moves; if False, it
        connects to and drives the arm.
    stt_model :
        ElevenLabs STT model id (default ``scribe_v1``).
    ssl_context :
        Optional SSL context for HTTPS (required for non-localhost mic
        access from browsers).
    autorun_orchestrator :
        When True, an internal worker dequeues each transcript and runs
        :func:`orchestrator._handle_prompt`, one at a time.
        When False (default for library use), no worker is started — the
        consumer is expected to drain the queue via :meth:`get_next_prompt`
        / :meth:`transcripts`, or to register an ``on_transcript`` callback.
    on_transcript :
        Optional sync or async callable invoked with each transcript
        immediately after it has been queued. Use this as a *notification*
        hook ("a new task is available") when integrating with another
        async program such as ``orchestrator.py``.
    print_transcripts :
        When True, every transcript is printed to stdout (defaults on for
        standalone CLI mode).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        api_key: Optional[str] = "sk_fcda42bc1f01dfa2236e22562170f8e89a4fc47973b28a2a",
        skills_path: Optional[Path] = None,
        poses_path: Optional[Path] = None,
        dry_run: bool = True,
        stt_model: str = ELEVENLABS_DEFAULT_MODEL,
        ssl_context: Optional[ssl.SSLContext] = None,
        *,
        autorun_orchestrator: bool = False,
        on_transcript: Optional[TranscriptCallback] = None,
        print_transcripts: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.stt_model = stt_model
        self.ssl_context = ssl_context

        self.autorun_orchestrator = autorun_orchestrator
        self.on_transcript = on_transcript
        self.print_transcripts = print_transcripts

        # The orchestrator handle is only needed when this object is
        # responsible for running prompts itself.
        self.skills: Optional[dict] = None
        self.ctx: Optional[ExecContext] = None
        if autorun_orchestrator:
            self.skills = load_skills(skills_path or DEFAULT_SKILLS_YAML)
            self.ctx = ExecContext(
                dry_run=dry_run,
                poses=load_poses(poses_path or DEFAULT_POSES_YAML),
            )

        # One-at-a-time prompt pipeline. Always present so library
        # consumers can pull from it via get_next_prompt() / transcripts().
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._running_prompt: Optional[str] = None
        # The orchestrator is synchronous and may block (sleeps, network,
        # serial bus). Keep it off the event-loop thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orch")

        self._app = web.Application(client_max_size=32 * 1024 * 1024)
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/status", self._status)
        self._app.router.add_post("/audio", self._audio)
        self._app.router.add_post("/text", self._text)
        self._app.on_shutdown.append(self._on_shutdown)

        # aiohttp ClientSession is created lazily on the running loop.
        self._http: Optional[aiohttp.ClientSession] = None
        self._worker_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _index(self, _request: web.Request) -> web.Response:
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def _status(self, _request: web.Request) -> web.Response:
        provider = (
            "elevenlabs" if self.api_key
            else "whisper" if self.openai_key
            else None
        )
        return web.json_response(
            {
                "stt_ready": provider is not None,
                "stt_provider": provider,
                "queued": self._queue.qsize(),
                "running": self._running_prompt is not None,
                "running_prompt": self._running_prompt,
            }
        )

    async def _audio(self, request: web.Request) -> web.Response:
        try:
            reader = await request.multipart()
        except Exception as exc:
            return web.json_response({"error": f"bad upload: {exc}"}, status=400)

        audio_bytes = b""
        filename = "clip.webm"
        content_type = "audio/webm"
        async for part in reader:
            if part.name == "audio":
                filename = part.filename or filename
                content_type = part.headers.get("Content-Type", content_type)
                audio_bytes = await part.read(decode=False)
                break
        if not audio_bytes:
            return web.json_response({"error": "no audio in request"}, status=400)

        try:
            transcript = await self._transcribe(audio_bytes, filename, content_type)
        except Exception as exc:
            return web.json_response({"error": f"STT failed: {exc}"}, status=502)

        transcript = (transcript or "").strip()
        if transcript:
            await self._publish(transcript, source="voice")
        else:
            print("[voice] empty transcript — ignored")

        return web.json_response(
            {
                "transcript": transcript,
                "queued": self._queue.qsize(),
                "running": self._running_prompt is not None,
            }
        )

    async def _text(self, request: web.Request) -> web.Response:
        """Manual / fallback prompt entry (handy for debugging without a mic)."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "missing prompt"}, status=400)
        await self._publish(prompt, source="text")
        return web.json_response({"queued": self._queue.qsize()})

    # ------------------------------------------------------------------
    # Publishing transcripts: queue + optional print + notify consumer
    # ------------------------------------------------------------------

    async def _publish(self, transcript: str, source: str) -> None:
        """Queue a new prompt and notify any registered consumer.

        The queue is the source of truth — both the optional internal
        worker and external consumers (``get_next_prompt`` / async
        iterator) read from it. The optional ``on_transcript`` callback
        is fired-and-forgotten as a notification so the consumer can,
        e.g. wake up its scheduler or log the new task.
        """
        await self._queue.put(transcript)

        if self.print_transcripts:
            print(f"\n[transcript] ({source}) {transcript}")
        else:
            print(f"[{source}] queued prompt: {transcript!r}")

        if self.on_transcript is not None:
            try:
                result = self.on_transcript(transcript)
                if inspect.isawaitable(result):
                    asyncio.create_task(result)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                print(f"[notify] on_transcript raised: {exc}")

    # ------------------------------------------------------------------
    # Speech-to-text (ElevenLabs primary, Whisper fallback)
    # ------------------------------------------------------------------

    async def _transcribe(
        self, audio_bytes: bytes, filename: str, content_type: str,
    ) -> str:
        if self.api_key:
            return await self._transcribe_elevenlabs(audio_bytes, filename, content_type)
        if self.openai_key:
            return await self._transcribe_whisper(audio_bytes, filename, content_type)
        raise RuntimeError(
            "No STT provider configured. Set ELEVENLABS_API_KEY or OPENAI_API_KEY."
        )

    async def _transcribe_elevenlabs(
        self, audio_bytes: bytes, filename: str, content_type: str,
    ) -> str:
        assert self.api_key
        assert self._http is not None
        form = aiohttp.FormData()
        form.add_field(
            "file", audio_bytes, filename=filename, content_type=content_type,
        )
        form.add_field("model_id", self.stt_model)
        async with self._http.post(
            ELEVENLABS_STT_URL,
            data=form,
            headers={"xi-api-key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"elevenlabs {resp.status}: {body[:200]}")
            data = await resp.json()
        # The scribe response shape is {"text": "...", "language_code": "...", ...}
        return str(data.get("text") or data.get("transcript") or "")

    async def _transcribe_whisper(
        self, audio_bytes: bytes, filename: str, content_type: str,
    ) -> str:
        """Fallback STT path using OpenAI Whisper (sync SDK in a thread)."""
        suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".webm"
        loop = asyncio.get_running_loop()

        def _run() -> str:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(f"openai package not installed: {exc}") from exc
            client = OpenAI()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    tr = client.audio.transcriptions.create(model="whisper-1", file=f)
                return (getattr(tr, "text", "") or "").strip()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return await loop.run_in_executor(self._executor, _run)

    # ------------------------------------------------------------------
    # Consumer-facing API (library mode)
    # ------------------------------------------------------------------

    async def get_next_prompt(self) -> str:
        """Wait for and return the next transcribed prompt.

        Intended for library-mode consumers (e.g. ``orchestrator.py``)
        that want to drive prompt processing themselves. Pair with
        :meth:`mark_running` / :meth:`mark_done` if you want the ``/status``
        endpoint to reflect the consumer's busy state.
        """
        return await self._queue.get()

    async def transcripts(self) -> AsyncIterator[str]:
        """Async-iterate over every transcript as it arrives."""
        while True:
            yield await self._queue.get()

    def mark_running(self, prompt: Optional[str]) -> None:
        """Tell the bridge a consumer has started executing ``prompt``."""
        self._running_prompt = prompt

    def mark_done(self) -> None:
        """Tell the bridge the consumer has finished the current prompt."""
        self._running_prompt = None
        try:
            self._queue.task_done()
        except ValueError:
            pass

    @property
    def queue(self) -> "asyncio.Queue[str]":
        """Direct access to the underlying transcript queue."""
        return self._queue

    # ------------------------------------------------------------------
    # Orchestrator worker — drains the queue serially (autorun mode)
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        assert self.skills is not None and self.ctx is not None, (
            "internal worker requires autorun_orchestrator=True"
        )
        loop = asyncio.get_running_loop()
        while True:
            prompt = await self._queue.get()
            self.mark_running(prompt)
            print(f"\n[orch] running prompt: {prompt!r}  (queue={self._queue.qsize()})")
            try:
                # _handle_prompt is synchronous and may take many seconds
                # (robot moves, VLM call, TTS, …). Push it onto the
                # dedicated single-thread executor so we keep guarantees:
                #   * the event loop / web server stays responsive,
                #   * only one prompt is ever in flight at a time.
                await loop.run_in_executor(
                    self._executor,
                    _handle_prompt,
                    prompt,
                    self.skills,
                    self.ctx,
                    False,  # plan_only
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                print(f"[orch] error while running {prompt!r}: {exc}")
            finally:
                self.mark_done()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_shutdown(self, _app: web.Application) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._http is not None:
            await self._http.close()
        if self.ctx is not None:
            try:
                self.ctx.close_robot()
            except Exception:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def serve(self) -> None:
        """Run the server forever. Cancel the task to stop."""
        self._http = aiohttp.ClientSession()
        if self.autorun_orchestrator:
            self._worker_task = asyncio.create_task(
                self._worker(), name="orchestrator-worker",
            )

        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(
            runner, self.host, self.port, ssl_context=self.ssl_context,
        )
        await site.start()

        scheme = "https" if self.ssl_context else "http"
        shown_host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        if not self.api_key and not self.openai_key:
            print(
                "[warn] No STT API key set — /audio will reject. "
                "Export ELEVENLABS_API_KEY (or OPENAI_API_KEY) to enable."
            )
        mode = "standalone" if self.autorun_orchestrator else "library"
        print(f"[server] listening on {scheme}://{shown_host}:{self.port}/  ({mode} mode)")
        if self.host in ("0.0.0.0", "::"):
            print(
                "[server] LAN clients can reach it at "
                f"{scheme}://<this-machine-ip>:{self.port}/  (mic needs HTTPS off-localhost)"
            )

        try:
            await asyncio.Event().wait()  # block until cancelled
        finally:
            await runner.cleanup()

    def run(self) -> None:
        """Synchronous entry point — runs the asyncio app forever."""
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:
            print("\n[server] shutting down …")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0",
                   help="Interface to bind (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=8000,
                   help="Port to bind (default 8000).")
    p.add_argument("--execute", action="store_true",
                   help="Drive the real arm. Default is dry-run.")
    p.add_argument("--skills", type=Path, default=None,
                   help="Override skills.yaml path.")
    p.add_argument("--poses", type=Path, default=None,
                   help="Override arm_poses.yaml path.")
    p.add_argument("--cert", type=Path, default=None,
                   help="TLS cert PEM (required for HTTPS).")
    p.add_argument("--key", type=Path, default=None,
                   help="TLS key PEM (required for HTTPS).")
    p.add_argument("--stt-model", default=ELEVENLABS_DEFAULT_MODEL,
                   help=f"ElevenLabs STT model id (default {ELEVENLABS_DEFAULT_MODEL}).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    ssl_ctx: Optional[ssl.SSLContext] = None
    if args.cert and args.key:
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(certfile=str(args.cert), keyfile=str(args.key))
    elif bool(args.cert) ^ bool(args.key):
        print("--cert and --key must be supplied together.", file=sys.stderr)
        return 2

    bridge = ElevenLabsInteraction(
        host=args.host,
        port=args.port,
        skills_path=args.skills,
        poses_path=args.poses,
        dry_run=not args.execute,
        stt_model=args.stt_model,
        ssl_context=ssl_ctx,
        # Standalone CLI defaults: drive the orchestrator ourselves and
        # print every transcript to stdout as it comes in.
        autorun_orchestrator=True,
        print_transcripts=True,
    )
    bridge.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
