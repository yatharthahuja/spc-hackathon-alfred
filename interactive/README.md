# `interactive/` — Voice front door for the reBot Arm

This package gives the orchestrator a press-and-hold microphone UI that
works from any phone or laptop browser. Audio is recorded in the browser,
streamed to the local machine, transcribed by **ElevenLabs Scribe**
(`scribe_v1`), and turned into prompts for the orchestrator's LLM.

```
                ┌──────────────────────────┐         ┌────────────────────┐
   browser  ──► │  ElevenLabsInteraction   │ ──────► │   orchestrator.py  │
   (mic UI)     │  (aiohttp async server)  │ queue   │   LLM + skills     │
                └──────────────────────────┘         └────────────────────┘
                       ▲                                       │
                       │  ElevenLabs STT                       ▼
                       └────────────────────────────  arm + camera + TTS
```

## Files

| File                  | Purpose                                                      |
|-----------------------|--------------------------------------------------------------|
| `elevenlabs.py`       | `ElevenLabsInteraction` class + `aiohttp` server + mic page. |
| `README.md`           | This file.                                                   |

> The module is named `elevenlabs.py` for clarity, but it does **not**
> `import elevenlabs`. It calls the REST API directly with `aiohttp`, so
> there is no shadow-import conflict with the official Python SDK.

## Requirements

```bash
pip install aiohttp pyyaml numpy
# Optional: pip install openai opencv-python pyttsx3 sounddevice scipy
```

Environment variables:

- `ELEVENLABS_API_KEY` — required for ElevenLabs STT.
- `OPENAI_API_KEY` — optional. Used as a Whisper fallback if no
  ElevenLabs key is set, and required for the orchestrator's LLM
  planner / vision queries.

## Two ways to run it

### 1. Standalone — run `elevenlabs.py` directly

The bridge owns the orchestrator. Each recording is transcribed,
**printed to stdout**, then dispatched to `orchestrator._handle_prompt`
through an internal worker. New recordings are queued while a prior
task is still running, and only one task is ever in flight.

```bash
# From the repo root, dry-run (arm doesn't move):
python -m interactive.elevenlabs

# Or with the real arm + bound to all interfaces, port 8000:
python interactive/elevenlabs.py --execute --host 0.0.0.0 --port 8000
```

After every recording the terminal looks like:

```
[transcript] (voice) pick up the red marker
[orch] running prompt: 'pick up the red marker'  (queue=0)
>>> prompt: pick up the red marker
[llm] skill = pick_marker_by_color  args = {'color': 'red'}
=== Executing skill: pick_marker_by_color ===
…
=== Skill pick_marker_by_color done ===
```

### 2. Library — `orchestrator.py --mic-server`

The orchestrator instantiates the bridge in **library mode**
(`autorun_orchestrator=False`). The bridge only does audio capture and
transcription; it then *notifies the orchestrator* that a new task is
available by:

- pushing the transcript onto its `asyncio.Queue` (the orchestrator
  awaits `bridge.get_next_prompt()`), **and**
- firing the `on_transcript` callback the orchestrator registered.

The orchestrator pulls each prompt off the queue, runs it through its
LLM planner and skill executor, and only after that finishes does it
look for the next prompt — so new tasks always wait for previous ones
to complete.

```bash
# Dry-run; visit http://localhost:8000/ from a browser:
python orchestrator.py --mic-server

# Drive the real arm:
python orchestrator.py --mic-server --execute

# Custom bind:
python orchestrator.py --mic-server --host 0.0.0.0 --port 8080
```

Sample log:

```
[server] listening on http://localhost:8000/  (library mode)
[voice] queued prompt: 'go home'
[orchestrator] notified of new task: 'go home'

>>> prompt: go home
[llm] skill = go_home  args = {}
…
=== Skill go_home done ===
```

## Using the class from your own code

```python
import asyncio
from interactive.elevenlabs import ElevenLabsInteraction

async def main():
    bridge = ElevenLabsInteraction(
        host="0.0.0.0",
        port=8000,
        autorun_orchestrator=False,         # library mode
        on_transcript=lambda t: print("new task:", t),
    )

    async def consumer():
        async for prompt in bridge.transcripts():
            bridge.mark_running(prompt)
            try:
                ...  # send `prompt` to your LLM / scheduler
            finally:
                bridge.mark_done()

    await asyncio.gather(bridge.serve(), consumer())

asyncio.run(main())
```

Public API of `ElevenLabsInteraction`:

| Member                  | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `serve()`               | Async — bind, serve, and (in autorun mode) run the worker.    |
| `run()`                 | Sync wrapper around `serve()` (used by the CLI).              |
| `get_next_prompt()`     | Async — wait for and return the next transcript.              |
| `transcripts()`         | Async iterator yielding each transcript as it arrives.        |
| `mark_running(prompt)`  | Update `/status` so the UI shows "busy".                      |
| `mark_done()`           | Update `/status` so the UI shows the task finished.           |
| `queue`                 | Direct access to the underlying `asyncio.Queue[str]`.         |

Constructor flags worth knowing:

| Parameter                | Default | Effect                                                          |
|--------------------------|---------|-----------------------------------------------------------------|
| `autorun_orchestrator`   | `False` | If True, an internal worker calls `_handle_prompt` per prompt.  |
| `on_transcript`          | `None`  | Sync or async callback fired after each transcript is queued.   |
| `print_transcripts`      | `False` | If True, each transcript is logged to stdout (CLI default).     |
| `dry_run`                | `True`  | Forwarded to `ExecContext` so hardware skills only print.       |
| `ssl_context`            | `None`  | Pass an `ssl.SSLContext` (or `--cert/--key`) for HTTPS.         |

## HTTP endpoints

| Method & path       | Purpose                                                            |
|---------------------|--------------------------------------------------------------------|
| `GET  /`            | Mobile-friendly press-and-hold microphone page.                    |
| `POST /audio`       | Multipart upload (`audio` field). Returns `{transcript, queued}`.  |
| `POST /text`        | JSON `{"prompt": "..."}` — fallback when there's no microphone.    |
| `GET  /status`      | `{stt_ready, stt_provider, queued, running, running_prompt}`.      |

## Browser / mobile notes

`getUserMedia()` only works on **secure origins**. That means:

- `http://localhost:8000/` works on the same machine.
- For a phone on the LAN, either:
  - Tunnel it: `ngrok http 8000` (or `cloudflared tunnel`), and open the
    public HTTPS URL on the phone, or
  - Run the bridge with TLS:
    ```bash
    python interactive/elevenlabs.py --cert cert.pem --key key.pem
    ```
- The page also supports **spacebar push-to-talk** on desktop, in case
  you want to test without a touch device.

## Troubleshooting

- **"no STT key" pill on the page** → export `ELEVENLABS_API_KEY` (or
  `OPENAI_API_KEY` for Whisper fallback) before starting the server.
- **Mic button does nothing on a phone** → you're on plain HTTP from a
  non-localhost origin; use a tunnel or `--cert/--key`.
- **`ModuleNotFoundError: aiohttp`** → `pip install aiohttp`.
- **Recordings come through but the arm doesn't move** → you're in
  dry-run. Add `--execute`.
- **Two prompts seem to overlap** → they shouldn't; the queue + worker
  enforces strict serial execution. If you see overlap, you're probably
  running both the standalone CLI *and* `--mic-server` against the same
  port. Pick one.
