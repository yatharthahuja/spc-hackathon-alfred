# spc-hackathon-alfred

Modular kitchen butler orchestrator. Alfred uses a **VLM orchestrator** (OpenAI multimodal) to see via webcam, reply in natural language, and sequence **skills** from a maintained catalogue.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (https://platform.openai.com/api-keys)
```

## Run

```bash
python alfred_core.py
```

Speak naturally at the `alfred>` prompt:

| Example | Behavior |
|---------|----------|
| What do you see on the counter? | VLM describes the scene (webcam image) |
| Take a note about the soup recipe | Runs `note` skill |
| Please tidy the kitchen | VLM may run `manipulate` reset |
| `status` | Print `WorldState` JSON |
| `mess` / `tidy` | Test flags for counter state |
| `quit` | Exit |

After **60 seconds** of idle time, Alfred runs a proactive normalcy check (VLM + optional tidy).

Without `OPENAI_API_KEY`, a **heuristic fallback** handles basic note/tidy/look patterns using `WorldState` only.

## Architecture

```
User → AlfredOrchestrator → Webcam frame
                          → OpenAIVLMDriver.orchestrate()  (see + plan)
                          → execute skills (note, manipulate)
```

- [`skill_catalog.py`](skill_catalog.py) — skill metadata for the VLM
- [`drivers/webcam_driver.py`](drivers/webcam_driver.py) — OpenCV capture
- [`drivers/openai_vlm_driver.py`](drivers/openai_vlm_driver.py) — multimodal orchestration
- [`skills/`](skills/) — dumb executors (no LLM inside skills)

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | VLM orchestrator |
| `OPENAI_VLM_MODEL` | `gpt-4o-mini` | Model name |
| `CAMERA_INDEX` | `0` | Webcam device |
| `VISION_ENABLED` | `true` | Send image to API; false = text-only |
| `MAX_PLAN_STEPS` | `5` | Cap skill steps per turn |
| `DEBUG_SAVE_FRAME` | `false` | Write `debug_frame.jpg` each turn to verify capture |

On startup you should see `Webcam: ready | Vision API: on`. Each turn should log:

```
[WebcamDriver] Captured frame (brightness=85.2)
[OpenAIVLMDriver] Multimodal: sending 640x480 frame to gpt-4o
```

**If Alfred says it cannot see the image:** check brightness in the log. If `brightness < 20`, the frame is black — set `CAMERA_INDEX=1` in `.env`, grant camera permissions, or set `DEBUG_SAVE_FRAME=true` and open `debug_frame.jpg`. Use `OPENAI_VLM_MODEL=gpt-4o` for best vision (not mini).

## Files (gitignored locally)

- `state.json` — persisted `WorldState`
- `notes.txt` — note log
- `.env` — secrets
