# Alfred Orchestrator

Hackathon MVP for an end-to-end Alfred loop:

1. Capture a user request.
2. Classify intent.
3. Select registered skills.
4. Capture a wrist-camera image.
5. Ask an OpenAI vision model what is visible.
6. Generate a concise answer.
7. Speak it with ElevenLabs.
8. Log every step for debugging.

The first supported task is:

> What is on my desk?

## Setup

This project is configured for Python 3.10+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### System audio dependencies

`sounddevice` is a CFFI wrapper around the PortAudio C library. The macOS
wheel bundles PortAudio, but on Linux you must install it (and an audio
player for TTS playback) from your distro:

```bash
# Debian / Ubuntu
sudo apt-get install -y libportaudio2 alsa-utils

# Fedora / RHEL
sudo dnf install -y portaudio alsa-utils

# Arch
sudo pacman -S portaudio alsa-utils
```

`alsa-utils` provides `aplay`, which `play_audio` uses for playback. `ffmpeg`
(`ffplay`) also works if you prefer.

On macOS:

```bash
brew install portaudio
```

Fill in `.env`:

```bash
OPENAI_API_KEY="..."
ELEVENLABS_API_KEY="..."
```

If your machine only has Python 3.9 on `PATH`, install Python 3.10+ first and then create the virtual environment.

## Verify Camera

```bash
python examples/test_camera.py
```

This probes camera indices and captures a still image into `runs/`.

## Typed Desk Demo

```bash
python examples/desk_inspection_demo.py --text "What is on my desk?"
```

## Main CLI

Typed mode:

```bash
python app/main.py --mode typed
```

Voice mode:

```bash
python app/main.py --mode voice
```

The voice loop is push-to-talk first: press Enter, speak for the configured duration, then Alfred transcribes and runs the desk-inspection pipeline.

## Safety Defaults

- The orchestrator can only execute registered skills.
- LLM output is parsed into schemas before use.
- Physical robot movement is disabled by default.
- `move_arm_noop` exists only to preserve the future arm interface.
- Real arm motion should use named poses and separate safety checks.

## Run Artifacts

Each demo run writes artifacts under `runs/<timestamp>/`, including:

- `events.jsonl`
- captured images
- `plan.json`
- `vlm_response.json`
- `final_answer.txt`
- recorded/transcribed audio in voice mode
