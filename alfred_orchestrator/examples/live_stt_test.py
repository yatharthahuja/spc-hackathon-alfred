from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from test_elevenlabs_api import load_env_value, transcribe_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def list_audio_devices() -> int:
    ffmpeg = require_ffmpeg()
    command = [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    print(result.stderr or result.stdout)
    return 0


def record_microphone(seconds: int, device: str, output_path: Path) -> Path:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        device,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    return ffmpeg


def main() -> int:
    parser = argparse.ArgumentParser(description="Record live microphone audio and test ElevenLabs STT.")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument(
        "--device",
        default=":0",
        help='AVFoundation device string. Use ":0" for first audio device or run --list-devices.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "elevenlabs_test" / "live_stt.wav",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        return list_audio_devices()

    api_key = load_env_value("ELEVENLABS_API_KEY")
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or "scribe_v1"
    if not api_key:
        print("ELEVENLABS_API_KEY was not found in the environment, .env, or .env.example.")
        return 1

    try:
        print(f"Recording from microphone for {args.seconds} seconds...")
        print("Speak now.")
        audio_path = record_microphone(args.seconds, args.device, args.output)
        print(f"Saved recording: {audio_path}")

        print(f"Sending recording to ElevenLabs STT with model: {stt_model}")
        transcript_payload = transcribe_audio(
            api_key=api_key,
            model_id=stt_model,
            audio_path=audio_path,
            timeout=args.timeout,
        )
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg recording failed with exit code {exc.returncode}.")
        print("Try: python3 examples/live_stt_test.py --list-devices")
        return 1
    except Exception as exc:
        print(f"Live STT test failed: {exc}")
        return 1

    print("Transcript:")
    print(transcript_payload.get("text") or json.dumps(transcript_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
