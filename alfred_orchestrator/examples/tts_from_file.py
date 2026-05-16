from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import urllib.error
from pathlib import Path

from test_elevenlabs_api import load_env_value, synthesize_speech


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def play_audio(path: Path) -> bool:
    afplay = shutil.which("afplay")
    if not afplay:
        return False
    subprocess.run([afplay, str(path)], check=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Read text from a file and test ElevenLabs TTS.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "examples" / "tts_sample.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "elevenlabs_test" / "tts_from_file.mp3",
    )
    parser.add_argument(
        "--voice-id",
        default=load_env_value("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb",
    )
    parser.add_argument("--tts-model", default="eleven_v3")
    parser.add_argument("--output-format", default="mp3_44100_128")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    api_key = load_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY was not found in the environment, .env, or .env.example.")
        return 1
    if not args.input.exists():
        print(f"Input text file does not exist: {args.input}")
        return 1

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        print(f"Input text file is empty: {args.input}")
        return 1

    try:
        print(f"Reading text from: {args.input}")
        print("Testing ElevenLabs text-to-speech...")
        audio_path = synthesize_speech(
            api_key=api_key,
            voice_id=args.voice_id,
            model_id=args.tts_model,
            output_format=args.output_format,
            text=text,
            output_path=args.output,
            timeout=args.timeout,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            message = body
        print(f"ElevenLabs API returned HTTP {exc.code}: {message}")
        return 1
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        print(f"Network error while contacting ElevenLabs: {exc}")
        return 1

    print(f"TTS audio saved: {audio_path}")
    if not args.no_play:
        played = play_audio(audio_path)
        print(f"Audio played: {played}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
