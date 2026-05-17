from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import Settings
from app.pipeline import AlfredRuntime
from live_stt_test import record_microphone
from test_elevenlabs_api import load_env_value, transcribe_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record live speech, transcribe it, then run Alfred's text pipeline."
    )
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--device", default=":2")
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "elevenlabs_test" / "m4_voice_command.wav",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--speak", action="store_true")
    args = parser.parse_args()

    api_key = load_env_value("ELEVENLABS_API_KEY")
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or Settings.load().elevenlabs_stt_model
    if not api_key:
        print("ELEVENLABS_API_KEY was not found in the environment, .env, or .env.example.")
        return 1

    print(f"Recording for {args.seconds} seconds. Say: What is on my desk?")
    for remaining in range(args.countdown, 0, -1):
        print(f"Starting in {remaining}...")
        time.sleep(1)
    audio_path = record_microphone(args.seconds, args.device, args.audio_output)
    print(f"Saved recording: {audio_path}")

    transcript_payload = transcribe_audio(
        api_key=api_key,
        model_id=stt_model,
        audio_path=audio_path,
        timeout=args.timeout,
    )
    transcript = str(transcript_payload.get("text") or "").strip()
    print(f"Transcript: {transcript}")
    if not transcript:
        print(json.dumps(transcript_payload, indent=2))
        return 1

    runtime = AlfredRuntime(Settings.load())
    result = runtime.handle_text(transcript, input_type="voice", speak=args.speak)
    print(f"Alfred: {result.response.answer_text}")
    print(f"Completion: {result.completion.task_complete} ({result.completion.reason})")
    print(f"Run artifacts: {result.run_dir}")
    return 0 if result.response.task_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
