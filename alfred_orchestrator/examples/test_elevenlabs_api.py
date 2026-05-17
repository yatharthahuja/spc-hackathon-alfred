from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import uuid
import urllib.error
import urllib.request
from urllib.parse import urlencode
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_value(name: str) -> Optional[str]:
    if os.getenv(name):
        return os.getenv(name)

    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"):
        value = read_env_file(env_path).get(name)
        if value:
            return value
    return None


def read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def synthesize_speech(
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    text: str,
    output_path: Path,
    timeout: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
            },
        }
    ).encode("utf-8")
    query = urlencode({"output_format": output_format})
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?{query}",
        data=body,
        headers={
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        output_path.write_bytes(response.read())
    return output_path


def transcribe_audio(api_key: str, model_id: str, audio_path: Path, timeout: int) -> Dict[str, Any]:
    boundary = f"----alfred-elevenlabs-{uuid.uuid4().hex}"
    body = build_multipart_body(
        boundary=boundary,
        fields={"model_id": model_id},
        file_field="file",
        file_path=audio_path,
    )
    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text",
        data=body,
        headers={
            "xi-api-key": api_key,
            "content-type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_multipart_body(
    boundary: str,
    fields: Dict[str, str],
    file_field: str,
    file_path: Path,
) -> bytes:
    parts = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                f"{value}\r\n".encode("utf-8"),
            ]
        )

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ElevenLabs TTS and STT APIs.")
    parser.add_argument("--text", default="The first move is what sets everything in motion.")
    parser.add_argument(
        "--voice-id",
        default=load_env_value("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb",
    )
    parser.add_argument("--tts-model", default="eleven_v3")
    parser.add_argument("--output-format", default="mp3_44100_128")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "elevenlabs_test" / "tts_test.mp3",
    )
    parser.add_argument(
        "--stt-audio",
        type=Path,
        help="Use an existing audio file for STT. If omitted, STT uses the TTS output.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Only test STT using --stt-audio.",
    )
    args = parser.parse_args()

    api_key = load_env_value("ELEVENLABS_API_KEY")
    stt_model = load_env_value("ELEVENLABS_STT_MODEL") or "scribe_v2"

    if not api_key:
        print("ELEVENLABS_API_KEY was not found in the environment, .env, or .env.example.")
        return 1

    if args.skip_tts and not args.stt_audio:
        print("--skip-tts requires --stt-audio.")
        return 1

    try:
        if args.skip_tts:
            audio_path = args.stt_audio
        else:
            print("Testing ElevenLabs text-to-speech...")
            audio_path = synthesize_speech(
                api_key=api_key,
                voice_id=args.voice_id,
                model_id=args.tts_model,
                output_format=args.output_format,
                text=args.text,
                output_path=args.output,
                timeout=args.timeout,
            )
            print(f"TTS audio saved: {audio_path}")

        if args.stt_audio:
            audio_path = args.stt_audio

        print(f"Testing ElevenLabs speech-to-text with model: {stt_model}")
        transcript_payload = transcribe_audio(
            api_key=api_key,
            model_id=stt_model,
            audio_path=audio_path,
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

    print("STT response text:")
    print(transcript_payload.get("text") or json.dumps(transcript_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
