from __future__ import annotations

import argparse
import json
import os
import socket
import urllib.error
import urllib.request
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


def call_openai(api_key: str, model: str, prompt: str, timeout: int) -> Dict[str, Any]:
    request_body = json.dumps({"model": model, "input": prompt}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    texts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the OpenAI Responses API key.")
    parser.add_argument(
        "--model",
        default=load_env_value("OPENAI_REASONING_MODEL") or "gpt-4.1-mini",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: Alfred key test ok",
    )
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    api_key = load_env_value("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY was not found in the environment, .env, or .env.example.")
        return 1

    print(f"Testing OpenAI Responses API with model: {args.model}")
    try:
        payload = call_openai(api_key, args.model, args.prompt, args.timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
            message = error_payload.get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body
        print(f"OpenAI API returned HTTP {exc.code}: {message}")
        return 1
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        print(f"Network error while contacting OpenAI: {exc}")
        return 1

    text = extract_response_text(payload)
    print("OpenAI response:")
    print(text or json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
