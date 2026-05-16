from __future__ import annotations

import argparse

from app.config import Settings
from app.pipeline import AlfredRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Alfred desk-inspection demo.")
    parser.add_argument("--text", default="What is on my desk?")
    parser.add_argument("--speak", action="store_true")
    args = parser.parse_args()

    runtime = AlfredRuntime(Settings.load())
    result = runtime.handle_text(args.text, speak=args.speak)

    print(f"Transcript: {result.request.raw_text}")
    print("Plan:")
    print("  1. capture_wrist_camera_image")
    print("  2. describe_image_with_vlm")
    if args.speak:
        print("  3. speak")
    print(f"Alfred: {result.response.answer_text}")
    print(f"Run artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()
