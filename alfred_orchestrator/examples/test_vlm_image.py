from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.orchestrator.prompt_registry import PromptRegistry
from app.skills.vlm_describe import DescribeImageWithVLMSkill


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OpenAI VLM description on one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", default="What objects are visible on the desk?")
    parser.add_argument("--user-text", default="What is on my desk?")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image does not exist: {args.image}")
        return 1

    settings = Settings.load()
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    skill = DescribeImageWithVLMSkill(settings, prompts)
    result = skill.run(
        image_path=str(args.image),
        question=args.question,
        user_text=args.user_text,
    )

    if result.status != "success":
        print(f"VLM test failed: {result.error}")
        return 1

    print("VLM description:")
    print(result.output.get("spoken_summary"))
    print("Objects:")
    print(json.dumps(result.output.get("objects", []), indent=2))
    print("Uncertainties:")
    print(json.dumps(result.output.get("uncertainties", []), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
