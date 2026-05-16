from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.orchestrator.json_utils import extract_json_object
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import SkillResult, VLMResult
from app.skills.base import Skill, failure, success


class DescribeImageWithVLMSkill(Skill):
    name = "describe_image_with_vlm"

    def __init__(self, settings: Settings, prompt_registry: PromptRegistry):
        self.settings = settings
        self.prompt_registry = prompt_registry

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for VLM description")

            image_path = Path(str(kwargs["image_path"]))
            question = str(kwargs.get("question", "What objects are visible on the desk?"))
            user_text = str(kwargs.get("user_text", "What is on my desk?"))
            prompt = self.prompt_registry.render(
                "desk_vlm",
                {
                    "user_text": user_text,
                    "question": question,
                },
            )

            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.responses.create(
                model=self.settings.openai_vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": self._image_data_url(image_path)},
                        ],
                    }
                ],
            )
            raw_text = response.output_text
            parsed = extract_json_object(raw_text)
            vlm_result = VLMResult.model_validate(parsed)
            return success(
                self.name,
                {
                    "image_path": str(image_path),
                    "raw_response": raw_text,
                    **vlm_result.model_dump(),
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
