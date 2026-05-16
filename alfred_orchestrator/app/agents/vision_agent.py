from __future__ import annotations

from app.orchestrator.schemas import SkillCall


class VisionAgent:
    def describe_image_call(self, image_reference: str, user_text: str) -> SkillCall:
        return SkillCall(
            skill_name="describe_image_with_vlm",
            arguments={
                "image_path": image_reference,
                "question": "What objects are visible on the desk?",
                "user_text": user_text,
            },
        )
