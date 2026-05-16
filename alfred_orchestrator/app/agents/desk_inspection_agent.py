from __future__ import annotations

from app.orchestrator.schemas import SkillCall


class DeskInspectionAgent:
    def create_skill_calls(self, user_text: str, camera_id: int = 0):
        return [
            SkillCall(
                skill_name="capture_wrist_camera_image",
                arguments={"camera_id": camera_id},
            ),
            SkillCall(
                skill_name="describe_image_with_vlm",
                arguments={
                    "image_path": "$capture_wrist_camera_image.image_path",
                    "question": "What objects are visible on the desk?",
                    "user_text": user_text,
                },
            ),
        ]
