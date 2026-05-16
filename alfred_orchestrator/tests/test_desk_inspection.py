from __future__ import annotations

from app.agents.desk_inspection_agent import DeskInspectionAgent


def test_desk_inspection_agent_creates_static_skill_composition():
    calls = DeskInspectionAgent().create_skill_calls("What is on my desk?", camera_id=1)

    assert [call.skill_name for call in calls] == [
        "capture_wrist_camera_image",
        "describe_image_with_vlm",
    ]
    assert calls[0].arguments["camera_id"] == 1
    assert calls[1].arguments["image_path"] == "$capture_wrist_camera_image.image_path"
