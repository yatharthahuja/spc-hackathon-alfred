from __future__ import annotations

import sys
from pathlib import Path

from app.orchestrator.schemas import SkillResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from interactive_voice_loop import (  # noqa: E402
    image_data_url,
    latest_successful_image_result,
    latest_successful_vlm_result,
    robot_move_payload,
)


def test_web_payload_helpers_use_latest_image_skill_result(tmp_path):
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    results = [
        SkillResult(
            skill_name="capture_wrist_camera_image",
            status="success",
            output={"image_path": str(first_image)},
        ),
        SkillResult(
            skill_name="answer_scene_question",
            status="success",
            output={
                "image_path": str(second_image),
                "answer_text": "I see the marker.",
                "robot_pose_name": "overlook",
                "robot_move_attempted": True,
                "robot_moved": True,
            },
        ),
    ]

    assert latest_successful_image_result(results).skill_name == "answer_scene_question"
    assert latest_successful_vlm_result(results).skill_name == "answer_scene_question"
    assert image_data_url(second_image).startswith("data:image/jpeg;base64,")
    assert robot_move_payload(results[-1].output) == {
        "robot_pose_name": "overlook",
        "robot_move_attempted": True,
        "robot_moved": True,
    }


def test_web_payload_helpers_include_read_notes_images(tmp_path):
    note_image = tmp_path / "note.jpg"
    note_image.write_bytes(b"note")
    results = [
        SkillResult(
            skill_name="read_notes",
            status="success",
            output={"image_path": str(note_image), "answer_text": "Your note says hello."},
        )
    ]

    assert latest_successful_image_result(results).skill_name == "read_notes"
    assert latest_successful_vlm_result(results).skill_name == "read_notes"
