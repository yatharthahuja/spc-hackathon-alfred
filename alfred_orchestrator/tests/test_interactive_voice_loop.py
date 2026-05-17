from __future__ import annotations

import sys
from pathlib import Path

from app.orchestrator.schemas import SkillResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from interactive_voice_loop import (  # noqa: E402
    _get_tts_events,
    _publish_tts_event,
    image_data_url,
    is_stream_disconnect,
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


def test_stream_disconnect_helper_treats_browser_ssl_eof_as_normal():
    assert is_stream_disconnect(BrokenPipeError("broken pipe")) is True
    assert is_stream_disconnect(OSError("EOF occurred in violation of protocol (_ssl.c:2426)"))
    assert is_stream_disconnect(RuntimeError("camera failed")) is False


def test_tts_events_are_published_incrementally():
    first = _publish_tts_event(
        "test-request",
        {"tts_audio_url": "/api/tts/test-request-pre-1.mp3", "tts_audio_bytes": 123},
    )
    second = _publish_tts_event(
        "test-request",
        {"tts_audio_url": "/api/tts/test-request.mp3", "tts_audio_bytes": 456},
    )

    assert first["index"] == 0
    assert second["index"] == 1
    assert _get_tts_events("test-request", after_index=-1) == [first, second]
    assert _get_tts_events("test-request", after_index=0) == [second]
