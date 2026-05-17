from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.hardware.resources import RobotMoveResult
from app.memory.session_memory import SessionMemory
from app.orchestrator.prompt_registry import PromptRegistry
from app.skills.scene_qa import AnswerSceneQuestionSkill


class FakeRobotResource:
    def __init__(self):
        self.moves: list[str] = []

    def move_to_pose(self, pose_name: str) -> RobotMoveResult:
        self.moves.append(pose_name)
        return RobotMoveResult(pose_name=pose_name, attempted=True, moved=True)


class FakeCameraResource:
    def __init__(self):
        self.capture_calls: list[dict[str, Any]] = []

    def capture_to_file(self, *, camera_id: int, save_dir: Path, prefix: str, flip: bool) -> Path:
        self.capture_calls.append(
            {
                "camera_id": camera_id,
                "save_dir": save_dir,
                "prefix": prefix,
                "flip": flip,
            }
        )
        image_path = save_dir / f"{prefix}.jpg"
        image_path.write_bytes(b"fake-jpeg")
        return image_path


class FakeHardwareContext:
    def __init__(self):
        self.robot = FakeRobotResource()
        self.camera = FakeCameraResource()


class FakeResponses:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return type(
            "FakeResponse",
            (),
            {
                "output_text": (
                    '{"answer_text": "The marker appears to be blue.", '
                    '"confidence": 0.84, "evidence": ["a blue marker is visible"], '
                    '"uncertainties": []}'
                )
            },
        )()


class FakeOpenAIClient:
    def __init__(self, **_kwargs: Any):
        self.responses = FakeResponses()


def test_scene_qa_moves_to_overlook_captures_image_and_answers(tmp_path):
    settings = replace(Settings.load(), openai_api_key="test-key")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    hardware = FakeHardwareContext()
    history = SessionMemory()
    history.add({"user_text": "pick the marker", "answer_text": "Picked up the blue marker."})
    created_clients: list[FakeOpenAIClient] = []

    def client_factory(**kwargs: Any) -> FakeOpenAIClient:
        client = FakeOpenAIClient(**kwargs)
        created_clients.append(client)
        return client

    skill = AnswerSceneQuestionSkill(
        settings=settings,
        prompt_registry=prompts,
        run_dir=tmp_path,
        hardware_context=hardware,
        task_history=history,
        openai_client_factory=client_factory,
    )

    result = skill.run(question="what color is the marker?", user_text="what color is the marker?")

    assert result.status == "success"
    assert hardware.robot.moves == ["overlook"]
    assert hardware.camera.capture_calls[0]["prefix"] == "scene_qa"
    assert result.output["answer_text"] == "The marker appears to be blue."
    assert result.output["confidence"] == 0.84
    assert result.output["question"] == "what color is the marker?"
    assert created_clients[0].responses.calls[0]["model"] == settings.openai_vision_model
    prompt_text = created_clients[0].responses.calls[0]["input"][0]["content"][0]["text"]
    assert "Full session memory" in prompt_text
    assert "pick the marker" in prompt_text
    assert "Use session memory only when it is relevant" in prompt_text
