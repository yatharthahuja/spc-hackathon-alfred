from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.hardware.resources import RobotMoveResult
from app.memory.session_memory import SessionMemory
from app.orchestrator.prompt_registry import PromptRegistry
from app.skills.read_notes import ReadNotesSkill


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
                    '{"extracted_text": "Buy milk\\nCall Sam", '
                    '"answer_text": "Your note says, buy milk, and call Sam.", '
                    '"confidence": 0.88, "uncertainties": []}'
                )
            },
        )()


class FakeOpenAIClient:
    def __init__(self, **_kwargs: Any):
        self.responses = FakeResponses()


def test_read_notes_moves_to_overlook_captures_image_and_reads_text(tmp_path):
    settings = replace(Settings.load(), openai_api_key="test-key")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    hardware = FakeHardwareContext()
    history = SessionMemory()
    history.add({"user_text": "order apples", "answer_text": "I sent the order signal."})
    created_clients: list[FakeOpenAIClient] = []

    def client_factory(**kwargs: Any) -> FakeOpenAIClient:
        client = FakeOpenAIClient(**kwargs)
        created_clients.append(client)
        return client

    skill = ReadNotesSkill(
        settings=settings,
        prompt_registry=prompts,
        run_dir=tmp_path,
        hardware_context=hardware,
        task_history=history,
        openai_client_factory=client_factory,
    )

    result = skill.run(user_text="read my notes")

    assert result.status == "success"
    assert hardware.robot.moves == ["overlook"]
    assert hardware.camera.capture_calls[0]["prefix"] == "read_notes"
    assert result.output["extracted_text"] == "Buy milk\nCall Sam"
    assert result.output["answer_text"] == "Your note says, buy milk, and call Sam."
    assert result.output["confidence"] == 0.88
    assert created_clients[0].responses.calls[0]["model"] == settings.openai_vision_model
    prompt_text = created_clients[0].responses.calls[0]["input"][0]["content"][0]["text"]
    assert "Full session memory" in prompt_text
    assert "order apples" in prompt_text
    assert "Use session memory only when it helps" in prompt_text


def test_read_notes_sends_contextual_note_question_to_vlm(tmp_path):
    settings = replace(Settings.load(), openai_api_key="test-key")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    hardware = FakeHardwareContext()
    created_clients: list[FakeOpenAIClient] = []

    def client_factory(**kwargs: Any) -> FakeOpenAIClient:
        client = FakeOpenAIClient(**kwargs)
        created_clients.append(client)
        return client

    skill = ReadNotesSkill(
        settings=settings,
        prompt_registry=prompts,
        run_dir=tmp_path,
        hardware_context=hardware,
        openai_client_factory=client_factory,
    )
    user_text = (
        "read my notes, in my notes I have what I need to buy, "
        "so tell me what I need to buy"
    )

    result = skill.run(user_text=user_text)

    assert result.status == "success"
    request_input = created_clients[0].responses.calls[0]["input"]
    prompt_text = request_input[0]["content"][0]["text"]
    assert user_text in prompt_text
    assert "you need to buy" in prompt_text
