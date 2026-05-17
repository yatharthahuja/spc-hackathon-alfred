from __future__ import annotations

from typing import Iterable

from app.config import Settings
from app.hardware.resources import RobotMoveResult, RobotSequenceResult
from app.memory.session_memory import TASK_HISTORY
from app.orchestrator.schemas import (
    CompletionResult,
    FinalResponse,
    Intent,
    OrchestratorPlan,
    SkillCall,
    SkillResult,
)
from app.pipeline import AlfredRuntime


class FakeRobotResource:
    def move_to_pose(self, pose_name: str) -> RobotMoveResult:
        return RobotMoveResult(pose_name=pose_name, attempted=True, moved=True)

    def move_through_poses(self, pose_names: Iterable[str]) -> RobotSequenceResult:
        names = list(pose_names)
        return RobotSequenceResult(trajectory_names=names, attempted=True, moved=True)


class FakeCameraResource:
    def capture_to_file(self, **_kwargs):
        return _kwargs["save_dir"] / "desk.jpg"


class FakeHardwareContext:
    def __init__(self):
        self.robot = FakeRobotResource()
        self.camera = FakeCameraResource()
        self.connect_calls = 0
        self.close_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_runtime_borrows_shared_hardware_context(tmp_path):
    settings = Settings.load()
    hardware = FakeHardwareContext()

    runtime = AlfredRuntime(settings, run_dir=tmp_path, hardware_context=hardware)
    capture_skill = runtime.router.get("capture_wrist_camera_image")

    assert hardware.connect_calls == 1
    assert capture_skill.hardware is hardware
    assert "general_conversation" in runtime.router.names()
    assert "answer_task_history" in runtime.router.names()
    assert "go_home" in runtime.router.names()
    assert "go_overlook" in runtime.router.names()
    assert "pick_blue_marker" in runtime.router.names()
    assert "pick_place_blue_marker" in runtime.router.names()
    assert "enzyme_experiments" in runtime.router.names()
    assert "save_notes" in runtime.router.names()
    assert "answer_scene_question" in runtime.router.names()
    assert "read_notes" in runtime.router.names()

    runtime.close()
    assert hardware.close_calls == 0


def test_runtime_records_tasks_and_history_queries(tmp_path):
    TASK_HISTORY.clear()
    settings = Settings.load()
    runtime = AlfredRuntime(settings, run_dir=tmp_path, hardware_context=FakeHardwareContext())
    request = runtime.orchestrator.classify_intent("go home")
    user_request = runtime.handle_text("go home").request

    assert request.intent
    assert TASK_HISTORY.count() == 1
    assert TASK_HISTORY.recent(1)[0]["user_text"] == "go home"

    history_plan = OrchestratorPlan(
        goal="what was the last task?",
        intent=Intent.RUN_SKILL,
        tasks=[],
        skill_calls=[SkillCall(skill_name="answer_task_history", arguments={"question": "what was the last task?"})],
        completion_criteria=[],
    )
    runtime.record_task_history(
        user_request,
        history_plan,
        [SkillResult(skill_name="answer_task_history", status="success", output={"answer_text": "Go home."})],
        [],
        [],
        CompletionResult(task_complete=True, reason="Answered history.", next_action="none"),
        FinalResponse(task_complete=True, answer_text="Go home.", confidence=0.9),
    )

    assert TASK_HISTORY.count() == 2
    assert TASK_HISTORY.recent(1)[0]["skill_names"] == ["answer_task_history"]
    runtime.close()
    TASK_HISTORY.clear()
