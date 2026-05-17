from __future__ import annotations

from pathlib import Path
from typing import Any

from app.hardware.resources import RobotMoveResult
from app.skills.camera import CaptureWristCameraImageSkill


class FakeRobotResource:
    def __init__(self, result: RobotMoveResult):
        self.result = result
        self.moves: list[str] = []

    def move_to_pose(self, pose_name: str) -> RobotMoveResult:
        self.moves.append(pose_name)
        return self.result


class FakeCameraResource:
    def __init__(self, image_name: str = "desk.jpg", error: Exception | None = None):
        self.image_name = image_name
        self.error = error
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
        if self.error:
            raise self.error
        return save_dir / self.image_name


class FakeHardwareContext:
    def __init__(self, robot_result: RobotMoveResult, camera: FakeCameraResource | None = None):
        self.robot = FakeRobotResource(robot_result)
        self.camera = camera or FakeCameraResource()


def test_capture_moves_shared_robot_to_overlook_before_taking_image(tmp_path):
    hardware = FakeHardwareContext(
        RobotMoveResult("overlook", attempted=True, moved=True),
    )
    skill = CaptureWristCameraImageSkill(
        run_dir=tmp_path,
        default_camera_id=4,
        hardware_context=hardware,
    )

    result = skill.run(max_attempts=1)

    assert result.status == "success"
    assert result.output["image_path"] == str(tmp_path / "desk.jpg")
    assert result.output["robot_pose_name"] == "overlook"
    assert result.output["robot_move_attempted"] is True
    assert result.output["robot_moved"] is True
    assert result.output["robot_unavailable"] is False
    assert hardware.robot.moves == ["overlook"]
    assert hardware.camera.capture_calls == [
        {
            "camera_id": 4,
            "save_dir": tmp_path,
            "prefix": "desk",
            "flip": False,
        }
    ]


def test_capture_falls_back_to_camera_when_robot_unavailable(tmp_path):
    hardware = FakeHardwareContext(
        RobotMoveResult(
            "overlook",
            attempted=True,
            moved=False,
            unavailable=True,
            error="robot bus unavailable",
        ),
    )
    skill = CaptureWristCameraImageSkill(
        run_dir=tmp_path,
        hardware_context=hardware,
    )

    result = skill.run(max_attempts=1)

    assert result.status == "success"
    assert result.output["image_path"] == str(tmp_path / "desk.jpg")
    assert result.output["robot_move_attempted"] is True
    assert result.output["robot_moved"] is False
    assert result.output["robot_unavailable"] is True
    assert "robot bus unavailable" in result.output["robot_error"]


def test_capture_can_require_robot_move_before_camera(tmp_path):
    camera = FakeCameraResource()
    hardware = FakeHardwareContext(
        RobotMoveResult(
            "overlook",
            attempted=True,
            moved=False,
            unavailable=True,
            error="robot bus unavailable",
        ),
        camera=camera,
    )
    skill = CaptureWristCameraImageSkill(
        run_dir=tmp_path,
        hardware_context=hardware,
    )

    result = skill.run(max_attempts=1, require_robot_move=True)

    assert result.status == "error"
    assert "Robot must reach pose 'overlook'" in result.error
    assert "robot bus unavailable" in result.error
    assert camera.capture_calls == []


def test_capture_fails_without_image_when_robot_does_not_reach_overlook(tmp_path):
    camera = FakeCameraResource()
    hardware = FakeHardwareContext(
        RobotMoveResult(
            "overlook",
            attempted=True,
            moved=False,
            error="Robot did not settle at pose 'overlook'",
        ),
        camera=camera,
    )
    skill = CaptureWristCameraImageSkill(
        run_dir=tmp_path,
        hardware_context=hardware,
    )

    result = skill.run(max_attempts=1)

    assert result.status == "error"
    assert "did not settle" in result.error
    assert camera.capture_calls == []
