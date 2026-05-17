from __future__ import annotations

from app.config import Settings
from app.hardware.resources import RobotMoveResult
from app.pipeline import AlfredRuntime


class FakeRobotResource:
    def move_to_pose(self, pose_name: str) -> RobotMoveResult:
        return RobotMoveResult(pose_name=pose_name, attempted=True, moved=True)


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

    runtime.close()
    assert hardware.close_calls == 0
