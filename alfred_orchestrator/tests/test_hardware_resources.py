from __future__ import annotations

from typing import Any

from app.hardware import resources


class FakeVideoCapture:
    instances: list["FakeVideoCapture"] = []

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.released = False
        self.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class FakeSequencer:
    instances: list["FakeSequencer"] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.executed: list[str] = []
        self.instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def has(self, pose_name: str) -> bool:
        return pose_name == "overlook"

    def execute(self, pose_name: str) -> bool:
        self.executed.append(pose_name)
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def test_camera_resource_reuses_handle_until_camera_id_changes(monkeypatch):
    FakeVideoCapture.instances.clear()
    monkeypatch.setattr(resources.cv2, "VideoCapture", FakeVideoCapture)

    camera = resources.CameraResource(default_camera_id=0)

    assert camera.connect(0) is True
    assert camera.set_camera_id(0) is True
    assert len(FakeVideoCapture.instances) == 1

    assert camera.set_camera_id(1) is True
    assert len(FakeVideoCapture.instances) == 2
    assert FakeVideoCapture.instances[0].released is True
    assert FakeVideoCapture.instances[1].released is False

    camera.close()
    assert FakeVideoCapture.instances[1].released is True


def test_robot_resource_connects_once_and_closes_once():
    FakeSequencer.instances.clear()
    robot = resources.RobotResource(sequencer_factory=FakeSequencer)

    assert robot.connect() is True
    assert robot.connect() is True
    assert len(FakeSequencer.instances) == 1
    assert FakeSequencer.instances[0].connect_calls == 1

    first_move = robot.move_to_pose("overlook")
    second_move = robot.move_to_pose("overlook")

    assert first_move.moved is True
    assert second_move.moved is True
    assert FakeSequencer.instances[0].executed == ["overlook", "overlook"]

    robot.close()
    assert FakeSequencer.instances[0].disconnect_calls == 1


def test_robot_resource_tolerates_unavailable_hardware():
    def unavailable_factory(**_kwargs: Any) -> Any:
        raise RuntimeError("serial bridge missing")

    robot = resources.RobotResource(sequencer_factory=unavailable_factory)

    assert robot.connect() is False
    result = robot.move_to_pose("overlook")

    assert result.attempted is True
    assert result.moved is False
    assert result.unavailable is True
    assert "serial bridge missing" in str(result.error)
