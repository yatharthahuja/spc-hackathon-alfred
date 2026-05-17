from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from app.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DRIVERS_DIR = PROJECT_ROOT / "robot_drivers"
DEFAULT_ARM_POSES_FILE = ROBOT_DRIVERS_DIR / "arm_poses.yaml"


@dataclass
class RobotMoveResult:
    pose_name: str
    attempted: bool
    moved: bool
    unavailable: bool = False
    error: str | None = None

    def output(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "robot_pose_name": self.pose_name,
            "robot_move_attempted": self.attempted,
            "robot_moved": self.moved,
            "robot_unavailable": self.unavailable,
        }
        if self.error:
            data["robot_error"] = self.error
        return data


def _load_default_sequencer_factory() -> Callable[..., Any]:
    try:
        from robot_drivers.robot import Sequencer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(str(exc)) from exc
    return Sequencer


class RobotResource:
    """Owns one robot sequencer connection for the process/runtime lifetime."""

    def __init__(
        self,
        *,
        poses_file: Path = DEFAULT_ARM_POSES_FILE,
        sequencer_factory: Callable[..., Any] | None = None,
        segment_duration: float = 3.0,
        home_duration: float = 3.0,
        zero_duration: float = 3.0,
        zero_on_disconnect: bool = True,
    ) -> None:
        self.poses_file = poses_file
        self.sequencer_factory = sequencer_factory
        self.segment_duration = segment_duration
        self.home_duration = home_duration
        self.zero_duration = zero_duration
        self.zero_on_disconnect = zero_on_disconnect
        self._lock = threading.RLock()
        self._sequencer: Any | None = None
        self.connected = False
        self.unavailable = False
        self.last_error: str | None = None

    def connect(self) -> bool:
        with self._lock:
            if self.connected and self._sequencer is not None:
                return True
            try:
                sequencer_factory = self.sequencer_factory or _load_default_sequencer_factory()
                sequencer = sequencer_factory(
                    poses_file=self.poses_file,
                    segment_duration=self.segment_duration,
                    home_duration=self.home_duration,
                    zero_duration=self.zero_duration,
                    zero_on_disconnect=self.zero_on_disconnect,
                )
                sequencer.connect()
            except Exception as exc:
                self._sequencer = None
                self.connected = False
                self.unavailable = True
                self.last_error = str(exc)
                print(f"[hardware] Robot unavailable: {exc}", flush=True)
                return False

            self._sequencer = sequencer
            self.connected = True
            self.unavailable = False
            self.last_error = None
            return True

    def has_pose(self, pose_name: str) -> bool:
        with self._lock:
            if not self.connected and not self.connect():
                return False
            if hasattr(self._sequencer, "has"):
                return bool(self._sequencer.has(pose_name))
            return True

    def move_to_pose(self, pose_name: str) -> RobotMoveResult:
        with self._lock:
            if not self.connected and not self.connect():
                return RobotMoveResult(
                    pose_name=pose_name,
                    attempted=True,
                    moved=False,
                    unavailable=True,
                    error=self.last_error or "robot is unavailable",
                )

            if hasattr(self._sequencer, "has") and not self._sequencer.has(pose_name):
                return RobotMoveResult(
                    pose_name=pose_name,
                    attempted=True,
                    moved=False,
                    error=f"Robot pose {pose_name!r} is not defined in {self.poses_file}",
                )

            try:
                print(f"[hardware] Moving robot to {pose_name!r}.", flush=True)
                moved = bool(self._sequencer.execute(pose_name))
            except Exception as exc:
                self.connected = False
                self.unavailable = True
                self.last_error = str(exc)
                return RobotMoveResult(
                    pose_name=pose_name,
                    attempted=True,
                    moved=False,
                    error=f"Robot move to {pose_name!r} failed: {exc}",
                )

            return RobotMoveResult(
                pose_name=pose_name,
                attempted=True,
                moved=moved,
                error=None if moved else f"Robot did not settle at pose {pose_name!r}",
            )

    def close(self) -> None:
        with self._lock:
            sequencer = self._sequencer
            self._sequencer = None
            self.connected = False
            if sequencer is None:
                return
            try:
                sequencer.disconnect()
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[hardware] Robot disconnect failed: {exc}", flush=True)


class CameraResource:
    """Owns one VideoCapture handle, reopening only when the selected camera changes."""

    def __init__(self, default_camera_id: int = 0) -> None:
        self.default_camera_id = default_camera_id
        self.camera_id: int | None = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.RLock()
        self.connected = False
        self.unavailable = False
        self.last_error: str | None = None

    def connect(self, camera_id: int | None = None) -> bool:
        with self._lock:
            selected = self.default_camera_id if camera_id is None else int(camera_id)
            if self.connected and self._cap is not None and self.camera_id == selected:
                return True

            self.close()
            cap = cv2.VideoCapture(selected)
            if not cap.isOpened():
                cap.release()
                self.connected = False
                self.unavailable = True
                self.last_error = f"Could not open camera {selected}"
                print(f"[hardware] Camera unavailable: {self.last_error}", flush=True)
                return False

            self._cap = cap
            self.camera_id = selected
            self.connected = True
            self.unavailable = False
            self.last_error = None
            return True

    def set_camera_id(self, camera_id: int) -> bool:
        return self.connect(camera_id)

    def grab_jpeg(self, camera_id: int | None = None, quality: int = 80) -> bytes:
        frame = self._read_frame(camera_id)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed for camera frame")
        return bytes(buf)

    def capture_to_file(
        self,
        *,
        camera_id: int | None = None,
        save_dir: Path,
        prefix: str = "desk",
        flip: bool = False,
    ) -> Path:
        frame = self._read_frame(camera_id)
        if flip:
            frame = cv2.flip(frame, -1)

        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"{prefix}_{stamp}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write image to {path}")
        return path

    def close(self) -> None:
        with self._lock:
            cap = self._cap
            self._cap = None
            self.camera_id = None
            self.connected = False
            if cap is not None:
                cap.release()

    def _read_frame(self, camera_id: int | None = None) -> Any:
        with self._lock:
            selected = self.default_camera_id if camera_id is None else int(camera_id)
            if not self.connect(selected):
                raise RuntimeError(self.last_error or f"Could not open camera {selected}")

            assert self._cap is not None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Could not capture image from camera {selected}")
            return frame


class HardwareContext:
    """Shared hardware resources owned by an Alfred runtime or server process."""

    def __init__(
        self,
        *,
        default_camera_id: int = 0,
        robot: RobotResource | None = None,
        camera: CameraResource | None = None,
    ) -> None:
        self.robot = robot or RobotResource()
        self.camera = camera or CameraResource(default_camera_id=default_camera_id)

    @classmethod
    def from_settings(cls, settings: Settings) -> "HardwareContext":
        return cls(default_camera_id=settings.camera_id)

    def connect(self) -> None:
        self.robot.connect()
        self.camera.connect()

    def close(self) -> None:
        self.camera.close()
        self.robot.close()

    def __enter__(self) -> "HardwareContext":
        self.connect()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
