from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

from app.hardware.resources import (
    DEFAULT_ARM_POSES_FILE,
    HardwareContext,
    RobotMoveResult,
    RobotResource,
)
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


class CaptureWristCameraImageSkill(Skill):
    name = "capture_wrist_camera_image"

    def __init__(
        self,
        run_dir: Path,
        default_camera_id: int = 0,
        hardware_context: HardwareContext | None = None,
        poses_file: Path = DEFAULT_ARM_POSES_FILE,
    ):
        self.run_dir = run_dir
        self.default_camera_id = default_camera_id
        self.hardware = hardware_context or HardwareContext(
            default_camera_id=default_camera_id,
            robot=RobotResource(poses_file=poses_file),
        )
        self.poses_file = poses_file

    def run(self, **kwargs: Any) -> SkillResult:
        camera_id = int(kwargs.get("camera_id", self.default_camera_id))
        save_dir = Path(kwargs.get("save_dir") or self.run_dir)
        max_attempts = int(kwargs.get("max_attempts", 3))
        retry_delay_seconds = float(kwargs.get("retry_delay_seconds", 0.75))
        last_error: Exception | None = None
        pose_name = str(kwargs.get("overlook_pose_name", "overlook"))
        move_to_overlook = _bool_arg(kwargs.get("move_to_overlook", True))
        require_robot_move = _bool_arg(kwargs.get("require_robot_move", False))
        robot_move = RobotMoveResult(
            pose_name=pose_name,
            attempted=False,
            moved=False,
        )

        if move_to_overlook:
            robot_move = self.hardware.robot.move_to_pose(pose_name)
            if robot_move.error and not robot_move.unavailable:
                return SkillResult(
                    skill_name=self.name,
                    status="error",
                    error=robot_move.error,
                    output=robot_move.output(),
                )

        if require_robot_move and not robot_move.moved:
            reason = robot_move.error or (
                "robot movement was disabled"
                if not robot_move.attempted
                else f"robot did not report reaching pose {pose_name!r}"
            )
            return SkillResult(
                skill_name=self.name,
                status="error",
                error=(
                    f"Robot must reach pose {pose_name!r} before wrist-camera capture: {reason}"
                ),
                output=robot_move.output(),
            )

        for attempt in range(1, max_attempts + 1):
            print(
                f"[camera] Image capture iteration {attempt}/{max_attempts} "
                f"using camera {camera_id}..."
            )
            try:
                image_path = self.hardware.camera.capture_to_file(
                    camera_id=camera_id,
                    save_dir=save_dir,
                    prefix=str(kwargs.get("prefix", "desk")),
                    flip=bool(kwargs.get("flip", False)),
                )
                print(
                    f"[camera] Image capture succeeded on iteration "
                    f"{attempt}/{max_attempts}."
                )
                output = {
                    "image_path": str(image_path),
                    "camera_id": camera_id,
                    "attempts": attempt,
                }
                output.update(robot_move.output())
                return success(self.name, output)
            except Exception as exc:
                last_error = exc
                print(
                    f"[camera] Image capture failed on iteration "
                    f"{attempt}/{max_attempts}: {exc}"
                )
                if attempt < max_attempts:
                    sleep(retry_delay_seconds)

        return failure(
            self.name,
            RuntimeError(
                f"Could not capture image from camera {camera_id} "
                f"after {max_attempts} iterations: {last_error}"
            ),
        )
