from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

from app.interfaces.camera import capture_image
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class CaptureWristCameraImageSkill(Skill):
    name = "capture_wrist_camera_image"

    def __init__(self, run_dir: Path, default_camera_id: int = 0):
        self.run_dir = run_dir
        self.default_camera_id = default_camera_id

    def run(self, **kwargs: Any) -> SkillResult:
        camera_id = int(kwargs.get("camera_id", self.default_camera_id))
        save_dir = Path(kwargs.get("save_dir") or self.run_dir)
        max_attempts = int(kwargs.get("max_attempts", 3))
        retry_delay_seconds = float(kwargs.get("retry_delay_seconds", 0.75))
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            print(
                f"[camera] Image capture iteration {attempt}/{max_attempts} "
                f"using camera {camera_id}..."
            )
            try:
                image_path = capture_image(
                    camera_id=camera_id,
                    save_dir=save_dir,
                    prefix=str(kwargs.get("prefix", "desk")),
                    flip=bool(kwargs.get("flip", False)),
                )
                print(f"[camera] Image capture succeeded on iteration {attempt}/{max_attempts}.")
                return success(
                    self.name,
                    {
                        "image_path": str(image_path),
                        "camera_id": camera_id,
                        "attempts": attempt,
                    },
                )
            except Exception as exc:
                last_error = exc
                print(f"[camera] Image capture failed on iteration {attempt}/{max_attempts}: {exc}")
                if attempt < max_attempts:
                    sleep(retry_delay_seconds)

        return failure(
            self.name,
            RuntimeError(
                f"Could not capture image from camera {camera_id} "
                f"after {max_attempts} iterations: {last_error}"
            ),
        )
