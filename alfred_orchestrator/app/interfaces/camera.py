from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2


@dataclass(frozen=True)
class CameraInfo:
    camera_id: int
    width: int
    height: int
    fps: float


def detect_cameras(max_index: int = 10) -> List[CameraInfo]:
    available: List[CameraInfo] = []
    previous_log_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    try:
        for camera_id in range(max_index):
            cap = cv2.VideoCapture(camera_id)
            try:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(
                            CameraInfo(
                                camera_id=camera_id,
                                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                            )
                        )
            finally:
                cap.release()
    finally:
        if previous_log_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(previous_log_level)
    return available


def capture_image(
    camera_id: int,
    save_dir: Path,
    prefix: str = "desk",
    flip: bool = False,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not capture image from camera {camera_id}")

        if flip:
            frame = cv2.flip(frame, -1)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"{prefix}_{stamp}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write image to {path}")
        return path
    finally:
        cap.release()
