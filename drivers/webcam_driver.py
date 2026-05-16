import asyncio
import base64
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from config import CAMERA_INDEX
from interfaces import BaseDriver
from models import CameraFrame

MIN_FRAME_BRIGHTNESS = 20.0
CAPTURE_ATTEMPTS = 12
WARMUP_FRAMES = 15


class CameraUnavailableError(RuntimeError):
    pass


def _frame_brightness(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


class WebcamDriver(BaseDriver):
    def __init__(self, device_index: int | None = None) -> None:
        self._index = device_index if device_index is not None else CAMERA_INDEX
        self._capture: cv2.VideoCapture | None = None

    def _open_capture_sync(self) -> None:
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise CameraUnavailableError(
                f"Could not open webcam at index {self._index}"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture = cap

    async def initialize(self) -> None:
        await asyncio.to_thread(self._open_capture_sync)
        for _ in range(WARMUP_FRAMES):
            await asyncio.to_thread(self._read_frame_sync)
        print(f"[WebcamDriver] Ready. Device index: {self._index}")

    def _read_frame_sync(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            raise CameraUnavailableError("Webcam not initialized")
        return self._capture.read()

    async def health_check(self) -> bool:
        if self._capture is None or not self._capture.isOpened():
            return False
        ok, frame = await asyncio.to_thread(self._read_frame_sync)
        return ok and frame is not None

    def _capture_best_frame_sync(self) -> tuple[np.ndarray, float]:
        best_frame: np.ndarray | None = None
        best_brightness = 0.0
        for _ in range(CAPTURE_ATTEMPTS):
            ok, frame = self._read_frame_sync()
            if not ok or frame is None:
                continue
            brightness = _frame_brightness(frame)
            if brightness > best_brightness:
                best_brightness = brightness
                best_frame = frame
            if brightness >= MIN_FRAME_BRIGHTNESS:
                break
        if best_frame is None:
            raise CameraUnavailableError("Failed to read frame from webcam")
        return best_frame, best_brightness

    async def capture_frame(self) -> CameraFrame:
        frame, brightness = await asyncio.to_thread(self._capture_best_frame_sync)
        if brightness < MIN_FRAME_BRIGHTNESS:
            print(
                f"[WebcamDriver] Warning: dark frame (brightness={brightness:.1f}). "
                f"Try CAMERA_INDEX=1, allow camera permissions, or remove lens cover."
            )
        else:
            print(f"[WebcamDriver] Captured frame (brightness={brightness:.1f})")

        def _encode() -> tuple[str, int, int]:
            success, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not success:
                raise CameraUnavailableError("Failed to encode frame as JPEG")
            h, w = frame.shape[:2]
            return base64.b64encode(buf.tobytes()).decode("ascii"), w, h

        jpeg_b64, width, height = await asyncio.to_thread(_encode)
        return CameraFrame(
            jpeg_b64=jpeg_b64,
            width=width,
            height=height,
            captured_at=datetime.now(timezone.utc),
        )

    async def shutdown(self) -> None:
        if self._capture is not None:

            def _release() -> None:
                self._capture.release()

            await asyncio.to_thread(_release)
            self._capture = None
