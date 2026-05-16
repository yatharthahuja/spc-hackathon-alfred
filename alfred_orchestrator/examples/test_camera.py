from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.interfaces.camera import capture_image, detect_cameras


def main() -> None:
    settings = Settings.load()
    print("Scanning for cameras...")
    cameras = detect_cameras()

    if not cameras:
        raise SystemExit("No cameras detected.")

    for camera in cameras:
        print(
            f"Camera {camera.camera_id}: "
            f"{camera.width}x{camera.height} @ {camera.fps:.0f} fps"
        )

    selected = settings.camera_id
    if selected not in {camera.camera_id for camera in cameras}:
        selected = cameras[0].camera_id
        print(f"Configured camera was not found. Falling back to camera {selected}.")

    run_dir = settings.new_run_dir("camera_test")
    image_path = capture_image(selected, Path(run_dir), prefix="camera_test")
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()
