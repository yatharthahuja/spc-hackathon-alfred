from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.hardware.resources import HardwareContext
from app.memory.session_memory import SessionMemory, TASK_HISTORY
from app.orchestrator.json_utils import extract_json_object
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import ReadNotesResult, SkillResult
from app.skills.base import Skill, failure, success


class ReadNotesSkill(Skill):
    name = "read_notes"

    def __init__(
        self,
        settings: Settings,
        prompt_registry: PromptRegistry,
        run_dir: Path,
        hardware_context: HardwareContext,
        task_history: SessionMemory = TASK_HISTORY,
        openai_client_factory: Callable[..., Any] | None = None,
    ):
        self.settings = settings
        self.prompt_registry = prompt_registry
        self.run_dir = run_dir
        self.hardware = hardware_context
        self.task_history = task_history
        self.openai_client_factory = openai_client_factory

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for note reading")

            user_text = str(kwargs.get("user_text", "Read my notes"))
            pose_name = str(kwargs.get("overlook_pose_name", "overlook"))
            print(f"[read_notes] Moving robot to overlook pose: {pose_name}")
            robot_move = self.hardware.robot.move_to_pose(pose_name)
            print(f"[read_notes] Robot move output: {robot_move.output()}")
            if not robot_move.moved:
                reason = robot_move.error or f"robot did not report reaching pose {pose_name!r}"
                return SkillResult(
                    skill_name=self.name,
                    status="error",
                    error=f"Robot must reach pose {pose_name!r} before note reading: {reason}",
                    output=robot_move.output(),
                )

            camera_id = int(kwargs.get("camera_id", self.settings.camera_id))
            print(f"[read_notes] Capturing image from camera: {camera_id}")
            image_path = self.hardware.camera.capture_to_file(
                camera_id=camera_id,
                save_dir=Path(kwargs.get("save_dir") or self.run_dir),
                prefix=str(kwargs.get("prefix", "read_notes")),
                flip=bool(kwargs.get("flip", False)),
            )
            print(f"[read_notes] Captured image path: {image_path}")
            prompt = self.prompt_registry.render(
                "read_notes_vlm",
                {
                    "user_text": user_text,
                    "task_history": self.task_history.all(),
                },
            )
            print("[read_notes] Prompt sent to VLM:")
            print(prompt)

            if self.openai_client_factory is None:
                from openai import OpenAI

                client = OpenAI(api_key=self.settings.openai_api_key)
            else:
                client = self.openai_client_factory(api_key=self.settings.openai_api_key)
            response = client.responses.create(
                model=self.settings.openai_vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": self._image_data_url(image_path)},
                        ],
                    }
                ],
            )
            raw_text = response.output_text
            print("[read_notes] Raw VLM output:")
            print(raw_text)
            parsed = extract_json_object(raw_text)
            notes_result = ReadNotesResult.model_validate(parsed)
            return success(
                self.name,
                {
                    "image_path": str(image_path),
                    "raw_response": raw_text,
                    **robot_move.output(),
                    **notes_result.model_dump(),
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
