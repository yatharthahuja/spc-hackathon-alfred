from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.interfaces.audio import record_wav
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class ListenSkill(Skill):
    name = "listen"

    def __init__(self, settings: Settings, run_dir: Path):
        self.settings = settings
        self.run_dir = run_dir

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            seconds = int(kwargs.get("seconds", self.settings.record_seconds))
            audio_path = Path(kwargs.get("audio_path") or self.run_dir / "audio_input.wav")
            record_wav(audio_path, seconds=seconds)
            return success(self.name, {"audio_file": str(audio_path), "seconds": seconds})
        except Exception as exc:
            return failure(self.name, exc)
