from __future__ import annotations

from pathlib import Path
from typing import Any

from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


DEFAULT_ORDER_SIGNAL_FILE = Path("/home/asjad/hackathon/drone/run")


class OrderingSkill(Skill):
    name = "ordering"

    def __init__(self, signal_file: Path = DEFAULT_ORDER_SIGNAL_FILE):
        self.signal_file = signal_file

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            user_text = str(kwargs.get("user_text") or "").strip()
            previous_value = ""
            if self.signal_file.exists():
                previous_value = self.signal_file.read_text(encoding="utf-8")

            self.signal_file.parent.mkdir(parents=True, exist_ok=True)
            self.signal_file.write_text("1", encoding="utf-8")

            return success(
                self.name,
                {
                    "signal_file": str(self.signal_file),
                    "signal_value": "1",
                    "previous_value": previous_value,
                    "user_text": user_text,
                    "message": "I sent the order signal.",
                },
            )
        except Exception as exc:
            return failure(self.name, exc)
