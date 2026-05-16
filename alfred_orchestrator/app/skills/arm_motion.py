from __future__ import annotations

from typing import Any

from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, success


class MoveArmNoopSkill(Skill):
    name = "move_arm_noop"

    def run(self, **kwargs: Any) -> SkillResult:
        pose_name = str(kwargs.get("pose_name", "desk_view"))
        return success(
            self.name,
            {
                "pose_name": pose_name,
                "moved": False,
                "message": "Robot arm motion is disabled; noop skill completed.",
            },
        )
