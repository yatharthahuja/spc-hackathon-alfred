from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from app.orchestrator.schemas import SkillResult


class Skill(ABC):
    name: str

    @abstractmethod
    def run(self, **kwargs: Any) -> SkillResult:
        raise NotImplementedError


def success(skill_name: str, output: Dict[str, Any]) -> SkillResult:
    return SkillResult(skill_name=skill_name, status="success", output=output)


def failure(skill_name: str, error: Exception) -> SkillResult:
    return SkillResult(skill_name=skill_name, status="error", error=str(error), output={})
