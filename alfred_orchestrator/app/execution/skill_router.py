from __future__ import annotations

from typing import Dict

from app.skills.base import Skill


class SkillRouter:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, skill_name: str) -> Skill:
        if skill_name not in self._skills:
            raise KeyError(f"No implementation registered for skill: {skill_name}")
        return self._skills[skill_name]

    def names(self):
        return sorted(self._skills.keys())
