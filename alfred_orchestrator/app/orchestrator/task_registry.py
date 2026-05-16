from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.config import load_yaml


class SkillCatalog:
    def __init__(self, skills_path: Path):
        data = load_yaml(skills_path)
        self.skills: Dict[str, Dict[str, Any]] = data.get("skills", {})

    def list_skills(self) -> List[str]:
        return list(self.skills.keys())

    def get(self, skill_name: str) -> Dict[str, Any]:
        if skill_name not in self.skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return self.skills[skill_name]

    def is_enabled(self, skill_name: str) -> bool:
        return bool(self.get(skill_name).get("enabled", True))

    def skill_type(self, skill_name: str) -> str:
        return str(self.get(skill_name).get("type", "digital"))
