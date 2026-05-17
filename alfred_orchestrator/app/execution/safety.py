from __future__ import annotations

from app.config import Settings
from app.orchestrator.task_registry import SkillCatalog


class SafetyGate:
    def __init__(self, settings: Settings, catalog: SkillCatalog):
        self.settings = settings
        self.catalog = catalog

    def validate_skill(self, skill_name: str) -> None:
        self.catalog.get(skill_name)
        if not self.catalog.is_enabled(skill_name):
            raise PermissionError(f"Skill is disabled: {skill_name}")
