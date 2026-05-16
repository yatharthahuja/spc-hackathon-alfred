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

        skill_type = self.catalog.skill_type(skill_name)
        allowed_physical_skills = {"capture_wrist_camera_image", "move_arm_noop"}
        if skill_type == "physical" and skill_name not in allowed_physical_skills:
            if not self.settings.enable_physical_skills:
                raise PermissionError(f"Physical skill is not enabled: {skill_name}")
