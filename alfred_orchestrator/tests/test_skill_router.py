from __future__ import annotations

from typing import Any

from app.config import Settings
from app.execution.executor import SkillExecutor
from app.execution.safety import SafetyGate
from app.execution.skill_router import SkillRouter
from app.logs.event_logger import EventLogger
from app.orchestrator.schemas import OrchestratorPlan, SkillCall, SkillResult
from app.orchestrator.schemas import Intent, TaskStep
from app.orchestrator.task_registry import SkillCatalog
from app.skills.base import Skill, success


class ProduceImageSkill(Skill):
    name = "capture_wrist_camera_image"

    def run(self, **kwargs: Any) -> SkillResult:
        return success(self.name, {"image_path": "runs/test/desk.jpg"})


class DescribeSkill(Skill):
    name = "describe_image_with_vlm"

    def run(self, **kwargs: Any) -> SkillResult:
        return success(self.name, {"image_path": kwargs["image_path"], "spoken_summary": "I see a cup."})


class PickBlueMarkerFakeSkill(Skill):
    name = "pick_blue_marker"

    def run(self, **kwargs: Any) -> SkillResult:
        return success(self.name, {"message": "Picked up the blue marker."})


def test_executor_resolves_skill_output_references(tmp_path):
    settings = Settings.load()
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    router = SkillRouter()
    router.register(ProduceImageSkill())
    router.register(DescribeSkill())
    executor = SkillExecutor(router, SafetyGate(settings, catalog), EventLogger(tmp_path, "req"))
    plan = OrchestratorPlan(
        goal="Describe desk",
        intent=Intent.DESCRIBE_DESK,
        tasks=[TaskStep(task_id="t1", agent="desk_inspection_agent", required_skills=[])],
        skill_calls=[
            SkillCall(skill_name="capture_wrist_camera_image", arguments={}),
            SkillCall(
                skill_name="describe_image_with_vlm",
                arguments={"image_path": "$capture_wrist_camera_image.image_path"},
            ),
        ],
        completion_criteria=[],
    )

    results = executor.execute_plan(plan)

    assert results[-1].output["image_path"] == "runs/test/desk.jpg"


def test_executor_allows_enabled_physical_skills_without_env_flag(tmp_path):
    settings = Settings.load()
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    router = SkillRouter()
    router.register(PickBlueMarkerFakeSkill())
    executor = SkillExecutor(router, SafetyGate(settings, catalog), EventLogger(tmp_path, "req"))
    plan = OrchestratorPlan(
        goal="Pick marker",
        intent=Intent.RUN_SKILL,
        tasks=[TaskStep(task_id="t1", agent="catalog_skill_planner", required_skills=[])],
        skill_calls=[SkillCall(skill_name="pick_blue_marker", arguments={})],
        completion_criteria=[],
    )

    results = executor.execute_plan(plan)

    assert results[0].status == "success"
    assert results[0].output["message"] == "Picked up the blue marker."
