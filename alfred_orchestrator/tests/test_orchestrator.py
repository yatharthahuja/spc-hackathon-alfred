from __future__ import annotations

from app.config import Settings
from app.logs.event_logger import EventLogger
from app.orchestrator.orchestrator import AlfredOrchestrator
from app.orchestrator.schemas import Intent, UserRequest
from app.orchestrator.task_registry import SkillCatalog


def test_rules_first_describe_desk_plan(tmp_path):
    settings = Settings.load()
    logger = EventLogger(tmp_path, "test-request")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    orchestrator = AlfredOrchestrator(catalog, logger, camera_id=2)
    request = UserRequest(request_id="test-request", raw_text="What is on my desk?")

    intent = orchestrator.classify_intent(request.raw_text)
    plan = orchestrator.create_plan(request, intent)

    assert intent.intent == Intent.DESCRIBE_DESK
    assert plan.skill_calls[0].skill_name == "capture_wrist_camera_image"
    assert plan.skill_calls[0].arguments["camera_id"] == 2
    assert plan.skill_calls[1].skill_name == "describe_image_with_vlm"


def test_rules_first_describe_desk_matches_natural_variants(tmp_path):
    settings = Settings.load()
    logger = EventLogger(tmp_path, "test-request")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    orchestrator = AlfredOrchestrator(catalog, logger)

    for text in [
        "Can you look at my workspace?",
        "What can you see?",
        "Describe the table please",
        "Inspect the work surface",
    ]:
        assert orchestrator.classify_intent(text).intent == Intent.DESCRIBE_DESK


def test_unknown_intent_has_no_skill_calls(tmp_path):
    settings = Settings.load()
    logger = EventLogger(tmp_path, "test-request")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    orchestrator = AlfredOrchestrator(catalog, logger)
    request = UserRequest(request_id="test-request", raw_text="Order soup ingredients")

    intent = orchestrator.classify_intent(request.raw_text)
    plan = orchestrator.create_plan(request, intent)

    assert intent.intent == Intent.UNKNOWN
    assert plan.skill_calls == []
