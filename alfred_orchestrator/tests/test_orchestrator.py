from __future__ import annotations

from dataclasses import replace

from app.config import Settings
from app.logs.event_logger import EventLogger
from app.orchestrator.orchestrator import AlfredOrchestrator
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import Intent, SkillResult, UserRequest
from app.orchestrator.skill_planner import CatalogSkillPlanner
from app.orchestrator.task_registry import SkillCatalog


def _orchestrator_with_catalog_planner(tmp_path):
    settings = replace(Settings.load(), openai_api_key="")
    logger = EventLogger(tmp_path, "test-request")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    planner = CatalogSkillPlanner(settings, catalog, prompts)
    return AlfredOrchestrator(catalog, logger, skill_planner=planner)


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


def test_catalog_planner_maps_pick_blue_marker_variants(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "pick blue marker",
        "pick this blue marker",
        "pick the blue marker",
        "pick up the blue marker",
        "pick marker",
        "pick the marker",
        "grab the marker",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["pick_blue_marker"]


def test_catalog_planner_maps_robot_pose_variants(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "go home",
        "return home",
        "move home",
        "go to home pose",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["go_home"]

    for text in [
        "go overlook",
        "go to overlook",
        "move to overlook",
        "go to the overlook pose",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["go_overlook"]


def test_catalog_planner_maps_task_history_questions(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    cases = [
        ("what tasks have you done so far?", None),
        ("tell me all tasks", None),
        ("what was the last task?", 1),
        ("what did you just do?", 1),
        ("tell me the last 2 tasks", 2),
    ]
    for text, expected_limit in cases:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["answer_task_history"]
        assert plan.skill_calls[0].arguments["question"] == text
        assert plan.skill_calls[0].arguments["limit"] == expected_limit


def test_catalog_planner_maps_pick_place_blue_marker_variants(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "pick and place the blue marker",
        "pick place blue marker",
        "pick up and drop the blue marker",
        "move the blue marker to the place location",
        "pick and place the marker",
        "drop the marker",
        "pick the sharpie",
        "pick the sharpy",
        "remove the marker from the table",
        "remove the sharpie from the table",
        "clean up the table",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["pick_place_blue_marker"]


def test_catalog_planner_maps_enzyme_experiment_variants(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "do enzyme experiments",
        "do the enzyme experiment",
        "run enzyme experiments",
        "do the biological experiments",
        "run the biological experiment",
        "perform the test tube experiment",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["enzyme_experiments"]


def test_catalog_planner_maps_compound_skill_sequences(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    cases = [
        (
            "go overlook and go home",
            ["go_overlook", "go_home"],
        ),
        (
            "what's on my desk and go home",
            ["answer_scene_question", "go_home"],
        ),
        (
            "pick and place the marker and go home",
            ["pick_place_blue_marker", "go_home"],
        ),
        (
            "pick the sharpy then go home",
            ["pick_place_blue_marker", "go_home"],
        ),
        (
            "do enzyme experiments and go home",
            ["enzyme_experiments", "go_home"],
        ),
    ]
    for text, expected_skills in cases:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == expected_skills

    desk_request = UserRequest(request_id="test-request", raw_text="what's on my desk and go home")
    desk_plan = orchestrator.create_plan(
        desk_request,
        orchestrator.classify_intent(desk_request.raw_text),
    )
    assert desk_plan.skill_calls[0].arguments["question"] == "what's on my desk"


def test_catalog_planner_maps_scene_questions_to_scene_qa(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "what color is the marker?",
        "what color is the pen?",
        "what color is the sharpie?",
        "how many markers are there?",
        "is there a cup on the table?",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["answer_scene_question"]
        assert plan.skill_calls[0].arguments["question"] == text


def test_catalog_planner_maps_note_reading_requests(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "read my notes",
        "what do my notes say?",
        "read the note",
        "what text is written on the paper?",
        "tell me what is written in my notes",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["read_notes"]
        assert plan.skill_calls[0].arguments["user_text"] == text


def test_catalog_planner_maps_general_conversation_fallback(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)

    for text in [
        "hi Alfred",
        "how are you today?",
        "what is your name?",
        "what skills do you have?",
        "what is the capital of the USA?",
        "Order soup ingredients",
    ]:
        request = UserRequest(request_id="test-request", raw_text=text)
        intent = orchestrator.classify_intent(request.raw_text)
        plan = orchestrator.create_plan(request, intent)

        assert plan.intent == Intent.RUN_SKILL
        assert [call.skill_name for call in plan.skill_calls] == ["general_conversation"]
        assert plan.skill_calls[0].arguments["question"] == text


def test_final_answer_uses_scene_qa_answer_text(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)
    request = UserRequest(request_id="test-request", raw_text="what color is the marker?")

    response = orchestrator.generate_final_answer(
        request,
        [
            SkillResult(
                skill_name="answer_scene_question",
                status="success",
                output={"answer_text": "The marker appears to be blue."},
            )
        ],
    )

    assert response.task_complete is True
    assert response.answer_text == "The marker appears to be blue."


def test_final_answer_combines_multiple_success_messages(tmp_path):
    orchestrator = _orchestrator_with_catalog_planner(tmp_path)
    request = UserRequest(request_id="test-request", raw_text="go overlook and go home")

    response = orchestrator.generate_final_answer(
        request,
        [
            SkillResult(
                skill_name="go_overlook",
                status="success",
                output={"message": "Moved to the overlook pose."},
            ),
            SkillResult(
                skill_name="go_home",
                status="success",
                output={"message": "Moved to the home pose."},
            ),
        ],
    )

    assert response.task_complete is True
    assert response.answer_text == "Moved to the overlook pose. Moved to the home pose."
