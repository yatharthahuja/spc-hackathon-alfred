from __future__ import annotations

from dataclasses import replace

from app.config import Settings
from app.memory.session_memory import SessionMemory
from app.orchestrator.prompt_registry import PromptRegistry
from app.skills.task_history import AnswerTaskHistorySkill


def test_task_history_skill_answers_from_recent_memory_without_openai():
    settings = replace(Settings.load(), openai_api_key="")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    memory = SessionMemory()
    memory.add(
        {
            "user_text": "go home",
            "skill_names": ["go_home"],
            "answer_text": "Moved to the home pose.",
        }
    )
    memory.add(
        {
            "user_text": "pick the marker",
            "skill_names": ["pick_blue_marker"],
            "answer_text": "Picked up the blue marker.",
        }
    )
    skill = AnswerTaskHistorySkill(settings, prompts, task_history=memory)

    result = skill.run(question="what was the last task?", limit=1)

    assert result.status == "success"
    assert result.output["task_count"] == 2
    assert result.output["selected_task_count"] == 1
    assert "pick the marker" in result.output["answer_text"]
    assert "Picked up the blue marker" in result.output["answer_text"]


def test_task_history_skill_handles_empty_history():
    settings = replace(Settings.load(), openai_api_key="")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    skill = AnswerTaskHistorySkill(settings, prompts, task_history=SessionMemory())

    result = skill.run(question="what tasks have you done so far?")

    assert result.status == "success"
    assert result.output["answer_text"] == "I have not completed any tasks yet."
    assert result.output["task_count"] == 0
    assert result.output["selected_task_count"] == 0
