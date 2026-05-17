from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.config import Settings
from app.memory.session_memory import SessionMemory
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.task_registry import SkillCatalog
from app.skills.conversation import GeneralConversationSkill


class FakeResponses:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return type(
            "FakeResponse",
            (),
            {
                "output_text": (
                    '{"answer_text": "Hi, I am Alfred, your robotics-savvy assistant.", '
                    '"confidence": 0.92}'
                )
            },
        )()


class FakeOpenAIClient:
    def __init__(self, **_kwargs: Any):
        self.responses = FakeResponses()


def test_general_conversation_uses_prompt_history_and_skills():
    settings = replace(Settings.load(), openai_api_key="test-key")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    history = SessionMemory()
    history.add({"user_text": "go home", "answer_text": "Moved to the home pose."})
    created_clients: list[FakeOpenAIClient] = []

    def client_factory(**kwargs: Any) -> FakeOpenAIClient:
        client = FakeOpenAIClient(**kwargs)
        created_clients.append(client)
        return client

    skill = GeneralConversationSkill(
        settings=settings,
        prompt_registry=prompts,
        skill_catalog=catalog,
        task_history=history,
        openai_client_factory=client_factory,
    )

    result = skill.run(question="hi Alfred")

    assert result.status == "success"
    assert result.output["answer_text"] == "Hi, I am Alfred, your robotics-savvy assistant."
    assert result.output["confidence"] == 0.92
    request = created_clients[0].responses.calls[0]
    assert request["model"] == settings.openai_reasoning_model
    assert "go home" in request["input"]
    assert "pick_blue_marker" in request["input"]


def test_general_conversation_has_offline_fallback_for_capabilities():
    settings = replace(Settings.load(), openai_api_key="")
    prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
    skill = GeneralConversationSkill(settings, prompts, catalog, task_history=SessionMemory())

    result = skill.run(question="what skills do you have?")

    assert result.status == "success"
    assert "registered skills" in result.output["answer_text"]
    assert "general_conversation" in result.output["answer_text"]
