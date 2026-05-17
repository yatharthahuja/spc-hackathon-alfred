from __future__ import annotations

from typing import Any, Callable

from app.config import Settings
from app.memory.session_memory import SessionMemory, TASK_HISTORY
from app.orchestrator.json_utils import extract_json_object
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import SkillResult
from app.orchestrator.task_registry import SkillCatalog
from app.skills.base import Skill, failure, success


class GeneralConversationSkill(Skill):
    name = "general_conversation"

    def __init__(
        self,
        settings: Settings,
        prompt_registry: PromptRegistry,
        skill_catalog: SkillCatalog,
        task_history: SessionMemory = TASK_HISTORY,
        openai_client_factory: Callable[..., Any] | None = None,
    ):
        self.settings = settings
        self.prompt_registry = prompt_registry
        self.skill_catalog = skill_catalog
        self.task_history = task_history
        self.openai_client_factory = openai_client_factory

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            question = str(kwargs.get("question") or "").strip()
            if not question:
                raise ValueError("Question is required for general conversation")

            available_skills = self._available_skills_context()
            task_history = self.task_history.all()
            prompt = self.prompt_registry.render(
                "general_conversation",
                {
                    "user_text": question,
                    "available_skills": available_skills,
                    "task_history": task_history,
                },
            )
            print("[general_conversation] Question:")
            print(question)
            print(f"[general_conversation] Available skill count: {len(available_skills)}")
            print(f"[general_conversation] Task history count: {len(task_history)}")
            print("[general_conversation] Prompt sent to LLM:")
            print(prompt)

            if not self.settings.openai_api_key:
                answer_text = self._fallback_answer(question, available_skills)
                return success(
                    self.name,
                    {
                        "answer_text": answer_text,
                        "confidence": 0.65,
                        "raw_response": "",
                    },
                )

            if self.openai_client_factory is None:
                from openai import OpenAI

                client = OpenAI(api_key=self.settings.openai_api_key)
            else:
                client = self.openai_client_factory(api_key=self.settings.openai_api_key)
            response = client.responses.create(
                model=self.settings.openai_reasoning_model,
                input=prompt,
            )
            raw_text = response.output_text
            print("[general_conversation] Raw LLM output:")
            print(raw_text)
            parsed = extract_json_object(raw_text)
            answer_text = str(parsed.get("answer_text", "")).strip()
            if not answer_text:
                answer_text = self._fallback_answer(question, available_skills)
            return success(
                self.name,
                {
                    "answer_text": answer_text,
                    "confidence": self._confidence(parsed.get("confidence", 0.8)),
                    "raw_response": raw_text,
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    def _available_skills_context(self) -> list[dict[str, Any]]:
        skills = []
        for name in self.skill_catalog.list_skills():
            metadata = self.skill_catalog.get(name)
            if not self.skill_catalog.is_enabled(name):
                continue
            skills.append(
                {
                    "skill_name": name,
                    "type": metadata.get("type", "digital"),
                    "description": metadata.get("description", ""),
                    "examples": metadata.get("examples", []),
                }
            )
        return skills

    @staticmethod
    def _fallback_answer(question: str, available_skills: list[dict[str, Any]]) -> str:
        normalized = question.lower()
        if "skill" in normalized or "can you do" in normalized:
            skill_names = ", ".join(str(skill["skill_name"]) for skill in available_skills)
            return f"I can chat and help through these registered skills: {skill_names}."
        if "name" in normalized:
            return "I'm Alfred, your smart, slightly funny robot assistant."
        if "how are" in normalized:
            return "I'm doing great, calibrated, caffeinated in spirit, and ready to help."
        return (
            "I'm Alfred, your all-in-one robot assistant. I can chat, answer questions, "
            "inspect the scene, read notes, remember recent tasks, and run safe scripted robot motions."
        )

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))
