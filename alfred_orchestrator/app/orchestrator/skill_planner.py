from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.orchestrator.json_utils import extract_json_object
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.task_registry import SkillCatalog


@dataclass(frozen=True)
class SkillPlanChoice:
    skill_name: str
    arguments: dict[str, Any]
    confidence: float
    reason: str

    @property
    def is_unknown(self) -> bool:
        return self.skill_name == "UNKNOWN"


class CatalogSkillPlanner:
    def __init__(
        self,
        settings: Settings,
        skill_catalog: SkillCatalog,
        prompt_registry: PromptRegistry,
    ):
        self.settings = settings
        self.skill_catalog = skill_catalog
        self.prompt_registry = prompt_registry

    def choose_skill(self, user_text: str) -> SkillPlanChoice:
        deterministic = self._deterministic_choice(user_text)
        if deterministic is not None:
            print("[planner] Deterministic skill selection")
            print(f"[planner] User text: {user_text}")
            print(f"[planner] Selected skill: {deterministic.skill_name}")
            print(f"[planner] Arguments: {deterministic.arguments}")
            print(f"[planner] Confidence: {deterministic.confidence}")
            print(f"[planner] Reason: {deterministic.reason}")
            return deterministic

        llm_choice = self._choose_with_llm(user_text)
        if llm_choice is not None:
            print("[planner] LLM skill selection")
            print(f"[planner] User text: {user_text}")
            print(f"[planner] Selected skill: {llm_choice.skill_name}")
            print(f"[planner] Arguments: {llm_choice.arguments}")
            print(f"[planner] Confidence: {llm_choice.confidence}")
            print(f"[planner] Reason: {llm_choice.reason}")
            return llm_choice

        print("[planner] No skill matched")
        print(f"[planner] User text: {user_text}")
        return SkillPlanChoice(
            skill_name="UNKNOWN",
            arguments={},
            confidence=0.2,
            reason="No enabled skill matched the request.",
        )

    def _choose_with_llm(self, user_text: str) -> SkillPlanChoice | None:
        if not self.settings.openai_api_key:
            return None

        from openai import OpenAI

        prompt = self.prompt_registry.render(
            "skill_planner",
            {
                "user_text": user_text,
                "available_skills": self._available_skills_context(),
            },
        )
        print("[planner] LLM prompt:")
        print(prompt)
        try:
            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.responses.create(
                model=self.settings.openai_reasoning_model,
                input=prompt,
            )
            raw_text = response.output_text
            print("[planner] LLM raw output:")
            print(raw_text)
            parsed = extract_json_object(raw_text)
            return self._validate_choice(parsed)
        except Exception as exc:
            print(f"[planner] LLM skill selection failed: {exc}")
            return None

    def _validate_choice(self, raw_choice: dict[str, Any]) -> SkillPlanChoice:
        skill_name = str(raw_choice.get("skill_name", "")).strip()
        if skill_name == "UNKNOWN":
            return SkillPlanChoice(
                skill_name="UNKNOWN",
                arguments={},
                confidence=self._confidence(raw_choice.get("confidence", 0.0)),
                reason=str(raw_choice.get("reason", "")).strip(),
            )

        if skill_name not in self.skill_catalog.list_skills() or not self.skill_catalog.is_enabled(skill_name):
            return SkillPlanChoice(
                skill_name="UNKNOWN",
                arguments={},
                confidence=0.0,
                reason=f"Planner selected unavailable skill: {skill_name}",
            )

        raw_arguments = raw_choice.get("arguments") or {}
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        allowed_inputs = set(self.skill_catalog.get(skill_name).get("inputs", []))
        arguments = {
            key: value
            for key, value in raw_arguments.items()
            if key in allowed_inputs
        }
        return SkillPlanChoice(
            skill_name=skill_name,
            arguments=arguments,
            confidence=self._confidence(raw_choice.get("confidence", 0.0)),
            reason=str(raw_choice.get("reason", "")).strip(),
        )

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
                    "inputs": metadata.get("inputs", []),
                    "outputs": metadata.get("outputs", []),
                    "examples": metadata.get("examples", []),
                }
            )
        return skills

    def _deterministic_choice(self, user_text: str) -> SkillPlanChoice | None:
        normalized = " ".join(user_text.lower().split())
        if "marker" not in normalized:
            visual_choice = self._deterministic_visual_question(user_text, normalized)
            if visual_choice is not None:
                return visual_choice
            return None

        pick_words = ("pick", "grab", "fetch", "get")
        place_words = ("place", "drop", "deliver", "move", "put")
        wants_pick = any(word in normalized for word in pick_words)
        wants_place = any(word in normalized for word in place_words)
        if not wants_pick and not wants_place:
            return self._deterministic_visual_question(user_text, normalized)

        if wants_place and self._can_use("pick_place_blue_marker"):
            return SkillPlanChoice(
                skill_name="pick_place_blue_marker",
                arguments={},
                confidence=0.95,
                reason="Detected a request to pick and place the marker; defaulting to blue.",
            )
        if wants_pick and self._can_use("pick_blue_marker"):
            return SkillPlanChoice(
                skill_name="pick_blue_marker",
                arguments={},
                confidence=0.95,
                reason="Detected a request to pick up the marker; defaulting to blue.",
            )
        return self._deterministic_visual_question(user_text, normalized)

    def _deterministic_visual_question(
        self,
        user_text: str,
        normalized: str,
    ) -> SkillPlanChoice | None:
        question_starters = ("what", "which", "how many", "is there", "are there", "do you see")
        visual_words = (
            "color",
            "colour",
            "count",
            "many",
            "see",
            "visible",
            "marker",
            "pen",
            "object",
            "cup",
            "desk",
            "table",
        )
        looks_like_question = "?" in normalized or normalized.startswith(question_starters)
        if (
            looks_like_question
            and any(word in normalized for word in visual_words)
            and self._can_use("answer_scene_question")
        ):
            return SkillPlanChoice(
                skill_name="answer_scene_question",
                arguments={"question": user_text},
                confidence=0.9,
                reason="Detected a visual question about the current scene.",
            )
        return None

    def _can_use(self, skill_name: str) -> bool:
        return (
            skill_name in self.skill_catalog.list_skills()
            and self.skill_catalog.is_enabled(skill_name)
        )

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))
