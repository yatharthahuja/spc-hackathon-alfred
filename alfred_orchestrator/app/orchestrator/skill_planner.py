from __future__ import annotations

import re
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


@dataclass(frozen=True)
class SkillPlanSequence:
    choices: list[SkillPlanChoice]
    confidence: float
    reason: str

    @property
    def is_unknown(self) -> bool:
        return not self.choices or all(choice.is_unknown for choice in self.choices)


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
        sequence = self.choose_skills(user_text)
        if sequence.choices:
            return sequence.choices[0]
        return SkillPlanChoice(
            skill_name="UNKNOWN",
            arguments={},
            confidence=0.2,
            reason="No enabled skill matched the request.",
        )

    def choose_skills(self, user_text: str) -> SkillPlanSequence:
        deterministic = self._deterministic_sequence(user_text)
        if deterministic is not None:
            print("[planner] Deterministic skill sequence selection")
            print(f"[planner] User text: {user_text}")
            self._print_sequence(deterministic.choices)
            print(f"[planner] Sequence confidence: {deterministic.confidence}")
            print(f"[planner] Sequence reason: {deterministic.reason}")
            return deterministic

        llm_sequence = self._choose_sequence_with_llm(user_text)
        if llm_sequence is not None:
            if not llm_sequence.is_unknown:
                print("[planner] LLM skill sequence selection")
                print(f"[planner] User text: {user_text}")
                self._print_sequence(llm_sequence.choices)
                print(f"[planner] Sequence confidence: {llm_sequence.confidence}")
                print(f"[planner] Sequence reason: {llm_sequence.reason}")
                return llm_sequence
            print("[planner] LLM returned UNKNOWN; trying conversation fallback")
            print(f"[planner] Reason: {llm_sequence.reason}")

        conversation_choice = self._conversation_choice(user_text)
        if conversation_choice is not None:
            print("[planner] Falling back to general conversation")
            print(f"[planner] User text: {user_text}")
            print(f"[planner] Selected skill: {conversation_choice.skill_name}")
            return SkillPlanSequence(
                choices=[conversation_choice],
                confidence=conversation_choice.confidence,
                reason=conversation_choice.reason,
            )

        print("[planner] No skill matched")
        print(f"[planner] User text: {user_text}")
        unknown = SkillPlanChoice(
            skill_name="UNKNOWN",
            arguments={},
            confidence=0.2,
            reason="No enabled skill matched the request.",
        )
        return SkillPlanSequence(
            choices=[unknown],
            confidence=unknown.confidence,
            reason=unknown.reason,
        )

    def _choose_sequence_with_llm(self, user_text: str) -> SkillPlanSequence | None:
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
            return self._validate_sequence(parsed)
        except Exception as exc:
            print(f"[planner] LLM skill selection failed: {exc}")
            return None

    def _validate_sequence(self, raw_plan: dict[str, Any]) -> SkillPlanSequence:
        raw_skill_calls = raw_plan.get("skill_calls")
        if not isinstance(raw_skill_calls, list):
            choice = self._validate_choice(raw_plan)
            return SkillPlanSequence(
                choices=[choice],
                confidence=choice.confidence,
                reason=choice.reason,
            )

        choices = [
            self._validate_choice(raw_choice)
            for raw_choice in raw_skill_calls
            if isinstance(raw_choice, dict)
        ]
        choices = [choice for choice in choices if not choice.is_unknown]
        if not choices:
            unknown = SkillPlanChoice(
                skill_name="UNKNOWN",
                arguments={},
                confidence=self._confidence(raw_plan.get("confidence", 0.0)),
                reason=str(raw_plan.get("reason", "")).strip(),
            )
            return SkillPlanSequence(
                choices=[unknown],
                confidence=unknown.confidence,
                reason=unknown.reason,
            )

        return SkillPlanSequence(
            choices=choices,
            confidence=min(choice.confidence for choice in choices),
            reason=str(raw_plan.get("reason", "")).strip()
            or "; ".join(choice.reason for choice in choices if choice.reason),
        )

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

    def _deterministic_sequence(self, user_text: str) -> SkillPlanSequence | None:
        clauses = self._split_compound_requests(user_text)
        if len(clauses) <= 1:
            choice = self._deterministic_choice(user_text)
            if choice is None:
                return None
            return SkillPlanSequence(
                choices=[choice],
                confidence=choice.confidence,
                reason=choice.reason,
            )

        choices = []
        for clause in clauses:
            choice = self._deterministic_choice(clause)
            if choice is None or choice.is_unknown:
                return None
            choices.append(choice)

        return SkillPlanSequence(
            choices=choices,
            confidence=min(choice.confidence for choice in choices),
            reason="Detected an ordered compound request with multiple supported skills.",
        )

    @staticmethod
    def _split_compound_requests(user_text: str) -> list[str]:
        protected = re.sub(
            r"\bpick\s+and\s+place\b",
            "pick__and__place",
            user_text,
            flags=re.IGNORECASE,
        )
        protected = re.sub(
            r"\bpick\s+up\s+and\s+drop\b",
            "pick__up__and__drop",
            protected,
            flags=re.IGNORECASE,
        )
        protected = re.sub(
            r"\bpick\s+and\s+drop\b",
            "pick__and__drop",
            protected,
            flags=re.IGNORECASE,
        )
        protected = re.sub(r"\band\s+then\b", " then ", protected, flags=re.IGNORECASE)
        raw_parts = re.split(
            r"\s*(?:,|;|\bthen\b|\bafter that\b)\s*|\s+and\s+",
            protected,
            flags=re.IGNORECASE,
        )
        restored_parts = []
        for part in raw_parts:
            restored = (
                part.replace("pick__and__place", "pick and place")
                .replace("pick__up__and__drop", "pick up and drop")
                .replace("pick__and__drop", "pick and drop")
                .strip()
            )
            if restored:
                restored_parts.append(restored)
        return restored_parts

    def _deterministic_choice(self, user_text: str) -> SkillPlanChoice | None:
        normalized = " ".join(user_text.lower().split())
        history_choice = self._deterministic_history_choice(user_text, normalized)
        if history_choice is not None:
            return history_choice

        pose_choice = self._deterministic_pose_choice(normalized)
        if pose_choice is not None:
            return pose_choice

        read_notes_choice = self._deterministic_read_notes_choice(user_text, normalized)
        if read_notes_choice is not None:
            return read_notes_choice

        enzyme_choice = self._deterministic_enzyme_experiment_choice(normalized)
        if enzyme_choice is not None:
            return enzyme_choice

        cleanup_choice = self._deterministic_table_cleanup_choice(normalized)
        if cleanup_choice is not None:
            return cleanup_choice

        marker_object_terms = ("marker", "sharpie", "sharpy")
        mentions_marker_object = any(term in normalized for term in marker_object_terms)
        if not mentions_marker_object:
            visual_choice = self._deterministic_visual_question(user_text, normalized)
            if visual_choice is not None:
                return visual_choice
            return None

        pick_words = ("pick", "grab", "fetch", "get")
        place_words = ("place", "drop", "deliver", "move", "put", "remove", "clean", "clear")
        wants_pick = any(word in normalized for word in pick_words)
        wants_place = any(word in normalized for word in place_words)
        if not wants_pick and not wants_place:
            return self._deterministic_visual_question(user_text, normalized)

        mentions_sharpie = "sharpie" in normalized or "sharpy" in normalized
        if mentions_sharpie and self._can_use("pick_place_blue_marker"):
            return SkillPlanChoice(
                skill_name="pick_place_blue_marker",
                arguments={},
                confidence=0.95,
                reason="Detected a request to pick up or remove the Sharpie; using the blue marker pick-and-place sequence.",
            )
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

    def _deterministic_table_cleanup_choice(self, normalized: str) -> SkillPlanChoice | None:
        cleanup_words = ("clean", "cleanup", "clear", "tidy", "remove")
        table_words = ("table", "desk", "workspace", "work surface")
        if (
            any(word in normalized for word in cleanup_words)
            and any(word in normalized for word in table_words)
            and self._can_use("pick_place_blue_marker")
        ):
            return SkillPlanChoice(
                skill_name="pick_place_blue_marker",
                arguments={},
                confidence=0.9,
                reason="Detected a table cleanup request; using the blue marker pick-and-place sequence.",
            )
        return None

    def _deterministic_enzyme_experiment_choice(self, normalized: str) -> SkillPlanChoice | None:
        experiment_words = ("experiment", "experiments", "lab", "protocol")
        domain_words = ("enzyme", "biological", "biology", "bio", "test tube", "tube")
        action_words = ("do", "run", "perform", "start", "execute", "begin")
        if (
            any(word in normalized for word in experiment_words)
            and any(word in normalized for word in domain_words)
            and any(word in normalized for word in action_words)
            and self._can_use("enzyme_experiments")
        ):
            return SkillPlanChoice(
                skill_name="enzyme_experiments",
                arguments={},
                confidence=0.95,
                reason="Detected a request to run the scripted enzyme or biological experiment trajectory.",
            )
        return None

    def _conversation_choice(self, user_text: str) -> SkillPlanChoice | None:
        if not self._can_use("general_conversation"):
            return None
        return SkillPlanChoice(
            skill_name="general_conversation",
            arguments={"question": user_text},
            confidence=0.75,
            reason="No specific skill matched, so using general conversation.",
        )

    def _deterministic_visual_question(
        self,
        user_text: str,
        normalized: str,
    ) -> SkillPlanChoice | None:
        question_starters = (
            "what",
            "what's",
            "whats",
            "which",
            "how many",
            "is there",
            "are there",
            "do you see",
        )
        visual_words = (
            "color",
            "colour",
            "count",
            "many",
            "see",
            "visible",
            "marker",
            "sharpie",
            "sharpy",
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

    def _deterministic_read_notes_choice(
        self,
        user_text: str,
        normalized: str,
    ) -> SkillPlanChoice | None:
        note_words = ("note", "notes", "paper", "written", "writing", "text")
        read_words = ("read", "say", "says", "written", "text", "transcribe")
        if (
            any(word in normalized for word in note_words)
            and any(word in normalized for word in read_words)
            and self._can_use("read_notes")
        ):
            return SkillPlanChoice(
                skill_name="read_notes",
                arguments={"user_text": user_text},
                confidence=0.95,
                reason="Detected a request to read visible notes.",
            )
        return None

    def _deterministic_history_choice(
        self,
        user_text: str,
        normalized: str,
    ) -> SkillPlanChoice | None:
        history_words = (
            "task",
            "tasks",
            "done",
            "history",
            "what did you do",
            "what have you done",
            "what did you just do",
        )
        if not any(word in normalized for word in history_words):
            return None
        if not self._can_use("answer_task_history"):
            return None

        limit = None
        last_match = re.search(r"\blast\s+(\d+)\s+tasks?\b", normalized)
        if last_match:
            limit = int(last_match.group(1))
        elif re.search(r"\b(last task|last thing|just do|previous task)\b", normalized):
            limit = 1

        return SkillPlanChoice(
            skill_name="answer_task_history",
            arguments={"question": user_text, "limit": limit},
            confidence=0.95,
            reason="Detected a question about completed task history.",
        )

    def _deterministic_pose_choice(self, normalized: str) -> SkillPlanChoice | None:
        home_phrases = (
            "go home",
            "return home",
            "move home",
            "go to home",
            "home pose",
        )
        if any(phrase in normalized for phrase in home_phrases) and self._can_use("go_home"):
            return SkillPlanChoice(
                skill_name="go_home",
                arguments={},
                confidence=0.95,
                reason="Detected a request to move to the home pose.",
            )

        overlook_phrases = (
            "go overlook",
            "go to overlook",
            "move to overlook",
            "overlook pose",
            "look at the table",
        )
        if any(phrase in normalized for phrase in overlook_phrases) and self._can_use("go_overlook"):
            return SkillPlanChoice(
                skill_name="go_overlook",
                arguments={},
                confidence=0.95,
                reason="Detected a request to move to the overlook pose.",
            )

        return None

    def _can_use(self, skill_name: str) -> bool:
        return (
            skill_name in self.skill_catalog.list_skills()
            and self.skill_catalog.is_enabled(skill_name)
        )

    @staticmethod
    def _print_sequence(choices: list[SkillPlanChoice]) -> None:
        for index, choice in enumerate(choices, start=1):
            print(f"[planner] Step {index} skill: {choice.skill_name}")
            print(f"[planner] Step {index} arguments: {choice.arguments}")
            print(f"[planner] Step {index} confidence: {choice.confidence}")
            print(f"[planner] Step {index} reason: {choice.reason}")

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))
