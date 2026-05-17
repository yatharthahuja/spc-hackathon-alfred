from __future__ import annotations

from typing import Any

from app.config import Settings
from app.memory.session_memory import SessionMemory, TASK_HISTORY
from app.orchestrator.json_utils import extract_json_object
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class AnswerTaskHistorySkill(Skill):
    name = "answer_task_history"

    def __init__(
        self,
        settings: Settings,
        prompt_registry: PromptRegistry,
        task_history: SessionMemory = TASK_HISTORY,
    ):
        self.settings = settings
        self.prompt_registry = prompt_registry
        self.task_history = task_history

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            question = str(kwargs.get("question") or "What tasks have you done so far?").strip()
            limit = self._parse_limit(kwargs.get("limit"))
            all_tasks = self.task_history.all()
            selected_tasks = all_tasks[-limit:] if limit is not None else all_tasks
            prompt = self.prompt_registry.render(
                "task_history_response",
                {
                    "user_text": question,
                    "task_history": selected_tasks,
                    "selected_task_count": len(selected_tasks),
                    "total_task_count": len(all_tasks),
                },
            )

            print("[answer_task_history] Question:")
            print(question)
            print(f"[answer_task_history] Total task count: {len(all_tasks)}")
            print(f"[answer_task_history] Selected task count: {len(selected_tasks)}")
            print("[answer_task_history] Prompt sent to LLM:")
            print(prompt)

            if not selected_tasks:
                answer_text = "I have not completed any tasks yet."
                return success(
                    self.name,
                    {
                        "answer_text": answer_text,
                        "confidence": 0.95,
                        "task_count": len(all_tasks),
                        "selected_task_count": 0,
                        "raw_response": "",
                    },
                )

            if not self.settings.openai_api_key:
                answer_text = self._fallback_answer(selected_tasks, limit)
                return success(
                    self.name,
                    {
                        "answer_text": answer_text,
                        "confidence": 0.7,
                        "task_count": len(all_tasks),
                        "selected_task_count": len(selected_tasks),
                        "raw_response": "",
                    },
                )

            from openai import OpenAI

            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.responses.create(
                model=self.settings.openai_reasoning_model,
                input=prompt,
            )
            raw_text = response.output_text
            print("[answer_task_history] Raw LLM output:")
            print(raw_text)
            parsed = extract_json_object(raw_text)
            answer_text = str(parsed.get("answer_text", "")).strip()
            if not answer_text:
                answer_text = self._fallback_answer(selected_tasks, limit)
            confidence = self._confidence(parsed.get("confidence", 0.8))
            return success(
                self.name,
                {
                    "answer_text": answer_text,
                    "confidence": confidence,
                    "task_count": len(all_tasks),
                    "selected_task_count": len(selected_tasks),
                    "raw_response": raw_text,
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    @staticmethod
    def _parse_limit(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    @staticmethod
    def _fallback_answer(tasks: list[dict[str, Any]], limit: int | None) -> str:
        if not tasks:
            return "I have not completed any tasks yet."
        prefix = "The last task was" if limit == 1 and len(tasks) == 1 else "The tasks were"
        summaries = []
        for task in tasks:
            user_text = str(task.get("user_text", "a task"))
            answer_text = str(task.get("answer_text", "")).strip()
            if answer_text:
                summaries.append(f"{user_text}: {answer_text}")
            else:
                summaries.append(user_text)
        return f"{prefix}: " + "; ".join(summaries) + "."

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))
