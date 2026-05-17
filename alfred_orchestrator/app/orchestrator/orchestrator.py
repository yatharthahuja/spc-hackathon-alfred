from __future__ import annotations

import re
from typing import List

from app.logs.event_logger import EventLogger
from app.orchestrator.schemas import (
    CompletionResult,
    FinalResponse,
    Intent,
    IntentResult,
    OrchestratorPlan,
    SkillCall,
    SkillResult,
    TaskStep,
    UserRequest,
)
from app.orchestrator.skill_planner import CatalogSkillPlanner, SkillPlanChoice, SkillPlanSequence
from app.orchestrator.task_registry import SkillCatalog


class AlfredOrchestrator:
    def __init__(
        self,
        skill_catalog: SkillCatalog,
        logger: EventLogger,
        camera_id: int = 0,
        skill_planner: CatalogSkillPlanner | None = None,
    ):
        self.skill_catalog = skill_catalog
        self.logger = logger
        self.camera_id = camera_id
        self.skill_planner = skill_planner

    def classify_intent(self, text: str) -> IntentResult:
        normalized = text.lower()
        if self._is_desk_inspection_request(normalized):
            result = IntentResult(
                intent=Intent.DESCRIBE_DESK,
                confidence=0.95,
                reason="Rules-first classifier matched a desk inspection request.",
            )
        else:
            result = IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.2,
                reason="No supported MVP intent matched the request.",
            )
        self.logger.log(
            stage="intent_classification",
            status="success",
            input_data={"text": text},
            output_data=result.model_dump(mode="json"),
        )
        return result

    def _is_desk_inspection_request(self, normalized_text: str) -> bool:
        compact = re.sub(r"\s+", " ", normalized_text).strip()
        if not compact:
            return False

        desk_targets = ("desk", "table", "workbench", "workspace", "work surface")
        if any(target in compact for target in desk_targets):
            inspection_words = (
                "what",
                "see",
                "look",
                "inspect",
                "describe",
                "identify",
                "objects",
                "items",
                "things",
            )
            if any(word in compact for word in inspection_words):
                return True

        direct_patterns = (
            r"\bwhat(?:'s| is)? on (?:the |my |this |that )?(?:desk|table|workbench)\b",
            r"\bwhat (?:can|do) you see\b",
            r"\blook at (?:the |my )?(?:desk|table|workspace|work surface)\b",
            r"\binspect (?:the |my )?(?:desk|table|workspace|work surface)\b",
            r"\bdescribe (?:the |my )?(?:desk|table|workspace|work surface)\b",
        )
        return any(re.search(pattern, compact) for pattern in direct_patterns)

    def create_plan(self, request: UserRequest, intent_result: IntentResult) -> OrchestratorPlan:
        use_catalog_planner = self.skill_planner is not None and (
            intent_result.intent != Intent.DESCRIBE_DESK
            or intent_result.confidence < 0.8
            or self._is_compound_request(request.raw_text)
        )
        if intent_result.intent == Intent.DESCRIBE_DESK and intent_result.confidence >= 0.8 and not use_catalog_planner:
            plan = OrchestratorPlan(
                goal="Describe what is on the user's desk",
                intent=Intent.DESCRIBE_DESK,
                tasks=[
                    TaskStep(
                        task_id="t1",
                        agent="desk_inspection_agent",
                        required_skills=[
                            "capture_wrist_camera_image",
                            "describe_image_with_vlm",
                        ],
                    )
                ],
                skill_calls=[
                    SkillCall(
                        skill_name="capture_wrist_camera_image",
                        arguments={"camera_id": self.camera_id, "max_attempts": 3},
                    ),
                    SkillCall(
                        skill_name="describe_image_with_vlm",
                        arguments={
                            "image_path": "$capture_wrist_camera_image.image_path",
                            "question": "What objects are visible on the desk?",
                            "user_text": request.raw_text,
                        },
                    ),
                ],
                completion_criteria=[
                    "Image captured",
                    "Desk contents described",
                    "User-facing answer generated",
                ],
            )
        elif self.skill_planner is not None:
            sequence = self.skill_planner.choose_skills(request.raw_text)
            plan = self._plan_from_skill_sequence(request, sequence)
        else:
            plan = OrchestratorPlan(
                goal="Ask user for a supported request",
                intent=Intent.UNKNOWN,
                tasks=[],
                skill_calls=[],
                completion_criteria=["User received unsupported intent message"],
            )

        self.logger.write_json("plan.json", plan.model_dump(mode="json"))
        self.logger.log(
            stage="planning",
            status="success",
            input_data={
                "request": request.model_dump(mode="json"),
                "intent_result": intent_result.model_dump(mode="json"),
            },
            output_data=plan.model_dump(mode="json"),
        )
        return plan

    @staticmethod
    def _is_compound_request(text: str) -> bool:
        protected = re.sub(r"\bpick\s+and\s+place\b", "pick__and__place", text, flags=re.IGNORECASE)
        protected = re.sub(
            r"\bpick\s+up\s+and\s+drop\b",
            "pick__up__and__drop",
            protected,
            flags=re.IGNORECASE,
        )
        protected = re.sub(r"\bpick\s+and\s+drop\b", "pick__and__drop", protected, flags=re.IGNORECASE)
        return bool(re.search(r"\b(?:and then|then|after that)\b|[,;]|\s+and\s+", protected, re.IGNORECASE))

    def _plan_from_skill_choice(
        self,
        request: UserRequest,
        choice: SkillPlanChoice,
    ) -> OrchestratorPlan:
        return self._plan_from_skill_sequence(
            request,
            SkillPlanSequence(
                choices=[choice],
                confidence=choice.confidence,
                reason=choice.reason,
            ),
        )

    def _plan_from_skill_sequence(
        self,
        request: UserRequest,
        sequence: SkillPlanSequence,
    ) -> OrchestratorPlan:
        choices = [
            choice
            for choice in sequence.choices
            if not choice.is_unknown and choice.confidence >= 0.5
        ]
        self.logger.log(
            stage="skill_planning",
            status="success" if choices else "info",
            input_data={"text": request.raw_text},
            output_data={
                "skill_calls": [
                    {
                        "skill_name": choice.skill_name,
                        "arguments": choice.arguments,
                        "confidence": choice.confidence,
                        "reason": choice.reason,
                    }
                    for choice in sequence.choices
                ],
                "confidence": sequence.confidence,
                "reason": sequence.reason,
            },
        )
        if not choices:
            return OrchestratorPlan(
                goal="Ask user for a supported request",
                intent=Intent.UNKNOWN,
                tasks=[],
                skill_calls=[],
                completion_criteria=["User received unsupported intent message"],
            )

        return OrchestratorPlan(
            goal=request.raw_text,
            intent=Intent.RUN_SKILL,
            tasks=[
                TaskStep(
                    task_id="t1",
                    agent="catalog_skill_planner",
                    required_skills=[choice.skill_name for choice in choices],
                )
            ],
            skill_calls=[
                SkillCall(
                    skill_name=choice.skill_name,
                    arguments=choice.arguments,
                )
                for choice in choices
            ],
            completion_criteria=["Selected skill completed successfully"],
        )

    def generate_final_answer(
        self,
        request: UserRequest,
        results: List[SkillResult],
    ) -> FinalResponse:
        errors = [result for result in results if result.status == "error"]
        if errors:
            first_error = errors[0]
            response = FinalResponse(
                task_complete=False,
                answer_text=f"I ran into a problem while using {first_error.skill_name}: {first_error.error}",
                confidence=0.0,
            )
        elif not results:
            response = FinalResponse(
                task_complete=False,
                answer_text="I do not know how to do that yet.",
                confidence=0.2,
            )
        else:
            vlm_result = next(
                (result for result in results if result.skill_name == "describe_image_with_vlm"),
                None,
            )
            summary = ""
            confidence = 0.7
            if vlm_result:
                summary = str(vlm_result.output.get("spoken_summary", ""))
                confidence = float(vlm_result.output.get("confidence", 0.8))

            if vlm_result and summary:
                response = FinalResponse(
                    task_complete=True,
                    answer_text=summary,
                    confidence=confidence,
                )
            elif vlm_result:
                response = FinalResponse(
                    task_complete=False,
                    answer_text="I captured an image, but I could not describe the desk clearly.",
                    confidence=0.3,
                )
            else:
                successful_results = [
                    result for result in results if result.status == "success"
                ]
                messages = [
                    str(
                        result.output.get("answer_text")
                        or result.output.get("message")
                        or ""
                    ).strip()
                    for result in successful_results
                ]
                messages = [message for message in messages if message]
                successful_result = successful_results[0] if successful_results else None
                response = FinalResponse(
                    task_complete=successful_result is not None,
                    answer_text=" ".join(messages) if messages else "Done.",
                    confidence=0.9 if successful_result else 0.3,
                )

        self.logger.write_text("final_answer.txt", response.answer_text)
        self.logger.log(
            stage="response_generation",
            status="success" if response.task_complete else "info",
            input_data={
                "request": request.model_dump(mode="json"),
                "results": [result.model_dump(mode="json") for result in results],
            },
            output_data=response.model_dump(mode="json"),
        )
        return response

    def evaluate_completion(
        self,
        request: UserRequest,
        results: List[SkillResult],
    ) -> CompletionResult:
        errors = [result for result in results if result.status == "error"]
        if not results:
            completion = CompletionResult(
                task_complete=False,
                reason="No executable plan was produced for this request.",
                next_action="ask_user",
            )
        elif errors:
            completion = CompletionResult(
                task_complete=False,
                reason=f"Skill failed: {errors[0].skill_name}",
                next_action="ask_user",
            )
        else:
            vlm_result = next(
                (result for result in results if result.skill_name == "describe_image_with_vlm"),
                None,
            )
            summary = str(vlm_result.output.get("spoken_summary", "")) if vlm_result else ""
            if vlm_result and len(summary.strip()) > 10:
                completion = CompletionResult(
                    task_complete=True,
                    reason="The VLM returned a usable desk description.",
                    next_action="none",
                )
            elif vlm_result:
                completion = CompletionResult(
                    task_complete=False,
                    reason="The vision result was empty or too short.",
                    next_action="retake_image",
                )
            else:
                completion = CompletionResult(
                    task_complete=True,
                    reason="The planned skill completed successfully.",
                    next_action="none",
                )

        self.logger.write_json("completion.json", completion.model_dump(mode="json"))
        self.logger.log(
            stage="completion_check",
            status="success" if completion.task_complete else "info",
            input_data={
                "request": request.model_dump(mode="json"),
                "results": [result.model_dump(mode="json") for result in results],
            },
            output_data=completion.model_dump(mode="json"),
        )
        return completion
