from __future__ import annotations

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
from app.orchestrator.task_registry import SkillCatalog


class AlfredOrchestrator:
    def __init__(self, skill_catalog: SkillCatalog, logger: EventLogger, camera_id: int = 0):
        self.skill_catalog = skill_catalog
        self.logger = logger
        self.camera_id = camera_id

    def classify_intent(self, text: str) -> IntentResult:
        normalized = text.lower()
        if "desk" in normalized or "what is on" in normalized:
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

    def create_plan(self, request: UserRequest, intent_result: IntentResult) -> OrchestratorPlan:
        if intent_result.intent == Intent.DESCRIBE_DESK and intent_result.confidence >= 0.8:
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

            response = FinalResponse(
                task_complete=bool(summary),
                answer_text=summary or "I captured an image, but I could not describe the desk clearly.",
                confidence=confidence if summary else 0.3,
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
            if len(summary.strip()) > 10:
                completion = CompletionResult(
                    task_complete=True,
                    reason="The VLM returned a usable desk description.",
                    next_action="none",
                )
            else:
                completion = CompletionResult(
                    task_complete=False,
                    reason="The vision result was empty or too short.",
                    next_action="retake_image",
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
