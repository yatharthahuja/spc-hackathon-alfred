from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from uuid import uuid4

from app.config import Settings
from app.execution.executor import SkillExecutor
from app.execution.safety import SafetyGate
from app.execution.skill_router import SkillRouter
from app.hardware.resources import HardwareContext
from app.logs.event_logger import EventLogger
from app.memory.session_memory import TASK_HISTORY
from app.orchestrator.announcements import completion_announcement, start_announcement
from app.orchestrator.orchestrator import AlfredOrchestrator
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.schemas import (
    CompletionResult,
    FinalResponse,
    OrchestratorPlan,
    SkillCall,
    SkillResult,
    UserRequest,
    utc_now_iso,
)
from app.orchestrator.skill_planner import CatalogSkillPlanner
from app.orchestrator.task_registry import SkillCatalog
from app.skills.amazon import AmazonAddToCartSkill, AmazonSearchSkill
from app.skills.camera import CaptureWristCameraImageSkill
from app.skills.arm_motion import MoveArmNoopSkill
from app.skills.conversation import GeneralConversationSkill
from app.skills.listen import ListenSkill
from app.skills.marker import (
    EnzymeExperimentsSkill,
    GoHomeSkill,
    GoOverlookSkill,
    PickBlueMarkerSkill,
    PickPlaceBlueMarkerSkill,
)
from app.skills.ordering import OrderingSkill
from app.skills.read_notes import ReadNotesSkill
from app.skills.scene_qa import AnswerSceneQuestionSkill
from app.skills.speech_to_text import SpeechToTextSkill
from app.skills.task_history import AnswerTaskHistorySkill
from app.skills.text_to_speech import TextToSpeechSkill
from app.skills.vlm_describe import DescribeImageWithVLMSkill


@dataclass
class PipelineResult:
    request: UserRequest
    response: FinalResponse
    completion: CompletionResult
    skill_results: List[SkillResult]
    run_dir: Path


class AlfredRuntime:
    def __init__(
        self,
        settings: Settings,
        run_dir: Optional[Path] = None,
        hardware_context: Optional[HardwareContext] = None,
        connect_hardware: bool = True,
    ):
        self.settings = settings
        self.settings.print_model_configuration()
        self.request_id = str(uuid4())
        self.run_dir = run_dir or settings.new_run_dir("alfred_demo")
        self.logger = EventLogger(self.run_dir, self.request_id)
        self.catalog = SkillCatalog(settings.configs_dir / "skills.yaml")
        self.prompts = PromptRegistry(settings.configs_dir / "prompts.yaml")
        self.hardware = hardware_context or HardwareContext.from_settings(settings)
        self._owns_hardware = hardware_context is None
        if connect_hardware:
            self.hardware.connect()
        self.router = self._build_router()
        self.executor = SkillExecutor(
            router=self.router,
            safety_gate=SafetyGate(settings, self.catalog),
            logger=self.logger,
        )
        self.orchestrator = AlfredOrchestrator(
            skill_catalog=self.catalog,
            logger=self.logger,
            camera_id=settings.camera_id,
            skill_planner=CatalogSkillPlanner(self.settings, self.catalog, self.prompts),
        )
        self.session_memory_json_path = self.run_dir / "session_memory.json"
        self.session_memory_jsonl_path = self.run_dir / "session_memory.jsonl"

    def close(self) -> None:
        if self._owns_hardware:
            self.hardware.close()

    def __enter__(self) -> "AlfredRuntime":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def handle_text(
        self,
        text: str,
        input_type: str = "text",
        speak: bool = False,
        input_processing_results: Optional[List[SkillResult]] = None,
    ) -> PipelineResult:
        request = UserRequest(
            request_id=self.request_id,
            input_type=input_type,  # type: ignore[arg-type]
            raw_text=text.strip(),
        )
        self.logger.write_text("transcript.txt", request.raw_text)
        self.logger.log(
            stage="user_request",
            status="success",
            output_data=request.model_dump(mode="json"),
        )

        intent_result = self.orchestrator.classify_intent(request.raw_text)
        plan = self.orchestrator.create_plan(request, intent_result)
        communication_results: List[SkillResult] = []
        before_skill = (
            self._before_skill_speaker(request.raw_text, communication_results)
            if speak
            else None
        )
        skill_results = self.executor.execute_plan(plan, before_skill=before_skill)
        completion = self.orchestrator.evaluate_completion(request, skill_results)
        response = self.orchestrator.generate_final_answer(request, skill_results)

        if speak:
            spoken_completion = completion_announcement(
                request.raw_text,
                [call.skill_name for call in plan.skill_calls],
                response.answer_text,
                any(result.status == "error" for result in skill_results),
            )
            if spoken_completion:
                speak_result = self._speak_text(spoken_completion)
                communication_results.append(speak_result)

        self.record_task_history(
            request,
            plan,
            skill_results,
            communication_results,
            input_processing_results or [],
            completion,
            response,
        )

        return PipelineResult(
            request=request,
            response=response,
            completion=completion,
            skill_results=[
                *(input_processing_results or []),
                *communication_results,
                *skill_results,
            ],
            run_dir=self.run_dir,
        )

    def _before_skill_speaker(
        self,
        user_text: str,
        communication_results: List[SkillResult],
    ) -> Callable[[SkillCall], None]:
        def before_skill(call: SkillCall) -> None:
            announcement = start_announcement(user_text, call.skill_name)
            if announcement:
                communication_results.append(self._speak_text(announcement))

        return before_skill

    def _speak_text(self, text: str) -> SkillResult:
        result = self.executor.execute(SkillCall(skill_name="speak", arguments={"text": text}))
        if result.status != "success":
            print(f"[pipeline] Speech announcement failed: {result.error}")
        return result

    def handle_voice(self, seconds: Optional[int] = None, speak: bool = True) -> PipelineResult:
        listen_args = {}
        if seconds is not None:
            listen_args["seconds"] = seconds
        listen_result = self.executor.execute(SkillCall(skill_name="listen", arguments=listen_args))
        if listen_result.status != "success":
            raise RuntimeError(listen_result.error or "Listen failed")

        stt_result = self.executor.execute(
            SkillCall(
                skill_name="speech_to_text",
                arguments={"audio_file": listen_result.output["audio_file"]},
            )
        )
        if stt_result.status != "success":
            raise RuntimeError(stt_result.error or "Speech-to-text failed")

        user_text = str(stt_result.output["text"])
        return self.handle_text(
            text=user_text,
            input_type="voice",
            speak=speak,
            input_processing_results=[listen_result, stt_result],
        )

    def record_task_history(
        self,
        request: UserRequest,
        plan: OrchestratorPlan,
        skill_results: List[SkillResult],
        communication_results: List[SkillResult],
        input_processing_results: List[SkillResult],
        completion: CompletionResult,
        response: FinalResponse,
    ) -> None:
        input_processing_result_records = [
            result.model_dump(mode="json")
            for result in input_processing_results
        ]
        skill_result_records = [
            result.model_dump(mode="json")
            for result in skill_results
        ]
        communication_result_records = [
            result.model_dump(mode="json")
            for result in communication_results
        ]
        record = {
            "timestamp": utc_now_iso(),
            "request_id": request.request_id,
            "input_type": request.input_type,
            "question": request.raw_text,
            "user_text": request.raw_text,
            "intent": plan.intent.value,
            "goal": plan.goal,
            "skill_names": [call.skill_name for call in plan.skill_calls],
            "skill_calls": [
                call.model_dump(mode="json")
                for call in plan.skill_calls
            ],
            "skill_statuses": {
                result.skill_name: result.status
                for result in skill_results
            },
            "input_processing_results": input_processing_result_records,
            "skill_results": skill_result_records,
            "communication_results": communication_result_records,
            "all_results": [
                *input_processing_result_records,
                *communication_result_records,
                *skill_result_records,
            ],
            "task_complete": completion.task_complete,
            "completion_reason": completion.reason,
            "next_action": completion.next_action,
            "answer_text": response.answer_text,
            "run_dir": str(self.run_dir),
        }
        TASK_HISTORY.add(record)
        print("[memory] Recorded task history entry:")
        print(record)
        self.logger.write_json("task_history_record.json", record)
        self._persist_session_memory(record)

    def _persist_session_memory(self, record: dict) -> None:
        all_records = TASK_HISTORY.all()
        self.logger.write_json("session_memory.json", all_records)
        with self.session_memory_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[memory] Session memory written: {self.session_memory_json_path}")
        print(f"[memory] Session memory appended: {self.session_memory_jsonl_path}")

    def _build_router(self) -> SkillRouter:
        router = SkillRouter()
        router.register(ListenSkill(self.settings, self.run_dir))
        router.register(SpeechToTextSkill(self.settings))
        router.register(
            CaptureWristCameraImageSkill(
                run_dir=self.run_dir,
                default_camera_id=self.settings.camera_id,
                hardware_context=self.hardware,
            )
        )
        router.register(DescribeImageWithVLMSkill(self.settings, self.prompts))
        router.register(AnswerTaskHistorySkill(self.settings, self.prompts))
        router.register(GeneralConversationSkill(self.settings, self.prompts, self.catalog))
        router.register(TextToSpeechSkill(self.settings, self.run_dir))
        router.register(MoveArmNoopSkill())
        router.register(GoHomeSkill(self.hardware))
        router.register(GoOverlookSkill(self.hardware))
        router.register(PickBlueMarkerSkill(self.hardware))
        router.register(PickPlaceBlueMarkerSkill(self.hardware))
        router.register(EnzymeExperimentsSkill(self.hardware))
        router.register(OrderingSkill())
        router.register(
            ReadNotesSkill(
                self.settings,
                self.prompts,
                self.run_dir,
                self.hardware,
            )
        )
        router.register(
            AnswerSceneQuestionSkill(
                self.settings,
                self.prompts,
                self.run_dir,
                self.hardware,
            )
        )
        router.register(AmazonSearchSkill(self.settings))
        router.register(AmazonAddToCartSkill(self.settings))
        return router
