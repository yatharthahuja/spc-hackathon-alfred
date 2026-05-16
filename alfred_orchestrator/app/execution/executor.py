from __future__ import annotations

import time
from typing import Any, Dict, List

from app.execution.safety import SafetyGate
from app.execution.skill_router import SkillRouter
from app.logs.event_logger import EventLogger
from app.orchestrator.schemas import OrchestratorPlan, SkillCall, SkillResult


class SkillExecutor:
    def __init__(self, router: SkillRouter, safety_gate: SafetyGate, logger: EventLogger):
        self.router = router
        self.safety_gate = safety_gate
        self.logger = logger

    def execute_plan(self, plan: OrchestratorPlan) -> List[SkillResult]:
        results: List[SkillResult] = []
        outputs_by_skill: Dict[str, Dict[str, Any]] = {}
        for call in plan.skill_calls:
            resolved_call = SkillCall(
                skill_name=call.skill_name,
                arguments=self._resolve_arguments(call.arguments, outputs_by_skill),
            )
            result = self.execute(resolved_call)
            results.append(result)
            outputs_by_skill[result.skill_name] = result.output
            if result.status == "error":
                break
        return results

    def execute(self, call: SkillCall) -> SkillResult:
        started = time.perf_counter()
        try:
            self.safety_gate.validate_skill(call.skill_name)
            skill = self.router.get(call.skill_name)
            self.logger.log(
                stage="skill_execution",
                status="info",
                skill=call.skill_name,
                input_data={"arguments": call.arguments},
            )
            result = skill.run(**call.arguments)
        except Exception as exc:
            result = SkillResult(skill_name=call.skill_name, status="error", error=str(exc))

        latency_ms = int((time.perf_counter() - started) * 1000)
        self.logger.log(
            stage="skill_execution",
            status=result.status,
            latency_ms=latency_ms,
            skill=call.skill_name,
            input_data={"arguments": call.arguments},
            output_data=result.output,
            error=result.error,
        )
        return result

    def _resolve_arguments(
        self,
        arguments: Dict[str, Any],
        outputs_by_skill: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            key: self._resolve_value(value, outputs_by_skill)
            for key, value in arguments.items()
        }

    def _resolve_value(self, value: Any, outputs_by_skill: Dict[str, Dict[str, Any]]) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            parts = value[1:].split(".")
            if len(parts) != 2:
                raise ValueError(f"Unsupported reference: {value}")
            skill_name, output_name = parts
            return outputs_by_skill[skill_name][output_name]
        return value
