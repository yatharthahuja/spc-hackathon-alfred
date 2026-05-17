from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Intent(str, Enum):
    DESCRIBE_DESK = "DESCRIBE_DESK"
    RUN_SKILL = "RUN_SKILL"
    FIND_OBJECT = "FIND_OBJECT"
    MONITOR_SCENE = "MONITOR_SCENE"
    OCR_READ_TEXT = "OCR_READ_TEXT"
    ORDER_MISSING_ITEMS = "ORDER_MISSING_ITEMS"
    UNKNOWN = "UNKNOWN"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    input_type: Literal["voice", "text"] = "text"
    raw_text: str
    timestamp: str = Field(default_factory=utc_now_iso)


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SkillCall(BaseModel):
    skill_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class TaskStep(BaseModel):
    task_id: str
    agent: str
    required_skills: List[str]


class OrchestratorPlan(BaseModel):
    goal: str
    intent: Intent
    tasks: List[TaskStep]
    skill_calls: List[SkillCall]
    completion_criteria: List[str]


class SkillResult(BaseModel):
    skill_name: str
    status: Literal["success", "error", "skipped"]
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class VLMResult(BaseModel):
    objects: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    spoken_summary: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class SceneQAResult(BaseModel):
    answer_text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class ReadNotesResult(BaseModel):
    extracted_text: str
    answer_text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)


class CompletionResult(BaseModel):
    task_complete: bool
    reason: str
    next_action: Literal["none", "retake_image", "move_camera", "ask_user"]


class FinalResponse(BaseModel):
    task_complete: bool
    answer_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class EventLogEntry(BaseModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    request_id: str
    stage: str
    status: Literal["success", "error", "skipped", "info"] = "info"
    latency_ms: Optional[int] = None
    skill: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
