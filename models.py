from datetime import datetime, timezone

from pydantic import BaseModel, Field

NORMALCY_INVENTORY: dict[str, str] = {
    "spatula": "drawer",
    "ladle": "drawer",
    "cutting_board": "cabinet",
    "mixing_bowl": "cabinet",
}

DEFAULT_MESSY_INVENTORY: dict[str, str] = {
    "spatula": "counter",
    "ladle": "drawer",
    "cutting_board": "counter",
    "mixing_bowl": "cabinet",
}


class WorldState(BaseModel):
    inventory: dict[str, str] = Field(default_factory=lambda: DEFAULT_MESSY_INVENTORY.copy())
    arm_location: str = "counter"
    is_messy: bool = True
    last_observation: str | None = None

    def is_out_of_place(self) -> bool:
        return self.inventory != NORMALCY_INVENTORY

    def apply_normalcy(self) -> None:
        self.inventory = NORMALCY_INVENTORY.copy()
        self.is_messy = False
        self.arm_location = "home"


class CameraFrame(BaseModel):
    jpeg_b64: str
    width: int
    height: int
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SkillSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, str] = Field(default_factory=dict)
    examples: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    skill: str
    params: dict = Field(default_factory=dict)


class OrchestratorDecision(BaseModel):
    reply: str
    steps: list[PlanStep] = Field(default_factory=list)
    update_messy: bool | None = None


class AgentAction(BaseModel):
    skill: str
    success: bool
    message: str
    details: dict | None = None
