from __future__ import annotations

from typing import Any

from app.hardware.resources import HardwareContext
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, success


PICK_BLUE_MARKER_SEQUENCE = [
    "pre_pick_marker",
    "pick_marker",
    "post_pick_marker",
]
PICK_PLACE_BLUE_MARKER_SEQUENCE = [
    "pre_pick_marker",
    "pick_marker",
    "post_pick_marker",
    "pre_place",
    "drop_marker",
]
GO_HOME_SEQUENCE = ["go_home"]
GO_OVERLOOK_SEQUENCE = ["overlook"]


class RobotTrajectorySkill(Skill):
    trajectory_names: list[str]
    success_message: str

    def __init__(self, hardware_context: HardwareContext):
        self.hardware = hardware_context

    def run(self, **_kwargs: Any) -> SkillResult:
        print(f"[{self.name}] Running robot trajectory sequence:")
        print(f"[{self.name}] {self.trajectory_names}")
        result = self.hardware.robot.move_through_poses(self.trajectory_names)
        print(f"[{self.name}] Robot sequence output: {result.output()}")
        output = {
            **result.output(),
            "message": self.success_message if result.moved else result.error,
        }
        if result.moved:
            return success(self.name, output)
        return SkillResult(
            skill_name=self.name,
            status="error",
            output=output,
            error=result.error or "Robot trajectory did not complete.",
        )


class PickBlueMarkerSkill(RobotTrajectorySkill):
    name = "pick_blue_marker"
    trajectory_names = PICK_BLUE_MARKER_SEQUENCE
    success_message = "Picked up the blue marker."


class PickPlaceBlueMarkerSkill(RobotTrajectorySkill):
    name = "pick_place_blue_marker"
    trajectory_names = PICK_PLACE_BLUE_MARKER_SEQUENCE
    success_message = "Picked up and placed the blue marker."


class GoHomeSkill(RobotTrajectorySkill):
    name = "go_home"
    trajectory_names = GO_HOME_SEQUENCE
    success_message = "Moved to the home pose."


class GoOverlookSkill(RobotTrajectorySkill):
    name = "go_overlook"
    trajectory_names = GO_OVERLOOK_SEQUENCE
    success_message = "Moved to the overlook pose."
