from __future__ import annotations

from typing import Iterable

from app.config import Settings
from app.hardware.resources import RobotSequenceResult
from app.orchestrator.task_registry import SkillCatalog
from app.skills.marker import (
    GO_HOME_SEQUENCE,
    GO_OVERLOOK_SEQUENCE,
    PICK_BLUE_MARKER_SEQUENCE,
    PICK_PLACE_BLUE_MARKER_SEQUENCE,
    GoHomeSkill,
    GoOverlookSkill,
    PickBlueMarkerSkill,
    PickPlaceBlueMarkerSkill,
)


class FakeRobotResource:
    def __init__(self):
        self.sequences: list[list[str]] = []

    def move_through_poses(self, pose_names: Iterable[str]) -> RobotSequenceResult:
        names = list(pose_names)
        self.sequences.append(names)
        return RobotSequenceResult(
            trajectory_names=names,
            attempted=True,
            moved=True,
        )


class FakeHardwareContext:
    def __init__(self):
        self.robot = FakeRobotResource()


def test_pick_blue_marker_runs_pick_sequence():
    hardware = FakeHardwareContext()
    skill = PickBlueMarkerSkill(hardware)

    result = skill.run()

    assert result.status == "success"
    assert hardware.robot.sequences == [PICK_BLUE_MARKER_SEQUENCE]
    assert result.output["trajectory_names"] == PICK_BLUE_MARKER_SEQUENCE
    assert result.output["robot_moved"] is True


def test_marker_skills_are_declared_in_catalog():
    settings = Settings.load()
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")

    assert catalog.get("general_conversation")["inputs"] == ["question"]
    assert catalog.get("go_home")["trajectory_names"] == GO_HOME_SEQUENCE
    assert catalog.get("go_overlook")["trajectory_names"] == GO_OVERLOOK_SEQUENCE
    assert catalog.get("pick_blue_marker")["trajectory_names"] == PICK_BLUE_MARKER_SEQUENCE
    assert catalog.get("pick_place_blue_marker")["trajectory_names"] == PICK_PLACE_BLUE_MARKER_SEQUENCE
    assert catalog.get("answer_scene_question")["inputs"] == ["question"]
    assert catalog.get("read_notes")["inputs"] == ["user_text"]


def test_pick_place_blue_marker_runs_pick_and_place_sequence():
    hardware = FakeHardwareContext()
    skill = PickPlaceBlueMarkerSkill(hardware)

    result = skill.run()

    assert result.status == "success"
    assert hardware.robot.sequences == [PICK_PLACE_BLUE_MARKER_SEQUENCE]
    assert result.output["trajectory_names"] == PICK_PLACE_BLUE_MARKER_SEQUENCE
    assert result.output["robot_moved"] is True


def test_go_home_runs_home_sequence():
    hardware = FakeHardwareContext()
    skill = GoHomeSkill(hardware)

    result = skill.run()

    assert result.status == "success"
    assert hardware.robot.sequences == [GO_HOME_SEQUENCE]
    assert result.output["trajectory_names"] == GO_HOME_SEQUENCE
    assert result.output["message"] == "Moved to the home pose."


def test_go_overlook_runs_overlook_sequence():
    hardware = FakeHardwareContext()
    skill = GoOverlookSkill(hardware)

    result = skill.run()

    assert result.status == "success"
    assert hardware.robot.sequences == [GO_OVERLOOK_SEQUENCE]
    assert result.output["trajectory_names"] == GO_OVERLOOK_SEQUENCE
    assert result.output["message"] == "Moved to the overlook pose."
