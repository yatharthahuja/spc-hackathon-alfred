from __future__ import annotations

from app.config import Settings
from app.memory.session_memory import SessionMemory
from app.orchestrator.task_registry import SkillCatalog
from app.skills.save_notes import SaveNotesSkill


def test_save_notes_skill_writes_previous_task_outcomes(tmp_path):
    memory = SessionMemory()
    memory.add(
        {
            "timestamp": "2026-05-17T20:00:00+00:00",
            "user_text": "read my notes",
            "task_complete": True,
            "completion_reason": "The planned skill completed successfully.",
            "answer_text": "Your notes say apples and oranges.",
            "skill_names": ["read_notes"],
            "skill_results": [
                {
                    "skill_name": "read_notes",
                    "status": "success",
                    "output": {
                        "extracted_text": "APPLES\nORANGES",
                        "answer_text": "Your notes say apples and oranges.",
                    },
                    "error": None,
                }
            ],
        }
    )
    skill = SaveNotesSkill(tmp_path, task_history=memory)

    result = skill.run(user_text="save my notes")

    assert result.status == "success"
    notes_file = tmp_path / "saved_notes.txt"
    assert notes_file.exists()
    notes_text = notes_file.read_text(encoding="utf-8")
    assert "read my notes" in notes_text
    assert "APPLES" in notes_text
    assert "Your notes say apples and oranges." in notes_text
    assert result.output["notes_file"] == str(notes_file)
    assert result.output["saved_record_count"] == 1


def test_save_notes_skill_can_save_last_task_only(tmp_path):
    memory = SessionMemory()
    memory.add(
        {
            "user_text": "go home",
            "task_complete": True,
            "answer_text": "Moved home.",
            "skill_names": ["go_home"],
            "skill_results": [],
        }
    )
    memory.add(
        {
            "user_text": "do enzyme experiments",
            "task_complete": True,
            "answer_text": "Experiment completed.",
            "skill_names": ["enzyme_experiments"],
            "skill_results": [],
        }
    )
    skill = SaveNotesSkill(tmp_path, task_history=memory)

    result = skill.run(user_text="save the result from the previous task")

    assert result.status == "success"
    assert result.output["saved_record_count"] == 1
    assert "do enzyme experiments" in result.output["notes_text"]
    assert "go home" not in result.output["notes_text"]


def test_save_notes_skill_can_use_requested_filename(tmp_path):
    memory = SessionMemory()
    memory.add(
        {
            "user_text": "read my notes",
            "task_complete": True,
            "answer_text": "Notes read.",
            "skill_names": ["read_notes"],
            "skill_results": [],
        }
    )
    skill = SaveNotesSkill(tmp_path, task_history=memory)

    result = skill.run(user_text="save my notes to experiment_notes.txt")

    assert result.status == "success"
    assert (tmp_path / "experiment_notes.txt").exists()
    assert result.output["notes_file"] == str(tmp_path / "experiment_notes.txt")


def test_save_notes_skill_is_declared_in_catalog():
    settings = Settings.load()
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")

    assert catalog.get("save_notes")["inputs"] == ["user_text", "filename"]
    assert catalog.get("save_notes")["outputs"] == [
        "notes_file",
        "saved_record_count",
        "notes_text",
        "message",
    ]
