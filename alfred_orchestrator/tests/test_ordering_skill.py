from __future__ import annotations

from app.config import Settings
from app.orchestrator.task_registry import SkillCatalog
from app.skills.ordering import OrderingSkill


def test_ordering_skill_writes_signal_file(tmp_path):
    signal_file = tmp_path / "drone" / "run"
    signal_file.parent.mkdir()
    signal_file.write_text("0", encoding="utf-8")
    skill = OrderingSkill(signal_file)

    result = skill.run(user_text="can you please get apple and oranges for me")

    assert result.status == "success"
    assert signal_file.read_text(encoding="utf-8") == "1"
    assert result.output["signal_file"] == str(signal_file)
    assert result.output["signal_value"] == "1"
    assert result.output["previous_value"] == "0"
    assert result.output["user_text"] == "can you please get apple and oranges for me"


def test_ordering_skill_is_declared_in_catalog():
    settings = Settings.load()
    catalog = SkillCatalog(settings.configs_dir / "skills.yaml")

    assert catalog.get("ordering")["inputs"] == ["user_text"]
    assert catalog.get("ordering")["outputs"] == [
        "signal_file",
        "signal_value",
        "previous_value",
        "user_text",
        "message",
    ]
