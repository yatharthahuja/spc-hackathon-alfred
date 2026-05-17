from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.memory.session_memory import SessionMemory, TASK_HISTORY
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class SaveNotesSkill(Skill):
    name = "save_notes"

    def __init__(
        self,
        run_dir: Path,
        task_history: SessionMemory = TASK_HISTORY,
    ):
        self.run_dir = run_dir
        self.task_history = task_history

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            user_text = str(kwargs.get("user_text") or "save my notes").strip()
            requested_filename = str(kwargs.get("filename") or "").strip()
            filename = self._safe_filename(
                requested_filename
                or self._filename_from_request(user_text)
                or "saved_notes.txt"
            )
            notes_file = self.run_dir / filename
            records = self._select_records(user_text)
            notes_text = self._format_notes(user_text, records)

            notes_file.parent.mkdir(parents=True, exist_ok=True)
            notes_file.write_text(notes_text, encoding="utf-8")

            return success(
                self.name,
                {
                    "notes_file": str(notes_file),
                    "saved_record_count": len(records),
                    "notes_text": notes_text,
                    "message": f"Saved notes from {len(records)} previous task record(s).",
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    def _select_records(self, user_text: str) -> list[dict[str, Any]]:
        records = [
            record
            for record in self.task_history.all()
            if "save_notes" not in record.get("skill_names", [])
        ]
        normalized = " ".join(user_text.lower().split())

        limit = self._requested_limit(normalized)
        if limit is not None:
            return records[-limit:]
        if any(word in normalized for word in ("last", "previous", "recent")):
            return records[-1:]
        return records

    @staticmethod
    def _requested_limit(normalized: str) -> int | None:
        match = re.search(r"\b(?:last|previous|recent)\s+(\d+)\b", normalized)
        if not match:
            return None
        limit = int(match.group(1))
        return limit if limit > 0 else None

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name.strip() or "saved_notes.txt"
        if not name.endswith(".txt"):
            name = f"{name}.txt"
        return name

    @staticmethod
    def _filename_from_request(user_text: str) -> str | None:
        match = re.search(
            r"\b(?:to|as|in)\s+([A-Za-z0-9_-]+\.txt)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _format_notes(
        self,
        user_text: str,
        records: list[dict[str, Any]],
    ) -> str:
        lines = [
            "Alfred Saved Notes",
            f"Saved at: {datetime.now().isoformat(timespec='seconds')}",
            f"User request: {user_text}",
            "",
        ]
        if not records:
            lines.append("No previous task outcomes were recorded yet.")
            lines.append("")
            return "\n".join(lines)

        for index, record in enumerate(records, start=1):
            lines.extend(
                [
                    f"## Task {index}",
                    f"Time: {record.get('timestamp', 'unknown')}",
                    f"Request: {record.get('user_text') or record.get('question') or 'unknown'}",
                    f"Task complete: {record.get('task_complete')}",
                    f"Completion reason: {record.get('completion_reason', '')}",
                    f"Final answer: {record.get('answer_text', '')}",
                    "Skill results:",
                ]
            )
            for result in record.get("skill_results", []):
                lines.extend(self._format_skill_result(result))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_skill_result(result: dict[str, Any]) -> list[str]:
        output = result.get("output") or {}
        lines = [
            f"- {result.get('skill_name', 'unknown')}: {result.get('status', 'unknown')}",
        ]
        for key in ("answer_text", "message", "spoken_summary", "extracted_text", "notes_text"):
            value = output.get(key)
            if value:
                lines.append(f"  {key}: {value}")
        if output:
            lines.append("  output_json:")
            lines.append(json.dumps(output, indent=2, ensure_ascii=False))
        if result.get("error"):
            lines.append(f"  error: {result['error']}")
        return lines
