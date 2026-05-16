from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.orchestrator.schemas import EventLogEntry


class EventLogger:
    def __init__(self, run_dir: Path, request_id: str):
        self.run_dir = run_dir
        self.request_id = request_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def log(
        self,
        stage: str,
        status: str = "info",
        latency_ms: Optional[int] = None,
        skill: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> EventLogEntry:
        entry = EventLogEntry(
            request_id=self.request_id,
            stage=stage,
            status=status,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            skill=skill,
            input=input_data or {},
            output=output_data or {},
            error=error,
        )
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(entry.model_dump_json() + "\n")
        return entry

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        path.write_text(text, encoding="utf-8")
        return path
