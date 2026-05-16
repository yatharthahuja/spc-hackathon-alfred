import asyncio
from datetime import datetime, timezone
from pathlib import Path

from interfaces import BaseDriver

NOTES_PATH = Path(__file__).resolve().parent.parent / "notes.txt"


class NoteDriver(BaseDriver):
    def __init__(self, notes_path: Path | None = None) -> None:
        self._notes_path = notes_path or NOTES_PATH

    async def initialize(self) -> None:
        await asyncio.sleep(0.05)
        print(f"[NoteDriver] Ready. Notes file: {self._notes_path}")

    async def health_check(self) -> bool:
        return True

    async def append_note(self, text: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {text}\n"

        def _write() -> None:
            with self._notes_path.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_write)
