from drivers.note_driver import NoteDriver
from interfaces import BaseSkill
from models import AgentAction


class NoteSkill(BaseSkill):
    def __init__(self, driver: NoteDriver) -> None:
        super().__init__(driver)
        self._notes: NoteDriver = driver

    async def execute(self, params: dict) -> AgentAction:
        text = params.get("text", "").strip()
        if not text:
            return AgentAction(
                skill="note",
                success=False,
                message="No note text provided.",
            )

        await self._notes.append_note(text)
        return AgentAction(
            skill="note",
            success=True,
            message=f"Noted, sir: {text}",
            details={"text": text},
        )
