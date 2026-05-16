import asyncio
import base64
import json
import time
from collections.abc import Callable
from pathlib import Path

from config import DEBUG_SAVE_FRAME, VISION_ENABLED
from drivers.mock_arm import MockArmDriver
from drivers.note_driver import NoteDriver
from drivers.openai_vlm_driver import NORMALCY_USER_MESSAGE, OpenAIVLMDriver
from drivers.webcam_driver import CameraUnavailableError, WebcamDriver
from interfaces import BaseSkill
from models import AgentAction, OrchestratorDecision, PlanStep, WorldState
from skill_catalog import build_catalogue
from skills.manipulate import ManipulateSkill
from skills.note import NoteSkill

STATE_PATH = Path(__file__).resolve().parent / "state.json"
INACTIVITY_SECONDS = 60


class AlfredOrchestrator:
    def __init__(self) -> None:
        self.state = self._load_state()
        self._state_path = STATE_PATH
        self._last_input = time.monotonic()
        self._normalcy_ran_for_idle = False
        self._running = False
        self._history: list[dict[str, str]] = []

        self._arm_driver = MockArmDriver()
        self._note_driver = NoteDriver()
        self._webcam = WebcamDriver()
        self._vlm = OpenAIVLMDriver()

        self.catalogue = build_catalogue()
        self.skill_registry: dict[str, BaseSkill] = {}
        self._webcam_ready = False

    def _load_state(self) -> WorldState:
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return WorldState.model_validate(data)
        return WorldState()

    async def _persist_state(self) -> None:
        payload = self.state.model_dump_json(indent=2)

        def _write() -> None:
            self._state_path.write_text(payload, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def _update_state(self, mutator: Callable[[WorldState], None]) -> None:
        mutator(self.state)
        await self._persist_state()

    async def initialize(self) -> None:
        await self._arm_driver.initialize()
        await self._note_driver.initialize()
        await self._vlm.initialize()

        try:
            await self._webcam.initialize()
            self._webcam_ready = True
        except CameraUnavailableError as exc:
            self._webcam_ready = False
            print(f"[Alfred] Webcam unavailable: {exc}")

        vision_status = "on" if VISION_ENABLED else "off"
        cam_status = "ready" if self._webcam_ready else "unavailable"
        print(f"[Alfred] Webcam: {cam_status} | Vision API: {vision_status}")

        await self._persist_state()

        self.skill_registry = {
            "manipulate": ManipulateSkill(
                self._arm_driver,
                on_state_change=self._update_state,
            ),
            "note": NoteSkill(self._note_driver),
        }

        self._running = True
        print("Alfred at your service, sir. Ask me anything — or type 'help'.")

    def _touch_input(self) -> None:
        self._last_input = time.monotonic()
        self._normalcy_ran_for_idle = False

    async def _capture_frame_or_none(self):
        if not self._webcam_ready:
            print("[Alfred] Skipping capture — webcam not ready.")
            return None
        try:
            frame = await self._webcam.capture_frame()
            if DEBUG_SAVE_FRAME:
                debug_path = Path(__file__).resolve().parent / "debug_frame.jpg"
                debug_path.write_bytes(base64.b64decode(frame.jpeg_b64))
                print(f"[Alfred] Saved debug frame to {debug_path}")
            return frame
        except CameraUnavailableError as exc:
            print(f"[Alfred] Capture failed: {exc}")
            self._webcam_ready = False
            return None

    async def _execute_steps(self, steps: list[PlanStep]) -> list[AgentAction]:
        results: list[AgentAction] = []
        for step in steps:
            if step.skill not in self.skill_registry:
                print(f"Unknown skill '{step.skill}', sir. Skipping.")
                continue
            skill = self.skill_registry[step.skill]
            action = await skill.execute(step.params)
            results.append(action)
            print(action.message)
            if not action.success:
                break
        return results

    def _apply_decision_state(self, decision: OrchestratorDecision) -> None:
        if decision.update_messy is not None:
            self.state.is_messy = decision.update_messy

    async def _handle_user_turn(self, command: str) -> None:
        frame = await self._capture_frame_or_none()
        decision = await self._vlm.orchestrate(
            user_message=command,
            catalogue=self.catalogue,
            state=self.state,
            frame=frame,
            history=self._history,
        )
        print(decision.reply)
        if decision.steps:
            await self._execute_steps(decision.steps)
        self._apply_decision_state(decision)
        self.state.last_observation = decision.reply[:500]
        await self._persist_state()
        self._history.append({"user": command, "reply": decision.reply})
        if len(self._history) > 10:
            self._history = self._history[-10:]

    async def perform_normalcy_check(self) -> None:
        frame = await self._capture_frame_or_none()
        decision = await self._vlm.orchestrate(
            user_message=NORMALCY_USER_MESSAGE,
            catalogue=self.catalogue,
            state=self.state,
            frame=frame,
            history=None,
        )
        print(decision.reply)
        if decision.steps:
            await self._execute_steps(decision.steps)
        elif (
            not decision.steps
            and (self.state.is_messy or self.state.is_out_of_place())
            and "manipulate" in self.skill_registry
        ):
            result = await self.skill_registry["manipulate"].execute({"action": "reset"})
            print(result.message)

        self._apply_decision_state(decision)
        self.state.last_observation = decision.reply[:500]
        await self._persist_state()

    async def _inactivity_monitor(self) -> None:
        while self._running:
            await asyncio.sleep(1)
            idle = time.monotonic() - self._last_input
            if idle > INACTIVITY_SECONDS and not self._normalcy_ran_for_idle:
                await self.perform_normalcy_check()
                self._normalcy_ran_for_idle = True

    async def _read_command(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input("alfred> "))

    async def _handle_command(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return True

        lowered = stripped.lower()

        if lowered in {"quit", "exit"}:
            print("Very good, sir. Alfred signing off.")
            return False

        if lowered == "help":
            print(
                "Speak naturally, sir. Examples:\n"
                "  What do you see on the counter?\n"
                "  Take a note about the soup recipe\n"
                "  Please tidy the kitchen\n"
                "  status — show kitchen state\n"
                "  mess / tidy — testing flags\n"
                "  quit — exit\n"
                "\nAlfred uses the webcam and VLM to see, reply, and run skills."
            )
            return True

        if lowered == "status":
            print(self.state.model_dump_json(indent=2))
            return True

        if lowered == "mess":
            await self._update_state(lambda s: setattr(s, "is_messy", True))
            print("Counter marked messy, sir.")
            return True

        if lowered == "tidy":
            await self._update_state(lambda s: setattr(s, "is_messy", False))
            print("Counter marked tidy, sir.")
            return True

        await self._handle_user_turn(stripped)
        return True

    async def run(self) -> None:
        await self.initialize()
        monitor_task = asyncio.create_task(self._inactivity_monitor())

        try:
            while self._running:
                command = await self._read_command()
                self._touch_input()
                should_continue = await self._handle_command(command)
                if not should_continue:
                    self._running = False
        finally:
            self._running = False
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            await self._webcam.shutdown()


async def main() -> None:
    orchestrator = AlfredOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
