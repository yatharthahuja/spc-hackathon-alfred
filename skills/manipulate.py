from collections.abc import Awaitable, Callable

from drivers.mock_arm import MockArmDriver
from interfaces import BaseSkill
from models import AgentAction, WorldState

StateMutator = Callable[[WorldState], None]
OnStateChange = Callable[[StateMutator], Awaitable[None]]


class ManipulateSkill(BaseSkill):
    def __init__(
        self,
        driver: MockArmDriver,
        on_state_change: OnStateChange,
    ) -> None:
        super().__init__(driver)
        self._arm: MockArmDriver = driver
        self._on_state_change = on_state_change

    async def execute(self, params: dict) -> AgentAction:
        action = params.get("action", "move")

        if action == "reset":
            move_msg = await self._arm.move_to("home")

            def apply_reset(state: WorldState) -> None:
                state.apply_normalcy()

            await self._on_state_change(apply_reset)
            return AgentAction(
                skill="manipulate",
                success=True,
                message=f"Kitchen restored to normalcy. {move_msg}",
                details={"action": "reset"},
            )

        if action == "move":
            target = params.get("target", "home")
            move_msg = await self._arm.move_to(target)

            def set_location(state: WorldState) -> None:
                state.arm_location = target

            await self._on_state_change(set_location)
            return AgentAction(
                skill="manipulate",
                success=True,
                message=move_msg,
                details={"action": "move", "target": target},
            )

        return AgentAction(
            skill="manipulate",
            success=False,
            message=f"Unknown action: {action}",
        )
