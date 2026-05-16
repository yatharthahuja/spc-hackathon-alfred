import asyncio

from interfaces import BaseDriver


class MockArmDriver(BaseDriver):
    def __init__(self) -> None:
        self._location = "home"

    async def initialize(self) -> None:
        await asyncio.sleep(0.1)
        print("[MockArmDriver] Arm ready.")

    async def health_check(self) -> bool:
        return True

    async def move_to(self, location: str) -> str:
        await asyncio.sleep(0.3)
        self._location = location
        return f"Arm moved to {location}"
