from abc import ABC, abstractmethod

from models import AgentAction


class BaseDriver(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class BaseSkill(ABC):
    def __init__(self, driver: BaseDriver) -> None:
        self.driver = driver

    @abstractmethod
    async def execute(self, params: dict) -> AgentAction:
        pass
