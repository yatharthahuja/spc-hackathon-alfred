from __future__ import annotations

from abc import ABC, abstractmethod

from app.integrations.amazon.models import AmazonProduct


class AmazonSearchClient(ABC):
    @abstractmethod
    def search_first(self, query: str) -> AmazonProduct:
        raise NotImplementedError
