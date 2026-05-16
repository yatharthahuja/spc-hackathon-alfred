from __future__ import annotations

from typing import Any, Dict, List


class SessionMemory:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(item)

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.items[-limit:]
