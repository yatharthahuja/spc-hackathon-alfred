from __future__ import annotations

import threading
from typing import Any, Dict, List


class SessionMemory:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self.items.append(item)

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.items[-limit:])

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.items)

    def clear(self) -> None:
        with self._lock:
            self.items.clear()

    def count(self) -> int:
        with self._lock:
            return len(self.items)


TASK_HISTORY = SessionMemory()
