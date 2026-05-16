from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app.config import load_yaml


class PromptRegistry:
    def __init__(self, prompts_path: Path):
        self.prompts = load_yaml(prompts_path)

    def render(self, name: str, context: Dict[str, Any]) -> str:
        if name not in self.prompts:
            raise KeyError(f"Unknown prompt: {name}")
        template = self.prompts[name]["system_prompt"]
        normalized = {
            key: self._format_value(value)
            for key, value in context.items()
        }
        return template.format(**normalized)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False)
