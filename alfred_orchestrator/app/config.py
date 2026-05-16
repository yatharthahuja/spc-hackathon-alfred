from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    runs_dir: Path
    configs_dir: Path
    openai_api_key: str
    elevenlabs_api_key: str
    openai_reasoning_model: str
    openai_vision_model: str
    elevenlabs_voice_id: str
    elevenlabs_tts_model: str
    elevenlabs_output_format: str
    elevenlabs_stt_model: str
    camera_id: int
    record_seconds: int
    enable_physical_skills: bool

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        runs_dir = Path(os.getenv("ALFRED_RUNS_DIR", "runs"))
        if not runs_dir.is_absolute():
            runs_dir = PROJECT_ROOT / runs_dir
        return cls(
            project_root=PROJECT_ROOT,
            runs_dir=runs_dir,
            configs_dir=PROJECT_ROOT / "configs",
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            openai_reasoning_model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1-mini"),
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            elevenlabs_tts_model=os.getenv("ELEVENLABS_TTS_MODEL", "eleven_v3"),
            elevenlabs_output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
            elevenlabs_stt_model=os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1"),
            camera_id=int(os.getenv("ALFRED_CAMERA_ID", "0")),
            record_seconds=int(os.getenv("ALFRED_RECORD_SECONDS", "5")),
            enable_physical_skills=_bool_env("ALFRED_ENABLE_PHYSICAL_SKILLS", False),
        )

    def new_run_dir(self, prefix: str = "alfred_demo") -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.runs_dir / f"{stamp}_{prefix}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data
