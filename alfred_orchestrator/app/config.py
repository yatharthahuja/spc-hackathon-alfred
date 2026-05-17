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
    serpapi_key: str
    amazon_associates_tag: str
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
            serpapi_key=os.getenv("SERPAPI_KEY", ""),
            amazon_associates_tag=os.getenv("AMAZON_ASSOCIATES_TAG", "sebilee2026-20"),
            openai_reasoning_model=os.getenv("OPENAI_REASONING_MODEL", "gpt-5.5"),
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.5"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            elevenlabs_tts_model=os.getenv("ELEVENLABS_TTS_MODEL", "eleven_v3"),
            elevenlabs_output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
            elevenlabs_stt_model=os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2"),
            camera_id=int(os.getenv("ALFRED_CAMERA_ID", "0")),
            record_seconds=int(os.getenv("ALFRED_RECORD_SECONDS", "5")),
            enable_physical_skills=_bool_env("ALFRED_ENABLE_PHYSICAL_SKILLS", False),
        )

    def print_model_configuration(self) -> None:
        print("[settings] Model configuration")
        print(f"[settings] OpenAI reasoning model: {self.openai_reasoning_model}")
        print(f"[settings] OpenAI vision model: {self.openai_vision_model}")
        print(f"[settings] ElevenLabs voice id: {self.elevenlabs_voice_id}")
        print(f"[settings] ElevenLabs TTS model: {self.elevenlabs_tts_model}")
        print(f"[settings] ElevenLabs output format: {self.elevenlabs_output_format}")
        print(f"[settings] ElevenLabs STT model: {self.elevenlabs_stt_model}")

    def new_run_dir(self, prefix: str = "alfred_demo") -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.runs_dir / f"{stamp}_{prefix}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data
