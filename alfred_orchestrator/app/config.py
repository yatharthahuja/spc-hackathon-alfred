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
    amazon_api: str
    amazon_use_mock: bool
    amazon_access_key: str
    amazon_secret_key: str
    amazon_credential_id: str
    amazon_credential_secret: str
    amazon_credential_version: str
    amazon_auth_endpoint: str
    amazon_partner_tag: str
    amazon_host: str
    amazon_region: str
    amazon_marketplace: str
    amazon_search_index: str

    @property
    def amazon_paapi_ready(self) -> bool:
        return bool(self.amazon_access_key and self.amazon_secret_key and self.amazon_partner_tag)

    @property
    def amazon_creators_ready(self) -> bool:
        return bool(
            self.amazon_credential_id
            and self.amazon_credential_secret
            and self.amazon_partner_tag
            and self.amazon_credential_version
        )

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
            openai_reasoning_model=os.getenv("OPENAI_REASONING_MODEL", "gpt-5.5"),
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.5"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            elevenlabs_tts_model=os.getenv("ELEVENLABS_TTS_MODEL", "eleven_v3"),
            elevenlabs_output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
            elevenlabs_stt_model=os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2"),
            camera_id=int(os.getenv("ALFRED_CAMERA_ID", "0")),
            record_seconds=int(os.getenv("ALFRED_RECORD_SECONDS", "5")),
            enable_physical_skills=_bool_env("ALFRED_ENABLE_PHYSICAL_SKILLS", False),
            amazon_api=os.getenv("AMAZON_API", "creators").strip().lower(),
            amazon_use_mock=_bool_env("AMAZON_USE_MOCK", True),
            amazon_access_key=os.getenv("AMAZON_ACCESS_KEY", ""),
            amazon_secret_key=os.getenv("AMAZON_SECRET_KEY", ""),
            amazon_credential_id=os.getenv("AMAZON_CREDENTIAL_ID")
            or os.getenv("AMAZON_ACCESS_KEY", ""),
            amazon_credential_secret=os.getenv("AMAZON_CREDENTIAL_SECRET")
            or os.getenv("AMAZON_SECRET_KEY", ""),
            amazon_credential_version=os.getenv("AMAZON_CREDENTIAL_VERSION", "2.1"),
            amazon_auth_endpoint=os.getenv("AMAZON_AUTH_ENDPOINT", ""),
            amazon_partner_tag=os.getenv("AMAZON_PARTNER_TAG", ""),
            amazon_host=os.getenv("AMAZON_PAAPI_HOST")
            or os.getenv("AMAZON_HOST", "webservices.amazon.com"),
            amazon_region=os.getenv("AMAZON_PAAPI_REGION")
            or os.getenv("AMAZON_REGION", "us-east-1"),
            amazon_marketplace=os.getenv("AMAZON_MARKETPLACE", "www.amazon.com"),
            amazon_search_index=os.getenv("AMAZON_SEARCH_INDEX", "All"),
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
