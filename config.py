import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().strip('"').strip("'").lower() in ("true", "1", "yes")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw.strip().strip('"').strip("'"))


OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_VLM_MODEL: str = os.getenv("OPENAI_VLM_MODEL", "gpt-4o")
CAMERA_INDEX: int = _env_int("CAMERA_INDEX", 0)
VISION_ENABLED: bool = _env_bool("VISION_ENABLED", True)
MAX_PLAN_STEPS: int = _env_int("MAX_PLAN_STEPS", 5)
DEBUG_SAVE_FRAME: bool = _env_bool("DEBUG_SAVE_FRAME", False)
