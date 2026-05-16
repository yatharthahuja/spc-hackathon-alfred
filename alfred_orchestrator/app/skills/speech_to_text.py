from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from app.config import Settings
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class SpeechToTextSkill(Skill):
    name = "speech_to_text"

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            audio_path = Path(str(kwargs["audio_file"]))
            text = self.transcribe(audio_path)
            return success(self.name, {"text": text, "audio_file": str(audio_path)})
        except Exception as exc:
            return failure(self.name, exc)

    def transcribe(self, audio_path: Path) -> str:
        if not self.settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for speech-to-text")

        with audio_path.open("rb") as audio_file:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.settings.elevenlabs_api_key},
                data={"model_id": self.settings.elevenlabs_stt_model},
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            raise RuntimeError(f"ElevenLabs STT returned no text: {payload}")
        return text
