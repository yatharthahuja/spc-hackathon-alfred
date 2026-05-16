from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from app.config import Settings
from app.interfaces.audio import play_audio
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class TextToSpeechSkill(Skill):
    name = "speak"

    def __init__(self, settings: Settings, run_dir: Path):
        self.settings = settings
        self.run_dir = run_dir

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            text = str(kwargs["text"]).strip()
            if not text:
                raise ValueError("Text is required for TTS")
            if not self.settings.elevenlabs_api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is required for TTS")

            audio_path = self.run_dir / "alfred_response.mp3"
            self._synthesize(text, audio_path)
            audio_played = play_audio(audio_path)
            return success(
                self.name,
                {
                    "text": text,
                    "audio_path": str(audio_path),
                    "audio_played": audio_played,
                },
            )
        except Exception as exc:
            return failure(self.name, exc)

    def _synthesize(self, text: str, audio_path: Path) -> None:
        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.settings.elevenlabs_voice_id}?"
            f"{urlencode({'output_format': self.settings.elevenlabs_output_format})}"
        )
        response = requests.post(
            url,
            headers={
                "xi-api-key": self.settings.elevenlabs_api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            },
            json={
                "text": text,
                "model_id": self.settings.elevenlabs_tts_model,
                "voice_settings": {
                    "stability": 0.45,
                    "similarity_boost": 0.8,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        audio_path.write_bytes(response.content)
