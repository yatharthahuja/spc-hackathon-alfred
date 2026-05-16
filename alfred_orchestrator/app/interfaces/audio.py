from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from scipy.io.wavfile import write as write_wav


def _load_sounddevice():
    # Deferred import: on Linux, `sounddevice` requires the PortAudio system
    # library (e.g. `libportaudio2`). Importing it at module load would break
    # unrelated paths (like audio playback via `aplay`) on machines without it.
    try:
        import sounddevice as sd  # noqa: WPS433 (intentional local import)
    except OSError as exc:
        hint = _portaudio_install_hint()
        raise RuntimeError(
            f"Audio recording is unavailable: {exc}.\n{hint}"
        ) from exc
    return sd


def _portaudio_install_hint() -> str:
    if sys.platform.startswith("linux"):
        return (
            "PortAudio is not installed. On Debian/Ubuntu run:\n"
            "    sudo apt-get install -y libportaudio2\n"
            "On Fedora/RHEL:  sudo dnf install -y portaudio\n"
            "On Arch:         sudo pacman -S portaudio"
        )
    if sys.platform == "darwin":
        return "Install PortAudio with Homebrew: brew install portaudio"
    return "Install the PortAudio library for your platform."


def record_wav(path: Path, seconds: int = 5, sample_rate: int = 16000) -> Path:
    sd = _load_sounddevice()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    write_wav(str(path), sample_rate, audio)
    return path


def play_audio(path: Path) -> bool:
    command = _playback_command(path)
    if command is None:
        return False
    subprocess.run(command, check=False)
    return True


def _playback_command(path: Path) -> Optional[list[str]]:
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    if shutil.which("afplay"):
        return ["afplay", str(path)]
    if shutil.which("aplay"):
        return ["aplay", str(path)]
    if shutil.which("mpg123"):
        return ["mpg123", "-q", str(path)]
    return None
