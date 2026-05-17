"""LLM-driven skill orchestrator for the reBot Arm B601-DM.

Flow
----
    prompt (text or voice)
        │
        ▼
    LLM agent  ──── picks ONE skill (+ args) from skills.yaml
        │
        ▼
    Executor   ──── runs the skill's ordered list of *subatomic* steps
                    (move_to_pose, take_picture, query_vlm, speak, ...)

Concepts
--------
- A **skill** (e.g. ``get_environment_info``) is a YAML-declared composition
  of subatomic steps. The LLM only ever sees the catalogue of skill names,
  their descriptions and (optionally) their params — never the steps.
- A **subatomic skill** (e.g. ``move_to_pose``) is a Python primitive,
  registered with the ``@subatomic("name")`` decorator. Each takes the
  current execution context plus keyword arguments and may return a value,
  which is stored under the step's ``outputs:`` key for later steps to use.
- Argument substitution: any string of the form ``"{name}"`` (or embedded
  ``"...{name}..."``) inside a step's ``args`` is resolved against the
  shared context, which is seeded with the LLM-extracted skill params and
  then accumulates outputs as steps run.

Quick start
-----------
    # Dry-run — never connects to the arm, prints what would happen:
    python orchestrator.py "what do you see on the table?"

    # Same prompt but really drive the arm:
    python orchestrator.py --execute "what do you see on the table?"

    # Interactive REPL:
    python orchestrator.py -i

    # Voice prompt (records 5 s from default mic, transcribes with Whisper):
    python orchestrator.py --voice

Dependencies
------------
- Required: ``pyyaml``, ``numpy`` (already used by trajectory.py).
- Optional (graceful degradation if missing):
    * ``openai``       — LLM skill selection + Whisper STT + GPT-4o vision.
    * ``opencv-python``— camera capture for ``take_picture``.
    * ``sounddevice``  — mic recording for ``--voice``.
    * ``pyttsx3``      — offline TTS for ``speak`` (falls back to ``espeak``
                         then to plain ``print``).

Environment
-----------
- ``OPENAI_API_KEY`` — enables the OpenAI LLM, Whisper and vision paths.
  Without it the orchestrator falls back to a transparent keyword-matching
  heuristic so the whole pipeline still works end-to-end offline.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml


# ──────────────────────────────────────────────────────────────────────
# Paths / defaults
# ──────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
DEFAULT_SKILLS_YAML = HERE / "skills.yaml"
DEFAULT_POSES_YAML = HERE / "arm_poses.yaml"

OPENAI_TEXT_MODEL = os.environ.get("ORCH_TEXT_MODEL", "gpt-5.5")
OPENAI_VISION_MODEL = os.environ.get("ORCH_VISION_MODEL", "gpt-5.5")
OPENAI_STT_MODEL = os.environ.get("ORCH_STT_MODEL", "whisper-1")


# ──────────────────────────────────────────────────────────────────────
# Execution context + subatomic-skill registry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecContext:
    """Shared state passed to every subatomic skill during a skill run.

    - ``vars``     accumulates skill params + step outputs; referenced by
                   ``"{name}"`` substitution in step args.
    - ``dry_run``  if True, hardware-touching skills only print what they
                   would do.
    - ``robot``    lazy singleton Robot handle (created on first need).
    """

    vars: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    robot: Optional[Any] = None
    poses: Dict[str, Any] = field(default_factory=dict)

    def get_robot(self):
        """Lazy-connect the arm. Only called by skills that move the arm."""
        if self.dry_run:
            return None
        if self.robot is not None:
            return self.robot
        from trajectory import Robot  # noqa: WPS433 (local import is intentional)
        self.robot = Robot()
        self.robot.connect()
        # auto-lift to home so subsequent named poses are reachable
        try:
            self.robot.home(duration=3.0)
        except Exception as exc:  # pragma: no cover
            print(f"[robot] home() failed: {exc}")
        return self.robot

    def close_robot(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None


SubatomicFn = Callable[..., Any]
SUBATOMIC_SKILLS: Dict[str, SubatomicFn] = {}


def subatomic(name: str) -> Callable[[SubatomicFn], SubatomicFn]:
    """Decorator: register a function as a subatomic skill addressable by name."""

    def _wrap(fn: SubatomicFn) -> SubatomicFn:
        if name in SUBATOMIC_SKILLS:
            raise RuntimeError(f"duplicate subatomic skill: {name!r}")
        SUBATOMIC_SKILLS[name] = fn
        return fn

    return _wrap


# ──────────────────────────────────────────────────────────────────────
# Subatomic skills — the atomic primitives that skills are composed of.
# Each takes (ctx, **kwargs) and may return a value. The returned value
# is stored in ctx.vars under the step's `outputs:` name.
# ──────────────────────────────────────────────────────────────────────


@subatomic("move_to_pose")
def _sk_move_to_pose(ctx: ExecContext, pose_name: str, duration: float = 3.0) -> None:
    """Drive the arm to a named pose from arm_poses.yaml (joint-space)."""
    entry = ctx.poses.get(pose_name)
    if entry is None:
        raise ValueError(
            f"move_to_pose: unknown pose {pose_name!r}. "
            f"Known: {sorted(ctx.poses.keys())}"
        )
    joints = entry.get("joints")
    gripper = float(entry.get("gripper", 0.0))
    print(f"  · move_to_pose({pose_name!r}, duration={duration:.1f}s)")

    if ctx.dry_run:
        print(f"      [dry-run] would drive joints→{joints}  grip→{gripper:+.3f}")
        return

    import numpy as np
    from trajectory import Tolerances
    robot = ctx.get_robot()
    if robot is None:
        return
    ok = robot.move_to_joints(
        np.asarray(joints, dtype=np.float64),
        gripper=gripper,
        duration=float(duration),
        wait=True,
        tolerances=Tolerances(joint=0.03, gripper=0.1),
    )
    if not ok:
        print(f"      WARNING: did not settle within tolerance at {pose_name!r}")


@subatomic("set_gripper")
def _sk_set_gripper(ctx: ExecContext, value: float, duration: float = 0.8) -> None:
    """Command an absolute gripper position (radians; negative = open)."""
    value = float(value)
    print(f"  · set_gripper(value={value:+.3f})")
    if ctx.dry_run:
        return
    robot = ctx.get_robot()
    if robot is None:
        return
    pose = robot.get_pose()
    robot.move_to(pose.with_gripper(value), duration=float(duration), wait=True)


@subatomic("home")
def _sk_home(ctx: ExecContext, duration: float = 3.0) -> None:
    """Lift the arm to the auto-captured home pose."""
    print(f"  · home(duration={duration:.1f}s)")
    if ctx.dry_run:
        return
    robot = ctx.get_robot()
    if robot is None:
        return
    robot.home(duration=float(duration))


@subatomic("halt")
def _sk_halt(ctx: ExecContext) -> None:
    """Resync target to current pose so the arm stops moving."""
    print("  · halt()")
    if ctx.dry_run:
        return
    robot = ctx.get_robot()
    if robot is None:
        return
    try:
        cur = robot.get_pose()
        robot.move_to(cur, duration=0.1, wait=False)
    except Exception as exc:
        print(f"      halt failed: {exc}")


@subatomic("wait")
def _sk_wait(ctx: ExecContext, seconds: float = 1.0) -> None:
    """Sleep for a fixed duration."""
    print(f"  · wait({seconds:.2f}s)")
    time.sleep(float(seconds))


@subatomic("take_picture")
def _sk_take_picture(
    ctx: ExecContext,
    camera_index: int = 0,
    save_to: Optional[str] = None,
) -> str:
    """Grab one frame from the default webcam, save it, return the file path.

    Falls back to a stub path when ``cv2`` isn't available or in dry-run.
    Downstream subatomic skills (e.g. ``query_vlm``) read the path from
    the context.
    """
    if save_to is None:
        save_to = str(Path(tempfile.gettempdir()) / f"orch_frame_{int(time.time())}.jpg")

    print(f"  · take_picture(camera_index={camera_index}) → {save_to}")

    try:
        import cv2  # type: ignore
    except ImportError:
        print("      [stub] cv2 not installed — returning empty image path")
        Path(save_to).write_bytes(b"")
        return save_to

    if ctx.dry_run:
        # Still attempt a real capture so the rest of the pipeline can run,
        # but failure is non-fatal in dry-run.
        try:
            cap = cv2.VideoCapture(int(camera_index))
            ok, frame = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(save_to, frame)
                return save_to
        except Exception as exc:
            print(f"      [dry-run] camera capture failed: {exc}")
        Path(save_to).write_bytes(b"")
        return save_to

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"take_picture: cannot open camera {camera_index}")
    try:
        # discard a few frames to let auto-exposure settle
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("take_picture: capture failed")
        cv2.imwrite(save_to, frame)
    finally:
        cap.release()
    return save_to


@subatomic("query_vlm")
def _sk_query_vlm(
    ctx: ExecContext,
    image: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """Send (image, prompt) to a vision LLM, return its textual answer."""
    model = model or OPENAI_VISION_MODEL
    print(f"  · query_vlm(model={model}, image={image!r})")
    print(f"      prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}")

    img_path = Path(image)
    if not img_path.exists() or img_path.stat().st_size == 0:
        msg = (
            "I can't see anything right now — the camera image is empty. "
            "(Stub answer; install opencv-python and connect a camera.)"
        )
        print(f"      [stub] {msg}")
        return msg

    client = _try_openai_client()
    if client is None:
        msg = (
            "[stub VLM] I would describe the captured image here, but no "
            "OPENAI_API_KEY / openai package is available."
        )
        print(f"      {msg}")
        return msg

    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"      VLM said: {text[:160]}{'…' if len(text) > 160 else ''}")
        return text
    except Exception as exc:
        msg = f"[VLM error] {exc}"
        print(f"      {msg}")
        return msg


@subatomic("speak")
def _sk_speak(ctx: ExecContext, text: str) -> None:
    """Speak ``text`` aloud (pyttsx3 → espeak → plain print fallback)."""
    print(f"  · speak: {text}")
    # Try pyttsx3 (offline cross-platform).
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass
    # Fallback: system 'espeak' if installed.
    if shutil.which("espeak"):
        try:
            subprocess.run(
                ["espeak", "-s", "165", text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    # Final fallback: already printed above.


@subatomic("log")
def _sk_log(ctx: ExecContext, text: str) -> None:
    """Plain stdout log line (no TTS)."""
    print(f"  · log: {text}")


# ──────────────────────────────────────────────────────────────────────
# Skill loading + validation
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SkillStep:
    subatomic: str
    args: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[str] = None


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str
    params: Dict[str, Any]
    steps: List[SkillStep]


def load_skills(path: Path) -> Dict[str, Skill]:
    if not path.exists():
        raise FileNotFoundError(f"skills file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("skills") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level 'skills' must be a mapping")

    out: Dict[str, Skill] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"{path}: skill {name!r} must be a mapping")
        steps_raw = body.get("steps") or []
        steps: List[SkillStep] = []
        for i, st in enumerate(steps_raw):
            if not isinstance(st, dict) or "subatomic" not in st:
                raise ValueError(
                    f"{path}: skill {name!r} step #{i} missing 'subatomic'"
                )
            sub = st["subatomic"]
            if sub not in SUBATOMIC_SKILLS:
                raise ValueError(
                    f"{path}: skill {name!r} step #{i} references unknown "
                    f"subatomic {sub!r}. Known: {sorted(SUBATOMIC_SKILLS.keys())}"
                )
            steps.append(
                SkillStep(
                    subatomic=sub,
                    args=dict(st.get("args") or {}),
                    outputs=st.get("outputs"),
                )
            )
        out[name] = Skill(
            name=name,
            description=str(body.get("description", "")).strip(),
            when_to_use=str(body.get("when_to_use", "")).strip(),
            params=dict(body.get("params") or {}),
            steps=steps,
        )
    if not out:
        raise ValueError(f"{path}: no skills defined")
    return out


def load_poses(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[warn] poses file {path} not found — move_to_pose will fail.")
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────
# Argument substitution
# ──────────────────────────────────────────────────────────────────────

_VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _substitute(value: Any, ctx: ExecContext) -> Any:
    """Recursively substitute ``{var}`` references using ``ctx.vars``.

    - ``"{name}"`` standalone → returns the raw value (any type).
    - ``"prefix {name} suffix"`` → string format (everything stringified).
    - dicts/lists are recursed into.
    """
    if isinstance(value, str):
        # Standalone reference: preserve type.
        m = re.fullmatch(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", value)
        if m:
            key = m.group(1)
            if key in ctx.vars:
                return ctx.vars[key]
            raise KeyError(f"unresolved variable in args: {{{key}}}")
        # Inline references.
        def _rep(match: "re.Match[str]") -> str:
            key = match.group(1)
            if key not in ctx.vars:
                raise KeyError(f"unresolved variable in args: {{{key}}}")
            return str(ctx.vars[key])

        return _VAR_RE.sub(_rep, value)
    if isinstance(value, dict):
        return {k: _substitute(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, ctx) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────
# LLM agent: pick ONE skill (+ args) for a user prompt
# ──────────────────────────────────────────────────────────────────────


def _try_openai_client():
    """Return an OpenAI client if the package + API key are available, else None."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore

        return OpenAI()
    except ImportError:
        return None
    except Exception as exc:
        print(f"[warn] openai client init failed: {exc}")
        return None


@dataclass
class SkillChoice:
    skill: str
    args: Dict[str, Any]
    reasoning: str = ""


def _build_catalog(skills: Dict[str, Skill]) -> str:
    """Render the skills catalog as compact JSON for the LLM."""
    catalog = []
    for s in skills.values():
        catalog.append(
            {
                "name": s.name,
                "description": s.description,
                "when_to_use": s.when_to_use,
                "params": s.params,
            }
        )
    return json.dumps(catalog, indent=2)


SYSTEM_PROMPT = """You are the planning brain of a tabletop robot arm.

You receive a single user request and a catalog of high-level SKILLS the
robot knows how to perform. Pick EXACTLY ONE skill that best satisfies the
request, and fill in any required `params` from the user's words.

Respond with STRICT JSON only, in this shape:

  {
    "skill": "<one of the skill names>",
    "args":  { ...params for that skill, may be empty... },
    "reasoning": "<one short sentence>"
  }

Rules:
- Choose the closest match. If nothing fits, pick the skill whose
  `when_to_use` is least wrong and explain in `reasoning`.
- Only emit param keys that appear in that skill's `params`.
- Never invent new skills or new params. Never wrap the JSON in prose
  or markdown fences.
"""


def choose_skill(prompt: str, skills: Dict[str, Skill]) -> SkillChoice:
    """Ask the LLM (or a heuristic fallback) which skill to run."""
    client = _try_openai_client()
    catalog_json = _build_catalog(skills)

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"SKILLS CATALOG:\n{catalog_json}\n\n"
                            f"USER REQUEST:\n{prompt}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = resp.choices[0].message.content or "{}"
            obj = json.loads(content)
            return _validate_choice(obj, skills)
        except Exception as exc:
            print(f"[warn] LLM call failed ({exc}); falling back to heuristic.")

    # Offline fallback — simple keyword scoring against descriptions.
    return _heuristic_choose(prompt, skills)


def _validate_choice(obj: Dict[str, Any], skills: Dict[str, Skill]) -> SkillChoice:
    name = str(obj.get("skill", "")).strip()
    if name not in skills:
        raise ValueError(
            f"LLM picked unknown skill {name!r}. "
            f"Known: {sorted(skills.keys())}"
        )
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    # Drop any keys the skill doesn't declare.
    allowed = set(skills[name].params.keys())
    filtered = {k: v for k, v in args.items() if k in allowed}
    return SkillChoice(
        skill=name,
        args=filtered,
        reasoning=str(obj.get("reasoning", "")).strip(),
    )


def _heuristic_choose(prompt: str, skills: Dict[str, Skill]) -> SkillChoice:
    """Last-resort offline keyword scoring. Predictable, not smart."""
    p = prompt.lower()
    keywords = {
        "stop": "stop",
        "halt": "stop",
        "freeze": "stop",
        "home": "go_home",
        "reset": "go_home",
        "stand by": "go_home",
        "what do you see": "get_environment_info",
        "what's around": "get_environment_info",
        "describe your surroundings": "get_environment_info",
        "environment": "get_environment_info",
        "where are you": "get_environment_info",
        "look around": "get_environment_info",
    }
    for kw, sk in keywords.items():
        if kw in p and sk in skills:
            return SkillChoice(
                skill=sk, args={}, reasoning=f"keyword match: {kw!r}",
            )
    # Color marker?
    if "pick_marker_by_color" in skills and (
        "pick" in p or "grab" in p or "fetch" in p
    ):
        for color in ("red", "yellow", "blue", "green", "white"):
            if color in p:
                return SkillChoice(
                    skill="pick_marker_by_color",
                    args={"color": color},
                    reasoning=f"detected color {color}",
                )
    if "describe_object" in skills and ("?" in p or p.startswith(("how", "what", "where", "is", "are"))):
        return SkillChoice(
            skill="describe_object",
            args={"question": prompt},
            reasoning="visual question fallback",
        )
    # Default: environment info.
    fallback = "get_environment_info" if "get_environment_info" in skills else next(iter(skills))
    return SkillChoice(
        skill=fallback, args={}, reasoning="no keyword match — defaulting",
    )


# ──────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────


def execute_skill(skill: Skill, args: Dict[str, Any], ctx: ExecContext) -> None:
    """Run every step of ``skill``, threading args + outputs through ``ctx.vars``."""
    print(f"\n=== Executing skill: {skill.name} ===")
    if args:
        print(f"  params: {args}")
    # Seed context with the LLM-extracted skill params.
    ctx.vars.update(args)

    for i, step in enumerate(skill.steps, 1):
        fn = SUBATOMIC_SKILLS[step.subatomic]
        try:
            resolved = _substitute(step.args, ctx)
        except KeyError as exc:
            print(f"[{i}/{len(skill.steps)}] {step.subatomic}: {exc} — aborting skill.")
            return
        print(f"[{i}/{len(skill.steps)}] {step.subatomic}")
        try:
            result = fn(ctx, **resolved)
        except Exception as exc:
            print(f"      ERROR in {step.subatomic}: {exc}")
            return
        if step.outputs:
            ctx.vars[step.outputs] = result
    print(f"=== Skill {skill.name} done ===\n")


# ──────────────────────────────────────────────────────────────────────
# Voice input (optional)
# ──────────────────────────────────────────────────────────────────────


def record_and_transcribe(seconds: float = 5.0, sample_rate: int = 16000) -> str:
    """Record from default mic for N seconds and transcribe with Whisper.

    Requires ``sounddevice`` + ``scipy`` (already installed) + ``openai`` +
    ``OPENAI_API_KEY``. Returns the transcribed string.
    """
    try:
        import sounddevice as sd  # type: ignore
        from scipy.io.wavfile import write as wav_write  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            f"voice input requires sounddevice + scipy: {exc}"
        ) from exc

    client = _try_openai_client()
    if client is None:
        raise RuntimeError(
            "voice input requires the openai package and OPENAI_API_KEY"
        )

    print(f"[voice] recording {seconds:.1f}s …")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("[voice] transcribing …")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_write(tmp.name, sample_rate, audio)
        wav_path = tmp.name
    try:
        with open(wav_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL, file=f,
            )
        text = (tr.text or "").strip()
        print(f"[voice] heard: {text!r}")
        return text
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "prompt", nargs="*",
        help="The user prompt (omit and use --voice or -i instead).",
    )
    p.add_argument(
        "--skills", type=Path, default=DEFAULT_SKILLS_YAML,
        help=f"Skills YAML file (default {DEFAULT_SKILLS_YAML.name}).",
    )
    p.add_argument(
        "--poses", type=Path, default=DEFAULT_POSES_YAML,
        help=f"Named poses YAML (default {DEFAULT_POSES_YAML.name}).",
    )
    p.add_argument(
        "--execute", action="store_true",
        help="Actually connect to and drive the robot. Default is dry-run.",
    )
    p.add_argument(
        "--voice", action="store_true",
        help="Take the prompt from the microphone instead of argv.",
    )
    p.add_argument(
        "--voice-seconds", type=float, default=5.0,
        help="Voice recording duration in seconds (default 5).",
    )
    p.add_argument(
        "-i", "--interactive", action="store_true",
        help="Loop: read prompts from stdin until EOF / quit.",
    )
    p.add_argument(
        "--plan-only", action="store_true",
        help="Only print the chosen skill + args, don't execute.",
    )
    p.add_argument(
        "--list-skills", action="store_true",
        help="List available skills and exit.",
    )
    p.add_argument(
        "--mic-server", action="store_true",
        help="Start the ElevenLabs voice bridge (interactive/elevenlabs.py) "
             "and consume each transcribed prompt as a new task. The "
             "orchestrator runs prompts strictly one at a time.",
    )
    p.add_argument(
        "--host", default="0.0.0.0",
        help="Bind interface for --mic-server (default 0.0.0.0).",
    )
    p.add_argument(
        "--port", type=int, default=8000,
        help="Bind port for --mic-server (default 8000).",
    )
    return p.parse_args()


def _list_skills(skills: Dict[str, Skill]) -> None:
    print("Available skills:")
    for s in skills.values():
        print(f"\n  {s.name}")
        if s.description:
            print(f"    {s.description}")
        if s.params:
            print(f"    params: {list(s.params.keys())}")
        print(f"    steps ({len(s.steps)}):")
        for st in s.steps:
            extras = []
            if st.args:
                extras.append(f"args={st.args}")
            if st.outputs:
                extras.append(f"→{st.outputs}")
            tail = ("  " + "  ".join(extras)) if extras else ""
            print(f"      - {st.subatomic}{tail}")


def _handle_prompt(prompt: str, skills: Dict[str, Skill], ctx: ExecContext, plan_only: bool) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    print(f"\n>>> prompt: {prompt}")
    choice = choose_skill(prompt, skills)
    print(f"[llm] skill = {choice.skill}  args = {choice.args}")
    if choice.reasoning:
        print(f"[llm] reason: {choice.reasoning}")
    if plan_only:
        return
    skill = skills[choice.skill]
    # Fresh per-run context vars (but keep robot handle + poses across runs).
    ctx.vars = {}
    execute_skill(skill, choice.args, ctx)


# ──────────────────────────────────────────────────────────────────────
# Voice bridge integration (library mode)
# ──────────────────────────────────────────────────────────────────────


def _serve_mic(
    skills: Dict[str, Skill],
    ctx: ExecContext,
    *,
    host: str,
    port: int,
    plan_only: bool,
) -> None:
    """Run the orchestrator behind the ElevenLabs voice bridge.

    The bridge captures + transcribes audio asynchronously and notifies
    us — via its transcript queue — every time a new task is available.
    We pull prompts off the queue and dispatch them through the same
    LLM-driven planner as the rest of the CLI, ensuring one task fully
    completes before the next is started.
    """
    import asyncio  # local: only needed for this code path

    # Local import keeps the orchestrator usable even without aiohttp
    # installed for users who never touch the voice path.
    from interactive.elevenlabs import ElevenLabsInteraction

    async def _run() -> None:
        # Library mode: the bridge only captures + transcribes; we take
        # ownership of running the LLM and the skill steps. The bridge
        # itself never imports the orchestrator.
        bridge = ElevenLabsInteraction(
            host=host,
            port=port,
            on_transcript=lambda t: print(
                f"[orchestrator] notified of new task: {t!r}"
            ),
        )

        loop = asyncio.get_running_loop()

        async def consumer() -> None:
            while True:
                prompt = await bridge.get_next_prompt()
                bridge.mark_running(prompt)
                try:
                    # _handle_prompt is synchronous (it calls the LLM,
                    # drives the arm, etc.). Awaiting run_in_executor
                    # here both keeps the web server responsive AND
                    # ensures we never start the next prompt until the
                    # previous one has finished.
                    await loop.run_in_executor(
                        None, _handle_prompt, prompt, skills, ctx, plan_only,
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"[orchestrator] error on {prompt!r}: {exc}")
                finally:
                    bridge.mark_done()

        consumer_task = asyncio.create_task(consumer(), name="orch-consumer")
        try:
            await bridge.serve()
        finally:
            consumer_task.cancel()
            try:
                await consumer_task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[orchestrator] mic server shutting down …")


def main() -> int:
    args = _parse_args()
    try:
        skills = load_skills(args.skills)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading skills: {exc}", file=sys.stderr)
        return 1

    if args.list_skills:
        _list_skills(skills)
        return 0

    ctx = ExecContext(
        dry_run=not args.execute,
        poses=load_poses(args.poses),
    )
    if ctx.dry_run:
        print("[mode] DRY-RUN (no hardware will move). Pass --execute to drive the arm.")
    else:
        print("[mode] EXECUTE — the arm will move.")

    try:
        if args.mic_server:
            _serve_mic(
                skills, ctx,
                host=args.host, port=args.port,
                plan_only=args.plan_only,
            )
            return 0

        if args.interactive:
            print("Interactive mode. Type a prompt; Ctrl-D or 'quit' to exit.")
            while True:
                try:
                    line = input("orchestrator> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if line.strip().lower() in {"quit", "exit", "q"}:
                    break
                _handle_prompt(line, skills, ctx, args.plan_only)
            return 0

        if args.voice:
            prompt = record_and_transcribe(seconds=args.voice_seconds)
        elif args.prompt:
            prompt = " ".join(args.prompt)
        else:
            print(
                "No prompt supplied. Pass one as args, use --voice, or -i for "
                "interactive mode. Try --list-skills.",
                file=sys.stderr,
            )
            return 2

        _handle_prompt(prompt, skills, ctx, args.plan_only)
        return 0
    finally:
        ctx.close_robot()


if __name__ == "__main__":
    sys.exit(main())
