import json
import re

from openai import AsyncOpenAI

from config import MAX_PLAN_STEPS, OPENAI_API_KEY, OPENAI_VLM_MODEL, VISION_ENABLED
from interfaces import BaseDriver
from models import CameraFrame, OrchestratorDecision, PlanStep, SkillSpec, WorldState

NORMALCY_USER_MESSAGE = "__normalcy_check__"

SYSTEM_PROMPT = """You are Alfred, a polite kitchen butler robot with access to skills.

You receive:
- The user's request (or a proactive kitchen check)
- A JSON skill catalogue (only use listed skills)
- Current WorldState
- Optionally a webcam image of the kitchen

Respond with JSON only (no markdown fences):
{{
  "reply": "What you say to the user — include what you see when relevant, address them as sir",
  "steps": [{{"skill": "skill_name", "params": {{...}}}}],
  "update_messy": true | false | null
}}

Rules:
- Use the fewest steps needed; empty steps is fine if only narration is required
- Only use skills from the catalogue; valid names: {skill_names}
- For notes, put the full note text in params.text
- For tidying, use manipulate with action reset
- Do not invent skills
- Max {max_steps} steps
"""

MULTIMODAL_SYSTEM = """You are Alfred, a vision-capable kitchen butler robot.

A webcam photo is attached to the user message. You CAN see it. Never say you cannot interpret images.
Describe visible objects, colors, surfaces, clutter, and people. The image overrides WorldState for what is true right now.

You may also run skills from the catalogue. Respond with JSON only:
{{
  "reply": "Polite response to the user, including what you see in the photo, address them as sir",
  "steps": [{{"skill": "skill_name", "params": {{...}}}}],
  "update_messy": true | false | null
}}

Rules:
- Valid skills only: {skill_names}
- For notes, params.text = exact content to save
- For tidying, manipulate with action reset
- Max {max_steps} steps
- If the user only asks what you see, steps may be empty but reply must describe the photo
"""

TEXT_ONLY_ADDENDUM = """
No webcam image is attached to this turn (capture failed or vision disabled).
Do not claim you can see the room. Say you are relying on stored WorldState and conversation only.
If the user asks what you see, explain that the camera frame was not available and suggest checking the webcam.
"""


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _parse_decision(raw: str, allowed_skills: set[str]) -> OrchestratorDecision:
    data = json.loads(_strip_json_fence(raw))
    steps: list[PlanStep] = []
    for item in data.get("steps", [])[:MAX_PLAN_STEPS]:
        skill = item.get("skill", "")
        if skill not in allowed_skills:
            continue
        steps.append(PlanStep(skill=skill, params=item.get("params") or {}))
    return OrchestratorDecision(
        reply=data.get("reply", "Very good, sir."),
        steps=steps,
        update_messy=data.get("update_messy"),
    )


class OpenAIVLMDriver(BaseDriver):
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def initialize(self) -> None:
        if OPENAI_API_KEY:
            self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            print(f"[OpenAIVLMDriver] Ready. Model: {OPENAI_VLM_MODEL}")
        else:
            print("[OpenAIVLMDriver] No OPENAI_API_KEY — using heuristic fallback.")

    async def health_check(self) -> bool:
        return OPENAI_API_KEY is not None and self._client is not None

    def _build_system_prompt(self, catalogue: list[SkillSpec]) -> str:
        names = ", ".join(s.name for s in catalogue)
        return SYSTEM_PROMPT.format(skill_names=names, max_steps=MAX_PLAN_STEPS)

    def _build_context_text(
        self,
        user_message: str,
        catalogue: list[SkillSpec],
        state: WorldState,
        history: list[dict[str, str]] | None,
    ) -> str:
        catalogue_json = json.dumps([s.model_dump() for s in catalogue], indent=2)
        state_json = state.model_dump_json(indent=2)
        history_text = ""
        if history:
            lines = [f"- User: {h['user']}\n  Alfred: {h['reply']}" for h in history[-5:]]
            history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
        return (
            f"{history_text}"
            f"Skill catalogue:\n{catalogue_json}\n\n"
            f"WorldState:\n{state_json}\n\n"
            f"User message: {user_message}"
        )

    async def orchestrate(
        self,
        *,
        user_message: str,
        catalogue: list[SkillSpec],
        state: WorldState,
        frame: CameraFrame | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> OrchestratorDecision:
        allowed = {s.name for s in catalogue}
        has_frame = frame is not None and bool(frame.jpeg_b64)
        use_vision = VISION_ENABLED and has_frame and self._client is not None

        if self._client is None:
            return _heuristic_orchestrate(
                user_message, catalogue, state, has_image=has_frame
            )

        if use_vision:
            print(
                f"[OpenAIVLMDriver] Multimodal: sending {frame.width}x{frame.height} frame to {OPENAI_VLM_MODEL}"
            )
            return await self._orchestrate_multimodal(
                user_message, catalogue, state, frame, history, allowed
            )

        reason = []
        if not VISION_ENABLED:
            reason.append("VISION_ENABLED=false")
        if not has_frame:
            reason.append("no webcam frame")
        print(f"[OpenAIVLMDriver] Text-only fallback ({', '.join(reason) or 'unknown'})")
        return await self._orchestrate_text_only(
            user_message, catalogue, state, history, allowed
        )

    async def _orchestrate_multimodal(
        self,
        user_message: str,
        catalogue: list[SkillSpec],
        state: WorldState,
        frame: CameraFrame,
        history: list[dict[str, str]] | None,
        allowed: set[str],
    ) -> OrchestratorDecision:
        assert self._client is not None
        context = self._build_context_text(user_message, catalogue, state, history)
        names = ", ".join(s.name for s in catalogue)
        system = MULTIMODAL_SYSTEM.format(
            skill_names=names, max_steps=MAX_PLAN_STEPS
        )
        response = await self._client.chat.completions.create(
            model=OPENAI_VLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame.jpeg_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "The image above is the live webcam view.\n\n" + context
                            ),
                        },
                    ],
                },
            ],
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return _parse_decision(raw, allowed)

    async def _orchestrate_text_only(
        self,
        user_message: str,
        catalogue: list[SkillSpec],
        state: WorldState,
        history: list[dict[str, str]] | None,
        allowed: set[str],
    ) -> OrchestratorDecision:
        assert self._client is not None
        context = self._build_context_text(user_message, catalogue, state, history)
        system = self._build_system_prompt(catalogue) + TEXT_ONLY_ADDENDUM
        response = await self._client.chat.completions.create(
            model=OPENAI_VLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return _parse_decision(raw, allowed)


def _heuristic_orchestrate(
    user_message: str,
    catalogue: list[SkillSpec],
    state: WorldState,
    *,
    has_image: bool,
) -> OrchestratorDecision:
    """Offline fallback when OPENAI_API_KEY is not set."""
    allowed = {s.name for s in catalogue}
    lowered = user_message.lower()
    steps: list[PlanStep] = []

    if user_message == NORMALCY_USER_MESSAGE:
        if state.is_messy or state.is_out_of_place():
            if "manipulate" in allowed:
                steps.append(PlanStep(skill="manipulate", params={"action": "reset"}))
            return OrchestratorDecision(
                reply=(
                    "I noticed the kitchen could use attention, sir. I shall restore order."
                    if has_image
                    else "The recorded state suggests tidying is needed, sir."
                ),
                steps=steps,
                update_messy=False if steps else None,
            )
        return OrchestratorDecision(
            reply="All appears in order, sir.",
            steps=[],
        )

    if any(w in lowered for w in ("note", "remember", "record")):
        text = user_message
        for prefix in ("take a note about", "note about", "remember", "record"):
            if prefix in lowered:
                idx = lowered.index(prefix) + len(prefix)
                text = user_message[idx:].strip(" :")
                break
        if "note" in allowed and text:
            steps.append(PlanStep(skill="note", params={"text": text}))
            return OrchestratorDecision(
                reply=f"Certainly, sir. I shall note that.",
                steps=steps,
            )

    if any(w in lowered for w in ("tidy", "clean", "reset", "fix", "messy", "untidy")):
        if "manipulate" in allowed:
            steps.append(PlanStep(skill="manipulate", params={"action": "reset"}))
        return OrchestratorDecision(
            reply="I shall tidy the kitchen for you, sir.",
            steps=steps,
            update_messy=False if steps else None,
        )

    if any(w in lowered for w in ("see", "look", "describe", "what", "counter", "kitchen")):
        obs = state.last_observation or "the kitchen as last recorded"
        return OrchestratorDecision(
            reply=(
                f"From what I can discern, sir: {obs}."
                if not has_image
                else "I am looking at the kitchen now, sir, though full vision requires an API key."
            ),
            steps=[],
        )

    return OrchestratorDecision(
        reply="I did not quite follow, sir. You might ask me to look around, take a note, or tidy up.",
        steps=[],
    )
