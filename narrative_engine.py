"""
narrative_engine.py — Stage 3 of the Qaivid cinematic pipeline.

Sits between the Context Engine (5 Ws) and the Creative Brief / Storyboard.

Role: Define HOW the narrative unfolds — storytelling strategy, not visuals.
      Upstream defines meaning. Narrative Intelligence defines expression logic.
      Downstream interprets creatively within that logic.

Rules (per master spec):
  - No scenes, no locations, no visual descriptions
  - No camera instructions
  - Lock meaning, keep visual expression open
  - Be decisive — pick one clear option per field

Output (stored as context_packet["narrative_intelligence"]):
  storytelling_mode     — observational | expressive | symbolic | performative
  presence_logic        — always_visible | fragmented | memory_only | absent
  timeline_behavior     — linear | fragmented | memory_based | cyclical
  emotional_progression — building | releasing | oscillating | still
  repetition_strategy   — visual_variation | emotional_shift | reinforce | none
  motion_philosophy     — still_dominant | dynamic_dominant | mixed
  expression_channels   — ordered list of what carries meaning most (character,
                          environment, objects, time, light, performance)
  continuity_style      — strict | associative | symbolic
  director_note         — one sentence: overall visual storytelling approach
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"


async def generate_narrative_intelligence(
    api_key: str,
    context_packet: Dict[str, Any],
    **_ignored_downstream: Any,
) -> Dict[str, Any]:
    """
    Stage 3 — Narrative Intelligence.

    Consumes ONLY the immediate predecessor's output (context_packet from
    Stage 2). MUST NOT read style_profile, storyboard, brief, or anything
    from a stage further downstream. Any unexpected kwargs are silently
    ignored so legacy callers don't crash while they get cleaned up.

    Output is purely strategic — no scenes, no locations, no camera.
    On LLM failure the pipeline continues with safe defaults.
    """
    if _ignored_downstream:
        logger.warning(
            "Narrative Engine (Stage 3) received downstream kwargs %s — "
            "chain violation; caller must stop passing these.",
            sorted(_ignored_downstream.keys()),
        )
    if not context_packet:
        return _fallback()

    client = AsyncOpenAI(api_key=api_key)
    system_msg, user_msg = _build_prompts(context_packet)

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=500,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return _repair(data)
    except Exception as exc:
        logger.warning("Narrative engine failed (%s) — using safe defaults", exc)
        return _fallback()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompts(cp: Dict[str, Any]) -> tuple[str, str]:
    wa  = cp.get("world_assumptions") or {}
    spk = cp.get("speaker") or {}
    arc = cp.get("emotional_arc") or {}
    mot = cp.get("motivation") or {}
    meta = cp.get("meta") or {}

    # Detect whether the piece has repetition (songs with chorus/hook)
    has_repetition = any(
        str(lm.get("repeat_status", "")).lower() == "repeat"
        for lm in (cp.get("line_meanings") or [])
    )

    context_summary = {
        "input_type":        cp.get("input_type") or cp.get("recognized_type"),
        "narrative_mode":    cp.get("narrative_mode"),
        "location_dna":      cp.get("location_dna"),
        "core_theme":        cp.get("core_theme"),
        "dramatic_premise":  cp.get("dramatic_premise"),
        "narrative_spine":   cp.get("narrative_spine"),
        "speaker_identity":  spk.get("identity"),
        "speaker_gender":    spk.get("gender"),
        "geography":         wa.get("geography"),
        "season":            wa.get("season"),
        "era":               wa.get("era"),
        "emotional_arc":     {
            "opening":    arc.get("opening"),
            "development": arc.get("development"),
            "climax":     arc.get("climax"),
            "resolution": arc.get("resolution"),
        },
        "motivation": {
            "inciting_cause":    mot.get("inciting_cause"),
            "underlying_desire": mot.get("underlying_desire"),
            "stakes":            mot.get("stakes"),
            "obstacle":          mot.get("obstacle"),
        },
        "has_repetition":    has_repetition,
        "symbolic_density":  meta.get("symbolic_density"),
        "abstraction_level": meta.get("abstraction_level"),
    }

    # PIPELINE CHAIN RULE: No style_profile here. Style is Stage 4 — it
    # comes AFTER Narrative, so Narrative cannot see it. style_line is kept
    # as an empty string so existing string interpolation below still works.
    style_line = ""

    system_msg = """\
You are a senior narrative director — the layer between story analysis and
visual production. Given a fully analyzed piece of content (song, script,
story, ad), you decide the narrative strategy: HOW the piece is told, not
WHAT it shows.

Your decisions shape how the creative brief and storyboard will approach
scene construction, character presence, emotional pacing, and continuity.

STRICT RULES:
- No scenes, no specific locations, no visual descriptions
- No camera instructions (no framing, no lens, no shot type)
- No color or lighting specifics
- Only strategic storytelling decisions
- Be decisive — pick one option per field, do not hedge
- Let the emotional truth of the content drive every choice

Return ONLY valid JSON with exactly these keys:

{
  "storytelling_mode": "observational|expressive|symbolic|performative",
  "presence_logic": "always_visible|fragmented|memory_only|absent",
  "timeline_behavior": "linear|fragmented|memory_based|cyclical",
  "emotional_progression": "building|releasing|oscillating|still",
  "repetition_strategy": "visual_variation|emotional_shift|reinforce|none",
  "motion_philosophy": "still_dominant|dynamic_dominant|mixed",
  "expression_channels": ["ordered list from most to least important — choose from: character, environment, objects, time, light, performance"],
  "continuity_style": "strict|associative|symbolic",
  "director_note": "one sentence capturing the overall visual storytelling approach"
}

Field guidance:
  storytelling_mode:
    observational  — camera watches, does not intrude; naturalistic
    expressive     — camera and environment reflect inner state
    symbolic       — meaning carried by objects, nature, abstraction
    performative   — artist performs directly; energy is the message

  presence_logic:
    always_visible — character on screen throughout
    fragmented     — character appears and disappears; present in parts
    memory_only    — character seen only in flashback or imagination
    absent         — no character; world carries the story

  timeline_behavior:
    linear        — story moves forward in time
    fragmented    — time cuts between moments non-linearly
    memory_based  — past and present intercut
    cyclical      — returns to the same moment or image

  repetition_strategy (for songs with chorus/hook):
    visual_variation  — same lyric, different visual treatment each time
    emotional_shift   — same lyric, rising emotional intensity each time
    reinforce         — same visual used as anchor motif
    none              — no special handling (not a song or no repetition)

  expression_channels: what carries meaning — order them by importance for THIS piece"""

    user_msg = f"""\
Content to analyze:
{json.dumps(context_summary, ensure_ascii=False, indent=2)}
{style_line}

Decide the narrative strategy. Return JSON only."""

    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Validation / repair
# ---------------------------------------------------------------------------

_VALID: Dict[str, set] = {
    "storytelling_mode":    {"observational", "expressive", "symbolic", "performative"},
    "presence_logic":       {"always_visible", "fragmented", "memory_only", "absent"},
    "timeline_behavior":    {"linear", "fragmented", "memory_based", "cyclical"},
    "emotional_progression":{"building", "releasing", "oscillating", "still"},
    "repetition_strategy":  {"visual_variation", "emotional_shift", "reinforce", "none"},
    "motion_philosophy":    {"still_dominant", "dynamic_dominant", "mixed"},
    "continuity_style":     {"strict", "associative", "symbolic"},
}

_DEFAULTS: Dict[str, str] = {
    "storytelling_mode":    "expressive",
    "presence_logic":       "always_visible",
    "timeline_behavior":    "linear",
    "emotional_progression":"building",
    "repetition_strategy":  "visual_variation",
    "motion_philosophy":    "mixed",
    "continuity_style":     "associative",
}

_VALID_CHANNELS = {"character", "environment", "objects", "time", "light", "performance"}
_DEFAULT_CHANNELS = ["character", "environment", "objects", "time", "light"]


def _repair(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, valid_set in _VALID.items():
        val = str(data.get(key) or "").strip().lower()
        out[key] = val if val in valid_set else _DEFAULTS[key]

    raw_channels = data.get("expression_channels")
    if isinstance(raw_channels, list):
        cleaned = [str(c).strip().lower() for c in raw_channels
                   if str(c).strip().lower() in _VALID_CHANNELS]
        out["expression_channels"] = cleaned or _DEFAULT_CHANNELS
    else:
        out["expression_channels"] = _DEFAULT_CHANNELS

    out["director_note"] = (
        str(data.get("director_note") or "").strip()
        or "Let the emotional arc drive visual interpretation."
    )
    return out


def _fallback() -> Dict[str, Any]:
    return {
        **_DEFAULTS,
        "expression_channels": _DEFAULT_CHANNELS,
        "director_note": "Let the emotional arc drive visual interpretation.",
    }


# ---------------------------------------------------------------------------
# Human-readable summary (used in downstream system prompts)
# ---------------------------------------------------------------------------

def format_for_prompt(ni: Dict[str, Any]) -> str:
    """Return a compact, readable block for injection into creative brief
    and storyboard system prompts."""
    if not ni:
        return ""

    channels = ", ".join(ni.get("expression_channels") or _DEFAULT_CHANNELS)

    return f"""\
NARRATIVE INTELLIGENCE (defines HOW this story is told — follow this strategy):
  Storytelling mode:     {ni.get('storytelling_mode')}
  Character presence:    {ni.get('presence_logic')}
  Timeline behavior:     {ni.get('timeline_behavior')}
  Emotional progression: {ni.get('emotional_progression')}
  Repetition strategy:   {ni.get('repetition_strategy')}
  Motion philosophy:     {ni.get('motion_philosophy')}
  Expression channels:   {channels}  (ordered most → least important)
  Continuity style:      {ni.get('continuity_style')}
  Director's approach:   {ni.get('director_note')}"""
