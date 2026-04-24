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
    wa   = cp.get("world_assumptions") or {}
    spk  = cp.get("speaker") or {}
    addr = cp.get("addressee") or {}
    arc  = cp.get("emotional_arc") or {}
    mot  = cp.get("motivation") or {}
    meta = cp.get("meta") or {}
    lines = cp.get("line_meanings") or []

    # --- Structure summary (item 3 of the input contract) -----------------
    # Pass through the actual verse / chorus / bridge / etc. breakdown,
    # not just a "has_repetition" boolean. Counts are enough for narrative
    # strategy — the engine doesn't need the full text.
    structure_counts: Dict[str, int] = {}
    for lm in lines:
        fn = str(lm.get("function") or "").strip().lower()
        if fn:
            structure_counts[fn] = structure_counts.get(fn, 0) + 1
    has_repetition = any(
        str(lm.get("repeat_status", "")).lower() == "repeat" for lm in lines
    )

    # --- Audio rhythm (item 7 of the input contract, optional) ------------
    # BPM rides through Context as cp["audio_meta"]["bpm"] when available.
    # Per-line timestamps are attached by the orchestrator as
    # line_meanings[].lyric_start_seconds / lyric_end_seconds. Total
    # duration is derived from the last timed line. Everything is optional;
    # a piece with no audio simply omits these fields.
    audio_meta = cp.get("audio_meta") or {}
    bpm = audio_meta.get("bpm")
    timed_lines = [
        lm for lm in lines
        if isinstance(lm.get("lyric_start_seconds"), (int, float))
        and isinstance(lm.get("lyric_end_seconds"), (int, float))
    ]
    total_duration_s: Optional[float] = None
    if timed_lines:
        try:
            total_duration_s = round(
                max(float(lm["lyric_end_seconds"]) for lm in timed_lines), 2
            )
        except (TypeError, ValueError):
            total_duration_s = None

    context_summary: Dict[str, Any] = {
        "input_type":        cp.get("input_type") or cp.get("recognized_type"),
        "narrative_mode":    cp.get("narrative_mode"),
        "location_dna":      cp.get("location_dna"),
        # 1. Meaning of the input
        "core_theme":        cp.get("core_theme"),
        "dramatic_premise":  cp.get("dramatic_premise"),
        "narrative_spine":   cp.get("narrative_spine"),
        # 2. Voice / perspective
        "speaker": {
            "identity":                  spk.get("identity"),
            "gender":                    spk.get("gender"),
            "age_range":                 spk.get("age_range"),
            "emotional_state":           spk.get("emotional_state"),
            "social_role":               spk.get("social_role"),
            "cultural_background":       spk.get("cultural_background"),
            "relationship_to_addressee": spk.get("relationship_to_addressee"),
        },
        "addressee": {
            "identity":     addr.get("identity"),
            "relationship": addr.get("relationship"),
            "presence":     addr.get("presence"),
        },
        # 4. Timeline nature + 5. Cultural / world grounding
        "world": {
            "geography":      wa.get("geography"),
            "era":            wa.get("era"),
            "season":         wa.get("season"),
            "timeline_nature": wa.get("timeline_nature"),
            "social_context": wa.get("social_context"),
        },
        # 6. Emotional progression
        "emotional_arc": {
            "opening":     arc.get("opening"),
            "development": arc.get("development"),
            "climax":      arc.get("climax"),
            "resolution":  arc.get("resolution"),
        },
        # 1c. Conflict
        "motivation": {
            "inciting_cause":    mot.get("inciting_cause"),
            "underlying_desire": mot.get("underlying_desire"),
            "stakes":            mot.get("stakes"),
            "obstacle":          mot.get("obstacle"),
        },
        # 3. Structure
        "structure": {
            "function_counts": structure_counts or None,
            "has_repetition":  has_repetition,
            "line_count":      len(lines),
        },
        # 7. Duration / rhythm (optional)
        "rhythm": {
            "bpm":                bpm,
            "total_duration_s":   total_duration_s,
            "timed_line_count":   len(timed_lines) or None,
        },
        # Meta — abstraction guidance for storytelling tone
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

  expression_channels: what carries meaning — order them by importance for THIS piece

INPUTS YOU RECEIVE (and how to use them):
  speaker + addressee
    — Use the relationship_to_addressee, addressee.presence and
      addressee.relationship to decide presence_logic. If the addressee is
      "absent" or "memory_only", lean toward fragmented or memory_only
      presence. If the addressee is on-screen and reciprocal, prefer
      always_visible.
  world.timeline_nature
    — real_time   → linear timeline_behavior is most natural
    — memory      → memory_based or fragmented
    — cyclical    → cyclical
    — ambiguous   → fragmented or symbolic, your call
  world.social_context + world.geography
    — Inform whether the storytelling_mode should feel intimate
      (observational), heightened (expressive), allegorical (symbolic), or
      direct (performative). They never become locations or visuals.
  structure.function_counts
    — Counts of verse / chorus / bridge / etc. lines. A high chorus count
      means the repetition_strategy field matters; a flat structure (no
      chorus) makes "none" appropriate. Bridges usually warrant an
      emotional_progression shift around their position.
  rhythm.bpm + rhythm.total_duration_s
    — High BPM + short duration → motion_philosophy=dynamic_dominant.
    — Low BPM + long duration → still_dominant.
    — Mixed energy → mixed.
    — These are optional; if missing, decide from the emotional arc alone.
  symbolic_density + abstraction_level
    — Higher symbolic density / abstraction → favor symbolic storytelling
      and associative or symbolic continuity_style."""

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
