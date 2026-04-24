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

Brain reads (all data must already exist on Project Brain before this runs):
  - raw_input         (Stage 0) — clean text, genre, audio meta, timed lyrics
  - input_structure   (Stage 1) — input_type, language, sections, units,
                                   repetition_map, speaker_boundaries, uncertainties
  - context_packet    (Stage 2) — core_theme, emotional_arc, speaker/addressee,
                                   timeline_nature, must_preserve, creative_freedom
  - project_settings  (Stage 0) — genre, duration_target, style_preset (optional)

Output written to brain["narrative_packet"]:
  storytelling_mode             — observational | expressive | symbolic | performative
  perspective                   — first_person | third_person_close | observer | omniscient
  timeline_strategy             — linear | fragmented | memory_based | cyclical
  presence_strategy             — always_visible | fragmented | memory_only | absent
  emotional_progression_strategy — building | releasing | oscillating | still
  repetition_strategy           — visual_variation | emotional_shift | reinforce | none
  motion_philosophy             — still_dominant | dynamic_dominant | mixed
  expression_channels           — ordered list: character, environment, objects,
                                   time, light, performance
  continuity_rules              — strict | associative | symbolic
  scene_interpretation_rules    — one rule sentence for how scenes connect
  variation_allowance           — strict | guided | open
  director_note                 — one sentence: overall visual storytelling approach
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
    input_structure: Optional[Dict[str, Any]] = None,
    project_settings: Optional[Dict[str, Any]] = None,
    **_ignored_downstream: Any,
) -> Dict[str, Any]:
    """
    Stage 3 — Narrative Intelligence.

    Reads from Project Brain:
      - context_packet    (required) — core meaning from Context Engine
      - input_structure   (optional) — structural facts from Input Processor
      - project_settings  (optional) — platform, duration, style preference

    Must NOT read style_profile directly, storyboard, brief, or anything
    from a stage further downstream.

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
    system_msg, user_msg = _build_prompts(
        context_packet,
        input_structure=input_structure or {},
        project_settings=project_settings or {},
    )

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=700,
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

def _build_prompts(
    cp: Dict[str, Any],
    input_structure: Dict[str, Any],
    project_settings: Dict[str, Any],
) -> tuple[str, str]:

    # ── Context packet fields ───────────────────────────────────────────────
    wa   = cp.get("world_assumptions") or {}
    spk  = cp.get("speaker") or {}
    addr = cp.get("addressee") or {}
    arc  = cp.get("emotional_arc") or {}
    mot  = cp.get("motivation") or {}
    meta = cp.get("meta") or {}
    lines = cp.get("line_meanings") or []

    # ── Structural facts from input_structure (Brain read) ──────────────────
    # If input_structure is available (Stage 1 brain output), prefer its
    # repetition_map and speaker_boundaries directly. Fall back to deriving
    # from context_packet line_meanings for backward compatibility.
    repetition_map = input_structure.get("repetition_map") or {}
    speaker_boundaries = input_structure.get("speaker_boundaries") or []
    uncertainties = input_structure.get("uncertainty_flags") or []
    sections = input_structure.get("sections") or []
    ip_input_type = input_structure.get("input_type")
    ip_language = input_structure.get("languages") or input_structure.get("language")
    lyrical_patterns = input_structure.get("lyrical_patterns") or {}

    # Section counts derived from input_structure sections (preferred) or
    # fall back to context_packet line_meanings for structure_counts.
    section_counts: Dict[str, int] = {}
    if sections:
        for sec in sections:
            label = str(sec.get("label") or "").strip().lower()
            if label:
                section_counts[label] = section_counts.get(label, 0) + 1
    else:
        for lm in lines:
            fn = str(lm.get("function") or "").strip().lower()
            if fn:
                section_counts[fn] = section_counts.get(fn, 0) + 1

    has_repetition = bool(repetition_map) or any(
        str(lm.get("repeat_status", "")).lower() == "repeat" for lm in lines
    )

    # Speaker count from speaker_boundaries
    unique_speakers = len({
        sb.get("speaker") for sb in speaker_boundaries
        if sb.get("speaker") and sb["speaker"] != "narrator"
    })

    # ── Audio rhythm from context_packet (or project_settings) ─────────────
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
            pass
    # Fall back to project_settings duration if available
    if total_duration_s is None:
        total_duration_s = project_settings.get("duration_seconds")

    # ── Compose the payload the LLM will reason over ───────────────────────
    context_summary: Dict[str, Any] = {
        # ── From context_packet (Stage 2 output) ──
        "input_type":       ip_input_type or cp.get("input_type") or cp.get("recognized_type"),
        "language":         ip_language,
        "narrative_mode":   cp.get("narrative_mode"),
        "location_dna":     cp.get("location_dna"),
        "core_theme":       cp.get("core_theme"),
        "dramatic_premise": cp.get("dramatic_premise"),
        "narrative_spine":  cp.get("narrative_spine"),
        "must_preserve":    cp.get("must_preserve"),
        "creative_freedom": cp.get("creative_freedom"),
        # ── Voice / perspective ──
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
        # ── World / timeline ──
        "world": {
            "geography":       wa.get("geography"),
            "era":             wa.get("era"),
            "season":          wa.get("season"),
            "timeline_nature": wa.get("timeline_nature"),
            "social_context":  wa.get("social_context"),
        },
        # ── Emotional arc ──
        "emotional_arc": {
            "opening":     arc.get("opening"),
            "development": arc.get("development"),
            "climax":      arc.get("climax"),
            "resolution":  arc.get("resolution"),
        },
        # ── Conflict / motivation ──
        "motivation": {
            "inciting_cause":    mot.get("inciting_cause"),
            "underlying_desire": mot.get("underlying_desire"),
            "stakes":            mot.get("stakes"),
            "obstacle":          mot.get("obstacle"),
        },
        # ── Structure (from input_structure — Brain read) ──
        "structure": {
            "section_counts":    section_counts or None,
            "has_repetition":    has_repetition,
            "unique_speakers":   unique_speakers if unique_speakers > 0 else None,
            "line_count":        len(lines) or len(sections),
            "repetition_map":    repetition_map if repetition_map else None,
            "lyrical_patterns":  lyrical_patterns if lyrical_patterns else None,
            "uncertainties":     uncertainties[:3] if uncertainties else None,
        },
        # ── Audio / rhythm ──
        "rhythm": {
            "bpm":              bpm,
            "total_duration_s": total_duration_s,
            "timed_line_count": len(timed_lines) or None,
        },
        # ── Meta ──
        "symbolic_density":  meta.get("symbolic_density"),
        "abstraction_level": meta.get("abstraction_level"),
        # ── Project settings (from Brain project_settings namespace) ──
        "project_settings": {
            "genre":         project_settings.get("genre"),
            "style_preset":  project_settings.get("style_preset"),
            "platform":      project_settings.get("platform"),
            "duration_target_s": project_settings.get("duration_seconds"),
        } if project_settings else None,
    }

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
  "perspective": "first_person|third_person_close|observer|omniscient",
  "timeline_strategy": "linear|fragmented|memory_based|cyclical",
  "presence_strategy": "always_visible|fragmented|memory_only|absent",
  "emotional_progression_strategy": "building|releasing|oscillating|still",
  "repetition_strategy": "visual_variation|emotional_shift|reinforce|none",
  "motion_philosophy": "still_dominant|dynamic_dominant|mixed",
  "expression_channels": ["ordered list from most to least important — choose from: character, environment, objects, time, light, performance"],
  "continuity_rules": "strict|associative|symbolic",
  "scene_interpretation_rules": "one rule sentence for how scenes connect or flow (strategy only, no visuals)",
  "variation_allowance": "strict|guided|open",
  "director_note": "one sentence capturing the overall visual storytelling approach"
}

Field guidance:
  storytelling_mode:
    observational  — camera watches, does not intrude; naturalistic
    expressive     — camera and environment reflect inner state
    symbolic       — meaning carried by objects, nature, abstraction
    performative   — artist performs directly; energy is the message

  perspective:
    first_person       — story experienced through the speaker's eyes
    third_person_close — story follows the speaker closely but from outside
    observer           — story watched from a neutral external viewpoint
    omniscient         — story moves freely between all viewpoints

  timeline_strategy:
    linear        — story moves forward in time
    fragmented    — time cuts between moments non-linearly
    memory_based  — past and present intercut
    cyclical      — returns to the same moment or image

  presence_strategy:
    always_visible — character on screen throughout
    fragmented     — character appears and disappears; present in parts
    memory_only    — character seen only in flashback or imagination
    absent         — no character; world carries the story

  repetition_strategy (for songs with chorus/hook):
    visual_variation  — same lyric, different visual treatment each time
    emotional_shift   — same lyric, rising emotional intensity each time
    reinforce         — same visual used as anchor motif
    none              — no special handling (not a song or no repetition)

  continuity_rules:
    strict       — scenes must connect logically and chronologically
    associative  — scenes connect by mood, feeling, or theme
    symbolic     — scenes connect through recurring symbols or motifs

  variation_allowance:
    strict  — downstream must follow strategy exactly; minimal deviation
    guided  — downstream may interpret within defined emotional bounds
    open    — downstream has creative freedom within the stated mode

  scene_interpretation_rules:
    Write ONE concise rule about HOW scenes relate to each other.
    Strategy only — no visuals, no camera, no locations.
    Examples:
      "Each scene is a memory fragment drifting further from the present."
      "Scenes follow the speaker's internal state, not external events."
      "Every chorus returns the viewer to the same emotional anchor."

INPUTS YOU RECEIVE (and how to use them):
  speaker + addressee
    — Use relationship_to_addressee and addressee.presence to decide
      presence_strategy and perspective. Absent addressee → lean toward
      memory_only or fragmented presence.
  world.timeline_nature
    — real_time → linear timeline_strategy is most natural
    — memory    → memory_based or fragmented
    — cyclical  → cyclical
    — ambiguous → fragmented or symbolic, your call
  structure.section_counts + structure.repetition_map
    — If high chorus count or repetition_map is non-empty:
      repetition_strategy matters significantly.
    — Flat structure (no chorus) → "none" for repetition_strategy.
  structure.unique_speakers
    — Multiple speakers → presence_strategy and perspective become critical
      to narrative coherence.
  rhythm.bpm + rhythm.total_duration_s
    — High BPM + short duration → motion_philosophy=dynamic_dominant.
    — Low BPM + long duration  → still_dominant.
    — Mixed energy             → mixed.
    — Missing → decide from emotional arc alone.
  symbolic_density + abstraction_level
    — Higher values → favor symbolic storytelling_mode and symbolic/
      associative continuity_rules.
  project_settings.style_preset
    — If present, use as soft guidance for motion_philosophy and
      variation_allowance. Do not let it override emotional truth."""

    user_msg = f"""\
Content to analyze:
{json.dumps(context_summary, ensure_ascii=False, indent=2)}

Decide the narrative strategy. Return JSON only."""

    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Validation / repair
# ---------------------------------------------------------------------------

_VALID: Dict[str, set] = {
    "storytelling_mode":              {"observational", "expressive", "symbolic", "performative"},
    "perspective":                    {"first_person", "third_person_close", "observer", "omniscient"},
    "timeline_strategy":              {"linear", "fragmented", "memory_based", "cyclical"},
    "presence_strategy":              {"always_visible", "fragmented", "memory_only", "absent"},
    "emotional_progression_strategy": {"building", "releasing", "oscillating", "still"},
    "repetition_strategy":            {"visual_variation", "emotional_shift", "reinforce", "none"},
    "motion_philosophy":              {"still_dominant", "dynamic_dominant", "mixed"},
    "continuity_rules":               {"strict", "associative", "symbolic"},
    "variation_allowance":            {"strict", "guided", "open"},
}

_DEFAULTS: Dict[str, str] = {
    "storytelling_mode":              "expressive",
    "perspective":                    "third_person_close",
    "timeline_strategy":              "linear",
    "presence_strategy":              "always_visible",
    "emotional_progression_strategy": "building",
    "repetition_strategy":            "visual_variation",
    "motion_philosophy":              "mixed",
    "continuity_rules":               "associative",
    "variation_allowance":            "guided",
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

    out["scene_interpretation_rules"] = (
        str(data.get("scene_interpretation_rules") or "").strip()
        or "Scenes connect by emotional state, not chronological order."
    )

    out["director_note"] = (
        str(data.get("director_note") or "").strip()
        or "Let the emotional arc drive visual interpretation."
    )

    return out


def _fallback() -> Dict[str, Any]:
    return {
        **_DEFAULTS,
        "expression_channels":       _DEFAULT_CHANNELS,
        "scene_interpretation_rules": "Scenes connect by emotional state, not chronological order.",
        "director_note":             "Let the emotional arc drive visual interpretation.",
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
  Storytelling mode:       {ni.get('storytelling_mode')}
  Perspective:             {ni.get('perspective')}
  Timeline strategy:       {ni.get('timeline_strategy')}
  Character presence:      {ni.get('presence_strategy')}
  Emotional progression:   {ni.get('emotional_progression_strategy')}
  Repetition strategy:     {ni.get('repetition_strategy')}
  Motion philosophy:       {ni.get('motion_philosophy')}
  Expression channels:     {channels}  (ordered most → least important)
  Continuity rules:        {ni.get('continuity_rules')}
  Scene rules:             {ni.get('scene_interpretation_rules')}
  Variation allowance:     {ni.get('variation_allowance')}
  Director's approach:     {ni.get('director_note')}"""
