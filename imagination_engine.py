"""
imagination_engine.py — Director's Imagination Engine (MetaMind 3.1, Stage 4.5)

ROLE:
    Sits between the Style Profile (Stage 4) and the Storyboard Engine (Stage 5).
    An AI director reads every upstream packet and imagines the FULL VISUAL WORLD
    of the music video — visual concept, motifs, section flow, shot ideas, and
    director notes — before any scene-level storyboard decisions are made.

    The output (imagination_packet) becomes the primary creative directive for:
      • Storyboard Engine v2  — scene intents must serve the imagined concept
      • Creative Brief Engine — visual_concept + motifs injected as anchor
      • Materializer Engine   — visual_concept + motifs inform identity rules

Reads from Project Brain:
    context_packet       — cultural world, emotional arc, speaker, must_preserve
    narrative_packet     — storytelling_mode, timeline, presence, expression channels
    style_packet         — cinematic style, color, texture, vibe label/direction
    emotional_mode_packet — emotional register, pacing biases, tone words
    input_structure      — sections, repetition_map (for section_flow count)

Writes to Project Brain:
    imagination_packet → {
        visual_concept   : str  — one evocative paragraph: the director's overall vision
        motifs           : list[dict]  — 3-5 recurring visual motifs with role + form
        visual_style     : str  — camera + light + texture philosophy in one paragraph
        section_flow     : list[dict]  — one entry per song section: mood + visual_idea
        shot_ideas       : list[str]   — 6-10 specific shot ideas from the director
        director_notes   : str  — one paragraph of notes to the storyboard team
    }
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o"  # Full model — this is the creative heart of the pipeline.


# ---------------------------------------------------------------------------
# Schema returned by the LLM
# ---------------------------------------------------------------------------

_SCHEMA = {
    "visual_concept": (
        "<one rich paragraph: the director's overall visual world — what this video "
        "FEELS like, what world the viewer enters, what visual truth it carries. "
        "Concrete, evocative, specific to this song — NOT generic.>"
    ),
    "motifs": [
        {
            "motif_id": "<slug>",
            "name": "<short name>",
            "visual_form": "<how it appears on screen — texture, light, object, gesture, colour>",
            "emotional_role": "<what this motif carries emotionally across the video>",
            "recurrence": "<when/how often it appears: 'every chorus', 'opening and closing', etc.>"
        }
    ],
    "visual_style": (
        "<one paragraph: the camera language, lighting philosophy, texture register, "
        "and colour temperature the director mandates. This is not generic — it is "
        "derived from THIS song's emotional world and the chosen style profile.>"
    ),
    "section_flow": [
        {
            "section_label": "<intro|verse1|chorus|...|outro>",
            "mood": "<the emotional register of this section — one phrase>",
            "visual_idea": "<what the camera shows or feels in this section — specific, not generic>"
        }
    ],
    "shot_ideas": [
        "<specific, evocative shot idea — e.g. 'a close-up of hands releasing a handful of grain into still water'>",
        "<another specific shot idea>",
        "<and so on — 6 to 10 total>"
    ],
    "director_notes": (
        "<one paragraph addressed to the storyboard team: the tone to protect, "
        "the visual traps to avoid, the one thing that must never be lost.>"
    )
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return """\
You are the Director's Imagination Engine for Qaivid — an AI creative director \
imagining a music video in full before the storyboard is built.

YOUR ROLE:
You receive everything the pipeline knows about the song — its cultural world, \
emotional arc, narrative strategy, and chosen visual style — and you imagine the \
COMPLETE VISUAL WORLD of this video. Your output becomes the primary creative \
directive for every downstream stage.

WHAT YOU MUST PRODUCE:
1. visual_concept — the director's overall vision in one rich, evocative paragraph.
   This is the "what is this video about visually" — NOT a scene list, NOT a shot list.
   It names the emotional world, the visual truth, the feeling the viewer is left with.

2. motifs — 3 to 5 recurring visual elements that anchor the video.
   Each motif has a concrete visual form AND an emotional role.
   Motifs must be specific to THIS song's cultural and emotional world — never generic.

3. visual_style — one paragraph: camera language, lighting philosophy, texture register,
   colour temperature. Specific to this song. Derived from the style packet.

4. section_flow — one entry per lyric section (use section labels from the input).
   For each section: what is its mood, and what does the camera show or feel?
   Be specific. One section = one visual idea. Not "emotional" — VISUAL.

5. shot_ideas — 6 to 10 specific, concrete shot ideas.
   These are not instructions — they are creative sparks for the storyboard team.
   Examples: "a close-up of hands releasing grain into still water"
             "the speaker's shadow stretched long across an empty road at golden hour"
             "two chairs at a table — one empty, tea cooling in the cup"

6. director_notes — one paragraph to the storyboard team.
   What tone must be protected. What visual trap to avoid.
   The one thing that must NEVER be lost.

RULES:
- Every field must be specific to THIS song — no generic descriptions.
- Do NOT name exact actors, locations, or props not derivable from the cultural world.
- Do NOT give camera directions (no "zoom in", no "pan left").
- Do NOT write scene-level instructions — that is the storyboard's job.
- DO write with the authority of a director who has heard this song a hundred times \
  and knows exactly what it should feel like on screen.

Respond ONLY with valid JSON matching the schema provided. No prose outside the JSON."""


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

def _build_user_prompt(
    context_packet: Dict[str, Any],
    narrative_packet: Dict[str, Any],
    style_packet: Dict[str, Any],
    emotional_mode_packet: Dict[str, Any],
    input_structure: Dict[str, Any],
) -> str:
    sections: list[str] = []

    # ── Context ──────────────────────────────────────────────────────────────
    sections.append("=== CONTEXT PACKET (Stage 2 — locked meaning, world, speaker) ===")
    location_dna = context_packet.get("location_dna") or ""
    language = context_packet.get("language") or ""
    core_theme = context_packet.get("core_theme") or ""
    dramatic_premise = context_packet.get("dramatic_premise") or ""
    narrative_spine = context_packet.get("narrative_spine") or ""
    arc = context_packet.get("emotional_arc") or {}
    speaker = context_packet.get("speaker") or {}
    addressee = context_packet.get("addressee") or {}
    must_preserve = context_packet.get("must_preserve") or []
    motif_map = context_packet.get("motif_map") or {}
    world_assumptions = context_packet.get("world_assumptions") or {}

    if language:
        sections.append(f"Song language: {language}")
    if location_dna:
        sections.append(f"Cultural world (Location DNA): {location_dna}")
    if core_theme:
        sections.append(f"Core theme: {core_theme}")
    if dramatic_premise:
        sections.append(f"Dramatic premise: {dramatic_premise}")
    if narrative_spine:
        sections.append(f"Narrative spine: {narrative_spine}")
    if arc:
        arc_str = " → ".join(
            str(v) for k, v in arc.items()
            if v and k in ("opening", "development", "climax", "resolution")
        )
        if arc_str:
            sections.append(f"Emotional arc: {arc_str}")
    if world_assumptions:
        sections.append(f"World: {json.dumps(world_assumptions, ensure_ascii=False)}")
    if speaker:
        sections.append(f"Speaker: {json.dumps(speaker, ensure_ascii=False)}")
    if addressee:
        sections.append(f"Addressee: {json.dumps(addressee, ensure_ascii=False)}")
    if must_preserve:
        sections.append(f"Must preserve: {must_preserve}")
    if motif_map:
        sections.append("Source motifs:")
        for mname, mpayload in list(motif_map.items())[:6]:
            sections.append(f"  {mname}: {mpayload}")

    # Pull cultural line meanings
    raw_line_meanings = context_packet.get("line_meanings") or []
    cultural_highlights: list[str] = []
    _seen: set[str] = set()
    for _lm in raw_line_meanings:
        if not isinstance(_lm, dict):
            continue
        for _key in ("cultural_meaning", "implied_meaning"):
            _val = str(_lm.get(_key) or "").strip()
            if _val and _val.lower() not in _seen:
                _seen.add(_val.lower())
                cultural_highlights.append(_val)
        if len(cultural_highlights) >= 10:
            break
    if cultural_highlights:
        sections.append(
            "Cultural meanings from lyrics:\n"
            + "\n".join(f"  - {h}" for h in cultural_highlights)
        )

    # ── Narrative ─────────────────────────────────────────────────────────────
    sections.append("=== NARRATIVE PACKET (Stage 3 — HOW this story is told) ===")
    if narrative_packet:
        try:
            from narrative_engine import format_for_prompt
            np_block = format_for_prompt(narrative_packet)
            if np_block:
                sections.append(np_block)
            else:
                sections.append(json.dumps(narrative_packet, ensure_ascii=False, indent=2)[:1500])
        except Exception:
            sections.append(json.dumps(narrative_packet, ensure_ascii=False, indent=2)[:1500])
    else:
        sections.append("(no narrative packet)")

    # ── Style ──────────────────────────────────────────────────────────────────
    sections.append("=== STYLE PACKET (Stage 4 — visual language chosen by user) ===")
    preset = style_packet.get("preset") or style_packet.get("cinematic_style") or ""
    vibe_label = style_packet.get("vibe_label") or ""
    vibe_storyboard = style_packet.get("vibe_storyboard_direction") or ""
    vibe_brief = style_packet.get("vibe_brief_direction") or ""
    cinematic = style_packet.get("cinematic") or {}
    color_psychology = cinematic.get("color_psychology") or ""
    texture_profile = cinematic.get("texture_profile") or ""
    realism_level = cinematic.get("realism_level") or ""
    storyboard_modifiers = style_packet.get("storyboard_modifiers") or {}

    if preset:
        sections.append(f"Cinematic style preset: {preset}")
    if vibe_label:
        sections.append(f"Vibe: {vibe_label}")
    if vibe_storyboard:
        sections.append(f"Vibe storyboard direction: {vibe_storyboard}")
    if vibe_brief:
        sections.append(f"Vibe brief direction: {vibe_brief}")
    if color_psychology:
        sections.append(f"Color psychology: {color_psychology}")
    if texture_profile:
        sections.append(f"Texture profile: {texture_profile}")
    if realism_level:
        sections.append(f"Realism level: {realism_level}")
    if storyboard_modifiers:
        sections.append(f"Storyboard modifiers: {json.dumps(storyboard_modifiers, ensure_ascii=False)}")

    # ── Emotional mode ─────────────────────────────────────────────────────────
    if emotional_mode_packet:
        sections.append("=== EMOTIONAL MODE PACKET (Stage 2b — register and pacing) ===")
        mode_id = emotional_mode_packet.get("mode_id") or ""
        mode_label = emotional_mode_packet.get("mode_label") or ""
        pacing = (emotional_mode_packet.get("pacing_profile") or {})
        tone_words = emotional_mode_packet.get("tone_words") or []
        shot_biases = emotional_mode_packet.get("shot_biases") or {}
        if mode_label or mode_id:
            sections.append(f"Emotional mode: {mode_label or mode_id}")
        if tone_words:
            sections.append(f"Tone words: {', '.join(tone_words[:6])}")
        if pacing:
            sections.append(f"Pacing: {json.dumps(pacing, ensure_ascii=False)}")
        if shot_biases:
            sections.append(f"Shot biases: {json.dumps(shot_biases, ensure_ascii=False)}")

    # ── Input structure: section list ──────────────────────────────────────────
    sections.append("=== SONG STRUCTURE (Stage 1 — section labels for section_flow) ===")
    raw_sections = input_structure.get("sections") or []
    if raw_sections:
        labels = [str(s.get("label") or "unknown") for s in raw_sections if isinstance(s, dict)]
        sections.append("Section labels (in order): " + ", ".join(labels))
        sections.append(
            f"Total sections: {len(labels)}. "
            "Your section_flow must have one entry per unique section label in this order."
        )
    else:
        sections.append("(no section labels — use: intro, verse1, chorus, verse2, outro)")

    # ── Schema ─────────────────────────────────────────────────────────────────
    sections.append("=== OUTPUT SCHEMA ===")
    sections.append(
        "Fill in the following JSON exactly. All fields are required.\n"
        "motifs: exactly 3 to 5 entries.\n"
        "shot_ideas: exactly 6 to 10 strings.\n"
        "section_flow: one entry per section listed above.\n\n"
        + json.dumps(_SCHEMA, indent=2, ensure_ascii=False)
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=_MODEL,
        temperature=0.72,
        max_tokens=3500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Validation / repair
# ---------------------------------------------------------------------------

def _validate_and_fix(data: Any) -> dict:
    if not isinstance(data, dict):
        data = {}

    # visual_concept
    if not isinstance(data.get("visual_concept"), str) or not data["visual_concept"].strip():
        data["visual_concept"] = "A cinematic exploration of the song's emotional world."

    # motifs: must be a list of dicts with required keys
    if not isinstance(data.get("motifs"), list):
        data["motifs"] = []
    cleaned_motifs = []
    for m in data["motifs"][:5]:
        if not isinstance(m, dict):
            continue
        cleaned_motifs.append({
            "motif_id":      str(m.get("motif_id") or m.get("name") or "motif").lower().replace(" ", "_")[:40],
            "name":          str(m.get("name") or m.get("motif_id") or "Visual motif"),
            "visual_form":   str(m.get("visual_form") or ""),
            "emotional_role": str(m.get("emotional_role") or ""),
            "recurrence":    str(m.get("recurrence") or "recurring"),
        })
    data["motifs"] = cleaned_motifs

    # visual_style
    if not isinstance(data.get("visual_style"), str) or not data["visual_style"].strip():
        data["visual_style"] = "A naturalistic cinematic style grounded in the song's emotional world."

    # section_flow
    if not isinstance(data.get("section_flow"), list):
        data["section_flow"] = []
    cleaned_sf = []
    for sf in data["section_flow"][:20]:
        if not isinstance(sf, dict):
            continue
        cleaned_sf.append({
            "section_label": str(sf.get("section_label") or sf.get("section") or "section"),
            "mood":          str(sf.get("mood") or ""),
            "visual_idea":   str(sf.get("visual_idea") or sf.get("visual") or ""),
        })
    data["section_flow"] = cleaned_sf

    # shot_ideas
    if not isinstance(data.get("shot_ideas"), list):
        data["shot_ideas"] = []
    cleaned_shots = [str(s) for s in data["shot_ideas"][:10] if s and str(s).strip()]
    data["shot_ideas"] = cleaned_shots

    # director_notes
    if not isinstance(data.get("director_notes"), str) or not data["director_notes"].strip():
        data["director_notes"] = "Protect the emotional honesty of the song in every visual choice."

    return data


def _fallback() -> dict:
    return {
        "visual_concept": "A cinematic exploration of the song's emotional world and cultural roots.",
        "motifs": [
            {
                "motif_id": "light_and_shadow",
                "name": "Light and Shadow",
                "visual_form": "The interplay of warm light and deep shadow across faces and spaces.",
                "emotional_role": "Carries the tension between presence and absence.",
                "recurrence": "Throughout — especially in chorus moments.",
            },
            {
                "motif_id": "threshold",
                "name": "Threshold",
                "visual_form": "Doorways, windows, and open roads — places of decision and crossing.",
                "emotional_role": "Marks moments of change, choice, and longing.",
                "recurrence": "Introduced in the verse; prominent in bridge and outro.",
            },
            {
                "motif_id": "hands_and_objects",
                "name": "Hands and Objects",
                "visual_form": "Close-ups of hands touching meaningful objects — worn things, gifted things, absent things.",
                "emotional_role": "Grounds emotion in the physical; makes inner life visible.",
                "recurrence": "Appears in quiet scenes; evolves toward release.",
            },
            {
                "motif_id": "stillness_in_motion",
                "name": "Stillness in Motion",
                "visual_form": "A static figure inside a moving world — or a moving figure in a frozen one.",
                "emotional_role": "Isolates the subject; heightens their interiority.",
                "recurrence": "Peak moments; used sparingly for maximum impact.",
            },
        ],
        "visual_style": "Naturalistic and intimate — close to the subject, honest about the world.",
        "section_flow": [
            {
                "section_label": "intro",
                "mood": "quiet and grounding",
                "visual_idea": "Establish the world through stillness — a person, a place, a light source.",
            },
            {
                "section_label": "verse",
                "mood": "reflective and close",
                "visual_idea": "Intimate coverage — faces, hands, small gestures that carry big feelings.",
            },
            {
                "section_label": "chorus",
                "mood": "expansive and emotional",
                "visual_idea": "Open out to wider frames; motion increases; the world around the subject feels larger.",
            },
            {
                "section_label": "outro",
                "mood": "resolving and honest",
                "visual_idea": "Return to stillness — the same world we opened on, changed by what has passed.",
            },
        ],
        "shot_ideas": [
            "A figure standing still in a doorway, light falling from one side.",
            "Hands touching an object that belongs to another person.",
            "A window with the outside world blurred, the inside in focus.",
            "A long road stretching to the horizon at golden hour.",
            "A close-up of an eye, a tear forming but not falling.",
            "An empty chair at a table set for two.",
        ],
        "director_notes": (
            "Protect the emotional honesty of the song. Every frame should earn its place "
            "by serving the song's emotional truth — never for visual spectacle alone."
        ),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_imagination_packet(
    context_packet: Dict[str, Any],
    narrative_packet: Dict[str, Any],
    style_packet: Dict[str, Any],
    emotional_mode_packet: Optional[Dict[str, Any]] = None,
    input_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stage 4.5 — Director's Imagination Engine.

    Returns the imagination_packet dict (always valid, falls back gracefully).
    """
    try:
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            context_packet=context_packet or {},
            narrative_packet=narrative_packet or {},
            style_packet=style_packet or {},
            emotional_mode_packet=emotional_mode_packet or {},
            input_structure=input_structure or {},
        )
        raw = await _call_llm(system_prompt, user_prompt)
        packet = _validate_and_fix(raw)
        logger.info(
            "Imagination Engine: produced %d motifs, %d shot ideas, %d section_flow entries.",
            len(packet.get("motifs") or []),
            len(packet.get("shot_ideas") or []),
            len(packet.get("section_flow") or []),
        )
        return packet
    except Exception as exc:
        logger.warning("Imagination Engine failed (%s) — using fallback.", exc)
        return _fallback()


# ---------------------------------------------------------------------------
# Human-readable formatter for downstream injections
# ---------------------------------------------------------------------------

def format_imagination_for_prompt(packet: Dict[str, Any]) -> str:
    """Return a compact block for injection into storyboard / brief system prompts."""
    if not packet:
        return ""
    lines = ["DIRECTOR'S IMAGINATION (Stage 4.5 — primary creative directive):"]

    vc = packet.get("visual_concept") or ""
    if vc:
        lines.append(f"  Visual Concept:\n    {vc}")

    vs = packet.get("visual_style") or ""
    if vs:
        lines.append(f"  Visual Style:\n    {vs}")

    motifs = packet.get("motifs") or []
    if motifs:
        lines.append("  Motifs:")
        for m in motifs[:5]:
            if isinstance(m, dict):
                lines.append(
                    f"    • {m.get('name', '')} [{m.get('recurrence', '')}]: "
                    f"{m.get('visual_form', '')} — {m.get('emotional_role', '')}"
                )

    section_flow = packet.get("section_flow") or []
    if section_flow:
        lines.append("  Section Flow (per-section visual direction — apply to matching storyboard sections):")
        for sf in section_flow:
            if not isinstance(sf, dict):
                continue
            label = sf.get("section_label", "")
            mood = sf.get("mood", "")
            idea = sf.get("visual_idea", "")
            lines.append(f"    [{label}] mood={mood} → {idea}")

    shot_ideas = packet.get("shot_ideas") or []
    if shot_ideas:
        lines.append("  Director's Shot Ideas (sparks — not instructions):")
        for si in shot_ideas[:8]:
            lines.append(f"    • {si}")

    dn = packet.get("director_notes") or ""
    if dn:
        lines.append(f"  Director's Notes:\n    {dn}")

    return "\n".join(lines)
