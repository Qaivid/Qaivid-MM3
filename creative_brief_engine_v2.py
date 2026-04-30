"""Creative Brief Engine v2 — controlled selection + variation anchoring.

ROLE (per master spec):
    The FIRST COMMITMENT LAYER.

    Convert storyboard possibilities (multiple valid_realizations per scene)
    into ONE executable direction per scene, while preserving flexibility,
    continuity, and controlled variation.

    Storyboard      = possibilities
    Creative Brief  = controlled selection + variation anchoring

INPUTS (read by the worker from Project Brain):
    storyboard_packet   — ordered scenes with valid_realizations, motion,
                          presence, motif_usage, continuity_hooks
    narrative_packet    — storytelling/perspective/timeline/presence/motion
                          /expression/repetition/continuity/scene rules
    context_packet      — core_theme, emotional_world, must_preserve,
                          creative_freedom
    style_packet        — cinematic_style, lighting/color/texture (HIGH-LEVEL)
    input_structure     — section type, repetition_map (light use)
    project_settings    — duration, platform, user prefs (optional)

OUTPUT:
    creative_brief_packet = {
        "schema_version": 2,
        "scenes":         [scene_brief, ...],
        "used_fallback":  bool,
    }

    where each scene_brief is:
    {
      "scene_id": "",
      "source_section": "",
      "narrative_phase": "",
      "scene_purpose": "",
      "chosen_direction": "",            # one of the valid_realizations
      "selection_basis": "",             # WHY this realization was picked
      "variation_anchor": "",            # what stays consistent across runs
      "subject_focus": "",               # character | environment | object
      "character_presence": "",          # continuous | intermittent | minimal
      "character_identity_hint": "",     # downstream consistency, NOT appearance
      "environment_type": "",            # general type, NOT exact location
      "key_elements": [str, ...],        # max 3–5, general, non-specific
      "emotional_state": "",             # micro-state
      "emotional_intensity": "",         # low | medium | high
      "lighting_condition": "",          # general, NOT technical
      "movement_type": "",               # static | slow | dynamic
      "timeline_mode": "",               # present | memory | mixed
      "motion_density": "",              # low | medium | high
      "repetition_handling": "",         # how a repeated section evolves
      "motif_usage": [str, ...],
      "continuity_hooks": {
        "character": "",
        "motifs":    [str, ...]
      }
    }

This engine MUST NOT:
    - define exact character appearance
    - define exact props (only general elements)
    - define camera shots / lens / angle
    - define shot count
    - generate executable visual prompts
    - over-specify details
    - collapse downstream flexibility

Async because it does an OpenAI Chat call. Returns a deterministic fallback
(picks the first valid_realization per scene with default execution fields)
if the LLM call fails so the pipeline never hard-stops here.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SUBJECT_FOCUS    = {"character", "environment", "object", "mixed"}
_PRESENCE         = {"continuous", "intermittent", "minimal"}
_INTENSITIES      = {"low", "medium", "high"}
_MOVEMENT         = {"static", "slow", "dynamic"}
_TIMELINE_MODES   = {"present", "memory", "mixed", "future", "dream"}
_MOTION_DENSITY   = {"low", "medium", "high"}


# ────────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────────
def _system_prompt() -> str:
    return (
        "You are the CREATIVE BRIEF ENGINE for Qaivid — the FIRST commitment "
        "layer of a music-video pipeline.\n\n"
        "ROLE: take the storyboard's already-defined scene and shot structure "
        "(valid_realizations) and make each shot richer and more executable "
        "by enriching it with narrative, style, and emotional context. "
        "You do NOT change what the shots are — you enhance them.\n\n"
        "RULES:\n"
        "1. ENRICH ALL valid_realizations as shot_directions: for every "
        "realization in the scene's valid_realizations list, produce one "
        "enhanced shot_direction entry that takes the storyboard's visual "
        "concept and adds execution detail — emotional state, lighting quality, "
        "character presence, atmosphere — drawn from the narrative, context, "
        "and style packets. Preserve the storyboard's visual concept exactly; "
        "only add execution depth. Order must match the storyboard order. "
        "chosen_direction = shot_directions[0] (the first enriched shot).\n"
        "2. For EVERY scene, also generate the per-scene execution intent "
        "fields (scene_purpose, subject_focus, environment_type, character_"
        "presence/identity_hint, key_elements, emotional_state/intensity, "
        "lighting_condition, movement_type, timeline_mode, motion_density, "
        "repetition_handling, motif_usage, continuity_hooks). "
        "Each field is GENERAL and NON-SPECIFIC — enough direction to execute, "
        "not a locked visual. These apply to the scene as a whole.\n"
        "3. APPLY narrative logic: presence_strategy → character_presence; "
        "motion_philosophy → movement_type/motion_density; timeline_strategy "
        "→ timeline_mode; expression_channels → which channels carry the "
        "emotion; repetition_strategy → repetition_handling on repeated "
        "sections.\n"
        "4. APPLY style guidance: tone, texture, color → lighting_condition + "
        "general atmosphere only. Style does NOT override narrative decisions.\n"
        "5. MAINTAIN CONTINUITY across scenes: consistent character identity "
        "(via hints, NEVER appearance), consistent world (culture / "
        "geography), motif continuity, smooth emotional progression.\n"
        "6. CONTROL VARIATION: every scene needs a variation_anchor — the "
        "thing that must stay the same even if the LLM picks differently on "
        "a re-run (e.g. 'protagonist remains the same person', 'the river is "
        "always present as a motif').\n\n"
        "7. If a DIRECTOR'S IMAGINATION block is present in the user message:\n"
        "   - It is the PRIMARY creative directive for the whole video.\n"
        "   - The visual_concept sets the overarching tone you must serve.\n"
        "   - Listed motifs MUST appear in motif_usage across scenes.\n"
        "   - Director's notes set constraints that override defaults.\n\n"
        "FORBIDDEN:\n"
        "- Do NOT define exact character appearance.\n"
        "- Do NOT define exact props — only general key_elements.\n"
        "- Do NOT define camera shots, lens, angles, or shot counts.\n"
        "- Do NOT generate executable prompts.\n"
        "- Do NOT over-specify details that downstream stages should choose.\n"
        "- Do NOT collapse flexibility.\n\n"
        "Return strict JSON. No prose outside the JSON object."
    )


def _format_scenes(scenes: List[Dict[str, Any]]) -> str:
    if not scenes:
        return "  (no storyboard scenes available — derive a minimal brief from context.)"
    lines: List[str] = []
    for s in scenes:
        sid       = s.get("scene_id") or "?"
        section   = s.get("source_section") or ""
        phase     = s.get("narrative_phase") or ""
        purpose   = s.get("purpose") or ""
        intensity = s.get("emotional_intensity") or ""
        presence  = s.get("presence_hint") or ""
        motion    = s.get("motion_density") or ""
        timeline  = s.get("timeline_position") or ""
        motifs    = s.get("motif_usage") or []
        hooks     = s.get("continuity_hooks") or {}
        realizations = s.get("valid_realizations") or []
        lines.append(
            f"  - scene_id: {sid} (section={section}, phase={phase}, "
            f"intensity={intensity}, presence={presence}, "
            f"motion={motion}, timeline={timeline})\n"
            f"    purpose: {purpose}\n"
            f"    motifs: {', '.join(motifs) if motifs else '—'}\n"
            f"    continuity: subject={hooks.get('subject') or '—'}, "
            f"motifs_carry={', '.join(hooks.get('motifs') or []) or '—'}\n"
            f"    valid_realizations:\n"
            + "\n".join(f"      [{i}] {r}" for i, r in enumerate(realizations))
        )
    return "\n".join(lines)


def _format_narrative(narrative_packet: Dict[str, Any]) -> str:
    if not narrative_packet:
        return "  (no narrative packet available)"
    try:
        from narrative_engine import format_for_prompt
        block = format_for_prompt(narrative_packet)
        if block:
            return block
    except Exception:
        logger.exception("BriefV2: failed to format narrative_packet")
    return json.dumps(narrative_packet, indent=2, ensure_ascii=False)[:2000]


def _format_context(context_packet: Dict[str, Any]) -> str:
    payload = {
        "core_theme":         context_packet.get("core_theme"),
        "emotional_world":    context_packet.get("emotional_world"),
        "emotional_arc":      context_packet.get("emotional_arc"),
        "speaker":            context_packet.get("speaker"),
        "location_dna":       context_packet.get("location_dna"),
        "era":                context_packet.get("era"),
        "world_assumptions":  context_packet.get("world_assumptions"),
        "must_preserve":      context_packet.get("must_preserve"),
        "creative_freedom":   context_packet.get("creative_freedom"),
        "cultural_grounding": context_packet.get("cultural_grounding"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_style(style_profile: Dict[str, Any]) -> str:
    cin  = (style_profile or {}).get("cinematic")  or {}
    prod = (style_profile or {}).get("production") or {}
    payload = {
        "cinematic_style":  cin.get("name") or cin.get("id"),
        "production_style": prod.get("name") or prod.get("id"),
        "preset":           (style_profile or {}).get("preset"),
        "lighting_logic":   (style_profile or {}).get("lighting_logic")
                                or cin.get("lighting"),
        "color_psychology": (style_profile or {}).get("color_psychology")
                                or cin.get("color"),
        "texture_profile":  (style_profile or {}).get("texture_profile")
                                or cin.get("texture"),
        "realism_level":    (style_profile or {}).get("realism_level"),
        "tone_keywords":    (style_profile or {}).get("tone_keywords"),
    }
    # Vibe preset direction — present only when user chose a vibe preset.
    # Gives the brief LLM the cultural/production vocabulary it needs to
    # select scene directions that match the chosen vibe (e.g. "Premium
    # Punjabi filmmaking with golden-hour mustard fields" vs "Bollywood
    # glamour with jewel tones and hero entrances").
    sp = style_profile or {}
    if sp.get("vibe_label"):
        payload["vibe_preset"] = sp["vibe_label"]
    if sp.get("vibe_brief_direction"):
        payload["vibe_direction"] = sp["vibe_brief_direction"]
    if sp.get("vibe_avoid"):
        avoid = sp["vibe_avoid"]
        payload["vibe_avoid"] = avoid if isinstance(avoid, list) else [avoid]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_input_structure_light(input_structure: Dict[str, Any]) -> str:
    sections = input_structure.get("sections") or []
    rep_map  = input_structure.get("repetition_map") or {}
    out = []
    if sections:
        out.append("  sections: " + ", ".join(
            f"{s.get('id')}:{s.get('type')}" for s in sections
        )[:600])
    if rep_map:
        out.append("  repetition_map: "
                   + json.dumps(rep_map, ensure_ascii=False)[:300])
    return "\n".join(out) or "  (no input structure)"


def _user_prompt(
    scenes:             List[Dict[str, Any]],
    narrative_packet:   Dict[str, Any],
    context_packet:     Dict[str, Any],
    style_profile:      Dict[str, Any],
    input_structure:    Dict[str, Any],
    project_settings:   Dict[str, Any],
    imagination_packet: Optional[Dict[str, Any]] = None,
) -> str:
    imagination_block = ""
    if imagination_packet:
        try:
            from imagination_engine import format_imagination_for_prompt
            imagination_block = format_imagination_for_prompt(imagination_packet) or ""
        except Exception:
            logger.exception("BriefV2: failed to format imagination_packet")

    return (
        ("\n\n" + imagination_block if imagination_block else "")
        + "\n\nSTORYBOARD SCENES (Stage 5 — possibilities, with valid_realizations):\n"
        + _format_scenes(scenes)
        + "\n\nNARRATIVE PACKET (Stage 3 — story logic to honor):\n"
        + _format_narrative(narrative_packet)
        + "\n\nCONTEXT PACKET (Stage 2 — locked meaning, world, speaker):\n"
        + _format_context(context_packet)
        + "\n\nSTYLE PACKET (Stage 4 — high-level only, do not override narrative):\n"
        + _format_style(style_profile)
        + "\n\nINPUT STRUCTURE (Stage 1 — light use):\n"
        + _format_input_structure_light(input_structure)
        + "\n\nPROJECT SETTINGS:\n"
        + json.dumps(project_settings or {}, indent=2, ensure_ascii=False)
        + "\n\nReturn JSON of the form:\n"
        + "{\n"
          '  "scenes": [\n'
          "    {\n"
          '      "scene_id":               "s1",\n'
          '      "source_section":         "intro|verse1|chorus|...",\n'
          '      "narrative_phase":        "intro|build|peak|breakdown|resolution",\n'
          '      "scene_purpose":          "the emotional purpose of this scene",\n'
          '      "shot_directions":        ["one enriched entry per valid_realization, same order — storyboard concept preserved, execution detail added"],\n'
          '      "chosen_direction":       "= shot_directions[0] — the first enriched shot direction",\n'
          '      "selection_basis":        "what execution context was added and why it serves the scene",\n'
          '      "variation_anchor":       "what stays consistent across re-runs (subject identity, motif, etc.)",\n'
          '      "subject_focus":          "character|environment|object|mixed",\n'
          '      "character_presence":     "continuous|intermittent|minimal",\n'
          '      "character_identity_hint":"downstream consistency hint, NOT appearance",\n'
          '      "environment_type":       "general environment type, NOT a named location",\n'
          '      "key_elements":           ["3-5 general elements", "..."],\n'
          '      "emotional_state":        "scene-specific micro-state",\n'
          '      "emotional_intensity":    "low|medium|high",\n'
          '      "lighting_condition":     "general lighting, e.g. \\"low warm interior\\", NOT technical",\n'
          '      "movement_type":          "static|slow|dynamic",\n'
          '      "timeline_mode":          "present|memory|mixed|future|dream",\n'
          '      "motion_density":         "low|medium|high",\n'
          '      "repetition_handling":    "for repeated sections: how this evolves vs prior occurrences",\n'
          '      "motif_usage":            ["motifs used in this scene"],\n'
          '      "continuity_hooks": {\n'
          '        "character": "subject continuity note",\n'
          '        "motifs":    ["motifs to carry forward"]\n'
          "      }\n"
          "    }\n"
          "  ]\n"
          "}\n\n"
          "Produce ONE entry per scene_id from the storyboard, in the same order. "
          "shot_directions must have the same number of entries as valid_realizations "
          "for that scene, in the same order — each entry enriches the corresponding "
          "storyboard concept without replacing it. "
          "All values must remain general and non-locking — never name an exact "
          "person, an exact place, an exact prop, or a camera move."
    )


# ────────────────────────────────────────────────────────────────────────
# Coercion & fallback
# ────────────────────────────────────────────────────────────────────────
def _enum(value: Any, allowed: set, default: str) -> str:
    v = str(value or "").strip().lower().replace(" ", "_")
    return v if v in allowed else default


def _coerce_scene_brief(raw: Any, src_scene: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        raw = {}
    valid_realizations = src_scene.get("valid_realizations") or []

    # shot_directions — one enriched entry per valid_realization, same order.
    # Fall back to raw valid_realizations if the LLM omits the field.
    sd_raw = raw.get("shot_directions")
    if isinstance(sd_raw, list) and sd_raw:
        shot_directions = [str(d).strip()[:500] for d in sd_raw if str(d).strip()]
    else:
        shot_directions = [str(r).strip()[:500] for r in valid_realizations if str(r).strip()]
    # Always have at least one entry
    if not shot_directions and valid_realizations:
        shot_directions = [str(valid_realizations[0])[:500]]

    chosen = str(raw.get("chosen_direction") or "").strip()
    if not chosen:
        chosen = shot_directions[0] if shot_directions else ""

    hooks_in = raw.get("continuity_hooks") or {}
    if not isinstance(hooks_in, dict):
        hooks_in = {}
    motifs_in = hooks_in.get("motifs") or []
    if not isinstance(motifs_in, list):
        motifs_in = []

    motif_usage_in = raw.get("motif_usage") or []
    if not isinstance(motif_usage_in, list):
        motif_usage_in = []

    key_elements_in = raw.get("key_elements") or []
    if not isinstance(key_elements_in, list):
        key_elements_in = []

    return {
        "scene_id":                str(raw.get("scene_id")
                                        or src_scene.get("scene_id")
                                        or f"s{idx + 1}").strip()[:24],
        "source_section":          str(raw.get("source_section")
                                        or src_scene.get("source_section")
                                        or "").strip()[:48],
        "narrative_phase":         str(raw.get("narrative_phase")
                                        or src_scene.get("narrative_phase")
                                        or "build").strip()[:32],
        "scene_purpose":           str(raw.get("scene_purpose")
                                        or src_scene.get("purpose")
                                        or "").strip()[:300],
        "shot_directions":         shot_directions,
        "chosen_direction":        chosen[:400],
        "selection_basis":         str(raw.get("selection_basis") or "").strip()[:300],
        "variation_anchor":        str(raw.get("variation_anchor") or "").strip()[:200],
        "subject_focus":           _enum(raw.get("subject_focus"), _SUBJECT_FOCUS, "character"),
        "character_presence":      _enum(raw.get("character_presence"), _PRESENCE, "continuous"),
        "character_identity_hint": str(raw.get("character_identity_hint") or "").strip()[:200],
        "environment_type":        str(raw.get("environment_type") or "").strip()[:120],
        "key_elements":            [str(k).strip()[:80] for k in key_elements_in if str(k).strip()][:5],
        "emotional_state":         str(raw.get("emotional_state") or "").strip()[:120],
        "emotional_intensity":     _enum(raw.get("emotional_intensity"), _INTENSITIES, "medium"),
        "lighting_condition":      str(raw.get("lighting_condition") or "").strip()[:120],
        "movement_type":           _enum(raw.get("movement_type"), _MOVEMENT, "slow"),
        "timeline_mode":           _enum(raw.get("timeline_mode"), _TIMELINE_MODES, "present"),
        "motion_density":          _enum(raw.get("motion_density"),
                                          _MOTION_DENSITY,
                                          str(src_scene.get("motion_density") or "medium")),
        "repetition_handling":     str(raw.get("repetition_handling") or "").strip()[:200],
        "motif_usage":             [str(m).strip()[:60] for m in motif_usage_in if str(m).strip()][:8],
        "continuity_hooks": {
            "character": str(hooks_in.get("character")
                              or hooks_in.get("subject")
                              or "").strip()[:200],
            "motifs":    [str(m).strip()[:60] for m in motifs_in if str(m).strip()][:8],
        },
    }


def _fallback(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic fallback: pass all valid_realizations as shot_directions
    and fill execution fields from the storyboard hints (no LLM)."""
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes or []):
        realizations = s.get("valid_realizations") or []
        shot_directions = [str(r).strip()[:500] for r in realizations if str(r).strip()]
        chosen = shot_directions[0] if shot_directions else "Express the scene's emotional intent."
        hooks = s.get("continuity_hooks") or {}
        out.append({
            "scene_id":                str(s.get("scene_id") or f"s{i + 1}"),
            "source_section":          str(s.get("source_section") or ""),
            "narrative_phase":         str(s.get("narrative_phase") or "build"),
            "scene_purpose":           str(s.get("purpose") or ""),
            "shot_directions":         shot_directions,
            "chosen_direction":        chosen[:400],
            "selection_basis":         "default fallback selection (first valid realization)",
            "variation_anchor":        "subject identity remains consistent",
            "subject_focus":           "character",
            "character_presence":      "continuous",
            "character_identity_hint": str(hooks.get("subject") or "same speaker throughout")[:200],
            "environment_type":        "matches the song's locked world",
            "key_elements":            [],
            "emotional_state":         "",
            "emotional_intensity":     str(s.get("emotional_intensity") or "medium"),
            "lighting_condition":      "natural to the chosen environment",
            "movement_type":           "slow",
            "timeline_mode":           str(s.get("timeline_position") or "present"),
            "motion_density":          str(s.get("motion_density") or "medium"),
            "repetition_handling":     "preserve scene if not a repeat; evolve if repeat",
            "motif_usage":             list(s.get("motif_usage") or [])[:8],
            "continuity_hooks": {
                "character": str(hooks.get("subject") or "")[:200],
                "motifs":    list(hooks.get("motifs") or [])[:8],
            },
        })
    return out


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────
async def generate_creative_brief_v2(
    api_key:             str,
    storyboard_scenes:   List[Dict[str, Any]],
    narrative_packet:    Dict[str, Any],
    context_packet:      Dict[str, Any],
    style_profile:       Optional[Dict[str, Any]] = None,
    input_structure:     Optional[Dict[str, Any]] = None,
    project_settings:    Optional[Dict[str, Any]] = None,
    imagination_packet:  Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Generate the v2 Creative Brief — one locked direction per storyboard scene.

    Returns (scene_briefs, used_fallback). Never raises — falls back to a
    deterministic per-scene first-realization pick if the LLM call fails.
    """
    style_profile    = style_profile    or {}
    input_structure  = input_structure  or {}
    project_settings = project_settings or {}

    if not storyboard_scenes:
        logger.warning("BriefV2: no storyboard scenes — using empty fallback")
        return [], True

    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=_MODEL,
            response_format={"type": "json_object"},
            temperature=0.7,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(
                    storyboard_scenes,
                    narrative_packet or {},
                    context_packet   or {},
                    style_profile,
                    input_structure,
                    project_settings,
                    imagination_packet=imagination_packet or {},
                )},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        out_scenes = data.get("scenes") or []
        # Index source scenes by id so coercion can pull defaults from
        # the original storyboard scene when LLM omits a field.
        src_by_id = {str(s.get("scene_id") or ""): s for s in storyboard_scenes}
        coerced: List[Dict[str, Any]] = []
        for i, raw_s in enumerate(out_scenes):
            sid = str((raw_s or {}).get("scene_id") or "")
            src = src_by_id.get(sid) or (
                storyboard_scenes[i] if i < len(storyboard_scenes) else {}
            )
            cb = _coerce_scene_brief(raw_s, src, i)
            if cb:
                coerced.append(cb)
        if not coerced:
            logger.warning("BriefV2: LLM returned no valid scenes; using fallback")
            return _fallback(storyboard_scenes), True
        return coerced, False
    except Exception:
        logger.exception("BriefV2: generation failed; using fallback")
        return _fallback(storyboard_scenes), True
