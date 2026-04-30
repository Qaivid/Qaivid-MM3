"""Creative Brief Engine v3 — scene-level commitment + per-shot enrichment.

ROLE (per master spec, v3 architecture):
    The FIRST COMMITMENT LAYER, now consuming the v3 storyboard packet.

    Storyboard v3 already gives us:
        - story:   the macro-arc (1 LLM call)
        - scenes:  ordered scenes with valid_realizations, motion, presence,
                   motif_usage, continuity_hooks, time_window_start/end
        - shots:   concrete time-anchored shots inside each scene with
                   shot_id, scene_id, start_time, end_time, duration,
                   lyric_text, action_intent, optional actions[] (multishot)

    The v3 brief turns this into ONE executable direction per scene
    (scene-level commitment, like v2) PLUS one enriched_direction per
    actual shot (so downstream stages don't re-invent shot intent).

INPUTS (read by the worker from Project Brain):
    storyboard_packet   — v3 dict with story / scenes / shots
    narrative_packet    — storytelling/perspective/timeline/presence/motion
                          /expression/repetition/continuity rules
    context_packet      — core_theme, emotional_world, must_preserve,
                          creative_freedom
    style_profile       — cinematic_style, lighting/color/texture (HIGH-LEVEL)
    input_structure     — section type, repetition_map (light use)
    project_settings    — duration, platform, user prefs (optional)
    imagination_packet  — optional director's imagination (PRIMARY directive)

OUTPUT:
    creative_brief_packet = {
        "schema_version": 3,
        "scenes":         [scene_brief_v3, ...],
        "used_fallback":  bool,
    }

    where each scene_brief_v3 is:
    {
      "scene_id": "",
      "source_section": "",
      "narrative_phase": "",
      "scene_purpose": "",
      "chosen_direction": "",            # one enriched valid_realization
      "selection_basis": "",             # WHY this realization was picked
      "variation_anchor": "",            # what stays consistent across runs
      "shots": [                         # NEW in v3: per-actual-shot enrichment
        {
          "shot_id": "shot_1",
          "start_time": 0.0,
          "end_time":   7.0,
          "duration":   7,
          "lyric_text": "...",
          "action_intent":     "raw shot intent from storyboard (preserved)",
          "enriched_direction":"executable direction; storyboard intent + "
                               "execution detail (light/atmosphere/emotion)",
        }, ...
      ],
      "shot_directions": ["enriched_direction text", ...],  # mirror, list-of-strings
      "subject_focus": "",
      "character_presence": "",
      "character_identity_hint": "",
      "environment_type": "",
      "key_elements": [str, ...],
      "emotional_state": "",
      "emotional_intensity": "",
      "lighting_condition": "",
      "movement_type": "",
      "timeline_mode": "",
      "motion_density": "",
      "repetition_handling": "",
      "motif_usage": [str, ...],
      "continuity_hooks": {"character": "", "motifs": [str, ...]},
    }

This engine MUST NOT:
    - define exact character appearance
    - define exact props (only general elements)
    - define camera shots / lens / angle
    - alter the storyboard's shot count or shot timings
    - generate executable visual prompts
    - over-specify details

Async because it does an OpenAI Chat call.  Returns a deterministic
fallback (passes through action_intents as enriched_directions and picks
the first valid_realization per scene) if the LLM call fails so the
pipeline never hard-stops here.
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
        "You are the CREATIVE BRIEF ENGINE for Qaivid (v3) — the FIRST "
        "commitment layer of a music-video pipeline.\n\n"
        "INPUT: a v3 storyboard with three layers:\n"
        "  • STORY   — the macro emotional arc of the whole video.\n"
        "  • SCENES  — ordered scenes with valid_realizations (creative "
        "possibilities), narrative phase, presence, motion, motifs, and "
        "an exact time_window.\n"
        "  • SHOTS   — concrete time-anchored shots inside each scene "
        "with start_time, end_time, duration, lyric_text, and a terse "
        "action_intent.  These shots are LOCKED — do not change their "
        "count, timings, lyric_text, or scene assignment.\n\n"
        "ROLE: produce a per-scene executable brief that:\n"
        "  1. ACCEPTS the storyboard's shots as the committed creative "
        "direction for each scene. The shots are already locked — their "
        "action_intent defines what happens on screen. Your job is NOT to "
        "pick a direction or choose from alternatives; the direction is "
        "already decided by the storyboard.\n"
        "  2. Enriches every shot by emitting an execution_detail string "
        "that adds execution depth — emotional state, lighting quality, "
        "character presence, atmosphere — drawn from the narrative, "
        "context, and style packets.  execution_detail is ADDITIVE only; "
        "the system will compose the final direction as "
        "`action_intent — execution_detail`.  Do NOT restate, paraphrase, "
        "or replace the action_intent inside execution_detail; assume the "
        "action_intent is already there.\n"
        "  3. Generates the per-scene production context fields "
        "(scene_purpose, subject_focus, environment_type, "
        "character_presence/identity_hint, key_elements, "
        "emotional_state/intensity, lighting_condition, movement_type, "
        "timeline_mode, motion_density, repetition_handling, "
        "motif_usage, continuity_hooks).  Each field is GENERAL and "
        "NON-SPECIFIC — these are the production envelope the shots live "
        "inside, not a restatement of shot content.\n\n"
        "RULES:\n"
        "1. SHOT FIDELITY — emit exactly one entry in shots[] per "
        "storyboard shot in the same order, preserving shot_id, "
        "start_time, end_time, duration, lyric_text, and action_intent. "
        "Add a new execution_detail string per shot.\n"
        "2. APPLY narrative logic: presence_strategy → character_presence; "
        "motion_philosophy → movement_type/motion_density; "
        "timeline_strategy → timeline_mode; expression_channels → which "
        "channels carry the emotion; repetition_strategy → "
        "repetition_handling on repeated sections.\n"
        "3. APPLY style guidance: tone, texture, color → "
        "lighting_condition + general atmosphere only.  Style does NOT "
        "override narrative decisions.\n"
        "4. MAINTAIN CONTINUITY across scenes: consistent character "
        "identity (via hints, NEVER appearance), consistent world, "
        "motif continuity, smooth emotional progression.\n"
        "5. CONTROL VARIATION: every scene needs a variation_anchor — "
        "what must stay the same on a re-run (e.g. 'protagonist remains "
        "the same person', 'mustard fields are always present').\n"
        "6. If a DIRECTOR'S IMAGINATION block is present:\n"
        "   - It is the PRIMARY creative directive for the whole video.\n"
        "   - The visual_concept sets the overarching tone you must serve.\n"
        "   - Listed motifs MUST appear in motif_usage across scenes.\n"
        "   - Director's notes set constraints that override defaults.\n\n"
        "FORBIDDEN:\n"
        "- Do NOT change the shot count, timings, or scene assignment.\n"
        "- Do NOT define exact character appearance.\n"
        "- Do NOT define exact props — only general key_elements.\n"
        "- Do NOT define camera shots, lens, or angles.\n"
        "- Do NOT generate executable prompts.\n"
        "- Do NOT over-specify details that downstream stages should choose.\n\n"
        "Return strict JSON.  No prose outside the JSON object."
    )


def _format_story(story: Dict[str, Any]) -> str:
    if not isinstance(story, dict) or not story:
        return "  (no story arc available)"
    bits: List[str] = []
    # Recognised v3 storyboard story keys (see storyboard_engine_v3._run_call1
    # output) — plus a few defensive aliases for forward compatibility.
    for k in ("summary", "arc", "macro_arc",
              "central_conflict", "logline",
              "emotional_journey", "thematic_throughline"):
        v = story.get(k)
        if v:
            bits.append(f"  - {k}: {str(v).strip()[:400]}")
    return "\n".join(bits) or "  (story present but no recognised fields)"


def _format_scene_with_shots(scene: Dict[str, Any],
                             shots_for_scene: List[Dict[str, Any]]) -> str:
    sid       = scene.get("scene_id") or "?"
    section   = scene.get("source_section") or ""
    phase     = scene.get("narrative_phase") or ""
    purpose   = scene.get("purpose") or ""
    intensity = scene.get("emotional_intensity") or ""
    presence  = scene.get("presence_hint") or ""
    motion    = scene.get("motion_density") or ""
    timeline  = scene.get("timeline_position") or ""
    ws        = scene.get("time_window_start")
    we        = scene.get("time_window_end")
    motifs    = scene.get("motif_usage") or []
    hooks     = scene.get("continuity_hooks") or {}

    lines = [
        f"  - scene_id: {sid} (section={section}, phase={phase}, "
        f"window=[{ws}, {we}], intensity={intensity}, presence={presence}, "
        f"motion={motion}, timeline={timeline})",
        f"    purpose: {str(purpose).strip()[:300]}",
        f"    motifs: {', '.join(motifs) if motifs else '—'}",
        (f"    continuity: subject={hooks.get('subject') or '—'}, "
         f"motifs_carry={', '.join(hooks.get('motifs') or []) or '—'}"),
    ]

    if shots_for_scene:
        lines.append("    shots (LOCKED — preserve count, timings, lyric, action_intent):")
        for sh in shots_for_scene:
            lyric = (sh.get("lyric_text") or "").strip().replace("\n", " ")
            if len(lyric) > 120:
                lyric = lyric[:117] + "…"
            lines.append(
                f"      • {sh.get('shot_id')} "
                f"[{sh.get('start_time')} → {sh.get('end_time')}] "
                f"= {sh.get('duration')}s | lyric: \"{lyric}\""
            )
            lines.append(
                f"          action_intent: {str(sh.get('action_intent') or '').strip()[:300]}"
            )
    else:
        lines.append("    shots: (none — this scene has no concrete shots; "
                     "skip per-shot enrichment for it)")
    return "\n".join(lines)


def _format_scenes_with_shots(scenes: List[Dict[str, Any]],
                              shots:  List[Dict[str, Any]]) -> str:
    if not scenes:
        return "  (no storyboard scenes available — derive a minimal brief from context.)"
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for sh in shots or []:
        sid = str(sh.get("scene_id") or "")
        by_scene.setdefault(sid, []).append(sh)
    return "\n".join(
        _format_scene_with_shots(s, by_scene.get(str(s.get("scene_id") or ""), []))
        for s in scenes
    )


def _format_narrative(narrative_packet: Dict[str, Any]) -> str:
    if not narrative_packet:
        return "  (no narrative packet available)"
    try:
        from narrative_engine import format_for_prompt
        block = format_for_prompt(narrative_packet)
        if block:
            return block
    except Exception:
        logger.exception("BriefV3: failed to format narrative_packet")
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
    story:              Dict[str, Any],
    scenes:             List[Dict[str, Any]],
    shots:              List[Dict[str, Any]],
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
            logger.exception("BriefV3: failed to format imagination_packet")

    return (
        ("\n\n" + imagination_block if imagination_block else "")
        + "\n\nSTORY (Stage 5 v3 — macro arc to honor):\n"
        + _format_story(story)
        + "\n\nSTORYBOARD SCENES & SHOTS (Stage 5 v3 — locked shots; accept as committed direction):\n"
        + _format_scenes_with_shots(scenes, shots)
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
          '      "scene_purpose":          "the emotional purpose of this scene (restate from storyboard, may enrich)",\n'
          '      "variation_anchor":       "what stays consistent across re-runs",\n'
          '      "shots": [\n'
          "        {\n"
          '          "shot_id":          "shot_1",\n'
          '          "execution_detail": "<lighting / atmosphere / emotional layer that adds depth to the locked visual concept — do NOT include the locked visual concept text itself; it is appended automatically by the system>"\n'
          "        }\n"
          "      ],\n"
          '      "subject_focus":          "character|environment|object|mixed",\n'
          '      "character_presence":     "continuous|intermittent|minimal",\n'
          '      "character_identity_hint":"downstream consistency hint, NOT appearance",\n'
          '      "environment_type":       "general environment type, NOT a named location",\n'
          '      "key_elements":           ["3-5 general elements"],\n'
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
          "shots[] must list one entry per LOCKED shot in that scene, in the same "
          "order, with the same shot_id — only add execution_detail. "
          "All values must remain general and non-locking — never name an exact "
          "person, an exact place, an exact prop, or a camera move."
    )


# ────────────────────────────────────────────────────────────────────────
# Coercion & fallback
# ────────────────────────────────────────────────────────────────────────
def _enum(value: Any, allowed: set, default: str) -> str:
    v = str(value or "").strip().lower().replace(" ", "_")
    return v if v in allowed else default


def _coerce_shot_brief(raw: Any, src_shot: Dict[str, Any]) -> Dict[str, Any]:
    """Always preserve the storyboard's locked metadata verbatim; the LLM
    may only contribute `execution_detail` (a short additive enrichment).
    `enriched_direction` is then deterministically composed as
    `action_intent — execution_detail` so the LLM cannot smuggle a
    changed visual concept into downstream stages.
    """
    if not isinstance(raw, dict):
        raw = {}

    # LOCKED FIELDS — pass through verbatim from src_shot.  No strip,
    # no truncate, no coerce — the storyboard owns these.
    #
    # `actions` carries the multishot decomposition for shots > the
    # video-render cap (each action is ≤8s and gets its own WAN/Kling
    # call in Phase 4).  We preserve it verbatim so the brief remains a
    # complete superset of the storyboard for downstream consumption.
    locked = {
        "shot_id":       src_shot.get("shot_id"),
        "scene_id":      src_shot.get("scene_id"),
        "start_time":    src_shot.get("start_time"),
        "end_time":      src_shot.get("end_time"),
        "duration":      src_shot.get("duration"),
        "lyric_text":    src_shot.get("lyric_text") or "",
        "action_intent": src_shot.get("action_intent") or "",
        "actions":       list(src_shot.get("actions") or []),
    }

    # ADDITIVE ENRICHMENT — the LLM contributes execution_detail only.
    # Accept either:
    #   (a) "execution_detail"     — preferred, additive only
    #   (b) "enriched_direction"   — legacy: keep only the suffix beyond
    #        the action_intent, so we still cannot smuggle a changed
    #        visual concept.
    ai = locked["action_intent"].strip()

    def _strip_ai_prefix(text: str) -> str:
        """If the LLM disobeyed and included the action_intent inside
        the detail, drop that prefix so we don't end up with the
        action_intent twice in the composed enriched_direction.  Also
        strips the literal token "action_intent" if the LLM echoed the
        placeholder phrase from the prompt template instead of writing
        real content (defensive — the template wording was reworded but
        we want belt-and-braces protection)."""
        t = text.strip()
        if ai and t.lower().startswith(ai.lower()):
            t = t[len(ai):].lstrip(" -—:;.,\t\n")
        # Defensive: drop a leading literal "action_intent" token that
        # the LLM may echo back from the prompt template.
        for tok in ("action_intent", "action intent"):
            if t.lower().startswith(tok):
                t = t[len(tok):].lstrip(" -—:;.,\t\n")
                break
        return t.strip()

    detail = _strip_ai_prefix(str(raw.get("execution_detail") or ""))
    if not detail:
        legacy = str(raw.get("enriched_direction") or "").strip()
        if ai and legacy.lower().startswith(ai.lower()):
            detail = _strip_ai_prefix(legacy)
        elif legacy and not legacy.lower().startswith("the storyboard"):
            # Legacy form replaced the visual concept entirely — do NOT
            # use it as a free-form replacement.  Treat as empty.
            detail = ""

    detail = detail[:400]

    # Compose enriched_direction deterministically.
    if detail and ai:
        enriched = f"{ai} — {detail}"
    elif detail:
        enriched = detail
    else:
        enriched = ai

    return {
        **locked,
        "execution_detail":   detail,
        "enriched_direction": enriched,
    }


def _coerce_scene_brief(raw: Any,
                        src_scene: Dict[str, Any],
                        src_shots: List[Dict[str, Any]],
                        idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        raw = {}
    # Per-shot enrichment — index LLM shots by shot_id, then pull in
    # source order so we never lose a storyboard shot to a missing LLM
    # entry.  Fall back to positional alignment for any LLM entry whose
    # shot_id we can't match (covers LLMs that omit shot_id).
    raw_shots = raw.get("shots") if isinstance(raw.get("shots"), list) else []
    raw_by_id: Dict[str, Any] = {}
    raw_unmatched: List[Any] = []
    for rsh in raw_shots:
        if not isinstance(rsh, dict):
            continue
        rid = str(rsh.get("shot_id") or "").strip()
        if rid:
            raw_by_id.setdefault(rid, rsh)
        else:
            raw_unmatched.append(rsh)
    coerced_shots: List[Dict[str, Any]] = []
    for pos, src_sh in enumerate(src_shots):
        sid = str(src_sh.get("shot_id") or "").strip()
        chosen = raw_by_id.get(sid)
        if chosen is None:
            # Positional fallback: prefer the LLM entry at the same
            # position (if it had no shot_id), else the next un-matched
            # entry, else None.
            if pos < len(raw_shots):
                cand = raw_shots[pos]
                if (isinstance(cand, dict)
                        and not str(cand.get("shot_id") or "").strip()):
                    chosen = cand
            if chosen is None and raw_unmatched:
                chosen = raw_unmatched.pop(0)
        coerced_shots.append(_coerce_shot_brief(chosen, src_sh))

    # chosen_direction is no longer LLM-generated — it is derived from
    # the storyboard's scene purpose (the direction is already committed
    # via the locked shots).  Keep the field for downstream compatibility.
    chosen = str(src_scene.get("purpose") or "").strip()[:400]

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

    # Legacy mirror — list-of-strings of the enriched per-shot directions,
    # kept so v2 consumers (timeline_builder_v2 etc.) keep working until
    # they're upgraded in Phase 3.
    shot_directions = [str(s.get("enriched_direction") or "").strip()[:500]
                       for s in coerced_shots
                       if str(s.get("enriched_direction") or "").strip()]

    return {
        # LOCKED FROM STORYBOARD — never let the LLM change these.
        "scene_id":                src_scene.get("scene_id") or f"s{idx + 1}",
        "source_section":          src_scene.get("source_section") or "",
        "narrative_phase":         src_scene.get("narrative_phase") or "build",
        # scene_purpose: prefer LLM's richer phrasing, fall back to source.
        "scene_purpose":           str(raw.get("scene_purpose")
                                        or src_scene.get("purpose")
                                        or "").strip()[:300],
        "chosen_direction":        chosen,
        "selection_basis":         "accepted from storyboard",
        "variation_anchor":        str(raw.get("variation_anchor") or "").strip()[:200],
        "shots":                   coerced_shots,
        "shot_directions":         shot_directions,
        "subject_focus":           _enum(raw.get("subject_focus"), _SUBJECT_FOCUS, "character"),
        "character_presence":      _enum(raw.get("character_presence"), _PRESENCE,
                                         _enum(src_scene.get("presence_hint"),
                                               _PRESENCE, "continuous")),
        "character_identity_hint": str(raw.get("character_identity_hint") or "").strip()[:200],
        "environment_type":        str(raw.get("environment_type") or "").strip()[:120],
        "key_elements":            [str(k).strip()[:80] for k in key_elements_in if str(k).strip()][:5],
        "emotional_state":         str(raw.get("emotional_state") or "").strip()[:120],
        "emotional_intensity":     _enum(raw.get("emotional_intensity"), _INTENSITIES,
                                         _enum(src_scene.get("emotional_intensity"),
                                               _INTENSITIES, "medium")),
        "lighting_condition":      str(raw.get("lighting_condition") or "").strip()[:120],
        "movement_type":           _enum(raw.get("movement_type"), _MOVEMENT, "slow"),
        "timeline_mode":           _enum(raw.get("timeline_mode"), _TIMELINE_MODES,
                                         _enum(src_scene.get("timeline_position"),
                                               _TIMELINE_MODES, "present")),
        "motion_density":          _enum(raw.get("motion_density"),
                                         _MOTION_DENSITY,
                                         _enum(src_scene.get("motion_density"),
                                               _MOTION_DENSITY, "medium")),
        "repetition_handling":     str(raw.get("repetition_handling") or "").strip()[:200],
        "motif_usage":             [str(m).strip()[:60] for m in motif_usage_in if str(m).strip()][:8],
        "continuity_hooks": {
            "character": str(hooks_in.get("character")
                              or hooks_in.get("subject")
                              or "").strip()[:200],
            "motifs":    [str(m).strip()[:60] for m in motifs_in if str(m).strip()][:8],
        },
    }


def _fallback(scenes: List[Dict[str, Any]],
              shots:  List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic fallback: pass through storyboard shots with action_intents
    as enriched_directions (no LLM). chosen_direction = scene purpose."""
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for sh in shots or []:
        sid = str(sh.get("scene_id") or "")
        by_scene.setdefault(sid, []).append(sh)

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes or []):
        sid = str(s.get("scene_id") or f"s{i + 1}")
        chosen = str(s.get("purpose") or "").strip()[:400]
        scene_shots = by_scene.get(sid, [])
        coerced_shots = [_coerce_shot_brief(None, sh) for sh in scene_shots]
        shot_directions = [s["enriched_direction"] for s in coerced_shots
                           if s.get("enriched_direction")]
        hooks = s.get("continuity_hooks") or {}
        out.append({
            "scene_id":                sid,
            "source_section":          str(s.get("source_section") or ""),
            "narrative_phase":         str(s.get("narrative_phase") or "build"),
            "scene_purpose":           str(s.get("purpose") or ""),
            "chosen_direction":        chosen,
            "selection_basis":         "accepted from storyboard",
            "variation_anchor":        "subject identity remains consistent",
            "shots":                   coerced_shots,
            "shot_directions":         shot_directions,
            "subject_focus":           "character",
            "character_presence":      _enum(s.get("presence_hint"), _PRESENCE, "continuous"),
            "character_identity_hint": str(hooks.get("subject") or "same speaker throughout")[:200],
            "environment_type":        "matches the song's locked world",
            "key_elements":            [],
            "emotional_state":         "",
            "emotional_intensity":     _enum(s.get("emotional_intensity"), _INTENSITIES, "medium"),
            "lighting_condition":      "natural to the chosen environment",
            "movement_type":           "slow",
            "timeline_mode":           _enum(s.get("timeline_position"),
                                             _TIMELINE_MODES, "present"),
            "motion_density":          _enum(s.get("motion_density"),
                                             _MOTION_DENSITY, "medium"),
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
async def generate_creative_brief_v3(
    api_key:             str,
    storyboard_packet:   Dict[str, Any],
    narrative_packet:    Dict[str, Any],
    context_packet:      Dict[str, Any],
    style_profile:       Optional[Dict[str, Any]] = None,
    input_structure:     Optional[Dict[str, Any]] = None,
    project_settings:    Optional[Dict[str, Any]] = None,
    imagination_packet:  Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Generate the v3 Creative Brief — one locked direction per scene,
    plus per-shot enrichment, from a v3 storyboard packet.

    Returns (scene_briefs, used_fallback).  Never raises — falls back to
    a deterministic per-scene first-realization pick (with action_intents
    passed through as enriched_directions) if the LLM call fails.
    """
    style_profile    = style_profile    or {}
    input_structure  = input_structure  or {}
    project_settings = project_settings or {}
    storyboard_packet = storyboard_packet or {}

    story  = storyboard_packet.get("story")  or {}
    scenes = list(storyboard_packet.get("scenes") or [])
    shots  = list(storyboard_packet.get("shots")  or [])

    if not scenes:
        logger.warning("BriefV3: no storyboard scenes — using empty fallback")
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
                    story,
                    scenes,
                    shots,
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

        # Index source scenes/shots by id for coercion.
        src_scene_by_id = {str(s.get("scene_id") or ""): s for s in scenes}
        src_shots_by_scene: Dict[str, List[Dict[str, Any]]] = {}
        for sh in shots:
            sid = str(sh.get("scene_id") or "")
            src_shots_by_scene.setdefault(sid, []).append(sh)

        # Index LLM scenes by id so we walk through SOURCE order — not LLM
        # order — and never drop a scene because the LLM omitted it.
        llm_by_id = {str((sc or {}).get("scene_id") or ""): sc for sc in out_scenes}
        coerced: List[Dict[str, Any]] = []
        for i, src in enumerate(scenes):
            sid = str(src.get("scene_id") or "")
            raw_s = llm_by_id.get(sid)
            if raw_s is None and i < len(out_scenes):
                # LLM didn't tag this entry by scene_id — fall back to
                # positional alignment for that one slot.
                raw_s = out_scenes[i]
            cb = _coerce_scene_brief(raw_s, src,
                                     src_shots_by_scene.get(sid, []), i)
            if cb:
                coerced.append(cb)
        if not coerced:
            logger.warning("BriefV3: LLM returned no usable scenes; using fallback")
            return _fallback(scenes, shots), True
        return coerced, False
    except Exception:
        logger.exception("BriefV3: generation failed; using fallback")
        return _fallback(scenes, shots), True


# ────────────────────────────────────────────────────────────────────────
# CLI runner — for offline phase-by-phase verification
# ────────────────────────────────────────────────────────────────────────
def _cli_main(argv: List[str]) -> int:
    """python creative_brief_engine_v3.py <project_id>

    Reads the v3 storyboard from /tmp/v3_output.json (or the brain if
    present) and the supporting packets from the brain, runs the brief,
    and writes the result to /tmp/brief_v3_output.json.
    """
    import asyncio
    import os
    import sys

    if len(argv) < 2:
        print("usage: python creative_brief_engine_v3.py <project_id>",
              file=sys.stderr)
        return 2
    project_id = argv[1]
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    # Try /tmp/v3_output.json first (Phase 1 artifact), fall back to brain
    storyboard_packet: Dict[str, Any] = {}
    try:
        with open("/tmp/v3_output.json", "r", encoding="utf-8") as fh:
            storyboard_packet = json.load(fh)
        print(f"[BriefV3 CLI] using /tmp/v3_output.json "
              f"({len(storyboard_packet.get('scenes') or [])} scenes, "
              f"{len(storyboard_packet.get('shots') or [])} shots)",
              file=sys.stderr)
    except Exception:
        print("[BriefV3 CLI] /tmp/v3_output.json not available — "
              "loading storyboard_packet from brain", file=sys.stderr)

    # Load supporting packets from brain
    try:
        from project_brain import ProjectBrain
        import psycopg
        from psycopg.rows import dict_row
        db_url = os.environ.get("DATABASE_URL")
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            brain = ProjectBrain.load(project_id, conn)
        if not storyboard_packet:
            storyboard_packet = brain.read("storyboard_packet") or {}
        narrative_packet  = brain.read("narrative_packet")  or {}
        context_packet    = brain.read("context_packet")    or {}
        style_profile     = brain.read("style_packet")      or {}
        input_structure   = brain.read("input_structure")   or {}
        project_settings  = brain.read("project_settings")  or {}
        imagination_packet = (brain.read("imagination_packet")
                              if brain.is_populated("imagination_packet")
                              else {})
    except Exception as exc:
        print(f"[BriefV3 CLI] failed to load brain: {exc}", file=sys.stderr)
        return 3

    scene_briefs, used_fallback = asyncio.run(generate_creative_brief_v3(
        api_key=api_key,
        storyboard_packet=storyboard_packet,
        narrative_packet=narrative_packet,
        context_packet=context_packet,
        style_profile=style_profile,
        input_structure=input_structure,
        project_settings=project_settings,
        imagination_packet=imagination_packet,
    ))

    out = {
        "schema_version": 3,
        "scenes":         scene_briefs,
        "used_fallback":  used_fallback,
        "scene_count":    len(scene_briefs),
        "shot_count":     sum(len(s.get("shots") or []) for s in scene_briefs),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main(sys.argv))
