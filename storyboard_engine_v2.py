"""Storyboard Engine v2 — pure intent layer (master spec).

ROLE:
    Translate narrative logic into structured scene-level intent
    WITHOUT locking visuals, shots, or execution.

    First stage where "what could be shown" is explored,
    NOT "what will be shown".

Inputs (read by the worker from Project Brain):
    input_structure   — sections, repetition_map, units, timing
    context_packet    — core_theme, emotional_world/arc, world, must_preserve
    narrative_packet  — storytelling_mode, presence/motion/expression strategy…
    style_packet      — cinematic_style, visual tone (LIGHT influence only)
    project_settings  — duration, platform, constraints (optional)

Output: a list of scenes; each scene has the schema:
    {
      "scene_id":            str,
      "source_section":      str,        # "intro", "verse1", "chorus", ...
      "narrative_phase":     str,        # intro|build|peak|breakdown|resolution
      "purpose":             str,        # what is this scene doing emotionally
      "emotional_intensity": str,        # low|medium|high|peak
      "presence_hint":       str,        # full|partial|absent|object_focus|...
      "motion_density":      str,        # low|medium|high
      "timeline_position":   str,        # present|memory|fragmented|future|dream
      "motif_usage":         [str, ...], # which motifs appear here
      "continuity_hooks": {
        "subject": str,                  # subject continuity note
        "motifs":  [str, ...]            # motifs to carry forward
      },
      "valid_realizations": [str, ...]   # 3-6 alternative ways to express purpose
    }

This engine MUST NOT:
    - lock characters, locations, props, shots, or prompts
    - pick a single realization
    - generate executable visual prompts

Async because it does an OpenAI Chat call. Returns a deterministic
fallback (one scene per section, with generic valid_realizations)
if the LLM call fails so the pipeline never hard-stops here.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"
_MIN_REALIZATIONS = 3
_MAX_REALIZATIONS = 6

_NARRATIVE_PHASES = {"intro", "build", "peak", "breakdown", "resolution"}
_INTENSITIES      = {"low", "medium", "high", "peak"}
_PRESENCE         = {"full", "partial", "absent", "object_focus", "silhouette",
                     "hands_only", "memory_presence"}
_MOTION_DENSITY   = {"low", "medium", "high"}
_TIMELINE_POS     = {"present", "memory", "fragmented", "future", "dream"}


# ────────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────────
def _system_prompt() -> str:
    return (
        "You are the STORYBOARD ENGINE for Qaivid — a music-video pipeline.\n\n"
        "ROLE: Translate narrative logic into structured scene-level INTENT.\n"
        "You do NOT lock visuals, shots, or execution. You explore "
        "'what could be shown', not 'what will be shown'.\n\n"
        "RULES:\n"
        "1. Segment the song into scenes using the lyric sections + emotional arc.\n"
        "   Maintain correct order. Merge or split for pacing only when justified.\n"
        "2. Assign a narrative_phase to each scene "
        "(intro|build|peak|breakdown|resolution) based on the emotional progression.\n"
        "3. Define a single, sharp purpose per scene (e.g. 'establish isolation',\n"
        "   'intensify longing', 'transition into memory'). Purpose answers:\n"
        "   what is this scene doing EMOTIONALLY?\n"
        "4. Apply narrative logic from the NARRATIVE INTELLIGENCE block:\n"
        "   - presence_strategy → presence_hint\n"
        "   - motion_philosophy → motion_density\n"
        "   - timeline_strategy → timeline_position\n"
        "   - expression_channels → which channels carry the emotion\n"
        "   - repetition_strategy → how repeated sections (chorus) evolve\n"
        "5. For EVERY scene, generate between "
        f"{_MIN_REALIZATIONS} and {_MAX_REALIZATIONS} valid_realizations — "
        "concrete but unlocked alternative ways the purpose can be expressed.\n"
        "   Each realization is a single sentence describing the scene's IDEA "
        "(not its locked execution). All must be contextually correct. None is "
        "the final pick — that happens later.\n"
        "6. Track continuity_hooks (subject + motifs carried across scenes).\n"
        "7. Track motif_usage per scene — introduce, repeat, evolve.\n"
        "8. Repeated sections (chorus, refrain) MUST evolve visually — never\n"
        "   duplicate blindly. Use the repetition strategy from narrative.\n"
        "9. Style is a LIGHT influence (tone only). Style does NOT dictate scene\n"
        "   content. Narrative + context dictate scene content.\n"
        "10. If a DIRECTOR'S IMAGINATION block is present in the user message:\n"
        "    - The visual_concept is the PRIMARY creative mandate for the whole video.\n"
        "    - Section Flow entries are PER-SECTION directives: for every scene that belongs\n"
        "      to a named section (intro, verse, chorus, bridge, outro, etc.), find the\n"
        "      matching [section_label] entry and apply its mood + visual_idea when\n"
        "      generating valid_realizations for that scene. The section_label is an\n"
        "      authoritative override of your own mood inference.\n"
        "    - Motifs listed in the imagination block MUST appear in motif_usage across\n"
        "      scenes — introduce early, evolve on repeats, resolve at end.\n"
        "    - Shot ideas are creative sparks — use them to inspire realizations.\n"
        "    - Director's notes contain hard constraints that override all defaults.\n\n"
        "FORBIDDEN:\n"
        "- Do NOT name exact locations, characters, props, camera shots, or prompts.\n"
        "- Do NOT pick a single realization.\n"
        "- Do NOT over-specify visuals.\n"
        "- Do NOT literally map lyrics to images.\n\n"
        "Return strict JSON. No prose outside the JSON object."
    )


def _format_input_structure(input_structure: Dict[str, Any]) -> str:
    """Compact lyric-section block for the prompt.

    Matches the actual InputProcessor schema:
      sections: [{id, type, label, is_inferred, unit_ids, repeat_of}, ...]
      units:    [{id, section_id, index, text, start_time, ...}, ...]

    When Stage 1 collapses everything into one inferred section (common for
    non-English or hard-to-parse audio), we surface more lyric content so the
    LLM can find natural breakpoints and create multiple scenes.
    """
    sections = input_structure.get("sections") or []
    units    = input_structure.get("units") or []
    if not sections:
        # Fall back: show first few raw lines so the LLM has something.
        clean = (input_structure.get("clean_text") or "").strip()
        if clean:
            preview = " / ".join(
                ln.strip() for ln in clean.splitlines()[:8] if ln.strip()
            )
            return f"  (no sections — preview: {preview[:400]})"
        return "  (no structured sections available)"

    # Build unit lookup (id → text + index) so we can surface real lyric
    # content per section without dumping the whole transcript into the prompt.
    unit_text: Dict[str, str] = {}
    unit_index: Dict[str, int] = {}
    for u in units:
        if isinstance(u, dict):
            uid = u.get("id")
            if uid:
                unit_text[uid]  = str(u.get("text") or "").strip()
                unit_index[uid] = int(u.get("index") or 0)

    # Detect collapsed-section: one section (likely inferred) with many units.
    all_inferred = all(s.get("is_inferred") for s in sections)
    collapsed    = len(sections) == 1 and all_inferred and len(units) > 10

    lines: List[str] = []
    for sec in sections:
        sid       = sec.get("id") or "?"
        stype     = sec.get("type") or "section"
        label     = sec.get("label") or stype
        repeat_of = sec.get("repeat_of")
        unit_ids  = sec.get("unit_ids") or []

        if collapsed:
            # Show first, middle, and last handful of lines so GPT can locate
            # emotional shifts and propose natural scene boundaries.
            n = len(unit_ids)
            sample_ids = (
                unit_ids[:5]
                + unit_ids[n // 4: n // 4 + 3]
                + unit_ids[n // 2: n // 2 + 3]
                + unit_ids[3 * n // 4: 3 * n // 4 + 3]
                + unit_ids[-4:]
            )
            # Deduplicate while preserving order.
            seen: set = set()
            sampled: List[str] = []
            for uid in sample_ids:
                if uid not in seen:
                    seen.add(uid)
                    sampled.append(uid)
            body_parts = [
                f"[unit {unit_index.get(uid, '?')}] {unit_text.get(uid, '')}"
                for uid in sampled if unit_text.get(uid)
            ]
            body = " / ".join(body_parts)
        else:
            body_parts = [unit_text.get(uid, "") for uid in unit_ids[:4]]
            body = " / ".join(p for p in body_parts if p)
            if len(unit_ids) > 4:
                body += " / …"

        tag = f" (repeats {repeat_of})" if repeat_of else ""
        inferred_tag = " [auto-detected, may need splitting]" if sec.get("is_inferred") else ""
        lines.append(
            f"  [{sid}] {stype} — {label}{tag}{inferred_tag} "
            f"({len(unit_ids)} units): {body}"
        )

    rep_map = input_structure.get("repetition_map") or {}
    if rep_map:
        lines.append(
            f"  repetition_map: "
            f"{json.dumps(rep_map, ensure_ascii=False)[:240]}"
        )
    return "\n".join(lines)


def _format_context(context_packet: Dict[str, Any]) -> str:
    spk = context_packet.get("speaker") or {}
    motivation = context_packet.get("motivation") or {}
    world = context_packet.get("world_assumptions") or {}
    payload = {
        "core_theme":         context_packet.get("core_theme"),
        "emotional_world":    context_packet.get("emotional_world"),
        "emotional_arc":      context_packet.get("emotional_arc"),
        "narrative_mode":     context_packet.get("narrative_mode"),
        "speaker_identity":   spk.get("identity"),
        "speaker_emotion":    spk.get("emotional_state"),
        "location_dna":       context_packet.get("location_dna"),
        "era":                context_packet.get("era"),
        "motivation":         motivation,
        "world":              world,
        "must_preserve":      context_packet.get("must_preserve"),
        "creative_freedom":   context_packet.get("creative_freedom"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_style_light(style_profile: Dict[str, Any]) -> str:
    cin  = (style_profile or {}).get("cinematic")  or {}
    prod = (style_profile or {}).get("production") or {}
    payload = {
        "cinematic_style":  cin.get("name") or cin.get("id"),
        "production_style": prod.get("name") or prod.get("id"),
        "preset":           (style_profile or {}).get("preset"),
        "tone_keywords":    (style_profile or {}).get("tone_keywords"),
    }
    # Vibe preset direction — only present when a vibe preset was chosen.
    # Informs the storyboard LLM about the specific shot-composition, framing-
    # ratio, and visual energy rules for this production identity.
    sp = style_profile or {}
    if sp.get("vibe_label"):
        payload["vibe_preset"] = sp["vibe_label"]
    if sp.get("vibe_storyboard_direction"):
        payload["vibe_storyboard_direction"] = sp["vibe_storyboard_direction"]
    if sp.get("vibe_avoid"):
        avoid = sp["vibe_avoid"]
        payload["vibe_avoid"] = avoid if isinstance(avoid, list) else [avoid]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_emotional_mode(emp: Dict[str, Any]) -> str:
    """Compact EMOTIONAL MODE block from emotional_mode_packet (Stage 2b).
    Returns "" if the packet is empty.
    """
    if not emp:
        return ""
    label    = emp.get("mode_label") or emp.get("primary_mode") or ""
    modifier = emp.get("cinematic_modifier") or ""
    if not label:
        return ""
    lines = [f"  Mode:           {label}"]
    if modifier:
        lines.append(f"  Aesthetic feel: {modifier}")
    prod_aff = emp.get("production_affinity") or {}
    if prod_aff.get("preferred"):
        lines.append(f"  Production:     {', '.join(prod_aff['preferred'])}")
    tone_words = emp.get("tone_words") or []
    if tone_words:
        lines.append(f"  Tone words:     {', '.join(tone_words[:5])}")
    return "EMOTIONAL MODE (Stage 2b — intent must serve this register):\n" + "\n".join(lines)


def _user_prompt(
    input_structure: Dict[str, Any],
    context_packet:  Dict[str, Any],
    narrative_packet: Dict[str, Any],
    style_profile:   Dict[str, Any],
    project_settings: Dict[str, Any],
    emotional_mode_packet: Optional[Dict[str, Any]] = None,
    imagination_packet: Optional[Dict[str, Any]] = None,
) -> str:
    # NARRATIVE INTELLIGENCE block (re-use the narrative_engine formatter)
    narrative_block = ""
    if narrative_packet:
        try:
            from narrative_engine import format_for_prompt
            narrative_block = format_for_prompt(narrative_packet) or ""
        except Exception:
            logger.exception("StoryboardV2: failed to format narrative_packet")

    # DIRECTOR'S IMAGINATION block — primary creative directive
    imagination_block = ""
    if imagination_packet:
        try:
            from imagination_engine import format_imagination_for_prompt
            imagination_block = format_imagination_for_prompt(imagination_packet) or ""
        except Exception:
            logger.exception("StoryboardV2: failed to format imagination_packet")

    sections  = input_structure.get("sections") or []
    n_sections = len(sections)
    n_units    = len(input_structure.get("units") or [])

    # Detect collapsed-section: Stage 1 auto-detected only 1 section but the
    # song actually has many lyric units.  In this case the default "1 section
    # → 1 scene" guidance produces a single, useless brief.  Override with a
    # unit-based target so the storyboard creates a realistic arc.
    all_inferred  = all(s.get("is_inferred") for s in sections)
    collapsed     = n_sections == 1 and all_inferred and n_units > 10

    if collapsed:
        import math
        # Aim for roughly one scene per 6-8 units, capped 4-8.
        target_n = min(8, max(4, math.ceil(n_units / 7)))
        min_n = max(2, target_n - 1)
        max_n = min(10, target_n + 2)
        scene_count_hint = (
            f"\n\nSCENE COUNT GUIDANCE (IMPORTANT): Stage 1 auto-detected only "
            f"1 section for this song, but it has {n_units} lyric units — "
            f"meaning the structural analysis is incomplete. "
            f"DO NOT create only 1 scene. "
            f"Use the lyric content above to identify natural emotional breakpoints "
            f"(opening, early verses, hook/chorus, middle section, emotional peak, "
            f"resolution/outro) and create between {min_n} and {max_n} scenes. "
            f"Each scene must have a distinct emotional purpose and narrative phase. "
            f"Distribute the {n_units} units evenly across your chosen number of scenes."
        )
    elif n_sections:
        scene_count_hint = (
            f"\n\nSCENE COUNT GUIDANCE: the song has ~{n_sections} lyric sections. "
            "Produce one scene per section by default. You MAY merge two adjacent "
            "sections into one scene, or split a long section into two scenes, "
            "but only when the emotional arc justifies it.\n\n"
            "VISUAL VARIETY MANDATE (non-negotiable for storyboard quality):\n"
            "- Each scene MUST describe a visually distinct environment, perspective, "
            "or temporal moment from its neighbours. Two consecutive scenes showing "
            "the same character in the same room is a failed storyboard.\n"
            "- Repeated sections (e.g. chorus appearing 3 times) MUST each have a "
            "different timeline_position, presence_hint, or visual environment — "
            "use memory/dream/present/fragmented to create contrast.\n"
            "- If the song has an addressee (a 'you' who is absent or lost), at "
            "least 2 scenes MUST evoke that person through memory_presence, "
            "object_focus on their belongings, or a flashback timeline_position. "
            "A video with only one person reacting to loss — never showing what "
            "was lost — has no story.\n"
            "- Include at least one scene set outside or in a clearly different "
            "spatial environment from the dominant indoor/domestic space."
        )
    else:
        scene_count_hint = ""

    mode_block = _format_emotional_mode(emotional_mode_packet or {})

    return (
        ("\n\n" + imagination_block if imagination_block else "")
        + "\n\nINPUT STRUCTURE (Stage 1 — sections + repetition):\n"
        + _format_input_structure(input_structure or {})
        + "\n\nCONTEXT PACKET (Stage 2 — locked meaning, world, speaker):\n"
        + _format_context(context_packet or {})
        + ("\n\n" + narrative_block if narrative_block else "")
        + ("\n\n" + mode_block if mode_block else "")
        + "\n\nSTYLE PACKET (Stage 4 — light tone influence ONLY):\n"
        + _format_style_light(style_profile or {})
        + "\n\nPROJECT SETTINGS:\n"
        + json.dumps(project_settings or {}, indent=2, ensure_ascii=False)
        + scene_count_hint
        + "\n\nReturn JSON of the form:\n"
        + "{\n"
          '  "scenes": [\n'
          "    {\n"
          '      "scene_id": "s1",\n'
          '      "source_section": "intro|verse1|chorus|...",\n'
          '      "narrative_phase": "intro|build|peak|breakdown|resolution",\n'
          '      "purpose": "What this scene is doing emotionally (one sentence)",\n'
          '      "emotional_intensity": "low|medium|high|peak",\n'
          '      "presence_hint": "full|partial|absent|object_focus|silhouette|hands_only|memory_presence",\n'
          '      "motion_density": "low|medium|high",\n'
          '      "timeline_position": "present|memory|fragmented|future|dream",\n'
          '      "motif_usage": ["motif name", "..."],\n'
          '      "continuity_hooks": {\n'
          '        "subject": "Subject continuity note (e.g. \\"same speaker, evolving emotion\\")",\n'
          '        "motifs": ["motif to carry forward", "..."]\n'
          "      },\n"
          '      "valid_realizations": [\n'
          '        "concrete idea 1 (one sentence, no locked execution)",\n'
          '        "concrete idea 2",\n'
          '        "concrete idea 3"\n'
          "      ]\n"
          "    }\n"
          "  ]\n"
          "}\n\n"
          f"Each scene MUST have between {_MIN_REALIZATIONS} and "
          f"{_MAX_REALIZATIONS} valid_realizations. None is the final pick."
    )


# ────────────────────────────────────────────────────────────────────────
# Coercion & fallback
# ────────────────────────────────────────────────────────────────────────
def _enum(value: Any, allowed: set, default: str) -> str:
    v = str(value or "").strip().lower().replace(" ", "_")
    return v if v in allowed else default


def _coerce_scene(raw: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    realizations_in = raw.get("valid_realizations") or []
    if not isinstance(realizations_in, list):
        return None
    realizations = [
        str(r).strip()[:300] for r in realizations_in if str(r).strip()
    ][:_MAX_REALIZATIONS]
    if len(realizations) < _MIN_REALIZATIONS:
        # Pad with deliberately under-specified variants so downstream is
        # never left with a single locked option.
        while len(realizations) < _MIN_REALIZATIONS:
            realizations.append("alternative framing of the same emotional intent")

    hooks_in = raw.get("continuity_hooks") or {}
    if not isinstance(hooks_in, dict):
        hooks_in = {}
    motifs_in = hooks_in.get("motifs") or []
    if not isinstance(motifs_in, list):
        motifs_in = []

    motif_usage_in = raw.get("motif_usage") or []
    if not isinstance(motif_usage_in, list):
        motif_usage_in = []

    return {
        "scene_id":            str(raw.get("scene_id") or f"s{idx + 1}").strip()[:24],
        "source_section":      str(raw.get("source_section") or "").strip()[:48],
        "narrative_phase":     _enum(raw.get("narrative_phase"), _NARRATIVE_PHASES, "build"),
        "purpose":             str(raw.get("purpose") or "").strip()[:300],
        "emotional_intensity": _enum(raw.get("emotional_intensity"), _INTENSITIES, "medium"),
        "presence_hint":       _enum(raw.get("presence_hint"), _PRESENCE, "full"),
        "motion_density":      _enum(raw.get("motion_density"), _MOTION_DENSITY, "medium"),
        "timeline_position":   _enum(raw.get("timeline_position"), _TIMELINE_POS, "present"),
        "motif_usage":         [str(m).strip()[:60] for m in motif_usage_in if str(m).strip()][:8],
        "continuity_hooks": {
            "subject": str(hooks_in.get("subject") or "").strip()[:200],
            "motifs":  [str(m).strip()[:60] for m in motifs_in if str(m).strip()][:8],
        },
        "valid_realizations":  realizations,
    }


def _fallback(input_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic fallback: one scene per section with generic options.

    When Stage 1 collapses everything into one inferred section with many
    units, we generate a proportional multi-scene arc instead of a single
    useless scene.
    """
    import math as _math

    sections = input_structure.get("sections") or []
    units    = input_structure.get("units") or []

    # Detect collapsed-section case.
    all_inferred = all(s.get("is_inferred") for s in sections)
    collapsed    = len(sections) == 1 and all_inferred and len(units) > 10

    if collapsed:
        target_n = min(8, max(4, _math.ceil(len(units) / 7)))
        arc = [
            ("intro",       "Establish the emotional world and draw the viewer in.",
             "low",  "full",  "present"),
            ("build",       "Deepen the emotional stakes as the tension rises.",
             "medium", "partial", "present"),
            ("build",       "Intensify the longing — the emotional core surfaces.",
             "medium", "full", "memory"),
            ("peak",        "The emotional climax — raw feeling at its most intense.",
             "high", "full", "present"),
            ("breakdown",   "Fragmentation — the weight of the emotion starts to break.",
             "high", "silhouette", "fragmented"),
            ("breakdown",   "Reflection and introspection after the peak.",
             "medium", "partial", "memory"),
            ("resolution",  "A quiet coming-to-terms — acceptance or lingering grief.",
             "low", "object_focus", "present"),
            ("resolution",  "Final image — the world as it is now, transformed by emotion.",
             "low", "absent", "present"),
        ]
        scenes: List[Dict[str, Any]] = []
        for i in range(target_n):
            phase, purpose, intensity, presence, timeline = arc[i % len(arc)]
            scenes.append({
                "scene_id":            f"s{i + 1}",
                "source_section":      f"segment_{i + 1}",
                "narrative_phase":     phase,
                "purpose":             purpose,
                "emotional_intensity": intensity,
                "presence_hint":       presence,
                "motion_density":      "medium",
                "timeline_position":   timeline,
                "motif_usage":         [],
                "continuity_hooks":    {"subject": "same speaker, evolving emotion", "motifs": []},
                "valid_realizations":  [
                    "subject in the primary setting expressing the emotion directly",
                    "environment-focused beat with the subject partially present",
                    "object or motif beat that echoes the subject's inner state",
                ],
            })
        return scenes

    # Normal path: one scene per section.
    scenes: List[Dict[str, Any]] = []
    n = max(len(sections), 1)
    for i, sec in enumerate(sections or [{"type": "scene", "label": "Scene"}]):
        # Real schema: type + label; older shape: name/section.
        name = (sec.get("label") or sec.get("type")
                or sec.get("name") or sec.get("section")
                or f"section_{i + 1}")
        progress = i / max(n - 1, 1) if n > 1 else 0
        if   progress < 0.20: phase = "intro"
        elif progress < 0.55: phase = "build"
        elif progress < 0.75: phase = "peak"
        elif progress < 0.90: phase = "breakdown"
        else:                 phase = "resolution"
        scenes.append({
            "scene_id":            f"s{i + 1}",
            "source_section":      str(name)[:48],
            "narrative_phase":     phase,
            "purpose":             "Express the emotional intent of this section.",
            "emotional_intensity": "medium",
            "presence_hint":       "full",
            "motion_density":      "medium",
            "timeline_position":   "present",
            "motif_usage":         [],
            "continuity_hooks":    {"subject": "same speaker", "motifs": []},
            "valid_realizations":  [
                "subject in the primary setting expressing the emotion directly",
                "environment-focused beat with the subject partially present",
                "object or motif beat that echoes the subject's inner state",
            ],
        })
    return scenes


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────
async def generate_storyboard_v2(
    api_key: str,
    input_structure: Dict[str, Any],
    context_packet:  Dict[str, Any],
    narrative_packet: Dict[str, Any],
    style_profile:   Optional[Dict[str, Any]] = None,
    project_settings: Optional[Dict[str, Any]] = None,
    emotional_mode_packet: Optional[Dict[str, Any]] = None,
    imagination_packet: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Generate the v2 storyboard intent layer.

    Returns (scenes, used_fallback). Never raises — falls back to a
    deterministic per-section scene list if the LLM call fails.
    """
    style_profile    = style_profile    or {}
    project_settings = project_settings or {}

    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=_MODEL,
            response_format={"type": "json_object"},
            temperature=0.8,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(
                    input_structure or {},
                    context_packet  or {},
                    narrative_packet or {},
                    style_profile,
                    project_settings,
                    emotional_mode_packet=emotional_mode_packet or {},
                    imagination_packet=imagination_packet or {},
                )},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        scenes_in = data.get("scenes") or []
        coerced: List[Dict[str, Any]] = []
        for i, s in enumerate(scenes_in):
            cs = _coerce_scene(s, i)
            if cs:
                coerced.append(cs)
        if not coerced:
            logger.warning("StoryboardV2: LLM returned no valid scenes; using fallback")
            return _fallback(input_structure or {}), True
        return coerced, False
    except Exception:
        logger.exception("StoryboardV2: generation failed; using fallback")
        return _fallback(input_structure or {}), True
