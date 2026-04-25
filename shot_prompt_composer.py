"""Tight, model-friendly prompt composer for per-shot still generation.

MM3.1 upgrade:
- Preserves the existing public API:
    compose_image_prompt(...)
    QUALITY_BOOSTERS
    DEFAULT_NEGATIVE
- Keeps prompts compact and model-friendly.
- Stops throwing away the most important cinematic information when present:
  action, trigger, object interaction, environment interaction, emotional shift,
  and visual contrast from the new MM3.1 shot-event layer.
- Falls back cleanly to the legacy MM3 shot fields when no new data exists.
"""

from __future__ import annotations

import re
from typing import Optional, Dict, Any


# ── Quality boosters and negative prompt (Flux-tuned) ─────────────────
QUALITY_BOOSTERS = (
    "cinematic realism, music video film still, "
    "warm color grade, lifted shadows, rich deep tones, "
    "shallow depth of field, beautiful intentional light, "
    "sharp focus on subject, professional music video cinematography, "
    "striking yet authentic, prestige production quality"
)

DEFAULT_NEGATIVE = (
    "blurry, watermark, text overlay, logo, deformed anatomy, "
    "extra fingers, mutated hands, cartoon, anime, illustration, "
    "jpeg artifacts, plastic skin, cgi look, "
    "amateur photography, bad composition, "
    "ugly, unflattering, harsh shadows, flat lighting, "
    "gritty documentary rawness, dull desaturated colors, mundane snapshot"
)

# Culture-specific negative additions keyed by a substring that must appear
# (case-insensitive) in the assembled environment text to activate.
_CULTURE_NEGATIVES: list[tuple[str, str]] = [
    (
        "punjab",
        "Mughal arches, ornate stone colonnade, haveli ornamentation, intricate carved stonework, "
        "Nawabi architecture, Lucknow haveli, arched gallery, decorative pillars, brick walls, "
        "tiled courtyard, concrete floor, marble, ornamental garden, paved path, stone fountain, "
        "studio set, Rajasthani facade, painted wall murals",
    ),
]


def _culture_negative(env_text: str) -> str:
    """Return additional negative-prompt terms for culture-specific location accuracy."""
    low = env_text.lower()
    extras: list[str] = []
    for trigger, neg in _CULTURE_NEGATIVES:
        if trigger in low:
            extras.append(neg)
    return ", ".join(extras)


def _build_negative(shot: dict, location: Optional[dict]) -> str:
    """Assemble the full negative prompt: default terms + any culture-specific additions.

    Culture detection reads the location name/dna and the shot's environment_profile
    so it works whether a location row is linked or the data is embedded in the shot.
    """
    env_text_parts: list[str] = []
    if location:
        env_text_parts.extend(filter(None, [
            location.get("name", ""),
            location.get("location_dna", ""),
            location.get("description", ""),
        ]))
    ep = shot.get("environment_profile") or {}
    if isinstance(ep, dict):
        env_text_parts.append(ep.get("location_dna") or "")
    env_text = " ".join(env_text_parts)
    culture_neg = _culture_negative(env_text)
    if culture_neg:
        return DEFAULT_NEGATIVE + ", " + culture_neg
    return DEFAULT_NEGATIVE

# ── Phrases to strip — pipeline-internal directives that must not reach the
#    image model as literal text.
#
#    MM3.1 removal: Performance, Function, Arc beat, and Motif handling are
#    intentionally NOT stripped.  In the new architecture these values arrive
#    in structured shot_event/cinematic_beat fields and provide useful context
#    for the image model about what the shot must convey.  Keeping these labels
#    when they appear in visual_prompt or user_override text is the correct
#    MM3.1 behaviour (they tell the model the emotional register and story
#    function of the shot).
_INSTRUCTION_PATTERNS = [
    r"Maintain (?:strict )?character continuity[^.]*\.",
    r"Repetition logic:[^.]*\.",
    r"Ambiguity handling:[^.]*\.",
    r"Relevant ambiguity notes:[^.]*\.",
    r"Source format awareness:[^.]*\.",
    r"Hard restrictions[^:]*:[^.]*\.",
    r"Visual constraints:[^.]*\.",
    r"Continuity:[^.]*\.",
    r"Cultural subtext to preserve:[^.]*\.",
    r"Why this song exists:[^.]*\.",
    r"Spine anchor:[^.]*\.",
    r"Treatment:[^.]*\.",
    r"Style notes:[^.]*\.",
    r"Rendering notes:[^.]*\.",
    r"Transition behavior:[^.]*\.",
    r"Repeat status:[^.]*\.",
    r"Speaker identity context:[^.]*\.",
    r"Addressee context:[^.]*\.",
    r"Intensity:[^.]*\.",
]
_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)


_ENV_LABEL_RE = re.compile(
    r"\b(?:world\s*dna|location\s*dna|geography|architecture(?:\s*style)?|"
    r"domestic\s*setting|characteristic\s*setting|characteristic\s*time|"
    r"time\s*of\s*day|season)\s*:\s*",
    re.IGNORECASE,
)


def _trim(s: Optional[str], limit: int = 140) -> str:
    """Sentence-aware trim. Returns at most ``limit`` chars, ending on a
    sentence/clause boundary where possible."""
    if not s:
        return ""
    s = " ".join(str(s).split())
    if len(s) <= limit:
        return s.rstrip(" .,;:") + "."
    cut = s[:limit]
    for sep in (". ", "; ", ", "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            return cut[:i].rstrip(" .,;:") + "."
    return cut.rstrip(" .,;:") + "."


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = _INSTRUCTION_RE.sub("", str(s))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _event_payload(shot: dict) -> Dict[str, Any]:
    for key in ("shot_event", "event", "cinematic_beat"):
        payload = shot.get(key)
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _gendered_subject(
    character: Optional[dict],
    brain_char: Optional[dict] = None,
) -> str:
    """Build a concrete, gendered subject phrase.

    Prefers identity_seed + archetype from the brain materializer entry
    (authoritative) over raw DB fields. Falls back cleanly when no brain
    data is available. Avoids the 'Adult Unspecified figure' fallback that
    breaks Flux identity.
    """
    bc = brain_char or {}

    identity_seed = (bc.get("identity_seed") or "").strip()
    archetype      = (bc.get("archetype")      or "").strip()
    cultural       = (bc.get("cultural_markers") or "").strip()

    if identity_seed:
        # Brain-anchored: identity_seed is the authoritative physical descriptor.
        base = identity_seed
        if archetype:
            base = f"{base} ({archetype})"
        if cultural:
            base = f"{base}, {cultural}"
        return base

    if not character:
        return "a person"

    parts: list[str] = []

    age = (character.get("age_range") or "").strip().lower()
    if age and age not in {"unspecified", "unknown"}:
        parts.append(age)

    ethnicity = (character.get("ethnicity") or "").strip()
    if ethnicity:
        parts.append(ethnicity)

    gender = (character.get("gender") or "").strip().lower()
    if gender in {"male", "man"}:
        parts.append("man")
    elif gender in {"female", "woman"}:
        parts.append("woman")
    else:
        parts.append("person")

    role = (character.get("role") or "").strip()
    if role:
        parts.append(f"({role.lower()})")

    if archetype:
        parts.append(f"[{archetype}]")

    return " ".join(parts).strip() or "a person"


def _wardrobe_clause(character: Optional[dict]) -> str:
    if not character:
        return ""

    wardrobe = (
        character.get("wardrobe_override")
        or character.get("wardrobe")
        or ""
    ).strip()
    grooming = (character.get("grooming") or "").strip()

    _WEARS_PREFIX = re.compile(
        r"^(?:the\s+\w+(?:\s+\w+)?\s+wears?\s+|wearing\s+)",
        re.IGNORECASE,
    )
    wardrobe = _WEARS_PREFIX.sub("", wardrobe).strip()

    bits = [b for b in (wardrobe, grooming) if b]
    if not bits:
        return ""
    return _trim("wearing " + ", ".join(bits), 160)


def _dedupe_phrases(s: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in re.split(r"\s*[,.]\s*", s):
        c = chunk.strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return ", ".join(out)


def _environment_clause(
    location: Optional[dict],
    shot: dict,
    brain_loc: Optional[dict] = None,
) -> str:
    raw = ""
    bl = brain_loc or {}

    # Brain-first: use location_profile world DNA as authoritative world descriptor.
    world_dna_parts: list[str] = []
    for field in ("world", "environment_type", "key_textures", "palette_anchor", "architecture"):
        val = (bl.get(field) or "").strip()
        if val:
            world_dna_parts.append(val)

    if location:
        name = (location.get("name") or "").strip()
        desc = (location.get("description") or "").strip()
        mood = (location.get("mood") or "").strip()
        bits: list[str] = []
        if name:
            bits.append(name)
        if world_dna_parts:
            # Brain world DNA enriches the DB location description
            bits.extend(world_dna_parts)
        elif desc:
            bits.append(desc)
        if mood:
            bits.append(mood + " atmosphere")
        if bits:
            raw = ", ".join(bits)
    elif world_dna_parts:
        raw = ", ".join(world_dna_parts)

    if not raw:
        env = shot.get("environment_profile") or {}
        if isinstance(env, dict):
            loc_dna = (env.get("location_dna") or "").strip()
            sf = env.get("scene_frame") or {}
            scene_loc = (sf.get("location") or "").strip() if isinstance(sf, dict) else ""
            scene_tod = (sf.get("time_of_day") or "").strip() if isinstance(sf, dict) else ""
            wa = env.get("world_assumptions") or {}
            if isinstance(wa, dict):
                season = (wa.get("season") or "").strip()
                arch = (wa.get("architecture_style") or "").strip()
                char_setting = (
                    wa.get("characteristic_setting")
                    or wa.get("domestic_setting")
                    or ""
                ).strip()
                char_time = (
                    wa.get("characteristic_time")
                    or wa.get("time_of_day")
                    or ""
                ).strip()
                active_location = scene_loc or char_setting
                active_time = scene_tod or char_time
                bits = [b for b in (loc_dna, active_location, arch, season, active_time) if b]
                if bits:
                    raw = ", ".join(bits)

    if not raw:
        return ""
    raw = _ENV_LABEL_RE.sub("", raw)
    raw = _dedupe_phrases(raw)
    return _trim(raw, 350)


def _framing_clause(shot: dict) -> str:
    fd = (shot.get("framing_directive") or "").strip()
    if fd:
        return _trim(fd, 140)
    cine = shot.get("cinematography") or {}
    if isinstance(cine, dict):
        lens = (cine.get("lens") or "").strip()
        if lens:
            return _trim(lens, 140)
    return ""


def _motion_clause(shot: dict) -> str:
    mp = (shot.get("motion_prompt") or "").strip()
    return _trim(mp, 120) if mp else ""


def _palette_clause(shot: dict) -> str:
    pal = (shot.get("color_palette") or "").strip()
    light = (shot.get("lighting_style") or "").strip()
    bits = [b for b in (light, pal) if b]
    return _trim(", ".join(bits), 180) if bits else ""


def _meaning_sentence(shot: dict) -> str:
    m = (shot.get("meaning") or "").strip()
    return _trim(m, 200) if m else ""


def _shot_event_lead_sentence(shot: dict) -> str:
    """Prefer a concrete event/action sentence over a generic mood sentence."""
    event = _event_payload(shot)
    action = _clean_text(event.get("action") or event.get("subject_action"))
    trigger = _clean_text(event.get("trigger") or event.get("trigger_event"))
    shift = _clean_text(event.get("emotional_shift"))
    contrast = _clean_text(event.get("visual_contrast"))

    parts: list[str] = []

    if action:
        parts.append(action.rstrip("."))
    if trigger:
        parts.append(f"triggered by {trigger.rstrip('.')}")
    if shift:
        parts.append(f"capturing {shift.rstrip('.')}")
    elif contrast:
        parts.append(f"carrying the tension of {contrast.rstrip('.')}")

    if not parts:
        return ""
    return _trim(", ".join(parts), 220)


def _object_interaction_clause(shot: dict) -> str:
    event = _event_payload(shot)
    obj = _clean_text(event.get("object_interaction") or event.get("object_usage"))
    if not obj:
        return ""
    return _trim(f"Object interaction: {obj}", 160)


def _environment_interaction_clause(shot: dict) -> str:
    event = _event_payload(shot)
    env = _clean_text(event.get("environment_interaction") or event.get("environment_usage"))
    if not env:
        return ""
    return _trim(f"Environment interaction: {env}", 180)


def _contrast_clause(shot: dict) -> str:
    event = _event_payload(shot)
    contrast = _clean_text(event.get("visual_contrast"))
    if not contrast:
        return ""
    return _trim(f"Visual tension: {contrast}", 180)


def _camera_plan_clause(shot: dict) -> str:
    event = _event_payload(shot)
    cp = shot.get("camera_plan") or event.get("camera_plan") or {}
    if not isinstance(cp, dict):
        return ""
    movement = (cp.get("movement") or "").strip()
    style = (cp.get("style") or "").strip()
    intensity = (cp.get("intensity") or "").strip()
    bits = [b for b in (movement, style, intensity) if b]
    if not bits:
        return ""
    return _trim("Camera behaviour: " + ", ".join(bits), 120)


_VERB_SIGNALS = re.compile(
    r"\b(walks?|stands?|sits?|holds?|lifts?|drops?|turns?|looks?|reaches?|"
    r"moves?|pauses?|stares?|leans?|touches?|pulls?|pushes?|runs?|falls?|"
    r"steps?|crosses?|waits?|watches?|raises?|bows?|presses?|"
    r"traces?|grips?|freezes?|slows?|breathes?|starts?|stops?|grabs?|"
    r"slips?|breaks?|folds?|wraps?|checks?|sweeps?|opens?|closes?|"
    r"gazes?|glances?|kneels?|collapses?|throws?|catches?|carries?|"
    r"rests?|catches?|drifts?|settles?|hangs?|floats?|fills?|casts?|"
    r"draws?|pulls?|shifts?|ripples?|glows?|lies?|leans?|faces?)\b",
    re.IGNORECASE,
)

_VERB_FALLBACKS: Dict[str, str] = {
    "face":        "turns face slightly away, eyes holding an unspoken thought",
    "body":        "pauses mid-movement, weight shifting as something catches attention",
    "environment": "light shifts across an empty space, carrying the mood of recent absence",
    "macro":       "object catches available light, drawing focus to its worn surface",
    "symbolic":    "silhouette stands at threshold between shadow and light",
}


def _has_verb(text: str) -> bool:
    """Return True if the composed text contains at least one action verb."""
    return bool(_VERB_SIGNALS.search(text))


def _inject_verb_fallback(body: str, shot: dict) -> str:
    """If body has no verb, prepend a fallback action sentence for the shot's mode."""
    if _has_verb(body):
        return body
    mode = (shot.get("expression_mode") or shot.get("llm_expression_mode") or "face").lower()
    fallback = _VERB_FALLBACKS.get(mode, _VERB_FALLBACKS["face"])
    return f"{fallback.capitalize()}. {body}".strip()


# ── Prompt-level environment interaction guarantee ────────────────────────────
# Mirrors _inject_verb_fallback: if the final body contains no spatial grounding
# (no Setting/Environment/location keyword), append a mode-appropriate phrase.
# This catches legacy/no-event shots that passed validation but never had an
# environment_usage field.

_ENV_SIGNALS = re.compile(
    r"\b(setting|environment|surround|space|room|street|courtyard|corridor|"
    r"field|forest|alley|doorway|window|ground|wall|floor|ceiling|water|"
    r"light\s+spill|shadow|horizon|landscape|interior|exterior|location|"
    r"background|architecture|texture|surface)\b",
    re.IGNORECASE,
)

_ENV_FALLBACKS: Dict[str, str] = {
    "face":        "The surrounding space holds quiet tension",
    "body":        "The character occupies and interacts with the surrounding space",
    "environment": "The space itself carries the weight of the scene",
    "macro":       "The object's detail anchors the viewer within the physical environment",
    "symbolic":    "The character's form echoes the surrounding architecture",
}


def _has_env_grounding(text: str) -> bool:
    """Return True if the text already references a setting or spatial context."""
    return bool(_ENV_SIGNALS.search(text))


def _inject_env_fallback(body: str, shot: dict) -> str:
    """If body lacks spatial grounding, append a mode-appropriate environment clause."""
    if _has_env_grounding(body):
        return body
    mode = (shot.get("expression_mode") or shot.get("llm_expression_mode") or "face").lower()
    fallback = _ENV_FALLBACKS.get(mode, _ENV_FALLBACKS["face"])
    return f"{body} {fallback}.".strip()


def compose_image_prompt(
    shot: dict,
    *,
    character: Optional[dict] = None,
    location: Optional[dict] = None,
    has_character_ref: bool = False,
    has_environment_ref: bool = False,
    user_override: Optional[str] = None,
    cine_prefix: str = "",
    brain_char: Optional[dict] = None,
    brain_loc: Optional[dict] = None,
    emotional_mode_modifier: str = "",
    project_id: Optional[str] = None,
) -> tuple[str, str]:
    """Compose a tight image prompt and matching negative prompt.

    brain_char: materializer_packet character_profile entry matched by db_id.
                Provides identity_seed, archetype, emotional_baseline,
                cultural_markers, continuity_rules.
    brain_loc:  materializer_packet location_profile entry matched by db_id.
                Provides world DNA (world, environment_type, key_textures,
                palette_anchor, architecture).
    Both fall back cleanly to DB-only behaviour when None or {}.

    emotional_mode_modifier: cinematic aesthetic string from emotional_mode_packet
        (Stage 2b Brain key).  Canonical design: callers read the Brain and pass
        the modifier here so this function stays pure and testable.  Pipeline reads
        happen in pipeline_worker._render_shot and image_generator.generate_shot_still.

    project_id: when provided AND emotional_mode_modifier is empty, the function
        self-reads `emotional_mode_packet` from Brain to derive the modifier.
        This path supports the spec-stated project_id self-read API contract.
        Callers that already hold the packet should pass emotional_mode_modifier
        directly to avoid a redundant DB round-trip.
    """
    # Spec-compliant self-read path: derive modifier from Brain when not injected.
    if not emotional_mode_modifier and project_id:
        try:
            from project_brain import ProjectBrain  # type: ignore
            import psycopg as _pg
            from psycopg.rows import dict_row as _dict_row
            import os as _os
            _db_url = _os.environ.get("DATABASE_URL", "")
            with _pg.connect(_db_url, row_factory=_dict_row) as _conn:
                _brain = ProjectBrain.load(project_id, _conn)
            if _brain.is_populated("emotional_mode_packet"):
                _emp = _brain.read("emotional_mode_packet") or {}
                emotional_mode_modifier = str(_emp.get("cinematic_modifier") or "").strip()
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "[compose_image_prompt] Brain self-read failed for project_id=%r — "
                "emotional_mode_modifier will be empty. Reason: %s",
                project_id, _exc,
            )
    bc = brain_char or {}
    identity_seed    = (bc.get("identity_seed")    or "").strip()
    archetype        = (bc.get("archetype")        or "").strip()
    emotional_base   = (bc.get("emotional_baseline") or "").strip()
    char_rules_raw   = bc.get("continuity_rules")  or []
    char_rules: list = char_rules_raw if isinstance(char_rules_raw, list) else [str(char_rules_raw)]

    # ── 1. User override path ────────────────────────────────────────────────
    if user_override and user_override.strip():
        body = _clean_text(user_override)
        body = _inject_verb_fallback(body, shot)
        body = _inject_env_fallback(body, shot)
        # Always prepend mode modifier so user-edited prompts remain mode-aware.
        effective_cine = (
            f"{emotional_mode_modifier}, {cine_prefix}".strip(", ")
            if emotional_mode_modifier and cine_prefix
            else emotional_mode_modifier or cine_prefix
        )
        prompt = _attach_envelope(
            body, effective_cine, has_character_ref, has_environment_ref,
            identity_seed=identity_seed,
            archetype=archetype,
            continuity_rules=char_rules or None,
        )
        return prompt, _build_negative(shot, location)

    parts: list[str] = []

    # 1) Lead with the strongest available shot idea.
    event_lead = _shot_event_lead_sentence(shot)
    meaning    = _meaning_sentence(shot)
    subject    = _gendered_subject(character, brain_char=brain_char)

    if event_lead:
        parts.append(event_lead)
        parts.append(f"Subject: {subject}.")
        if meaning:
            parts.append(f"Story beat: {meaning}".rstrip(".") + ".")
    elif meaning:
        parts.append(meaning)
        parts.append(f"Subject: {subject}.")
    else:
        parts.append(f"{subject.capitalize()}.")

    # 2) Emotional baseline from brain (only when no event_lead overrides tone)
    if emotional_base and not event_lead:
        parts.append(f"Emotional register: {emotional_base}.")

    # 3) Identity / wardrobe
    wardrobe = _wardrobe_clause(character)
    if wardrobe:
        parts.append(wardrobe.capitalize())

    # 4) Environment — brain world DNA + DB location
    env = _environment_clause(location, shot, brain_loc=brain_loc)
    if env:
        parts.append(f"Setting: {env}".rstrip(".") + ".")

    # 5) MM3.1 action support
    obj_clause = _object_interaction_clause(shot)
    if obj_clause:
        parts.append(obj_clause)

    env_interaction = _environment_interaction_clause(shot)
    if env_interaction:
        parts.append(env_interaction)

    contrast = _contrast_clause(shot)
    if contrast:
        parts.append(contrast)

    # 6) Framing + camera
    framing = _framing_clause(shot)
    if framing:
        parts.append(f"Framing: {framing}".rstrip(".") + ".")

    if not (cine_prefix or "").strip():
        cam_plan = _camera_plan_clause(shot)
        if cam_plan:
            parts.append(cam_plan)
        else:
            motion = _motion_clause(shot)
            if motion:
                parts.append(f"Camera: {motion}".rstrip(".") + ".")

    # 7) Lighting / palette last
    palette = _palette_clause(shot)
    if palette:
        parts.append(palette.capitalize())

    body = " ".join(p for p in parts if p)
    body = _clean_text(body)
    body = re.sub(r"\.\.+", ".", body)
    body = re.sub(r"\s+", " ", body).strip()

    body = _inject_verb_fallback(body, shot)
    body = _inject_env_fallback(body, shot)

    # Prepend emotional mode modifier so it primes the diffusion model's
    # interpretation before all other descriptors (mood-first prompting).
    if emotional_mode_modifier and emotional_mode_modifier.strip():
        body = emotional_mode_modifier.strip().rstrip(".") + ". " + body

    prompt = _attach_envelope(
        body, cine_prefix, has_character_ref, has_environment_ref,
        identity_seed=identity_seed,
        archetype=archetype,
        continuity_rules=char_rules or None,
    )
    return prompt, _build_negative(shot, location)


def _attach_envelope(
    body: str,
    cine_prefix: str,
    has_character_ref: bool,
    has_environment_ref: bool,
    identity_seed: str = "",
    archetype: str = "",
    continuity_rules: Optional[list] = None,
) -> str:
    """Attach cinematography prefix, continuity cues, and quality boosters.

    When brain identity data is supplied (identity_seed, archetype,
    continuity_rules), it is embedded in the continuity cue so the model
    stays anchored even when no reference image is available (FLUX mode).
    """
    continuity_cues: list[str] = []

    if has_character_ref:
        anchor = ""
        if identity_seed:
            anchor = f" The character is: {identity_seed}"
            if archetype:
                anchor += f" ({archetype})"
            anchor += "."
        continuity_cues.append(
            "Match the exact face, complexion, and skin tone of the character "
            "reference image — same identity across all shots."
            + anchor +
            " Clothing and jewelry should match the scene context described in the prompt."
        )
    elif identity_seed:
        # No ref image (FLUX text-only path): text anchor is all we have.
        anchor = f"Character identity: {identity_seed}"
        if archetype:
            anchor += f" ({archetype})"
        anchor += ". Maintain this appearance consistently across all shots."
        continuity_cues.append(anchor)

    if has_environment_ref:
        continuity_cues.append(
            "Match the lighting, color palette, materials, and architectural "
            "details of the established environment reference plate."
        )

    # Global character continuity rules from materializer
    if continuity_rules:
        rules_str = "; ".join(
            str(r).strip() for r in continuity_rules if r
        )
        if rules_str:
            continuity_cues.append(f"Continuity rules: {rules_str}.")

    cues = " ".join(continuity_cues)

    cine = (cine_prefix or "").strip()
    if cine and not cine.endswith("."):
        cine += "."

    pieces = [cine, body, cues, QUALITY_BOOSTERS]
    full = " ".join(p for p in pieces if p).strip()
    full = re.sub(r"\s+", " ", full)

    if len(full) <= 1100:
        return full
    cut = full[:1100]
    i = cut.rfind(". ")
    if i > 700:
        return cut[: i + 1]
    return cut.rstrip(" .,;:") + "."
