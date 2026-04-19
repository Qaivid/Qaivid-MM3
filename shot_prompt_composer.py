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
    "shot on Arri Alexa, 35mm anamorphic, fine film grain, "
    "professional cinematography, photorealistic, sharp focus, "
    "ultra-detailed, natural skin texture"
)

DEFAULT_NEGATIVE = (
    "blurry, watermark, text overlay, logo, deformed anatomy, "
    "extra fingers, mutated hands, oversaturated cartoon, low quality, "
    "jpeg artifacts, oversharpened, plastic skin, cgi look, "
    "amateur photography, bad composition"
)

# ── Phrases to strip — pipeline-internal directives that should not reach
#    the image model.  Cinematic intelligence fields (Function, Performance,
#    Arc beat, Motif handling) are intentionally KEPT so the model understands
#    the shot's intent. ─────────────────────────────────────────────────────
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
    # NOTE: Performance, Function, Arc beat, Central metaphor, Dramatic premise,
    # and Motif handling are intentionally NOT stripped — they give the image
    # model context about what the shot must convey.
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


def _gendered_subject(character: Optional[dict]) -> str:
    """Build a concrete, gendered subject phrase. Avoids the dreaded
    'Adult Unspecified figure' fallback that breaks Flux identity."""
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


def _environment_clause(location: Optional[dict], shot: dict) -> str:
    raw = ""
    if location:
        name = (location.get("name") or "").strip()
        desc = (location.get("description") or "").strip()
        mood = (location.get("mood") or "").strip()
        bits: list[str] = []
        if name:
            bits.append(name)
        if desc:
            bits.append(desc)
        if mood:
            bits.append(mood + " atmosphere")
        if bits:
            raw = ", ".join(bits)

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
    return _trim(raw, 220)


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


def compose_image_prompt(
    shot: dict,
    *,
    character: Optional[dict] = None,
    location: Optional[dict] = None,
    has_character_ref: bool = False,
    has_environment_ref: bool = False,
    user_override: Optional[str] = None,
    cine_prefix: str = "",
) -> tuple[str, str]:
    """Compose a tight image prompt and matching negative prompt.

    Existing MM3 signature preserved.
    """
    # ── 1. User override path: keep user's text, but still clean hard instructions ─
    if user_override and user_override.strip():
        body = _clean_text(user_override)
        prompt = _attach_envelope(
            body,
            cine_prefix,
            has_character_ref,
            has_environment_ref,
        )
        return prompt, DEFAULT_NEGATIVE

    parts: list[str] = []

    # 1) Lead with the strongest available shot idea.
    event_lead = _shot_event_lead_sentence(shot)
    meaning = _meaning_sentence(shot)
    subject = _gendered_subject(character)

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

    # 2) Identity / wardrobe
    wardrobe = _wardrobe_clause(character)
    if wardrobe:
        parts.append(wardrobe.capitalize())

    # 3) Environment
    env = _environment_clause(location, shot)
    if env:
        parts.append(f"Setting: {env}".rstrip(".") + ".")

    # 4) MM3.1 action support
    obj_clause = _object_interaction_clause(shot)
    if obj_clause:
        parts.append(obj_clause)

    env_interaction = _environment_interaction_clause(shot)
    if env_interaction:
        parts.append(env_interaction)

    contrast = _contrast_clause(shot)
    if contrast:
        parts.append(contrast)

    # 5) Framing + camera
    framing = _framing_clause(shot)
    if framing:
        parts.append(f"Framing: {framing}".rstrip(".") + ".")

    # If there is no explicit cine_prefix from cinematography_engine, still pass motion/camera hints.
    if not (cine_prefix or "").strip():
        cam_plan = _camera_plan_clause(shot)
        if cam_plan:
            parts.append(cam_plan)
        else:
            motion = _motion_clause(shot)
            if motion:
                parts.append(f"Camera: {motion}".rstrip(".") + ".")

    # 6) Lighting / palette last
    palette = _palette_clause(shot)
    if palette:
        parts.append(palette.capitalize())

    body = " ".join(p for p in parts if p)
    body = _clean_text(body)
    body = re.sub(r"\.\.+", ".", body)
    body = re.sub(r"\s+", " ", body).strip()

    # Verb validator: every still prompt must contain a concrete action verb.
    # If none is found, a mode-appropriate fallback action is prepended so
    # Flux always receives a verb-led description rather than a mood noun.
    body = _inject_verb_fallback(body, shot)

    prompt = _attach_envelope(
        body,
        cine_prefix,
        has_character_ref,
        has_environment_ref,
    )
    return prompt, DEFAULT_NEGATIVE


def _attach_envelope(
    body: str,
    cine_prefix: str,
    has_character_ref: bool,
    has_environment_ref: bool,
) -> str:
    """Attach cinematography prefix, continuity cues, and quality boosters."""
    continuity_cues: list[str] = []
    if has_character_ref:
        continuity_cues.append(
            "Match the exact face, complexion, and skin tone of the character "
            "reference image — same identity across all shots. "
            "Clothing and jewelry should match the scene context described in the prompt."
        )
    if has_environment_ref:
        continuity_cues.append(
            "Match the lighting, color palette, materials, and architectural "
            "details of the established environment reference plate."
        )
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
