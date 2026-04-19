"""Tight, model-friendly prompt composer for per-shot still generation.

Replaces the wall-of-text styled_visual_prompt (which dumps director's
notes, story spine, ambiguity handling, etc. into the prompt and is
truncated mid-sentence at 1800 chars) with a focused 60-180 word
visual description that diffusion models can actually attend to.

The composer is deterministic and works directly off the structured
fields produced by the upstream engines plus the linked character /
location records. The full styled_visual_prompt remains in the DB for
inspection, UI display, and editing — but the model only ever sees
this clean version unless the user has hand-edited the prompt
(``user_override``), in which case the user's verbatim text is used
with the cinematography prefix and quality boosters reattached.

Returns:
    A tuple ``(prompt, negative_prompt)`` ready for FAL.
"""

from __future__ import annotations

import re
from typing import Optional


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

# ── Phrases to strip — these are director instructions, not visual prose
_INSTRUCTION_PATTERNS = [
    r"Maintain (?:strict )?character continuity[^.]*\.",
    r"Performance:[^.]*\.",
    r"Function:[^.]*\.",
    r"Repetition logic:[^.]*\.",
    r"Ambiguity handling:[^.]*\.",
    r"Relevant ambiguity notes:[^.]*\.",
    r"Source format awareness:[^.]*\.",
    r"Hard restrictions[^:]*:[^.]*\.",
    r"Visual constraints:[^.]*\.",
    r"Continuity:[^.]*\.",
    r"Motif handling:[^.]*\.",
    r"Cultural subtext to preserve:[^.]*\.",
    r"Why this song exists:[^.]*\.",
    r"Spine anchor:[^.]*\.",
    r"Dramatic premise:[^.]*\.",
    r"Treatment:[^.]*\.",
    r"Central metaphor[^:]*:[^.]*\.",
    r"Arc beat[^:]*:[^.]*\.",
    r"Style notes:[^.]*\.",
    r"Rendering notes:[^.]*\.",
    r"Transition behavior:[^.]*\.",
    r"Repeat status:[^.]*\.",
    r"Speaker identity context:[^.]*\.",
    r"Addressee context:[^.]*\.",
    r"Intensity:[^.]*\.",
]

_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)


def _trim(s: Optional[str], limit: int = 140) -> str:
    """Sentence-aware trim. Returns at most ``limit`` chars, ending on a
    sentence/clause boundary where possible."""
    if not s:
        return ""
    s = " ".join(s.split())
    if len(s) <= limit:
        return s.rstrip(" .,;:") + "."
    cut = s[:limit]
    for sep in (". ", "; ", ", "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            return cut[:i].rstrip(" .,;:") + "."
    return cut.rstrip(" .,;:") + "."


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
        # Skip the gender word entirely — let PuLID + ref image carry it.
        parts.append("person")

    role = (character.get("role") or "").strip()
    if role:
        parts.append(f"({role.lower()})")

    return " ".join(parts).strip() or "a person"


def _wardrobe_clause(character: Optional[dict]) -> str:
    if not character:
        return ""
    # Prefer per-shot scene-appropriate wardrobe (from wardrobe_engine) over
    # the single global default stored on the character record.  This is how
    # real filmmaking works: the face stays the same across shots but the
    # clothes change with the scene (wedding → home → field → performance).
    wardrobe = (
        character.get("wardrobe_override")
        or character.get("wardrobe")
        or ""
    ).strip()
    grooming = (character.get("grooming") or "").strip()

    # Normalise: strip LLM "The [subject] wears " prefix so we can re-attach
    # a clean "wearing" ourselves, avoiding "wearing The woman wears…"
    _WEARS_PREFIX = re.compile(
        r"^(?:the\s+\w+(?:\s+\w+)?\s+wears?\s+|wearing\s+)",
        re.IGNORECASE,
    )
    wardrobe = _WEARS_PREFIX.sub("", wardrobe).strip()

    bits = [b for b in (wardrobe, grooming) if b]
    if not bits:
        return ""
    return _trim("wearing " + ", ".join(bits), 160)


_ENV_LABEL_RE = re.compile(
    r"\b(?:world\s*dna|location\s*dna|geography|architecture(?:\s*style)?|"
    r"domestic\s*setting|characteristic\s*setting|characteristic\s*time|"
    r"time\s*of\s*day|season)\s*:\s*",
    re.IGNORECASE,
)


def _dedupe_phrases(s: str) -> str:
    """Remove case-insensitive duplicate comma/period-separated phrases
    while preserving order. Avoids 'Punjabi village, ..., Punjabi village'."""
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
    """Build a concrete environment description.

    Prefers the linked location plate's name + description, falls back to
    the shot's environment_profile when no location is linked. Strips
    upstream field labels ('World DNA:', 'geography:', etc.) and
    deduplicates repeated phrases."""
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
            # Prefer scene_frame (shot-specific location) over global characteristic_setting
            sf = env.get("scene_frame") or {}
            scene_loc = (sf.get("location") or "").strip() if isinstance(sf, dict) else ""
            scene_tod = (sf.get("time_of_day") or "").strip() if isinstance(sf, dict) else ""
            wa = env.get("world_assumptions") or {}
            if isinstance(wa, dict):
                season = (wa.get("season") or "").strip()
                arch = (wa.get("architecture_style") or "").strip()
                # Support both new names and legacy pre-rename names
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
                # Scene-specific values take priority over cultural defaults
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
    """The shot's actual story beat — must lead the prompt."""
    m = (shot.get("meaning") or "").strip()
    return _trim(m, 200) if m else ""


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

    Args:
        shot: The styled timeline shot dict.
        character: The linked character row (with ethnicity, gender,
            age_range, wardrobe, grooming, role, etc.) or None.
        location: The linked location row (name, description, mood) or None.
        has_character_ref: True if a character reference image will be
            passed to the model. Adds a continuity cue saying "the same
            person as the reference image" so non-PuLID layers (style,
            wardrobe, complexion) stay consistent across all shots.
        has_environment_ref: True if an env reference will be passed.
            Adds a parallel continuity cue for setting consistency.
        user_override: If the user hand-edited the prompt in the UI,
            their verbatim text is used (with cine_prefix + boosters
            reattached). The composer is bypassed entirely.
        cine_prefix: Optional cinematography prefix from
            ``cinematography_engine.lens_clause`` to prepend.

    Returns:
        ``(prompt, negative_prompt)`` — the prompt is sentence-aware
        trimmed to roughly ~1100 chars, leaving headroom for cine_prefix
        and continuity cues.
    """
    # ── 1. User override path: keep the user's verbatim text ────────
    if user_override and user_override.strip():
        body = user_override.strip()
        # Still strip director instructions if the user pasted them in.
        body = _INSTRUCTION_RE.sub("", body).strip()
        body = re.sub(r"\s+", " ", body)
        prompt = _attach_envelope(body, cine_prefix, has_character_ref,
                                  has_environment_ref)
        return prompt, DEFAULT_NEGATIVE

    # ── 2. Compose from structured fields ───────────────────────────
    parts: list[str] = []

    # Subject sentence (leads the prompt — most-attended position)
    meaning = _meaning_sentence(shot)
    subject = _gendered_subject(character)
    if meaning:
        parts.append(meaning)
        parts.append(f"Subject: {subject}.")
    else:
        parts.append(f"{subject.capitalize()}.")

    # Wardrobe / grooming (drives identity-consistent appearance)
    wardrobe = _wardrobe_clause(character)
    if wardrobe:
        parts.append(wardrobe.capitalize())

    # Environment
    env = _environment_clause(location, shot)
    if env:
        parts.append(f"Setting: {env}".rstrip(".") + ".")

    # Framing + lens
    framing = _framing_clause(shot)
    if framing:
        parts.append(f"Framing: {framing}".rstrip(".") + ".")

    # Motion — only emit when cine_prefix is empty, otherwise the
    # cinematography rig already encodes movement + lens + DoF and
    # repeating it doubles the model's signal toward the same look
    # (e.g. "shallow depth of field" appearing twice in every prompt).
    if not (cine_prefix or "").strip():
        motion = _motion_clause(shot)
        if motion:
            parts.append(f"Camera: {motion}".rstrip(".") + ".")

    # Palette + lighting
    palette = _palette_clause(shot)
    if palette:
        parts.append(palette.capitalize())

    body = " ".join(p for p in parts if p)
    body = re.sub(r"\.\.+", ".", body)
    body = re.sub(r"\s+", " ", body).strip()

    prompt = _attach_envelope(body, cine_prefix, has_character_ref,
                              has_environment_ref)
    return prompt, DEFAULT_NEGATIVE


def _attach_envelope(
    body: str,
    cine_prefix: str,
    has_character_ref: bool,
    has_environment_ref: bool,
) -> str:
    """Attach cinematography prefix, continuity cues, and quality boosters.

    The continuity cues are the key piece for cross-shot consistency:
    when a reference image is being passed to the model (PuLID for
    character, image-to-image for environment), the text prompt
    explicitly tells the model to match that reference's appearance,
    wardrobe, complexion, lighting, and palette. Without these cues,
    the prompt's other descriptors can pull the output away from the
    reference.
    """
    continuity_cues: list[str] = []
    if has_character_ref:
        # Lock FACE and COMPLEXION only — NOT wardrobe.
        # The reference plate establishes the character's identity (face,
        # skin tone, bone structure). Clothing and jewelry must be driven by
        # the per-shot wardrobe description so the character's look changes
        # between scenes (casual courtyard ≠ wedding ceremony ≠ wheat-field
        # walk), exactly as real film costume design works.
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

    # Order: cinematography → body → continuity cues → quality boosters.
    pieces = [cine, body, cues, QUALITY_BOOSTERS]
    full = " ".join(p for p in pieces if p).strip()
    full = re.sub(r"\s+", " ", full)

    # Sentence-aware trim to ~1100 chars (well under FAL's effective
    # attention window, leaving room for the cues and boosters).
    if len(full) <= 1100:
        return full
    cut = full[:1100]
    i = cut.rfind(". ")
    if i > 700:
        return cut[: i + 1]
    return cut.rstrip(" .,;:") + "."
