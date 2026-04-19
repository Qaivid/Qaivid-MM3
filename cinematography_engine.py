"""Cinematography Engine — deterministic per-shot rig + motion derivation.

Adds a structured `cinematography` block to every storyboard shot so the
downstream stills, video, and continuity prompts share a single canonical
camera taxonomy instead of free-form prose.

The engine is intentionally non-LLM: given (shot, context_packet,
style_profile) it always returns the same block, and gracefully degrades
(returns ``None`` for legacy shots that lack the inputs).

Public surface:
  - ``RIGS``                     — canonical rig vocabulary
  - ``EMOTION_TO_RIG``           — emotion → ranked rig list
  - ``derive(shot, ctx, sp)``    — return the cinematography block (or None)
  - ``motion_prompt_from_block(block)`` — short Kling-friendly motion line
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---- Canonical rig vocabulary ----------------------------------------------
RIGS: tuple[str, ...] = (
    "tripod",
    "gimbal",
    "steadicam",
    "dolly",
    "crane",
    "drone",
    "vehicle",
    "handheld",
)

# ---- Emotion → rig affinity (ranked, first match wins) ---------------------
# Keys are *substrings* matched against the shot's emotional signal.
EMOTION_TO_RIG: Dict[str, List[str]] = {
    "grief":      ["tripod", "dolly", "steadicam"],
    "longing":    ["dolly", "tripod", "steadicam"],
    "tender":     ["steadicam", "dolly", "tripod"],
    "intimate":   ["steadicam", "tripod", "dolly"],
    "calm":       ["tripod", "dolly", "steadicam"],
    "still":      ["tripod", "dolly", "steadicam"],
    "reverent":   ["crane", "tripod", "dolly"],
    "awe":        ["crane", "drone", "dolly"],
    "wonder":     ["crane", "drone", "steadicam"],
    "anger":      ["handheld", "gimbal", "vehicle"],
    "rage":       ["handheld", "gimbal", "vehicle"],
    "tension":    ["gimbal", "handheld", "steadicam"],
    "anxious":    ["handheld", "gimbal", "steadicam"],
    "panic":      ["handheld", "gimbal", "vehicle"],
    "joy":        ["gimbal", "steadicam", "drone"],
    "celebrate":  ["gimbal", "drone", "steadicam"],
    "triumph":    ["crane", "drone", "gimbal"],
    "release":    ["crane", "drone", "gimbal"],
    "memory":     ["dolly", "steadicam", "tripod"],
    "nostalg":    ["dolly", "steadicam", "tripod"],
    "epic":       ["crane", "drone", "dolly"],
    "kinetic":    ["gimbal", "vehicle", "handheld"],
    "movement":   ["vehicle", "gimbal", "steadicam"],
    "drift":      ["dolly", "steadicam", "drone"],
}

_DEFAULT_RANK: List[str] = ["dolly", "steadicam", "tripod", "gimbal", "crane",
                            "handheld", "drone", "vehicle"]

# ---- Style preset rig affinities (cinematic_style id → ranked rigs) --------
STYLE_RIG_AFFINITY: Dict[str, List[str]] = {
    "cinematic_natural":   ["dolly", "steadicam", "tripod", "crane"],
    "noir_dramatic":       ["dolly", "tripod", "crane", "steadicam"],
    "soft_poetic":         ["tripod", "dolly", "steadicam"],
    "vibrant_bold":        ["gimbal", "drone", "steadicam", "crane"],
    "vintage_grain":       ["tripod", "dolly", "handheld"],
    "monochrome":          ["tripod", "dolly", "steadicam"],
    "arthouse_minimalist": ["tripod", "dolly", "crane"],
}

# ---- Expression-mode framing → rig nudge -----------------------------------
MODE_RIG_BIAS: Dict[str, List[str]] = {
    "face":        ["tripod", "dolly", "steadicam"],
    "body":        ["steadicam", "gimbal", "dolly"],
    "environment": ["crane", "drone", "dolly"],
    "symbolic":    ["dolly", "tripod", "crane"],
    "macro":       ["tripod", "dolly"],
}

# ---- Lens recommendations per rig + mode -----------------------------------
# Default lens used only when the framing_directive provides no usable cue.
LENS_FOR_MODE: Dict[str, str] = {
    "face":        "50mm portrait, medium depth",
    "body":        "50mm natural, medium depth",
    "environment": "24mm wide, deep focus",
    "symbolic":    "85mm short-tele, shallow stop",
    "macro":       "100mm macro, very shallow",
}

# Framing-directive keyword → lens. Specific patterns first; the first
# matching keyword wins. This breaks the "every face shot is 85mm/shallow
# DoF" monoculture by letting framing cues drive lens choice.
LENS_BY_FRAMING: List[tuple[tuple[str, ...], str]] = [
    (("extreme close-up", "macro detail", "tear-line", "eyes and brow"),
     "100mm macro, very shallow depth of field"),
    (("tight facial", "tight portrait", "tight close"),
     "85mm portrait, shallow depth of field"),
    (("soft pull-focus", "rack focus"),
     "50mm with rack focus, selective depth"),
    (("medium close-up", "two-thirds", "shoulder", "head and shoulders"),
     "50mm portrait, medium depth of field"),
    (("side profile", "profile"),
     "50mm portrait, medium-deep depth of field"),
    (("over-the-shoulder", "ots"),
     "35mm classic, medium-deep depth"),
    (("medium shot", "waist-up", "mid shot"),
     "35mm classic, deep enough background"),
    (("wide", "establishing", "landscape", "vista", "aerial"),
     "24mm wide, deep focus"),
    (("low angle", "hero shot"),
     "28mm wide hero, deep focus"),
    (("high angle", "top-down", "overhead"),
     "35mm classic, deep focus"),
    (("symbol", "object", "still life"),
     "85mm short-tele, shallow stop"),
]


def _pick_lens(mode: str, framing_directive: str) -> str:
    fd = (framing_directive or "").lower()
    for kws, lens in LENS_BY_FRAMING:
        if any(kw in fd for kw in kws):
            return lens
    return LENS_FOR_MODE.get(mode, "50mm natural, medium depth")


# ---- Direction templates (rig, mode) → motion direction --------------------
# When multiple shots land on the same (rig, mode), the engine cycles through
# the available directions using ``shot_index`` so seven dolly+face shots in a
# row don't all "slow push-in".
_DIRECTION_VARIANTS: Dict[tuple[str, str], List[str]] = {
    ("dolly", "face"):        ["slow push-in", "slow pull-back", "slight lateral arc"],
    ("dolly", "body"):        ["slow push-in", "lateral track", "pull-back reveal"],
    ("dolly", "environment"): ["lateral track left→right", "slow pull-back reveal", "diagonal track-in"],
    ("dolly", "symbolic"):    ["slow push-in to object", "slow pull-back from object", "circular dolly arc"],
    ("dolly", "macro"):       ["very slow push-in", "tiny lateral drift"],
    ("steadicam", "face"):    ["floating follow", "gentle orbit", "slow lateral float"],
    ("steadicam", "body"):    ["floating follow", "tracking pace", "slow orbit"],
    ("steadicam", "environment"): ["slow drift through space", "wandering reveal", "lateral float-by"],
    ("tripod", "face"):       ["locked frame, micro breath only", "locked frame, slight rack focus", "locked frame with subject motion"],
    ("tripod", "body"):       ["locked frame", "locked frame with subtle zoom"],
    ("tripod", "environment"):["locked wide, ambient motion in frame", "locked frame, slow internal action"],
    ("gimbal", "face"):       ["smooth orbit", "slow drift around subject"],
    ("gimbal", "body"):       ["fluid follow", "smooth orbit"],
    ("gimbal", "environment"):["smooth lateral travel", "low fluid sweep"],
    ("crane", "face"):        ["descend onto subject", "rising over subject"],
    ("crane", "environment"): ["rising tilt-up reveal", "descending wide approach", "boom-up over scene"],
    ("drone", "environment"): ["high pull-back reveal", "low fly-in across landscape", "ascending vertical reveal"],
    ("drone", "face"):        ["low fly-in toward subject", "circling drone arc"],
    ("handheld", "face"):     ["loose follow", "shaky push-in"],
    ("handheld", "body"):     ["shaky follow", "verite handheld"],
    ("vehicle", "environment"):["tracking alongside motion", "parallel travel shot"],
}


def _direction_for(rig: str, mode: str, intensity: float,
                   shot_index: Optional[int] = None) -> str:
    variants = _DIRECTION_VARIANTS.get((rig, mode))
    if variants:
        if shot_index is not None and len(variants) > 1:
            return variants[int(shot_index) % len(variants)]
        return variants[0]
    # Fallback for combos not in the variants table.
    if rig == "tripod":
        return "locked frame, micro breath only"
    if rig == "dolly":
        return "slow push-in" if intensity < 0.6 else "fast push-in"
    if rig == "steadicam":
        return "floating follow"
    if rig == "gimbal":
        return "smooth orbit" if intensity < 0.7 else "whip orbit + push"
    if rig == "crane":
        return "rising tilt-up"
    if rig == "drone":
        return "high pull-back reveal"
    if rig == "vehicle":
        return "tracking alongside motion"
    if rig == "handheld":
        return "shaky push-in" if intensity >= 0.7 else "loose follow"
    return "subtle drift"


def _speed_for(intensity: float) -> str:
    if intensity >= 0.8:
        return "fast"
    if intensity >= 0.55:
        return "moderate"
    return "slow"


def _intensity_for(intensity: float) -> str:
    if intensity >= 0.8:
        return "high"
    if intensity >= 0.5:
        return "medium"
    return "low"


def _emotion_signal(shot: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    parts = [
        str(shot.get("meaning") or ""),
        str(shot.get("function") or ""),
        str(shot.get("expression_mode") or ""),
        str(ctx.get("emotional_arc") or ""),
    ]
    spk = ctx.get("speaker") or {}
    if isinstance(spk, dict):
        parts.append(str(spk.get("emotional_state") or ""))
    return " ".join(parts).lower()


def _emotion_rank(signal: str) -> List[str]:
    for kw, ranked in EMOTION_TO_RIG.items():
        if kw in signal:
            return ranked
    return list(_DEFAULT_RANK)


def _style_rank(style_profile: Dict[str, Any]) -> List[str]:
    cin = (style_profile or {}).get("cinematic") or {}
    sid = cin.get("id") or (style_profile or {}).get("preset")
    if isinstance(sid, str) and sid in STYLE_RIG_AFFINITY:
        return list(STYLE_RIG_AFFINITY[sid])
    return list(_DEFAULT_RANK)


def _mode_rank(mode: str) -> List[str]:
    return list(MODE_RIG_BIAS.get(mode, _DEFAULT_RANK))


def _pick_rig(emotion_rank: List[str], style_rank: List[str],
              mode_rank: List[str], director_note: str = "",
              prev_rig: Optional[str] = None,
              recent_rigs: Optional[List[str]] = None,
              emotion_signal: str = "",
              mode: str = "") -> tuple[str, str]:
    """Score-blend the three rankings; return (chosen_rig, justification).

    ``prev_rig`` and ``recent_rigs`` apply an anti-repeat penalty so a long
    run of same-mode shots doesn't all collapse onto the same rig (which
    is what produces the "every shot is a dolly push-in" monotony).

    ``emotion_signal`` and ``mode`` are used to build a plain-language
    justification sentence shown to the user in the storyboard UI."""
    note = (director_note or "").lower()
    score: Dict[str, float] = {r: 0.0 for r in RIGS}
    for i, rig in enumerate(emotion_rank):
        if rig in score:
            score[rig] += (len(emotion_rank) - i) * 1.5
    for i, rig in enumerate(style_rank):
        if rig in score:
            score[rig] += (len(style_rank) - i) * 1.0
    for i, rig in enumerate(mode_rank):
        if rig in score:
            score[rig] += (len(mode_rank) - i) * 1.2

    # Director-note keyword boosts
    boost_map = {
        "handheld":  ["handheld", "verite", "vérité", "raw", "doc-style"],
        "drone":     ["aerial", "drone", "epic wide"],
        "crane":     ["crane", "epic", "majestic"],
        "tripod":    ["locked", "tarkovsky", "static", "still"],
        "gimbal":    ["kinetic", "fluid", "music video"],
        "steadicam": ["floating", "dreamlike", "smooth"],
        "vehicle":   ["car chase", "tracking", "in motion"],
        "dolly":     ["push-in", "dolly", "intimate slow"],
    }
    for rig, kws in boost_map.items():
        if any(kw in note for kw in kws):
            score[rig] += 6.0

    # Anti-repeat: penalize the immediately previous rig and any rig that
    # has dominated the last few shots. Penalty is bounded so a strongly
    # justified rig can still re-win when needed.
    if prev_rig and prev_rig in score:
        score[prev_rig] -= 3.0
    if recent_rigs:
        from collections import Counter
        counts = Counter(recent_rigs[-4:])
        for rig, n in counts.items():
            if rig in score and n >= 2:
                score[rig] -= 1.5 * (n - 1)

    chosen = max(score.items(), key=lambda kv: kv[1])[0]

    # Plain-language justification sentence for the storyboard UI.
    _rig_labels: Dict[str, str] = {
        "tripod":    "A locked tripod",
        "dolly":     "A dolly move",
        "steadicam": "Steadicam",
        "gimbal":    "A gimbal",
        "crane":     "A crane shot",
        "drone":     "A drone shot",
        "handheld":  "Handheld",
        "vehicle":   "A vehicle-mounted shot",
    }
    rig_label = _rig_labels.get(chosen, chosen.capitalize())

    has_director_note = bool(note and any(
        kw in note for kws in boost_map.values() for kw in kws
    ))
    is_emotion_top = bool(emotion_rank and emotion_rank[0] == chosen)
    is_emotion_near = not is_emotion_top and chosen in emotion_rank[:2]
    is_style_top = bool(style_rank and style_rank[0] == chosen)
    is_mode_top = bool(mode_rank and mode_rank[0] == chosen)

    matched_emotion = next(
        (kw for kw in EMOTION_TO_RIG if kw in (emotion_signal or "")), None
    )
    # Map internal stem keys to readable display labels
    _emotion_display: Dict[str, str] = {
        "nostalg":   "nostalgic",
        "celebrate": "celebratory",
        "kinetic":   "kinetic energy",
        "movement":  "movement",
        "drift":     "drifting",
        "still":     "stillness",
        "tender":    "tender",
        "intimate":  "intimate",
        "reverent":  "reverent",
        "anxious":   "anxious",
        "triumph":   "triumphant",
        "release":   "release",
    }
    emotion_display = _emotion_display.get(matched_emotion, matched_emotion) if matched_emotion else None

    if has_director_note:
        justification = f"{rig_label} — the director's note called for this choice."
    elif is_emotion_top and emotion_display:
        justification = (
            f"{rig_label} captures the {emotion_display} tone best"
            f" — it's the strongest match for that emotional weight."
        )
    elif is_emotion_near and emotion_display:
        justification = (
            f"{rig_label} complements the {emotion_display} mood in this shot."
        )
    elif is_emotion_top:
        justification = (
            f"{rig_label} is the best match for the emotional tone of this shot."
        )
    elif is_style_top:
        justification = (
            f"{rig_label} fits the visual style chosen for this project."
        )
    elif is_mode_top and mode:
        justification = (
            f"{rig_label} is well-suited to {mode} shots."
        )
    else:
        justification = (
            f"{rig_label} is a balanced choice for this shot."
        )

    return chosen, justification


def derive(
    shot: Dict[str, Any],
    context_packet: Optional[Dict[str, Any]] = None,
    style_profile: Optional[Dict[str, Any]] = None,
    prev_block: Optional[Dict[str, Any]] = None,
    recent_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the structured cinematography block for one shot.

    Returns ``None`` if shot lacks an expression_mode (legacy / malformed).

    ``prev_block`` and ``recent_blocks`` (if provided) feed the anti-repeat
    rig penalty and the direction-cycling so consecutive same-mode shots
    don't all collapse onto the same lens / rig / direction. Backward
    compatible — both default to None.
    """
    if not isinstance(shot, dict):
        return None
    mode = (shot.get("expression_mode") or "").lower()
    if mode not in ("face", "body", "environment", "symbolic", "macro"):
        return None

    ctx = context_packet or {}
    sp = style_profile or {}

    # Backward-compat guard (Task #69): legacy projects that have no locked
    # Creative Brief should fall through with no rig block — only opt-in once
    # the user has approved a brief variant.
    chosen_brief = ((ctx.get("creative_brief") or {}).get("chosen") or {})
    if not isinstance(chosen_brief, dict) or not chosen_brief:
        return None

    director_note = str(chosen_brief.get("director_note") or "")

    intensity_raw = shot.get("intensity")
    try:
        intensity = float(intensity_raw) if intensity_raw is not None else 0.5
    except (TypeError, ValueError):
        intensity = 0.5
    intensity = max(0.0, min(1.0, intensity))

    signal = _emotion_signal(shot, ctx)
    e_rank = _emotion_rank(signal)
    s_rank = _style_rank(sp)
    m_rank = _mode_rank(mode)

    prev_rig = (prev_block or {}).get("rig") if isinstance(prev_block, dict) else None
    recent_rigs = [
        (b or {}).get("rig") for b in (recent_blocks or [])
        if isinstance(b, dict) and (b or {}).get("rig")
    ]

    rig, justification = _pick_rig(e_rank, s_rank, m_rank, director_note,
                                   prev_rig=prev_rig, recent_rigs=recent_rigs,
                                   emotion_signal=signal, mode=mode)

    # Shot-index drives direction cycling so seven dolly+face shots in a
    # row rotate through push-in / pull-back / lateral-arc instead of all
    # being "slow push-in".
    shot_index = shot.get("shot_index") or shot.get("timeline_index")
    direction = _direction_for(rig, mode, intensity, shot_index=shot_index)
    speed = _speed_for(intensity)
    intensity_label = _intensity_for(intensity)
    lens = _pick_lens(mode, str(shot.get("framing_directive") or ""))

    return {
        "rig": rig,
        "direction": direction,
        "speed": speed,
        "lens": lens,
        "intensity": intensity_label,
        "justification": justification,
    }


def motion_prompt_from_block(block: Optional[Dict[str, Any]]) -> str:
    """Compact Kling-friendly camera-motion string from a cinematography block."""
    if not isinstance(block, dict):
        return ""
    rig = block.get("rig")
    direction = block.get("direction")
    if not rig or not direction:
        return ""
    speed = block.get("speed") or "moderate"
    lens = block.get("lens") or ""
    parts = [f"{speed} {rig} — {direction}"]
    if lens:
        parts.append(lens)
    out = ", ".join(parts)
    return out[:240]


def lens_clause(block: Optional[Dict[str, Any]]) -> str:
    """One-line lens/framing clause for stills prompts."""
    if not isinstance(block, dict):
        return ""
    lens = block.get("lens") or ""
    rig = block.get("rig") or ""
    direction = block.get("direction") or ""
    if not (lens or rig):
        return ""
    return f"Cinematography: {rig} rig, {direction}, shot on {lens}.".strip()
