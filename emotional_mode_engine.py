"""
emotional_mode_engine.py — Stage 2b: Emotional Mode Classification

Reads context_packet (Stage 2) from the Brain, classifies the project's
primary + secondary emotional mode, and writes emotional_mode_packet to
the Brain.  Every downstream stage reads from the Brain — nothing is
hardcoded in this module's callers.

7 Emotional Modes
-----------------
romantic | sad_loss | nostalgic | hopeful | angry_intense |
spiritual_reflective | energetic_celebration

Brain namespace written: emotional_mode_packet
  {
    primary_mode:         str   — canonical mode id
    secondary_mode:       str   — second-strongest mode id
    primary_weight:       float — 0.7 (always)
    secondary_weight:     float — 0.3 (always)
    pacing_profile:       dict  — effective blended pacing values
    shot_intensity_biases:dict  — blended variety-target fractions
    camera_movement_profile: dict
    cinematic_modifier:   str   — prepended to shot prompts as flavour text only
    style_modifier_injection: dict — movement + atmosphere_note only (no lighting)
    classifier_method:    str   — "llm" | "keyword_fallback"
    raw_scores:           dict  — per-mode float score from classifier
  }

NOTE: Emotional mode governs only the BEHAVIOURAL register of the scene —
pacing, camera movement, shot intensity, and acting energy. It does NOT
control which cinematic or production style is used. Style selection is
entirely the user's choice and is never overridden by mode constraints.
A sad person can stand in a sunlit field; a celebration can happen in a
dim room. Location, visual aesthetic, and environment are not mode-owned.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 7-MODE REGISTRY — pure data, no logic
# ---------------------------------------------------------------------------
_MODE_REGISTRY: Dict[str, Dict[str, Any]] = {

    "romantic": {
        "id": "romantic",
        "label": "Romantic",
        "keywords": [
            "love", "romance", "heart", "together", "kiss", "darling",
            "beloved", "forever", "hold", "embrace", "tenderness", "intimate",
            "mohabbat", "ishq", "pyaar", "dil", "sanam", "mehboob",
        ],
        "pacing_profile": {
            "min_shot_duration":    2.5,
            "max_shot_duration":    9.0,
            "preferred_avg_duration": 3.8,
            "merge_threshold":      2.0,
            "split_threshold":      9.0,
            "long_shot_ratio":      0.30,
            "medium_shot_ratio":    0.55,
            "short_shot_ratio":     0.15,
        },
        "shot_intensity_biases": {
            "face":        0.42,
            "body":        0.34,
            "environment": 0.12,
            "macro":       0.07,
            "symbolic":    0.05,
        },
        "camera_movement_profile": {
            "static_threshold": 0.50,
            "drift_threshold":  0.75,
            "active_threshold": 1.01,
        },
        "cinematic_modifier": (
            "intimate romantic cinema, golden-hour warmth, shallow depth of field, "
            "soft radiant light on faces"
        ),
        "style_modifier_injection": {
            "movement": "slow drift or barely perceptible float — tenderness in every frame",
            "atmosphere_note": "warmth of closeness, the world narrowed to two people",
        },
    },

    "sad_loss": {
        "id": "sad_loss",
        "label": "Sad / Loss",
        "keywords": [
            "loss", "grief", "tears", "cry", "pain", "heartbreak", "miss",
            "gone", "empty", "alone", "broken", "goodbye", "left", "sorrow",
            "judai", "dard", "rona", "yaad", "bichhad", "tanha", "alvida",
        ],
        "pacing_profile": {
            "min_shot_duration":    3.0,
            "max_shot_duration":    10.0,
            "preferred_avg_duration": 4.5,
            "merge_threshold":      2.5,
            "split_threshold":      10.0,
            "long_shot_ratio":      0.40,
            "medium_shot_ratio":    0.45,
            "short_shot_ratio":     0.15,
        },
        "shot_intensity_biases": {
            "face":        0.45,
            "body":        0.28,
            "environment": 0.15,
            "macro":       0.06,
            "symbolic":    0.06,
        },
        "camera_movement_profile": {
            "static_threshold": 0.55,
            "drift_threshold":  0.80,
            "active_threshold": 1.01,
        },
        "cinematic_modifier": (
            "grief-weight stillness, overcast or fading light, "
            "muted palette — the quiet after the fall"
        ),
        "style_modifier_injection": {
            "movement": "barely perceptible — held-breath quality, stillness is the emotion",
            "atmosphere_note": "weight of loss, quiet grief, the air before a tear falls",
        },
    },

    "nostalgic": {
        "id": "nostalgic",
        "label": "Nostalgic",
        "keywords": [
            "remember", "memory", "used to", "back then", "once", "past",
            "childhood", "days gone", "fade", "old times", "yesterday",
            "yaadein", "zamaana", "waqt", "purana", "kal", "beet gaye",
        ],
        "pacing_profile": {
            "min_shot_duration":    3.0,
            "max_shot_duration":    10.0,
            "preferred_avg_duration": 4.2,
            "merge_threshold":      2.5,
            "split_threshold":      10.0,
            "long_shot_ratio":      0.35,
            "medium_shot_ratio":    0.50,
            "short_shot_ratio":     0.15,
        },
        "shot_intensity_biases": {
            "face":        0.38,
            "body":        0.30,
            "environment": 0.16,
            "macro":       0.08,
            "symbolic":    0.08,
        },
        "camera_movement_profile": {
            "static_threshold": 0.45,
            "drift_threshold":  0.72,
            "active_threshold": 1.01,
        },
        "cinematic_modifier": (
            "soft-focus memory texture, warm faded tones, "
            "Super-8 or vintage film grain — time slipping through fingers"
        ),
        "style_modifier_injection": {
            "movement": "gentle drift or slow pull-back — the camera remembering",
            "atmosphere_note": "haze of memory, warmth of what was, bittersweet distance",
        },
    },

    "hopeful": {
        "id": "hopeful",
        "label": "Hopeful",
        "keywords": [
            "hope", "dream", "rise", "new", "tomorrow", "believe", "light",
            "forward", "possibility", "open", "morning", "begin", "sky",
            "umeed", "asha", "naya", "subah", "roshan", "parwaz",
        ],
        "pacing_profile": {
            "min_shot_duration":    2.5,
            "max_shot_duration":    8.0,
            "preferred_avg_duration": 3.5,
            "merge_threshold":      2.0,
            "split_threshold":      8.0,
            "long_shot_ratio":      0.25,
            "medium_shot_ratio":    0.55,
            "short_shot_ratio":     0.20,
        },
        "shot_intensity_biases": {
            "face":        0.38,
            "body":        0.38,
            "environment": 0.16,
            "macro":       0.04,
            "symbolic":    0.04,
        },
        "camera_movement_profile": {
            "static_threshold": 0.35,
            "drift_threshold":  0.62,
            "active_threshold": 0.88,
        },
        "cinematic_modifier": (
            "uplifting open light, rising momentum, "
            "wide skies and expanding space — possibility made visible"
        ),
        "style_modifier_injection": {
            "movement": "slow rise or gentle forward drift — optimism as motion",
            "atmosphere_note": "possibility and forward momentum, breath before a new beginning",
        },
    },

    "angry_intense": {
        "id": "angry_intense",
        "label": "Angry / Intense",
        "keywords": [
            "fight", "anger", "rage", "war", "scream", "power", "rise",
            "rebel", "force", "fire", "storm", "shake", "break", "burn",
            "ghussa", "jung", "toofan", "aag", "lalkar", "himmat",
        ],
        "pacing_profile": {
            "min_shot_duration":    1.0,
            "max_shot_duration":    5.0,
            "preferred_avg_duration": 2.0,
            "merge_threshold":      1.0,
            "split_threshold":      5.0,
            "long_shot_ratio":      0.05,
            "medium_shot_ratio":    0.35,
            "short_shot_ratio":     0.60,
        },
        "shot_intensity_biases": {
            "face":        0.36,
            "body":        0.44,
            "environment": 0.10,
            "macro":       0.05,
            "symbolic":    0.05,
        },
        "camera_movement_profile": {
            "static_threshold": 0.20,
            "drift_threshold":  0.45,
            "active_threshold": 0.65,
        },
        "cinematic_modifier": (
            "raw confrontational energy, tight aggressive framing, "
            "high contrast — the body as weapon and statement"
        ),
        "style_modifier_injection": {
            "movement": "aggressive push-in or fast pan — urgency as motion language",
            "atmosphere_note": "tension at breaking point, controlled explosion of feeling",
        },
    },

    "spiritual_reflective": {
        "id": "spiritual_reflective",
        "label": "Spiritual / Reflective",
        "keywords": [
            "soul", "spirit", "god", "divine", "pray", "faith", "surrender",
            "silence", "peace", "inner", "beyond", "truth", "light", "empty",
            "rooh", "khuda", "ibadat", "sukoon", "dargah", "qawwali", "sufi",
        ],
        "pacing_profile": {
            "min_shot_duration":    4.0,
            "max_shot_duration":    12.0,
            "preferred_avg_duration": 5.5,
            "merge_threshold":      3.5,
            "split_threshold":      12.0,
            "long_shot_ratio":      0.45,
            "medium_shot_ratio":    0.45,
            "short_shot_ratio":     0.10,
        },
        "shot_intensity_biases": {
            "face":        0.35,
            "body":        0.28,
            "environment": 0.20,
            "macro":       0.08,
            "symbolic":    0.09,
        },
        "camera_movement_profile": {
            "static_threshold": 0.60,
            "drift_threshold":  0.82,
            "active_threshold": 1.01,
        },
        "cinematic_modifier": (
            "meditative stillness, sacred atmosphere, "
            "devotional light — the space between words and the divine"
        ),
        "style_modifier_injection": {
            "movement": "near-static — breath of the camera, reverence in every frame",
            "atmosphere_note": "spiritual gravity, the weight of something larger than the self",
        },
    },

    "energetic_celebration": {
        "id": "energetic_celebration",
        "label": "Energetic / Celebration",
        "keywords": [
            "dance", "celebrate", "party", "joy", "energy", "fun", "move",
            "jump", "spin", "run", "alive", "free", "go", "wild", "beat",
            "naach", "khushi", "jashan", "masti", "dhamaal", "dhol",
        ],
        "pacing_profile": {
            "min_shot_duration":    0.8,
            "max_shot_duration":    4.0,
            "preferred_avg_duration": 1.8,
            "merge_threshold":      0.8,
            "split_threshold":      4.0,
            "long_shot_ratio":      0.05,
            "medium_shot_ratio":    0.30,
            "short_shot_ratio":     0.65,
        },
        "shot_intensity_biases": {
            "face":        0.30,
            "body":        0.50,
            "environment": 0.12,
            "macro":       0.04,
            "symbolic":    0.04,
        },
        "camera_movement_profile": {
            "static_threshold": 0.15,
            "drift_threshold":  0.40,
            "active_threshold": 0.60,
        },
        "cinematic_modifier": (
            "kinetic joy, saturated vivid colour, "
            "movement and rhythm — life at full volume"
        ),
        "style_modifier_injection": {
            "movement": "active pans, fast orbit, energy in every camera decision",
            "atmosphere_note": "pure joy and release, the body surrendering to the beat",
        },
    },
}

# ---------------------------------------------------------------------------
# DETERMINISTIC KEYWORD CLASSIFIER
# ---------------------------------------------------------------------------

# Per-mode BPM affinity ranges: (min_bpm, max_bpm, score_bonus)
_BPM_AFFINITY: Dict[str, Tuple[float, float, float]] = {
    "romantic":             (60,  95,  0.15),
    "sad_loss":             (50,  85,  0.15),
    "nostalgic":            (55,  90,  0.10),
    "hopeful":              (75, 115,  0.10),
    "angry_intense":        (120, 200, 0.20),
    "spiritual_reflective": (55,  90,  0.15),
    "energetic_celebration":(110, 200, 0.20),
}


def _keyword_scores(text: str, genre: str, bpm: Optional[float]) -> Dict[str, float]:
    """Return a raw score per mode based on keyword frequency + BPM bonus."""
    text_lower = (text or "").lower()
    genre_lower = (genre or "").lower()
    scores: Dict[str, float] = {m: 0.0 for m in _MODE_REGISTRY}

    for mode_id, cfg in _MODE_REGISTRY.items():
        count = sum(1 for kw in cfg["keywords"] if kw in text_lower or kw in genre_lower)
        scores[mode_id] = count * 0.1

    if bpm is not None:
        for mode_id, (lo, hi, bonus) in _BPM_AFFINITY.items():
            if lo <= bpm <= hi:
                scores[mode_id] += bonus

    # Genre overrides
    if "qawwali" in genre_lower or "sufi" in genre_lower:
        scores["spiritual_reflective"] += 0.4
    if "ghazal" in genre_lower:
        scores["sad_loss"]     += 0.2
        scores["nostalgic"]    += 0.2
        scores["romantic"]     += 0.15
    if any(g in genre_lower for g in ("pop", "dance", "edm", "bhangra")):
        scores["energetic_celebration"] += 0.3
    if any(g in genre_lower for g in ("hip hop", "rap", "trap")):
        scores["angry_intense"]         += 0.2
        scores["energetic_celebration"] += 0.1

    return scores


def _classify_deterministic(
    context_packet: Dict[str, Any],
    audio_data: Dict[str, Any],
) -> Tuple[str, str, Dict[str, float]]:
    """Deterministic keyword + BPM classifier.

    Returns (primary_mode_id, secondary_mode_id, raw_scores).
    """
    text = (context_packet.get("lyrics") or context_packet.get("text_excerpt") or "")
    meaning = context_packet.get("meaning") or {}
    if isinstance(meaning, dict):
        text += " " + (meaning.get("summary") or meaning.get("statement") or "")
    theme = context_packet.get("theme") or {}
    if isinstance(theme, dict):
        text += " " + (theme.get("summary") or "")
    elif isinstance(theme, str):
        text += " " + theme

    genre = audio_data.get("genre") or context_packet.get("genre") or ""
    bpm   = audio_data.get("bpm")
    try:
        bpm = float(bpm) if bpm else None
    except (TypeError, ValueError):
        bpm = None

    scores = _keyword_scores(text, genre, bpm)
    sorted_modes = sorted(scores, key=lambda m: scores[m], reverse=True)

    primary   = sorted_modes[0]
    secondary = sorted_modes[1] if len(sorted_modes) > 1 else primary

    # If all scores are 0 (no signal at all), default to romantic
    if scores[primary] == 0.0:
        primary   = "romantic"
        secondary = "nostalgic"

    return primary, secondary, scores


# ---------------------------------------------------------------------------
# LLM CLASSIFIER
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a music video creative director specialising in emotional intelligence.

Your job: read the song context below and classify its DOMINANT emotional mode.

Emotional modes available (use the exact id string):
  romantic              — love, intimacy, longing, tenderness
  sad_loss              — grief, heartbreak, absence, mourning
  nostalgic             — memory, yearning for the past, bittersweet retrospection
  hopeful               — aspiration, new beginnings, rising, possibility
  angry_intense         — rage, confrontation, power, rebellion
  spiritual_reflective  — devotion, transcendence, inner searching, surrender
  energetic_celebration — joy, dance, festivity, kinetic energy, liberation

Return ONLY valid JSON, no markdown:
{
  "primary_mode": "<mode_id>",
  "secondary_mode": "<mode_id>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}
""".strip()


async def _classify_llm(
    context_packet: Dict[str, Any],
    audio_data: Dict[str, Any],
    api_key: str,
) -> Tuple[Optional[str], Optional[str], float, str]:
    """Call GPT-4o-mini to classify emotional mode.  Returns (primary, secondary, confidence, reasoning).
    Returns (None, None, 0, "") on any failure — caller falls back to deterministic.
    """
    import json as _json
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        meaning = context_packet.get("meaning") or context_packet.get("theme") or {}
        if isinstance(meaning, dict):
            meaning_text = meaning.get("summary") or meaning.get("statement") or ""
        else:
            meaning_text = str(meaning)[:200]

        speaker = context_packet.get("speaker") or {}
        if isinstance(speaker, dict):
            speaker_text = ", ".join(f"{k}={v}" for k, v in speaker.items() if v)[:120]
        else:
            speaker_text = str(speaker)[:120]

        lyrics_excerpt = (context_packet.get("lyrics") or context_packet.get("text_excerpt") or "")[:600]

        user_prompt = f"""Genre: {audio_data.get('genre') or context_packet.get('genre') or 'unknown'}
BPM: {audio_data.get('bpm') or 'unknown'}
Energy: {audio_data.get('avg_energy') or 'unknown'}
Meaning: {meaning_text}
Speaker: {speaker_text}
Lyrics excerpt:
{lyrics_excerpt}

Classify the dominant emotional mode."""

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=200,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
        parsed = _json.loads(raw)
        primary   = parsed.get("primary_mode",   "").strip()
        secondary = parsed.get("secondary_mode", "").strip()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning  = str(parsed.get("reasoning", ""))
        if primary not in _MODE_REGISTRY:
            return None, None, 0.0, ""
        if secondary not in _MODE_REGISTRY:
            secondary = primary
        return primary, secondary, confidence, reasoning
    except Exception as exc:
        logger.warning("EmotionalModeEngine LLM classifier failed: %s", exc)
        return None, None, 0.0, ""


# ---------------------------------------------------------------------------
# PACKET BUILDER
# ---------------------------------------------------------------------------

def _blend_biases(
    primary: Dict[str, float],
    secondary: Dict[str, float],
    w_primary: float = 0.7,
) -> Dict[str, float]:
    """Blend two shot-intensity-bias dicts with a weighted average."""
    w_sec = 1.0 - w_primary
    keys = set(primary) | set(secondary)
    blended = {k: round(primary.get(k, 0.0) * w_primary + secondary.get(k, 0.0) * w_sec, 4)
               for k in keys}
    total = sum(blended.values())
    if total > 0:
        blended = {k: round(v / total, 4) for k, v in blended.items()}
    return blended


def _blend_pacing(
    primary: Dict[str, float],
    secondary: Dict[str, float],
    w_primary: float = 0.7,
) -> Dict[str, float]:
    """Weighted blend of two pacing profile dicts."""
    w_sec = 1.0 - w_primary
    keys = set(primary) | set(secondary)
    return {k: round(primary.get(k, 0.0) * w_primary + secondary.get(k, 0.0) * w_sec, 3)
            for k in keys}


def build_emotional_mode_packet(
    primary_mode_id: str,
    secondary_mode_id: str,
    raw_scores: Dict[str, float],
    classifier_method: str,
    reasoning: str = "",
) -> Dict[str, Any]:
    """Assemble the Brain-ready emotional_mode_packet from two mode IDs.

    Blends pacing profiles and shot-intensity biases at 70 / 30.
    Primary mode's cinematic_modifier and style_modifier_injection are used
    as-is (no blending — they must be coherent strings).
    """
    pm = _MODE_REGISTRY.get(primary_mode_id) or _MODE_REGISTRY["romantic"]
    sm = _MODE_REGISTRY.get(secondary_mode_id) or pm

    blended_pacing = _blend_pacing(pm["pacing_profile"], sm["pacing_profile"])
    blended_biases = _blend_biases(pm["shot_intensity_biases"], sm["shot_intensity_biases"])

    return {
        "primary_mode":               primary_mode_id,
        "secondary_mode":             secondary_mode_id,
        "primary_weight":             0.7,
        "secondary_weight":           0.3,
        "pacing_profile":             blended_pacing,
        "shot_intensity_biases":      blended_biases,
        "camera_movement_profile":    pm["camera_movement_profile"],
        "cinematic_modifier":         pm["cinematic_modifier"],
        "style_modifier_injection":   pm["style_modifier_injection"],
        "classifier_method":          classifier_method,
        "raw_scores":                 {m: round(s, 4) for m, s in raw_scores.items()},
        "reasoning":                  reasoning,
        "mode_label":                 pm["label"],
    }


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

async def classify_emotional_mode(
    context_packet: Dict[str, Any],
    audio_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify the emotional mode for a project and return the Brain packet.

    Tries the LLM first; falls back to the deterministic classifier on any
    failure or low-confidence result.  Always returns a valid packet.

    Parameters
    ----------
    context_packet : Brain namespace context_packet (Stage 2 output)
    audio_data     : audio_data dict from the projects table / raw_input Brain namespace
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    llm_primary = llm_secondary = None
    llm_confidence = 0.0
    llm_reasoning  = ""

    if api_key:
        llm_primary, llm_secondary, llm_confidence, llm_reasoning = await _classify_llm(
            context_packet, audio_data, api_key,
        )

    det_primary, det_secondary, raw_scores = _classify_deterministic(context_packet, audio_data)

    if llm_primary and llm_confidence >= 0.5:
        primary   = llm_primary
        secondary = llm_secondary or det_secondary
        method    = "llm"
        reasoning = llm_reasoning
        if not raw_scores:
            raw_scores = {m: 0.0 for m in _MODE_REGISTRY}
        logger.info(
            "EmotionalModeEngine: LLM classified primary=%s secondary=%s confidence=%.2f",
            primary, secondary, llm_confidence,
        )
    else:
        primary   = det_primary
        secondary = det_secondary
        method    = "keyword_fallback"
        reasoning = f"Keyword + BPM fallback (top scores: {dict(list(sorted(raw_scores.items(), key=lambda x: -x[1]))[:3])})"
        logger.info(
            "EmotionalModeEngine: deterministic classified primary=%s secondary=%s",
            primary, secondary,
        )

    return build_emotional_mode_packet(primary, secondary, raw_scores, method, reasoning)


def get_mode(mode_id: str) -> Optional[Dict[str, Any]]:
    """Return the mode registry entry for a given mode_id, or None."""
    return _MODE_REGISTRY.get(mode_id)


def all_mode_ids() -> List[str]:
    """Return all 7 mode ids."""
    return list(_MODE_REGISTRY.keys())
