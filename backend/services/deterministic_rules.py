"""
Pacing Profiles — Ported from Qaivid 1.0 storyboard-engine-v2/types.ts
Per-content-type shot duration and density rules.
Used by the Shot Engine to determine shot count and duration per scene.

MM3.1 extension: EMOTIONAL_PACING_PROFILES adds per-mode overrides keyed by
(content_type, emotional_mode_id).  get_pacing_profile() accepts an optional
emotional_mode_id param and merges the override on top of the base profile.
"""
from typing import Dict, Optional

PACING_PROFILES: Dict[str, Dict[str, float]] = {
    "song": {
        "min_shot_duration": 2.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 6.0,
        "merge_threshold": 2.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.25,
        "medium_shot_ratio": 0.55,
        "short_shot_ratio": 0.20,
    },
    "poem": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 8.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.35,
        "medium_shot_ratio": 0.50,
        "short_shot_ratio": 0.15,
    },
    "ghazal": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 8.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.35,
        "medium_shot_ratio": 0.50,
        "short_shot_ratio": 0.15,
    },
    "qawwali": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 7.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.30,
        "medium_shot_ratio": 0.50,
        "short_shot_ratio": 0.20,
    },
    "script": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 8.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.35,
        "medium_shot_ratio": 0.45,
        "short_shot_ratio": 0.20,
    },
    "story": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 8.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.35,
        "medium_shot_ratio": 0.45,
        "short_shot_ratio": 0.20,
    },
    "ad": {
        "min_shot_duration": 2.0,
        "max_shot_duration": 10.0,
        "preferred_avg_duration": 3.5,
        "merge_threshold": 1.5,
        "split_threshold": 10.0,
        "long_shot_ratio": 0.10,
        "medium_shot_ratio": 0.35,
        "short_shot_ratio": 0.55,
    },
    "documentary": {
        "min_shot_duration": 4.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 9.0,
        "merge_threshold": 4.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.40,
        "medium_shot_ratio": 0.45,
        "short_shot_ratio": 0.15,
    },
    "voiceover": {
        "min_shot_duration": 3.0,
        "max_shot_duration": 15.0,
        "preferred_avg_duration": 8.0,
        "merge_threshold": 3.0,
        "split_threshold": 15.0,
        "long_shot_ratio": 0.35,
        "medium_shot_ratio": 0.50,
        "short_shot_ratio": 0.15,
    },
}

# Common motifs for detection (ported from Qaivid 1.0)
COMMON_MOTIFS = [
    "water", "fire", "night", "dust", "train", "crowd", "home", "distance",
    "sky", "road", "mirror", "rain", "light", "shadow", "door", "window",
    "clock", "moon", "sun", "ocean", "mountain", "bridge", "city", "forest",
    "flower", "bird", "storm", "star", "blood", "hand", "eye", "heart",
    "ghost", "dream", "fog", "smoke", "glass", "stone", "river", "wind",
    "charkha", "phulkari", "doli", "mehendi", "payal", "chunni", "charpai",
    "haveli", "kotha", "pipal", "beri", "dargah", "shama", "aaina",
]

# Emotional shift patterns (ported from Qaivid 1.0)
EMOTIONAL_SHIFTS = [
    {"pattern": r"\b(but|however|yet|still|though)\b", "shift": "contrast"},
    {"pattern": r"\b(remember|used to|back then|once)\b", "shift": "nostalgia"},
    {"pattern": r"\b(now|today|here|this moment)\b", "shift": "present"},
    {"pattern": r"\b(dream|imagine|wish|hope|one day)\b", "shift": "aspiration"},
    {"pattern": r"\b(cry|tears|pain|hurt|broken|lost|gone|miss|empty|alone)\b", "shift": "sorrow"},
    {"pattern": r"\b(love|heart|soul|together|forever|hold|kiss|embrace)\b", "shift": "love"},
    {"pattern": r"\b(fight|rise|stand|strong|power|rage|anger|war|scream)\b", "shift": "empowerment"},
    {"pattern": r"\b(dance|move|jump|fly|run|go|spin|turn|shake)\b", "shift": "energy"},
    {"pattern": r"\b(quiet|silence|whisper|soft|gentle|calm|peace|still)\b", "shift": "serenity"},
    {"pattern": r"\b(dark|death|fear|dread|haunt|shadow|grave|cold)\b", "shift": "darkness"},
    {"pattern": r"\b(free|wild|open|escape|break|beyond|horizon)\b", "shift": "liberation"},
    {"pattern": r"\b(wait|time|passing|fading|slow|moment|breath)\b", "shift": "contemplation"},
]

# Symbolic vs literal indicators (ported from Qaivid 1.0 semantic-analyzer.ts)
SYMBOLIC_INDICATORS = [
    r"\blike\s+a\b",
    r"\bas\s+if\b",
    r"\bmetaphor\b",
    r"\bsymbol\b",
    r"\bdrown(?:ing|ed|s)?\s+in\b",
    r"\bburning\s+(?:with|inside)\b",
    r"\bwings?\s+(?:of|to)\b",
    r"\bchains?\s+(?:of|that)\b",
    r"\bcage[ds]?\b",
    r"\bmask(?:s|ed)?\b",
    r"\bmirror(?:s|ed)?\b",
    r"\bshadow(?:s)?\s+(?:of|that|follow)\b",
    r"\bghost(?:s)?\s+(?:of|that|from)\b",
    r"\bwall(?:s)?\s+(?:around|between|closing)\b",
    r"\bbleed(?:ing|s)?\s+(?:into|through|out)\b",
    r"\bfalling\s+(?:into|through|apart)\b",
    r"\brising\s+(?:from|above|up)\b",
]

LITERAL_INDICATORS = [
    r"\bwalking\b",
    r"\bsitting\b",
    r"\bstanding\b",
    r"\blooking\s+at\b",
    r"\bdriving\b",
    r"\btalking\s+to\b",
    r"\bholding\b",
    r"\bwearing\b",
    r"\bin\s+the\s+(room|street|car|house|building|office)\b",
    r"\bat\s+the\s+(table|door|window|bar|park)\b",
]

# Setting detection patterns (ported from Qaivid 1.0 semantic-analyzer.ts)
SETTING_PATTERNS = [
    {"pattern": r"\b(street|road|highway|alley|sidewalk)\b", "setting": "urban street"},
    {"pattern": r"\b(room|bedroom|living room|kitchen|apartment)\b", "setting": "interior room"},
    {"pattern": r"\b(ocean|sea|beach|shore|wave|coast)\b", "setting": "coastal/ocean"},
    {"pattern": r"\b(forest|woods|trees|jungle|grove)\b", "setting": "forest/nature"},
    {"pattern": r"\b(mountain|hill|cliff|peak)\b", "setting": "mountain/elevated"},
    {"pattern": r"\b(city|downtown|skyline|skyscraper)\b", "setting": "urban cityscape"},
    {"pattern": r"\b(stage|concert|performance|spotlight)\b", "setting": "performance stage"},
    {"pattern": r"\b(temple|dargah|cathedral|altar|gurudwara)\b", "setting": "sacred space"},
    {"pattern": r"\b(desert|sand|dunes|wasteland)\b", "setting": "desert/arid"},
    {"pattern": r"\b(rain|storm|thunder|lightning)\b", "setting": "stormy atmosphere"},
    {"pattern": r"\b(night|dark|midnight|moonlight)\b", "setting": "nighttime"},
    {"pattern": r"\b(garden|field|meadow|flower|khet)\b", "setting": "garden/pastoral"},
    {"pattern": r"\b(car|bus|train|metro|airport)\b", "setting": "transit/vehicle"},
    {"pattern": r"\b(rooftop|balcony|terrace|kotha)\b", "setting": "elevated overlook"},
    {"pattern": r"\b(courtyard|haveli|vehra|chowk)\b", "setting": "courtyard/haveli"},
    {"pattern": r"\b(river|nadi|dariya|ghat)\b", "setting": "riverside"},
]


# ── MM3.1 Emotional Mode Pacing Overrides ────────────────────────────────────
# Indexed as EMOTIONAL_PACING_PROFILES[content_type][emotional_mode_id].
# Each entry is a partial dict — only keys that differ from the base profile.
# get_pacing_profile() merges these on top of the content-type base.
EMOTIONAL_PACING_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "song": {
        "romantic": {
            "preferred_avg_duration": 7.5,
            "long_shot_ratio":  0.35,
            "medium_shot_ratio": 0.50,
            "short_shot_ratio":  0.15,
        },
        "sad_loss": {
            "preferred_avg_duration": 8.0,
            "long_shot_ratio":  0.40,
            "medium_shot_ratio": 0.48,
            "short_shot_ratio":  0.12,
        },
        "nostalgic": {
            "preferred_avg_duration": 7.0,
            "long_shot_ratio":  0.35,
            "medium_shot_ratio": 0.52,
            "short_shot_ratio":  0.13,
        },
        "hopeful": {
            "preferred_avg_duration": 6.5,
            "long_shot_ratio":  0.28,
            "medium_shot_ratio": 0.55,
            "short_shot_ratio":  0.17,
        },
        "angry_intense": {
            "preferred_avg_duration": 4.5,
            "long_shot_ratio":  0.12,
            "medium_shot_ratio": 0.43,
            "short_shot_ratio":  0.45,
        },
        "spiritual_reflective": {
            "preferred_avg_duration": 9.0,
            "long_shot_ratio":  0.45,
            "medium_shot_ratio": 0.45,
            "short_shot_ratio":  0.10,
        },
        "energetic_celebration": {
            "preferred_avg_duration": 4.0,
            "long_shot_ratio":  0.10,
            "medium_shot_ratio": 0.40,
            "short_shot_ratio":  0.50,
        },
    },
}


def get_pacing_profile(
    content_type: str,
    emotional_mode_id: Optional[str] = None,
) -> Dict[str, float]:
    """Return the pacing profile for a given content type.

    When emotional_mode_id is supplied, the per-mode override from
    EMOTIONAL_PACING_PROFILES is merged on top of the base profile so
    e.g. a spiritual song uses longer shot durations than the default.
    """
    base = dict(PACING_PROFILES.get(content_type, PACING_PROFILES["song"]))
    if emotional_mode_id:
        overrides = (
            EMOTIONAL_PACING_PROFILES.get(content_type, {}).get(emotional_mode_id) or
            EMOTIONAL_PACING_PROFILES.get("song", {}).get(emotional_mode_id) or
            {}
        )
        base.update(overrides)
    return base


def detect_emotional_shift(text: str) -> str:
    import re as _re
    text_lower = text.lower()
    for es in EMOTIONAL_SHIFTS:
        if _re.search(es["pattern"], text_lower, _re.IGNORECASE):
            return es["shift"]
    return "neutral"


def detect_symbolic_density(text: str) -> str:
    import re as _re
    text_lower = text.lower()
    symbolic_count = sum(1 for p in SYMBOLIC_INDICATORS if _re.search(p, text_lower, _re.IGNORECASE))
    literal_count = sum(1 for p in LITERAL_INDICATORS if _re.search(p, text_lower, _re.IGNORECASE))
    if symbolic_count >= 3:
        return "high"
    if symbolic_count >= 1 and literal_count <= 1:
        return "medium"
    return "low"


def infer_setting(text: str) -> str:
    import re as _re
    text_lower = text.lower()
    for sp in SETTING_PATTERNS:
        if _re.search(sp["pattern"], text_lower, _re.IGNORECASE):
            return sp["setting"]
    return "unspecified"
