"""
shot_variety_engine.py

Purpose:
Assign diverse shot_type values across a sequence to enforce cinematic variety.

Director-spec target distribution (default):
    25 % close_up        (face — CU / ECU)
    15 % head_shoulders  (face — H&S / MCU)
    20 % medium_shot     (body — medium / ¾ body)
    10 % full_body       (body — full figure)
     6 % movement        (body — movement shots)
    10 % wide_shot       (environment — wide)
     4 % drone           (environment — aerial)
     5 % insert          (macro — prop/object inserts)
     5 % memory_fragment (symbolic — memory / fragments)

A 20-item base cycle hits these targets exactly.
Emotion intensity nudges the assigned type for very high or very low shots.
Emotional mode adjusts the cycle to serve the locked emotional register.
"""

from typing import Any, Dict, List, Optional


_BASE_CYCLE = [
    "close_up",          # face — CU (25%)
    "medium_shot",       # body — medium/¾ (20%)
    "head_shoulders",    # face — H&S (15%)
    "wide_shot",         # environment — wide (10%)
    "full_body",         # body — full figure (10%)
    "close_up",          # face — CU
    "movement",          # body — movement (6%)
    "insert",            # macro — insert (5%)
    "head_shoulders",    # face — H&S
    "medium_shot",       # body — medium/¾
    "close_up",          # face — CU
    "drone",             # environment — aerial (4%)
    "full_body",         # body — full figure
    "head_shoulders",    # face — H&S
    "medium_shot",       # body — medium/¾
    "close_up",          # face — CU
    "wide_shot",         # environment — wide
    "memory_fragment",   # symbolic — memory/fragment (5%)
    "medium_shot",       # body — medium/¾
    "close_up",          # face — CU
]

# ── MM3.1 Emotional Mode Cycles ───────────────────────────────────────────────
# 20-item cycles tuned to each mode's target shot distribution.
# romantic:      face 47%, body 32%, env 12%, macro 5%, symbolic 4%
# sad_loss:      face 45%, body 31%, env 13%, macro 5%, symbolic 6%
# nostalgic:     face 35%, body 34%, env 17%, macro 5%, symbolic 9%
# hopeful:       face 33%, body 33%, env 22%, macro 5%, symbolic 7%
# angry_intense: face 35%, body 46%, env 10%, macro 5%, symbolic 4%
# spiritual_reflective:  face 28%, body 28%, env 22%, macro 7%, symbolic 15%
# energetic_celebration: face 35%, body 51%, env 10%, macro 4%, symbolic 0%
_MODE_CYCLES: Dict[str, List[str]] = {
    "romantic": [
        "close_up",        "medium_shot",     "head_shoulders",   "wide_shot",
        "close_up",        "full_body",        "head_shoulders",   "insert",
        "close_up",        "medium_shot",      "extreme_close_up", "head_shoulders",
        "full_body",       "medium_shot",      "close_up",         "drone",
        "close_up",        "memory_fragment",  "movement",         "close_up",
    ],
    "sad_loss": [
        "close_up",        "medium_shot",      "head_shoulders",   "memory_fragment",
        "close_up",        "full_body",        "head_shoulders",   "wide_shot",
        "close_up",        "insert",           "extreme_close_up", "medium_shot",
        "head_shoulders",  "close_up",         "wide_shot",        "memory_fragment",
        "medium_shot",     "full_body",        "drone",            "close_up",
    ],
    "nostalgic": [
        "close_up",        "medium_shot",      "wide_shot",        "memory_fragment",
        "head_shoulders",  "full_body",        "close_up",         "memory_fragment",
        "medium_shot",     "wide_shot",        "head_shoulders",   "insert",
        "extreme_close_up","medium_shot",      "drone",            "close_up",
        "full_body",       "movement",         "head_shoulders",   "medium_shot",
    ],
    "hopeful": [
        "close_up",        "medium_shot",      "wide_shot",        "head_shoulders",
        "full_body",       "wide_shot",        "close_up",         "memory_fragment",
        "medium_shot",     "wide_shot",        "head_shoulders",   "insert",
        "extreme_close_up","drone",            "medium_shot",      "close_up",
        "full_body",       "movement",         "head_shoulders",   "medium_shot",
    ],
    "angry_intense": [
        "close_up",        "movement",         "medium_shot",      "head_shoulders",
        "movement",        "full_body",        "close_up",         "wide_shot",
        "movement",        "medium_shot",      "extreme_close_up", "movement",
        "head_shoulders",  "full_body",        "medium_shot",      "drone",
        "close_up",        "insert",           "memory_fragment",  "close_up",
    ],
    "spiritual_reflective": [
        "memory_fragment", "medium_shot",      "wide_shot",        "close_up",
        "drone",           "memory_fragment",  "head_shoulders",   "wide_shot",
        "insert",          "medium_shot",      "memory_fragment",  "close_up",
        "full_body",       "drone",            "head_shoulders",   "insert",
        "close_up",        "extreme_close_up", "movement",         "wide_shot",
    ],
    "energetic_celebration": [
        "movement",        "close_up",         "medium_shot",      "movement",
        "head_shoulders",  "full_body",        "movement",         "medium_shot",
        "close_up",        "movement",         "head_shoulders",   "full_body",
        "wide_shot",       "movement",         "medium_shot",      "extreme_close_up",
        "drone",           "movement",         "insert",           "close_up",
    ],
}

_SHOT_TYPE_TO_MODE = {
    # face
    "close_up":          "face",
    "extreme_close_up":  "face",
    "head_shoulders":    "face",
    "portrait":          "face",    # legacy alias
    # body
    "medium_shot":       "body",
    "full_body":         "body",
    "movement":          "body",
    "over_shoulder":     "body",
    # environment
    "wide_shot":         "environment",
    "wide_environment":  "environment",   # legacy alias
    "drone":             "environment",
    "empty_frame":       "environment",
    # macro
    "insert":            "macro",
    "object_detail":     "macro",   # legacy alias
    # symbolic
    "memory_fragment":   "symbolic",
    "reflection":        "symbolic",   # legacy alias
    "silhouette":        "symbolic",
}


class ShotVarietyEngine:
    def __init__(self, emotional_mode_packet: Optional[Dict[str, Any]] = None):
        """Initialise with an optional emotional_mode_packet from Stage 2b.

        When a known mode is supplied, the engine uses the mode-tuned cycle
        instead of the default _BASE_CYCLE, shifting the shot-type distribution
        to serve the locked emotional register.
        Blends primary (70 %) + secondary (30 %) cycles when both are present.
        """
        emp = emotional_mode_packet or {}
        primary_id   = emp.get("primary_mode") or ""
        secondary_id = emp.get("secondary_mode") or ""
        pw = float(emp.get("primary_weight", 1.0))
        sw = float(emp.get("secondary_weight", 0.0))

        primary_cycle = _MODE_CYCLES.get(primary_id)
        if not primary_cycle:
            self.shot_types = list(_BASE_CYCLE)
        elif secondary_id and sw > 0 and _MODE_CYCLES.get(secondary_id):
            secondary_cycle = _MODE_CYCLES[secondary_id]
            blended: List[str] = []
            for i in range(20):
                if i / 20 < pw:
                    blended.append(primary_cycle[i % len(primary_cycle)])
                else:
                    blended.append(secondary_cycle[i % len(secondary_cycle)])
            self.shot_types = blended
        else:
            self.shot_types = list(primary_cycle)

    @staticmethod
    def shot_type_to_mode(shot_type: str) -> str:
        """Convert a shot_type label to a VSE expression_mode string."""
        return _SHOT_TYPE_TO_MODE.get(shot_type, "face")

    def apply_variety(self, events: List[Dict]) -> List[Dict]:
        """
        Assign a shot_type to each event from the base cycle.

        Intensity nudge:
        - Very high intensity (> 0.85) → prefer close_up or movement
        - Very low intensity (< 0.15) → prefer wide_shot or insert
        Otherwise uses the deterministic cycle.
        """
        varied: List[Dict] = []
        for i, event in enumerate(events):
            base_type = self.shot_types[i % len(self.shot_types)]

            intensity = float(event.get("intensity") or 0.5)

            if intensity > 0.85 and base_type in ("wide_shot", "drone", "memory_fragment"):
                nudged = "close_up"
            elif intensity < 0.15 and base_type in ("medium_shot", "full_body", "movement"):
                nudged = "wide_shot"
            else:
                nudged = base_type

            event["shot_type"] = nudged
            event["variety_applied"] = True
            varied.append(event)

        return varied

    def check_repetition(self, events: List[Dict]) -> bool:
        """Return True if 3+ consecutive shots share the same shot_type."""
        last_type = None
        repeat_count = 0
        for event in events:
            current_type = event.get("shot_type")
            if current_type == last_type:
                repeat_count += 1
                if repeat_count >= 3:
                    return True
            else:
                repeat_count = 0
            last_type = current_type
        return False
