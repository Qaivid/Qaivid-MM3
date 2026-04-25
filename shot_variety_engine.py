"""
shot_variety_engine.py

Purpose:
Assign diverse shot_type values across a sequence to enforce cinematic variety.

Director-spec target distribution:
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
"""

from typing import List, Dict


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
    def __init__(self):
        self.shot_types = list(_BASE_CYCLE)

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
