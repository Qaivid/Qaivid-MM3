"""
shot_variety_engine.py

Purpose:
Assign diverse shot_type values across a sequence to enforce cinematic variety.

MM3.1 target distribution (matches task spec):
    ~20% portrait      (face closeup)
    ~30% body/action   (movement + over_shoulder)
    ~20% environment   (wide_environment + empty_frame)
    ~20% object/detail (object_detail)
    ~10% symbolic      (reflection)

A 10-item base cycle hits these targets exactly.
Emotion intensity nudges the assigned type for very high or very low shots.
"""

from typing import List, Dict


_BASE_CYCLE = [
    "portrait",           # face
    "movement",           # body — physical action
    "wide_environment",   # environment — breath of space
    "object_detail",      # macro — meaningful object
    "movement",           # body
    "portrait",           # face
    "empty_frame",        # environment — absence / space
    "over_shoulder",      # body — relational frame
    "object_detail",      # macro
    "reflection",         # symbolic
]

_SHOT_TYPE_TO_MODE = {
    "portrait":          "face",
    "wide_environment":  "environment",
    "object_detail":     "macro",
    "movement":          "body",
    "empty_frame":       "environment",
    "over_shoulder":     "body",
    "reflection":        "symbolic",
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
        - Very high intensity (> 0.85) → prefer movement or wide_environment
        - Very low intensity (< 0.15) → prefer empty_frame or object_detail
        Otherwise uses the deterministic cycle.
        """
        varied: List[Dict] = []
        for i, event in enumerate(events):
            base_type = self.shot_types[i % len(self.shot_types)]

            intensity = float(event.get("intensity") or 0.5)

            if intensity > 0.85 and base_type in ("portrait", "reflection"):
                nudged = "movement"
            elif intensity < 0.15 and base_type in ("movement", "over_shoulder"):
                nudged = "empty_frame"
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
