"""
generic_shot_validator.py

Purpose:
Detect and flag weak / generic shots that lack cinematic value.

MM3.1 design decision:
- MARKS events as generic rather than removing them.
- Index continuity is preserved so _shot_events_by_index maps correctly.
- is_generic=True / is_valid=False flags are set; downstream callers decide
  whether to rewrite or skip those shots.
"""

from typing import Dict, List


class GenericShotValidator:
    def __init__(self):
        self.generic_patterns = [
            "looking away",
            "emotional eyes",
            "distant gaze",
            "sad expression",
            "cinematic portrait"
        ]

    def is_generic(self, event: Dict) -> bool:
        """
        Detect if a shot is generic based on lack of action and overused phrases.
        """
        action = event.get("action", "") or ""
        contrast = event.get("visual_contrast", "") or ""
        camera = event.get("camera_motivation", "") or ""

        text_blob = f"{action} {contrast} {camera}".lower()

        if not action or len(action.split()) < 3:
            return True

        for pattern in self.generic_patterns:
            if pattern in text_blob:
                return True

        return False

    def validate_sequence(self, events: List[Dict]) -> List[Dict]:
        """
        Mark each event as generic or valid — never remove events.
        Preserving list length keeps _shot_events_by_index aligned with
        the storyboard's line_meanings list.
        """
        for event in events:
            generic = self.is_generic(event)
            event["is_generic"] = generic
            event["is_valid"] = not generic

        return events
