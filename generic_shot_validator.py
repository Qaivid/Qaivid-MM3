"""
generic_shot_validator.py

Purpose:
Reject weak / generic shots that lack cinematic value.
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
        Detect if a shot is generic based on lack of action and overused patterns.
        """

        action = event.get("action", "") or ""
        contrast = event.get("visual_contrast", "") or ""
        camera = event.get("camera_motivation", "") or ""

        text_blob = f"{action} {contrast} {camera}".lower()

        # If no action → generic
        if not action or len(action.split()) < 3:
            return True

        # If matches generic phrases
        for pattern in self.generic_patterns:
            if pattern in text_blob:
                return True

        return False

    def validate_sequence(self, events: List[Dict]) -> List[Dict]:
        """
        Filter out generic shots from a sequence.
        """

        valid_events = []

        for event in events:
            if not self.is_generic(event):
                event["is_generic"] = False
                valid_events.append(event)
            else:
                event["is_generic"] = True

        return valid_events
