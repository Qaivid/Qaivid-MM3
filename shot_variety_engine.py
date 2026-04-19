"""
shot_variety_engine.py

Purpose:
Ensure diversity across a sequence of shots to avoid repetition and flat visuals.
"""

from typing import List, Dict


class ShotVarietyEngine:
    def __init__(self):
        self.shot_types = [
            "portrait",
            "wide_environment",
            "object_detail",
            "movement",
            "empty_frame",
            "over_shoulder",
            "reflection",
            "silhouette"
        ]

    def apply_variety(self, events: List[Dict]) -> List[Dict]:
        """
        Assign variety tags to each shot event.
        """

        varied_events = []
        for i, event in enumerate(events):
            shot_type = self.shot_types[i % len(self.shot_types)]

            event["shot_type"] = shot_type
            event["variety_applied"] = True

            varied_events.append(event)

        return varied_events

    def check_repetition(self, events: List[Dict]) -> bool:
        """
        Detect if too many consecutive shots are similar.
        """

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
