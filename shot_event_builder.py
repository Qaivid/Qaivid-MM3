"""
shot_event_builder.py

Purpose:
Convert cinematic beats into concrete shot events that downstream
engines can use for prompting and rendering.
"""

from typing import Dict, Any


class ShotEventBuilder:
    def __init__(self):
        pass

    def build_event(self, beat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a cinematic beat into a shot event.
        """

        return {
            "shot_function": beat.get("shot_function"),
            "action": beat.get("subject_action"),
            "trigger": beat.get("trigger_event"),
            "emotional_shift": beat.get("emotional_shift"),
            "object_interaction": beat.get("object_usage"),
            "environment_interaction": beat.get("environment_usage"),
            "camera_motivation": beat.get("camera_motive"),
            "visual_contrast": beat.get("visual_contrast"),
            "is_valid": self._validate_event(beat)
        }

    def _validate_event(self, beat: Dict[str, Any]) -> bool:
        """
        Ensure the shot has minimum cinematic requirements.
        """

        required_fields = [
            "subject_action",
            "camera_motive",
            "visual_contrast"
        ]

        for field in required_fields:
            if not beat.get(field):
                return False

        return True

    def build_sequence(self, beats: list) -> list:
        """
        Build full sequence of shot events.
        """

        return [self.build_event(b) for b in beats]
