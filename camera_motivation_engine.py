"""
camera_motivation_engine.py

Purpose:
Assign camera behaviour based on action/event (not emotion).
"""

from typing import Dict, List


class CameraMotivationEngine:
    def __init__(self):
        pass

    def assign_camera(self, event: Dict) -> Dict:
        """
        Determine camera behaviour from subject action.
        """

        action = (event.get("action") or "").lower()

        camera_plan = {
            "movement": "static",
            "style": "locked",
            "intensity": "low"
        }

        if any(k in action for k in ["walk", "move", "approach"]):
            camera_plan = {
                "movement": "follow",
                "style": "steady",
                "intensity": "medium"
            }

        elif any(k in action for k in ["freeze", "pause", "stop"]):
            camera_plan = {
                "movement": "none",
                "style": "locked",
                "intensity": "low"
            }

        elif any(k in action for k in ["drop", "fall", "slip"]):
            camera_plan = {
                "movement": "snap",
                "style": "quick_cut",
                "intensity": "high"
            }

        elif any(k in action for k in ["turn", "look back"]):
            camera_plan = {
                "movement": "slow_push",
                "style": "cinematic",
                "intensity": "medium"
            }

        event["camera_plan"] = camera_plan
        return event

    def apply_to_sequence(self, events: List[Dict]) -> List[Dict]:
        """
        Apply camera logic across sequence.
        """

        return [self.assign_camera(e) for e in events]
