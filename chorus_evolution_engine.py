"""
chorus_evolution_engine.py

Purpose:
Evolve repeated chorus shots so they do not visually repeat.
Each recurrence escalates or shifts emotional state.
"""

from typing import List, Dict


class ChorusEvolutionEngine:
    def __init__(self):
        self.evolution_stages = [
            "hope",
            "anticipation",
            "doubt",
            "hurt",
            "acceptance",
            "numbness"
        ]

    def apply_evolution(self, events: List[Dict]) -> List[Dict]:
        """
        Apply evolution stages to repeated chorus-tagged events.
        """

        stage_index = 0

        for event in events:
            if event.get("is_chorus"):
                stage = self.evolution_stages[min(stage_index, len(self.evolution_stages) - 1)]
                event["chorus_stage"] = stage
                event = self._evolve_event(event, stage)
                stage_index += 1

        return events

    def _evolve_event(self, event: Dict, stage: str) -> Dict:
        """
        Modify event based on emotional stage.
        """

        action = event.get("action", "")

        if stage == "hope":
            event["action"] = f"{action} with expectation"
        elif stage == "anticipation":
            event["action"] = f"{action} more urgently"
        elif stage == "doubt":
            event["action"] = f"{action} with hesitation"
        elif stage == "hurt":
            event["action"] = f"{action} but slows down in pain"
        elif stage == "acceptance":
            event["action"] = f"{action} calmly without reaction"
        elif stage == "numbness":
            event["action"] = f"{action} mechanically with no emotion"

        return event
