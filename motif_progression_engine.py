"""
motif_progression_engine.py

Purpose:
Manage recurring motifs and evolve them across a sequence instead of repeating flatly.
"""

from typing import List, Dict


class MotifProgressionEngine:
    def __init__(self):
        self.progression_stages = [
            "introduction",
            "reinforcement",
            "variation",
            "distortion",
            "resolution"
        ]

    def apply_progression(self, events: List[Dict]) -> List[Dict]:
        """
        Assign progression stage to each recurring motif.
        """

        motif_counter = {}
        for event in events:
            motif = event.get("object_interaction") or "generic"

            if motif not in motif_counter:
                motif_counter[motif] = 0

            stage_index = min(motif_counter[motif], len(self.progression_stages) - 1)
            event["motif_stage"] = self.progression_stages[stage_index]

            motif_counter[motif] += 1

        return events

    def evolve_motif(self, event: Dict) -> Dict:
        """
        Slightly modify event behaviour based on motif stage.
        """

        stage = event.get("motif_stage")

        if stage == "variation":
            event["action"] = f"variation of {event.get('action', '')}"
        elif stage == "distortion":
            event["action"] = f"distorted version of {event.get('action', '')}"
        elif stage == "resolution":
            event["action"] = f"final version of {event.get('action', '')}"

        return event

    def apply_full_progression(self, events: List[Dict]) -> List[Dict]:
        """
        Apply both stage assignment and evolution.
        """

        events = self.apply_progression(events)
        return [self.evolve_motif(e) for e in events]
