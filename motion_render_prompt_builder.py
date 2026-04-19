"""
motion_render_prompt_builder.py

Purpose:
Generate motion/render prompts for video generation.
Focus: how the shot unfolds over time (not just a still moment).
"""

from typing import Dict, List


class MotionRenderPromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, event: Dict) -> str:
        """
        Convert shot event into a motion/render prompt.
        """

        action = event.get("action", "")
        trigger = event.get("trigger", "")
        shift = event.get("emotional_shift", "")
        obj = event.get("object_interaction", "")
        env = event.get("environment_interaction", "")
        camera = event.get("camera_plan", {})

        movement = camera.get("movement", "static")
        style = camera.get("style", "cinematic")
        intensity = camera.get("intensity", "low")

        parts = [
            f"start: {action}" if action else "",
            f"triggered by: {trigger}" if trigger else "",
            f"emotional shift: {shift}" if shift else "",
            f"object interaction: {obj}" if obj else "",
            f"environment: {env}" if env else "",
            f"camera movement: {movement}, style: {style}, intensity: {intensity}",
            "natural cinematic motion, realistic timing, no abrupt transitions, filmic quality"
        ]

        prompt = ", ".join([p for p in parts if p])
        return prompt.strip()

    def build_sequence(self, events: List[Dict]) -> List[str]:
        """
        Generate motion prompts for full sequence.
        """

        return [self.build_prompt(e) for e in events]
