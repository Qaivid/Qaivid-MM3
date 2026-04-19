"""
still_keyframe_prompt_builder.py

Purpose:
Generate strong still-image prompts from shot events.
Focus: one frozen cinematic moment (NOT generic description).
"""

from typing import Dict, List


class StillKeyframePromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, event: Dict) -> str:
        """
        Convert shot event into a still-image prompt.
        """

        action = event.get("action", "")
        obj = event.get("object_interaction", "")
        env = event.get("environment_interaction", "")
        contrast = event.get("visual_contrast", "")
        camera = event.get("camera_plan", {})

        movement = camera.get("movement", "static")
        style = camera.get("style", "cinematic")

        prompt_parts = [
            action,
            f"interacting with {obj}" if obj else "",
            f"within {env}" if env else "",
            f"visual tension: {contrast}" if contrast else "",
            f"camera: {movement} {style}",
            "cinematic lighting, natural skin texture, shallow depth of field, high detail"
        ]

        # Clean empty parts
        prompt = ", ".join([p for p in prompt_parts if p])

        return prompt.strip()

    def build_sequence(self, events: List[Dict]) -> List[str]:
        """
        Generate prompts for full sequence.
        """

        return [self.build_prompt(e) for e in events]
