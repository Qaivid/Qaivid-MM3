"""
motion_render_prompt_builder.py

Builds video motion prompts sized to fit the active video model.

Components are added in priority order — the most essential cinematic
information comes first.  Lower-priority parts are dropped cleanly when
the budget is exhausted, so the prompt is always complete and never
truncated mid-sentence.

The max_chars budget is read from video_generator.MOTION_PROMPT_MAX_CHARS
at import time, so switching video models only requires changing one
constant in video_generator.py.
"""

from typing import Dict, List, Optional

# Read the model's prompt budget from the single source of truth.
# Fall back to 400 if video_generator can't be imported (e.g. missing deps).
try:
    from video_generator import MOTION_PROMPT_MAX_CHARS as _MODEL_MAX_CHARS
except Exception:
    _MODEL_MAX_CHARS = 400

# Component priority list — highest priority first.
# Each entry is (label, event_key).  label=None means no "label: " prefix.
_COMPONENTS = [
    (None,                  "action"),
    ("triggered by",        "trigger"),
    ("emotional shift",     "emotional_shift"),
    ("object interaction",  "object_interaction"),
    ("environment",         "environment_interaction"),
]

# Quality suffix appended only if there is room.
_QUALITY_SUFFIX = (
    "natural cinematic motion, realistic timing, "
    "no abrupt transitions, filmic quality"
)


def _camera_clause(camera: Dict) -> str:
    movement  = camera.get("movement", "")
    style     = camera.get("style", "")
    intensity = camera.get("intensity", "")
    parts     = [p for p in [movement, style, intensity] if p]
    if not parts:
        return ""
    return "camera: " + ", ".join(parts)


class MotionRenderPromptBuilder:
    def __init__(self, max_chars: Optional[int] = None):
        self._max_chars = max_chars or _MODEL_MAX_CHARS

    def build_prompt(self, event: Dict, max_chars: Optional[int] = None) -> str:
        """Build a complete motion prompt that fits within *max_chars*.

        Components are added in priority order.  A component is skipped
        (not truncated) if adding it would exceed the budget.  The camera
        clause and quality suffix are appended last.
        """
        budget = max_chars or self._max_chars

        chosen: List[str] = []

        def _fits(candidate: str) -> bool:
            tentative = ", ".join(chosen + [candidate])
            return len(tentative) <= budget

        # Priority components
        for label, key in _COMPONENTS:
            value = (event.get(key) or "").strip()
            if not value:
                continue
            clause = f"{label}: {value}" if label else value
            if _fits(clause):
                chosen.append(clause)

        # Camera clause — medium priority
        camera = event.get("camera_plan") or {}
        cam_clause = _camera_clause(camera)
        if cam_clause and _fits(cam_clause):
            chosen.append(cam_clause)

        # Quality suffix — lowest priority, only if room remains
        if _fits(_QUALITY_SUFFIX):
            chosen.append(_QUALITY_SUFFIX)

        prompt = ", ".join(chosen)
        return prompt.strip()

    def build_sequence(self, events: List[Dict]) -> List[str]:
        """Generate motion prompts for a full shot sequence."""
        return [self.build_prompt(e) for e in events]
