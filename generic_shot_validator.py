"""
generic_shot_validator.py

Purpose:
Detect and rewrite weak / generic shots that lack cinematic value.

MM3.1 design decision:
- MARKS events as generic rather than removing them (index continuity preserved
  so _shot_events_by_index maps correctly to the storyboard's line_meanings).
- REWRITES generic events in-place: patches missing/weak `action` and
  `environment_usage` fields with mode-appropriate fallbacks so every shot
  that exits this validator carries a verb-bearing action and an explicit
  environment interaction.
- is_generic=True / is_valid=False flags are set so callers can inspect
  whether a rewrite occurred.
"""

from typing import Dict, List, Optional

# ── Mode-appropriate fallback actions (verb-bearing) ──────────────────────────
_FALLBACK_ACTIONS: Dict[str, str] = {
    "face":        "turns slightly, expression shifting as a private thought passes",
    "body":        "pauses mid-movement, weight settling as something catches attention",
    "environment": "light shifts across the empty space, marking the passage of a moment",
    "macro":       "fingers trace the object's surface, detail catching the available light",
    "symbolic":    "silhouette stands at the threshold between shadow and open air",
}
_DEFAULT_FALLBACK_ACTION = "stands still, presence filling the space around them"

# ── Mode-appropriate fallback environment interactions ────────────────────────
_FALLBACK_ENV_USAGE: Dict[str, str] = {
    "face":        "the surrounding light and shadow frame the character's expression",
    "body":        "the character's movement responds to the texture and scale of the space",
    "environment": "the empty space itself becomes the subject, atmosphere rendered tangible",
    "macro":       "the object's detail anchors the viewer within the physical environment",
    "symbolic":    "the character's form echoes or contrasts with the surrounding architecture",
}
_DEFAULT_FALLBACK_ENV = "character occupies and interacts with the surrounding space"


class GenericShotValidator:
    def __init__(self):
        self.generic_patterns = [
            "looking away",
            "emotional eyes",
            "distant gaze",
            "sad expression",
            "cinematic portrait",
        ]

    # ── Detection ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_env(event: Dict) -> str:
        """Canonical environment field resolver — ShotEventBuilder emits
        `environment_interaction`; some callers use `environment_usage`.
        Both are treated as equivalent; whichever is non-empty wins.
        """
        return (
            (event.get("environment_usage") or "")
            or (event.get("environment_interaction") or "")
        ).strip()

    def is_generic(self, event: Dict) -> bool:
        """Return True if the shot lacks sufficient cinematic specificity.

        A shot is generic if:
        - action is absent or shorter than 3 words (no concrete behaviour)
        - action contains a known empty-phrase pattern
        - environment_usage / environment_interaction is absent (no spatial grounding)
        """
        action = (event.get("action") or event.get("subject_action") or "").strip()
        contrast = (event.get("visual_contrast") or "").strip()
        camera = (event.get("camera_motivation") or "").strip()
        env_field = self._get_env(event)

        # Absent or skeletal action
        if not action or len(action.split()) < 3:
            return True

        # Known filler phrases
        text_blob = f"{action} {contrast} {camera}".lower()
        for pattern in self.generic_patterns:
            if pattern in text_blob:
                return True

        # No spatial grounding in either env field
        if not env_field:
            return True

        return False

    # ── Rewrite ───────────────────────────────────────────────────────────────

    def rewrite_generic(self, event: Dict) -> Dict:
        """Patch a generic event in-place with mode-appropriate fallbacks.

        - Overwrites a weak/absent `action` (and its `subject_action` alias)
          with a verb-bearing fallback.
        - Fills missing environment spatial grounding in BOTH canonical fields
          (`environment_usage` and `environment_interaction`) so whichever the
          downstream caller reads, it gets a non-empty value.
        Original values are preserved under `_original_*` keys for debugging.
        """
        mode = (event.get("expression_mode") or "face").lower()

        # --- action fallback ---
        action = (event.get("action") or event.get("subject_action") or "").strip()
        if not action or len(action.split()) < 3:
            fallback = _FALLBACK_ACTIONS.get(mode, _DEFAULT_FALLBACK_ACTION)
            event["_original_action"] = action
            event["action"] = fallback
            event["subject_action"] = fallback

        # --- environment fallback (both alias fields) ---
        if not self._get_env(event):
            fallback_env = _FALLBACK_ENV_USAGE.get(mode, _DEFAULT_FALLBACK_ENV)
            event["_original_environment_usage"] = ""
            event["environment_usage"] = fallback_env
            event["environment_interaction"] = fallback_env  # alias

        return event

    # ── Main entry ────────────────────────────────────────────────────────────

    def validate_sequence(self, events: List[Dict]) -> List[Dict]:
        """Mark and rewrite each event in the sequence.

        - Marks every event with is_generic / is_valid flags.
        - Rewrites generic events in-place so they exit with a concrete action
          and a spatial environment_usage (required by shot_prompt_composer).
        - Never removes events — list length is preserved for index alignment.
        """
        for event in events:
            generic = self.is_generic(event)
            event["is_generic"] = generic
            event["is_valid"] = not generic
            if generic:
                self.rewrite_generic(event)

        return events
