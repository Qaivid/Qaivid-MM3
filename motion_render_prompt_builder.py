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

# Spec motion modes — used by _motion_mode_for_intensity()
_MOTION_MODES = {
    "static":        "static hold, minimal camera movement",
    "slow_zoom_in":  "very slow zoom in, subtle push",
    "slow_zoom_out": "very slow zoom out, gentle pull",
    "pan_left":      "slow pan left, drifting reveal",
    "pan_right":     "slow pan right, drifting reveal",
    "drift":         "subtle camera drift, organic handheld quality",
}

# Aliases mapping common camera_plan.movement strings → canonical spec mode keys
_MOTION_ALIASES = {
    # static / hold
    "hold": "static", "fixed": "static", "locked": "static", "still": "static",
    "lockdown": "static", "tripod": "static", "no_movement": "static",
    "none": "static", "": "static",
    # zoom in
    "zoom_in": "slow_zoom_in", "push_in": "slow_zoom_in", "dolly_in": "slow_zoom_in",
    "slow_push": "slow_zoom_in", "push": "slow_zoom_in",
    # zoom out
    "zoom_out": "slow_zoom_out", "pull_out": "slow_zoom_out", "dolly_out": "slow_zoom_out",
    "pull_back": "slow_zoom_out", "pullback": "slow_zoom_out", "pull": "slow_zoom_out",
    # pans
    "pan": "pan_right", "slow_pan": "pan_right", "pan_r": "pan_right",
    "pan_l": "pan_left", "tilt": "pan_right", "tilt_up": "pan_right",
    "tilt_down": "pan_left", "whip_pan": "pan_right",
    # drift / handheld
    "handheld": "drift", "shake": "drift", "wobble": "drift", "float": "drift",
    "drift_left": "drift", "drift_right": "drift", "subtle_drift": "drift",
    "tracking": "drift", "track": "drift",
}


def normalize_camera_movement(raw: str) -> str:
    """Map any camera_plan.movement string to a canonical spec mode key.

    Returns one of: static, slow_zoom_in, slow_zoom_out, pan_left, pan_right, drift.
    Falls back to 'static' if the input cannot be parsed.
    """
    key = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in _MOTION_MODES:
        return key
    if key in _MOTION_ALIASES:
        return _MOTION_ALIASES[key]
    if "zoom_in" in key or "push" in key:
        return "slow_zoom_in"
    if "zoom_out" in key or "pull" in key:
        return "slow_zoom_out"
    if "pan_left" in key or "left" in key:
        return "pan_left"
    if "pan_right" in key or "right" in key or "pan" in key:
        return "pan_right"
    if "drift" in key or "hand" in key or "track" in key:
        return "drift"
    return "static"


def _camera_clause(camera: Dict) -> str:
    movement  = camera.get("movement", "")
    style     = camera.get("style", "")
    intensity = camera.get("intensity", "")
    parts     = [p for p in [movement, style, intensity] if p]
    if not parts:
        return ""
    return "camera: " + ", ".join(parts)


# ── MM3.1 Emotional Mode Intensity Thresholds ────────────────────────────────
# Per-mode overrides for the low/mid/high intensity band cutoffs.
# Keys: emotional_mode_id (matches emotional_mode_engine.py mode IDs)
# Values: (low_cutoff, high_cutoff) — replaces global (0.35, 0.65)
#   low_cutoff:  below this → static
#   high_cutoff: above this → active pan/tilt
_MODE_INTENSITY_THRESHOLDS: Dict[str, tuple] = {
    "romantic":      (0.30, 0.60),   # lower threshold — subtle drift starts earlier
    "sad_loss":      (0.25, 0.55),   # very soft — motion is restrained throughout
    "nostalgic":     (0.30, 0.60),   # gentle; memory-like drifts preferred
    "hopeful":       (0.30, 0.60),   # smooth upward energy, not harsh
    "angry_intense": (0.40, 0.60),   # compresses range — goes active quickly
    "spiritual":     (0.25, 0.55),   # very minimal movement; stillness honoured
    "energetic":     (0.40, 0.65),   # high floor — nearly always active
}


def _motion_mode_for_intensity(
    emotional_intensity: float,
    motion_philosophy: str,
    emotional_mode_id: str = "",
) -> str:
    """Return a spec motion mode string scaled to emotional intensity and philosophy.

    Default intensity bands (spec rule):
        < 0.35   → static / minimal  (still_dominant philosophies held here)
        0.35–0.65 → subtle drift or slow zoom
        > 0.65   → active pan/tilt aligned with motion_philosophy direction

    When emotional_mode_id is supplied, per-mode thresholds from
    _MODE_INTENSITY_THRESHOLDS override the defaults, so e.g. a spiritual
    song stays minimal longer than an energetic one at the same intensity.

    motion_philosophy values: still_dominant | dynamic_dominant | mixed
    """
    phi = (motion_philosophy or "mixed").lower().strip()

    thresholds = _MODE_INTENSITY_THRESHOLDS.get(emotional_mode_id or "", (0.35, 0.65))
    low_cut, high_cut = thresholds

    if emotional_intensity < low_cut:
        return _MOTION_MODES["static"]

    if emotional_intensity <= high_cut:
        if phi == "still_dominant":
            return _MOTION_MODES["slow_zoom_in"]
        return _MOTION_MODES["drift"]

    # High intensity (> high_cut) — active pan/tilt for all philosophies
    # Direction is tuned by motion_philosophy.
    if phi == "still_dominant":
        return _MOTION_MODES["pan_left"]    # controlled active pan
    if phi == "dynamic_dominant":
        return _MOTION_MODES["pan_right"]   # vigorous lateral motion
    return _MOTION_MODES["pan_right"]       # mixed: default active pan


def _intensity_float(intensity) -> float:
    """Convert an emotional_intensity value to a 0.0–1.0 float.

    Accepts both numeric values (0.0–1.0, passed through directly) and
    the legacy label strings (low/medium/high/peak).  Safe against
    unexpected types — always returns a clamped float.
    """
    if isinstance(intensity, (int, float)):
        return float(max(0.0, min(1.0, intensity)))
    _MAP = {
        "low":    0.2,
        "medium": 0.5,
        "high":   0.75,
        "peak":   0.95,
    }
    label = str(intensity or "medium").lower().strip()
    return _MAP.get(label, 0.5)


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


def build_video_clip_prompt(
    shot: Dict,
    *,
    motion_philosophy: str = "mixed",
    identity_seed: Optional[str] = None,
    cinematic_style: Optional[str] = None,
    lighting_logic: Optional[str] = None,
    continuity_rules: Optional[List[str]] = None,
    max_chars: int = _MODEL_MAX_CHARS,
    emotional_mode_id: str = "",
) -> str:
    """Build a brain-aware WAN 2.6 Flash video prompt for a single shot.

    Constructs the prompt strictly from provided brain data — no invented
    content.  Components (in priority order):

        1. Subject / identity (identity_seed or shot subject)
        2. Action (shot action field)
        3. Environment (shot environment_interaction or location world)
        4. Lighting (style_packet.lighting_logic)
        5. Emotion (shot emotional_micro_state)
        6. Cinematic style hint (style_packet.cinematic_style)
        7. Motion instruction (spec motion mode scaled by emotional_intensity)
        8. Continuity rule excerpt (first rule, if present)
        9. Quality suffix (if budget allows)

    The output is always <= max_chars and never truncated mid-sentence.
    """
    budget = max_chars
    chosen: List[str] = []

    def _fits(candidate: str) -> bool:
        tentative = ", ".join(chosen + [candidate])
        return len(tentative) <= budget

    def _add(clause: str) -> None:
        clause = clause.strip()
        if clause and _fits(clause):
            chosen.append(clause)

    # 1. Subject / identity
    subject = (identity_seed or shot.get("subject") or shot.get("action") or "").strip()
    if subject:
        _add(subject)

    # 2. Action (if distinct from subject)
    action = (shot.get("action") or "").strip()
    if action and action != subject:
        _add(action)

    # 3. Environment
    env = (shot.get("environment_interaction") or shot.get("environment") or "").strip()
    if env:
        _add(env)

    # 4. Lighting
    if lighting_logic:
        _add(lighting_logic.strip())

    # 5. Emotional micro-state
    emotion = (shot.get("emotional_micro_state") or shot.get("emotional_shift") or "").strip()
    if emotion:
        _add(emotion)

    # 6. Cinematic style hint (brief)
    if cinematic_style:
        hint = cinematic_style.strip()
        _add(hint)

    # 7. Motion instruction — scaled to emotional_intensity + motion_philosophy + mode
    raw_intensity = shot.get("emotional_intensity") or "medium"
    intensity_f = _intensity_float(raw_intensity)
    motion_clause = _motion_mode_for_intensity(intensity_f, motion_philosophy, emotional_mode_id)
    if motion_clause:
        _add(motion_clause)

    # 8. First continuity rule (brief excerpt — strictly from brain)
    rules = continuity_rules or []
    if rules:
        rule = str(rules[0]).strip()
        if len(rule) > 80:
            rule = rule[:80].rsplit(" ", 1)[0]
        if rule:
            _add(rule)

    prompt = ", ".join(chosen)
    return prompt.strip()
