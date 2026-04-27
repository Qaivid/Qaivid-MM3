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
    "romantic":               (0.45, 0.70),   # STILLER — wide static zone; drift only at high intensity
    "sad_loss":               (0.50, 0.75),   # STILLEST — very restrained; active only at peak
    "nostalgic":              (0.40, 0.65),   # slight stillness bias; memory-drift aesthetic
    "hopeful":                (0.30, 0.60),   # standard; gentle upward energy permitted
    "angry_intense":          (0.20, 0.45),   # MOST ACTIVE — motion kicks in at low intensity
    "spiritual_reflective":   (0.50, 0.75),   # STILLEST alongside sad_loss; stillness is the message
    "energetic_celebration":  (0.15, 0.40),   # MOST ACTIVE — nearly always moving
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
    _MODE_INTENSITY_THRESHOLDS override the defaults, so e.g. a spiritual_reflective
    song stays minimal longer than an energetic_celebration one at the same intensity.

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


def _pick(shot: Dict, *keys: str) -> str:
    """Return the first non-empty string value from shot for any of the given keys."""
    for key in keys:
        val = (shot.get(key) or "").strip()
        if val:
            return val
    return ""


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
    vibe_shot_direction: str = "",
    vibe_avoid: Optional[List[str]] = None,
) -> str:
    """Build a brain-aware WAN 2.6 Flash video prompt for a single shot.

    Reads timeline fields with full fallback coverage — the timeline builder
    uses different key names across versions (e.g. chosen_direction vs action,
    emotional_state vs emotional_micro_state, intensity vs emotional_intensity).
    All known aliases are checked so no story data is silently dropped.

    Components (in priority order):
        0. Vibe shot direction (cultural/aesthetic identity)
        1. Subject / identity (identity_seed or shot subject)
        2. Scene action / direction (chosen_direction → action → visual_prompt excerpt)
        3. Environment (environment_interaction → environment_type → key_elements)
        4. Lighting (style_packet.lighting_logic → shot lighting_style)
        5. Emotion (emotional_micro_state → emotional_state → meaning)
        6. Atmosphere / meaning (atmosphere_profile → meaning for story weight)
        7. Cinematic style hint
        8. Motion instruction (scaled to intensity + philosophy + mode)
        9. Continuity rule excerpt
       10. Quality suffix (if budget allows)

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

    # 0. Vibe shot direction — cultural/aesthetic identity of the production.
    if vibe_shot_direction:
        _add(vibe_shot_direction.strip())

    # 1. Subject / identity — identity_seed is the authoritative physical anchor.
    subject = (identity_seed or "").strip()
    if not subject:
        subject = _pick(shot, "subject", "character_name")
    if subject:
        _add(subject)

    # 2. Scene action / direction — what is HAPPENING in this shot.
    # Timeline v2: chosen_direction / _v2_chosen_direction
    # MM3.1+:      action / shot_event
    # Fallback:    first 120 chars of visual_prompt (contains the scene sentence)
    action = _pick(shot,
                   "chosen_direction", "_v2_chosen_direction",
                   "action", "shot_event")
    if not action:
        vp = (shot.get("visual_prompt") or "").strip()
        if vp:
            # Take only the first sentence — avoids lyric fragments / noise.
            first = vp.split(".")[0].strip()
            if len(first) > 20:
                action = first[:120]
    if action and action != subject:
        _add(action)

    # 3. Environment — setting / physical space.
    # Timeline v2: environment_type (str), key_elements (list)
    # MM3.1+:      environment_interaction, environment
    env = _pick(shot, "environment_interaction", "environment",
                "environment_type")
    if not env:
        # Build from environment_profile dict
        ep = shot.get("environment_profile") or {}
        env = (ep.get("environment_type") or "").strip()
    if env:
        _add(env)

    # Append up to 3 key visual elements as enrichment.
    key_elems = shot.get("key_elements") or []
    if isinstance(key_elems, list) and key_elems:
        elems_str = ", ".join(str(e).strip() for e in key_elems[:3] if e)
        if elems_str:
            _add(elems_str)

    # 4. Lighting — brain style_packet first, then shot-level field.
    light = (lighting_logic or "").strip() or _pick(shot, "lighting_style", "lighting_condition")
    if light:
        _add(light)

    # 5. Emotional state / micro-state.
    # MM3.1+:      emotional_micro_state, emotional_shift
    # Timeline v2: emotional_state, emotional_tone
    emotion = _pick(shot,
                    "emotional_micro_state", "emotional_shift",
                    "emotional_state", "emotional_tone")
    if emotion:
        _add(emotion)

    # 6. Atmosphere / story meaning — gives WAN the narrative weight of the shot.
    # Trimmed to 80 chars so it doesn't eat the whole budget.
    meaning = _pick(shot, "atmosphere_profile", "meaning")
    if meaning and len(meaning) > 80:
        meaning = meaning[:80].rsplit(" ", 1)[0]
    if meaning:
        _add(meaning)

    # 7. Cinematic style hint (brief).
    if cinematic_style:
        _add(cinematic_style.strip())

    # 8. Motion instruction — scaled to emotional_intensity + motion_philosophy + mode.
    # Timeline v2 uses "intensity" (float 0–1); MM3.1 uses "emotional_intensity".
    raw_intensity = (
        shot.get("emotional_intensity")
        or shot.get("intensity")
        or shot.get("audio_intensity")
        or "medium"
    )
    intensity_f = _intensity_float(raw_intensity)
    motion_clause = _motion_mode_for_intensity(intensity_f, motion_philosophy, emotional_mode_id)
    if motion_clause:
        _add(motion_clause)

    # 9. First continuity rule (brief excerpt — strictly from brain).
    rules = continuity_rules or []
    if rules:
        rule = str(rules[0]).strip()
        if len(rule) > 80:
            rule = rule[:80].rsplit(" ", 1)[0]
        if rule:
            _add(rule)

    # 10. Vibe avoid — compact negative clause.
    # WAN 2.6 has no negative prompt field; avoid terms go in the positive prompt.
    _avoid_items = [str(a).strip() for a in (vibe_avoid or []) if a and str(a).strip()]
    if _avoid_items:
        _avoid_clause = "avoid: " + ", ".join(_avoid_items[:4])
        _add(_avoid_clause)

    # Quality suffix — appended last only if budget allows.
    if _fits(_QUALITY_SUFFIX):
        chosen.append(_QUALITY_SUFFIX)

    prompt = ", ".join(chosen)
    return prompt.strip()
