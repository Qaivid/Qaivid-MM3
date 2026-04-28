"""
motion_render_prompt_builder.py

Builds video motion prompts sized to fit the active video model.

Output format: labelled sections separated by " | " so WAN 2.6 can parse
a clear hierarchy of meaning rather than a flat comma-joined string.

    Style: <vibe direction> | Scene: <subject, action> | Environment: <setting, elements>
    | Lighting: <lighting> | Mood: <emotion, atmosphere> | Camera: <motion, quality>
    | Avoid: <negative terms>

Each section is omitted (never truncated) if it would push the output over the
budget.  The budget is read from video_generator.MOTION_PROMPT_MAX_CHARS at
import time, so switching video models only requires changing one constant.
"""

from typing import Dict, List, Optional

# Read the model's prompt budget from the single source of truth.
# Fall back to 400 if video_generator can't be imported (e.g. missing deps).
try:
    from video_generator import MOTION_PROMPT_MAX_CHARS as _MODEL_MAX_CHARS
except Exception:
    _MODEL_MAX_CHARS = 400

# Section separator used to join the labelled blocks.
_SEP = " | "

# Quality clause folded into the Camera section.
_QUALITY_CLAUSE = (
    "natural cinematic motion, realistic timing, "
    "no abrupt transitions, filmic quality"
)

# Legacy quality suffix — kept for MotionRenderPromptBuilder compatibility.
_QUALITY_SUFFIX = _QUALITY_CLAUSE

# Spec motion modes — used by _motion_mode_for_intensity()
_MOTION_MODES = {
    "static":        "static hold, minimal camera movement",
    "slow_zoom_in":  "very slow zoom in, subtle push",
    "slow_zoom_out": "very slow zoom out, gentle pull",
    "pan_left":      "slow pan left, drifting reveal",
    "pan_right":     "slow pan right, drifting reveal",
    "drift":         "subtle camera drift, organic handheld quality",
}

# Component priority list for the legacy MotionRenderPromptBuilder.
# Each entry is (label, event_key).  label=None means no "label: " prefix.
_COMPONENTS = [
    (None,                  "action"),
    ("triggered by",        "trigger"),
    ("emotional shift",     "emotional_shift"),
    ("object interaction",  "object_interaction"),
    ("environment",         "environment_interaction"),
]

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_trailing_punct(text: str) -> str:
    """Strip trailing periods, commas, and semicolons so clauses join cleanly."""
    return text.rstrip(".,;: ").strip()


def _truncate_to_clause(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, always ending at a complete clause.

    Splits on ', ' boundaries and drops the last fragment when the full text
    is too long.  This prevents dangling prepositional phrases such as
    "aligned with the," that appear when truncating by character position alone.
    Returns an empty string if even the first clause exceeds max_chars.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return _clean_trailing_punct(text)
    parts = text.split(", ")
    result = ""
    for part in parts:
        candidate = f"{result}, {part}" if result else part
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break
    return _clean_trailing_punct(result)


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


# ── Legacy builder ────────────────────────────────────────────────────────────

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

        # Priority components — strip trailing punctuation before joining.
        for label, key in _COMPONENTS:
            value = _clean_trailing_punct((event.get(key) or "").strip())
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick(shot: Dict, *keys: str) -> str:
    """Return the first non-empty string value from shot for any of the given keys."""
    for key in keys:
        val = (shot.get(key) or "").strip()
        if val:
            return val
    return ""


# ── Main builder ──────────────────────────────────────────────────────────────

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

    Output format — labelled sections separated by ' | ':

        Style: <vibe direction> | Scene: <subject, action> | Environment: <setting>
        | Lighting: <lighting> | Mood: <emotion, atmosphere> | Camera: <motion, quality>
        | Avoid: <negative terms>

    This gives WAN 2.6 a clear parsing hierarchy.  Each section is included only
    if its content is non-empty and it fits within *max_chars*; sections are never
    truncated mid-clause.

    Data sources (unchanged from previous flat format):
        Style    ← vibe_shot_direction (brain style_packet)
        Scene    ← identity_seed / character_name, chosen_direction / action / shot_event
        Environ  ← environment_interaction / environment_type, key_elements (up to 3)
        Lighting ← lighting_logic (brain) / lighting_style / lighting_condition
        Mood     ← emotional_micro_state / emotional_shift / emotional_state,
                   atmosphere_profile / meaning (truncated to a clean clause boundary)
        Camera   ← cinematic_style (brain), motion instruction (intensity-scaled),
                   first continuity rule, quality clause
        Avoid    ← vibe_avoid (brain, up to 4 items)
    """
    budget = max_chars
    sections: List[str] = []

    def _current_len(extra_section: str = "") -> int:
        """Length of the assembled prompt if *extra_section* were appended."""
        all_secs = sections + ([extra_section] if extra_section else [])
        return len(_SEP.join(all_secs))

    def _add_section(label: str, content: str) -> None:
        """Append a labelled section if it fits in the budget."""
        content = content.strip()
        if not content:
            return
        sec = f"{label}: {content}" if label else content
        if _current_len(sec) <= budget:
            sections.append(sec)

    # ── Style — vibe identity, goes first as production fingerprint ───────────
    if vibe_shot_direction:
        vsd = _clean_trailing_punct(vibe_shot_direction.strip())
        if vsd:
            _add_section("Style", vsd)

    # ── Scene — who + what is happening ──────────────────────────────────────
    scene_parts: List[str] = []

    subject = _clean_trailing_punct((identity_seed or "").strip())
    if not subject:
        subject = _clean_trailing_punct(_pick(shot, "subject", "character_name"))
    if subject:
        scene_parts.append(subject)

    # chosen_direction ends with a period on many timeline entries — strip it.
    action = _pick(shot, "chosen_direction", "_v2_chosen_direction", "action", "shot_event")
    if not action:
        vp = (shot.get("visual_prompt") or "").strip()
        if vp:
            first = vp.split(".")[0].strip()
            if len(first) > 20:
                action = first[:120]
    action = _clean_trailing_punct(action.strip()) if action else ""
    if action and action != subject:
        scene_parts.append(action)

    if scene_parts:
        _add_section("Scene", ", ".join(scene_parts))

    # ── Environment — setting + key visual elements ───────────────────────────
    env_parts: List[str] = []

    env = _pick(shot, "environment_interaction", "environment", "environment_type")
    if not env:
        ep = shot.get("environment_profile") or {}
        env = (ep.get("environment_type") or "").strip()
    if env:
        env_parts.append(_clean_trailing_punct(env))

    key_elems = shot.get("key_elements") or []
    if isinstance(key_elems, list) and key_elems:
        env_parts.extend(
            str(e).strip() for e in key_elems[:3] if e and str(e).strip()
        )

    if env_parts:
        _add_section("Environment", ", ".join(env_parts))

    # ── Lighting — brain style_packet first, then shot-level field ────────────
    light = _clean_trailing_punct(
        (lighting_logic or "").strip()
        or _pick(shot, "lighting_style", "lighting_condition")
    )
    if light:
        _add_section("Lighting", light)

    # ── Mood — emotional state + atmosphere/meaning ───────────────────────────
    mood_parts: List[str] = []

    emotion = _pick(
        shot,
        "emotional_micro_state", "emotional_shift",
        "emotional_state", "emotional_tone",
    )
    if emotion:
        mood_parts.append(_clean_trailing_punct(emotion))

    # atmosphere_profile can be very long ("aligned with the emotional meaning of …")
    # — truncate to a complete clause boundary so no dangling phrase leaks through.
    atmosphere = _pick(shot, "atmosphere_profile", "meaning")
    if atmosphere:
        atmosphere = _truncate_to_clause(atmosphere, 100)
    if atmosphere:
        mood_parts.append(atmosphere)

    if mood_parts:
        _add_section("Mood", ", ".join(mood_parts))

    # ── Camera — cinematic style + motion + continuity + quality ──────────────
    # Quality clause is the lowest priority within this section: we first try
    # to include it, and fall back to the section without it if budget is tight.
    # This ensures the motion instruction is never dropped just because quality
    # would push the total over the limit.
    camera_parts: List[str] = []

    if cinematic_style:
        camera_parts.append(_clean_trailing_punct(cinematic_style.strip()))

    raw_intensity = (
        shot.get("emotional_intensity")
        or shot.get("intensity")
        or shot.get("audio_intensity")
        or "medium"
    )
    intensity_f = _intensity_float(raw_intensity)
    motion_clause = _motion_mode_for_intensity(intensity_f, motion_philosophy, emotional_mode_id)
    if motion_clause:
        camera_parts.append(motion_clause)

    rules = continuity_rules or []
    if rules:
        rule = _truncate_to_clause(str(rules[0]).strip(), 80)
        if rule:
            camera_parts.append(rule)

    if camera_parts:
        core_content = ", ".join(camera_parts)
        full_content = core_content + ", " + _QUALITY_CLAUSE
        # Prefer full content (with quality); fall back to core if it doesn't fit.
        if _current_len(f"Camera: {full_content}") <= budget:
            _add_section("Camera", full_content)
        else:
            _add_section("Camera", core_content)

    # ── Avoid — compact negative clause (WAN has no dedicated negative field) ─
    avoid_items = [str(a).strip() for a in (vibe_avoid or []) if a and str(a).strip()]
    if avoid_items:
        _add_section("Avoid", ", ".join(avoid_items[:4]))

    return _SEP.join(sections).strip()
