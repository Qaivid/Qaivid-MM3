"""V3 Timeline Builder — thin decorator over the v3 storyboard + v3 brief.

This is the Phase-3 "shrink" replacement for ``timeline_builder_v2``.
The v3 storyboard engine (Phase 1) and v3 creative-brief engine (Phase 2)
between them already commit:

  * the exact list of shots,
  * each shot's start_time / end_time / duration (locked, ≤15s by
    construction in storyboard_engine_v3 — narrative shots may span up
    to the storyboard cap; any shot >8s carries an `actions[]`
    decomposition where each action is ≤8s for Phase-4 WAN/Kling
    rendering),
  * each shot's shot_id, scene_id, lyric_text, action_intent (locked),
  * each shot's enriched_direction (action_intent + execution_detail),
  * each shot's `actions[]` multishot decomposition (when present).

Everything that timeline_builder_v2 had to *derive* — lyric-unit-to-scene
assignment, shots-per-scene chunking, duration calculation, beat-snap,
lyric-anchor pass, audio-duration normalization, the ≤8s "splitter"
(the v2 builder used to slice over-long shots itself), fallback-shot
synthesis — is therefore redundant under v3 and is dropped here.  The
v3 builder only *validates* the multishot contract (actions exist when
duration>8s, each action ≤8s, sum(actions.duration)≈shot.duration) and
carries `actions[]` through to Phase 4.  What remains is purely *decoration*:

  * compose visual_prompt from enriched_direction + brief intent fields,
  * derive expression_mode / motion_scale / transition / framing,
  * build camera_profile / environment_profile / continuity_anchor,
  * stamp shot_type via ShotVarietyEngine,
  * apply StyleGradingEngine,
  * merge brief-intent passthrough fields back after grading.

Because every shot is sourced one-for-one from the v3 brief, the output
``styled_timeline`` has the *same* count and *same* timings as the
storyboard.  Downstream stages (materializer, references, stills, video,
final cut) consume the same schema as before.

Public API
----------
build_timeline_from_brief_v3(
    brief_packet,           # output of creative_brief_engine_v3 (dict
                            # with `scenes`: each scene has `shots[]`
                            # with locked metadata + enriched_direction)
    style_packet,
    narrative_packet,
    emotional_mode_packet,
    audio_data,
) -> List[Dict]

Phase 3 is *standalone*: pipeline_worker.py is NOT modified by this
file.  v2 remains in production until Phase 5 wiring.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from shot_variety_engine import ShotVarietyEngine

logger = logging.getLogger(__name__)

# ── Pacing defaults (only used to compute start_beat/bar_index labels) ───────
_DEFAULT_BPM             = 120.0
_DEFAULT_BEATS_PER_BAR   = 4
_DEFAULT_INTENSITY       = 0.5
_VIDEO_MAX_DURATION      = 8    # WAN/Kling per-render hard cap
                                # (per-action, NOT per-shot — a shot
                                # over this cap is decomposed into
                                # multishot `actions[]` upstream)
_STORYBOARD_MAX_SHOT      = 15   # narrative-shot cap from
                                # storyboard_engine_v3._enforce_max_shot_duration

# ── Intensity mapping from brief emotional_intensity string ──────────────────
_INTENSITY_MAP: Dict[str, float] = {
    "low":    0.3,
    "medium": 0.6,
    "high":   0.9,
    "peak":   1.0,
}

# ── expression_mode derivation from brief subject_focus + character_presence ─
_SUBJECT_TO_MODE: Dict[str, str] = {
    "character":   "face",
    "environment": "environment",
    "object":      "macro",
    "mixed":       "symbolic",
}

# ── Motion scale templates keyed by expression_mode ──────────────────────────
_MOTION_SCALE: Dict[str, List[str]] = {
    "face": [
        "minimal motion, intimate hold",
        "barely perceptible drift toward face",
        "static hold with breath-weight stillness",
        "subtle rack focus shift during hold",
    ],
    "body": [
        "gentle cinematic drift",
        "slow body-weight follow",
        "static frame, subject carries motion",
        "slow drift across body plane",
    ],
    "environment": [
        "slow environmental drift",
        "wide static hold, scene breathes naturally",
        "gentle push through open space",
        "slow tilt across landscape",
    ],
    "symbolic": [
        "slow dolly toward object, background gradually blurs",
        "static with organic atmospheric movement",
        "gentle parallax shift revealing depth in scene",
        "slow pull back from detail to wider symbolic context",
    ],
    "macro": [
        "precise micro-motion, premium controlled movement",
        "static macro hold, subject depth-of-field breathing",
        "slow push-in on surface detail",
        "barely perceptible circular drift around focal point",
    ],
}


# ── Decoration helpers ───────────────────────────────────────────────────────
def _derive_transition(shot_index: int, current_mode: str,
                       intensity: float) -> str:
    if shot_index == 0:
        return "opening_hold"
    if current_mode == "symbolic":
        return "poetic_dissolve"
    if intensity > 0.82:
        return "hard_cut"
    return "straight_cut"


def _build_camera_profile(movement_type: str, motion_density: str) -> Dict[str, Any]:
    mt = (movement_type or "slow").lower()
    md = (motion_density or "medium").lower()
    return {
        "movement":         mt,
        "motion_density":   md,
        "stabilisation":    "gimbal" if mt == "dynamic" else "tripod",
        "depth_of_field":   "shallow" if md == "low" else "deep",
    }


def _compose_visual_prompt(
    scene: Dict[str, Any],
    enriched_direction: str,
    lyric_text: str = "",
) -> str:
    """Compose a visual prompt string from v3 brief fields.

    enriched_direction comes from the v3 brief shot (verbatim composed
    in creative_brief_engine_v3 as `action_intent — execution_detail`),
    so the storyboard's shot-level visual concept is preserved exactly
    and only enriched.
    """
    parts: List[str] = []

    chosen = (enriched_direction or "").strip()
    if chosen:
        parts.append(chosen)

    keys = list(scene.get("key_elements") or [])
    if keys:
        parts.append(", ".join(str(k) for k in keys[:4]))

    env = str(scene.get("environment_type") or "").strip()
    if env:
        parts.append(env)

    lighting = str(scene.get("lighting_condition") or "").strip()
    if lighting:
        parts.append(lighting)

    emotion = str(scene.get("emotional_state") or "").strip()
    if emotion:
        parts.append(emotion)

    lt = (lyric_text or "").strip()
    if lt:
        parts.append(f'"{lt}"')

    return ". ".join(p.rstrip(".") for p in parts if p) or "cinematic shot"


def _compose_framing_directive(scene: Dict[str, Any], expression_mode: str) -> str:
    subject  = str(scene.get("subject_focus")      or "").strip()
    presence = str(scene.get("character_presence") or "").strip()
    movement = str(scene.get("movement_type")      or "slow").strip()

    if expression_mode == "face":
        return f"face — {presence} presence, {movement} movement"
    if expression_mode == "body":
        return f"body — {movement} movement"
    if expression_mode == "environment":
        env = str(scene.get("environment_type") or "").strip()
        return f"environment — {env}, {movement} movement"
    if expression_mode == "macro":
        return "macro detail, controlled movement"
    return f"{subject}, {movement} movement"


def _audio_intensity_at(audio_data: Dict[str, Any],
                        time_seconds: float) -> float:
    """Look up the audio intensity at a given timestamp.

    intensity_curve is either a list of floats (one per equally-spaced
    sample) or a list of {time, intensity} dicts.  Falls back to
    _DEFAULT_INTENSITY when the curve is empty or unparsable.
    """
    curve = (audio_data or {}).get("intensity_curve") or []
    if not curve:
        return _DEFAULT_INTENSITY

    # Dict-of-{time, intensity} form — pick the closest sample by time
    if isinstance(curve[0], dict):
        best_v = _DEFAULT_INTENSITY
        best_d = float("inf")
        for item in curve:
            try:
                t = float(item.get("time") or item.get("t") or 0.0)
                v = float(item.get("intensity") or item.get("value") or 0.5)
            except (TypeError, ValueError):
                continue
            d = abs(t - time_seconds)
            if d < best_d:
                best_d = d
                best_v = v
        return max(0.0, min(1.0, best_v))

    # Flat list of floats — assume evenly spaced over audio duration
    try:
        audio_dur = float((audio_data or {}).get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        audio_dur = 0.0
    if audio_dur <= 0:
        # No duration → treat curve as 1-sample-per-second
        idx = max(0, min(len(curve) - 1, int(round(time_seconds))))
    else:
        idx = max(0, min(len(curve) - 1,
                         int(round(time_seconds * len(curve) / audio_dur))))
    try:
        return max(0.0, min(1.0, float(curve[idx])))
    except (TypeError, ValueError):
        return _DEFAULT_INTENSITY


# ── Main public function ─────────────────────────────────────────────────────
def build_timeline_from_brief_v3(
    brief_packet:           Dict[str, Any],
    style_packet:           Dict[str, Any],
    narrative_packet:       Dict[str, Any],
    emotional_mode_packet:  Dict[str, Any],
    audio_data:             Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a v3 creative brief into a styled_timeline.

    The v3 brief packet already contains, per scene, the locked shot
    list with timings + enriched_direction.  This function adds the
    decoration layer (visual_prompt, expression_mode, camera_profile,
    motion_scale, transition, framing_directive), runs ShotVarietyEngine
    + StyleGradingEngine, and returns the final styled_timeline in the
    same schema downstream stages already consume.

    Raises
    ------
    ValueError
        If brief_packet has no scenes or no shots.  v3 must NEVER
        silently synthesise fallback shots — if upstream engines failed,
        we want a loud failure here.
    """
    scenes: List[Dict[str, Any]] = list(
        (brief_packet or {}).get("scenes") or []
    )
    if not scenes:
        raise ValueError(
            "TimelineBuilderV3: brief_packet.scenes is empty — "
            "the v3 storyboard / brief must have produced at least one "
            "scene.  Refusing to synthesise fallback shots."
        )

    total_shots = sum(len(sc.get("shots") or []) for sc in scenes)
    if total_shots == 0:
        raise ValueError(
            "TimelineBuilderV3: brief_packet has scenes but zero shots.  "
            "Refusing to synthesise fallback shots."
        )

    # ── Narrative packet — director-level decisions ────────────────────────
    _np = narrative_packet or {}
    narr_motion_philosophy = str(_np.get("motion_philosophy") or "mixed").strip().lower()
    narr_presence_strategy = str(_np.get("presence_strategy") or "").strip().lower()
    narr_timeline_strategy = str(_np.get("timeline_strategy") or "").strip().lower()
    _default_timeline_mode = narr_timeline_strategy or "present"
    _narr_presence: Optional[str] = narr_presence_strategy or None

    # ── Audio + pacing parameters (used for start_beat / bar_index only) ──
    _ad = audio_data or {}
    try:
        bpm = float(_ad.get("bpm") or _DEFAULT_BPM)
        if bpm <= 0:
            bpm = _DEFAULT_BPM
    except (TypeError, ValueError):
        bpm = _DEFAULT_BPM
    try:
        beats_per_bar = int(_ad.get("beats_per_bar") or _DEFAULT_BEATS_PER_BAR)
        if beats_per_bar <= 0:
            beats_per_bar = _DEFAULT_BEATS_PER_BAR
    except (TypeError, ValueError):
        beats_per_bar = _DEFAULT_BEATS_PER_BAR
    beat_dur = 60.0 / bpm
    bar_dur  = beat_dur * beats_per_bar

    logger.info(
        "TimelineBuilderV3: %d scenes, %d shots, narrative motion=%s "
        "presence=%s timeline=%s, bpm=%.1f",
        len(scenes), total_shots, narr_motion_philosophy,
        narr_presence_strategy or "(brief-level)",
        _default_timeline_mode, bpm,
    )

    raw_shots: List[Dict[str, Any]] = []
    timeline_index = 0

    for scene in scenes:
        # Brief intent fields — all decoration draws from these
        scene_id            = scene.get("scene_id") or ""
        scene_purpose       = str(scene.get("scene_purpose") or "").strip()
        narrative_phase     = str(scene.get("narrative_phase") or "").strip()
        scene_chosen        = str(scene.get("chosen_direction") or "").strip()
        _subject_focus      = str(scene.get("subject_focus")      or "character").strip()
        _character_presence = (str(scene.get("character_presence") or "").strip()
                               or _narr_presence or "continuous")
        _environment_type   = str(scene.get("environment_type")   or "").strip()
        _key_elements       = list(scene.get("key_elements")       or [])
        _emotional_state    = str(scene.get("emotional_state")     or "").strip()
        _lighting_condition = str(scene.get("lighting_condition")  or "").strip()
        _movement_type      = str(scene.get("movement_type")       or "").strip()
        _motion_density     = str(scene.get("motion_density")      or "").strip()
        _timeline_mode      = (str(scene.get("timeline_mode") or "").strip()
                               or _default_timeline_mode)
        ei_str              = str(scene.get("emotional_intensity") or "medium").lower()
        scene_intensity     = _INTENSITY_MAP.get(ei_str, 0.6)

        # Expression mode — brief wins, narrative fills gaps
        sf   = _subject_focus.lower()
        cp   = _character_presence.lower()
        mode = _SUBJECT_TO_MODE.get(sf, "environment")
        if sf == "character" and cp in ("intermittent", "minimal"):
            mode = "body"
        if narr_motion_philosophy == "static" and mode not in ("face", "body"):
            mode = "environment"

        scene_shots = list(scene.get("shots") or [])
        for shot_within_scene, brief_shot in enumerate(scene_shots):
            timeline_index += 1

            # ── LOCKED FIELDS — verbatim from brief shot ────────────────
            shot_id            = brief_shot.get("shot_id") or f"shot_{timeline_index}"
            start_time         = brief_shot.get("start_time")
            end_time           = brief_shot.get("end_time")
            duration           = brief_shot.get("duration")
            lyric_text         = str(brief_shot.get("lyric_text") or "")
            action_intent      = str(brief_shot.get("action_intent") or "")
            enriched_direction = str(brief_shot.get("enriched_direction") or "")

            # Sanity guards — v3 storyboard must never produce missing
            # timings or a shot longer than the storyboard cap.  Shots
            # >_VIDEO_MAX_DURATION are valid here so long as they carry
            # an `actions[]` decomposition for Phase-4 multishot
            # consumption (each action is ≤_VIDEO_MAX_DURATION).
            if start_time is None or end_time is None or duration is None:
                raise ValueError(
                    f"TimelineBuilderV3: brief shot {shot_id} (scene "
                    f"{scene_id}) is missing timings — refusing to "
                    f"fabricate them."
                )
            try:
                _dur_f = float(duration)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"TimelineBuilderV3: brief shot {shot_id} has "
                    f"non-numeric duration ({duration!r})."
                ) from exc
            if _dur_f > _STORYBOARD_MAX_SHOT + 1e-6:
                raise ValueError(
                    f"TimelineBuilderV3: brief shot {shot_id} "
                    f"duration={_dur_f}s exceeds the storyboard cap of "
                    f"{_STORYBOARD_MAX_SHOT}s — upstream contract "
                    f"violation."
                )

            # Multishot decomposition (carries each ≤_VIDEO_MAX_DURATION
            # render unit).  When the shot is >_VIDEO_MAX_DURATION, we
            # require it to have actions[] whose durations sum to the
            # shot duration; otherwise Phase-4 has no way to render it.
            actions: List[Dict[str, Any]] = list(brief_shot.get("actions") or [])
            if _dur_f > _VIDEO_MAX_DURATION + 1e-6:
                if not actions:
                    raise ValueError(
                        f"TimelineBuilderV3: brief shot {shot_id} "
                        f"duration={_dur_f}s exceeds the video-render "
                        f"cap of {_VIDEO_MAX_DURATION}s but has no "
                        f"actions[] decomposition.  The storyboard must "
                        f"emit a multishot decomposition for any shot "
                        f"longer than {_VIDEO_MAX_DURATION}s."
                    )
                _act_sum = 0.0
                for _a in actions:
                    try:
                        _ad = float(_a.get("duration") or 0)
                    except (TypeError, ValueError):
                        _ad = 0.0
                    if _ad > _VIDEO_MAX_DURATION + 1e-6:
                        raise ValueError(
                            f"TimelineBuilderV3: brief shot {shot_id} "
                            f"action order={_a.get('order')} duration="
                            f"{_ad}s exceeds the video-render cap of "
                            f"{_VIDEO_MAX_DURATION}s."
                        )
                    _act_sum += _ad
                # Hard contract: per-action durations must sum to (≈) the
                # parent shot duration.  This is the bridge invariant for
                # Phase-4 rendering — a desync here would cause render
                # length and final assembly to drift.  Tolerance is 0.25s
                # to allow for storyboard rounding when actions are split
                # by the multishot decomposer (it stores integer/half-
                # second durations).
                if abs(_act_sum - _dur_f) > 0.25:
                    raise ValueError(
                        f"TimelineBuilderV3: shot {shot_id} actions sum="
                        f"{_act_sum:.2f}s differs from shot duration="
                        f"{_dur_f:.2f}s by more than 0.25s — multishot "
                        f"decomposition is desynced from the storyboard "
                        f"timing.  Phase-4 rendering would drift."
                    )

            # ── DECORATION ────────────────────────────────────────────────
            audio_int = _audio_intensity_at(audio_data, float(start_time))
            blended_intensity = max(
                0.0, min(1.0, scene_intensity * 0.6 + audio_int * 0.4)
            )

            motion_scale_pool = (
                _MOTION_SCALE.get(mode) or _MOTION_SCALE["environment"]
            )
            motion_scale = motion_scale_pool[
                (timeline_index - 1) % len(motion_scale_pool)
            ]
            transition  = _derive_transition(timeline_index - 1, mode,
                                              blended_intensity)
            beat_start  = round(float(start_time) / beat_dur)
            bar_index   = int(float(start_time) // bar_dur) + 1

            # chosen_direction is the per-shot enriched_direction from the
            # brief.  Falls back to the scene-level chosen_direction only
            # when the brief shot somehow lacked an enriched form (which
            # _coerce_shot_brief already guarantees won't happen, but we
            # keep the fallback defensively).
            per_shot_dir = enriched_direction.strip() or scene_chosen
            visual_prompt = _compose_visual_prompt(scene, per_shot_dir,
                                                    lyric_text)
            framing_dir   = _compose_framing_directive(scene, mode)

            # Normalise start_time int when integral, preserving v2
            # downstream's typing convention.
            try:
                _st_norm = (
                    int(start_time) if float(start_time).is_integer()
                    else round(float(start_time), 3)
                )
            except (TypeError, ValueError):
                _st_norm = start_time

            raw_shots.append({
                "timeline_index":           timeline_index,
                "shot_index":               timeline_index,
                "shot_id":                  shot_id,
                "start_time":               _st_norm,
                "duration":                 duration,
                "end_time":                 round(float(end_time), 3),
                "start_beat":               beat_start,
                "bar_index":                bar_index,
                "visual_prompt":            (
                    f"{visual_prompt} Motion scale: {motion_scale}. "
                    f"Transition behavior: {transition}."
                ),
                "meaning":                  scene_purpose,
                "function":                 narrative_phase,
                "repeat_status":            "original",
                "intensity":                round(blended_intensity, 3),
                "raw_shot_intensity":       round(scene_intensity, 3),
                "audio_intensity":          round(audio_int, 3),
                "motion_scale":             motion_scale,
                "transition":               transition,
                "expression_mode":          mode,
                "reference_image":          None,
                "fidelity_lock":            0.72,
                "character_consistency_id": None,
                # ── First-class brief intent fields ────────────────────
                "subject_focus":            _subject_focus,
                "character_presence":       _character_presence,
                "environment_type":         _environment_type,
                "key_elements":             _key_elements,
                "emotional_state":          _emotional_state,
                "lighting_condition":       _lighting_condition,
                "movement_type":            _movement_type,
                "motion_density":           _motion_density,
                # ── Structured sub-dicts ──────────────────────────────
                "camera_profile":           _build_camera_profile(
                                                _movement_type or narr_motion_philosophy,
                                                _motion_density,
                                            ),
                "environment_profile": {
                    "environment_type":     _environment_type,
                    "timeline_mode":        _timeline_mode,
                },
                "continuity_anchor": {
                    "motifs":               list(scene.get("motif_usage") or []),
                    "character":            str(
                        (scene.get("continuity_hooks") or {}).get("character")
                        or (scene.get("continuity_hooks") or {}).get("subject")
                        or ""
                    ).strip(),
                },
                "rendering_notes":          [],
                "motion_prompt":            motion_scale,
                "framing_directive":        framing_dir,
                "composition_note":         "",
                "cinematography":           None,
                "cinematic_beat":           None,
                "shot_event":               None,
                "shot_type":                None,
                "shot_validation":          None,
                "llm_expression_mode":      mode,
                "variety_cap_reclassified": False,
                # v3 shots derive their times from the storyboard, which
                # was itself driven by the lyric timings — start_time IS
                # the lyric anchor.
                "lyric_start_seconds":      float(start_time),
                "lyric_end_seconds":        float(end_time),
                "lyric_text":               lyric_text,
                "action_intent":            action_intent,
                "execution_detail":         str(brief_shot.get("execution_detail") or ""),
                # Multishot decomposition for Phase-4 WAN/Kling
                # consumption.  Empty list when the shot is a single
                # render unit (≤_VIDEO_MAX_DURATION).
                "actions":                  actions,
                # First-class timeline decisions (top-level)
                "chosen_direction":         per_shot_dir,
                "timeline_mode":            _timeline_mode,
                # Compat aliases — downstream (shot_prompt_composer,
                # motion_render_prompt_builder, pipeline_worker) still
                # reads `_v2_chosen_direction`.  Keep the alias so v3 is
                # a drop-in replacement.
                "_v2_scene_id":             str(scene_id),
                "_v2_chosen_direction":     per_shot_dir,
                "_v2_lyric_units_count":    1,
                # v3 trace fields
                "_v3_scene_id":             str(scene_id),
                "_v3_chosen_direction":     per_shot_dir,
                "_v3_enriched_direction":   enriched_direction,
                "_v3_action_intent":        action_intent,
                "_shot_within_scene":       shot_within_scene,
                "scene_id":                 str(scene_id),
            })

    logger.info(
        "TimelineBuilderV3: built %d shots from %d brief scenes "
        "(no fallback, no splitter — durations locked by storyboard v3)",
        len(raw_shots), len(scenes),
    )

    # Per-scene direction variety log
    _scene_dirs: Dict[str, list] = {}
    for _s in raw_shots:
        _sid = _s.get("_v3_scene_id") or "?"
        _scene_dirs.setdefault(_sid, []).append(_s.get("chosen_direction") or "")
    logger.info(
        "TimelineBuilderV3: distinct chosen_direction per scene — %s",
        ", ".join(f"{sid}:{len(set(dirs))}" for sid, dirs in _scene_dirs.items()),
    )

    # ── Shot Variety Engine — stamp each shot with its director-spec shot_type
    try:
        emp = emotional_mode_packet or {}
        variety_engine = ShotVarietyEngine(emotional_mode_packet=emp)
        raw_shots = variety_engine.apply_variety(raw_shots)
        _dist: Dict[str, int] = {}
        for _s in raw_shots:
            _t = _s.get("shot_type") or "unknown"
            _dist[_t] = _dist.get(_t, 0) + 1
        logger.info(
            "TimelineBuilderV3: shot variety applied (mode=%s) — %s",
            emp.get("primary_mode") or "base",
            ", ".join(f"{k}×{v}" for k, v in sorted(_dist.items())),
        )
    except Exception:
        logger.exception(
            "TimelineBuilderV3: ShotVarietyEngine failed (non-fatal) — "
            "shot_type left as None for all shots."
        )

    # ── Style grading pass ────────────────────────────────────────────────
    styled_timeline = _apply_style_grading(raw_shots, style_packet)

    # ── Merge brief-intent + v3 trace fields back (StyleGrading strips them)
    _PASSTHROUGH_FIELDS = (
        # Rhythm / timing
        "start_beat", "bar_index", "audio_intensity", "raw_shot_intensity",
        "lyric_start_seconds", "lyric_end_seconds",
        "lyric_text", "action_intent", "execution_detail", "actions",
        # v2 compat trace
        "_v2_scene_id", "_v2_chosen_direction", "_v2_lyric_units_count",
        # v3 trace
        "_v3_scene_id", "_v3_chosen_direction",
        "_v3_enriched_direction", "_v3_action_intent",
        # First-class brief intent
        "subject_focus", "character_presence", "environment_type",
        "key_elements", "emotional_state", "lighting_condition",
        "movement_type", "motion_density",
        # Top-level timeline decisions
        "chosen_direction", "timeline_mode", "scene_id",
        # Shot Variety Engine
        "shot_type", "variety_applied",
        # Within-scene shot position
        "_shot_within_scene",
    )
    raw_by_idx = {r.get("shot_index"): r for r in raw_shots}
    for styled in styled_timeline:
        src = raw_by_idx.get(styled.get("shot_index"))
        if not src:
            continue
        for k in _PASSTHROUGH_FIELDS:
            if k in src and (k not in styled or styled[k] is None):
                styled[k] = src[k]

    if len(styled_timeline) != len(raw_shots):
        # StyleGradingEngine should never drop shots — a count mismatch
        # would silently lose footage.  Log loudly and prefer raw_shots.
        logger.error(
            "TimelineBuilderV3: StyleGrading changed shot count "
            "(%d → %d) — falling back to raw_shots to preserve all shots.",
            len(raw_shots), len(styled_timeline),
        )
        return raw_shots

    return styled_timeline


def _apply_style_grading(
    timeline: List[Dict[str, Any]],
    style_packet: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Pass the raw shots through StyleGradingEngine.  Mirrors v2's
    behaviour exactly so styled_timeline output remains compatible.
    """
    try:
        from style_grading_engine import StyleGradingEngine
        engine = StyleGradingEngine()
        styled = engine.apply_style(
            timeline=timeline,
            style_profile=style_packet or {},
        )
        if styled:
            return styled
    except Exception:
        logger.exception(
            "TimelineBuilderV3: StyleGradingEngine failed (non-fatal) — "
            "returning raw shots."
        )
    return timeline


# ── CLI runner ───────────────────────────────────────────────────────────────
def _cli_main(argv: List[str]) -> int:
    """Standalone CLI for Phase-3 testing.

    Reads /tmp/brief_v3_output.json (Phase-2 artifact), pulls supporting
    packets (style/narrative/emotional/audio) from the project brain,
    builds the v3 timeline and writes the result to stdout.
    """
    import json
    import os
    import sys

    if len(argv) < 2:
        print("usage: python timeline_builder_v3.py <project_id>",
              file=sys.stderr)
        return 2
    project_id = argv[1]

    # 1. Load Phase-2 brief packet from /tmp
    brief_packet: Dict[str, Any] = {}
    try:
        with open("/tmp/brief_v3_output.json", "r", encoding="utf-8") as fh:
            brief_packet = json.load(fh)
        print(f"[TimelineV3 CLI] using /tmp/brief_v3_output.json "
              f"({brief_packet.get('scene_count')} scenes, "
              f"{brief_packet.get('shot_count')} shots)",
              file=sys.stderr)
    except Exception as exc:
        print(f"[TimelineV3 CLI] failed to load /tmp/brief_v3_output.json: "
              f"{exc}", file=sys.stderr)
        return 3

    # 2. Pull supporting packets from brain + audio_data from projects row
    try:
        from project_brain import ProjectBrain
        import psycopg
        from psycopg.rows import dict_row
        db_url = os.environ.get("DATABASE_URL")
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            brain = ProjectBrain.load(project_id, conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT audio_data FROM projects WHERE id = %s",
                    (project_id,),
                )
                row = cur.fetchone()
                audio_data = (row or {}).get("audio_data") or {}
        narrative_packet      = brain.read("narrative_packet")      or {}
        style_packet          = brain.read("style_packet")          or {}
        emotional_mode_packet = brain.read("emotional_mode_packet") or {}
    except Exception as exc:
        print(f"[TimelineV3 CLI] failed to load brain/audio: {exc}",
              file=sys.stderr)
        return 4

    timeline = build_timeline_from_brief_v3(
        brief_packet=brief_packet,
        style_packet=style_packet,
        narrative_packet=narrative_packet,
        emotional_mode_packet=emotional_mode_packet,
        audio_data=audio_data,
    )

    out = {
        "schema_version":   3,
        "shot_count":       len(timeline),
        "shots":            timeline,
    }
    import json as _json
    print(_json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main(sys.argv))
