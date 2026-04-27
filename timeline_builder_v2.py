"""V2 Timeline Builder — converts Creative Brief + audio timing into styled_timeline.

This is the bridge that was missing from the V2 pipeline.  It replaces the
V1 chain of VisualStoryboardEngine → RhythmicAssemblyEngine → StyleGradingEngine
but reads from the brain's creative_briefs (the FIRST COMMITMENT LAYER) instead
of re-deriving from raw lyrics.  Every generated image is now driven by the
brief's locked chosen_direction rather than V1's independently-invented prompts.

Public API
----------
build_timeline_from_brief(
    creative_briefs,
    input_structure,
    emotional_mode_packet,
    style_packet,
    narrative_packet,
    audio_data,
) -> List[Dict]

Output schema matches V1 styled_timeline exactly so all downstream stages
(materializer, references, stills, video assembly, final cut) consume it
without any changes.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from shot_variety_engine import ShotVarietyEngine

logger = logging.getLogger(__name__)

# ── Pacing defaults (mirrors RhythmicAssemblyEngine constants) ────────────────
_DEFAULT_BPM             = 120.0
_DEFAULT_BEATS_PER_BAR   = 4
_DEFAULT_MIN_DURATION    = 2.0
_DEFAULT_MAX_DURATION    = 15.0
_DEFAULT_PREFERRED_AVG   = 6.0
_DEFAULT_INTENSITY       = 0.5

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

# ── Motion scale templates keyed by expression_mode ─────────────────────────
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

# ── Transition type by context ───────────────────────────────────────────────
def _derive_transition(shot_index: int, prev_mode: Optional[str],
                       current_mode: str, intensity: float) -> str:
    if shot_index == 0:
        return "opening_hold"
    if current_mode == "symbolic":
        return "poetic_dissolve"
    if intensity > 0.82:
        return "hard_cut"
    return "straight_cut"


# ── Camera profile from movement_type + motion_density ───────────────────────
def _build_camera_profile(movement_type: str, motion_density: str) -> Dict[str, Any]:
    mt = (movement_type or "slow").lower()
    md = (motion_density or "medium").lower()
    return {
        "movement":         mt,
        "motion_density":   md,
        "stabilisation":    "gimbal" if mt == "dynamic" else "tripod",
        "depth_of_field":   "shallow" if md == "low" else "deep",
    }


# ── Visual prompt composer from brief fields ──────────────────────────────────
def _compose_visual_prompt(scene: Dict[str, Any], lyric_text: str = "") -> str:
    parts: List[str] = []

    chosen = str(scene.get("chosen_direction") or "").strip()
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

    if lyric_text:
        parts.append(f'"{lyric_text.strip()}"')

    return ". ".join(p.rstrip(".") for p in parts if p) or "cinematic shot"


def _compose_framing_directive(scene: Dict[str, Any], expression_mode: str) -> str:
    subject = str(scene.get("subject_focus") or "").strip()
    presence = str(scene.get("character_presence") or "").strip()
    movement = str(scene.get("movement_type") or "slow").strip()

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


# ── BPM helpers (mirrors RhythmicAssemblyEngine) ─────────────────────────────
def _snap_to_beat(duration: float, beat_dur: float) -> float:
    if beat_dur <= 0:
        return duration
    beats = max(1, round(duration / beat_dur))
    return beats * beat_dur


def _clamp_duration(duration: float, min_d: float, max_d: float) -> int:
    return int(max(min_d, min(max_d, round(duration))))


def _calculate_base_duration(intensity: float, beat_dur: float,
                              expression_mode: str) -> float:
    if intensity >= 0.85:
        beats = 2
    elif intensity >= 0.70:
        beats = 3
    elif intensity >= 0.50:
        beats = 4
    elif intensity >= 0.30:
        beats = 6
    else:
        beats = 8
    if expression_mode in {"face", "body"} and intensity < 0.75:
        beats += 1
    return beat_dur * beats


# ── Main public function ──────────────────────────────────────────────────────
def build_timeline_from_brief(
    creative_briefs: Dict[str, Any],
    input_structure: Dict[str, Any],
    emotional_mode_packet: Dict[str, Any],
    style_packet: Dict[str, Any],
    narrative_packet: Dict[str, Any],
    audio_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert the locked Creative Brief into a timed styled_timeline.

    This is the V2 replacement for the V1 chain:
        VisualStoryboardEngine.build_storyboard()
        → RhythmicAssemblyEngine.assemble_timeline()
        → StyleGradingEngine.apply_style()

    Returns a list of shot dicts in the exact same schema as V1 styled_timeline
    so all downstream stages (materializer, refs, stills, video, final cut)
    consume it without any changes.
    """
    scenes: List[Dict[str, Any]] = list(
        (creative_briefs or {}).get("scenes") or []
    )
    if not scenes:
        logger.warning(
            "TimelineBuilderV2: creative_briefs.scenes is empty — "
            "producing a single fallback shot."
        )
        scenes = [{
            "scene_id": "fallback_scene",
            "source_section": "verse",
            "narrative_phase": "development",
            "scene_purpose": "cinematic expression",
            "chosen_direction": "cinematic still of the main subject",
            "subject_focus": "character",
            "character_presence": "continuous",
            "environment_type": "natural setting",
            "key_elements": [],
            "emotional_state": "reflective",
            "emotional_intensity": "medium",
            "lighting_condition": "natural light",
            "movement_type": "slow",
            "motion_density": "medium",
            "timeline_mode": "present",
        }]

    # ── Narrative packet — extract director-level decisions ────────────────
    _np = narrative_packet or {}
    narr_motion_philosophy = str(_np.get("motion_philosophy") or "mixed").strip().lower()
    narr_presence_strategy = str(_np.get("presence_strategy") or "").strip().lower()
    narr_timeline_strategy = str(_np.get("timeline_strategy") or "").strip().lower()

    # If narrative dictates a global timeline strategy, shots without their own
    # timeline_mode will default to this.  Brief-level timeline_mode wins.
    _default_timeline_mode = narr_timeline_strategy if narr_timeline_strategy else "present"

    # Presence strategy from narrative layer (overrides brief character_presence
    # only when brief leaves it empty/absent).
    _narr_presence: Optional[str] = narr_presence_strategy or None

    logger.info(
        "TimelineBuilderV2: narrative motion_philosophy=%s presence=%s timeline=%s",
        narr_motion_philosophy, narr_presence_strategy or "(brief-level)",
        _default_timeline_mode,
    )

    # ── Audio + pacing parameters ──────────────────────────────────────────
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

    try:
        audio_duration = float(_ad.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        audio_duration = 0.0

    intensity_curve: List[float] = []
    for item in list(_ad.get("intensity_curve") or []):
        try:
            if isinstance(item, dict):
                intensity_curve.append(float(item.get("intensity") or item.get("value") or 0.5))
            else:
                intensity_curve.append(float(item))
        except (TypeError, ValueError):
            intensity_curve.append(0.5)

    beat_dur = 60.0 / bpm
    bar_dur  = beat_dur * beats_per_bar

    # ── Pacing overrides from emotional_mode_packet ────────────────────────
    emp  = emotional_mode_packet or {}
    pp   = emp.get("pacing_profile") or {}
    try:
        min_dur = float(pp.get("min_shot_duration") or _DEFAULT_MIN_DURATION)
    except (TypeError, ValueError):
        min_dur = _DEFAULT_MIN_DURATION
    try:
        max_dur = float(pp.get("max_shot_duration") or _DEFAULT_MAX_DURATION)
    except (TypeError, ValueError):
        max_dur = _DEFAULT_MAX_DURATION
    try:
        pref_avg = float(pp.get("preferred_avg_duration") or _DEFAULT_PREFERRED_AVG)
        pref_avg = max(min_dur, min(max_dur, pref_avg))
    except (TypeError, ValueError):
        pref_avg = _DEFAULT_PREFERRED_AVG

    # ── Lyric units → determine shot list ─────────────────────────────────
    # input_structure.units carries one entry per lyric line with optional
    # timestamps.  Each unit becomes one shot.
    # If no units are available, derive a shot count from the audio duration
    # and preferred average.
    units: List[Dict[str, Any]] = list((input_structure or {}).get("units") or [])

    # Fallback: derive equally-timed units from sections or audio duration
    if not units:
        sections: List[Dict[str, Any]] = list(
            (input_structure or {}).get("sections") or []
        )
        if sections and audio_duration > 0:
            n_units = sum(
                max(1, len(list(s.get("units") or [])) or 1) for s in sections
            ) or len(sections)
            step = audio_duration / n_units
            units = [
                {
                    "text": "",
                    "start": round(i * step, 3),
                    "end":   round((i + 1) * step, 3),
                    "unit_index": i,
                    "section_id": (sections[int(i * len(sections) / n_units)].get("id") or ""),
                }
                for i in range(n_units)
            ]
        else:
            # Absolute fallback: one shot per scene, evenly distributed
            n_units = len(scenes) * 2
            if audio_duration > 0:
                step = audio_duration / max(n_units, 1)
                units = [
                    {
                        "text": "",
                        "start": round(i * step, 3),
                        "end":   round((i + 1) * step, 3),
                        "unit_index": i,
                    }
                    for i in range(n_units)
                ]
            else:
                units = [
                    {
                        "text": "",
                        "start": round(i * pref_avg, 3),
                        "end":   round((i + 1) * pref_avg, 3),
                        "unit_index": i,
                    }
                    for i in range(n_units)
                ]

    n_units  = len(units)
    n_scenes = len(scenes)

    # ── Assign lyric units to scenes (section-key first, positional fallback) ─
    # Build section_id → scene index map
    _section_to_scene_idx: Dict[str, int] = {}
    for _s_idx, _sc in enumerate(scenes):
        _sec_id = str(_sc.get("source_section") or "").strip().lower()
        if _sec_id:
            _section_to_scene_idx[_sec_id] = _s_idx

    # Assign each unit to its scene
    unit_scene_idxs: List[int] = []
    for _ui, _unit in enumerate(units):
        _sec = str(_unit.get("section_id") or "").strip().lower()
        if _sec and _sec in _section_to_scene_idx:
            unit_scene_idxs.append(_section_to_scene_idx[_sec])
        else:
            # Positional mapping — proportional by unit position
            unit_scene_idxs.append(
                min(n_scenes - 1, int(_ui * n_scenes / max(n_units, 1)))
            )

    # Group units by scene — preserving order within each scene
    scene_units_map: Dict[int, List[int]] = {i: [] for i in range(n_scenes)}
    for _ui, _si in enumerate(unit_scene_idxs):
        scene_units_map[_si].append(_ui)

    # ── Determine shots per scene (~1 shot per 2–3 lyric units, min 1) ───
    # Compute target units-per-shot from pref_avg and avg unit duration.
    _avg_unit_dur = (audio_duration / max(n_units, 1)) if audio_duration > 0 else pref_avg
    _target_ups   = max(1, round(pref_avg / max(_avg_unit_dur, 0.5)))   # units per shot

    # Build flat list of (scene, unit_group, first_flat_unit_idx) triples — each → one shot
    # first_flat_unit_idx is the flat index into `units` for the group's first unit;
    # used to look up audio intensity from intensity_curve without ValueError.
    shot_plan: List[tuple] = []
    for _sc_i, scene in enumerate(scenes):
        _u_idxs = scene_units_map.get(_sc_i, [])
        if not _u_idxs:
            # Scene has no units (short/empty section) — give it one shot
            shot_plan.append((scene, [{}], 0))
            continue
        # Chunk unit indices into groups of ~_target_ups
        _shots_this_scene = max(1, round(len(_u_idxs) / _target_ups))
        _chunk_size = max(1, len(_u_idxs) // _shots_this_scene)
        _remainder  = len(_u_idxs) % _shots_this_scene
        _pos = 0
        for _gi in range(_shots_this_scene):
            _extra = 1 if _gi < _remainder else 0
            _grp_idxs = _u_idxs[_pos: _pos + _chunk_size + _extra]
            _pos += _chunk_size + _extra
            if not _grp_idxs:
                continue
            shot_plan.append((scene, [units[_k] for _k in _grp_idxs], _grp_idxs[0]))

    logger.info(
        "TimelineBuilderV2: planned %d shots from %d scenes "
        "(units=%d, units_per_shot=~%d)",
        len(shot_plan), n_scenes, n_units, _target_ups,
    )

    # ── Build raw shots from plan ──────────────────────────────────────────
    raw_shots: List[Dict[str, Any]] = []
    current_ts = 0.0
    prev_mode: Optional[str] = None
    any_lyric_anchor = False

    for shot_i, (scene, unit_group, first_flat_unit_idx) in enumerate(shot_plan):
        # Use first unit in group for lyric anchor + representative lyric text
        first_unit = unit_group[0] if unit_group else {}
        last_unit  = unit_group[-1] if unit_group else {}

        # Timing — use lyric anchor when first unit has real timestamps
        lyric_start: Optional[float] = None
        lyric_end:   Optional[float] = None
        try:
            _ls = float(first_unit.get("start") or 0)
            _le = float(last_unit.get("end")    or last_unit.get("start") or 0)
            if _le > _ls:
                lyric_start = _ls
                lyric_end   = _le
        except (TypeError, ValueError):
            pass

        if lyric_start is not None:
            raw_snap = round(round(lyric_start / beat_dur) * beat_dur, 3)
            current_ts = 0.0 if shot_i == 0 else max(raw_snap, raw_shots[-1]["start_time"] if raw_shots else 0.0)
            any_lyric_anchor = True

        # Audio intensity — look up by first unit's flat index in the units list
        # (never use shot_i or scene_idx to index intensity_curve — they have different lengths)
        audio_int = (
            intensity_curve[first_flat_unit_idx] if first_flat_unit_idx < len(intensity_curve)
            else (intensity_curve[-1] if intensity_curve else _DEFAULT_INTENSITY)
        )

        ei_str = str(scene.get("emotional_intensity") or "medium").lower()
        shot_intensity = _INTENSITY_MAP.get(ei_str, 0.6)
        blended_intensity = max(0.0, min(1.0, shot_intensity * 0.6 + audio_int * 0.4))

        # Brief intent fields — first-class on the shot entry
        _subject_focus      = str(scene.get("subject_focus")      or "character").strip()
        _character_presence = (str(scene.get("character_presence") or "").strip()
                               or _narr_presence or "continuous")
        _environment_type   = str(scene.get("environment_type")   or "").strip()
        _key_elements       = list(scene.get("key_elements")       or [])
        _emotional_state    = str(scene.get("emotional_state")     or "").strip()
        _lighting_condition = str(scene.get("lighting_condition")  or "").strip()
        _movement_type      = str(scene.get("movement_type")       or "").strip()
        _motion_density     = str(scene.get("motion_density")      or "").strip()

        # Expression mode — brief wins, narrative fills gaps
        sf   = _subject_focus.lower()
        cp   = _character_presence.lower()
        mode = _SUBJECT_TO_MODE.get(sf, "environment")
        if sf == "character" and cp in ("intermittent", "minimal"):
            mode = "body"
        if narr_motion_philosophy == "static" and mode not in ("face", "body"):
            mode = "environment"

        # Duration — scale by narrative motion_philosophy
        raw_dur = _calculate_base_duration(blended_intensity, beat_dur, mode)
        if narr_motion_philosophy == "dynamic":
            raw_dur *= 0.80
        elif narr_motion_philosophy == "static":
            raw_dur *= 1.25
        snap_dur = _snap_to_beat(raw_dur, beat_dur)
        duration = _clamp_duration(snap_dur, min_dur, max_dur)

        motion_scale = (
            _MOTION_SCALE.get(mode) or _MOTION_SCALE["environment"]
        )[shot_i % len(_MOTION_SCALE.get(mode) or _MOTION_SCALE["environment"])]

        transition = _derive_transition(shot_i, prev_mode, mode, blended_intensity)
        beat_start = round(current_ts / beat_dur)
        bar_index  = int(current_ts // bar_dur) + 1

        lyric_text    = " / ".join(
            str(u.get("text") or "").strip() for u in unit_group if u.get("text")
        )
        visual_prompt = _compose_visual_prompt(scene, lyric_text)
        framing_dir   = _compose_framing_directive(scene, mode)

        raw_shots.append({
            "timeline_index":           shot_i + 1,
            "shot_index":               shot_i + 1,
            "shot_id":                  f"shot_{shot_i + 1}",
            "start_time":               (
                int(current_ts) if float(current_ts).is_integer()
                else round(current_ts, 3)
            ),
            "duration":                 duration,
            "end_time":                 round(current_ts + duration, 3),
            "start_beat":               beat_start,
            "bar_index":                bar_index,
            "visual_prompt":            (
                f"{visual_prompt} Motion scale: {motion_scale}. "
                f"Transition behavior: {transition}."
            ),
            "meaning":                  str(scene.get("scene_purpose") or "").strip(),
            "function":                 str(scene.get("narrative_phase") or "").strip(),
            "repeat_status":            "original",
            "intensity":                round(blended_intensity, 3),
            "raw_shot_intensity":       round(shot_intensity, 3),
            "audio_intensity":          round(audio_int, 3),
            "motion_scale":             motion_scale,
            "transition":               transition,
            "expression_mode":          mode,
            "reference_image":          None,
            "fidelity_lock":            0.72,
            "character_consistency_id": None,
            # ── First-class brief intent fields ────────────────────────────
            "subject_focus":            _subject_focus,
            "character_presence":       _character_presence,
            "environment_type":         _environment_type,
            "key_elements":             _key_elements,
            "emotional_state":          _emotional_state,
            "lighting_condition":       _lighting_condition,
            "movement_type":            _movement_type,
            "motion_density":           _motion_density,
            # ── Structured sub-dicts ────────────────────────────────────────
            "camera_profile":           _build_camera_profile(
                                            _movement_type or narr_motion_philosophy,
                                            _motion_density,
                                        ),
            "environment_profile": {
                "environment_type":   _environment_type,
                "timeline_mode":      (str(scene.get("timeline_mode") or "").strip()
                                       or _default_timeline_mode),
            },
            "continuity_anchor": {
                "motifs":    list(scene.get("motif_usage") or []),
                "character": str(
                    (scene.get("continuity_hooks") or {}).get("character") or ""
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
            "lyric_start_seconds":      lyric_start,
            "lyric_end_seconds":        lyric_end,
            # First-class timeline decisions (top-level, per task spec)
            "chosen_direction":         str(scene.get("chosen_direction") or ""),
            "timeline_mode":            (str(scene.get("timeline_mode") or "").strip()
                                         or _default_timeline_mode),
            # V2-specific: trace which brief scene drove this shot
            "_v2_scene_id":             str(scene.get("scene_id") or ""),
            "_v2_chosen_direction":     str(scene.get("chosen_direction") or ""),
            # Lyric unit coverage (for downstream editors)
            "_v2_lyric_units_count":    len(unit_group),
        })

        prev_mode = mode
        current_ts += duration

    # ── Lyric-anchor gap pass (matches RhythmicAssemblyEngine logic) ──────
    if any_lyric_anchor and raw_shots:
        for j in range(len(raw_shots) - 1):
            gap = raw_shots[j + 1]["start_time"] - raw_shots[j]["start_time"]
            max_cap = max(max_dur, gap) if j == 0 else max_dur
            dur = int(max(int(min_dur), min(int(max_cap), round(gap))))
            raw_shots[j]["duration"]     = dur
            raw_shots[j]["end_time"]     = round(raw_shots[j]["start_time"] + dur, 3)
            raw_shots[j]["lyric_anchored"] = True
        last = raw_shots[-1]
        last["lyric_anchored"] = True
        if audio_duration > 0:
            raw_last = audio_duration - last["start_time"]
            last["duration"] = int(max(int(min_dur), min(int(max_dur), round(raw_last))))
            last["end_time"]  = round(last["start_time"] + last["duration"], 3)

    # ── Audio-duration normalization (non-anchor mode) ─────────────────────
    if not any_lyric_anchor and audio_duration > 0 and raw_shots:
        total_assigned = sum(s["duration"] for s in raw_shots)
        if total_assigned > 0 and abs(total_assigned - audio_duration) > 0.5:
            scale = audio_duration / total_assigned
            logger.info(
                "TimelineBuilderV2: normalizing %.1fs → %.1fs (audio) "
                "×%.3f for %d shots",
                total_assigned, audio_duration, scale, len(raw_shots),
            )
            ts = 0
            for s in raw_shots:
                scaled  = s["duration"] * scale
                snapped = int(max(int(min_dur), min(int(max_dur), round(scaled))))
                s["duration"]   = snapped
                s["start_time"] = ts
                s["end_time"]   = ts + snapped
                s["start_beat"] = round(ts / beat_dur)
                s["bar_index"]  = int(ts // bar_dur) + 1
                ts += snapped

    logger.info(
        "TimelineBuilderV2: built %d shots from %d brief scenes "
        "(lyric_anchored=%s, audio_dur=%.1fs, bpm=%.1f)",
        len(raw_shots), len(scenes), any_lyric_anchor, audio_duration, bpm,
    )

    # ── Shot Variety Engine — stamp each shot with its director-spec shot_type ─
    try:
        variety_engine = ShotVarietyEngine(emotional_mode_packet=emp)
        raw_shots = variety_engine.apply_variety(raw_shots)
        _dist: Dict[str, int] = {}
        for _s in raw_shots:
            _t = _s.get("shot_type") or "unknown"
            _dist[_t] = _dist.get(_t, 0) + 1
        logger.info(
            "TimelineBuilderV2: shot variety applied (mode=%s) — %s",
            emp.get("primary_mode") or "base",
            ", ".join(f"{k}×{v}" for k, v in sorted(_dist.items())),
        )
    except Exception:
        logger.exception(
            "TimelineBuilderV2: ShotVarietyEngine failed (non-fatal) — "
            "shot_type left as None for all shots."
        )

    # ── Style grading pass ─────────────────────────────────────────────────
    styled_timeline = _apply_style_grading(raw_shots, style_packet)

    # ── Merge rhythm + brief-intent fields back (StyleGradingEngine strips them)
    _PASSTHROUGH_FIELDS = (
        # Rhythm / timing
        "start_beat", "bar_index", "audio_intensity", "raw_shot_intensity",
        "lyric_start_seconds", "lyric_end_seconds",
        # V2 tracing
        "_v2_scene_id", "_v2_chosen_direction", "_v2_lyric_units_count",
        # First-class brief intent — required by task spec
        "subject_focus", "character_presence", "environment_type",
        "key_elements", "emotional_state", "lighting_condition",
        "movement_type", "motion_density",
        # Top-level timeline decisions
        "chosen_direction", "timeline_mode",
        # Shot Variety Engine — cinematic shot distribution
        "shot_type", "variety_applied",
    )
    raw_by_idx = {r.get("shot_index"): r for r in raw_shots}
    for styled in styled_timeline:
        src = raw_by_idx.get(styled.get("shot_index"))
        if not src:
            continue
        for k in _PASSTHROUGH_FIELDS:
            # Always overwrite from raw for list fields (key_elements); use
            # raw value when styled doesn't have the key or has None.
            if k in src and (k not in styled or styled[k] is None):
                styled[k] = src[k]

    return styled_timeline


def _apply_style_grading(
    timeline: List[Dict[str, Any]],
    style_packet: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Pass the raw shots through StyleGradingEngine with the full style_profile.

    Passes the complete style_packet (not just {"preset": ...}) so that
    cinematic_style, color_psychology, texture_profile, and lighting_logic
    all reach the engine — fixing audit issue #3.
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
            "TimelineBuilderV2: StyleGradingEngine failed (non-fatal) — "
            "returning raw shots."
        )
    return timeline
