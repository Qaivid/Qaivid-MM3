"""
Shot Engine Service — v2
Converts scenes into individual shot units.
Now wired with pacing profiles for content-type-specific duration rules.
One shot = one intention. Lean, precise, model-friendly.
"""
from typing import Dict, List, Any
import uuid
from models import now_utc
from services.deterministic_rules import get_pacing_profile, detect_emotional_shift

SHOT_TYPES = ["wide", "medium", "medium-close", "close-up", "extreme-close-up", "over-shoulder", "aerial", "pov"]
CAMERA_HEIGHTS = ["eye-level", "low-angle", "high-angle", "overhead", "dutch-tilt"]
CAMERA_BEHAVIORS = ["static", "slow-pan-left", "slow-pan-right", "track-forward", "track-back", "dolly-in", "dolly-out", "handheld", "crane-up", "crane-down", "orbit"]


def build_shots_for_scene(scene: Dict[str, Any], context_packet: Dict[str, Any], project_settings: Dict[str, Any] = None, content_type: str = "song", emotional_mode_id: str = "") -> List[Dict[str, Any]]:
    settings = project_settings or {}
    density = settings.get("shot_density", "medium")
    temporal_status = scene.get("temporal_status", "present")
    emotional_temp = scene.get("emotional_temperature", "neutral")
    narrative_mode = context_packet.get("narrative_mode", "realist")
    line_meanings = context_packet.get("line_meanings", [])
    lyric_span = scene.get("lyric_span", [])

    # Get pacing profile — mode-aware when emotional_mode_id is provided
    pacing = get_pacing_profile(content_type, emotional_mode_id)

    # Get relevant line meanings for this scene
    scene_lines = [lm for lm in line_meanings if lm.get("line_index") in lyric_span]
    visualizable = [lm for lm in scene_lines if lm.get("visualization_mode") not in ("performance_only", "absorbed")]

    if not visualizable:
        visualizable = scene_lines[:1] if scene_lines else [{"text": scene.get("purpose", ""), "line_index": 0, "visualization_mode": "direct"}]

    # Determine shot count using pacing profile ratios
    density_multiplier = {"low": 0.6, "medium": 1.0, "high": 1.5}.get(density, 1.0)
    base_shots = max(1, len(visualizable))
    target_shots = max(1, round(base_shots * density_multiplier))

    shots = []
    shot_num = 1

    # AI VIDEO GENERATION LIMITS (WAN 2.6 / Atlas Cloud):
    # most clips render best between 3.0s and 8.0s; clamp accordingly.
    AI_MIN, AI_MAX = 3.0, 8.0

    # Determine if we have real lyric timing for this scene
    scene_has_timing = any(lm.get("start_time") is not None for lm in visualizable)
    scene_start = scene.get("start_time")
    scene_end = scene.get("end_time")

    # Distribute lines across target shot count
    group_size = max(1, len(visualizable) // target_shots)
    shot_groups = []
    for i in range(0, len(visualizable), max(1, group_size)):
        group = visualizable[i:i + group_size]
        if group:
            shot_groups.append(group)

    for gi, group in enumerate(shot_groups):
        primary_line = group[0]

        viz_mode = primary_line.get("visualization_mode", "direct")
        expression_mode = (primary_line.get("expression_mode") or "").lower()
        line_intensity = (primary_line.get("emotional_intensity") or "").lower()
        # Shot type now considers expression_mode so wide shots express via environment, not faces
        shot_type = _select_shot_type(viz_mode, temporal_status, shot_num, target_shots, expression_mode)
        # Camera height only goes high-angle for grief when intensity is genuinely HIGH
        camera_height = _select_camera_height(viz_mode, narrative_mode, emotional_temp, line_intensity)
        camera_behavior = _select_camera_behavior(viz_mode, temporal_status, emotional_temp)
        subject_action = _build_subject_action(primary_line, viz_mode, narrative_mode, expression_mode, scene)
        emotional_micro = primary_line.get("emotional_meaning", emotional_temp)
        light = _determine_lighting(scene, temporal_status, emotional_temp)
        objects = scene.get("objects_of_significance", [])[:3]
        negatives = _build_negative_constraints(viz_mode, narrative_mode, context_packet)

        # Duration: prefer real lyric timing (variety driven by song pacing),
        # fall back to pacing-profile heuristic. Always clamp to AI gen limits.
        shot_start = None
        shot_end = None
        if scene_has_timing:
            group_starts = [lm.get("start_time") for lm in group if lm.get("start_time") is not None]
            group_ends = [lm.get("end_time") for lm in group if lm.get("end_time") is not None]
            if group_starts:
                shot_start = min(group_starts)
                # Extend to start of next shot's first line if available, else group end
                if gi + 1 < len(shot_groups):
                    next_starts = [lm.get("start_time") for lm in shot_groups[gi + 1] if lm.get("start_time") is not None]
                    if next_starts:
                        shot_end = min(next_starts)
                if shot_end is None and group_ends:
                    shot_end = max(group_ends)
                if shot_end is None and scene_end is not None:
                    shot_end = scene_end

        if shot_start is not None and shot_end is not None and shot_end > shot_start:
            raw_duration = shot_end - shot_start
            duration = round(max(AI_MIN, min(AI_MAX, raw_duration)), 2)
        else:
            duration = _compute_duration(pacing, shot_type, shot_num, target_shots)
            duration = round(max(AI_MIN, min(AI_MAX, duration)), 2)

        shot = {
            "id": str(uuid.uuid4()),
            "project_id": scene.get("project_id", ""),
            "scene_id": scene.get("id", ""),
            "shot_number": shot_num,
            "visual_priority": primary_line.get("text", "")[:100],
            "lyric_text": " / ".join(lm.get("text", "") for lm in group if lm.get("text"))[:240],
            "lyric_line_indices": [lm.get("line_index") for lm in group if lm.get("line_index") is not None],
            "shot_type": shot_type,
            "camera_height": camera_height,
            "camera_behavior": camera_behavior,
            "subject_action": subject_action,
            "emotional_micro_state": emotional_micro,
            "light_description": light,
            "secondary_objects": objects,
            "motion_constraints": _motion_constraints(camera_behavior, duration),
            "negative_constraints": negatives,
            "duration_hint": duration,
            "start_time": shot_start,
            "end_time": shot_end,
            "generation_safe_wording": _build_generation_wording(
                subject_action, shot_type, camera_height, light, objects, viz_mode
            ),
            "created_at": now_utc(),
        }
        shots.append(shot)
        shot_num += 1

    return shots


def _compute_duration(pacing: Dict, shot_type: str, shot_num: int, total: int) -> float:
    """Compute shot duration using pacing profile ratios."""
    avg = pacing["preferred_avg_duration"]
    min_d = pacing["min_shot_duration"]
    max_d = pacing["max_shot_duration"]

    # Wide/establishing shots get longer, close-ups get shorter
    type_factor = {
        "wide": pacing["long_shot_ratio"] / 0.25 if pacing["long_shot_ratio"] > 0 else 1.2,
        "medium": 1.0,
        "medium-close": 0.9,
        "close-up": pacing["short_shot_ratio"] / 0.2 if pacing["short_shot_ratio"] > 0 else 0.7,
        "extreme-close-up": 0.6,
        "over-shoulder": 0.9,
        "aerial": 1.3,
        "pov": 0.8,
    }
    factor = type_factor.get(shot_type, 1.0)
    duration = round(avg * factor, 1)
    return max(min_d, min(max_d, duration))


def _select_shot_type(viz_mode: str, temporal: str, shot_num: int, total: int, expression_mode: str = "") -> str:
    # Expression-mode-driven shot framing (CINEMATIC ENGAGEMENT RULE):
    # - environment / absence / silence  → wide shot (let the space carry the emotion)
    # - object                           → extreme-close-up of the object
    # - body_posture                     → medium (read the body, not the face)
    # - memory_warmth                    → medium-close, soft
    # - face                             → close-up (reserved for actual emotional peaks)
    em = (expression_mode or "").lower()
    if em in ("environment", "absence", "silence"):
        return "wide"
    if em == "object":
        return "extreme-close-up"
    if em == "body_posture":
        return "medium"
    if em == "memory_warmth":
        return "medium-close"
    if em == "face":
        return "close-up"

    if shot_num == 1 and total > 1:
        # Establishing shot — wide unless the scene's dominant mode is intimate (face/object/memory_warmth)
        return "wide"
    if viz_mode == "symbolic":
        return "close-up"
    if viz_mode == "indirect":
        return "medium"
    if temporal == "memory":
        return "medium-close"
    if shot_num == total:
        return "medium"
    # Vary shot types for middle shots
    cycle = ["medium", "close-up", "medium-close", "medium"]
    return cycle[(shot_num - 1) % len(cycle)]


def _select_camera_height(viz_mode: str, narrative_mode: str, emotion: str, intensity: str = "") -> str:
    if narrative_mode == "symbolic" or viz_mode == "symbolic":
        return "eye-level"
    emotion_lower = emotion.lower()
    intensity_lower = (intensity or "").lower()
    if "power" in emotion_lower or "anger" in emotion_lower or "empowerment" in emotion_lower:
        return "low-angle"
    # High-angle for grief/sorrow ONLY at genuine peak intensity — otherwise eye-level keeps it dignified
    if "vulnerability" in emotion_lower or "sorrow" in emotion_lower or "grief" in emotion_lower or "breakdown" in emotion_lower:
        return "high-angle" if intensity_lower == "high" else "eye-level"
    if "liberation" in emotion_lower:
        return "low-angle"
    return "eye-level"


def _select_camera_behavior(viz_mode: str, temporal: str, emotion: str) -> str:
    if temporal == "memory":
        return "slow-pan-right"
    if viz_mode == "symbolic":
        return "static"
    emotion_lower = emotion.lower()
    if "intensity" in emotion_lower or "urgency" in emotion_lower or "energy" in emotion_lower:
        return "handheld"
    if "contemplation" in emotion_lower or "serenity" in emotion_lower:
        return "dolly-in"
    return "static"


def _build_subject_action(line_meaning: Dict, viz_mode: str, narrative_mode: str, expression_mode: str = "", scene: Dict = None) -> str:
    text = line_meaning.get("text", "")
    literal = line_meaning.get("literal_meaning", "")
    implied = line_meaning.get("implied_meaning", "")
    emotion = line_meaning.get("emotional_meaning", "contemplative")
    scene = scene or {}
    location = scene.get("location") or "the space"
    objs = scene.get("objects_of_significance") or []
    obj_hint = objs[0] if objs else "an object she has been holding"

    em = (expression_mode or "").lower()
    # Expression-mode-driven action: keeps the camera off the face when the line calls for environment/object/silence
    if em == "environment":
        return f"no person in frame — {location} carrying the emotion ({emotion}); space, light, weather do the work"
    if em == "absence":
        return f"the empty place where someone used to be — {location}, traces of presence (an unworn shawl, two cups but one untouched), no figure"
    if em == "silence":
        return f"stillness in {location} — held breath, no movement, ambient quiet conveying {emotion}"
    if em == "object":
        return f"tight focus on {obj_hint} — its stillness carries {emotion}, no face in frame"
    if em == "body_posture":
        return f"subject seen from behind or in profile — slumped shoulders, still hands, the way she sits at the threshold; face not the focus ({emotion})"
    if em == "memory_warmth":
        return f"a remembered moment — softer, warmer light than the present scene, gentle gesture between two people ({emotion})"
    if em == "face":
        return f"close on subject's face — {emotion} held quietly; small involuntary movement (a swallow, a blink), no theatrical crying unless intensity is HIGH"

    if viz_mode == "direct":
        return literal if literal else f"subject present — {text[:60]}"
    if viz_mode == "indirect":
        return implied if implied else f"suggested through environment — {text[:60]}"
    if viz_mode == "symbolic":
        return f"symbolic visual — {implied[:80]}" if implied else f"metaphorical image for: {text[:60]}"
    return f"atmospheric presence — mood: {emotion}"


def _determine_lighting(scene: Dict, temporal: str, emotion: str) -> str:
    tod = scene.get("time_of_day", "unspecified")
    if tod == "unspecified":
        if temporal == "memory":
            return "warm golden light, slightly overexposed, soft focus edges"
        if "sorrow" in emotion.lower() or "melancholy" in emotion.lower():
            return "muted natural light, cool tones, shadow-heavy"
        if "darkness" in emotion.lower():
            return "harsh single-source light, deep shadows, noir atmosphere"
        return "natural ambient light, balanced exposure"
    tod_map = {
        "dawn": "soft pink-gold pre-sunrise light, gentle rim lighting",
        "morning": "clean warm morning light, long soft shadows",
        "afternoon": "bright overhead light, defined shadows",
        "evening": "golden hour light, warm tones, long shadows",
        "night": "low-key lighting, practical light sources, deep shadows",
        "late_night": "minimal cool light, near darkness, isolated light pools",
    }
    return tod_map.get(tod, "natural ambient light")


def _build_negative_constraints(viz_mode: str, narrative_mode: str, context: Dict) -> List[str]:
    negatives = ["no text overlay", "no watermark", "no split screen"]
    restrictions = context.get("restrictions", [])
    cultural = context.get("cultural_setting", {})
    pack_restrictions = cultural.get("restrictions", [])
    for r in (restrictions + pack_restrictions)[:3]:
        negatives.append(r)
    if viz_mode == "symbolic":
        negatives.append("no literal interpretation of metaphor")
    if narrative_mode == "memory":
        negatives.append("no sharp modern digital look")
    return negatives[:6]


def _motion_constraints(camera_behavior: str, duration: float) -> str:
    if camera_behavior == "static":
        return f"no camera movement, hold for {duration}s"
    if "pan" in camera_behavior:
        return f"smooth {camera_behavior}, complete arc within {duration}s"
    if camera_behavior == "handheld":
        return f"subtle handheld sway, controlled within {duration}s"
    if camera_behavior == "dolly-in":
        return f"slow dolly push-in over {duration}s"
    return f"{camera_behavior} movement over {duration}s"


def _build_generation_wording(action: str, shot_type: str, camera_height: str, light: str, objects: List[str], viz_mode: str) -> str:
    parts = [f"{shot_type} shot", f"{camera_height}", action[:120]]
    if light:
        parts.append(light[:60])
    if objects:
        parts.append(f"includes: {', '.join(objects[:3])}")
    return ". ".join(parts)
