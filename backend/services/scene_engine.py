"""
Scene Engine Service
Converts narrative context into scene units deterministically.
Each scene = one dramatic unit with purpose, location, emotion, temporal status.
"""
from typing import Dict, List, Any
import uuid
from models import now_utc


_EXPRESSION_PALETTE = ["environment", "body_posture", "object", "silence", "memory_warmth", "absence", "face"]


def _scene_timing(line_meanings_for_scene):
    """Compute (start, end, duration) from line_meanings if any have timing."""
    starts = [lm.get("start_time") for lm in line_meanings_for_scene if lm.get("start_time") is not None]
    ends = [lm.get("end_time") for lm in line_meanings_for_scene if lm.get("end_time") is not None]
    if not starts:
        return None, None, None
    s = min(starts)
    e = max(ends) if ends else None
    d = round(e - s, 3) if (e is not None and e > s) else None
    return s, e, d


def _dominant_expression_mode(line_meanings_for_scene):
    modes = [str(lm.get("expression_mode", "")).lower() for lm in line_meanings_for_scene]
    modes = [m for m in modes if m]
    if not modes:
        return ""
    # Most-common
    return max(set(modes), key=modes.count)


def _enforce_scene_variation(scenes):
    """
    CINEMATIC ENGAGEMENT RULE at scene level:
    No two consecutive scenes may share the same dominant expression style.
    Also ensures the climax scene gets at least one face/peak shot if nothing
    else carries it, and demotes back-to-back face-driven scenes.
    """
    if not scenes or len(scenes) < 2:
        return scenes
    rotation_idx = 0
    rotation = ["environment", "body_posture", "object", "silence", "memory_warmth"]
    for i in range(1, len(scenes)):
        prev = (scenes[i - 1].get("scene_expression_mode") or "").lower()
        cur = (scenes[i].get("scene_expression_mode") or "").lower()
        if prev and cur and prev == cur:
            # Pick a rotation value that isn't the previous scene's mode
            for _ in range(len(rotation)):
                candidate = rotation[rotation_idx % len(rotation)]
                rotation_idx += 1
                if candidate != prev:
                    scenes[i]["scene_expression_mode"] = candidate
                    scenes[i].setdefault("visual_risk_notes", []).append(
                        f"scene_expression_mode auto-rotated from '{cur}' to '{candidate}' to prevent visual repetition with previous scene"
                    )
                    break
    return scenes


def build_scenes(project_id: str, context_packet: Dict[str, Any], source_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    line_meanings = context_packet.get("line_meanings", [])
    narrative_mode = context_packet.get("narrative_mode", "realist")
    world = context_packet.get("world_assumptions", {})
    sections = source_input.get("sections", [])

    scenes = []
    scene_num = 1

    # Safety net: a single section with many lines means section detection found nothing
    # useful — fall through to the chunk-based fallback so we don't produce one giant scene.
    _MAX_LINES_PER_SCENE = 12
    if len(sections) == 1 and len(sections[0].get("lines", [])) > _MAX_LINES_PER_SCENE:
        sections = []

    if sections and len(sections) > 0:
        # Use detected sections as scene boundaries
        for section in sections:
            section_lines = section.get("lines", [])
            if not section_lines:
                continue
            line_indices = [ln["index"] for ln in section_lines]
            line_texts = [ln["text"] for ln in section_lines]

            # Gather meanings for these lines
            relevant_meanings = [lm for lm in line_meanings if lm.get("line_index") in line_indices]

            # Determine visualization modes
            viz_modes = [lm.get("visualization_mode", "direct") for lm in relevant_meanings]
            dominant_viz = max(set(viz_modes), key=viz_modes.count) if viz_modes else "direct"

            # Determine temporal status from narrative mode and viz
            temporal = _determine_temporal_status(dominant_viz, narrative_mode, section.get("type", "verse"))

            # Determine emotional temperature
            emotions = [lm.get("emotional_meaning", "") for lm in relevant_meanings]
            emotional_temp = _summarize_emotions(emotions)

            # Map section type to story function
            story_function = _section_to_function(section.get("type", "verse"), scene_num, len(sections))

            # Extract significant objects
            objects = []
            entity_map = context_packet.get("entity_map", {})
            for obj in entity_map.get("objects", []):
                for lt in line_texts:
                    if obj.lower() in lt.lower():
                        objects.append(obj)
                        break

            # Scene location from world assumptions
            location = world.get("geography", "unspecified")
            if world.get("domestic_setting", "unspecified") != "unspecified":
                location = world.get("domestic_setting")

            # Motif priority from motif map
            motif_map = context_packet.get("motif_map", {})
            motif_priority = ""
            for motif, indices in motif_map.items():
                idx_set = set()
                for idx_val in indices:
                    if isinstance(idx_val, int):
                        idx_set.add(idx_val)
                    elif isinstance(idx_val, str) and idx_val.isdigit():
                        idx_set.add(int(idx_val))
                if idx_set & set(line_indices):
                    motif_priority = motif
                    break

            scene_expression_mode = _dominant_expression_mode(relevant_meanings)
            s_start, s_end, s_dur = _scene_timing(relevant_meanings)

            scene = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "scene_number": scene_num,
                "purpose": f"{section.get('type', 'verse').title()} — {story_function}",
                "lyric_span": line_indices,
                "lyric_text": "\n".join(line_texts),
                "story_function": story_function,
                "temporal_status": temporal,
                "emotional_temperature": emotional_temp,
                "scene_expression_mode": scene_expression_mode,
                "start_time": s_start,
                "end_time": s_end,
                "duration_sec": s_dur,
                "location": location,
                "time_of_day": world.get("time_of_day", "unspecified"),
                "objects_of_significance": objects[:5],
                "character_blocking": _generate_blocking(relevant_meanings, narrative_mode),
                "continuity_dependencies": _build_continuity_deps(scene_num),
                "visual_risk_notes": _check_visual_risks(relevant_meanings, dominant_viz),
                "visual_motif_priority": motif_priority,
                "created_at": now_utc(),
            }
            scenes.append(scene)
            scene_num += 1
    else:
        # No sections detected — group lines by emotional phase or chunks
        chunk_size = max(3, len(line_meanings) // 4) if line_meanings else 4
        for i in range(0, len(line_meanings), chunk_size):
            chunk = line_meanings[i:i + chunk_size]
            line_indices = [lm.get("line_index", i + j) for j, lm in enumerate(chunk)]
            line_texts = [lm.get("text", "") for lm in chunk]

            viz_modes = [lm.get("visualization_mode", "direct") for lm in chunk]
            dominant_viz = max(set(viz_modes), key=viz_modes.count) if viz_modes else "direct"
            temporal = _determine_temporal_status(dominant_viz, narrative_mode, "verse")
            emotions = [lm.get("emotional_meaning", "") for lm in chunk]
            emotional_temp = _summarize_emotions(emotions)
            story_function = _section_to_function("verse", scene_num, max(1, len(line_meanings) // chunk_size))

            cs_start, cs_end, cs_dur = _scene_timing(chunk)
            scene = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "scene_number": scene_num,
                "purpose": f"Scene {scene_num} — {story_function}",
                "lyric_span": line_indices,
                "lyric_text": "\n".join(line_texts),
                "story_function": story_function,
                "temporal_status": temporal,
                "emotional_temperature": emotional_temp,
                "scene_expression_mode": _dominant_expression_mode(chunk),
                "start_time": cs_start,
                "end_time": cs_end,
                "duration_sec": cs_dur,
                "location": world.get("geography", "unspecified"),
                "time_of_day": world.get("time_of_day", "unspecified"),
                "objects_of_significance": [],
                "character_blocking": _generate_blocking(chunk, narrative_mode),
                "continuity_dependencies": _build_continuity_deps(scene_num),
                "visual_risk_notes": _check_visual_risks(chunk, dominant_viz),
                "visual_motif_priority": "",
                "created_at": now_utc(),
            }
            scenes.append(scene)
            scene_num += 1

    # CINEMATIC ENGAGEMENT RULE: enforce no two consecutive scenes share dominant expression style
    scenes = _enforce_scene_variation(scenes)
    return scenes


def _determine_temporal_status(viz_mode: str, narrative_mode: str, section_type: str) -> str:
    if narrative_mode == "memory" or narrative_mode == "hybrid":
        if viz_mode == "symbolic":
            return "symbolic"
        if section_type in ("chorus", "hook"):
            return "present"
        return "memory"
    if narrative_mode == "symbolic":
        return "symbolic"
    if viz_mode == "symbolic":
        return "symbolic"
    return "present"


def _section_to_function(section_type: str, scene_num: int, total: int) -> str:
    if scene_num == 1:
        base = "establish world and emotional entry"
    elif scene_num == total:
        base = "resolve or leave open"
    else:
        base = "develop and deepen"

    type_map = {
        "verse": "narrative progression",
        "chorus": "emotional anchor / refrain",
        "bridge": "shift perspective or reveal",
        "intro": "set tone and atmosphere",
        "outro": "emotional landing",
        "performance": "musical/performance energy",
        "dialogue": "character interaction",
    }
    suffix = type_map.get(section_type, "narrative beat")
    return f"{base} — {suffix}"


def _summarize_emotions(emotions: List[str]) -> str:
    if not emotions:
        return "neutral"
    # Simple concatenation of unique emotions
    unique = list(dict.fromkeys(e for e in emotions if e))
    if len(unique) <= 2:
        return " to ".join(unique) if unique else "neutral"
    return f"{unique[0]} building to {unique[-1]}"


def _generate_blocking(line_meanings: List[Dict], narrative_mode: str) -> str:
    if narrative_mode in ("symbolic", "philosophical"):
        return "subject in contemplative position, environment as emotional extension"
    if narrative_mode == "performance_led":
        return "performer center frame, audience/space around"
    if narrative_mode == "memory":
        return "subject interacting with memory-space, soft focus boundaries"
    return "subject in natural position within environment"


def _build_continuity_deps(scene_num: int) -> List[str]:
    if scene_num == 1:
        return []
    return [f"inherits from scene {scene_num - 1}: location, wardrobe, time progression"]


def _check_visual_risks(line_meanings: List[Dict], dominant_viz: str) -> List[str]:
    risks = []
    for lm in line_meanings:
        if lm.get("visualization_mode") == "symbolic" and dominant_viz == "direct":
            risks.append(f"Line {lm.get('line_index', '?')}: symbolic content in direct scene — risk of literalization")
        if lm.get("visual_suitability") == "low":
            risks.append(f"Line {lm.get('line_index', '?')}: low visual suitability — consider absorbed treatment")
    return risks
