"""
Continuity Tracking Engine
Tracks character, location, prop, wardrobe, and weather consistency across scenes and shots.
Ported from Qaivid 1.0 continuity-tracker.ts concept.
"""
from typing import Dict, List, Any
import uuid
from models import now_utc


def build_continuity_report(
    project_id: str,
    scenes: List[Dict[str, Any]],
    shots: List[Dict[str, Any]],
    context_packet: Dict[str, Any],
    characters: List[Dict[str, Any]] = None,
    environments: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a comprehensive continuity report for a project."""

    # Track subjects across shots
    subject_tracking = _track_subjects(shots, scenes, context_packet)

    # Track motifs across scenes
    motif_tracking = _track_motifs(scenes, context_packet)

    # Track location continuity
    location_tracking = _track_locations(scenes)

    # Track emotional progression
    emotional_tracking = _track_emotional_arc(scenes)

    # Track temporal logic
    temporal_tracking = _track_temporal_logic(scenes)

    # Build warnings
    warnings = _detect_continuity_breaks(
        scenes, shots, subject_tracking, location_tracking, temporal_tracking, characters, environments
    )

    return {
        "id": str(uuid.uuid4()),
        "project_id": project_id,
        "subjects": subject_tracking,
        "motifs": motif_tracking,
        "locations": location_tracking,
        "emotional_progression": emotional_tracking,
        "temporal_flow": temporal_tracking,
        "warnings": warnings,
        "total_subjects_tracked": len(subject_tracking),
        "total_motifs_tracked": len(motif_tracking),
        "total_warnings": len(warnings),
        "created_at": now_utc(),
    }


def _track_subjects(shots: List[Dict], scenes: List[Dict], context: Dict) -> List[Dict]:
    """Track which subjects appear in which shots and scenes."""
    subjects = {}
    speaker = context.get("speaker_model", {})
    speaker_name = speaker.get("identity", "speaker")
    if speaker_name and speaker_name != "unspecified":
        subjects[speaker_name] = {
            "name": speaker_name,
            "type": "character",
            "first_appearance_shot": None,
            "shot_ids": [],
            "scene_ids": set(),
        }

    entity_map = context.get("entity_map", {})
    for char in entity_map.get("characters", []):
        if char not in subjects:
            subjects[char] = {"name": char, "type": "character", "first_appearance_shot": None, "shot_ids": [], "scene_ids": set()}

    # Scan shots for subject presence
    for shot in shots:
        action = (shot.get("subject_action", "") + " " + shot.get("visual_priority", "")).lower()
        for name, sub in subjects.items():
            if name.lower() in action or "subject" in action:
                sub["shot_ids"].append(shot.get("id", ""))
                sub["scene_ids"].add(shot.get("scene_id", ""))
                if sub["first_appearance_shot"] is None:
                    sub["first_appearance_shot"] = shot.get("id", "")

    return [
        {**v, "scene_ids": list(v["scene_ids"]), "appearance_count": len(v["shot_ids"])}
        for v in subjects.values() if v["shot_ids"]
    ]


def _track_motifs(scenes: List[Dict], context: Dict) -> List[Dict]:
    """Track motif recurrence across scenes."""
    motif_map = context.get("motif_map", {})
    result = []
    for motif_name, line_indices in motif_map.items():
        scene_ids = []
        for scene in scenes:
            span = set(scene.get("lyric_span", []))
            idx_set = set()
            for idx in line_indices:
                if isinstance(idx, int):
                    idx_set.add(idx)
                elif isinstance(idx, str) and idx.isdigit():
                    idx_set.add(int(idx))
            if span & idx_set:
                scene_ids.append(scene.get("id", ""))
        result.append({
            "motif": motif_name,
            "scene_ids": scene_ids,
            "occurrence_count": len(scene_ids),
            "is_recurring": len(scene_ids) > 1,
        })
    return result


def _track_locations(scenes: List[Dict]) -> List[Dict]:
    """Track location usage across scenes."""
    locations = {}
    for scene in scenes:
        loc = scene.get("location", "unspecified")
        if loc not in locations:
            locations[loc] = {"location": loc, "scene_numbers": [], "scene_ids": []}
        locations[loc]["scene_numbers"].append(scene.get("scene_number", 0))
        locations[loc]["scene_ids"].append(scene.get("id", ""))
    return list(locations.values())


def _track_emotional_arc(scenes: List[Dict]) -> List[Dict]:
    """Track emotional progression across scenes."""
    return [
        {
            "scene_number": s.get("scene_number", 0),
            "emotion": s.get("emotional_temperature", "neutral"),
            "temporal": s.get("temporal_status", "present"),
        }
        for s in scenes
    ]


def _track_temporal_logic(scenes: List[Dict]) -> List[Dict]:
    """Track present/memory/symbolic status flow."""
    return [
        {
            "scene_number": s.get("scene_number", 0),
            "status": s.get("temporal_status", "present"),
            "location": s.get("location", "unspecified"),
        }
        for s in scenes
    ]


def _detect_continuity_breaks(
    scenes: List[Dict], shots: List[Dict],
    subjects: List[Dict], locations: List[Dict],
    temporal: List[Dict],
    characters: List[Dict] = None,
    environments: List[Dict] = None,
) -> List[Dict]:
    """Detect potential continuity issues."""
    warnings = []

    # Check for location jumps without scene transition
    for i in range(1, len(scenes)):
        prev_loc = scenes[i - 1].get("location", "unspecified")
        curr_loc = scenes[i].get("location", "unspecified")
        prev_temp = scenes[i - 1].get("temporal_status", "present")
        curr_temp = scenes[i].get("temporal_status", "present")
        if prev_loc != curr_loc and prev_temp == curr_temp == "present":
            warnings.append({
                "type": "location_jump",
                "severity": "info",
                "scene_number": scenes[i].get("scene_number", 0),
                "message": f"Location changes from '{prev_loc}' to '{curr_loc}' without temporal shift",
                "suggestion": "Consider adding a transition scene or changing temporal status",
            })

    # Check for missing character definitions
    if characters is not None:
        char_names = {c.get("name", "").lower() for c in characters}
        for sub in subjects:
            if sub["name"].lower() not in char_names and sub["type"] == "character":
                warnings.append({
                    "type": "unresolved_character",
                    "severity": "warning",
                    "message": f"Subject '{sub['name']}' appears in {sub['appearance_count']} shots but has no character profile",
                    "suggestion": "Create a character reference profile for visual consistency",
                })

    # Check for unreferenced environments
    if environments is not None:
        env_names = {e.get("name", "").lower() for e in environments}
        for loc in locations:
            if loc["location"] != "unspecified" and loc["location"].lower() not in env_names:
                warnings.append({
                    "type": "unresolved_environment",
                    "severity": "info",
                    "message": f"Location '{loc['location']}' used in {len(loc['scene_ids'])} scenes but has no environment profile",
                    "suggestion": "Create an environment reference profile for visual consistency",
                })

    # Check for rapid temporal switching
    for i in range(2, len(temporal)):
        if (temporal[i - 2]["status"] != temporal[i - 1]["status"] and
                temporal[i - 1]["status"] != temporal[i]["status"] and
                temporal[i - 2]["status"] != temporal[i]["status"]):
            warnings.append({
                "type": "rapid_temporal_shift",
                "severity": "warning",
                "scene_number": temporal[i]["scene_number"],
                "message": f"Rapid temporal shifts: {temporal[i-2]['status']} → {temporal[i-1]['status']} → {temporal[i]['status']}",
                "suggestion": "Consider grouping temporal modes for visual coherence",
            })

    return warnings
