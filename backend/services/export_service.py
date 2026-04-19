"""
Export Service
Exports project data in multiple formats: JSON, CSV, prompt list.
"""
import json
import csv
import io
from typing import Dict, List, Any


def export_json(project: Dict, context: Dict, scenes: List[Dict], shots: List[Dict], prompts: List[Dict]) -> str:
    export_data = {
        "project": {
            "id": project.get("id"),
            "name": project.get("name"),
            "status": project.get("status"),
            "input_mode": project.get("input_mode"),
            "settings": project.get("settings"),
        },
        "context_packet": _clean_for_export(context),
        "scenes": [_clean_for_export(s) for s in scenes],
        "shots": [_clean_for_export(s) for s in shots],
        "prompts": [_clean_for_export(p) for p in prompts],
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_csv_shots(shots: List[Dict], prompts: List[Dict]) -> str:
    prompt_map = {p.get("shot_id", ""): p for p in prompts}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "shot_number", "scene_id", "shot_type", "camera_height", "camera_behavior",
        "subject_action", "emotional_micro_state", "light_description",
        "secondary_objects", "duration_hint", "positive_prompt", "negative_prompt"
    ])
    for shot in shots:
        prompt = prompt_map.get(shot.get("id", ""), {})
        writer.writerow([
            shot.get("shot_number", ""),
            shot.get("scene_id", ""),
            shot.get("shot_type", ""),
            shot.get("camera_height", ""),
            shot.get("camera_behavior", ""),
            shot.get("subject_action", ""),
            shot.get("emotional_micro_state", ""),
            shot.get("light_description", ""),
            "|".join(shot.get("secondary_objects", [])),
            shot.get("duration_hint", ""),
            prompt.get("positive_prompt", ""),
            prompt.get("negative_prompt", ""),
        ])
    return output.getvalue()


def export_prompt_list(prompts: List[Dict]) -> str:
    lines = []
    for i, p in enumerate(prompts, 1):
        lines.append(f"--- Shot {i} ---")
        lines.append(f"Model: {p.get('model_target', 'generic')}")
        lines.append(f"Positive: {p.get('positive_prompt', '')}")
        lines.append(f"Negative: {p.get('negative_prompt', '')}")
        if p.get("style_injection"):
            lines.append(f"Style: {p['style_injection']}")
        lines.append(f"Aspect Ratio: {p.get('aspect_ratio', '16:9')}")
        lines.append(f"Duration: {p.get('duration', 3.0)}s")
        lines.append("")
    return "\n".join(lines)


def export_storyboard(scenes: List[Dict], shots: List[Dict]) -> str:
    shot_map = {}
    for s in shots:
        sid = s.get("scene_id", "")
        if sid not in shot_map:
            shot_map[sid] = []
        shot_map[sid].append(s)

    lines = ["STORYBOARD", "=" * 60, ""]
    for scene in scenes:
        lines.append(f"SCENE {scene.get('scene_number', '?')}: {scene.get('purpose', '')}")
        lines.append(f"  Location: {scene.get('location', 'unspecified')}")
        lines.append(f"  Time: {scene.get('time_of_day', 'unspecified')}")
        lines.append(f"  Emotion: {scene.get('emotional_temperature', '')}")
        lines.append(f"  Temporal: {scene.get('temporal_status', 'present')}")
        lines.append(f"  Lyrics: {scene.get('lyric_text', '')[:100]}")
        lines.append("")

        scene_shots = shot_map.get(scene.get("id", ""), [])
        for shot in scene_shots:
            lines.append(f"  SHOT {shot.get('shot_number', '?')}:")
            lines.append(f"    Type: {shot.get('shot_type', '')} | Height: {shot.get('camera_height', '')}")
            lines.append(f"    Camera: {shot.get('camera_behavior', '')}")
            lines.append(f"    Action: {shot.get('subject_action', '')}")
            lines.append(f"    Light: {shot.get('light_description', '')}")
            lines.append(f"    Duration: {shot.get('duration_hint', 3.0)}s")
            lines.append("")
        lines.append("-" * 60)

    return "\n".join(lines)


def _clean_for_export(doc: Dict) -> Dict:
    cleaned = {}
    skip_keys = {"_id", "created_at", "updated_at"}
    for k, v in doc.items():
        if k not in skip_keys:
            cleaned[k] = v
    return cleaned
