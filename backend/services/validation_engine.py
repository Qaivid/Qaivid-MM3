"""
Validation Engine Service
Validates context, scenes, shots, and prompts before export.
"""
from typing import Dict, List, Any


def validate_project(context_packet: Dict, scenes: List[Dict], shots: List[Dict], prompts: List[Dict]) -> Dict[str, Any]:
    issues = []

    # Context validation
    issues.extend(_validate_context(context_packet))

    # Scene validation
    issues.extend(_validate_scenes(scenes, context_packet))

    # Shot validation
    issues.extend(_validate_shots(shots))

    # Prompt validation
    issues.extend(_validate_prompts(prompts))

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]

    return {
        "total_issues": len(issues),
        "errors": len(errors),
        "warnings": len(warnings),
        "infos": len(infos),
        "issues": issues,
    }


def _validate_context(ctx: Dict) -> List[Dict]:
    issues = []
    if not ctx:
        return [{"severity": "error", "layer": "context", "target_id": "", "message": "No context packet found", "suggestion": "Run interpretation first"}]

    scores = ctx.get("confidence_scores", {})
    for field, score in scores.items():
        if isinstance(score, (int, float)) and score < 0.5:
            issues.append({
                "severity": "warning",
                "layer": "context",
                "target_id": ctx.get("id", ""),
                "message": f"Low confidence in '{field}': {score}",
                "suggestion": f"Consider reviewing and overriding {field} assumptions",
            })

    ambiguities = ctx.get("ambiguity_flags", [])
    for amb in ambiguities:
        issues.append({
            "severity": "warning",
            "layer": "context",
            "target_id": ctx.get("id", ""),
            "message": f"Ambiguity in '{amb.get('field', 'unknown')}': {amb.get('reason', '')}",
            "suggestion": "User override recommended",
        })

    speaker = ctx.get("speaker_model", {})
    if speaker.get("identity", "unspecified") == "unspecified":
        issues.append({
            "severity": "warning",
            "layer": "context",
            "target_id": ctx.get("id", ""),
            "message": "Speaker identity is unspecified",
            "suggestion": "Set speaker identity in overrides for better scene design",
        })

    return issues


def _validate_scenes(scenes: List[Dict], ctx: Dict) -> List[Dict]:
    issues = []
    for scene in scenes:
        sid = scene.get("id", "")
        if not scene.get("location") or scene["location"] == "unspecified":
            issues.append({
                "severity": "info",
                "layer": "scene",
                "target_id": sid,
                "message": f"Scene {scene.get('scene_number', '?')}: no specific location set",
                "suggestion": "Consider setting a location for better shot design",
            })

        risks = scene.get("visual_risk_notes", [])
        for risk in risks:
            issues.append({
                "severity": "warning",
                "layer": "scene",
                "target_id": sid,
                "message": f"Scene {scene.get('scene_number', '?')}: {risk}",
                "suggestion": "Review visualization mode assignments",
            })

        # Check for too many objects
        objs = scene.get("objects_of_significance", [])
        if len(objs) > 5:
            issues.append({
                "severity": "warning",
                "layer": "scene",
                "target_id": sid,
                "message": f"Scene {scene.get('scene_number', '?')}: too many significant objects ({len(objs)})",
                "suggestion": "Reduce to 3-5 key objects per scene",
            })

    return issues


def _validate_shots(shots: List[Dict]) -> List[Dict]:
    issues = []
    for shot in shots:
        sid = shot.get("id", "")

        # Check secondary objects count
        objs = shot.get("secondary_objects", [])
        if len(objs) > 3:
            issues.append({
                "severity": "error",
                "layer": "shot",
                "target_id": sid,
                "message": f"Shot {shot.get('shot_number', '?')}: too many secondary objects ({len(objs)}). Max is 3.",
                "suggestion": "Remove excess objects to maintain shot clarity",
            })

        # Check for multi-phase instructions
        action = shot.get("subject_action", "")
        if " then " in action.lower() or " and then " in action.lower():
            issues.append({
                "severity": "error",
                "layer": "shot",
                "target_id": sid,
                "message": f"Shot {shot.get('shot_number', '?')}: multi-phase instruction detected ('then')",
                "suggestion": "Split into separate shots. One shot = one intention.",
            })

        # Check prompt length via generation_safe_wording
        wording = shot.get("generation_safe_wording", "")
        if len(wording) > 400:
            issues.append({
                "severity": "warning",
                "layer": "shot",
                "target_id": sid,
                "message": f"Shot {shot.get('shot_number', '?')}: generation wording too long ({len(wording)} chars)",
                "suggestion": "Simplify to under 300 characters for best model results",
            })

        if not shot.get("visual_priority"):
            issues.append({
                "severity": "warning",
                "layer": "shot",
                "target_id": sid,
                "message": f"Shot {shot.get('shot_number', '?')}: no visual priority set",
                "suggestion": "Set a clear visual priority for this shot",
            })

    return issues


def _validate_prompts(prompts: List[Dict]) -> List[Dict]:
    issues = []
    for prompt in prompts:
        pid = prompt.get("id", "")
        positive = prompt.get("positive_prompt", "")

        if not positive:
            issues.append({
                "severity": "error",
                "layer": "prompt",
                "target_id": pid,
                "message": "Empty positive prompt",
                "suggestion": "Regenerate prompts from shot data",
            })
            continue

        if len(positive) > 500:
            issues.append({
                "severity": "warning",
                "layer": "prompt",
                "target_id": pid,
                "message": f"Prompt too long ({len(positive)} chars)",
                "suggestion": "Trim to under 400 chars for best generation results",
            })

    return issues
