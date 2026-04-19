"""
Prompt Inheritance System
Formal cascade: Project → Scene → Shot

Each level defines inheritable properties. Lower levels override higher ones.
The final prompt is compiled by merging all three levels, with shot-specific
content being the unique part and project/scene context inherited automatically.

This means:
- Changing project palette changes ALL prompts
- Changing scene location updates all shots in that scene
- Shot prompts stay lean (one shot = one intention)
- Cultural restrictions can never be accidentally dropped
"""
from typing import Dict, List, Any, Optional


def build_project_context(
    project: Dict[str, Any],
    context_packet: Dict[str, Any],
    creative_brief: Optional[Dict[str, Any]] = None,
    vibe_preset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the PROJECT-level inheritable context.
    This is the "world" that every scene and shot lives in.
    """
    settings = project.get("settings", {})
    cultural = context_packet.get("cultural_setting", {})
    brief_aesthetic = {}
    if creative_brief:
        brief_aesthetic = creative_brief.get("visual_aesthetic", {})

    return {
        # Visual world
        "realism_style": _resolve(
            brief_aesthetic.get("style"),
            _realism_label(settings.get("realism_level", "")),
        ),
        "color_palette": _resolve(
            brief_aesthetic.get("color_palette"),
            cultural.get("visual_palette"),
        ),
        "cinematography_style": brief_aesthetic.get("cinematography_style", ""),
        "lighting_mood": brief_aesthetic.get("lighting_mood", ""),
        "aspect_ratio": settings.get("aspect_ratio", "16:9"),

        # Cultural
        "culture_pack": cultural.get("pack_name", ""),
        "restrictions": (
            context_packet.get("restrictions", [])
            + cultural.get("restrictions", [])
        ),
        "common_misinterpretations": cultural.get("common_misinterpretations", []),

        # Narrative
        "narrative_mode": context_packet.get("narrative_mode", ""),
        "input_type": context_packet.get("input_type", project.get("input_mode", "")),

        # Vibe
        "vibe_shot_direction": vibe_preset.get("shot_direction", "") if vibe_preset else "",
        "vibe_avoid": vibe_preset.get("avoid", []) if vibe_preset else [],
    }


def build_scene_context(
    scene: Dict[str, Any],
    project_ctx: Dict[str, Any],
    characters: Optional[List[Dict]] = None,
    environments: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Build the SCENE-level inheritable context.
    Inherits from project, adds scene-specific setting.
    """
    # Find matching environment
    env_desc = ""
    if environments:
        loc = scene.get("location", "").lower()
        for env in environments:
            if env.get("name", "").lower() in loc or loc in env.get("name", "").lower():
                env_desc = env.get("description", env.get("visual_details", ""))[:80]
                break
        if not env_desc and environments:
            env_desc = environments[0].get("description", "")[:80]

    # Find characters in this scene
    scene_chars = []
    if characters:
        scene_text = scene.get("lyric_text", "").lower() + " " + scene.get("purpose", "").lower()
        for ch in characters:
            if ch.get("name", "").lower() in scene_text:
                scene_chars.append(ch)
        if not scene_chars and characters:
            scene_chars = [characters[0]]

    return {
        # Inherited from project (can be overridden per-scene)
        **project_ctx,

        # Scene-specific
        "location": scene.get("location", ""),
        "time_of_day": scene.get("time_of_day", ""),
        "emotional_temperature": scene.get("emotional_temperature", ""),
        "temporal_status": scene.get("temporal_status", "present"),
        "visual_motif": scene.get("visual_motif_priority", ""),
        "environment_description": env_desc,
        "scene_characters": [
            f"{c.get('name', '')}: {c.get('appearance', c.get('description', ''))[:50]}"
            for c in scene_chars[:2]
        ],
    }


def compile_shot_prompt(
    shot: Dict[str, Any],
    scene_ctx: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Compile the final prompt for a shot by inheriting from scene_ctx
    (which already inherited from project_ctx).

    The shot only adds what's UNIQUE to this shot:
    - Camera framing
    - Subject action
    - Emotional micro-state
    - Light (if different from scene default)
    - Secondary objects

    Everything else cascades from scene → project.
    """
    sep = model_config.get("separator", ". ")
    max_len = model_config.get("max_length", 300)
    supports_motion = model_config.get("supports_motion", False)
    supports_negative = model_config.get("supports_negative", True)

    parts = []

    # Prefix
    if model_config.get("prefix"):
        parts.append(model_config["prefix"])

    # ─── SHOT-SPECIFIC (unique to this frame) ───
    parts.append(f"{shot.get('shot_type', 'medium')} shot")
    if shot.get("camera_height") and shot["camera_height"] != "eye-level":
        parts.append(shot["camera_height"])

    action = shot.get("subject_action", "")
    if action:
        parts.append(action)

    emotion = shot.get("emotional_micro_state", "")
    if emotion:
        parts.append(f"mood: {emotion}")

    light = shot.get("light_description", "")
    if light:
        parts.append(light[:60])

    objects = shot.get("secondary_objects", [])
    if objects:
        parts.append(f"includes {', '.join(objects[:3])}")

    if supports_motion:
        cam = shot.get("camera_behavior", "static")
        if cam != "static":
            parts.append(f"camera: {cam}")

    # ─── INHERITED FROM SCENE ───
    scene_chars = scene_ctx.get("scene_characters", [])
    if scene_chars:
        parts.append(sep.join(scene_chars[:2]))

    env = scene_ctx.get("environment_description", "")
    if env:
        parts.append(f"setting: {env[:60]}")
    elif scene_ctx.get("location") and scene_ctx["location"] != "unspecified":
        parts.append(f"setting: {scene_ctx['location']}")

    tod = scene_ctx.get("time_of_day", "")
    if tod and tod != "unspecified":
        parts.append(f"time: {tod}")

    # ─── INHERITED FROM PROJECT ───
    style = scene_ctx.get("realism_style", "")
    if style:
        parts.append(style[:60])

    palette = scene_ctx.get("color_palette", [])
    if palette and isinstance(palette, list):
        parts.append(f"palette: {', '.join(palette[:3])}")

    vibe_dir = scene_ctx.get("vibe_shot_direction", "")
    if vibe_dir:
        parts.append(vibe_dir[:60])

    ar = scene_ctx.get("aspect_ratio", "16:9")

    # Suffix
    suffix = model_config.get("suffix", "")
    if suffix:
        suffix = suffix.replace("{ar}", ar)

    positive = sep.join(p for p in parts if p)
    if suffix:
        positive += suffix
    if len(positive) > max_len:
        positive = positive[:max_len - 3] + "..."

    # ─── NEGATIVE (inherited restrictions + shot constraints + model + vibe) ───
    negative = ""
    if supports_negative:
        neg_parts = list(shot.get("negative_constraints", []))
        neg_parts.extend(scene_ctx.get("restrictions", [])[:3])
        neg_parts.extend(scene_ctx.get("vibe_avoid", [])[:3])
        model_negs = _model_negatives(model_config.get("_model_target", ""))
        neg_parts.extend(model_negs)
        # Platform safety net: songs always get clean-aesthetic negatives regardless of
        # whether the context engine expressed it in cultural restrictions this run.
        if scene_ctx.get("input_type", "") in ("song", "ghazal", "qawwali"):
            neg_parts.extend([
                "dirt or grime on surfaces",
                "soiled clothing",
                "raw poverty aesthetic",
                "grimy textures",
            ])
        negative = ", ".join(list(dict.fromkeys(neg_parts)))[:300]

    # Style injection
    culture = scene_ctx.get("culture_pack", "")
    style_injection = f"Cultural context: {culture}" if culture else ""

    return {
        "positive_prompt": positive,
        "negative_prompt": negative,
        "style_injection": style_injection,
        "aspect_ratio": ar,
    }


def _resolve(*values):
    """Return first non-empty value."""
    for v in values:
        if v:
            return v
    return ""


def _realism_label(key: str) -> str:
    return {
        "realist": "photorealistic, grounded naturalism",
        "poetic_realism": "cinematic, poetic realism, soft film grain",
        "symbolic_abstraction": "stylized, symbolic visual language",
        "commercial_gloss": "polished, high-production commercial look",
        "documentary_naturalism": "documentary style, natural lighting",
    }.get(key, "")


def _model_negatives(model_target: str) -> List[str]:
    negs = {
        "wan_2_6": ["complex hand gestures", "fast dancing", "large crowd"],
        "sdxl": ["blurry", "low quality", "deformed"],
        "flux": ["blurry", "low quality", "deformed"],
    }
    return negs.get(model_target, [])
