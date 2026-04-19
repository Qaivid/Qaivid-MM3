"""
Prompt Engine Service — v2
Translates shot logic into model-specific prompts.
Now includes full model adapter system ported from Qaivid 1.0 prompt-engine.
Does NOT reinterpret meaning — only renders existing cinematic decisions.
"""
from typing import Dict, List, Any
import uuid
from models import now_utc

# Model-specific configurations (expanded with Qaivid 1.0 adapters)
MODEL_CONFIGS = {
    "generic": {
        "max_length": 300, "separator": ". ", "style": "descriptive",
        "supports_negative": True, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
    "midjourney": {
        "max_length": 250, "separator": ", ", "style": "comma-separated",
        "supports_negative": True, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": " --ar {ar} --v 6.1",
    },
    "dall-e": {
        "max_length": 400, "separator": ". ", "style": "descriptive",
        "supports_negative": False, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
    "flux": {
        "max_length": 350, "separator": ", ", "style": "tag-flow",
        "supports_negative": True, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
    "sdxl": {
        "max_length": 300, "separator": ", ", "style": "tag-based",
        "supports_negative": True, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
    "runway": {
        "max_length": 200, "separator": ". ", "style": "concise-cinematic",
        "supports_negative": False, "supports_motion": True, "supports_camera": True,
        "prefix": "", "suffix": "",
    },
    "kling": {
        "max_length": 250, "separator": ". ", "style": "concise-cinematic",
        "supports_negative": True, "supports_motion": True, "supports_camera": True,
        "prefix": "", "suffix": "",
    },
    "wan_2_6": {
        "max_length": 200, "separator": ", ", "style": "concise-safe",
        "supports_negative": True, "supports_motion": True, "supports_camera": True,
        "prefix": "", "suffix": "",
        "risk_notes": "Avoid complex hand movements, crowds, fast dance. Prefer static/slow-pan.",
    },
    "veo": {
        "max_length": 300, "separator": ". ", "style": "descriptive-cinematic",
        "supports_negative": False, "supports_motion": True, "supports_camera": True,
        "prefix": "", "suffix": "",
    },
    "pika": {
        "max_length": 200, "separator": ", ", "style": "concise-cinematic",
        "supports_negative": True, "supports_motion": True, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
    "stable-diffusion": {
        "max_length": 300, "separator": ", ", "style": "tag-based",
        "supports_negative": True, "supports_motion": False, "supports_camera": False,
        "prefix": "", "suffix": "",
    },
}


def build_prompts_for_shot(
    shot: Dict[str, Any],
    scene: Dict[str, Any],
    context_packet: Dict[str, Any],
    project_settings: Dict[str, Any] = None,
    model_target: str = "generic",
    characters: List[Dict] = None,
    environments: List[Dict] = None,
) -> Dict[str, Any]:
    config = MODEL_CONFIGS.get(model_target, MODEL_CONFIGS["generic"])
    sep = config["separator"]
    max_len = config["max_length"]
    style = config["style"]

    parts = []

    # Model prefix
    if config.get("prefix"):
        parts.append(config["prefix"])

    # Shot framing
    parts.append(f"{shot.get('shot_type', 'medium')} shot")
    if shot.get("camera_height") and shot["camera_height"] != "eye-level":
        parts.append(f"{shot['camera_height']}")

    # Subject action — core
    action = shot.get("subject_action", "")
    if action:
        parts.append(action)

    # Character reference injection (if available)
    if characters:
        scene_chars = _match_characters(shot, scene, characters)
        if scene_chars:
            char_desc = sep.join(f"{c['name']}: {c.get('appearance', c.get('description', ''))[:50]}" for c in scene_chars[:2])
            parts.append(char_desc)

    # Environment reference injection (if available)
    if environments:
        env = _match_environment(scene, environments)
        if env:
            parts.append(f"setting: {env.get('description', env.get('name', ''))[:60]}")

    # Emotional micro-state
    emotion = shot.get("emotional_micro_state", "")
    if emotion and style != "tag-based":
        parts.append(f"mood: {emotion}")

    # Lighting
    light = shot.get("light_description", "")
    if light:
        parts.append(light[:60])

    # Secondary objects
    objects = shot.get("secondary_objects", [])
    if objects:
        parts.append(f"includes {', '.join(objects[:3])}")

    # Camera behavior (only for video models)
    if config["supports_motion"]:
        cam = shot.get("camera_behavior", "static")
        if cam != "static":
            parts.append(f"camera: {cam}")

    # Scene location fallback
    location = scene.get("location", "")
    if location and location != "unspecified" and not environments:
        parts.append(f"setting: {location}")

    tod = scene.get("time_of_day", "")
    if tod and tod != "unspecified":
        parts.append(f"time: {tod}")

    # Style injection
    settings = project_settings or {}
    realism = settings.get("realism_level", "")
    if realism:
        style_map = {
            "realist": "photorealistic, grounded naturalism",
            "poetic_realism": "cinematic, poetic realism, soft film grain",
            "symbolic_abstraction": "stylized, symbolic visual language",
            "commercial_gloss": "polished, high-production commercial look",
            "documentary_naturalism": "documentary style, natural lighting, observational",
        }
        if realism in style_map:
            parts.append(style_map[realism])

    # Cultural palette
    cultural = context_packet.get("cultural_setting", {})
    palette = cultural.get("visual_palette", [])
    if palette and style not in ("concise-safe",):
        parts.append(f"color palette: {', '.join(palette[:3])}")

    # Aspect ratio
    ar = settings.get("aspect_ratio", "16:9")

    # Model suffix
    suffix = config.get("suffix", "")
    if suffix:
        suffix = suffix.replace("{ar}", ar)

    positive = sep.join(parts)
    if suffix:
        positive = positive + suffix
    if len(positive) > max_len:
        positive = positive[:max_len - 3] + "..."

    # Negative prompt (model-aware)
    negative = ""
    if config["supports_negative"]:
        neg_parts = list(shot.get("negative_constraints", []))
        cultural_restrictions = cultural.get("restrictions", [])
        neg_parts.extend(cultural_restrictions[:2])
        # Model-specific negatives
        if model_target == "wan_2_6":
            neg_parts.extend(["complex hand gestures", "fast dancing", "large crowd"])
        if model_target in ("sdxl", "flux"):
            neg_parts.extend(["blurry", "low quality", "deformed"])
        negative = ", ".join(list(dict.fromkeys(neg_parts)))[:250]

    # Style injection
    pack_name = cultural.get("pack_name", "")
    style_injection = f"Cultural context: {pack_name}" if pack_name else ""

    # Risk notes for this model
    risk_notes = config.get("risk_notes", "")

    return {
        "id": str(uuid.uuid4()),
        "project_id": shot.get("project_id", ""),
        "shot_id": shot.get("id", ""),
        "scene_id": shot.get("scene_id", ""),
        "model_target": model_target,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "style_injection": style_injection,
        "aspect_ratio": ar,
        "duration": shot.get("duration_hint", 3.0),
        "risk_notes": risk_notes,
        "created_at": now_utc(),
    }


def build_all_prompts(
    shots: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    context_packet: Dict[str, Any],
    project_settings: Dict[str, Any] = None,
    model_target: str = "generic",
    characters: List[Dict] = None,
    environments: List[Dict] = None,
    project: Dict[str, Any] = None,
    creative_brief: Dict[str, Any] = None,
    vibe_preset: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Build all prompts using the inheritance system:
    Project context → Scene context → Shot compilation
    """
    from services.prompt_inheritance import build_project_context, build_scene_context, compile_shot_prompt

    config = MODEL_CONFIGS.get(model_target, MODEL_CONFIGS["generic"])
    config_with_target = {**config, "_model_target": model_target}

    # Build project-level context (computed once, inherited everywhere)
    proj_ctx = build_project_context(
        project or {"settings": project_settings or {}},
        context_packet,
        creative_brief,
        vibe_preset,
    )

    # Build scene-level contexts (computed once per scene)
    scene_ctx_cache = {}
    for scene in scenes:
        scene_ctx_cache[scene["id"]] = build_scene_context(
            scene, proj_ctx, characters, environments
        )

    # Compile each shot using inherited context
    prompts = []
    for shot in shots:
        scene_id = shot.get("scene_id", "")
        scene_ctx = scene_ctx_cache.get(scene_id, proj_ctx)
        compiled = compile_shot_prompt(shot, scene_ctx, config_with_target)

        prompts.append({
            "id": str(uuid.uuid4()),
            "project_id": shot.get("project_id", ""),
            "shot_id": shot.get("id", ""),
            "scene_id": scene_id,
            "model_target": model_target,
            "positive_prompt": compiled["positive_prompt"],
            "negative_prompt": compiled["negative_prompt"],
            "style_injection": compiled["style_injection"],
            "aspect_ratio": compiled["aspect_ratio"],
            "duration": shot.get("duration_hint", 3.0),
            "risk_notes": config.get("risk_notes", ""),
            "inheritance": "project→scene→shot",
            "created_at": now_utc(),
        })

    return prompts


def _match_characters(shot: Dict, scene: Dict, characters: List[Dict]) -> List[Dict]:
    """Find characters relevant to this shot/scene."""
    matched = []
    scene_text = (scene.get("lyric_text", "") + " " + shot.get("subject_action", "")).lower()
    for char in characters:
        name_lower = char.get("name", "").lower()
        if name_lower and name_lower in scene_text:
            matched.append(char)
    if not matched and characters:
        # Return primary character if no match
        return [characters[0]]
    return matched[:2]


def _match_environment(scene: Dict, environments: List[Dict]) -> Dict:
    """Find environment matching this scene."""
    location = scene.get("location", "").lower()
    for env in environments:
        env_name = env.get("name", "").lower()
        if env_name and (env_name in location or location in env_name):
            return env
    return environments[0] if environments else {}


def get_available_models() -> List[Dict[str, Any]]:
    """Return list of available model targets with capabilities."""
    return [
        {"id": k, "supports_negative": v["supports_negative"],
         "supports_motion": v["supports_motion"], "supports_camera": v["supports_camera"],
         "max_length": v["max_length"], "style": v["style"]}
        for k, v in MODEL_CONFIGS.items()
    ]
