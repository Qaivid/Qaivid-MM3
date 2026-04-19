from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid


def gen_id():
    return str(uuid.uuid4())


def now_utc():
    return datetime.now(timezone.utc).isoformat()


# ─── Project ───────────────────────────────────────────────
class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    input_mode: str = "song"  # song, poem, script, story, ad, documentary
    language: str = "auto"
    culture_pack: str = "auto"
    settings: Dict[str, Any] = Field(default_factory=lambda: {
        "target_platform": "general",
        "aspect_ratio": "16:9",
        "shot_density": "medium",
        "realism_level": "poetic_realism",
        "generation_model": "none",
    })


class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    name: str
    description: str = ""
    status: str = "draft"  # draft, input_added, interpreting, interpreted, scenes_built, shots_built, prompts_ready, complete
    input_mode: str = "song"
    language: str = "auto"
    culture_pack: str = "auto"
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)


# ─── Source Input ──────────────────────────────────────────
class SourceInputCreate(BaseModel):
    raw_text: str
    language_hint: str = "auto"
    culture_hint: str = "auto"


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    raw_text: str
    cleaned_text: str = ""
    detected_language: str = ""
    detected_script: str = ""
    detected_type: str = ""
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    lines: List[Dict[str, Any]] = Field(default_factory=list)
    line_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)


# ─── Context Packet (THE CORE OBJECT) ─────────────────────
class LineMeaning(BaseModel):
    line_index: int = 0
    text: str = ""
    literal_meaning: str = ""
    implied_meaning: str = ""
    emotional_meaning: str = ""
    cultural_meaning: str = ""
    visual_suitability: str = "medium"  # high, medium, low
    visualization_mode: str = "direct"  # direct, indirect, symbolic, absorbed, performance_only


class SpeakerModel(BaseModel):
    identity: str = "unspecified"
    gender: str = "unspecified"
    age_range: str = "unspecified"
    social_role: str = "unspecified"
    emotional_state: str = "unspecified"
    relationship_to_addressee: str = "unspecified"


class WorldAssumptions(BaseModel):
    geography: str = "unspecified"
    era: str = "contemporary"
    season: str = "unspecified"
    time_of_day: str = "unspecified"
    social_context: str = "unspecified"
    economic_context: str = "unspecified"
    architecture_style: str = "unspecified"
    domestic_setting: str = "unspecified"


class ContextPacket(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    input_type: str = ""  # song, poem, ghazal, qawwali, script, story, ad, voiceover
    narrative_mode: str = ""  # realist, memory, symbolic, psychological, philosophical, performance_led, hybrid
    language_info: Dict[str, str] = Field(default_factory=lambda: {"primary": "", "script": "", "dialect": ""})
    speaker_model: Dict[str, str] = Field(default_factory=dict)
    addressee_model: Dict[str, str] = Field(default_factory=dict)
    world_assumptions: Dict[str, str] = Field(default_factory=dict)
    cultural_setting: Dict[str, str] = Field(default_factory=dict)
    emotional_arc: List[Dict[str, str]] = Field(default_factory=list)
    narrative_spine: str = ""
    core_theme: str = ""
    dramatic_premise: str = ""
    line_meanings: List[Dict[str, Any]] = Field(default_factory=list)
    motif_map: Dict[str, List[str]] = Field(default_factory=dict)
    entity_map: Dict[str, List[str]] = Field(default_factory=dict)
    restrictions: List[str] = Field(default_factory=list)
    ambiguity_flags: List[Dict[str, str]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=lambda: {
        "overall": 0.0, "cultural": 0.0, "emotional": 0.0, "speaker": 0.0, "narrative_mode": 0.0
    })
    locked_assumptions: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)


# ─── User Override ─────────────────────────────────────────
class UserOverrideCreate(BaseModel):
    field_path: str
    override_value: Any
    locked: bool = True


class UserOverride(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    field_path: str
    original_value: Any = None
    override_value: Any = None
    locked: bool = True
    created_at: str = Field(default_factory=now_utc)


# ─── Scene ─────────────────────────────────────────────────
class Scene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    scene_number: int = 0
    purpose: str = ""
    lyric_span: List[int] = Field(default_factory=list)
    lyric_text: str = ""
    story_function: str = ""
    temporal_status: str = "present"  # present, memory, symbolic
    emotional_temperature: str = ""
    location: str = ""
    time_of_day: str = ""
    objects_of_significance: List[str] = Field(default_factory=list)
    character_blocking: str = ""
    continuity_dependencies: List[str] = Field(default_factory=list)
    visual_risk_notes: List[str] = Field(default_factory=list)
    visual_motif_priority: str = ""
    created_at: str = Field(default_factory=now_utc)


# ─── Shot ──────────────────────────────────────────────────
class Shot(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    scene_id: str
    shot_number: int = 0
    visual_priority: str = ""
    shot_type: str = ""  # wide, medium, close-up, extreme-close-up, over-shoulder, aerial, etc.
    camera_height: str = ""  # eye-level, low-angle, high-angle, overhead, dutch
    camera_behavior: str = ""  # static, slow-pan, track, dolly, handheld, crane
    subject_action: str = ""
    emotional_micro_state: str = ""
    light_description: str = ""
    secondary_objects: List[str] = Field(default_factory=list)  # max 3
    motion_constraints: str = ""
    negative_constraints: List[str] = Field(default_factory=list)
    duration_hint: float = 3.0
    generation_safe_wording: str = ""
    created_at: str = Field(default_factory=now_utc)


# ─── Prompt Variant ────────────────────────────────────────
class PromptVariant(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    shot_id: str
    scene_id: str
    model_target: str = "generic"  # generic, midjourney, dall-e, runway, stable-diffusion, kling, pika
    positive_prompt: str = ""
    negative_prompt: str = ""
    style_injection: str = ""
    aspect_ratio: str = "16:9"
    duration: float = 3.0
    created_at: str = Field(default_factory=now_utc)


# ─── Character Profile (from Qaivid 1.0 CharacterContext) ──
class CharacterProfileCreate(BaseModel):
    name: str
    role: str = ""
    description: str = ""
    appearance: str = ""
    age_range: str = ""
    wardrobe: str = ""
    emotional_range: str = ""
    reference_image_url: str = ""


class CharacterProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    name: str
    role: str = ""
    description: str = ""
    appearance: str = ""
    age_range: str = ""
    wardrobe: str = ""
    emotional_range: str = ""
    reference_image_url: str = ""
    look_variants: List[Dict[str, Any]] = Field(default_factory=list)
    scene_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)


# ─── Environment Profile (from Qaivid 1.0 LocationContext) ─
class EnvironmentProfileCreate(BaseModel):
    name: str
    description: str = ""
    time_of_day: str = ""
    mood: str = ""
    visual_details: str = ""
    architecture: str = ""
    reference_image_url: str = ""


class EnvironmentProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=gen_id)
    project_id: str
    name: str
    description: str = ""
    time_of_day: str = ""
    mood: str = ""
    visual_details: str = ""
    architecture: str = ""
    reference_image_url: str = ""
    scene_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)



# ─── Validation Result ─────────────────────────────────────
class ValidationIssue(BaseModel):
    severity: str = "warning"  # error, warning, info
    layer: str = ""  # context, scene, shot, prompt
    target_id: str = ""
    message: str = ""
    suggestion: str = ""


class ValidationResult(BaseModel):
    project_id: str
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    issues: List[Dict[str, str]] = Field(default_factory=list)
    validated_at: str = Field(default_factory=now_utc)
