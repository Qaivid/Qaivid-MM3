"""
Creative Brief Generator — Direct OpenAI SDK
"""
import json
import uuid
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from models import now_utc

logger = logging.getLogger(__name__)


async def generate_creative_brief(
    project_id: str,
    context_packet: Dict[str, Any],
    project: Dict[str, Any],
    vibe_preset: Optional[str] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("OpenAI API key not configured. Go to Settings to add your key.")

    client = AsyncOpenAI(api_key=api_key)
    cp = context_packet

    vibe_rules = ""
    if vibe_preset:
        from services.vibe_presets import get_vibe_preset
        vp = get_vibe_preset(vibe_preset)
        if vp:
            vibe_rules = f"\n\nMANDATORY VISUAL STYLE — Vibe: \"{vp['label']}\"\n{vp['brief_direction']}"

    prompt = f"""You are a world-class creative director. Create a detailed Creative Brief from this structured interpretation.

PROJECT: {project.get('name', 'Untitled')}
CONTENT TYPE: {cp.get('input_type', 'song')}
NARRATIVE MODE: {cp.get('narrative_mode', 'realist')}
CORE THEME: {cp.get('core_theme', '')}
DRAMATIC PREMISE: {cp.get('dramatic_premise', '')}
NARRATIVE SPINE: {cp.get('narrative_spine', '')}
SPEAKER: {json.dumps(cp.get('speaker_model', {}), ensure_ascii=False)}
ADDRESSEE: {json.dumps(cp.get('addressee_model', {}), ensure_ascii=False)}
WORLD: {json.dumps(cp.get('world_assumptions', {}), ensure_ascii=False)}
CULTURE: {json.dumps(cp.get('cultural_setting', {}), ensure_ascii=False)}
EMOTIONAL ARC: {json.dumps(cp.get('emotional_arc', []), ensure_ascii=False)}
MOTIFS: {json.dumps(cp.get('motif_map', {}), ensure_ascii=False)}
RESTRICTIONS: {json.dumps(cp.get('restrictions', []), ensure_ascii=False)}
{vibe_rules}

Return ONLY valid JSON:
{{
  "title": "Punchy project title",
  "tagline": "One evocative sentence",
  "narrative_arc": "2-3 sentences: overall story arc",
  "emotional_journey": "What audience feels at start, middle, end",
  "visual_aesthetic": {{
    "style": "e.g. Poetic realism with warm amber tones",
    "color_palette": ["4-5 specific colors"],
    "lighting_mood": "Primary lighting approach",
    "cinematography_style": "Camera work description"
  }},
  "characters": [
    {{
      "name": "Name", "role": "protagonist|love_interest|supporting",
      "age": "e.g. mid-20s",
      "physical_description": "Detailed: hair, skin, features, build. Min 40 words.",
      "wardrobe": "Specific clothing",
      "emotional_arc": "How emotion changes through video"
    }}
  ],
  "locations": [
    {{
      "name": "Location name", "description": "What and why it matters",
      "time_of_day": "e.g. golden hour", "mood": "Atmosphere",
      "visual_details": "Architecture, textures, props. Min 30 words."
    }}
  ],
  "visual_motifs": ["3-5 recurring visual symbols"],
  "production_notes": "Key directorial guidance"
}}

Create 2-4 characters and 3-6 locations from ContextPacket data."""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cinematic creative director. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )

    brief = json.loads(response.choices[0].message.content)
    brief["id"] = str(uuid.uuid4())
    brief["project_id"] = project_id
    brief["vibe_preset"] = vibe_preset
    brief["created_at"] = now_utc()
    return brief
