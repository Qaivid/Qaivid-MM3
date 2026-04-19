"""
Context Engine Service — Direct OpenAI SDK
The ONE deep LLM call that produces the ContextPacket.
Uses OpenAI GPT directly via user's own API key.
"""
import json
import uuid
import logging
from typing import Dict, Any
from openai import AsyncOpenAI
from services.culture_packs import apply_culture_enrichment, get_metaphor_meanings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a world-class literary analyst, dramaturg, and cultural interpreter. Your job is to deeply interpret a piece of expressive text (song lyrics, poem, ghazal, qawwali, script, story, voiceover) and produce a structured analysis.

You must return ONLY valid JSON with this exact structure:
{
  "input_type": "song|poem|ghazal|qawwali|script|story|ad|voiceover",
  "narrative_mode": "realist|memory|symbolic|psychological|philosophical|performance_led|hybrid",
  "core_theme": "one sentence describing the central theme",
  "dramatic_premise": "one sentence dramatic premise",
  "narrative_spine": "2-3 sentence narrative summary",
  "speaker_model": {
    "identity": "who is speaking",
    "gender": "male|female|non_binary|unspecified",
    "age_range": "young|middle|elder|unspecified",
    "social_role": "their social position",
    "emotional_state": "their dominant emotional state",
    "relationship_to_addressee": "their relationship"
  },
  "addressee_model": {
    "identity": "who is being addressed",
    "relationship": "relationship to speaker"
  },
  "world_assumptions": {
    "geography": "where this is set",
    "era": "when this is set",
    "season": "seasonal context if any",
    "time_of_day": "time context if any",
    "social_context": "social world described",
    "economic_context": "economic reality if relevant",
    "architecture_style": "built environment",
    "domestic_setting": "home/indoor setting if relevant"
  },
  "emotional_arc": [
    {"phase": "opening", "emotion": "...", "intensity": "low|medium|high"},
    {"phase": "development", "emotion": "...", "intensity": "..."},
    {"phase": "climax", "emotion": "...", "intensity": "..."},
    {"phase": "resolution", "emotion": "...", "intensity": "..."}
  ],
  "line_meanings": [
    {
      "line_index": 0,
      "text": "original line text",
      "literal_meaning": "what it literally says",
      "implied_meaning": "what it actually means",
      "emotional_meaning": "what emotion it carries",
      "cultural_meaning": "cultural significance if any",
      "visual_suitability": "high|medium|low",
      "visualization_mode": "direct|indirect|symbolic|absorbed|performance_only"
    }
  ],
  "motif_map": {"motif_name": ["line indices where it appears"]},
  "entity_map": {"characters": [], "locations": [], "objects": [], "time_references": []},
  "restrictions": ["things to avoid in visualization"],
  "ambiguity_flags": [{"field": "field_name", "reason": "why ambiguous", "confidence": 0.0}],
  "confidence_scores": {"overall": 0.0, "cultural": 0.0, "emotional": 0.0, "speaker": 0.0, "narrative_mode": 0.0}
}

CRITICAL RULES:
- Treat metaphors as metaphors. Do NOT literalize symbolic language.
- Identify visualization_mode carefully.
- Be culturally precise. If South Asian content, use appropriate frameworks.
- Be honest about confidence. If unsure, flag it.
- Return ONLY the JSON. No markdown, no explanation."""


async def build_context_packet(
    project_id: str,
    cleaned_text: str,
    lines: list,
    detected_type: str,
    detected_language: str,
    culture_pack_id: str,
    user_settings: Dict[str, Any] = None,
    pre_enrichment_hints: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("OpenAI API key not configured. Go to Settings to add your key.")

    client = AsyncOpenAI(api_key=api_key)

    # Build prompt
    parts = [
        f"INPUT TYPE: {detected_type}",
        f"DETECTED LANGUAGE: {detected_language}",
        f"CULTURE PACK: {culture_pack_id}",
    ]
    if user_settings:
        for k, v in user_settings.items():
            if v and v != "auto":
                parts.append(f"USER SETTING - {k}: {v}")
    parts.append(f"\nTEXT TO INTERPRET:\n{cleaned_text}")

    metaphors = get_metaphor_meanings(cleaned_text, culture_pack_id)
    if metaphors:
        parts.append("\nKNOWN CULTURAL METAPHORS:")
        for m in metaphors:
            parts.append(f"  - '{m['trigger']}' means: {m['cultural_meaning']}")

    line_texts = [ln.get("text", "") for ln in lines] if lines else cleaned_text.split('\n')
    line_list = "\n".join(f"[{i}] {t}" for i, t in enumerate(line_texts) if t.strip())
    parts.append(f"\nINDEXED LINES:\n{line_list}")
    if pre_enrichment_hints:
        parts.append(f"\n{pre_enrichment_hints}")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    context_data = json.loads(content)

    # Apply culture pack enrichment
    context_data = apply_culture_enrichment(context_data, culture_pack_id)

    context_data["id"] = str(uuid.uuid4())
    context_data["project_id"] = project_id
    context_data["locked_assumptions"] = {}
    if "language_info" not in context_data:
        context_data["language_info"] = {"primary": detected_language, "script": "", "dialect": ""}

    from models import now_utc
    context_data["created_at"] = now_utc()
    context_data["updated_at"] = now_utc()

    return context_data
