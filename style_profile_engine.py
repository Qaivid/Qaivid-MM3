"""
Qaivid Style Profile Engine

Reads all available inputs (song text, genre, audio analytics, culture pack)
and uses an LLM — backed entirely by the StyleProfileRegistry — to suggest
2-3 style profiles for the user to choose from.

DESIGN PRINCIPLE:
- This engine contains ZERO hardcoded style knowledge.
- All style definitions live in style_profile_registry.py.
- This engine only knows how to read the registry and call the LLM.
- Adding a new style = add to the registry. No changes here needed.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from style_profile_registry import StyleProfileRegistry

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.5
_MAX_TOKENS = 1500


class StyleProfileEngine:
    """
    Given the inputs available after audio analysis, suggest 2-3 style profiles
    that would make a great music video for this particular song/content.

    Returns a list of resolved style_profile dicts from the registry.
    """

    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)

    async def suggest(
        self,
        text: str,
        genre: str,
        audio_analytics: Optional[Dict[str, Any]] = None,
        culture_pack_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Suggest 2-3 style profiles for this content.

        Returns a list of style_profile dicts, each fully resolved from the registry.
        Each dict has keys: production, cinematic, preset, justification.
        """
        audio_analytics = audio_analytics or {}
        culture_pack_id = culture_pack_id or "none"

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, genre, audio_analytics, culture_pack_id)

        try:
            raw = await self._call_model(system_prompt, user_prompt)
            suggestions_raw = self._safe_parse_json(raw)
            return self._resolve_suggestions(suggestions_raw)
        except Exception:
            logger.exception("StyleProfileEngine.suggest failed — returning defaults")
            return [self._default_suggestion()]

    def _build_system_prompt(self) -> str:
        registry_summary = StyleProfileRegistry.registry_summary_for_llm()

        return f"""
You are a world-class music video creative director and visual stylist.

Your job is to analyse a song (or other expressive content) and recommend
2-3 distinct, compelling visual style directions for its music video.

You have access to a fixed registry of Production Styles and Cinematic Styles.
You must only use IDs from this registry — do not invent new IDs.

{registry_summary}

For each recommendation you must:
1. Choose one production_style_id from the PRODUCTION STYLES list above.
2. Choose one cinematic_style_id from the CINEMATIC STYLES list above.
3. Ensure the pairing is creatively coherent.
4. Write a short justification (2-3 sentences) explaining why this combination
   fits THIS specific song — reference the lyrics, genre, culture, and audio feel.

SINGER-GENDER GUIDANCE (only applies when "Singer gender" is provided in the input):
- If singer is FEMALE → prefer profiles that center an intimate female protagonist
  (e.g. narrative or split_narrative_performance with cinematic_natural / soft_poetic).
- If singer is MALE → prefer profiles that center a male protagonist
  (e.g. narrative or performance with noir_dramatic / cinematic_natural).
- If singer is MIXED / DUET → suggest split or conversational framings
  (e.g. split_narrative_performance) that can carry two protagonists.
- Always reflect this choice in the justification.

Return ONLY valid JSON in this exact shape:
{{
  "suggestions": [
    {{
      "production_style_id": "string",
      "cinematic_style_id": "string",
      "justification": "string"
    }},
    ...
  ]
}}

Rules:
- Suggest 2-3 options. Never fewer than 2.
- Make the options genuinely different from each other — give the user real choice.
- The first suggestion should be the strongest/most fitting recommendation.
- Justification must reference the actual song content, not be generic.
- Return ONLY valid JSON. No markdown fences. No extra text.
""".strip()

    def _build_user_prompt(
        self,
        text: str,
        genre: str,
        audio_analytics: Dict[str, Any],
        culture_pack_id: str,
    ) -> str:
        bpm = audio_analytics.get("bpm") or "unknown"
        energy = audio_analytics.get("avg_energy") or audio_analytics.get("energy") or "unknown"
        energy_profile = audio_analytics.get("energy_profile") or "unknown"
        duration = audio_analytics.get("duration_seconds") or "unknown"
        brightness = audio_analytics.get("brightness_profile") or "unknown"

        vocal_gender = str(audio_analytics.get("vocal_gender") or "").strip().lower()
        gender_label_map = {"male": "Male", "female": "Female", "mixed": "Mixed / Duet"}
        gender_line = (
            f"\nSinger gender: {gender_label_map[vocal_gender]}"
            if vocal_gender in gender_label_map else ""
        )

        lyrics_excerpt = text[:800].strip() if text else "(no lyrics provided)"

        return f"""
SONG / CONTENT:
Genre: {genre}
Culture Pack: {culture_pack_id}
Duration: {duration}s
BPM: {bpm}
Energy profile: {energy_profile}
Avg energy: {energy}
Brightness: {brightness}{gender_line}

LYRICS (excerpt):
{lyrics_excerpt}

Suggest 2-3 style profile combinations that would make a great music video for this content.
""".strip()

    async def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=_MODEL,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _safe_parse_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("StyleProfileEngine: JSON parse failed, returning empty")
            return {"suggestions": []}

    def _resolve_suggestions(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_suggestions = raw.get("suggestions") or []
        if not isinstance(raw_suggestions, list):
            raw_suggestions = []

        resolved = []
        for item in raw_suggestions[:3]:
            if not isinstance(item, dict):
                continue
            prod_id = str(item.get("production_style_id") or "").strip()
            cin_id = str(item.get("cinematic_style_id") or "").strip()
            justification = str(item.get("justification") or "").strip()

            prod = StyleProfileRegistry.get_production_style(prod_id)
            cin = StyleProfileRegistry.get_cinematic_style(cin_id)

            if not prod or not cin:
                logger.warning(
                    "StyleProfileEngine: unknown style IDs prod=%s cin=%s — skipping",
                    prod_id, cin_id,
                )
                continue

            profile = StyleProfileRegistry.build_style_profile(prod_id, cin_id)
            profile["justification"] = justification
            resolved.append(profile)

        if not resolved:
            resolved.append(self._default_suggestion())

        if len(resolved) < 2:
            resolved.append(self._default_suggestion())

        return resolved

    def _default_suggestion(self) -> Dict[str, Any]:
        profile = StyleProfileRegistry.default_style_profile()
        profile["justification"] = (
            "A versatile narrative-performance blend with naturalistic cinematography — "
            "suitable for most song types and easily refined once the context is analysed."
        )
        return profile
