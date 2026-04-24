"""Materializer Engine v2 (MetaMind 3.1 — master spec)

ROLE:
  Convert abstract scene directions from the locked Creative Brief into
  CONSISTENT visual identity anchors for characters and locations, without
  locking exact visuals.

Reads from Project Brain:
  1. creative_briefs     — scenes, character_presence, environment_type, etc.
  2. context_packet      — cultural_grounding, must_preserve, creative_freedom
  3. narrative_packet    — presence_strategy, expression_channels (light use)
  4. style_packet        — cinematic_style, color_psychology, texture (light)
  5. project_settings    — variation preference, output constraints (optional)

Writes to Project Brain:
  materializer_packet → {character_bible, location_bible, motif_anchors,
                         continuity_rules}
  Also mirrors:
  character_bible, location_bible as top-level namespaces (backward compat)

Always calls the three legacy materializers (character/location/motif) after
writing to brain so the DB tables stay populated for the UI.

The LLM is told to define IDENTITY RULES, not exact visuals — per spec:
  - NO exact face details
  - NO exact outfits
  - NO scene compositions or camera directions
  - NO collapsed variation (keep multiple visual realizations possible)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON schema for the LLM to fill in
# ---------------------------------------------------------------------------

_MATERIALIZER_SCHEMA = {
    "character_bible": {
        "characters": [
            {
                "character_id": "<string — stable id slug, e.g. 'primary_speaker'>",
                "identity_seed": "<one sentence — the non-negotiable core of who this person is>",
                "archetype": "<e.g. 'young rural woman in mourning'>",
                "age_range": "<e.g. '22-30'>",
                "physical_range": "<broad physical description range — no exact features>",
                "wardrobe_logic": "<wardrobe RULES, not specific outfit — e.g. 'traditional attire, earthy tones'>",
                "styling_logic": "<grooming and styling rules — e.g. 'minimal, natural'>",
                "emotional_baseline": "<resting emotional expression — e.g. 'restrained grief'>",
                "behavior_traits": "<e.g. 'slow, contained, deliberate movement'>",
                "variation_rules": {
                    "allowed": ["<list of things that may change between scenes>"],
                    "restricted": ["<list of things that must never change>"]
                }
            }
        ]
    },
    "location_bible": {
        "locations": [
            {
                "location_id": "<string — stable id slug, e.g. 'rural_punjab_world'>",
                "world": "<e.g. 'Rural Punjab, Pakistan, 1980s'>",
                "architecture_style": "<e.g. 'mud-brick, low courtyard walls'>",
                "structural_elements": ["<e.g. 'narrow lanes'>", "<doorways>", "<courtyards>"],
                "textures": ["<e.g. 'raw earth'>", "<weathered plaster>"],
                "environment_density": "<empty | sparse | moderate | busy>",
                "color_palette": ["<e.g. 'warm ochre'>", "<dusty beige>", "<faded terracotta>"],
                "lighting_tendency": "<e.g. 'soft diffused morning light, harsh midday shadow'>",
                "environment_rules": ["<what belongs>", "<what is forbidden>"]
            }
        ]
    },
    "motif_anchors": [
        {
            "motif_id": "<string slug>",
            "general_form": "<how the motif appears visually — general, not exact>",
            "variation_range": "<how much variation is allowed across scenes>"
        }
    ],
    "continuity_rules": {
        "character_consistency": True,
        "world_consistency": True,
        "motif_consistency": True,
        "tone_consistency": True
    }
}


def _build_system_prompt() -> str:
    return (
        "You are the Materializer for a cinematic music video pipeline.\n"
        "Your role is to define IDENTITY RULES and WORLD ANCHORS — not exact visuals.\n\n"
        "CORE PRINCIPLE: define what makes a character recognizable and a world believable, "
        "while preserving creative variation across scenes.\n\n"
        "WHAT YOU MUST NOT DO:\n"
        "- Do NOT define exact face details\n"
        "- Do NOT define exact outfits\n"
        "- Do NOT define exact props\n"
        "- Do NOT define scene compositions or camera directions\n"
        "- Do NOT generate image prompts\n"
        "- Do NOT collapse variation — keep multiple visual realizations possible\n\n"
        "WHAT YOU MUST DO:\n"
        "- Define identity_seed: the non-negotiable core of who each character is\n"
        "- Define physical_range: broad parameters, NOT a face description\n"
        "- Define wardrobe_logic: rules and cultural constraints, NOT a specific outfit\n"
        "- Define location world anchors: architecture, texture, color range, lighting tendency\n"
        "- Define what can vary vs. what must remain stable across scenes\n\n"
        "Respond ONLY with valid JSON matching the provided schema. No prose."
    )


def _build_user_prompt(
    brief_packet: dict,
    context_packet: dict,
    narrative_packet: dict,
    style_packet: dict,
    project_settings: dict,
    schema: dict,
) -> str:
    # ── Creative Brief: the primary driver ──
    brief_scenes = list(brief_packet.get("scenes") or [])
    brief_summary_lines = []
    for i, sc in enumerate(brief_scenes[:8]):
        if not isinstance(sc, dict):
            continue
        line_parts = [f"Scene {i+1}:"]
        for field in ("subject_focus", "character_presence", "character_identity_hint",
                      "environment_type", "continuity_hooks", "motif_usage",
                      "chosen_direction", "variation_anchor"):
            val = sc.get(field)
            if val and str(val).strip():
                line_parts.append(f"  {field}: {val}")
        if len(line_parts) > 1:
            brief_summary_lines.append("\n".join(line_parts))

    # ── Context: cultural grounding ──
    location_dna = context_packet.get("location_dna") or ""
    speaker = context_packet.get("speaker") or {}
    world_assumptions = context_packet.get("world_assumptions") or {}
    must_preserve = context_packet.get("must_preserve") or []
    creative_freedom = context_packet.get("creative_freedom") or {}
    motif_map = context_packet.get("motif_map") or {}

    # ── Narrative: light presence signals ──
    presence_strategy = narrative_packet.get("presence_strategy") or ""
    expression_channels = narrative_packet.get("expression_channels") or []
    continuity_rules_narr = narrative_packet.get("continuity_rules") or ""

    # ── Style: visual tone direction ──
    cinematic_style = style_packet.get("preset") or style_packet.get("cinematic_style") or ""
    color_psychology = (style_packet.get("cinematic") or {}).get("color_psychology") or ""
    texture_profile = (style_packet.get("cinematic") or {}).get("texture_profile") or ""
    realism_level = (style_packet.get("cinematic") or {}).get("realism_level") or ""

    # ── Project settings ──
    variation_pref = project_settings.get("variation_preference") or "moderate"
    output_constraints = project_settings.get("output_constraints") or ""

    sections: list[str] = []

    sections.append("=== CREATIVE BRIEF (primary input) ===")
    if brief_summary_lines:
        sections.append("\n\n".join(brief_summary_lines))
    else:
        sections.append("(no brief scenes available — rely on context/narrative)")

    sections.append("=== CULTURAL CONTEXT ===")
    if location_dna:
        sections.append(f"Location DNA: {location_dna}")
    if world_assumptions:
        sections.append(f"World assumptions: {json.dumps(world_assumptions)}")
    if speaker:
        sections.append(f"Primary speaker: {json.dumps(speaker)}")
    if must_preserve:
        sections.append(f"Must preserve: {must_preserve}")
    if creative_freedom:
        sections.append(f"Creative freedom: {json.dumps(creative_freedom)}")

    sections.append("=== MOTIFS ===")
    if motif_map:
        for mname, mpayload in list(motif_map.items())[:6]:
            sections.append(f"  {mname}: {mpayload}")

    sections.append("=== NARRATIVE SIGNALS ===")
    if presence_strategy:
        sections.append(f"Presence strategy: {presence_strategy}")
    if expression_channels:
        sections.append(f"Expression channels: {expression_channels}")
    if continuity_rules_narr:
        sections.append(f"Narrative continuity: {continuity_rules_narr}")

    sections.append("=== VISUAL STYLE ===")
    if cinematic_style:
        sections.append(f"Cinematic style: {cinematic_style}")
    if color_psychology:
        sections.append(f"Color psychology: {color_psychology}")
    if texture_profile:
        sections.append(f"Texture profile: {texture_profile}")
    if realism_level:
        sections.append(f"Realism level: {realism_level}")
    if variation_pref:
        sections.append(f"Variation preference: {variation_pref}")
    if output_constraints:
        sections.append(f"Output constraints: {output_constraints}")

    sections.append("=== OUTPUT SCHEMA ===")
    sections.append(
        "Fill in the following JSON completely. Use the scene data above to "
        "derive character identities and locations. Do NOT leave fields empty. "
        "Include ALL characters mentioned across scenes and ALL distinct environments."
    )
    sections.append(json.dumps(schema, indent=2))

    return "\n\n".join(sections)


async def _call_llm(user_prompt: str) -> dict:
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.35,
        max_tokens=2500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def _validate_and_fix(data: Any) -> dict:
    """Ensure the LLM output has the four required top-level keys."""
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("character_bible"), dict):
        data["character_bible"] = {"characters": []}
    if not isinstance(data["character_bible"].get("characters"), list):
        data["character_bible"]["characters"] = []
    if not isinstance(data.get("location_bible"), dict):
        data["location_bible"] = {"locations": []}
    if not isinstance(data["location_bible"].get("locations"), list):
        data["location_bible"]["locations"] = []
    if not isinstance(data.get("motif_anchors"), list):
        data["motif_anchors"] = []
    if not isinstance(data.get("continuity_rules"), dict):
        data["continuity_rules"] = {
            "character_consistency": True,
            "world_consistency": True,
            "motif_consistency": True,
            "tone_consistency": True,
        }
    # Ensure each character has variation_rules
    for ch in data["character_bible"]["characters"]:
        if not isinstance(ch, dict):
            continue
        if "character_id" not in ch or not ch["character_id"]:
            ch["character_id"] = "char_" + uuid.uuid4().hex[:6]
        if not isinstance(ch.get("variation_rules"), dict):
            ch["variation_rules"] = {"allowed": [], "restricted": []}
    # Ensure each location has required lists
    for loc in data["location_bible"]["locations"]:
        if not isinstance(loc, dict):
            continue
        if "location_id" not in loc or not loc["location_id"]:
            loc["location_id"] = "loc_" + uuid.uuid4().hex[:6]
        for list_field in ("structural_elements", "textures", "color_palette", "environment_rules"):
            if not isinstance(loc.get(list_field), list):
                loc[list_field] = []
    return data


def _build_fallback_from_db(project_id: str, context_packet: dict) -> dict:
    """Synthesize a minimal materializer_packet from legacy DB rows.

    Called when the LLM fails. The three legacy materializers are expected to
    have already run (or will run after this returns) so DB tables are populated.
    """
    import os
    import psycopg
    from psycopg.rows import dict_row

    try:
        conn_str = os.environ["DATABASE_URL"]
        with psycopg.connect(conn_str, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, entity_type, appearance, age_range, gender, "
                "ethnicity, complexion, wardrobe, grooming, emotional_notes, role "
                "FROM characters WHERE project_id=%s ORDER BY id LIMIT 10",
                (project_id,),
            )
            db_chars = cur.fetchall() or []
            cur.execute(
                "SELECT name, entity_type, description, geography, architecture_style, "
                "weather_or_atmosphere, cultural_dna, visual_details, cultural_notes "
                "FROM locations WHERE project_id=%s ORDER BY id LIMIT 10",
                (project_id,),
            )
            db_locs = cur.fetchall() or []
            cur.execute(
                "SELECT name, motif_type, significance, visual_form "
                "FROM motifs WHERE project_id=%s ORDER BY id LIMIT 10",
                (project_id,),
            )
            db_motifs = cur.fetchall() or []
    except Exception:
        logger.exception("Materializer v2 fallback: DB query failed for project=%s", project_id)
        db_chars, db_locs, db_motifs = [], [], []

    characters = []
    for ch in db_chars:
        characters.append({
            "character_id": ch["name"].lower().replace(" ", "_") if ch.get("name") else "char",
            "identity_seed": ch.get("appearance") or ch.get("role") or ch.get("name") or "Primary character",
            "archetype": ch.get("role") or "",
            "age_range": ch.get("age_range") or "",
            "physical_range": " ".join(filter(None, [
                ch.get("gender"), ch.get("ethnicity"), ch.get("complexion")
            ])) or "",
            "wardrobe_logic": ch.get("wardrobe") or "",
            "styling_logic": ch.get("grooming") or "",
            "emotional_baseline": ch.get("emotional_notes") or "",
            "behavior_traits": "",
            "variation_rules": {
                "allowed": ["face expression", "clothing color variations", "minor accessories"],
                "restricted": ["cultural identity", "emotional baseline", "gender presentation"],
            },
        })

    locations = []
    for loc in db_locs:
        loc_palette = []
        if loc.get("cultural_dna"):
            loc_palette.append(loc["cultural_dna"])
        locations.append({
            "location_id": (loc.get("name") or "loc").lower().replace(" ", "_"),
            "world": loc.get("description") or loc.get("name") or "",
            "architecture_style": loc.get("architecture_style") or "",
            "structural_elements": [loc.get("visual_details") or ""] if loc.get("visual_details") else [],
            "textures": [],
            "environment_density": "moderate",
            "color_palette": loc_palette,
            "lighting_tendency": loc.get("weather_or_atmosphere") or "",
            "environment_rules": [loc.get("cultural_notes") or ""] if loc.get("cultural_notes") else [],
        })

    motif_anchors = []
    for m in db_motifs:
        motif_anchors.append({
            "motif_id": (m.get("name") or "motif").lower().replace(" ", "_"),
            "general_form": m.get("visual_form") or m.get("significance") or m.get("name") or "",
            "variation_range": "moderate variation allowed",
        })

    return {
        "character_bible": {"characters": characters},
        "location_bible": {"locations": locations},
        "motif_anchors": motif_anchors,
        "continuity_rules": {
            "character_consistency": True,
            "world_consistency": True,
            "motif_consistency": True,
            "tone_consistency": True,
        },
        "_source": "fallback_from_db",
    }


async def run_materializer(
    project_id: str,
    brain: Any,
    vocal_gender: Optional[str] = None,
) -> dict:
    """Primary async entry point — LLM identity bible only.

    Reads five brain namespaces (creative_briefs, context_packet,
    narrative_packet, style_packet, project_settings), calls the LLM,
    validates the result, and writes three brain namespaces:
    - brain.materializer_packet  (v2 spec schema)
    - brain.character_bible      (backward-compat mirror of character_bible sub-key)
    - brain.location_bible       (backward-compat mirror of location_bible sub-key)

    Does NOT call legacy DB materializers — the caller (_stage_refs_job) is
    responsible for invoking those AFTER brain is saved, in the correct order.

    Returns the materializer_packet dict.
    """
    # ── Read all brain namespaces ──────────────────────────────────────────
    brief_packet     = dict(brain.read("creative_briefs") or {})
    context_packet   = dict(brain.read("context_packet") or {})
    narrative_packet = dict(brain.read("narrative_packet") or {})
    style_packet     = dict(brain.read("style_packet") or {})
    project_settings = dict(brain.read("project_settings") or {})

    # ── Call LLM for the rich identity bible ─────────────────────────────
    mat_packet: Optional[dict] = None
    try:
        user_prompt = _build_user_prompt(
            brief_packet=brief_packet,
            context_packet=context_packet,
            narrative_packet=narrative_packet,
            style_packet=style_packet,
            project_settings=project_settings,
            schema=_MATERIALIZER_SCHEMA,
        )
        llm_result = await _call_llm(user_prompt)
        mat_packet = _validate_and_fix(llm_result)
        logger.info(
            "Materializer v2: LLM produced %d character(s), %d location(s), %d motif(s) for project=%s",
            len(mat_packet["character_bible"]["characters"]),
            len(mat_packet["location_bible"]["locations"]),
            len(mat_packet.get("motif_anchors") or []),
            project_id,
        )
    except Exception:
        logger.exception(
            "Materializer v2: LLM call failed — falling back to DB synthesis for project=%s",
            project_id,
        )
        mat_packet = _build_fallback_from_db(project_id, context_packet)

    # ── Write to brain ────────────────────────────────────────────────────
    brain.write("materializer_packet", mat_packet)
    brain.write("character_bible", mat_packet.get("character_bible") or {})
    brain.write("location_bible", mat_packet.get("location_bible") or {})
    return mat_packet


def run_materializer_sync(
    project_id: str,
    brain: Any,
    vocal_gender: Optional[str] = None,
) -> dict:
    """Synchronous wrapper around run_materializer (for use in pipeline threads)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            run_materializer(project_id, brain, vocal_gender=vocal_gender)
        )
    finally:
        try:
            loop.close()
        except Exception:
            pass
