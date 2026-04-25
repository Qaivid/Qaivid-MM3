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
        context_packet: Optional[Dict[str, Any]] = None,
        narrative_packet: Optional[Dict[str, Any]] = None,
        emotional_mode_packet: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Suggest 2-3 style profiles for this content.

        Reads from Project Brain (per master spec — Style is Stage 4):
          • context_packet  (Stage 2) — meaning, world, speaker, motivation
          • narrative_packet (Stage 3) — storytelling mode, perspective, arc, motifs

        These ground the visual recommendations in the locked story decisions
        instead of guessing from lyrics alone. They are optional — if not
        provided the engine falls back to lyrics + audio only (legacy mode).

        Returns a list of style_profile dicts, each fully resolved from the registry.
        Each dict has keys: production, cinematic, preset, justification.
        """
        audio_analytics = audio_analytics or {}
        culture_pack_id = culture_pack_id or "none"
        context_packet = context_packet or {}
        narrative_packet = narrative_packet or {}
        emotional_mode_packet = emotional_mode_packet or {}

        mode_constraints = self._extract_mode_constraints(emotional_mode_packet)
        system_prompt = self._build_system_prompt(mode_constraints=mode_constraints)
        user_prompt = self._build_user_prompt(
            text, genre, audio_analytics, culture_pack_id,
            context_packet, narrative_packet, emotional_mode_packet,
        )

        try:
            raw = await self._call_model(system_prompt, user_prompt)
            suggestions_raw = self._safe_parse_json(raw)
            resolved = self._resolve_suggestions(suggestions_raw, mode_constraints=mode_constraints)
            return [self._apply_mode_merge(s, emotional_mode_packet) for s in resolved]
        except Exception:
            logger.exception("StyleProfileEngine.suggest failed — returning mode-constrained defaults")
            # Run fallback through the same avoid/incompatible post-filter as normal suggestions
            # so mode constraints (avoid/incompatible) hold even on model failure.
            # Use a raw-format dict so _resolve_suggestions can apply replacements.
            synthetic_raw = {
                "suggestions": [{
                    "production_style_id": "split_narrative_performance",
                    "cinematic_style_id":  "cinematic_realism",
                    "justification":       (
                        "Versatile narrative blend with naturalistic cinematography — "
                        "default fallback when style analysis is unavailable."
                    ),
                }]
            }
            resolved_fallback = self._resolve_suggestions(synthetic_raw, mode_constraints=mode_constraints)
            fallback = self._apply_mode_merge(resolved_fallback[0], emotional_mode_packet)
            return [fallback]

    def _extract_mode_constraints(self, emp: Dict[str, Any]) -> Dict[str, Any]:
        """Extract filtering constraints from an emotional_mode_packet (or empty dict)."""
        if not emp:
            return {}
        return {
            "mode_label":             emp.get("mode_label", ""),
            "cinematic_modifier":     emp.get("cinematic_modifier", ""),
            "preferred_production":   (emp.get("production_affinity") or {}).get("preferred") or [],
            "avoid_production":       (emp.get("production_affinity") or {}).get("avoid") or [],
            "incompatible_cinematic": emp.get("incompatible_cinematic_styles") or [],
            "compatible_cinematic":   emp.get("compatible_cinematic_styles") or [],
        }

    def _build_system_prompt(self, mode_constraints: Optional[Dict[str, Any]] = None) -> str:
        registry_summary = StyleProfileRegistry.registry_summary_for_llm()

        mode_section = ""
        if mode_constraints and mode_constraints.get("mode_label"):
            preferred   = mode_constraints.get("preferred_production") or []
            avoid       = mode_constraints.get("avoid_production") or []
            incompat    = mode_constraints.get("incompatible_cinematic") or []
            compatible  = mode_constraints.get("compatible_cinematic") or []
            modifier    = mode_constraints.get("cinematic_modifier") or ""
            mode_section = f"""
EMOTIONAL MODE CONSTRAINTS (locked by Stage 2b — MUST be respected):
  Emotional mode:  {mode_constraints['mode_label']}
  Aesthetic feel:  {modifier}
  Production style — MUST choose from: {', '.join(preferred) if preferred else '(any)'}
  Production style — MUST NOT use:    {', '.join(avoid) if avoid else '(none)'}
  Cinematic style  — PREFER from:     {', '.join(compatible) if compatible else '(any)'}
  Cinematic style  — MUST NOT use:    {', '.join(incompat) if incompat else '(none)'}
These constraints are absolute. Violating them means ignoring the locked emotional truth.
""".strip()

        return f"""
You are a world-class music video creative director and visual stylist.

Your job is to analyse a song (or other expressive content) and recommend
2-3 distinct, compelling visual style directions for its music video.

You have access to a fixed registry of Production Styles and Cinematic Styles.
You must only use IDs from this registry — do not invent new IDs.

{registry_summary}

{mode_section}

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
        context_packet: Optional[Dict[str, Any]] = None,
        narrative_packet: Optional[Dict[str, Any]] = None,
        emotional_mode_packet: Optional[Dict[str, Any]] = None,
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

        context_block   = self._build_context_block(context_packet or {})
        narrative_block = self._build_narrative_block(narrative_packet or {})
        mode_block      = self._build_mode_block(emotional_mode_packet or {})

        return f"""
SONG / CONTENT:
Genre: {genre}
Culture Pack: {culture_pack_id}
Duration: {duration}s
BPM: {bpm}
Energy profile: {energy_profile}
Avg energy: {energy}
Brightness: {brightness}{gender_line}
{context_block}{narrative_block}{mode_block}
LYRICS (excerpt):
{lyrics_excerpt}

Suggest 2-3 style profile combinations that would make a great music video for this content.
When STORY CONTEXT, NARRATIVE INTELLIGENCE, and EMOTIONAL MODE are present above, your
suggestions MUST honour them — choose visual styles that serve the locked meaning, world,
speaker, emotional register, and storytelling strategy. Reference these in your justifications.
""".strip()

    def _build_context_block(self, ctx: Dict[str, Any]) -> str:
        """Compact STORY CONTEXT block from the locked context_packet (Stage 2).

        Returns "" if the packet is empty so legacy / pre-brain projects keep
        working unchanged.
        """
        if not isinstance(ctx, dict) or not ctx:
            return ""

        lines: List[str] = []

        meaning = ctx.get("meaning") or ctx.get("theme")
        if isinstance(meaning, dict):
            meaning = meaning.get("summary") or meaning.get("statement") or ""
        if meaning:
            lines.append(f"  Meaning / theme:    {str(meaning)[:240].strip()}")

        speaker = ctx.get("speaker") if isinstance(ctx.get("speaker"), dict) else {}
        if speaker:
            sp_bits = []
            for key in ("name", "identity", "gender", "age", "archetype"):
                val = speaker.get(key)
                if val:
                    sp_bits.append(f"{key}={val}")
            if sp_bits:
                lines.append(f"  Speaker:            {', '.join(sp_bits)}")

        loc = ctx.get("location_dna") or ctx.get("location")
        if isinstance(loc, dict):
            loc = loc.get("summary") or loc.get("name") or ""
        if loc:
            lines.append(f"  Location DNA:       {str(loc)[:160].strip()}")

        era = ctx.get("era")
        if era:
            lines.append(f"  Era:                {str(era)[:80].strip()}")

        world = ctx.get("world_assumptions") if isinstance(ctx.get("world_assumptions"), dict) else {}
        if world:
            world_bits = []
            for key in ("geography", "season", "characteristic_time",
                        "social_context", "architecture_style",
                        "characteristic_setting"):
                val = world.get(key)
                if val:
                    world_bits.append(f"{key}={val}")
            if world_bits:
                lines.append(f"  World:              {', '.join(world_bits)}")

        motivation = ctx.get("motivation") if isinstance(ctx.get("motivation"), dict) else {}
        if motivation:
            mot_bits = []
            for key in ("inciting_cause", "underlying_desire", "stakes", "obstacle"):
                val = motivation.get(key)
                if val:
                    mot_bits.append(f"{key}: {str(val)[:80]}")
            if mot_bits:
                lines.append(f"  Motivation:         {' | '.join(mot_bits)}")

        if not lines:
            return ""

        return "\nSTORY CONTEXT (locked by Stage 2 — visuals must serve this meaning):\n" + "\n".join(lines) + "\n"

    def _build_narrative_block(self, narr: Dict[str, Any]) -> str:
        """Compact NARRATIVE INTELLIGENCE block from narrative_packet (Stage 3).

        Reuses narrative_engine.format_for_prompt for a stable, shared rendering.
        Returns "" if the packet is empty.
        """
        if not isinstance(narr, dict) or not narr:
            return ""
        try:
            from narrative_engine import format_for_prompt
            block = format_for_prompt(narr)
            if not block:
                return ""
            return "\n" + block + "\n"
        except Exception:
            logger.exception("StyleProfileEngine: failed to format narrative_packet")
            return ""

    def _build_mode_block(self, emp: Dict[str, Any]) -> str:
        """Compact EMOTIONAL MODE block from emotional_mode_packet (Stage 2b).
        Returns "" if the packet is empty.
        """
        if not isinstance(emp, dict) or not emp:
            return ""
        label    = emp.get("mode_label") or emp.get("primary_mode") or ""
        modifier = emp.get("cinematic_modifier") or ""
        if not label:
            return ""
        lines = [f"  Mode:             {label}"]
        if modifier:
            lines.append(f"  Cinematic feel:   {modifier}")
        prod_aff = emp.get("production_affinity") or {}
        if prod_aff.get("preferred"):
            lines.append(f"  Preferred prod:   {', '.join(prod_aff['preferred'])}")
        if prod_aff.get("avoid"):
            lines.append(f"  Avoid prod:       {', '.join(prod_aff['avoid'])}")
        compat = emp.get("compatible_cinematic_styles") or []
        if compat:
            lines.append(f"  Prefer cinematic: {', '.join(compat)}")
        incompat = emp.get("incompatible_cinematic_styles") or []
        if incompat:
            lines.append(f"  Avoid cinematic:  {', '.join(incompat)}")
        return "\nEMOTIONAL MODE (locked by Stage 2b — style must serve this register):\n" + "\n".join(lines) + "\n"

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

    def _resolve_suggestions(
        self,
        raw: Dict[str, Any],
        mode_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raw_suggestions = raw.get("suggestions") or []
        if not isinstance(raw_suggestions, list):
            raw_suggestions = []

        avoid_prod      = set((mode_constraints or {}).get("avoid_production") or [])
        incompat_cin    = set((mode_constraints or {}).get("incompatible_cinematic") or [])
        preferred_prod  = list((mode_constraints or {}).get("preferred_production") or [])
        compatible_cin  = list((mode_constraints or {}).get("compatible_cinematic") or [])

        resolved = []
        for item in raw_suggestions[:3]:
            if not isinstance(item, dict):
                continue
            prod_id = str(item.get("production_style_id") or "").strip()
            cin_id = str(item.get("cinematic_style_id") or "").strip()
            justification = str(item.get("justification") or "").strip()

            # Post-filter: replace disallowed production style with preferred
            if prod_id in avoid_prod:
                replacement = next(
                    (p for p in preferred_prod if p not in avoid_prod and StyleProfileRegistry.get_production_style(p)),
                    None,
                )
                if replacement:
                    logger.info(
                        "StyleProfileEngine: replaced disallowed production style %r → %r (mode constraint)",
                        prod_id, replacement,
                    )
                    prod_id = replacement

            # Post-filter: replace incompatible cinematic style.
            # Prefer a mode-compatible style first; fall back to any valid non-incompatible style.
            if cin_id in incompat_cin:
                all_cin = [c["id"] for c in StyleProfileRegistry.all_cinematic_styles()]
                replacement = next(
                    (c for c in compatible_cin if c not in incompat_cin and StyleProfileRegistry.get_cinematic_style(c)),
                    None,
                ) or next((c for c in all_cin if c not in incompat_cin), None)
                if replacement:
                    logger.info(
                        "StyleProfileEngine: replaced incompatible cinematic style %r → %r (mode constraint, compatible=%s)",
                        cin_id, replacement, bool(replacement in compatible_cin),
                    )
                    cin_id = replacement

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

    def _apply_mode_merge(self, profile: Dict[str, Any], emp: Dict[str, Any]) -> Dict[str, Any]:
        """Blend emotional_mode_packet's style_modifier_injection into the resolved style profile.

        The profile already has a storyboard_modifiers block from the registry.
        We overlay the mode's style_modifier_injection dict on top so mode-specific
        defaults win while allowing registry values to remain where the mode doesn't
        specify anything.
        """
        if not emp:
            return profile

        injection = emp.get("style_modifier_injection") or {}
        if not injection:
            return profile

        import copy
        profile = copy.deepcopy(profile)

        sb = profile.get("storyboard_modifiers")
        if isinstance(sb, dict):
            # Mode injection OVERRIDES registry values for specified keys.
            # The emotional mode's aesthetic is the locked truth — it must win
            # over whatever the cinematic style registry provides.
            for k, v in injection.items():
                sb[k] = v
        else:
            profile["storyboard_modifiers"] = dict(injection)

        profile.setdefault("emotional_mode", {
            "mode_id":    emp.get("primary_mode") or emp.get("mode_id") or "",
            "mode_label": emp.get("mode_label") or "",
            "weight":     emp.get("primary_weight", 1.0),
        })

        return profile

    def _default_suggestion(self) -> Dict[str, Any]:
        profile = StyleProfileRegistry.default_style_profile()
        profile["justification"] = (
            "A versatile narrative-performance blend with naturalistic cinematography — "
            "suitable for most song types and easily refined once the context is analysed."
        )
        return profile
