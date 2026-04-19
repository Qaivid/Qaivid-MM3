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

SYSTEM_PROMPT = """You are a world-class literary analyst, dramaturg, and cultural interpreter — and crucially, also a cinematographer's brain. Your job is to deeply interpret a piece of expressive text (song lyrics, poem, ghazal, qawwali, script, story, voiceover) and produce a structured analysis that downstream scene/shot engines can turn into a CINEMATIC piece — not a uniform emotional wash.

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
      "emotional_meaning": "specific emotion (see EMOTIONAL PALETTE below — avoid reusing 'grief'/'sorrow' on every line)",
      "emotional_intensity": "low|medium|high",
      "expression_mode": "face|body_posture|environment|object|absence|silence|memory_warmth",
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

CRITICAL RULES — LITERARY:
- Treat metaphors as metaphors. Do NOT literalize symbolic language.
- Identify visualization_mode carefully.
- Be culturally precise. If South Asian content, use appropriate frameworks.
- Return ONLY the JSON. No markdown, no explanation.

CRITICAL RULES — EMOTIONAL CINEMATOGRAPHY (these prevent uniformly sad / crying-in-every-shot videos):

1. DO NOT apply a single dominant emotion (grief, sorrow, joy, anger) globally. Sad songs are NOT 100% crying. They are mostly restraint, routine, waiting, memory warmth, silence, and dignity — punctuated by a few peak moments of breakdown. Reflect that distribution.

2. EMOTIONAL PALETTE for `emotional_meaning` — pick from this vocabulary, do NOT collapse to one word:
   restraint, longing, waiting, silence, dignity, numbness, memory_warmth, quiet_routine, private_breakdown, public_composure, ache, tenderness, resignation, hollowness, fading_hope, suspended_time, ritual_continuation, absence_felt_in_objects, dawn_lucidity, night_solitude, performative_grief, breakdown.
   Reserve "breakdown" / "private_breakdown" / "performative_grief" for AT MOST 1–2 peak lines per piece. Most lines should sit in restraint, longing, waiting, routine, memory_warmth, silence.

3. `emotional_intensity` MUST vary across lines. A 20-line song should show a mix of low/medium/high — not all high. Most lines in a lament are LOW or MEDIUM intensity; HIGH is reserved for the climax and one or two emotional spikes.

4. `emotional_arc` intensities MUST NOT all be the same. If you find yourself writing high/high/high/high, you are wrong. A real arc has rise and fall — e.g. low → medium → high → medium, or medium → low → high → low. Resolution is rarely HIGH; it is usually medium or low (resignation, acceptance, exhaustion).

5. `expression_mode` per line tells the shot engine HOW to externalise the emotion. Use the full range:
   - face         → close-up of expression (USE SPARINGLY — only for true peaks)
   - body_posture → slumped shoulders, still hands, the way she sits at the threshold
   - environment  → empty courtyard, fog over fields, an unmade manji, winter light through a doorway
   - object       → a cold cup of tea, an unworn shawl, a charkha that has stopped, his abandoned chair
   - absence      → the empty space where someone used to be, a half-set table, two cups but one untouched
   - silence      → stillness, no movement, held breath, ambient quiet
   - memory_warmth→ a remembered moment shown in warmer light/color than the present
   For wide shots, prefer environment / absence / object / silence — NEVER force a face/crying read into a wide shot.

6. MOTIF MAP must be DIVERSE. For any sad/separation/longing piece, you MUST include at least these motifs (in addition to whatever the text literally names):
   waiting, absence, domestic_routine, silence, memory_warmth, winter_light, fading_presence, threshold (door/courtyard edge), unfinished_objects.
   Do NOT let any single motif (especially "tears" or "crying") dominate. If "tears" appears in the text, map it ONLY to the lines that literally invoke it — do not extend it to every emotional line. The motif_map is a directorial palette, not a word-frequency count.

7. RESOLVE-WHEN-IMPLIED for ambiguity_flags. Do not flag a field as ambiguous if context already implies it:
   - If the piece is a rural Punjabi separation lament with a domestic-female voice, then social_role=wife/beloved/daughter-in-law, domestic_setting=courtyard or single-room home, economic_context=agrarian/working-class are IMPLIED — fill them with your best inference and lower their confidence rather than punting to "unspecified".
   - Only raise an ambiguity_flag when the text genuinely admits multiple irreconcilable readings (e.g. is the addressee a lover, God, or motherland? is he dead or merely abroad?). Those are real ambiguities worth flagging.
   - Never flag speaker_model.gender as ambiguous if AUDIO ANALYSIS hints below specify it.

8. CULTURAL TRANSLATION — when motifs are culturally loaded (pardes, charkha, chunni, dupatta, manji, chulha), the motif_map name should be the cultural concept (e.g. "pardes_separation", "charkha_devotion") not the literal object, and `expression_mode` should usually be `object` or `environment`, not `face`.

9. CONFIDENCE — prefer to make a confident inference at confidence 0.6 over leaving a field unspecified at confidence 0.0. Downstream stages can build with a soft-locked inference; they cannot build with a blank.

10. CINEMATIC ENGAGEMENT RULE (global, non-negotiable) — emotional accuracy alone is not enough; the result must be WATCHABLE.
    Even when the piece is sad, the visual treatment must vary across scenes:
    - Adjacent lines/scenes must not repeat the same `expression_mode`. If line N is `face`, line N+1 should NOT be `face` — make it `environment`, `object`, `body_posture`, `silence`, `absence`, or `memory_warmth`.
    - Spread these expression modes across the piece so a director gets a real palette to cut between: restraint, routine, silence, memory_warmth, distance, observation, and only sparingly — private emotional release.
    - Wide framings express through environment and space. Mid framings express through body language and posture. Close-ups are reserved for genuine emotional peaks (HIGH intensity only).
    - The piece should feel narratively progressive — early lines often live in routine/restraint/observation, the middle in longing/memory_warmth/silence, the climax in private_breakdown or performative_grief, the resolution in resignation/dignity/quiet_routine. Not the same emotional register on every line.
    - Repetition kills engagement. If two consecutive lines share the same `emotional_meaning` AND the same `expression_mode`, you are wrong — vary one of them.
    Goal: preserve emotional depth, lose visual monotony.

11. SONG VISUAL AESTHETIC STANDARD (applies only when input_type is "song") —
    Music video audiences disengage from raw grime. Regardless of how poor, rural, or economically deprived the setting is,
    all visual depictions for songs must be CLEAN and CINEMATICALLY DIGNIFIED.
    - A mud-floor courtyard should feel like a still from a Satyajit Ray film — composed, lit, intentional — not like documentary poverty footage.
    - Humble settings are not dirty settings. Worn cloth is not soiled cloth. Empty rooms are not neglected rooms.
    - Your `cultural_setting.restrictions` output for songs MUST include: "no dirt or grime on surfaces", "no soiled clothing",
      "no raw poverty aesthetic — humble settings must feel dignified and cinematic".
    - This rule does NOT apply to scripts, stories, voiceovers, or documentary content — those follow their own world rules.

Return ONLY the JSON."""


_PEAK_EMOTIONS = {"breakdown", "private_breakdown", "performative_grief", "anguish"}
_OVERUSED_DOMINANT_MOTIFS = {"tears", "crying", "weeping", "sorrow", "grief"}
_REQUIRED_LAMENT_MOTIFS = [
    "waiting", "absence", "domestic_routine", "silence",
    "memory_warmth", "winter_light", "fading_presence",
]


def _normalize_emotional_distribution(ctx: Dict[str, Any]) -> None:
    """
    Post-LLM cinematographic guardrails. Catches the most common failure modes
    (uniform-HIGH arc, tears-dominance, every-line-is-grief) and either logs a
    warning or applies a gentle correction. Does NOT rewrite directorial intent.
    """
    # 1) Emotional arc must vary in intensity
    arc = ctx.get("emotional_arc") or []
    intensities = [str(p.get("intensity", "")).lower() for p in arc if isinstance(p, dict)]
    if intensities and len(set(intensities)) == 1 and intensities[0] in {"high", "medium", "low"}:
        logger.warning(
            "[context_engine] emotional_arc has uniform intensity '%s' across all phases — "
            "applying gentle taper so downstream stages get contrast.",
            intensities[0],
        )
        # Gentle taper: opening->medium, dev->high, climax->high, resolution->medium (or low if was high)
        taper = ["medium", "high", "high", "medium"]
        for i, phase in enumerate(arc):
            if isinstance(phase, dict) and i < len(taper):
                phase["intensity"] = taper[i]

    # 2) Per-line emotional intensity distribution: cap "high" at ~30% of lines
    lms = ctx.get("line_meanings") or []
    if lms:
        high_count = sum(1 for lm in lms if str(lm.get("emotional_intensity", "")).lower() == "high")
        if high_count / max(len(lms), 1) > 0.4:
            logger.warning(
                "[context_engine] %d/%d lines marked HIGH intensity (>40%%). "
                "Real laments live mostly in low/medium with selected peaks.",
                high_count, len(lms),
            )

        # 3) Peak emotions (breakdown / performative_grief) capped at 2 lines
        peak_lines = [i for i, lm in enumerate(lms)
                      if str(lm.get("emotional_meaning", "")).lower().replace(" ", "_") in _PEAK_EMOTIONS]
        if len(peak_lines) > 2:
            logger.warning(
                "[context_engine] %d lines marked as peak-breakdown emotion. "
                "Reserving top 2 by line position; demoting rest to 'restraint'.",
                len(peak_lines),
            )
            keep = set(peak_lines[:2])
            for i, lm in enumerate(lms):
                if i in peak_lines and i not in keep:
                    lm["emotional_meaning"] = "restraint"
                    if str(lm.get("emotional_intensity", "")).lower() == "high":
                        lm["emotional_intensity"] = "medium"

        # 4) expression_mode must not be face-only
        modes = [str(lm.get("expression_mode", "")).lower() for lm in lms]
        face_count = sum(1 for m in modes if m == "face")
        if modes and face_count / len(modes) > 0.5:
            logger.warning(
                "[context_engine] %d/%d lines set expression_mode=face. "
                "Wide/environmental shots will look hollow. Diversifying surplus to environment/object/silence.",
                face_count, len(modes),
            )
            rotation = ["environment", "object", "silence", "body_posture", "absence"]
            r = 0
            seen_face = 0
            for lm in lms:
                if str(lm.get("expression_mode", "")).lower() == "face":
                    seen_face += 1
                    # keep first ~25% of face lines; rotate the rest
                    if seen_face > max(1, len(lms) // 4):
                        lm["expression_mode"] = rotation[r % len(rotation)]
                        r += 1

    # 4b) CINEMATIC ENGAGEMENT: no two consecutive lines share BOTH emotional_meaning
    # AND expression_mode. Rotate the offending line through the available palette.
    if lms and len(lms) > 1:
        rotation = ["environment", "body_posture", "object", "silence", "memory_warmth", "absence"]
        r = 0
        adjusted = 0
        for i in range(1, len(lms)):
            prev_em = str(lms[i - 1].get("expression_mode", "")).lower()
            cur_em = str(lms[i].get("expression_mode", "")).lower()
            prev_emo = str(lms[i - 1].get("emotional_meaning", "")).lower()
            cur_emo = str(lms[i].get("emotional_meaning", "")).lower()
            if prev_em and cur_em and prev_em == cur_em and prev_emo == cur_emo:
                # Pick the next rotation value that differs from the previous
                for _ in range(len(rotation)):
                    candidate = rotation[r % len(rotation)]
                    r += 1
                    if candidate != prev_em:
                        lms[i]["expression_mode"] = candidate
                        adjusted += 1
                        break
        if adjusted:
            logger.info(
                "[context_engine] Diversified %d consecutive duplicate expression_modes "
                "(CINEMATIC ENGAGEMENT RULE).",
                adjusted,
            )

    # 5) Motif map: forbid single-motif dominance and ensure complementary motifs exist
    motif_map = ctx.get("motif_map") or {}
    if isinstance(motif_map, dict) and lms:
        total_lines = len(lms)
        dominant = None
        for name, indices in motif_map.items():
            if name.lower() in _OVERUSED_DOMINANT_MOTIFS and isinstance(indices, list):
                if len(indices) / max(total_lines, 1) > 0.5:
                    dominant = (name, len(indices))
                    break
        if dominant:
            logger.warning(
                "[context_engine] motif '%s' covers %d/%d lines (>50%%). "
                "Trimming to its top 3 line indices so it does not dominate the visual palette.",
                dominant[0], dominant[1], total_lines,
            )
            motif_map[dominant[0]] = motif_map[dominant[0]][:3]

        # Ensure complementary lament motifs at least exist as empty palette entries
        # so the scene engine can pull from them. (Empty list = "available motif, no specific
        # line index" — downstream uses motif name as visual_motif_priority.)
        added = []
        for req in _REQUIRED_LAMENT_MOTIFS:
            if req not in motif_map and not any(req in k.lower() for k in motif_map.keys()):
                motif_map[req] = []
                added.append(req)
        if added:
            logger.info(
                "[context_engine] Added complementary lament motifs to palette: %s",
                ", ".join(added),
            )
        ctx["motif_map"] = motif_map


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
    audio_hints: Dict[str, Any] = None,
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

    # Build INDEXED LINES with optional [start→end] timing tags (only when
    # source_input has audio-derived timestamps). The LLM sees pacing as a
    # SOFT hint — it informs expression_mode/intensity choices but does not
    # override the literary analysis.
    line_objs = lines if lines else [{"text": t} for t in cleaned_text.split('\n')]
    has_timing = any(isinstance(ln, dict) and ln.get("start_time") is not None for ln in line_objs)

    def _fmt_t(s):
        try:
            s = float(s); m = int(s // 60); sec = s - m * 60
            return f"{m}:{sec:05.2f}"
        except Exception:
            return "?"

    line_list_lines = []
    for i, ln in enumerate(line_objs):
        t = ln.get("text", "") if isinstance(ln, dict) else str(ln)
        if not t.strip():
            continue
        if has_timing and isinstance(ln, dict) and ln.get("start_time") is not None:
            st = ln.get("start_time"); et = ln.get("end_time")
            dur = ln.get("duration")
            if et is not None:
                line_list_lines.append(f"[{i}] ({_fmt_t(st)}→{_fmt_t(et)}, {dur}s) {t}")
            else:
                line_list_lines.append(f"[{i}] ({_fmt_t(st)}) {t}")
        else:
            line_list_lines.append(f"[{i}] {t}")
    line_list = "\n".join(line_list_lines)
    parts.append(f"\nINDEXED LINES:\n{line_list}")

    if has_timing:
        parts.append(
            "\nLYRIC PACING HINT (soft — these timings come from the actual audio recording):\n"
            "- Each line's parenthetical shows when it is sung and how long it lasts.\n"
            "- Long sustained lines (>5s) and large gaps between consecutive timestamps are breathing room — favor 'environment', 'silence', 'memory_warmth', or 'absence' for those moments.\n"
            "- Rapid-fire consecutive lines (<2s each) carry their own visual rhythm — keep their expression_mode coherent rather than switching every line; the engagement comes from the words themselves.\n"
            "- Use this to inform pacing of emotional_intensity (peaks should land on the longer/emphasised lines, not arbitrary lines), but do NOT let timing override the literary analysis."
        )
    if pre_enrichment_hints:
        parts.append(f"\n{pre_enrichment_hints}")

    if audio_hints:
        vg = (audio_hints.get("vocal_gender") or "").lower()
        va = (audio_hints.get("vocal_age_range") or "").lower()
        audio_lines = []
        if vg in ("male", "female", "non_binary", "mixed"):
            mapped = {"male": "male", "female": "female", "non_binary": "non_binary", "mixed": "unspecified"}[vg]
            audio_lines.append(f"- Lead vocalist gender (heard in audio): {mapped}. Set speaker_model.gender to '{mapped}'. Do NOT add an ambiguity_flag for speaker_model.gender — this was determined directly from the audio recording.")
        if va in ("young", "middle", "elder"):
            audio_lines.append(f"- Lead vocalist age range (heard in audio): {va}. Use this to inform speaker_model.age_range unless the lyrics clearly contradict it.")
        if audio_lines:
            parts.append("\nAUDIO ANALYSIS (authoritative — derived from the actual recording, not the text):\n" + "\n".join(audio_lines))

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

    # Attach source-line timing onto each line_meaning by line_index so
    # downstream scene/shot engines can pace to the actual lyrics.
    if has_timing:
        timing_by_idx = {}
        for i, ln in enumerate(line_objs):
            if isinstance(ln, dict) and ln.get("start_time") is not None:
                timing_by_idx[i] = {
                    "start_time": ln.get("start_time"),
                    "end_time": ln.get("end_time"),
                    "duration": ln.get("duration"),
                }
        for lm in context_data.get("line_meanings", []) or []:
            idx = lm.get("line_index")
            if idx in timing_by_idx:
                t = timing_by_idx[idx]
                if t.get("start_time") is not None:
                    lm["start_time"] = t["start_time"]
                if t.get("end_time") is not None:
                    lm["end_time"] = t["end_time"]
                if t.get("duration") is not None:
                    lm["duration"] = t["duration"]

    # Post-LLM cinematographic normalization (logs warnings, mildly auto-corrects obvious violations)
    _normalize_emotional_distribution(context_data)

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
