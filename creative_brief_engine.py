"""Creative Brief Engine — generate 2–4 director treatment variants.

Sits between JARVIS Dialogue and Storyboard. Consumes the locked
context_packet (5W incl. WHY) and the chosen style_profile and returns a
list of treatment variants the user picks from before storyboarding.

Output shape (each variant dict):
    {
      "id": "v1",
      "title":            str,
      "pitch":            str,   # one-line
      "treatment":        str,   # 3–5 paragraphs
      "scenes":           [ {name, beat_range, summary}, ... ],
      "cast_roster":      [ str, ... ],   # subset of context entity names
      "central_metaphor": str,
      "self_critique":    {"score": int(1-10), "rationale": str},
    }

The engine is async because it does an OpenAI Chat call. A deterministic
fallback variant is returned if the LLM call fails so the pipeline never
hard-stops here.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"
_MAX_VARIANTS = 4
_MIN_VARIANTS = 2

_BEAT_SECTIONS = ["intro", "verse1", "chorus", "verse2", "outro"]
_BEAT_BOUNDARIES = [0.0, 0.15, 0.40, 0.62, 0.80, 1.0]


def _section_lyrics(
    lyrics_text: str,
    lyrics_timed: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """Divide lyrics into beat sections.

    When timed lyrics are available AND have usable temporal spread (i.e.
    timestamps are not all zero / collapsed), uses timestamps relative to
    the final timestamp to assign each line to a section.  Falls back to
    positional proportioning otherwise — including the known all-zero case
    produced by the Whisper fallback path.

    Returns a dict keyed by section name (intro/verse1/chorus/verse2/outro),
    each mapping to the list of lyric lines in that section.
    """
    sections: Dict[str, List[str]] = {s: [] for s in _BEAT_SECTIONS}

    if lyrics_timed and isinstance(lyrics_timed, list) and len(lyrics_timed) > 1:
        entries = [
            e for e in lyrics_timed
            if isinstance(e, dict) and str(e.get("text") or "").strip()
        ]
        if entries:
            timestamps = [
                float(e.get("timestamp") or e.get("start") or 0.0)
                for e in entries
            ]
            max_ts = max(timestamps)
            min_ts = min(timestamps)
            # Only use timestamp-based sectioning when there is meaningful
            # temporal spread.  If all timestamps are 0 (or nearly identical),
            # the Whisper fallback produced a degenerate list — fall through
            # to positional distribution below.
            if max_ts > 0 and (max_ts - min_ts) > 1.0:
                for e, ts in zip(entries, timestamps):
                    ratio = ts / max_ts
                    text = str(e.get("text") or "").strip()
                    for i, boundary in enumerate(_BEAT_BOUNDARIES[1:], 0):
                        if ratio <= boundary:
                            sections[_BEAT_SECTIONS[i]].append(text)
                            break
                    else:
                        sections["outro"].append(text)
                return sections
            # Degenerate timestamps — fall through to positional using text
            # extracted from the timed entries so nothing is lost
            lyrics_text = "\n".join(
                str(e.get("text") or "").strip() for e in entries
            ) or lyrics_text

    lines = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
    if not lines:
        return sections
    n = len(lines)
    for i, line in enumerate(lines):
        ratio = i / n
        for j, boundary in enumerate(_BEAT_BOUNDARIES[1:], 0):
            if ratio < boundary:
                sections[_BEAT_SECTIONS[j]].append(line)
                break
        else:
            sections["outro"].append(line)
    return sections


def _format_lyric_sections(sections: Dict[str, List[str]]) -> str:
    """Render sectioned lyrics as a human-readable block for the GPT prompt."""
    parts: List[str] = []
    labels = {
        "intro":  "INTRO",
        "verse1": "VERSE 1",
        "chorus": "CHORUS",
        "verse2": "VERSE 2",
        "outro":  "OUTRO",
    }
    for key in _BEAT_SECTIONS:
        lines = sections.get(key) or []
        if lines:
            parts.append(f"[{labels[key]}]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _entity_names(context_packet: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    spk = context_packet.get("speaker") or {}
    if isinstance(spk, dict) and spk.get("identity"):
        out.append(str(spk["identity"]))
    addr = context_packet.get("addressee") or {}
    if isinstance(addr, dict) and addr.get("identity"):
        ident = str(addr["identity"])
        if ident and ident.lower() not in {n.lower() for n in out}:
            out.append(ident)
    for ent in (context_packet.get("entities") or []):
        if isinstance(ent, dict):
            t = str(ent.get("type") or "").lower()
            n = str(ent.get("name") or "").strip()
            if n and ("person" in t or "character" in t):
                if n.lower() not in {x.lower() for x in out}:
                    out.append(n)
    return out[:12]


def _system_prompt(cultural_grounding: str = "") -> str:
    base = (
        "You are a senior music-video director crafting immersive, story-rich "
        "creative treatments for a song. You return strict JSON only.\n\n"
        "MANDATORY RULES — every variant MUST follow all of these:\n"
        "1. TELL A VISUAL STORY. Each treatment must have a clear narrative arc "
        "(beginning → middle → end), not just abstract emotions.\n"
        "2. USE 3-5 DISTINCT VISUAL LOCATIONS. No treatment may be set in a "
        "single location. Each scene must specify a different, specific place. "
        "Variety is mandatory.\n"
        "3. LOCATIONS MUST COME FROM THE LYRICS. When a lyric block is supplied "
        "for a beat section (intro/verse1/chorus/etc.), the scene assigned to "
        "that beat section MUST derive its location from the concrete imagery, "
        "objects, and settings mentioned in those specific lines. Read the words "
        "literally — if a line mentions a field, a river, a doorway, a window, "
        "a rooftop — that is the location. Do not invent generic defaults. Two "
        "different songs should produce two different sets of locations because "
        "their lyrics describe different things.\n"
        "4. HONOUR THE ARCHITECTURE. The context packet supplies "
        "architecture_style and characteristic_setting drawn from expert cultural "
        "knowledge. Every location must be consistent with those descriptions — "
        "use the specific materials, finishes, and spatial vocabulary given. "
        "Never override them with generic substitutes.\n"
        "5. SHOW SECONDARY CHARACTERS. The beloved, a friend, or family member "
        "must appear visually in at least two scenes — even as a memory, "
        "silhouette, or presence — not just referenced in text.\n"
        "6. INCLUDE VISUAL PROPS AND OBJECTS. Each scene must specify 1-3 "
        "culturally authentic props drawn from the song's own lyric world "
        "that carry symbolic weight. Choose props that belong to the specific "
        "culture and geography of THIS song — do not default to any single "
        "cultural tradition.\n"
        "7. VARY TIME OF DAY. Different scenes should happen at different times "
        "(dawn, morning, golden hour, dusk, night).\n"
        "8. CONTRAST SCENE MOODS. Alternate between intimate/close and wide/open "
        "settings so the video breathes visually.\n"
        "Never repeat the same treatment twice. Cast members must come from the "
        "supplied entity list."
    )
    if cultural_grounding:
        base += (
            "\n\nCULTURAL ARCHITECTURE MANDATE — this overrides any generic "
            "defaults you might otherwise apply:\n"
            + cultural_grounding
        )
    return base


# Deterministic cultural grounding keyed by geography/location markers.
# Used as a reliable fallback when context_packet["culture_pack"] is absent
# (e.g. older projects) or when the pack object was not serialised to the DB.
# Keys are lowercase fragments that may appear in location_dna or geography.
_MARKER_GROUNDING: Dict[str, str] = {
    "punjab": (
        "Required architecture: kuchha (mud-plastered) village architecture — "
        "thick bare ochre/sand-toned mud walls, flat clay rooftops with exterior "
        "stone or mud staircases, small deep-set windows with blue or turquoise "
        "painted wooden shutters, heavy carved wooden doors painted blue, smooth "
        "plastered parapet walls, no glass or modern cladding.\n\n"
        "Required setting vocabulary: clean swept earthen vehra (open courtyard) "
        "— packed bare mud floor, borders of thick kuchha walls, charpai "
        "(rope-strung wooden bed) in open air, terracotta matkas near the "
        "entrance, mustard or wheat fields visible beyond the compound wall, "
        "open sky overhead, golden or warm afternoon light; no concrete or tile, "
        "no synthetic materials.\n\n"
        "Visual restrictions (every location must comply):\n"
        "  - Do not use Rajasthani haveli ornamentation.\n"
        "  - Walls must be bare plastered mud (kuchha), not brick or painted plaster.\n"
        "  - Courtyard floor must be packed earth, not tile or concrete.\n"
        "  - No Mughal arches, ornate columns, or Indo-Saracenic domes.\n"
        "  - Do not render speakers with East Asian, European, or African features "
        "unless the text explicitly demands it.\n\n"
        "Common misinterpretations to avoid:\n"
        "  - Do not substitute a generic 'rural Indian village'.\n"
        "  - Punjab village architecture is plain, massive, and earthen — not "
        "Rajasthani or Mughal ornate.\n"
        "  - Do not flatten domestic imagery into random rustic props."
    ),
    "urdu ghazal": (
        "Required aesthetic: classical Mughal-Deccani courtly interior or walled "
        "garden — cool marble or polished stone surfaces, arched niches with oil "
        "lamps, latticed jaali screens filtering moonlight, geometric tile work.\n\n"
        "Visual restrictions:\n"
        "  - Avoid generic contemporary interiors.\n"
        "  - Avoid bright saturated colours — favour ivory, deep indigo, "
        "terracotta, and candlelight gold.\n\n"
        "Common misinterpretations to avoid:\n"
        "  - Do not substitute a Bollywood disco aesthetic.\n"
        "  - Do not render the space as a modern apartment or café."
    ),
}

# Ordered list of (marker_fragment, grounding_key) for detection.
# The fragment is lowercased and checked with 'in' against location_dna + geography.
_MARKER_KEYS: List[tuple] = [
    ("punjab", "punjab"),
    ("ghazal", "urdu ghazal"),
    ("urdu",   "urdu ghazal"),
]


def _build_cultural_grounding(context_packet: Dict[str, Any]) -> str:
    """Build a culture-specific grounding block for the system prompt.

    Strategy (in priority order):
    1. Use the full culture_pack object stored in context_packet (richest source).
    2. Fall back to a deterministic marker lookup from location_dna / geography
       (works for older projects where culture_pack may not be serialised).
    3. Return empty string when no marker matches — non-cultural songs are unaffected.
    """
    pack = context_packet.get("culture_pack") or {}

    if pack:
        world_defaults = pack.get("world_defaults") or {}
        restrictions: List[str] = list(pack.get("visual_restrictions") or [])
        misinterps: List[str] = list(pack.get("common_misinterpretations") or [])

        arch = str(world_defaults.get("architecture_style") or "").strip()
        setting = str(world_defaults.get("characteristic_setting") or
                      world_defaults.get("domestic_setting") or "").strip()

        parts: List[str] = []
        if arch:
            parts.append(f"Required architecture: {arch}")
        if setting:
            parts.append(f"Required setting vocabulary: {setting}")
        if restrictions:
            parts.append("Visual restrictions (every location must comply):\n"
                         + "\n".join(f"  - {r}" for r in restrictions))
        if misinterps:
            parts.append("Common misinterpretations to avoid:\n"
                         + "\n".join(f"  - {m}" for m in misinterps))
        if parts:
            return "\n\n".join(parts)

    # Fallback: inspect location_dna and geography for known cultural markers
    world = context_packet.get("world_assumptions") or {}
    location_dna = str(context_packet.get("location_dna") or "").lower()
    geography = str(world.get("geography") or "").lower()
    combined = location_dna + " " + geography

    for fragment, key in _MARKER_KEYS:
        if fragment in combined:
            return _MARKER_GROUNDING.get(key, "")

    return ""


def _user_prompt(
    context_packet: Dict[str, Any],
    style_profile: Dict[str, Any],
    entity_names: List[str],
    lyrics: Optional[str] = None,
    lyrics_timed: Optional[List[Dict[str, Any]]] = None,
) -> str:
    spk = context_packet.get("speaker") or {}
    motivation = context_packet.get("motivation") or {}
    world = context_packet.get("world_assumptions") or {}
    cin = (style_profile or {}).get("cinematic") or {}
    prod = (style_profile or {}).get("production") or {}
    payload = {
        "speaker_identity":    spk.get("identity"),
        "emotional_state":     spk.get("emotional_state"),
        "narrative_mode":      context_packet.get("narrative_mode"),
        "location_dna":        context_packet.get("location_dna"),
        "era":                 context_packet.get("era"),
        "motivation": {
            "inciting_cause":    motivation.get("inciting_cause"),
            "underlying_desire": motivation.get("underlying_desire"),
            "stakes":            motivation.get("stakes"),
            "obstacle":          motivation.get("obstacle"),
        },
        "world_assumptions": {
            "geography":             world.get("geography"),
            "architecture_style":    world.get("architecture_style"),
            "characteristic_setting": world.get("characteristic_setting"),
            "social_context":        world.get("social_context"),
            "season":                world.get("season"),
            "era":                   world.get("era"),
        },
        "cinematic_style":  cin.get("name") or cin.get("id"),
        "production_style": prod.get("name") or prod.get("id"),
        "available_cast":   entity_names,
    }

    lyric_block = ""
    if lyrics or lyrics_timed:
        clean_lyrics = (lyrics or "").strip()
        sections = _section_lyrics(clean_lyrics, lyrics_timed)
        formatted = _format_lyric_sections(sections)
        parts: List[str] = []
        if clean_lyrics:
            parts.append(
                "Full song text (all lyrics in original language — use for "
                "overall tone, recurring imagery, and dominant setting cues):\n\n"
                + clean_lyrics
            )
        if formatted.strip():
            parts.append(
                "Lyrics divided by beat section — each scene assigned to a "
                "beat_range MUST derive its 'location' from the concrete images, "
                "places, and objects in the lines for that section. Read them "
                "literally. Two different songs will produce two different sets "
                "of locations because their lyrics describe different worlds:\n\n"
                + formatted
            )
        if parts:
            lyric_block = (
                "\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n\nIMPORTANT: The architecture_style and characteristic_setting "
                "in world_assumptions define the exact built-environment vocabulary "
                "for this song's world. Every location description must use those "
                "specific materials and spatial terms — never override them with "
                "generic substitutes."
            )

    return (
        "Context for this song:\n"
        + json.dumps(payload, indent=2)
        + lyric_block
        + "\n\nReturn JSON of the form:\n"
        + "{\n"
          '  "variants": [\n'
          "    {\n"
          '      "title": "Short evocative name",\n'
          '      "pitch": "One-line pitch (<= 22 words)",\n'
          '      "treatment": "3-5 short paragraphs of director treatment",\n'
          '      "scenes": [\n'
          '        {\n'
          '          "name": "Scene name",\n'
          '          "beat_range": "intro|verse1|chorus|verse2|outro",\n'
          '          "summary": "What happens visually in this scene (2-3 sentences)",\n'
          '          "location": "Specific visual setting drawn from the lyrics of this beat section",\n'
          '          "time_of_day": "dawn|morning|afternoon|golden_hour|dusk|night",\n'
          '          "props": ["prop1", "prop2"]\n'
          "        }\n"
          "      ],\n"
          '      "visual_locations": ["All distinct locations used across all scenes (2-5 entries)"],\n'
          '      "cast_roster": ["Name", "..."],\n'
          '      "central_metaphor": "Single visual metaphor that ties the whole video together",\n'
          '      "justification": "Why this variant is the strongest choice for this song (1-2 sentences)",\n'
          '      "self_critique": {"score": 1-10, "rationale": "Honest critique of risks / weaknesses"}\n'
          "    }\n"
          f"  ]  (between {_MIN_VARIANTS} and {_MAX_VARIANTS} variants, mutually distinct)\n"
          "}"
    )


def _coerce_variant(raw: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "").strip() or f"Variant {idx + 1}"
    pitch = str(raw.get("pitch") or "").strip()
    treatment = str(raw.get("treatment") or "").strip()
    scenes_in = raw.get("scenes") or []
    scenes: List[Dict[str, Any]] = []
    if isinstance(scenes_in, list):
        for s in scenes_in[:8]:
            if isinstance(s, dict):
                props_in = s.get("props") or []
                scenes.append({
                    "name":       str(s.get("name") or "").strip()[:80],
                    "beat_range": str(s.get("beat_range") or "").strip()[:60],
                    "summary":    str(s.get("summary") or "").strip()[:400],
                    "location":   str(s.get("location") or "").strip()[:200],
                    "time_of_day": str(s.get("time_of_day") or "").strip()[:50],
                    "props":      [str(p).strip() for p in (props_in if isinstance(props_in, list) else [])[:6] if p],
                })
    vl_in = raw.get("visual_locations") or []
    visual_locations = [str(x).strip() for x in (vl_in if isinstance(vl_in, list) else []) if str(x).strip()][:8]
    cast_in = raw.get("cast_roster") or []
    cast = [str(x).strip() for x in cast_in if isinstance(x, (str, int))][:10]
    metaphor = str(raw.get("central_metaphor") or "").strip()
    justification = str(raw.get("justification") or "").strip()[:400]
    sc_in = raw.get("self_critique") or {}
    score = 7
    rationale = ""
    if isinstance(sc_in, dict):
        try:
            score = int(sc_in.get("score") or 7)
        except (TypeError, ValueError):
            score = 7
        score = max(1, min(10, score))
        rationale = str(sc_in.get("rationale") or "").strip()[:600]

    if not (pitch or treatment):
        return None

    return {
        "id":               f"v{idx + 1}",
        "title":            title,
        "pitch":            pitch,
        "treatment":        treatment,
        "scenes":           scenes,
        "visual_locations": visual_locations,
        "cast_roster":      cast,
        "central_metaphor": metaphor,
        "justification":    justification,
        "self_critique":    {"score": score, "rationale": rationale},
    }


def _fallback_variants(context_packet: Dict[str, Any],
                       entity_names: List[str]) -> List[Dict[str, Any]]:
    spk = context_packet.get("speaker") or {}
    speaker_ident = str(spk.get("identity") or "the speaker")
    motivation = context_packet.get("motivation") or {}
    cause = str(motivation.get("inciting_cause") or "an unresolved emotional moment")
    desire = str(motivation.get("underlying_desire") or "to be heard")
    location = str(context_packet.get("location_dna") or "an evocative interior")

    return [
        {
            "id": "v1",
            "title": "Literal Narrative",
            "pitch": f"Follow {speaker_ident} through the inciting moment in real space.",
            "treatment": (
                f"A grounded, observational treatment. We stay close to "
                f"{speaker_ident} as they move through {location}, witnessing "
                f"the lived experience of {cause}. Performance is internal; the "
                f"camera respects breath and silence.\n\n"
                f"Each chorus returns to a single ritual gesture that "
                f"externalizes the desire {desire}. The visual world is "
                f"naturalistic, lit by practical sources only."
            ),
            "scenes": [
                {"name": "Arrival",   "beat_range": "intro",   "summary": f"Speaker enters {location} at dawn.",        "location": f"{location} — dawn light", "time_of_day": "dawn",        "props": ["worn path", "gate"]},
                {"name": "Memory",    "beat_range": "verse1",  "summary": "Intimate indoor memory flashes.",             "location": "dimly lit home interior",  "time_of_day": "morning",     "props": ["letter", "candle"]},
                {"name": "Reckoning", "beat_range": "chorus",  "summary": "Ritual gesture outdoors with rising weight.", "location": "open field",               "time_of_day": "golden_hour", "props": ["wind-swept fabric", "worn object"]},
                {"name": "Release",   "beat_range": "outro",   "summary": "Final breath at the water's edge.",           "location": "riverside at dusk",        "time_of_day": "dusk",        "props": ["rippling water", "lone silhouette"]},
            ],
            "visual_locations": [f"{location} — dawn light", "dimly lit home interior", "open field", "riverside at dusk"],
            "cast_roster": entity_names[:1] or [speaker_ident],
            "central_metaphor": "the journey from interior pain to open release",
            "justification": "Emotionally legible for a wide audience; grounds performance across varied locations.",
            "self_critique": {"score": 7, "rationale": "Safe and emotionally legible but visually conservative."},
        },
        {
            "id": "v2",
            "title": "Symbolic Object Study",
            "pitch": "No people on camera — a lone object carries the entire emotional arc.",
            "treatment": (
                "Strip the song down to a single recurring object that embodies "
                f"the underlying desire {desire}. Macro lens, painterly light, "
                "no face ever resolves on screen.\n\n"
                "Each verse adds a new tactile detail; each chorus accelerates "
                "a transformation of the object (water rising, paper burning, "
                "petals falling). The viewer projects the speaker onto the "
                "object."
            ),
            "scenes": [
                {"name": "Object",   "beat_range": "intro",  "summary": "Object introduced in stillness on a window sill.", "location": "stone window sill at morning light", "time_of_day": "morning",     "props": ["flower", "dust particles"]},
                {"name": "Pressure", "beat_range": "verses", "summary": "Object transforms — textures change with each verse.", "location": "outdoor courtyard",              "time_of_day": "afternoon",    "props": ["earth", "scattered petals"]},
                {"name": "Rupture",  "beat_range": "chorus", "summary": "Transformation accelerates in open air.",            "location": "open hilltop",                    "time_of_day": "golden_hour",  "props": ["burning paper", "wind"]},
                {"name": "Aftermath","beat_range": "outro",  "summary": "Stillness returns at the water's edge.",             "location": "riverside at night",              "time_of_day": "night",        "props": ["ripples", "reflection"]},
            ],
            "visual_locations": ["stone window sill", "outdoor courtyard", "open hilltop", "riverside at night"],
            "cast_roster": [],
            "central_metaphor": "the object as the unspoken self",
            "justification": "No cast required — purely visual; maximises cinematic freedom and avoids character consistency issues.",
            "self_critique": {"score": 8, "rationale": "Bold but risks emotional distance if metaphor isn't read."},
        },
    ]


async def generate_variants(
    api_key: str,
    context_packet: Dict[str, Any],
    style_profile: Optional[Dict[str, Any]] = None,
    n: int = 3,
    lyrics: Optional[str] = None,
    lyrics_timed: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Generate `n` (clamped 2–4) distinct creative-brief variants.

    lyrics and lyrics_timed are used to derive scene-specific locations from
    the actual imagery in the song rather than generic cultural defaults.

    Returns a tuple of (variants, used_fallback) where used_fallback is True
    when the LLM failed to produce enough valid variants and hardcoded defaults
    were substituted instead.
    """
    n = max(_MIN_VARIANTS, min(_MAX_VARIANTS, int(n or 3)))
    entity_names = _entity_names(context_packet or {})
    sp = style_profile or {}

    try:
        client = AsyncOpenAI(api_key=api_key)
        cultural_grounding = _build_cultural_grounding(context_packet or {})
        user_content = (
            _user_prompt(context_packet, sp, entity_names, lyrics, lyrics_timed)
            + f"\n\nProduce exactly {n} variants."
        )
        resp = await client.chat.completions.create(
            model=_MODEL,
            response_format={"type": "json_object"},
            temperature=0.9,
            messages=[
                {"role": "system", "content": _system_prompt(cultural_grounding)},
                {"role": "user",   "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        variants_in = data.get("variants") or []
        coerced: List[Dict[str, Any]] = []
        for i, v in enumerate(variants_in):
            cv = _coerce_variant(v, i)
            if cv:
                coerced.append(cv)
            if len(coerced) >= n:
                break
        if len(coerced) < _MIN_VARIANTS:
            logger.warning("Creative brief LLM returned <2 variants; using fallback")
            return _fallback_variants(context_packet or {}, entity_names), True
        return coerced, False
    except Exception:
        logger.exception("Creative brief generation failed; returning fallback variants")
        return _fallback_variants(context_packet or {}, entity_names), True


def coerce_chosen(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize user-submitted chosen-variant form payload."""
    out = {
        "variant_id":       str(payload.get("variant_id") or "").strip()[:20],
        "title":            str(payload.get("title") or "").strip()[:120],
        "pitch":            str(payload.get("pitch") or "").strip()[:300],
        "treatment":        str(payload.get("treatment") or "").strip()[:6000],
        "central_metaphor": str(payload.get("central_metaphor") or "").strip()[:300],
        "director_note":    str(payload.get("director_note") or "").strip()[:600],
        "justification":    str(payload.get("justification") or "").strip()[:400],
    }
    raw_cast = payload.get("cast_roster")
    if isinstance(raw_cast, str):
        cast = [c.strip() for c in raw_cast.split(",") if c.strip()]
    elif isinstance(raw_cast, list):
        cast = [str(c).strip() for c in raw_cast if str(c).strip()]
    else:
        cast = []
    out["cast_roster"] = cast[:10]
    return out
