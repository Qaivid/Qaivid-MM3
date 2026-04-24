"""Creative Brief Engine — generate 2–4 director treatment variants.

Sits between METAMAN Dialogue and Storyboard. Consumes the locked
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
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"
_MAX_VARIANTS = 4
_MIN_VARIANTS = 2

_BEAT_SECTIONS = ["intro", "verse1", "chorus", "verse2", "outro"]
_BEAT_BOUNDARIES = [0.0, 0.15, 0.40, 0.62, 0.80, 1.0]

# Matches lines like "[Chorus]", "[Verse 2]", "[Pre-Chorus]", "[Bridge]" etc.
_SECTION_TAG_RE = re.compile(r'^\s*\[([^\]]+)\]\s*$')


def _detect_repeated_structure(
    lines: List[str],
) -> Optional[Dict[str, List[str]]]:
    """Detect song structure from unlabeled lyrics by finding repeated line blocks.

    Scans for the longest block of consecutive lines that repeats at least twice
    (the chorus/mukhda).  Unique blocks between repeats become verse sections.

    Returns an ordered dict like:
        {"chorus": [...], "verse": [...], "chorus_2": [...], "verse_2": [...], ...}
    or None if no clear repetition is detected (too short / no repeating block).
    """
    if not lines or len(lines) < 8:
        return None

    normalized = [ln.strip().lower() for ln in lines]
    n = len(normalized)

    best_positions: Optional[List[int]] = None
    best_len = 0

    # Search from longest blocks down to 3 lines — prefer longer blocks
    max_search = min(16, n // 3)
    for block_len in range(max_search, 2, -1):
        seen: Dict[tuple, List[int]] = {}
        for i in range(n - block_len + 1):
            fp = tuple(normalized[i : i + block_len])
            seen.setdefault(fp, []).append(i)

        for positions in seen.values():
            # Remove overlapping occurrences
            valid: List[int] = [positions[0]]
            for pos in positions[1:]:
                if pos >= valid[-1] + block_len:
                    valid.append(pos)
            if len(valid) >= 2:
                # Prefer more repetitions; break ties by longer block
                if not best_positions or len(valid) > len(best_positions) or (
                    len(valid) == len(best_positions) and block_len > best_len
                ):
                    best_positions = valid
                    best_len = block_len
        if best_positions:
            break

    if not best_positions or len(best_positions) < 2:
        return None

    # Build ordered sections: verse blocks between chorus repeats
    sections: Dict[str, List[str]] = {}
    chorus_count = 0
    verse_count = 0
    pos = 0

    for block_start in sorted(best_positions):
        if pos < block_start:
            verse_count += 1
            key = "verse" if verse_count == 1 else f"verse_{verse_count}"
            sections[key] = lines[pos:block_start]

        chorus_count += 1
        ckey = "chorus" if chorus_count == 1 else f"chorus_{chorus_count}"
        sections[ckey] = lines[block_start : block_start + best_len]
        pos = block_start + best_len

    if pos < n:
        verse_count += 1
        key = "verse" if verse_count == 1 else f"verse_{verse_count}"
        sections[key] = lines[pos:]

    # Only return if structure makes sense (3+ sections, no empty ones)
    sections = {k: v for k, v in sections.items() if v}
    if len(sections) >= 3:
        logger.debug(
            "Detected song structure: %s (sections=%d, chorus_len=%d, chorus_repeats=%d)",
            list(sections.keys()), len(sections), best_len, chorus_count,
        )
        return sections
    return None


def _section_lyrics(
    lyrics_text: str,
    lyrics_timed: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """Divide lyrics into sections.

    Priority order:
    1. Structural [Section] tags — if the lyrics contain bracket markers such
       as [Chorus], [Verse], [Pre-Chorus], [Bridge], these are used directly
       to group lines into named sections.  Repeated tags get numeric suffixes
       (chorus, chorus_2, chorus_3; verse, verse_2, verse_3 etc.) so every
       occurrence is a distinct section rather than collapsing into one.
       This handles complex song structures with 3+ verses, multiple
       pre-choruses, etc. without any cap.
    2. Timestamp-based — when timed lyrics are available with meaningful
       temporal spread, time boundaries map lines to the legacy 5-section set.
    3. Repetition detection — find the longest block of lines that repeats
       (the chorus/mukhda) and segment the full song around those repeats.
       Unique blocks between chorus repeats become verse sections.  Works on
       any unlabeled lyrics as long as the chorus appears at least twice.
    4. Positional — proportional distribution across the 5-section fallback
       (last resort when no structure can be inferred).

    Returns an ordered dict keyed by section name mapping to lyric lines.
    No section is created with zero lines.
    """
    # --- Priority 1: structural [Section] tags ---
    raw_lines = list((lyrics_text or "").splitlines())
    if any(_SECTION_TAG_RE.match(ln) for ln in raw_lines):
        sections: Dict[str, List[str]] = {}
        tag_counts: Dict[str, int] = {}
        current_key = "intro"
        sections[current_key] = []
        for ln in raw_lines:
            m = _SECTION_TAG_RE.match(ln)
            if m:
                raw_tag = m.group(1).strip()
                # Normalise to snake_case key: treat whitespace AND hyphens as
                # word separators before stripping remaining non-alnum chars.
                # e.g. "Pre-Chorus" → "pre_chorus", "Verse 2" → "verse_2"
                base_key = re.sub(r'[\s\-]+', '_', raw_tag.lower())
                base_key = re.sub(r'[^a-z0-9_]', '', base_key)
                if not base_key:
                    continue
                tag_counts[base_key] = tag_counts.get(base_key, 0) + 1
                count = tag_counts[base_key]
                # First occurrence uses bare key; subsequent add _2, _3 …
                current_key = base_key if count == 1 else f"{base_key}_{count}"
                if current_key not in sections:
                    sections[current_key] = []
            else:
                text = ln.strip()
                if text:
                    sections[current_key].append(text)
        # Drop any empty sections (e.g. intro had no lines before first tag)
        sections = {k: v for k, v in sections.items() if v}
        if sections:
            return sections

    # --- Priority 2: timestamp-based sectioning ---
    fallback: Dict[str, List[str]] = {s: [] for s in _BEAT_SECTIONS}
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
            if max_ts > 0 and (max_ts - min_ts) > 1.0:
                for e, ts in zip(entries, timestamps):
                    ratio = ts / max_ts
                    text = str(e.get("text") or "").strip()
                    for i, boundary in enumerate(_BEAT_BOUNDARIES[1:], 0):
                        if ratio <= boundary:
                            fallback[_BEAT_SECTIONS[i]].append(text)
                            break
                    else:
                        fallback["outro"].append(text)
                return {k: v for k, v in fallback.items() if v}
            # Degenerate timestamps — extract text and fall through
            lyrics_text = "\n".join(
                str(e.get("text") or "").strip() for e in entries
            ) or lyrics_text

    # --- Priority 3: repetition-based structure detection ---
    content_lines = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
    detected = _detect_repeated_structure(content_lines)
    if detected:
        return detected

    # --- Priority 4: positional distribution (last resort) ---
    lines = content_lines
    if not lines:
        return {k: v for k, v in fallback.items() if v}
    n = len(lines)
    for i, line in enumerate(lines):
        ratio = i / n
        for j, boundary in enumerate(_BEAT_BOUNDARIES[1:], 0):
            if ratio < boundary:
                fallback[_BEAT_SECTIONS[j]].append(line)
                break
        else:
            fallback["outro"].append(line)
    return {k: v for k, v in fallback.items() if v}


def _format_lyric_sections(sections: Dict[str, List[str]]) -> str:
    """Render sectioned lyrics as a human-readable block for the GPT prompt.

    Handles both the legacy 5-key set and the dynamic tag-derived keys that
    _section_lyrics now produces (e.g. pre_chorus, verse_2, chorus_3, bridge).
    Sections are rendered in their original insertion order.
    """
    _STATIC_LABELS: Dict[str, str] = {
        "intro":  "INTRO",
        "verse1": "VERSE 1",
        "chorus": "CHORUS",
        "verse2": "VERSE 2",
        "outro":  "OUTRO",
    }
    parts: List[str] = []
    for key, lines in sections.items():
        if not lines:
            continue
        if key in _STATIC_LABELS:
            label = _STATIC_LABELS[key]
        else:
            # Convert snake_case key back to a readable heading.
            # "pre_chorus_2" → "PRE-CHORUS 2", "verse_3" → "VERSE 3" etc.
            label = re.sub(r'_(\d+)$', r' \1', key)  # trailing _N → space N
            label = label.replace('_', '-').upper()
        parts.append(f"[{label}]\n" + "\n".join(lines))
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
        "4. HONOUR THE CULTURAL WORLD. Each location must be grounded in the "
        "geographic and cultural context of this song's world. But each location "
        "type has its own distinct visual identity — a mustard field looks "
        "different from a courtyard, which looks different from a rooftop. "
        "Use the location name, the lyrics, and the cultural context to determine "
        "what each space looks like. Do not apply the same visual template to "
        "every location.\n"
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
        "Cultural world: Punjab region, South Asia — Punjabi folk tradition.\n\n"
        "Character appearance: South Asian (Punjabi) — warm wheatish to tan complexion; "
        "phulkari dupatta, salwar-kameez or kurta-pajama; turban for adult men where appropriate; "
        "kohl-lined eyes and simple traditional jewelry for women.\n\n"
        "Visual guidance:\n"
        "  - Ground every location in the Punjabi cultural world, but let each location's "
        "name and the lyrics determine its specific appearance — a mustard field, a rooftop, "
        "a river bank, and a courtyard are all distinct spaces with different visual identities.\n"
        "  - Do not apply the same architectural template to every scene.\n"
        "  - Do not render speakers with East Asian, European, or African features "
        "unless the text explicitly demands it.\n"
        "  - Do not default to Rajasthani haveli ornamentation or Mughal architectural style.\n"
        "  - Do not substitute a generic 'rural Indian village' — the Punjabi cultural world "
        "is specific; let the story and location name guide the visual."
    ),
    "urdu ghazal": (
        "Cultural world: Urdu ghazal / classical South Asian poetry tradition.\n\n"
        "Visual guidance:\n"
        "  - Ground visuals in the classical Urdu cultural sphere — subdued, restrained elegance.\n"
        "  - Avoid bright saturated colours — favour muted, deep, candlelit tones.\n"
        "  - Do not substitute a Bollywood pop aesthetic.\n"
        "  - Do not literalize symbolic or metaphorical language into crude plot imagery."
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

    if pack and isinstance(pack, dict):
        world_defaults = pack.get("world_defaults") or {}
        restrictions: List[str] = list(pack.get("visual_restrictions") or [])
        misinterps: List[str] = list(pack.get("common_misinterpretations") or [])

        geography = str(world_defaults.get("geography") or "").strip()
        cultural_dna = str(world_defaults.get("cultural_dna") or "").strip()

        parts: List[str] = []
        if geography or cultural_dna:
            ctx = " — ".join(filter(None, [geography, cultural_dna]))
            parts.append(f"Cultural world: {ctx}")
        if restrictions:
            parts.append("Visual guidance:\n"
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
    narrative_packet: Optional[Dict[str, Any]] = None,
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
            n_sections = len(sections) if (lyrics or lyrics_timed) else 0
            scene_count_instruction = (
                f"\n\nSCENE COUNT RULE: The lyrics above have exactly {n_sections} "
                f"labeled sections. Each variant MUST contain exactly {n_sections} "
                "scenes — one scene per section. Do NOT merge sections, do NOT drop "
                "sections, and do NOT add extra scenes beyond this count."
            ) if n_sections >= 2 else ""
            lyric_block = (
                "\n\n"
                + "\n\n---\n\n".join(parts)
                + scene_count_instruction
            )

    # ── Narrative Intelligence (Stage 3) ──────────────────────────────────
    # When narrative_packet is provided (post-brain pipeline), inject the
    # locked narrative strategy so every variant honours storytelling mode,
    # perspective, motion philosophy, etc. — not just the raw context+style.
    narrative_block = ""
    if isinstance(narrative_packet, dict) and narrative_packet:
        try:
            from narrative_engine import format_for_prompt
            _ni_block = format_for_prompt(narrative_packet)
            if _ni_block:
                narrative_block = (
                    "\n\n" + _ni_block
                    + "\n\nEvery variant below MUST honour the NARRATIVE INTELLIGENCE "
                    "above (storytelling mode, perspective, motion philosophy, "
                    "expression channels). Treatments that contradict the locked "
                    "strategy are invalid."
                )
        except Exception:
            logger.exception("CreativeBriefEngine: failed to format narrative_packet")

    return (
        "Context for this song:\n"
        + json.dumps(payload, indent=2)
        + narrative_block
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
          '          "beat_range": "Name of the lyric section this scene covers (e.g. intro, verse1, chorus, pre-chorus, verse_2, bridge, chorus_3, outro — must match the actual structure of the provided lyrics)",\n'
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
        for s in scenes_in:
            if isinstance(s, dict):
                props_in = s.get("props") or []
                scenes.append({
                    "name":       str(s.get("name") or "").strip()[:80],
                    "beat_range": str(s.get("beat_range") or "").strip()[:80],
                    "summary":    str(s.get("summary") or "").strip()[:600],
                    "location":   str(s.get("location") or "").strip()[:300],
                    "time_of_day": str(s.get("time_of_day") or "").strip()[:50],
                    "props":      [str(p).strip() for p in (props_in if isinstance(props_in, list) else []) if p],
                })
    vl_in = raw.get("visual_locations") or []
    visual_locations = [str(x).strip() for x in (vl_in if isinstance(vl_in, list) else []) if str(x).strip()]
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
    narrative_packet: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Generate `n` (clamped 2–4) distinct creative-brief variants.

    Inputs (per master spec — Brief is Stage 5):
      • context_packet   (Stage 2) — locked meaning, world, speaker, motivation
      • narrative_packet (Stage 3) — storytelling mode, perspective, motifs
      • style_profile    (Stage 4) — chosen production + cinematic style
      • lyrics / lyrics_timed     — drive scene-specific locations

    narrative_packet is optional for backward compatibility with pre-brain
    projects. When provided, every variant must honour the locked narrative
    strategy.

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
            _user_prompt(
                context_packet, sp, entity_names,
                lyrics, lyrics_timed,
                narrative_packet=narrative_packet,
            )
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
