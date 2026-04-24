"""
input_processor.py — Stage 1 of the Qaivid cinematic pipeline.

Role: Prepare clean, structured, and reliable data so the Context Engine
      can extract meaning correctly.

CORE RULE: Clean → Segment → Label → Preserve → Pass Forward

This stage must NOT:
  - interpret meaning
  - infer emotion
  - create visuals
  - define characters beyond labels
  - influence storytelling decisions

Five input types handled:
  song     — lyrics / ghazal / qawwali / poem (singing/lyric form)
  script   — screenplay / stage play / ad / voiceover
  story    — narrative prose / documentary text
  document — concept / idea / heading-based content
  audio    — no text provided; pure audio transcription

Output: input_packet consumed by Stage 2 (Context Engine).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type constants
# ---------------------------------------------------------------------------

TYPE_SONG     = "song"
TYPE_SCRIPT   = "script"
TYPE_STORY    = "story"
TYPE_DOCUMENT = "document"
TYPE_AUDIO    = "audio"

# Song section canonical labels
SONG_SECTION_ORDER = ["intro", "verse", "pre_chorus", "chorus", "post_chorus",
                      "bridge", "hook", "refrain", "interlude", "outro"]

# Bracket marker aliases → canonical section type
_BRACKET_ALIASES: Dict[str, str] = {
    "intro": "intro", "introduction": "intro",
    "verse": "verse", "v": "verse",
    "pre-chorus": "pre_chorus", "pre chorus": "pre_chorus", "prechorus": "pre_chorus",
    "chorus": "chorus", "ch": "chorus", "hook": "chorus", "refrain": "refrain",
    "post-chorus": "post_chorus", "post chorus": "post_chorus",
    "bridge": "bridge", "b": "bridge",
    "interlude": "interlude",
    "outro": "outro", "outtro": "outro", "ending": "outro", "end": "outro",
    "instrumental": "interlude",
    "breakdown": "bridge",
    "spoken": "verse",
}

# Script patterns
_SCENE_HEADER_RE = re.compile(
    r"^\s*(INT\.|EXT\.|INT/EXT\.|I/E\.)", re.IGNORECASE
)
_CHARACTER_NAME_RE = re.compile(
    r"^([A-Z][A-Z\s\.\-\']{1,40})(?:\s*\([^)]*\))?\s*$"
)
_INLINE_DIALOGUE_RE = re.compile(
    r"^([A-Z][A-Z ]{1,30}):\s+(.+)$"
)
_STAGE_DIRECTION_RE = re.compile(r"^\s*\((.+)\)\s*$")
_TRANSITION_RE = re.compile(
    r"^\s*(FADE\s*(IN|OUT|TO)?|CUT\s*TO|SMASH\s*CUT|DISSOLVE(\s*TO)?|"
    r"MATCH\s*CUT|JUMP\s*CUT|WIPE(\s*TO)?|IRIS\s*(IN|OUT))\s*[:\.]?\s*$",
    re.IGNORECASE,
)
_NARRATOR_RE = re.compile(
    r"^(NARRATOR|V\.O\.|V\.O|VO|VOICE\s*OVER|VOICE-OVER|OMNISCIENT|NARRATION)$",
    re.IGNORECASE,
)

# Document heading patterns
_DOC_HEADING_RE = re.compile(
    r"^(\d+[\.\)]\s.{2,80}|#{1,6}\s.{2,80}|[A-Z][A-Z ]{5,60}:?\s*)$"
)

# Whisper SRT / timecoded input
_SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2}[,\.]\d{2,3}")
_TIMECODED_LINE_RE = re.compile(
    r"^\[?(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\]?\s+(.*)$"
)


# ===========================================================================
# Main processor
# ===========================================================================

class InputProcessor:
    """
    Stage 1 — Input Processing.

    Converts raw user input (text + optional audio metadata) into a clean,
    structured input_packet for the Context Engine.
    """

    def process(
        self,
        raw_text: str,
        genre_hint: Optional[str] = None,
        audio_meta: Optional[Dict[str, Any]] = None,
        timed_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            raw_text:       Raw input text (may be empty if pure audio input)
            genre_hint:     User-provided genre label (song/poem/ghazal/script/…)
            audio_meta:     Features from AudioProcessor.extract_features()
            timed_segments: Per-segment Whisper timestamps [{text, start, end}, …]

        Returns:
            input_packet — structured, labelled, ready for Context Engine
        """
        audio_meta     = audio_meta or {}
        timed_segments = timed_segments or []
        genre_hint     = (genre_hint or "").strip().lower()

        # --- Step 1: preserve raw, produce clean --------------------------
        raw_text_preserved = raw_text
        clean_text, source_format = self._clean_and_detect_format(raw_text)

        # --- Step 2: detect type and sub-type -----------------------------
        is_pure_audio = (not clean_text.strip()) and bool(audio_meta)
        if is_pure_audio:
            input_type = TYPE_AUDIO
            sub_type   = "instrumental"
        else:
            input_type, sub_type = self._detect_type(
                clean_text, genre_hint, audio_meta, timed_segments
            )

        # --- Step 3: detect language(s) -----------------------------------
        languages = self._detect_languages(clean_text)

        # --- Step 4: segment into atomic units ----------------------------
        raw_lines = self._split_raw_lines(clean_text, source_format, timed_segments)

        # --- Step 5: strip explicit bracket markers, record them ----------
        raw_lines, bracket_markers = self._extract_bracket_markers(raw_lines)

        # --- Step 6: build sections and assign unit IDs -------------------
        sections, units = self._build_structure(
            raw_lines, bracket_markers, input_type
        )

        # --- Step 7: attach timing to units (if available) ----------------
        units = self._attach_timing(units, timed_segments, audio_meta)

        # --- Step 8: build repetition map ---------------------------------
        repetition_map = self._build_repetition_map(units, sections)

        # --- Step 8.5: detect lyrical patterns (songs only) ---------------
        lyrical_patterns = self._detect_lyrical_patterns(sections, units, input_type)

        # --- Step 9: detect speaker boundaries ----------------------------
        speaker_data = self._detect_speakers(units, input_type)

        # --- Step 10: flag uncertainties ----------------------------------
        uncertainties = self._flag_uncertainties(
            units, sections, input_type, clean_text, bracket_markers
        )

        # --- Step 11: build timing summary --------------------------------
        timing = self._build_timing_summary(audio_meta, timed_segments, units)

        return {
            "schema_version":  1,
            "raw_text":        raw_text_preserved,
            "clean_text":      clean_text,
            "source_format":   source_format,
            "input_type":      input_type,
            "sub_type":        sub_type,
            "languages":       languages,
            "sections":        sections,
            "units":           units,
            "repetition_map":  repetition_map,
            "lyrical_patterns": lyrical_patterns,
            "speaker_map":     speaker_data["speakers"],
            "speaker_types":   speaker_data["speaker_types"],
            "timing":          timing,
            "audio_meta":      self._compact_audio_meta(audio_meta),
            "uncertainties":   uncertainties,
            "resegmented_by_llm": False,
        }

    # -----------------------------------------------------------------------
    # Step 1 — Clean text + detect source format
    # -----------------------------------------------------------------------

    def _clean_and_detect_format(self, raw: str) -> Tuple[str, str]:
        if not isinstance(raw, str):
            raw = str(raw or "")

        text = raw.strip()

        # Fix common encoding artefacts
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", " - ").replace("\u2013", "-")
        text = text.replace("\u00a0", " ")

        if _SRT_TIMESTAMP_RE.search(text):
            return self._clean_srt(text), "srt"

        if re.search(_TIMECODED_LINE_RE.pattern, text, re.MULTILINE):
            return self._clean_timecoded(text), "timecoded"

        # Normalise line endings; collapse excessive blank lines to max 2
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text, "plain_text"

    def _clean_srt(self, text: str) -> str:
        blocks = re.split(r"\n\s*\n", text.strip())
        lines = []
        for block in blocks:
            blines = [x.strip() for x in block.splitlines() if x.strip()]
            if len(blines) < 2:
                continue
            # Skip index line and timestamp line
            start = 2 if re.match(r"^\d+$", blines[0]) and "-->" in blines[1] else \
                    1 if "-->" in blines[0] else 0
            content = " ".join(blines[start:]).strip()
            if content:
                lines.append(content)
        return "\n".join(lines)

    def _clean_timecoded(self, text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            m = _TIMECODED_LINE_RE.match(raw_line.strip())
            if m:
                lines.append(m.group(2).strip())
            elif raw_line.strip():
                lines.append(raw_line.strip())
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Step 2 — Detect type and sub-type
    # -----------------------------------------------------------------------

    def _detect_type(
        self,
        text: str,
        genre_hint: str,
        audio_meta: Dict[str, Any],
        timed_segments: List[Dict[str, Any]],
    ) -> Tuple[str, str]:

        hint_map: Dict[str, Tuple[str, str]] = {
            "song":        (TYPE_SONG,     "lyric"),
            "lyrics":      (TYPE_SONG,     "lyric"),
            "poem":        (TYPE_SONG,     "poem"),
            "poetry":      (TYPE_SONG,     "poem"),
            "ghazal":      (TYPE_SONG,     "ghazal"),
            "qawwali":     (TYPE_SONG,     "qawwali"),
            "sufi":        (TYPE_SONG,     "qawwali"),
            "nasheeds":    (TYPE_SONG,     "nasheed"),
            "script":      (TYPE_SCRIPT,   "screenplay"),
            "screenplay":  (TYPE_SCRIPT,   "screenplay"),
            "stage_play":  (TYPE_SCRIPT,   "stage_play"),
            "ad":          (TYPE_SCRIPT,   "ad"),
            "advertisement": (TYPE_SCRIPT, "ad"),
            "voiceover":   (TYPE_SCRIPT,   "voiceover"),
            "voice_over":  (TYPE_SCRIPT,   "voiceover"),
            "story":       (TYPE_STORY,    "prose"),
            "narrative":   (TYPE_STORY,    "prose"),
            "documentary": (TYPE_STORY,    "documentary"),
            "document":    (TYPE_DOCUMENT, "concept"),
            "concept":     (TYPE_DOCUMENT, "concept"),
        }
        if genre_hint in hint_map:
            return hint_map[genre_hint]

        lower = text.lower()
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Script signals
        has_scene_headers = any(_SCENE_HEADER_RE.match(l) for l in lines)
        caps_lines = [l for l in lines if _CHARACTER_NAME_RE.match(l)]
        if has_scene_headers or (len(caps_lines) >= 3 and len(caps_lines) / max(len(lines), 1) > 0.15):
            sub = "ad" if any(x in lower for x in ["buy now", "order now", "limited offer", "introducing"]) else "screenplay"
            return TYPE_SCRIPT, sub

        # Voiceover / documentary
        if "voiceover" in lower or "voice over" in lower:
            return TYPE_SCRIPT, "voiceover"
        if "documentary" in lower:
            return TYPE_STORY, "documentary"

        # Song signals — explicit labels OR lyric form
        has_brackets = bool(re.search(r"\[(verse|chorus|bridge|intro|outro|hook|refrain)", lower))
        high_repetition = self._has_high_repetition(lines)
        short_line_form  = len(lines) >= 4 and sum(1 for l in lines if len(l) < 60) / max(len(lines), 1) > 0.7
        has_bpm = bool(audio_meta.get("bpm"))

        if has_brackets or (short_line_form and high_repetition) or (has_bpm and short_line_form):
            sub = self._detect_song_sub_type(lower)
            return TYPE_SONG, sub

        # Poem signals (lyric form, no repetition, no audio)
        if short_line_form and not high_repetition:
            sub = self._detect_song_sub_type(lower)
            return TYPE_SONG, sub

        # Story signals
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if len(paragraphs) >= 2 and avg_para_len > 100:
            return TYPE_STORY, "prose"

        # Document signals
        heading_lines = [l for l in lines if _DOC_HEADING_RE.match(l)]
        if len(heading_lines) >= 2:
            return TYPE_DOCUMENT, "concept"

        # Fallback: if it has audio and short lines → song
        if has_bpm:
            return TYPE_SONG, "lyric"

        return TYPE_SONG, "lyric"  # most common use case for this app

    def _detect_song_sub_type(self, lower: str) -> str:
        if "ghazal" in lower:
            return "ghazal"
        if any(x in lower for x in ["qawwali", "alaap", "sufi"]):
            return "qawwali"
        if any(x in lower for x in ["nasheed", "naat"]):
            return "nasheed"
        return "lyric"

    def _has_high_repetition(self, lines: List[str]) -> bool:
        normed = [self._norm(l) for l in lines if l.strip()]
        counts = Counter(normed)
        repeated = sum(1 for v in counts.values() if v > 1)
        return repeated >= 2

    # -----------------------------------------------------------------------
    # Step 3 — Language detection (character-block based)
    # -----------------------------------------------------------------------

    def _detect_languages(self, text: str) -> List[str]:
        if not text.strip():
            return ["unknown"]

        block_counts: Dict[str, int] = Counter()
        for ch in text:
            cp = ord(ch)
            if 0x0A00 <= cp <= 0x0A7F:
                block_counts["Punjabi (Gurmukhi)"] += 1
            elif 0x0600 <= cp <= 0x06FF:
                block_counts["Arabic/Urdu"] += 1
            elif 0x0900 <= cp <= 0x097F:
                block_counts["Hindi (Devanagari)"] += 1
            elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
                block_counts["Korean"] += 1
            elif 0x4E00 <= cp <= 0x9FFF:
                block_counts["Chinese/Japanese"] += 1
            elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
                block_counts["Latin"] += 1

        if not block_counts:
            return ["unknown"]

        total = sum(block_counts.values())
        # Return languages that account for ≥ 10% of script characters
        detected = [lang for lang, cnt in sorted(block_counts.items(), key=lambda x: -x[1])
                    if cnt / total >= 0.10]
        return detected or ["unknown"]

    # -----------------------------------------------------------------------
    # Step 4 — Split into raw lines
    # -----------------------------------------------------------------------

    def _split_raw_lines(
        self,
        clean_text: str,
        source_format: str,
        timed_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return a list of raw line dicts: {text, start_time, end_time, blank_before}.
        blank_before=True means there was a blank line before this line in the input.
        """
        if not clean_text.strip():
            return []

        raw_lines: List[Dict[str, Any]] = []
        prev_blank = False

        for raw_line in clean_text.split("\n"):
            if not raw_line.strip():
                prev_blank = True
                continue
            raw_lines.append({
                "text":         raw_line.strip(),
                "start_time":   None,
                "end_time":     None,
                "blank_before": prev_blank,
            })
            prev_blank = False

        return raw_lines

    # -----------------------------------------------------------------------
    # Step 5 — Extract explicit bracket markers
    # -----------------------------------------------------------------------

    def _extract_bracket_markers(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
        """
        Scan lines for [Verse], [Chorus], etc.
        Remove them from the line list.
        Return: (cleaned_lines, {line_index_after_removal: canonical_section_type})
        """
        _BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*$", re.IGNORECASE)
        cleaned: List[Dict[str, Any]] = []
        markers: Dict[int, str] = {}
        pending_marker: Optional[str] = None

        for line in raw_lines:
            m = _BRACKET_RE.match(line["text"])
            if m:
                raw_label = m.group(1).strip().lower()
                # Strip trailing numbers: "verse 1" → "verse"
                base = re.sub(r"\s*\d+\s*$", "", raw_label).strip()
                canonical = _BRACKET_ALIASES.get(base) or _BRACKET_ALIASES.get(raw_label) or base
                pending_marker = canonical
            else:
                if pending_marker is not None:
                    markers[len(cleaned)] = pending_marker
                    pending_marker = None
                cleaned.append(line)

        # If a marker was at the very end with no following lines
        # just discard it (nothing to label)

        return cleaned, markers

    # -----------------------------------------------------------------------
    # Step 6 — Build sections + units (type-specific)
    # -----------------------------------------------------------------------

    def _build_structure(
        self,
        raw_lines: List[Dict[str, Any]],
        bracket_markers: Dict[int, str],
        input_type: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        if input_type == TYPE_SONG:
            return self._structure_song(raw_lines, bracket_markers)
        elif input_type == TYPE_SCRIPT:
            return self._structure_script(raw_lines)
        elif input_type == TYPE_STORY:
            return self._structure_story(raw_lines)
        elif input_type == TYPE_DOCUMENT:
            return self._structure_document(raw_lines)
        else:  # audio / fallback
            return self._structure_plain(raw_lines)

    # --- SONG structure ---------------------------------------------------

    def _structure_song(
        self,
        raw_lines: List[Dict[str, Any]],
        bracket_markers: Dict[int, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        has_explicit = bool(bracket_markers)

        if has_explicit:
            return self._song_from_brackets(raw_lines, bracket_markers)
        else:
            return self._song_inferred(raw_lines)

    def _song_from_brackets(
        self,
        raw_lines: List[Dict[str, Any]],
        bracket_markers: Dict[int, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build sections directly from explicit bracket markers."""
        sections: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        u_idx = 0
        s_idx = 0

        # Determine where each section starts/ends
        marker_positions = sorted(bracket_markers.keys())
        # Add sentinel end
        boundaries = marker_positions + [len(raw_lines)]

        type_counters: Counter = Counter()

        for i, pos in enumerate(marker_positions):
            section_type = bracket_markers[pos]
            type_counters[section_type] += 1
            count = type_counters[section_type]
            label = f"{section_type.replace('_', '-').title()} {count}" if count > 1 else section_type.replace('_', '-').title()

            s_idx += 1
            section_id = f"s{s_idx:03d}"
            section_unit_ids: List[str] = []

            end = boundaries[i + 1]
            for j in range(pos, end):
                line = raw_lines[j]
                if not line["text"].strip():
                    continue
                u_idx += 1
                uid = f"u{u_idx:03d}"
                units.append({
                    "id":           uid,
                    "section_id":   section_id,
                    "index":        u_idx,
                    "text":         line["text"],
                    "start_time":   line.get("start_time"),
                    "end_time":     line.get("end_time"),
                    "speaker":      None,
                    "is_inferred":  False,
                })
                section_unit_ids.append(uid)

            if section_unit_ids:
                sections.append({
                    "id":          section_id,
                    "type":        section_type,
                    "label":       label,
                    "is_inferred": False,
                    "unit_ids":    section_unit_ids,
                    "repeat_of":   None,
                })

        # Lines before the first marker (if any) → inferred intro
        if marker_positions and marker_positions[0] > 0:
            intro_unit_ids: List[str] = []
            s_idx += 1
            section_id = f"s{s_idx:03d}"  # will be renumbered below
            for j in range(0, marker_positions[0]):
                line = raw_lines[j]
                if not line["text"].strip():
                    continue
                u_idx += 1
                uid = f"u{u_idx:03d}"
                units.insert(j, {
                    "id":          uid,
                    "section_id":  section_id,
                    "index":       u_idx,
                    "text":        line["text"],
                    "start_time":  line.get("start_time"),
                    "end_time":    line.get("end_time"),
                    "speaker":     None,
                    "is_inferred": True,
                })
                intro_unit_ids.append(uid)
            if intro_unit_ids:
                sections.insert(0, {
                    "id":          section_id,
                    "type":        "intro",
                    "label":       "Intro",
                    "is_inferred": True,
                    "unit_ids":    intro_unit_ids,
                    "repeat_of":   None,
                })

        # Renumber everything cleanly
        sections, units = self._renumber(sections, units)
        return sections, units

    def _song_inferred(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Infer song structure from blank-line stanza groupings + repetition.
        """
        # Group into stanzas by blank lines
        stanzas: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for line in raw_lines:
            if line.get("blank_before") and current:
                stanzas.append(current)
                current = []
            current.append(line)
        if current:
            stanzas.append(current)

        if not stanzas:
            return [], []

        # Fingerprint each stanza
        def fingerprint(stanza: List[Dict[str, Any]]) -> str:
            return self._norm(" ".join(l["text"] for l in stanza))

        fps = [fingerprint(s) for s in stanzas]
        fp_counts = Counter(fps)

        # Classify each stanza
        section_types: List[str] = []
        verse_count = 0
        chorus_assigned_fp: Optional[str] = None

        for i, (stanza, fp) in enumerate(zip(stanzas, fps)):
            count = fp_counts[fp]
            n = len(stanzas)

            if count >= 2:
                # Repeating → chorus (use the first repeating fingerprint as chorus)
                if chorus_assigned_fp is None:
                    chorus_assigned_fp = fp
                if fp == chorus_assigned_fp:
                    section_types.append("chorus")
                else:
                    section_types.append("refrain")
            elif i == 0 and len(stanza) <= 2:
                section_types.append("intro")
            elif i == n - 1 and len(stanza) <= 2 and count == 1:
                section_types.append("outro")
            else:
                # Check if sandwiched between two chorus occurrences → bridge
                before_chorus = any(fps[j] == chorus_assigned_fp for j in range(i))
                after_chorus  = any(fps[j] == chorus_assigned_fp for j in range(i + 1, n))
                if before_chorus and after_chorus and chorus_assigned_fp and len(stanza) <= 3:
                    section_types.append("bridge")
                else:
                    section_types.append("verse")

        # Build sections + units
        sections: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        type_counters: Counter = Counter()
        u_idx = 0
        s_idx = 0

        for stanza, stype in zip(stanzas, section_types):
            type_counters[stype] += 1
            count = type_counters[stype]
            label = f"{stype.replace('_', '-').title()} {count}" if count > 1 else stype.replace('_', '-').title()

            s_idx += 1
            section_id = f"s{s_idx:03d}"
            section_unit_ids: List[str] = []

            for line in stanza:
                if not line["text"].strip():
                    continue
                u_idx += 1
                uid = f"u{u_idx:03d}"
                units.append({
                    "id":          uid,
                    "section_id":  section_id,
                    "index":       u_idx,
                    "text":        line["text"],
                    "start_time":  line.get("start_time"),
                    "end_time":    line.get("end_time"),
                    "speaker":     None,
                    "is_inferred": True,
                })
                section_unit_ids.append(uid)

            if section_unit_ids:
                sections.append({
                    "id":          section_id,
                    "type":        stype,
                    "label":       label,
                    "is_inferred": True,
                    "unit_ids":    section_unit_ids,
                    "repeat_of":   None,
                })

        return sections, units

    # --- SCRIPT structure -------------------------------------------------

    def _structure_script(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sections: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        u_idx = 0
        s_idx = 0

        current_scene_lines: List[Dict[str, Any]] = []
        current_scene_label = "Scene 1"
        current_scene_type  = "scene"
        scene_count = 0
        transition_holder: List[Optional[str]] = [None]  # [current_transition]

        def flush_scene():
            nonlocal s_idx
            if not current_scene_lines:
                return
            s_idx += 1
            section_id = f"s{s_idx:03d}"
            section_unit_ids = [u["id"] for u in current_scene_lines]
            for u in current_scene_lines:
                u["section_id"] = section_id
            sections.append({
                "id":               section_id,
                "type":             current_scene_type,
                "label":            current_scene_label,
                "is_inferred":      False,
                "unit_ids":         section_unit_ids,
                "repeat_of":        None,
                "scene_transition": transition_holder[0],
            })
            current_scene_lines.clear()
            transition_holder[0] = None

        pending_character: Optional[str] = None
        act_count = 0

        for line in raw_lines:
            text = line["text"].strip()
            if not text:
                continue

            # Scene transition (FADE TO, CUT TO, DISSOLVE, etc.) — label only
            if _TRANSITION_RE.match(text):
                transition_holder[0] = re.sub(r"\s+", " ", text.upper().strip(" :."))
                continue

            # Act marker
            if re.match(r"^ACT\s+[IVX\d]+", text, re.IGNORECASE):
                flush_scene()
                act_count += 1
                current_scene_label = text
                current_scene_type  = "act"
                pending_character   = None
                continue

            # Scene header
            if _SCENE_HEADER_RE.match(text):
                flush_scene()
                scene_count += 1
                current_scene_label = text[:80]
                current_scene_type  = "scene"
                pending_character   = None
                continue

            # Stage direction
            if _STAGE_DIRECTION_RE.match(text):
                u_idx += 1
                uid = f"u{u_idx:03d}"
                units.append({
                    "id":          uid,
                    "section_id":  None,
                    "index":       u_idx,
                    "text":        text,
                    "start_time":  line.get("start_time"),
                    "end_time":    line.get("end_time"),
                    "speaker":     None,
                    "speaker_type": None,
                    "unit_type":   "stage_direction",
                    "is_inferred": False,
                })
                current_scene_lines.append(units[-1])
                continue

            # Character name (ALL CAPS short line)
            if _CHARACTER_NAME_RE.match(text) and len(text.split()) <= 5:
                pending_character = text.strip()
                continue

            # Inline dialogue "CHAR: text"
            m = _INLINE_DIALOGUE_RE.match(text)
            if m:
                speaker = m.group(1).strip()
                dialogue = m.group(2).strip()
                u_idx += 1
                uid = f"u{u_idx:03d}"
                units.append({
                    "id":           uid,
                    "section_id":   None,
                    "index":        u_idx,
                    "text":         dialogue,
                    "start_time":   line.get("start_time"),
                    "end_time":     line.get("end_time"),
                    "speaker":      speaker,
                    "speaker_type": "narrator" if _NARRATOR_RE.match(speaker) else "character",
                    "unit_type":    "dialogue",
                    "is_inferred":  False,
                })
                current_scene_lines.append(units[-1])
                pending_character = None
                continue

            # Regular line (dialogue after character name, or action)
            u_idx += 1
            uid = f"u{u_idx:03d}"
            unit_type = "dialogue" if pending_character else "action"
            spk_type: Optional[str] = None
            if pending_character:
                spk_type = "narrator" if _NARRATOR_RE.match(pending_character) else "character"
            units.append({
                "id":           uid,
                "section_id":   None,
                "index":        u_idx,
                "text":         text,
                "start_time":   line.get("start_time"),
                "end_time":     line.get("end_time"),
                "speaker":      pending_character,
                "speaker_type": spk_type,
                "unit_type":    unit_type,
                "is_inferred":  False,
            })
            current_scene_lines.append(units[-1])
            if pending_character:
                pending_character = None  # character name applies to ONE block

        flush_scene()

        # If no scenes detected, treat whole thing as one scene
        if not sections:
            s_idx += 1
            section_id = f"s{s_idx:03d}"
            for u in units:
                u["section_id"] = section_id
            sections.append({
                "id":          section_id,
                "type":        "scene",
                "label":       "Scene 1",
                "is_inferred": True,
                "unit_ids":    [u["id"] for u in units],
                "repeat_of":   None,
            })

        return sections, units

    # --- STORY structure --------------------------------------------------

    def _structure_story(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sections: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        u_idx = 0
        s_idx = 0

        # Group by blank lines → paragraphs
        paras: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for line in raw_lines:
            if line.get("blank_before") and current:
                paras.append(current)
                current = []
            current.append(line)
        if current:
            paras.append(current)

        para_count = 0
        for para in paras:
            para_count += 1
            s_idx += 1
            section_id = f"s{s_idx:03d}"
            section_unit_ids: List[str] = []

            for line in para:
                if not line["text"].strip():
                    continue
                u_idx += 1
                uid = f"u{u_idx:03d}"
                # Simple dialogue detection: line starting with a quote or
                # ending with attribution
                is_dialogue = (
                    line["text"].startswith('"') or
                    line["text"].startswith('"') or
                    bool(re.search(r'\b(said|asked|replied|whispered|shouted)\b', line["text"], re.IGNORECASE))
                )
                units.append({
                    "id":          uid,
                    "section_id":  section_id,
                    "index":       u_idx,
                    "text":        line["text"],
                    "start_time":  line.get("start_time"),
                    "end_time":    line.get("end_time"),
                    "speaker":     None,
                    "unit_type":   "dialogue" if is_dialogue else "narrative",
                    "is_inferred": False,
                })
                section_unit_ids.append(uid)

            if section_unit_ids:
                sections.append({
                    "id":          section_id,
                    "type":        "paragraph",
                    "label":       f"Paragraph {para_count}",
                    "is_inferred": False,
                    "unit_ids":    section_unit_ids,
                    "repeat_of":   None,
                })

        return sections, units

    # --- DOCUMENT structure -----------------------------------------------

    def _structure_document(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sections: List[Dict[str, Any]] = []
        units: List[Dict[str, Any]] = []
        u_idx = 0
        s_idx = 0

        current_heading = "Section 1"
        current_unit_ids: List[str] = []
        section_count = 0

        def flush_section():
            nonlocal s_idx, section_count
            if not current_unit_ids:
                return
            section_count += 1
            s_idx += 1
            section_id = f"s{s_idx:03d}"
            for u in units:
                if u["id"] in current_unit_ids:
                    u["section_id"] = section_id
            sections.append({
                "id":          section_id,
                "type":        "section",
                "label":       current_heading,
                "is_inferred": False,
                "unit_ids":    list(current_unit_ids),
                "repeat_of":   None,
            })
            current_unit_ids.clear()

        for line in raw_lines:
            text = line["text"].strip()
            if not text:
                continue
            if _DOC_HEADING_RE.match(text):
                flush_section()
                current_heading = text[:80]
                continue
            u_idx += 1
            uid = f"u{u_idx:03d}"
            units.append({
                "id":          uid,
                "section_id":  None,
                "index":       u_idx,
                "text":        text,
                "start_time":  line.get("start_time"),
                "end_time":    line.get("end_time"),
                "speaker":     None,
                "unit_type":   "item",
                "is_inferred": False,
            })
            current_unit_ids.append(uid)

        flush_section()

        if not sections and units:
            s_idx += 1
            section_id = f"s{s_idx:03d}"
            for u in units:
                u["section_id"] = section_id
            sections.append({
                "id":          section_id,
                "type":        "section",
                "label":       "Content",
                "is_inferred": True,
                "unit_ids":    [u["id"] for u in units],
                "repeat_of":   None,
            })

        return sections, units

    # --- Plain fallback ---------------------------------------------------

    def _structure_plain(
        self, raw_lines: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        units: List[Dict[str, Any]] = []
        for i, line in enumerate(raw_lines, 1):
            if line["text"].strip():
                units.append({
                    "id":          f"u{i:03d}",
                    "section_id":  "s001",
                    "index":       i,
                    "text":        line["text"],
                    "start_time":  line.get("start_time"),
                    "end_time":    line.get("end_time"),
                    "speaker":     None,
                    "unit_type":   "line",
                    "is_inferred": False,
                })
        sections = [{
            "id":          "s001",
            "type":        "section",
            "label":       "Content",
            "is_inferred": True,
            "unit_ids":    [u["id"] for u in units],
            "repeat_of":   None,
        }] if units else []
        return sections, units

    # -----------------------------------------------------------------------
    # Step 7 — Attach timing from Whisper timed_segments
    # -----------------------------------------------------------------------

    def _attach_timing(
        self,
        units: List[Dict[str, Any]],
        timed_segments: List[Dict[str, Any]],
        audio_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not timed_segments or not units:
            return units

        # Align segments to units by order (best-effort; lengths may differ)
        for i, unit in enumerate(units):
            if i < len(timed_segments):
                seg = timed_segments[i]
                if unit.get("start_time") is None:
                    try:
                        unit["start_time"] = float(seg.get("start") or 0.0)
                    except (TypeError, ValueError):
                        pass
                if unit.get("end_time") is None:
                    try:
                        unit["end_time"] = float(seg.get("end") or 0.0)
                    except (TypeError, ValueError):
                        pass

        return units

    # -----------------------------------------------------------------------
    # Step 8 — Repetition map
    # -----------------------------------------------------------------------

    def _build_repetition_map(
        self,
        units: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Unit-level: which units share identical content
        unit_normed: Dict[str, str] = {u["id"]: self._norm(u["text"]) for u in units}
        norm_to_ids: Dict[str, List[str]] = defaultdict(list)
        for uid, norm in unit_normed.items():
            if norm:
                norm_to_ids[norm].append(uid)

        unit_repetitions: Dict[str, List[str]] = {}
        for uid in unit_normed:
            norm = unit_normed[uid]
            peers = [x for x in norm_to_ids.get(norm, []) if x != uid]
            if peers:
                unit_repetitions[uid] = peers

        # Section-level: which sections share content fingerprint
        def section_fingerprint(sec: Dict[str, Any]) -> str:
            sec_units = [u for u in units if u["id"] in sec["unit_ids"]]
            return self._norm(" ".join(u["text"] for u in sec_units))

        sec_fp: Dict[str, str] = {s["id"]: section_fingerprint(s) for s in sections}
        fp_to_sec: Dict[str, List[str]] = defaultdict(list)
        for sid, fp in sec_fp.items():
            if fp:
                fp_to_sec[fp].append(sid)

        section_repetitions: Dict[str, List[str]] = {}
        for sid in sec_fp:
            fp = sec_fp[sid]
            peers = [x for x in fp_to_sec.get(fp, []) if x != sid]
            if peers:
                section_repetitions[sid] = peers
                # Back-fill repeat_of on sections
                for sec in sections:
                    if sec["id"] == sid and sec.get("repeat_of") is None:
                        sec["repeat_of"] = peers[0]

        return {
            "unit_repetitions":    unit_repetitions,
            "section_repetitions": section_repetitions,
            "repeated_section_ids": list({
                sid for ids in section_repetitions.values() for sid in ids
            } | set(section_repetitions.keys())),
        }

    # -----------------------------------------------------------------------
    # Step 9 — Speaker detection
    # -----------------------------------------------------------------------

    def _detect_speakers(
        self, units: List[Dict[str, Any]], input_type: str
    ) -> Dict[str, Any]:
        speaker_map: Dict[str, List[str]] = defaultdict(list)
        speaker_types: Dict[str, str] = {}
        for u in units:
            sp = u.get("speaker")
            if sp:
                speaker_map[sp].append(u["id"])
                if sp not in speaker_types:
                    spk_type = u.get("speaker_type")
                    if spk_type is None:
                        spk_type = "narrator" if _NARRATOR_RE.match(sp) else "character"
                    speaker_types[sp] = spk_type
        return {
            "speakers":      dict(speaker_map),
            "speaker_types": speaker_types,
        }

    # -----------------------------------------------------------------------
    # Step 8.5 — Lyrical pattern detection (songs only)
    # -----------------------------------------------------------------------

    def _detect_lyrical_patterns(
        self,
        sections: List[Dict[str, Any]],
        units: List[Dict[str, Any]],
        input_type: str,
    ) -> Dict[str, Any]:
        """Lightweight structural detection only — no poetic analysis."""
        if input_type != TYPE_SONG:
            return {}

        unit_by_id = {u["id"]: u for u in units}

        def end_word(text: str) -> str:
            words = re.sub(r"[^a-z\s]", "", text.lower()).split()
            return words[-1] if words else ""

        def rhymes(a: str, b: str) -> bool:
            if not a or not b or a == b:
                return False
            # Require min length 4 so short function words (her, the, of)
            # don't create false matches via shared suffix
            if len(a) < 4 or len(b) < 4:
                return False
            # Share last 2+ chars (vowel nucleus + coda)
            return a[-2:] == b[-2:]

        rhyme_detected = False
        for sec in sections:
            end_words = [
                end_word(unit_by_id[uid]["text"])
                for uid in sec.get("unit_ids", [])
                if uid in unit_by_id
            ]
            # Check any pair of adjacent or nearby lines for rhyme
            for i in range(len(end_words) - 1):
                if rhymes(end_words[i], end_words[i + 1]):
                    rhyme_detected = True
                    break
                if i + 2 < len(end_words) and rhymes(end_words[i], end_words[i + 2]):
                    rhyme_detected = True
                    break
            if rhyme_detected:
                break

        return {"rhyme_repetition_detected": rhyme_detected}

    # -----------------------------------------------------------------------
    # Step 10 — Uncertainties
    # -----------------------------------------------------------------------

    def _flag_uncertainties(
        self,
        units: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        input_type: str,
        clean_text: str,
        bracket_markers: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []

        if not clean_text.strip():
            flags.append({"code": "empty_input", "detail": "No text content found."})
            return flags

        inferred_sections = [s for s in sections if s.get("is_inferred")]
        if inferred_sections and not bracket_markers:
            flags.append({
                "code":   "structure_inferred",
                "detail": f"{len(inferred_sections)} section(s) labelled by inference — no explicit markers found.",
            })

        if not sections:
            flags.append({"code": "no_sections", "detail": "Could not identify any sections."})

        # Mixed language
        # (languages detected separately; flag if more than one non-latin script)
        langs = self._detect_languages(clean_text)
        if len(langs) > 1:
            flags.append({
                "code":   "mixed_language",
                "detail": f"Multiple scripts detected: {', '.join(langs)}",
            })

        if input_type == TYPE_SONG and not any(
            s["type"] in ("chorus", "refrain") for s in sections
        ):
            flags.append({
                "code":   "no_chorus_detected",
                "detail": "No chorus or repeating section was identified.",
            })

        return flags

    # -----------------------------------------------------------------------
    # Step 11 — Timing summary
    # -----------------------------------------------------------------------

    def _build_timing_summary(
        self,
        audio_meta: Dict[str, Any],
        timed_segments: List[Dict[str, Any]],
        units: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        bpm           = audio_meta.get("bpm")
        duration_s    = audio_meta.get("duration_seconds")
        timed_count   = sum(1 for u in units if u.get("start_time") is not None)
        total_units   = len(units)

        last_end: Optional[float] = None
        for u in reversed(units):
            if u.get("end_time") is not None:
                last_end = u["end_time"]
                break

        return {
            "bpm":                bpm,
            "duration_seconds":   duration_s,
            "timed_units":        timed_count,
            "total_units":        total_units,
            "coverage":           round(timed_count / total_units, 2) if total_units else 0.0,
            "inferred_end_s":     last_end,
        }

    def _compact_audio_meta(self, audio_meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "bpm", "duration_seconds", "vocal_gender", "vocal_f0_hz",
            "vocal_gender_confidence", "energy_profile", "brightness_profile",
            "silence_ratio",
        ]
        return {k: audio_meta.get(k) for k in keys if audio_meta.get(k) is not None}

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _renumber(
        self,
        sections: List[Dict[str, Any]],
        units: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assign clean sequential IDs after any insertions."""
        # Renumber sections
        old_to_new_s: Dict[str, str] = {}
        for i, sec in enumerate(sections, 1):
            new_id = f"s{i:03d}"
            old_to_new_s[sec["id"]] = new_id
            sec["id"] = new_id

        # Renumber units and update their section_id references
        for i, unit in enumerate(units, 1):
            unit["index"] = i
            unit["id"] = f"u{i:03d}"
            old_sec = unit.get("section_id")
            if old_sec and old_sec in old_to_new_s:
                unit["section_id"] = old_to_new_s[old_sec]

        # Rebuild each section's unit_ids from the units that reference it
        unit_by_section: Dict[str, List[str]] = defaultdict(list)
        for unit in units:
            sid = unit.get("section_id")
            if sid:
                unit_by_section[sid].append(unit["id"])

        for sec in sections:
            sec["unit_ids"] = unit_by_section.get(sec["id"], [])

        return sections, units

    def _norm(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u0600-\u06FF\u0A00-\u0A7F\u0900-\u097F]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


# ===========================================================================
# Module-level helper: LLM-powered lyric resegmentation
# ===========================================================================

_CANONICAL_SECTION_TYPES = frozenset({
    "intro", "verse", "chorus", "pre_chorus", "post_chorus",
    "bridge", "hook", "refrain", "interlude", "outro",
})


def llm_resegment_lyrics(
    input_packet: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    """Re-segment a collapsed single-section song into proper named sections.

    Called from ``_stage0_job`` in pipeline_worker.py when
    ``InputProcessor._song_inferred()`` collapses everything into one giant
    inferred section — which happens when lyrics are pasted without blank
    lines between stanzas (the norm for Punjabi/Urdu/Hindi songs copied from
    lyric websites).

    Sends the numbered lyric lines to GPT-4o-mini and asks it to return
    canonical section boundaries.  The response is validated strictly:

    - Every unit (1 … N) must be covered exactly once (no gaps, no overlaps)
    - At least 2 sections must be returned to justify the resegmentation
    - Any section type not in ``_CANONICAL_SECTION_TYPES`` is mapped to
      "verse" so downstream stages always receive known types

    On any failure (network, parse error, validation failure) the function
    logs a warning and returns the original ``input_packet`` unchanged.  The
    pipeline never hard-stops here.

    Returns:
        The (possibly updated) ``input_packet`` dict.  If resegmentation
        succeeded, ``input_packet["resegmented_by_llm"]`` is ``True`` and
        ``input_packet["sections"]`` / ``input_packet["repetition_map"]``
        reflect the new structure.
    """
    sections = input_packet.get("sections") or []
    units    = input_packet.get("units") or []

    # Only act when blank-line inference collapsed everything into one section.
    if (len(sections) != 1
            or not sections[0].get("is_inferred")
            or len(units) < 10):
        return input_packet

    logger.info(
        "llm_resegment_lyrics: collapsed single-section detected "
        "(%d units) — calling GPT-4o-mini for structural analysis",
        len(units),
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        numbered_lines = "\n".join(
            f"{u['index']}. {u['text']}" for u in units
        )

        system_prompt = (
            "You are a music structure analyst. "
            "Your only task is to identify section boundaries in song lyrics.\n\n"
            "MAP any regional or language-specific section names to these canonical types:\n"
            "  intro       — opening lines before the first verse or chorus\n"
            "  verse       — antara, stanza, kuplet, mukhda (if not repeating)\n"
            "  chorus      — mukhda / hook / refrain — the main repeating hook\n"
            "  pre_chorus  — build-up before the chorus\n"
            "  bridge      — a contrasting section that usually appears once\n"
            "  interlude   — instrumental or transitional break\n"
            "  outro       — closing / fade-out lines\n\n"
            "RULES:\n"
            "1. Every lyric line MUST belong to exactly one section "
            "(no gaps, no overlaps).\n"
            "2. Identify repeated sections by comparing lyric content — "
            "repeated mukhda/chorus lines should be the same type.\n"
            "3. Use start_unit and end_unit as 1-based line numbers (inclusive).\n"
            "4. Return between 3 and 10 sections.\n"
            "5. Return STRICT JSON only — no prose outside the JSON object."
        )

        user_prompt = (
            f"Identify the section structure of these {len(units)} lyric lines.\n\n"
            f"LYRIC LINES:\n{numbered_lines}\n\n"
            f"Return this exact JSON format:\n"
            f'{{\n'
            f'  "sections": [\n'
            f'    {{"type": "intro",  "label": "Intro",   "start_unit": 1, "end_unit": 4}},\n'
            f'    {{"type": "verse",  "label": "Verse 1", "start_unit": 5, "end_unit": 12}},\n'
            f'    {{"type": "chorus", "label": "Chorus",  "start_unit": 13, "end_unit": 18}}\n'
            f'  ]\n'
            f'}}\n\n'
            f"IMPORTANT: The last section's end_unit MUST be {len(units)}. "
            f"Every line from 1 to {len(units)} must be covered."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw  = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        gpt_sections = data.get("sections") or []

        if not gpt_sections:
            logger.warning("llm_resegment_lyrics: GPT returned no sections — keeping original")
            return input_packet

        # ── Validate coverage ────────────────────────────────────────────────
        n = len(units)
        covered = [False] * (n + 1)  # 1-indexed

        for spec in gpt_sections:
            try:
                start = int(spec.get("start_unit") or 0)
                end   = int(spec.get("end_unit")   or 0)
            except (TypeError, ValueError):
                logger.warning("llm_resegment_lyrics: non-integer boundary — aborting")
                return input_packet

            if start < 1 or end > n or start > end:
                logger.warning(
                    "llm_resegment_lyrics: boundary out-of-range "
                    "start=%d end=%d n=%d — aborting",
                    start, end, n,
                )
                return input_packet

            for i in range(start, end + 1):
                if covered[i]:
                    logger.warning(
                        "llm_resegment_lyrics: overlap at unit %d — aborting", i
                    )
                    return input_packet
                covered[i] = True

        missing = [i for i in range(1, n + 1) if not covered[i]]
        if missing:
            logger.warning(
                "llm_resegment_lyrics: %d uncovered unit(s) (e.g. %s) — aborting",
                len(missing), missing[:5],
            )
            return input_packet

        if len(gpt_sections) < 2:
            logger.warning(
                "llm_resegment_lyrics: only 1 section returned — no improvement, keeping original"
            )
            return input_packet

        # ── Rebuild sections + update unit.section_id ────────────────────────
        units_by_index: Dict[int, Dict[str, Any]] = {u["index"]: u for u in units}
        type_counters: Counter = Counter()
        new_sections: List[Dict[str, Any]] = []

        for i, spec in enumerate(gpt_sections, 1):
            raw_type = (
                str(spec.get("type") or "verse")
                .lower().strip()
                .replace(" ", "_").replace("-", "_")
            )
            stype = raw_type if raw_type in _CANONICAL_SECTION_TYPES else "verse"
            type_counters[stype] += 1
            cnt   = type_counters[stype]

            gpt_label = str(spec.get("label") or "").strip()
            if not gpt_label:
                gpt_label = (
                    stype.replace("_", "-").title()
                    if cnt == 1
                    else f"{stype.replace('_', '-').title()} {cnt}"
                )
            label = gpt_label[:60]

            section_id      = f"s{i:03d}"
            start           = int(spec["start_unit"])
            end             = int(spec["end_unit"])
            section_unit_ids: List[str] = []

            for idx in range(start, end + 1):
                u = units_by_index.get(idx)
                if u:
                    u["section_id"] = section_id
                    section_unit_ids.append(u["id"])

            new_sections.append({
                "id":          section_id,
                "type":        stype,
                "label":       label,
                "is_inferred": True,
                "unit_ids":    section_unit_ids,
                "repeat_of":   None,
            })

        # ── Rebuild repetition map ────────────────────────────────────────────
        ip          = InputProcessor()
        new_rep_map = ip._build_repetition_map(units, new_sections)

        updated = dict(input_packet)
        updated["sections"]           = new_sections
        updated["units"]              = units
        updated["repetition_map"]     = new_rep_map
        updated["resegmented_by_llm"] = True

        logger.info(
            "llm_resegment_lyrics: resegmented %d units into %d sections: %s",
            n,
            len(new_sections),
            [f"{s['type']}({len(s['unit_ids'])}u)" for s in new_sections],
        )
        return updated

    except Exception:
        logger.exception(
            "llm_resegment_lyrics: unexpected error — returning original input_packet"
        )
        return input_packet
