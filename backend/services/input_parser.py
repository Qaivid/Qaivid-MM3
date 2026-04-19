"""
Input Parser Service
Deterministic text cleaning, section detection, language detection, content type classification.
Includes SRT parsing, timestamped lyrics parsing ported from Qaivid 1.0.
No AI calls - uses rules and patterns only.
"""
import re
from typing import Dict, List, Any, Optional, Tuple


# Language detection patterns (deterministic)
SCRIPT_PATTERNS = {
    "devanagari": re.compile(r'[\u0900-\u097F]'),
    "gurmukhi": re.compile(r'[\u0A00-\u0A7F]'),
    "arabic_urdu": re.compile(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]'),
    "latin": re.compile(r'[a-zA-Z]'),
}

SECTION_MARKERS = {
    "verse": re.compile(r'^\s*\[?\s*(verse|stanza|bait|sher|doha)\s*\d*\s*\]?\s*$', re.IGNORECASE),
    "chorus": re.compile(r'^\s*\[?\s*(chorus|hook|refrain|mukhda|pallavi|asthayi)\s*\]?\s*$', re.IGNORECASE),
    "bridge": re.compile(r'^\s*\[?\s*(bridge|interlude|antara)\s*\]?\s*$', re.IGNORECASE),
    "intro": re.compile(r'^\s*\[?\s*(intro|opening|alaap)\s*\]?\s*$', re.IGNORECASE),
    "outro": re.compile(r'^\s*\[?\s*(outro|closing|end)\s*\]?\s*$', re.IGNORECASE),
    "dialogue": re.compile(r'^\s*\[?\s*(dialogue|dialog|scene)\s*\d*\s*\]?\s*$', re.IGNORECASE),
    "performance": re.compile(r'^\s*\[?\s*(instrumental|music|taan|murki|sargam)\s*\]?\s*$', re.IGNORECASE),
    "pre_chorus": re.compile(r'^\s*\[?\s*(pre-chorus|pre chorus|build)\s*\]?\s*$', re.IGNORECASE),
    "drop": re.compile(r'^\s*\[?\s*(drop|breakdown)\s*\]?\s*$', re.IGNORECASE),
}

CONTENT_TYPE_SIGNALS = {
    "song": ["chorus", "verse", "hook", "mukhda", "pallavi", "refrain"],
    "ghazal": ["sher", "bait", "matla", "maqta", "radif", "qaafiya"],
    "qawwali": ["alaap", "taan", "murki", "sargam", "dam mast qalandar"],
    "poem": ["stanza", "couplet"],
    "script": ["int.", "ext.", "cut to", "fade in", "scene", "dialogue"],
    "story": [],
}

# SRT parsing (ported from Qaivid 1.0)
SRT_BLOCK_PATTERN = re.compile(
    r'(\d+)\s*\n(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*\n([\s\S]*?)(?=\n\n|\n*$)',
    re.MULTILINE
)

# Timestamped lyrics patterns (ported from Qaivid 1.0)
TIMESTAMP_PATTERNS = [
    re.compile(r'^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]\s*(.*)$'),
    re.compile(r'^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-\u2013]\s*(.*)$'),
    re.compile(r'^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s+(.+)$'),
]

# Script format detection
SCRIPT_LINE_PATTERN = re.compile(r'^(?:(?:INT|EXT|INT/EXT|I/E)[\s.]+.+|[A-Z][A-Z\s]+(?:\(.*?\))?\s*$)', re.MULTILINE)
DIALOGUE_PATTERN = re.compile(r'^([A-Z][A-Z\s]+):\s*(.+)$', re.MULTILINE)


def clean_text(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def detect_script(text: str) -> str:
    counts = {}
    for script_name, pattern in SCRIPT_PATTERNS.items():
        counts[script_name] = len(pattern.findall(text))
    if not any(counts.values()):
        return "unknown"
    return max(counts, key=counts.get)


def detect_language_hint(text: str, script: str) -> str:
    mapping = {
        "devanagari": "hindi",
        "gurmukhi": "punjabi",
        "arabic_urdu": "urdu",
        "latin": "english",
    }
    return mapping.get(script, "unknown")


def detect_input_format(text: str) -> str:
    """Detect whether input is SRT, timestamped lyrics, script, dialogue, or plain text."""
    if SRT_BLOCK_PATTERN.search(text):
        return "srt"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    ts_count = 0
    for line in lines:
        for pattern in TIMESTAMP_PATTERNS:
            if pattern.match(line.strip()):
                ts_count += 1
                break
    if ts_count > len(lines) * 0.3:
        return "timestamped_lyrics"
    if SCRIPT_LINE_PATTERN.search(text):
        return "script"
    dialogue_lines = [ln for ln in lines if DIALOGUE_PATTERN.match(ln.strip())]
    if len(dialogue_lines) > len(lines) * 0.3:
        return "dialogue"
    return "plain_text"


def parse_srt(text: str) -> List[Dict[str, Any]]:
    """Parse SRT subtitle format into segments with timestamps."""
    segments = []
    for match in SRT_BLOCK_PATTERN.finditer(text):
        start_time = _hhmmss_to_sec(match.group(2), match.group(3), match.group(4), match.group(5))
        end_time = _hhmmss_to_sec(match.group(6), match.group(7), match.group(8), match.group(9))
        line_text = re.sub(r'<[^>]+>', '', match.group(10)).strip()
        if line_text:
            segments.append({
                "index": len(segments),
                "text": line_text,
                "start_time": start_time,
                "end_time": end_time,
            })
    return segments


def parse_timestamped_lyrics(text: str) -> List[Dict[str, Any]]:
    """Parse timestamped lyrics (e.g., [0:15] line text)."""
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        matched = False
        # [MM:SS.ms] text
        m = re.match(r'^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]\s*(.*)$', line)
        if m:
            start = _mmss_to_sec(m.group(1), m.group(2), m.group(3))
            txt = (m.group(4) or "").strip()
            if txt:
                segments.append({"index": len(segments), "text": txt, "start_time": start})
            matched = True
        if not matched:
            m = re.match(r'^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-\u2013]\s*(.*)$', line) or \
                re.match(r'^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s+(.+)$', line)
            if m:
                start = _mmss_to_sec(m.group(1), m.group(2), m.group(3))
                txt = (m.group(4) or "").strip()
                if txt:
                    segments.append({"index": len(segments), "text": txt, "start_time": start})
                matched = True
        if not matched and line and not re.match(r'^\[.*\]$', line) and not re.match(r'^\d+$', line):
            segments.append({"index": len(segments), "text": line})
    # Fill end times from next segment's start
    for i in range(len(segments) - 1):
        if segments[i].get("start_time") is not None and segments[i + 1].get("start_time") is not None:
            segments[i]["end_time"] = segments[i + 1]["start_time"]
    return segments


def detect_content_type(text: str, lines: List[str]) -> str:
    text_lower = text.lower()
    scores = {}
    for ctype, signals in CONTENT_TYPE_SIGNALS.items():
        score = sum(1 for s in signals if s in text_lower)
        scores[ctype] = score
    if re.search(r'\b(INT\.|EXT\.)\b', text):
        return "script"
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if len(lines) < 6:
        return "poem"
    line_lengths = [len(ln) for ln in lines if ln.strip()]
    avg_len = sum(line_lengths) / max(len(line_lengths), 1)
    if avg_len > 80:
        return "story"
    return "song"


def detect_sections(lines: List[str]) -> List[Dict[str, Any]]:
    sections = []
    current_section = {"type": "verse", "start_line": 0, "lines": []}
    for i, line in enumerate(lines):
        stripped = line.strip()
        matched_section = None
        for sec_type, pattern in SECTION_MARKERS.items():
            if pattern.match(stripped):
                matched_section = sec_type
                break
        if matched_section:
            if current_section["lines"]:
                sections.append(current_section)
            current_section = {"type": matched_section, "start_line": i + 1, "lines": []}
        elif stripped:
            current_section["lines"].append({"index": i, "text": stripped})
        elif not stripped and current_section["lines"]:
            sections.append(current_section)
            current_section = {"type": "verse", "start_line": i + 1, "lines": []}
    if current_section["lines"]:
        sections.append(current_section)

    # Fallback: if only one large section was detected (no blank lines, no markers),
    # try to split by repetition — repeated line clusters signal chorus/refrain boundaries.
    if len(sections) == 1 and len(sections[0]["lines"]) > 10:
        split = _split_by_repetition(sections[0]["lines"])
        if len(split) > 1:
            return split

    return sections


def _split_by_repetition(line_objs: List[Dict]) -> List[Dict[str, Any]]:
    """
    Splits a single unsectioned block into verse/chorus sections by detecting
    repeated line clusters. A cluster of lines whose text recurs later in the
    piece is treated as a chorus. Everything between chorus clusters is a verse.
    Songs like Punjabi folk laments always repeat the refrain — this catches them.
    """
    # Build a frequency map on normalised text
    norm = [ln["text"].strip().lower() for ln in line_objs]
    freq: Dict[str, int] = {}
    for t in norm:
        if t:
            freq[t] = freq.get(t, 0) + 1

    # Mark each line as repeated (chorus candidate) or unique (verse candidate)
    is_repeated = [bool(t and freq.get(t, 0) > 1) for t in norm]

    sections: List[Dict[str, Any]] = []
    current_type: Optional[str] = None
    current_lines: List[Dict] = []
    current_start = 0

    for i, ln in enumerate(line_objs):
        ltype = "chorus" if is_repeated[i] else "verse"
        if ltype != current_type:
            if current_lines:
                sections.append({
                    "type": current_type,
                    "start_line": current_start,
                    "lines": current_lines,
                })
            current_type = ltype
            current_start = ln.get("index", i)
            current_lines = [ln]
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append({
            "type": current_type,
            "start_line": current_start,
            "lines": current_lines,
        })

    # Merge tiny fragments (< 2 lines) into their neighbour to avoid noise
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and len(sec["lines"]) < 2:
            merged[-1]["lines"].extend(sec["lines"])
        else:
            merged.append(sec)

    return merged if len(merged) > 1 else sections


def detect_repeated_lines(lines: List[str]) -> List[Dict[str, Any]]:
    clean_lines = [ln.strip().lower() for ln in lines if ln.strip()]
    seen = {}
    for i, line in enumerate(clean_lines):
        if line in seen:
            seen[line]["count"] += 1
            seen[line]["indices"].append(i)
        else:
            seen[line] = {"count": 1, "indices": [i], "text": lines[i].strip() if i < len(lines) else line}
    return [v for v in seen.values() if v["count"] > 1]


def parse_input(raw_text: str) -> Dict[str, Any]:
    cleaned = clean_text(raw_text)
    all_lines = cleaned.split('\n')
    content_lines = [ln for ln in all_lines if ln.strip()]
    script = detect_script(cleaned)
    language = detect_language_hint(cleaned, script)
    input_format = detect_input_format(cleaned)
    content_type = detect_content_type(cleaned, content_lines)

    # Parse based on detected format
    timed_segments = None
    has_timestamps = False
    if input_format == "srt":
        timed_segments = parse_srt(cleaned)
        has_timestamps = True
    elif input_format == "timestamped_lyrics":
        timed_segments = parse_timestamped_lyrics(cleaned)
        has_timestamps = any(s.get("start_time") is not None for s in timed_segments)

    sections = detect_sections(all_lines)
    repeated = detect_repeated_lines(all_lines)

    line_objects = []
    for i, line in enumerate(content_lines):
        line_objects.append({
            "index": i,
            "text": line.strip(),
            "is_repeated": any(line.strip().lower() == r["text"].lower() for r in repeated),
            "char_count": len(line.strip()),
        })

    return {
        "cleaned_text": cleaned,
        "detected_language": language,
        "detected_script": script,
        "detected_type": content_type,
        "input_format": input_format,
        "has_timestamps": has_timestamps,
        "timed_segments": timed_segments,
        "sections": sections,
        "lines": line_objects,
        "line_count": len(content_lines),
        "repeated_lines": repeated,
        "metadata": {
            "total_characters": len(cleaned),
            "total_words": len(cleaned.split()),
            "has_section_markers": any(
                any(p.match(ln.strip()) for p in SECTION_MARKERS.values())
                for ln in all_lines
            ),
            "symbolic_density": "unknown",
            "input_format": input_format,
        }
    }


def _mmss_to_sec(mm: str, ss: str, ms: Optional[str] = None) -> float:
    m = int(mm)
    s = int(ss)
    millis = int((ms or "0").ljust(3, "0")) if ms else 0
    return m * 60 + s + millis / 1000


def _hhmmss_to_sec(hh: str, mm: str, ss: str, ms: Optional[str] = None) -> float:
    h = int(hh)
    m = int(mm)
    s = int(ss)
    millis = int((ms or "0").ljust(3, "0")) if ms else 0
    return h * 3600 + m * 60 + s + millis / 1000
