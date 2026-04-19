"""
Pre-LLM Enrichment Service
Applies deterministic symbolic/literal detection to assign visualization modes
BEFORE the LLM call, reducing AI workload.
Uses patterns ported from Qaivid 1.0 semantic-analyzer.ts.
"""
import re
from typing import Dict, List, Any
from services.deterministic_rules import (
    SYMBOLIC_INDICATORS, LITERAL_INDICATORS,
    COMMON_MOTIFS, EMOTIONAL_SHIFTS,
    detect_emotional_shift, detect_symbolic_density, infer_setting,
)


def pre_enrich_lines(lines: List[Dict[str, Any]], culture_pack_id: str = "") -> List[Dict[str, Any]]:
    """
    Pre-assign visualization modes and emotional hints to lines
    before sending to LLM. This reduces the LLM's workload and
    provides it with better-structured input.
    """
    enriched = []
    for line in lines:
        text = line.get("text", "")
        if not text.strip():
            enriched.append(line)
            continue

        # Detect visualization mode
        viz_mode = _detect_viz_mode(text)

        # Detect emotional shift
        emotion = detect_emotional_shift(text)

        # Detect motifs
        motifs = _detect_motifs_in_line(text)

        # Detect setting
        setting = infer_setting(text)

        # Symbolic density
        sym_density = detect_symbolic_density(text)

        enriched.append({
            **line,
            "pre_viz_mode": viz_mode,
            "pre_emotion": emotion,
            "pre_motifs": motifs,
            "pre_setting": setting if setting != "unspecified" else None,
            "symbolic_density": sym_density,
        })

    return enriched


def _detect_viz_mode(text: str) -> str:
    """
    Deterministically classify visualization mode:
    - direct: literal action/scene description
    - indirect: implied meaning through environment
    - symbolic: metaphorical/abstract imagery
    - absorbed: line shapes mood, not separate image
    - performance_only: musical notation/instruction
    """
    text_lower = text.lower().strip()

    # Performance markers
    if re.match(r'^\[?(instrumental|music|taan|murki|sargam|alaap)\]?$', text_lower, re.IGNORECASE):
        return "performance_only"

    # Count symbolic and literal indicators
    sym_score = sum(1 for p in SYMBOLIC_INDICATORS if re.search(p, text_lower, re.IGNORECASE))
    lit_score = sum(1 for p in LITERAL_INDICATORS if re.search(p, text_lower, re.IGNORECASE))

    # Very short lines (< 3 words) are often absorbed
    word_count = len(text.split())
    if word_count <= 2:
        return "absorbed"

    # Clear symbolic content
    if sym_score >= 2 and lit_score == 0:
        return "symbolic"
    if sym_score >= 1 and lit_score == 0:
        return "indirect"

    # Mixed
    if sym_score >= 1 and lit_score >= 1:
        return "indirect"

    # Clear literal
    if lit_score >= 2:
        return "direct"
    if lit_score >= 1:
        return "direct"

    # Default: let LLM decide
    return "auto"


def _detect_motifs_in_line(text: str) -> List[str]:
    """Detect common motifs in a single line."""
    text_lower = text.lower()
    found = []
    for motif in COMMON_MOTIFS:
        pattern = r'\b' + re.escape(motif) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            found.append(motif)
    return found


def build_pre_enrichment_context(enriched_lines: List[Dict]) -> str:
    """
    Build a summary of pre-enrichment to include in the LLM prompt,
    so the LLM can confirm or override the deterministic assignments.
    """
    hints = []
    for line in enriched_lines:
        if line.get("pre_viz_mode") and line["pre_viz_mode"] != "auto":
            hints.append(f"  Line {line.get('index', '?')}: pre-classified as {line['pre_viz_mode']}")
        if line.get("pre_motifs"):
            hints.append(f"  Line {line.get('index', '?')}: detected motifs: {', '.join(line['pre_motifs'])}")
        if line.get("pre_emotion") and line["pre_emotion"] != "neutral":
            hints.append(f"  Line {line.get('index', '?')}: emotional shift: {line['pre_emotion']}")
    if hints:
        return "PRE-ANALYSIS HINTS (confirm or override):\n" + "\n".join(hints[:30])
    return ""
