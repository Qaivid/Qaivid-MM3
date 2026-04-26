import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# CULTURE PACKS
# =============================================================================

class CulturePackRegistry:
    """
    Deterministic enrichment layer.

    Purpose:
    - infer likely cultural pack from trigger words
    - inject metaphor hints
    - provide world defaults
    - provide appearance defaults
    - provide visual restrictions and misinterpretation guards
    """

    PACKS: Dict[str, Dict[str, Any]] = {
        "punjabi_rural_lament": {
            "triggers": [
                "pind", "vehra", "charpai", "phulkari", "poh", "chetar", "vaisakh",
                "dupatta", "jhhanjar", "marvee", "khet", "sanjh", "chaadar"
            ],
            "metaphors": {
                "winter sun": "emotional distance, weak warmth, seasonal abandonment",
                "courtyard": "domestic emotional center, family memory, intimate lived space",
                "anklet": "feminine adornment, desire, youth, celebration, withheld fulfillment",
            },
            "world_defaults": {
                "geography": "Punjab region, South Asia",
                "cultural_dna": "Punjabi rural lament",
            },
            "visual_restrictions": [
                "Avoid generic Western suburban visuals.",
                "Do not render speakers with East Asian, European, or African features unless the text explicitly demands it.",
            ],
            "common_misinterpretations": [
                "Do not treat agrarian or seasonal references as generic decoration.",
                "Do not flatten feminine domestic imagery into random rustic props.",
                "Do not substitute a generic rural Indian village aesthetic — Punjab has its own distinct cultural world; let the story and location name guide the specific visual.",
            ],
        },
        "punjabi_diaspora_memory": {
            "triggers": [
                "pardes", "london", "vilayat", "foreign", "distance", "missed call", "airport"
            ],
            "metaphors": {
                "phone silence": "emotional abandonment, distance, suspended bond",
                "abroad": "separation, economic duty, migration ache, emotional split",
            },
            "world_defaults": {
                "geography": "diaspora split between homeland and abroad",
                "cultural_dna": "Punjabi diaspora memory",
            },
            "visual_restrictions": [
                "Do not erase the tension between homeland memory and present migration reality.",
                "Do not render speakers with East Asian, European, or African features unless the text explicitly demands it.",
            ],
            "common_misinterpretations": [
                "Do not make diaspora purely glamorous or purely urban without emotional contrast.",
            ],
        },
        "urdu_philosophical_ghazal": {
            "triggers": [
                "ishq", "hijr", "vasl", "dil", "khudi", "wajood", "fana", "safar", "khamoshi"
            ],
            "metaphors": {
                "heart": "inner self, emotional consciousness, metaphysical vulnerability",
                "journey": "existential passage, inner becoming, loss or seeking",
                "silence": "unspoken intensity, metaphysical restraint, unresolved interiority",
            },
            "world_defaults": {
                "cultural_dna": "Urdu philosophical ghazal",
                "geography": "Urdu cultural sphere (North India / Pakistan)",
            },
            "visual_restrictions": [
                "Do not literalize abstract ghazal language into crude plot imagery.",
                "Do not render speakers with East Asian, European, or African features unless the text explicitly demands it.",
            ],
            "common_misinterpretations": [
                "Do not assume every romantic image is a literal relationship scene.",
            ],
        },
        "devotional_qawwali": {
            "triggers": [
                "ali", "maula", "allah", "hussain", "karam", "darbar", "zikr", "ishq-e-haqiqi"
            ],
            "metaphors": {
                "wine": "spiritual ecstasy, mystical absorption",
                "beloved": "divine beloved or sacred focus, not always romantic human figure",
                "door": "sacred threshold, longing for nearness, devotional seeking",
            },
            "world_defaults": {
                "cultural_dna": "devotional qawwali",
                "geography": "South Asian devotional space (dargah, shrine, gathering)",
            },
            "visual_restrictions": [
                "Avoid trivializing devotional intensity into nightclub or pop-romance imagery.",
                "Do not render speakers with East Asian, European, or African features unless the text explicitly demands it.",
            ],
            "common_misinterpretations": [
                "Do not force human romance interpretation onto devotional address.",
            ],
        },
        "universal": {
            "triggers": [],
            "metaphors": {},
            "world_defaults": {},
            "visual_restrictions": [
                "Do not impose any single culture's visual language unless the text demands it.",
                "Render characters, locations, and props faithful to the language and geography of the submitted content.",
            ],
            "common_misinterpretations": [
                "Do not default to South Asian, Western, or East Asian aesthetics without textual justification.",
            ],
        },
    }

    @classmethod
    def detect_pack(cls, text: str, explicit_pack: Optional[str] = None) -> Optional[str]:
        if explicit_pack and explicit_pack in cls.PACKS:
            return explicit_pack

        lowered = text.lower()
        scores: Dict[str, int] = {}

        for pack_id, pack in cls.PACKS.items():
            score = 0
            for trigger in pack.get("triggers", []):
                if trigger.lower() in lowered:
                    score += 1
            if score > 0:
                scores[pack_id] = score

        return max(scores, key=scores.get) if scores else None

    @classmethod
    def get_pack(cls, pack_id: Optional[str]) -> Dict[str, Any]:
        if pack_id and pack_id in cls.PACKS:
            return cls.PACKS[pack_id]
        return cls.PACKS["universal"]

    @classmethod
    def get_triggered_metaphors(cls, text: str, pack_id: Optional[str]) -> Dict[str, str]:
        pack = cls.get_pack(pack_id)
        metaphors = pack.get("metaphors", {})
        lowered = text.lower()

        active: Dict[str, str] = {}
        for trigger, meaning in metaphors.items():
            if trigger.lower() in lowered:
                active[trigger] = meaning
        return active


# =============================================================================
# STRUCTURED INPUT PARSER
# =============================================================================

class StructuredInputParser:
    """
    Cheap deterministic parser for:
    - plain text
    - SRT
    - time-coded text
    - simple structured dict / JSON-like input
    """

    def parse(self, raw_input: Any) -> Dict[str, Any]:
        if isinstance(raw_input, dict):
            return self._parse_dict_input(raw_input)

        if not isinstance(raw_input, str):
            raw_input = str(raw_input)

        text = raw_input.strip()

        if self._looks_like_json(text):
            try:
                parsed = json.loads(text)
                return self._parse_dict_input(parsed)
            except Exception:
                pass

        if self._looks_like_srt(text):
            return self._parse_srt(text)

        if self._looks_like_timecoded_text(text):
            return self._parse_timecoded_lines(text)

        return self._parse_plain_text(text)

    def _parse_plain_text(self, text: str) -> Dict[str, Any]:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        return {
            "source_format": "plain_text",
            "cleaned_text": "\n".join(lines),
            "lines": [
                {
                    "line_index": i + 1,
                    "text": line,
                    "start_time": None,
                    "end_time": None,
                    "section_label": None,
                    "annotation_tags": [],
                }
                for i, line in enumerate(lines)
            ],
            "sections": [],
        }

    def _parse_srt(self, text: str) -> Dict[str, Any]:
        blocks = re.split(r"\n\s*\n", text.strip())
        lines: List[Dict[str, Any]] = []
        idx = 1

        for block in blocks:
            block_lines = [x.strip() for x in block.splitlines() if x.strip()]
            if len(block_lines) < 2:
                continue

            time_line = None
            content_start = 0

            if re.match(r"^\d+$", block_lines[0]) and len(block_lines) >= 3:
                time_line = block_lines[1]
                content_start = 2
            elif "-->" in block_lines[0]:
                time_line = block_lines[0]
                content_start = 1

            if not time_line:
                continue

            start_time, end_time = self._split_time_range(time_line)
            content = " ".join(block_lines[content_start:]).strip()
            if not content:
                continue

            lines.append(
                {
                    "line_index": idx,
                    "text": content,
                    "start_time": start_time,
                    "end_time": end_time,
                    "section_label": None,
                    "annotation_tags": [],
                }
            )
            idx += 1

        cleaned_text = "\n".join(x["text"] for x in lines)
        return {
            "source_format": "srt",
            "cleaned_text": cleaned_text,
            "lines": lines,
            "sections": [],
        }

    def _parse_timecoded_lines(self, text: str) -> Dict[str, Any]:
        lines: List[Dict[str, Any]] = []
        idx = 1

        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            match = re.match(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\]?\s+(.*)$", raw_line)
            if match:
                start_time = match.group(1)
                content = match.group(2).strip()
                lines.append(
                    {
                        "line_index": idx,
                        "text": content,
                        "start_time": start_time,
                        "end_time": None,
                        "section_label": None,
                        "annotation_tags": [],
                    }
                )
            else:
                lines.append(
                    {
                        "line_index": idx,
                        "text": raw_line,
                        "start_time": None,
                        "end_time": None,
                        "section_label": None,
                        "annotation_tags": [],
                    }
                )
            idx += 1

        cleaned_text = "\n".join(x["text"] for x in lines)
        return {
            "source_format": "timecoded_text",
            "cleaned_text": cleaned_text,
            "lines": lines,
            "sections": [],
        }

    def _parse_dict_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lines: List[Dict[str, Any]] = []

        if isinstance(data.get("lines"), list):
            for i, item in enumerate(data["lines"], start=1):
                if isinstance(item, dict):
                    lines.append(
                        {
                            "line_index": item.get("line_index", i),
                            "text": str(item.get("text", "")).strip(),
                            "start_time": item.get("start_time"),
                            "end_time": item.get("end_time"),
                            "section_label": item.get("section_label"),
                            "annotation_tags": item.get("annotation_tags", []),
                        }
                    )
                else:
                    lines.append(
                        {
                            "line_index": i,
                            "text": str(item).strip(),
                            "start_time": None,
                            "end_time": None,
                            "section_label": None,
                            "annotation_tags": [],
                        }
                    )

        if not lines and isinstance(data.get("text"), str):
            return self._parse_plain_text(data["text"])

        cleaned_text = "\n".join(x["text"] for x in lines if x["text"])
        return {
            "source_format": "structured_dict",
            "cleaned_text": cleaned_text,
            "lines": lines,
            "sections": data.get("sections", []) if isinstance(data.get("sections"), list) else [],
        }

    def _looks_like_json(self, text: str) -> bool:
        return text.startswith("{") or text.startswith("[")

    def _looks_like_srt(self, text: str) -> bool:
        return "-->" in text and bool(re.search(r"\d{2}:\d{2}:\d{2}[,\.]\d{2,3}", text))

    def _looks_like_timecoded_text(self, text: str) -> bool:
        return bool(re.search(r"^\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\]?\s+", text, re.MULTILINE))

    def _split_time_range(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        parts = text.split("-->")
        if len(parts) != 2:
            return None, None
        return parts[0].strip(), parts[1].strip()


# =============================================================================
# ROUTER / GENRE SPECIALIZATION
# =============================================================================

class GenreSpecialization:
    GENRE_RULES: Dict[str, Dict[str, Any]] = {
        "song": {
            "directive": (
                "Prioritize lyrical meaning, emotional flow, symbolic motifs, "
                "and repetition as chorus/hook emphasis rather than new plot."
            ),
            "default_visual_constraints": [
                "Allow lyrical abstraction, but keep recurring motifs visually consistent.",
                "Treat repeated lines as emotional emphasis unless the text clearly introduces a new event.",
            ],
        },
        "poem": {
            "directive": (
                "Prioritize metaphor, symbolic language, emotional compression, and interpretive restraint."
            ),
            "default_visual_constraints": [
                "Do not flatten metaphor into overly literal plot unless strongly supported by the text.",
            ],
        },
        "ghazal": {
            "directive": (
                "Treat each couplet as potentially semi-autonomous while preserving emotional climate. "
                "Respect ambiguity and symbolic excess."
            ),
            "default_visual_constraints": [
                "Do not collapse ghazal ambiguity into a single literal storyline.",
            ],
        },
        "qawwali": {
            "directive": (
                "Preserve devotional performance energy, refrain logic, spiritual metaphor, and call-response possibility."
            ),
            "default_visual_constraints": [
                "Do not force devotional language into secular romance unless clearly warranted.",
            ],
        },
        "story": {
            "directive": (
                "Prioritize continuity, character-location clarity, and causal narrative progression."
            ),
            "default_visual_constraints": [
                "Preserve character and location continuity across line meanings.",
            ],
        },
        "script": {
            "directive": (
                "Prioritize acting beats, subtext, body language, pauses, and performance-driven meaning. "
                "No lip-sync speculation."
            ),
            "default_visual_constraints": [
                "Focus interpretation on face, posture, gesture, silence, and scene tension rather than lyric-style montage.",
            ],
        },
        "documentary": {
            "directive": (
                "Prioritize realism, observational framing, factual restraint, and naturalistic interpretation."
            ),
            "default_visual_constraints": [
                "Avoid stylized interpretation unless directly justified by the text.",
                "Prefer realism, natural light, and observational tone.",
            ],
        },
        "ad": {
            "directive": (
                "Prioritize clarity, appeal, product emphasis, premium aesthetics, and clean intent."
            ),
            "default_visual_constraints": [
                "Favor clean, high-clarity visual interpretation.",
                "Avoid ambiguity that weakens the core selling focus.",
            ],
        },
        "voiceover": {
            "directive": (
                "Preserve narration logic, pacing intent, and cinematic illustrative compatibility."
            ),
            "default_visual_constraints": [
                "Do not overcomplicate narration-led content with symbolic overload.",
            ],
        },
        "mixed": {
            "directive": "Identify dominant mode but preserve mixed structure honestly.",
            "default_visual_constraints": [],
        },
        "unknown": {
            "directive": (
                "Be cautious, infer dominant expressive mode, preserve uncertainty, and still return usable structure."
            ),
            "default_visual_constraints": [],
        },
    }

    @classmethod
    def get(cls, input_type: str) -> Dict[str, Any]:
        return cls.GENRE_RULES.get(
            (input_type or "").lower(),
            {
                "directive": "Stay grounded, clear, and downstream-friendly.",
                "default_visual_constraints": [],
            },
        )


class MetaMindInputRouter:
    SUPPORTED_TYPES = {
        "song", "poem", "ghazal", "qawwali", "script", "story",
        "ad", "voiceover", "documentary", "mixed", "unknown"
    }

    def detect(self, text: str, hinted_type: Optional[str] = None) -> Dict[str, Any]:
        hinted = (hinted_type or "").strip().lower()
        cleaned = text.lower()

        detected_type = hinted if hinted in self.SUPPORTED_TYPES else self._infer_type(cleaned)

        line_count = len([x for x in text.splitlines() if x.strip()])
        repetition = self._has_repetition(text)

        return {
            "recognized_type": detected_type,
            "raw_detected_type": hinted if hinted else detected_type,
            "is_mixed_input": detected_type == "mixed",
            "structure_quality": self._infer_structure_quality(text, line_count),
            "line_count": line_count,
            "has_repetition": repetition,
            "symbolic_density": self._infer_symbolic_density(cleaned),
            "abstraction_level": self._infer_abstraction_level(cleaned),
        }

    def _infer_type(self, text: str) -> str:
        if any(x in text for x in ["chorus", "verse", "hook"]):
            return "song"
        if "ghazal" in text:
            return "ghazal"
        if "qawwali" in text or "alaap" in text or "sufi" in text:
            return "qawwali"
        if any(x in text for x in ["scene", "dialogue", "int.", "ext."]):
            return "script"
        if any(x in text for x in ["buy now", "introducing", "limited offer"]):
            return "ad"
        if "voiceover" in text:
            return "voiceover"
        if "documentary" in text:
            return "documentary"
        if any(x in text for x in ["once upon", "he said", "she said"]):
            return "story"
        if len(text.splitlines()) >= 2:
            return "poem"
        return "unknown"

    def _has_repetition(self, text: str) -> bool:
        lines = [self._norm(x) for x in text.splitlines() if x.strip()]
        if not lines:
            return False
        counts = Counter(lines)
        return any(v > 1 for v in counts.values())

    def _infer_structure_quality(self, text: str, line_count: int) -> str:
        if not text.strip():
            return "empty"
        if line_count <= 1:
            return "fragmented"
        if len(text) < 80:
            return "short"
        return "clean"

    def _infer_symbolic_density(self, text: str) -> str:
        markers = ["like", "as if", "shadow", "dream", "heart", "sun", "winter", "name", "silence"]
        score = sum(1 for m in markers if m in text)
        if score >= 5:
            return "high"
        if score >= 2:
            return "medium"
        return "low"

    def _infer_abstraction_level(self, text: str) -> str:
        abstract_markers = ["existence", "silence", "memory", "shadow", "soul", "heart", "absence"]
        score = sum(1 for m in abstract_markers if m in text)
        if score >= 5:
            return "high"
        if score >= 2:
            return "medium"
        return "low"

    def _norm(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u0600-\u06FF\u0A00-\u0A7F]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


# =============================================================================
# FINAL ENGINE
# =============================================================================

class MetaMindContextEngineFinal:
    """
    Cleaned final unified MetaMind context engine.

    Goals:
    - keep all important capability from the live engine
    - preserve backward compatibility
    - keep code readable and maintainable
    - output one reusable context packet for the rest of the workflow
    """

    MODEL = "gpt-4o"
    DEFAULT_CONFIDENCE = 0.72

    LANGUAGE_LOCATION_DEFAULTS = {
        # South Asian
        "punjabi": "Punjab cultural region (South Asian)",
        "urdu": "Urdu cultural sphere (North India / Pakistan, South Asian)",
        "urdu/punjabi": "Urdu/Punjabi cultural sphere (South Asian)",
        "hindi": "Hindi cultural heartland (North/Central India, South Asian)",
        "bengali": "Bengali cultural region (Bengal, South Asian)",
        "tamil": "Tamil cultural region (South India / Sri Lanka)",
        "telugu": "Telugu cultural region (South India)",
        # European
        "english": "English-speaking cultural world (UK / Ireland / North America / Australia)",
        "english/romanized": "English-speaking cultural world (UK / Ireland / North America / Australia)",
        "french": "French cultural world (France / French-speaking regions)",
        "spanish": "Spanish-speaking cultural world (Spain / Latin America)",
        "portuguese": "Portuguese-speaking cultural world (Portugal / Brazil)",
        "german": "German-speaking cultural world (Germany / Austria / Switzerland)",
        "italian": "Italian cultural world",
        "dutch": "Dutch / Flemish cultural world",
        # East Asian
        "korean": "Korean cultural world",
        "japanese": "Japanese cultural world",
        "mandarin": "Chinese cultural world (Mandarin-speaking regions)",
        "chinese": "Chinese cultural world",
        "cantonese": "Cantonese / Southern Chinese cultural world",
        # Middle Eastern / African
        "arabic": "Arabic-speaking cultural world (Middle East / North Africa)",
        "persian": "Persian / Farsi cultural world (Iran / Afghanistan)",
        "turkish": "Turkish cultural world",
        "swahili": "East African cultural world",
        "amharic": "Ethiopian cultural world",
        # South-East Asian
        "thai": "Thai cultural world",
        "indonesian": "Indonesian cultural world",
        "tagalog": "Filipino cultural world",
    }

    # NOTE: LANGUAGE_APPEARANCE_DEFAULTS (complexion / wardrobe / grooming
    # per language) was removed. Visual appearance is NOT a meaning concern
    # and must not be decided here. The Character Materializer (Stage 7) is
    # the only stage allowed to derive appearance, and it must derive it
    # from its immediate predecessor's output, not from this engine.

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing OpenAI API key.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.router = MetaMindInputRouter()
        self.parser = StructuredInputParser()

    async def generate(
        self,
        raw_input: Any,
        hinted_type: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
        explicit_culture_pack: Optional[str] = None,
        locked_assumptions: Optional[Dict[str, Any]] = None,
        input_packet: Optional[Dict[str, Any]] = None,
        **_ignored_downstream: Any,
    ) -> Dict[str, Any]:
        # PIPELINE CHAIN RULE: this is Stage 2.
        # When input_packet is supplied it is the SOLE source of text,
        # structure, language and type — Stage 1 output consumed directly.
        # Any unexpected kwargs are absorbed (legacy callers don't crash)
        # but logged loudly so the chain violation is visible.
        if _ignored_downstream:
            logger.warning(
                "Context Engine (Stage 2) received downstream kwargs %s — "
                "chain violation; caller must stop passing these.",
                sorted(_ignored_downstream.keys()),
            )
        pre_analysis = pre_analysis or {}
        locked_assumptions = locked_assumptions or {}

        if input_packet:
            # ---------------------------------------------------------------
            # CHAIN-COMPLIANT PATH — Stage 1 output consumed directly.
            # No re-parsing, no re-routing, no re-detecting language.
            # ---------------------------------------------------------------
            cleaned_text = input_packet.get("clean_text", "")
            if not cleaned_text.strip():
                raise ValueError("Input is empty after parsing.")
            parsed_input  = self._parsed_input_from_packet(input_packet)
            lines         = self._lines_from_packet(input_packet)
            routed        = self._routed_from_packet(input_packet, cleaned_text)
            language_info = self._language_info_from_packet(input_packet)
        else:
            # ---------------------------------------------------------------
            # LEGACY PATH — raw text passed directly (no Stage 1 packet).
            # ---------------------------------------------------------------
            parsed_input = self.parser.parse(raw_input)
            cleaned_text = parsed_input["cleaned_text"]
            lines = parsed_input["lines"]
            if not cleaned_text.strip():
                raise ValueError("Input is empty after parsing.")
            if len(lines) <= 1 and len(cleaned_text) > 120:
                lines = self._segment_blob_into_lines(cleaned_text)
            routed = self.router.detect(cleaned_text, hinted_type=hinted_type)
            language_info = self._detect_language(cleaned_text)

        culture_pack_id = CulturePackRegistry.detect_pack(cleaned_text, explicit_pack=explicit_culture_pack)
        culture_pack = CulturePackRegistry.get_pack(culture_pack_id)
        active_metaphors = CulturePackRegistry.get_triggered_metaphors(cleaned_text, culture_pack_id)
        genre_cfg = GenreSpecialization.get(routed["recognized_type"])

        hard_logic = {
            "parsed_input":  parsed_input,
            "routing":       routed,
            "language":      language_info,
            "genre_directive": genre_cfg["directive"],
            "pre_analysis":  pre_analysis,
            "culture_pack_id": culture_pack_id,
            "culture_pack":  culture_pack,
            "active_metaphors": active_metaphors,
            "locked_assumptions": locked_assumptions,
            "input_packet":  input_packet,  # None when legacy path
        }

        system_prompt = self._build_system_prompt(hard_logic)
        user_prompt = self._build_user_prompt(cleaned_text, lines, hard_logic)

        raw = await self._call_model(system_prompt, user_prompt)
        parsed = self._safe_parse_json(raw)

        validated = self._validate_and_repair(
            data=parsed,
            lines=lines,
            hard_logic=hard_logic,
        )
        self._normalize_by_genre(validated)
        return validated

    # -------------------------------------------------------------------------
    # INPUT PACKET → INTERNAL STRUCTURES  (Stage 1 → Stage 2 conversion)
    # -------------------------------------------------------------------------

    def _parsed_input_from_packet(self, pkt: Dict[str, Any]) -> Dict[str, Any]:
        """Build the parsed_input structure that hard_logic expects, from the
        Stage 1 input_packet.  No text re-parsing."""
        return {
            "source_format": pkt.get("source_format", "plain_text"),
            "cleaned_text":  pkt.get("clean_text", ""),
            "lines":         [],  # populated separately via _lines_from_packet
            "sections":      pkt.get("sections", []),
        }

    def _lines_from_packet(self, pkt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Stage 1 units → the line format Context Engine expects."""
        sections_by_id = {s["id"]: s for s in pkt.get("sections", [])}
        lines: List[Dict[str, Any]] = []
        for unit in pkt.get("units", []):
            sec = sections_by_id.get(unit.get("section_id", ""), {})
            lines.append({
                "line_index":      unit.get("index", len(lines) + 1),
                "text":            unit.get("text", ""),
                "start_time":      unit.get("start_time"),
                "end_time":        unit.get("end_time"),
                "section_label":   sec.get("label"),
                "section_type":    sec.get("type"),
                "annotation_tags": [],
                "speaker":         unit.get("speaker"),
                "speaker_type":    unit.get("speaker_type"),
                "unit_type":       unit.get("unit_type"),
                "is_inferred":     unit.get("is_inferred", False),
            })
        return lines

    def _routed_from_packet(
        self, pkt: Dict[str, Any], cleaned_text: str
    ) -> Dict[str, Any]:
        """Build the routing dict from Stage 1 data.
        symbolic_density and abstraction_level still run on the clean text
        (they analyse text signals, not downstream data — no chain violation)."""
        input_type  = pkt.get("input_type", "unknown")
        sections    = pkt.get("sections", [])
        rep_map     = pkt.get("repetition_map", {})
        has_rep     = bool(rep_map.get("section_repetitions")) or bool(
            rep_map.get("unit_repetitions")
        )
        inferred    = [s for s in sections if s.get("is_inferred")]
        if not sections:
            quality = "low"
        elif len(inferred) == 0:
            quality = "explicit"
        elif len(inferred) == len(sections):
            quality = "inferred"
        else:
            quality = "partial"

        return {
            "recognized_type":  input_type,
            "raw_detected_type": pkt.get("sub_type", input_type),
            "is_mixed_input":   input_type == "mixed",
            "structure_quality": quality,
            "has_repetition":   has_rep,
            "symbolic_density": self.router._infer_symbolic_density(cleaned_text),
            "abstraction_level": self.router._infer_abstraction_level(cleaned_text),
        }

    # Language label from Input Engine → {primary, script, dialect}
    _LANG_LABEL_MAP: Dict[str, Dict[str, str]] = {
        "punjabi":   {"primary": "Punjabi",          "script": "Gurmukhi",         "dialect": ""},
        "urdu":      {"primary": "Urdu",              "script": "Arabic/Shahmukhi", "dialect": ""},
        "hindi":     {"primary": "Hindi",             "script": "Devanagari",       "dialect": ""},
        "bengali":   {"primary": "Bengali",           "script": "Bengali",          "dialect": ""},
        "korean":    {"primary": "Korean",            "script": "Hangul",           "dialect": ""},
        "japanese":  {"primary": "Japanese",          "script": "CJK",              "dialect": ""},
        "mandarin":  {"primary": "Mandarin Chinese",  "script": "CJK",              "dialect": ""},
        "chinese":   {"primary": "Chinese",           "script": "CJK",              "dialect": ""},
        "arabic":    {"primary": "Arabic",            "script": "Arabic",           "dialect": ""},
        "persian":   {"primary": "Persian",           "script": "Arabic/Naskh",     "dialect": ""},
        "turkish":   {"primary": "Turkish",           "script": "Latin",            "dialect": ""},
        "french":    {"primary": "French",            "script": "Latin",            "dialect": ""},
        "spanish":   {"primary": "Spanish",           "script": "Latin",            "dialect": ""},
        "portuguese":{"primary": "Portuguese",        "script": "Latin",            "dialect": ""},
        "german":    {"primary": "German",            "script": "Latin",            "dialect": ""},
        "italian":   {"primary": "Italian",           "script": "Latin",            "dialect": ""},
        "latin":     {"primary": "English/Romanized", "script": "Latin",            "dialect": ""},
        "english":   {"primary": "English/Romanized", "script": "Latin",            "dialect": ""},
    }

    def _language_info_from_packet(self, pkt: Dict[str, Any]) -> Dict[str, str]:
        """Convert Stage 1 languages list → {primary, script, dialect}."""
        langs = pkt.get("languages") or []
        primary_label = (langs[0] if langs else "").lower()
        return self._LANG_LABEL_MAP.get(
            primary_label,
            {"primary": primary_label.title() or "Unknown", "script": "", "dialect": ""},
        )

    def _structure_block_from_packet(self, pkt: Dict[str, Any]) -> str:
        """Build the STRUCTURE FROM INPUT PROCESSOR block for the user prompt."""
        sections      = pkt.get("sections", [])
        rep_map       = pkt.get("repetition_map", {})
        lyr           = pkt.get("lyrical_patterns", {})
        spk_types     = pkt.get("speaker_types", {})
        timing        = pkt.get("timing", {})
        uncertainties = pkt.get("uncertainties", [])
        sub_type      = pkt.get("sub_type", "")

        lines: List[str] = ["STRUCTURE FROM INPUT PROCESSOR (Stage 1):"]

        if sub_type:
            lines.append(f"- Sub-type: {sub_type}")

        if sections:
            lines.append(f"- Sections ({len(sections)} total):")
            for s in sections:
                inferred_tag = " [inferred]" if s.get("is_inferred") else ""
                rep_of = f" [repeat of {s['repeat_of']}]" if s.get("repeat_of") else ""
                trans  = f" → {s['scene_transition']}" if s.get("scene_transition") else ""
                lines.append(
                    f"    {s['id']} | {s['type']:10s} | {s['label']}{inferred_tag}{rep_of}{trans}"
                    f" ({len(s.get('unit_ids', []))} lines)"
                )

        rep_ids = rep_map.get("repeated_section_ids", [])
        if rep_ids:
            lines.append(f"- Repeated sections: {', '.join(rep_ids)}")

        if lyr.get("rhyme_repetition_detected"):
            lines.append("- Lyrical pattern: rhyme repetition detected")

        if spk_types:
            narrators   = [n for n, t in spk_types.items() if t == "narrator"]
            characters  = [n for n, t in spk_types.items() if t == "character"]
            if narrators:
                lines.append(f"- Narrator voice(s): {', '.join(narrators)}")
            if characters:
                lines.append(f"- Character voice(s): {', '.join(characters)}")

        if timing.get("bpm"):
            lines.append(f"- BPM: {timing['bpm']}")
        if timing.get("duration_seconds"):
            lines.append(f"- Duration: {timing['duration_seconds']}s")
        if timing.get("timed_units", 0) > 0:
            lines.append(
                f"- Timed lines: {timing['timed_units']}/{timing['total_units']} "
                f"({int(timing.get('coverage', 0) * 100)}% coverage)"
            )

        if uncertainties:
            lines.append(
                "- Input uncertainties: "
                + "; ".join(u.get("code", "") for u in uncertainties)
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # INPUT SPLITTING
    # -------------------------------------------------------------------------

    def _segment_blob_into_lines(self, blob: str) -> List[Dict[str, Any]]:
        raw_lines = self._split_blob_into_phrases(blob)
        return [
            {
                "line_index": i + 1,
                "text": line,
                "start_time": None,
                "end_time": None,
                "section_label": None,
                "annotation_tags": [],
            }
            for i, line in enumerate(raw_lines)
        ]

    def _split_blob_into_phrases(self, blob: str) -> List[str]:
        if not blob:
            return []

        sentence_split = re.split(r"(?<=[\.!?।۔])\s+", blob)
        sentence_split = [s.strip() for s in sentence_split if s.strip()]
        if len(sentence_split) >= 4:
            return sentence_split

        clause_split: List[str] = []
        for chunk in sentence_split or [blob]:
            parts = re.split(r"\s*[,;—–]\s*", chunk)
            for p in parts:
                p = p.strip()
                if p:
                    clause_split.append(p)
        if len(clause_split) >= 4:
            return clause_split

        words = blob.split()
        wrapped: List[str] = []
        cur = ""
        for w in words:
            if len(cur) + 1 + len(w) > 80 and cur:
                wrapped.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            wrapped.append(cur)

        return wrapped or [blob]

    # -------------------------------------------------------------------------
    # PROMPTS
    # -------------------------------------------------------------------------

    def _build_system_prompt(self, hard_logic: Dict[str, Any]) -> str:
        routed = hard_logic["routing"]
        language = hard_logic["language"]
        genre_directive = hard_logic["genre_directive"]
        culture_pack_id = hard_logic["culture_pack_id"] or "none"
        active_metaphors = hard_logic["active_metaphors"]
        locked_assumptions = hard_logic["locked_assumptions"]

        metaphor_text = "\n".join(
            f"- '{k}' means: {v}" for k, v in active_metaphors.items()
        ) if active_metaphors else "None"

        # VOCAL GENDER DIRECTIVE — hard fact derived from audio analysis
        # (or explicit user override). This is a meaning-level fact (who
        # the speaker is), not a visual one, so it belongs here.
        vocal_gender_block = ""
        audio_hints = (hard_logic.get("pre_analysis") or {}).get("audio_hints") or {}
        vg_final = str(audio_hints.get("vocal_gender_final") or audio_hints.get("vocal_gender") or "").lower().strip()
        if vg_final in ("male", "female", "mixed"):
            f0 = audio_hints.get("vocal_f0_hz") or 0
            label_map = {"male": "Male", "female": "Female", "mixed": "Mixed / Duet"}
            vocal_gender_block = f"""
VOCAL GENDER DIRECTIVE (from audio analysis{f' — median F0 ≈ {f0:.0f} Hz' if f0 else ''}):
- The singer is: {label_map[vg_final]}
- You MUST set speaker.gender to "{label_map[vg_final]}" — do NOT output "Unclear", "Unspecified", or any other value.
- For "Mixed / Duet", set speaker.gender to "Mixed" and reflect both voices in speaker.identity.
"""

        return f"""
You are Qaivid MetaMind — Stage 2 of the cinematic pipeline. Your only job is to extract
the MEANING of the input. Downstream stages (Narrative Intelligence, Style, Storyboard,
Creative Brief, Materializer, Stills) handle every visual decision.

CORE PRINCIPLE: Lock meaning. Keep visual expression open.

INPUT ROUTING:
- recognized_type: {routed["recognized_type"]}
- raw_detected_type: {routed["raw_detected_type"]}
- structure_quality: {routed["structure_quality"]}
- has_repetition: {routed["has_repetition"]}
- symbolic_density: {routed["symbolic_density"]}
- abstraction_level: {routed["abstraction_level"]}

LANGUAGE:
- primary: {language["primary"]}
- script: {language["script"]}

CULTURE PACK:
- selected: {culture_pack_id}

ACTIVE CULTURAL METAPHORS:
{metaphor_text}

LOCKED ASSUMPTIONS:
{json.dumps(locked_assumptions, ensure_ascii=False)}

GENRE DIRECTIVE:
{genre_directive}
{vocal_gender_block}
WHAT YOU MUST DO:
1. Identify content type and language.
2. Extract core_theme, dramatic_premise, narrative_spine — meaning only.
3. Fill all 5 Ws (WHO / WHAT / WHEN / WHERE / WHY).
4. Classify narrative_mode (literal / symbolic / hybrid / etc.).
5. Define the emotional arc and the motivation block.
6. Per line: literal, implied, emotional, and cultural meaning + intensity + repeat status + function.
7. Identify literary devices and named entities.
8. Define cultural_constraints (what culturally MUST not break).
9. Define preservation_rules (what MUST be locked downstream).
10. Define creative_freedom (what downstream stages MAY vary).
11. Surface assumptions honestly and flag ambiguity.
12. Respect locked assumptions where provided.

WHAT YOU MUST NEVER DO (THESE ARE DOWNSTREAM CONCERNS):
- ❌ No complexion, wardrobe, grooming, hair, jewelry — Character Materializer (Stage 7).
- ❌ No architecture style, building type, room type — Location Materializer (Stage 7).
- ❌ No props, objects, plants, garments per line — Storyboard / Stills (Stage 5/9).
- ❌ No shot framing, camera, lens, lighting, color, motion — Style / Storyboard / Stills.
- ❌ No scenes, locations, time-of-day per shot — Creative Brief (Stage 6).
- ❌ No visualization_mode / expression_mode per line — Narrative Intelligence (Stage 3).
- ❌ No motifs as visual symbols — Narrative Intelligence decides expression channels.
- ❌ No "visual_constraints" or "restrictions" — Style decides those.

5W FRAMEWORK:
- WHO  → speaker / addressee (identity, role, relationship — broad, not visual)
- WHAT → core_theme / dramatic_premise / narrative_spine
- WHEN → world_assumptions.era / season / timeline_nature / emotional_arc
- WHERE → world_assumptions.geography / location_dna (cultural anchor only)
- WHY  → motivation (REQUIRED — do not leave blank)
The "motivation" object captures the emotional engine. Fill:
  - inciting_cause: the concrete event/loss/longing/decision that triggered this expression
  - underlying_desire: what the speaker ultimately wants
  - stakes: what is at risk if the desire is not met
  - obstacle: what stands between the speaker and the desire
  - confidence: 0..1 — how confident you are in the motivation reading
If the lyrics are abstract, infer motivation from emotional arc + cultural context — never leave blank.

REQUIRED JSON SHAPE (meaning-only):
{{
  "input_profile": {{
    "recognized_type": "string",
    "raw_detected_type": "string",
    "is_mixed_input": false,
    "structure_quality": "string",
    "source_format": "string",
    "language": {{ "primary": "string", "script": "string", "dialect": "string" }},
    "analysis_confidence": 0.0
  }},
  "input_type": "string",
  "language": "string",
  "narrative_mode": "string  (literal | symbolic | hybrid | performative — meaning mode, not visual mode)",
  "location_dna": "string  (broad cultural sphere — e.g. 'Punjab cultural region (South Asian)', 'French-speaking world', 'Korean cultural world'. NOT architecture, NOT setting type)",
  "genre_directive": "string",
  "core_theme": "string",
  "dramatic_premise": "string",
  "narrative_spine": "string",
  "speaker": {{
    "identity": "string  (broad — e.g. 'a young woman addressing an absent lover', 'an elder recounting a memory'. NO appearance.)",
    "gender": "string  (Male | Female | Mixed | Unclear)",
    "age_range": "string  (e.g. 'young adult', 'middle-aged', 'elder')",
    "social_role": "string",
    "emotional_state": "string",
    "relationship_to_addressee": "string",
    "cultural_background": "string  (broad cultural identity — e.g. 'South Asian (Punjabi)', 'French', 'Korean', 'West African'. NO complexion, NO wardrobe, NO grooming.)"
  }},
  "addressee": {{
    "identity": "string",
    "relationship": "string",
    "presence": "string  (literal_present | absent | symbolic | divine | self)"
  }},
  "world_assumptions": {{
    "geography": "string  (broad geographic anchor — e.g. 'Punjab region, South Asia', 'rural France', 'urban Seoul')",
    "era": "string  (e.g. 'contemporary', 'mid-20th century', 'timeless')",
    "season": "string",
    "timeline_nature": "string  (real_time | memory | cyclical | ambiguous — how time behaves in the meaning of the piece)",
    "social_context": "string",
    "economic_context": "string"
  }},
  "emotional_arc": {{
    "opening": "string",
    "development": "string",
    "climax": "string",
    "resolution": "string"
  }},
  "motivation": {{
    "inciting_cause": "string",
    "underlying_desire": "string",
    "stakes": "string",
    "obstacle": "string",
    "confidence": 0.0
  }},
  "line_meanings": [
    {{
      "line_index": 1,
      "text": "string",
      "literal_meaning": "string",
      "implied_meaning": "string",
      "emotional_meaning": "string",
      "cultural_meaning": "string",
      "meaning": "string  (one-line summary)",
      "function": "string  (e.g. 'verse', 'chorus', 'bridge', 'invocation', 'declaration', 'lament')",
      "repeat_status": "original|repeat",
      "intensity": 0.5
    }}
  ],
  "entities": [
    {{ "name": "string", "type": "person|place|symbol|concept", "role": "string  (meaning role, not visual prop)" }}
  ],
  "literary_devices": ["string"],
  "cultural_constraints": ["string  (meaning-level cultural rules that MUST hold — e.g. 'the beloved in a ghazal is symbolic and may be human, divine, or abstract; do not literalize as a single named woman')"],
  "preservation_rules": ["string  (what downstream MUST NOT change — e.g. 'speaker gender', 'cultural identity', 'emotional truth of the climax')"],
  "creative_freedom": ["string  (what downstream MAY vary — e.g. 'casting specifics', 'architecture and locations', 'props and objects', 'era visualization', 'color and motion')"],
  "surfaced_assumptions": [
    {{ "field": "string", "value": "string", "confidence": 0.0, "reason": "string" }}
  ],
  "locked_assumptions": {{}},
  "ambiguity_flags": [
    {{ "field": "string", "reason": "string", "confidence": 0.0 }}
  ],
  "confidence": 0.0,
  "confidence_scores": {{
    "overall": 0.0,
    "cultural": 0.0,
    "emotional": 0.0,
    "speaker": 0.0,
    "narrative_mode": 0.0,
    "motivation": 0.0
  }}
}}

Return ONLY valid JSON.
""".strip()

    def _build_user_prompt(
        self,
        text: str,
        lines: List[Dict[str, Any]],
        hard_logic: Dict[str, Any],
    ) -> str:
        indexed_lines = []
        for ln in lines:
            bits = [f"[{ln['line_index']}] {ln['text']}"]
            if ln.get("start_time"):
                bits.append(f"(start: {ln['start_time']})")
            if ln.get("end_time"):
                bits.append(f"(end: {ln['end_time']})")
            if ln.get("section_label"):
                bits.append(f"(section: {ln['section_label']})")
            if ln.get("annotation_tags"):
                bits.append(f"(tags: {', '.join(map(str, ln['annotation_tags']))})")
            indexed_lines.append(" ".join(bits))

        lang = (hard_logic["language"].get("primary") or "").lower()
        cultural_hint = ""
        # Cultural framework hints — MEANING-LEVEL ONLY (literary tradition,
        # thematic motifs, cultural truth). No visual prescriptions.
        if "punjabi" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Punjabi folk / Sufi / qissa tradition. "
                "Common thematic motifs: separation (judaai), beloved (mahi/sajjan), agrarian land, "
                "migration, oral-song memory, divine longing. Be true to the Punjabi cultural world."
            )
        elif "urdu" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Urdu ghazal / nazm / qawwali tradition. "
                "Common thematic motifs: hijr, wasl, mehboob, divine beloved, silence, existential longing. "
                "The beloved is often symbolic — may be human, divine, or abstract. Do not literalize."
            )
        elif "hindi" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Hindi film song / bhajan / folk tradition. "
                "Common thematic motifs: love, separation, devotion, nature, seasons, inner longing."
            )
        elif "bengali" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Bengali song tradition (Rabindra Sangeet / Baul / folk). "
                "Common thematic motifs: river, rain, seasons, philosophical longing, nature."
            )
        elif "tamil" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Tamil song / Carnatic / folk tradition. "
                "Common thematic motifs: nature, devotion, love, separation, classical dance."
            )
        elif "french" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: French chanson / pop / classical song tradition. "
                "Be true to the French cultural world."
            )
        elif "spanish" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Spanish / Latin American song tradition (bolero, cumbia, flamenco, pop). "
                "Identify region from lyric content; be true to that cultural world."
            )
        elif "portuguese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Portuguese / Brazilian song tradition (fado, bossa nova, sertanejo, pop). "
                "Be true to the relevant Portuguese-speaking cultural world."
            )
        elif "korean" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Korean song tradition (K-pop, ballad, trot, folk). "
                "Be true to the Korean cultural world."
            )
        elif "japanese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Japanese song tradition (J-pop, enka, folk, shoegaze). "
                "Be true to the Japanese cultural world."
            )
        elif "mandarin" in lang or "chinese" in lang or "cantonese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Chinese song tradition (Mandopop, Cantopop, folk, classical). "
                "Identify mainland / Hong Kong / Taiwan from lyric content; be true to that cultural world."
            )
        elif "arabic" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Arabic song tradition (tarab, khaleeji, pop, mawwal). "
                "Identify region (Gulf, Levant, Egypt, Maghreb) from lyric content."
            )
        elif "persian" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Persian / Farsi song tradition (classical, pop, ghazal). "
                "Be true to the Iranian / Persian cultural world."
            )
        elif "turkish" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Turkish song tradition (Türkü folk, arabesk, pop). "
                "Be true to the Turkish / Anatolian cultural world."
            )
        elif "english" in lang or "romanized" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Latin-script content — first identify the ACTUAL language and "
                "cultural origin from the lyric text. Be true to that cultural world's literary tradition."
            )
        elif "german" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: German-language song tradition. "
                "Be true to the German / Austrian / Swiss cultural world as appropriate."
            )
        elif "italian" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Italian song tradition (canzone, opera pop, folk). "
                "Be true to the Italian cultural world."
            )

        pkt = hard_logic.get("input_packet")
        structure_block = self._structure_block_from_packet(pkt) if pkt else ""

        return f"""
TEXT TO INTERPRET:
{text}

INDEXED LINES:
{chr(10).join(indexed_lines)}

HELPFUL CONTEXT:
- Input type: {hard_logic["routing"]["recognized_type"]}
- Language: {hard_logic["language"]["primary"]}
- Source format: {hard_logic["parsed_input"]["source_format"]}
- Culture pack: {hard_logic["culture_pack_id"] or "none"}
- Repetition detected: {hard_logic["routing"]["has_repetition"]}
- Symbolic density: {hard_logic["routing"]["symbolic_density"]}
- Abstraction level: {hard_logic["routing"]["abstraction_level"]}

{cultural_hint}

{structure_block}

OPTIONAL PRE-ANALYSIS:
{json.dumps(hard_logic.get("pre_analysis", {}), ensure_ascii=False)}

REQUIREMENTS:
- line_meanings must contain exactly one entry per indexed line, in order.
- Fill literal_meaning, implied_meaning, emotional_meaning, and cultural_meaning for every line.
- Mark repeated lines with repeat_status="repeat".
- Keep narrative_spine compact — meaning, not visuals.
- world_assumptions should be inferred from textual evidence, not invented.
- cultural_constraints, preservation_rules, creative_freedom must each contain at least one entry.
- confidence_scores must be honest.
- Do NOT output any visual fields — no props, no clothing, no architecture, no shot framing.
""".strip()

    async def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _safe_parse_json(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Invalid JSON from MetaMind final: %s", exc)
            return {}

    # -------------------------------------------------------------------------
    # VALIDATION / REPAIR
    # -------------------------------------------------------------------------

    def _validate_and_repair(
        self,
        data: Dict[str, Any],
        lines: List[Dict[str, Any]],
        hard_logic: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}

        routed = hard_logic["routing"]
        language = hard_logic["language"]
        parsed_input = hard_logic["parsed_input"]
        culture_pack_id = hard_logic["culture_pack_id"]
        culture_pack = hard_logic["culture_pack"]
        locked_assumptions = hard_logic["locked_assumptions"]
        genre_cfg = GenreSpecialization.get(routed["recognized_type"])

        self._ensure_top_level_defaults(data)

        data["input_profile"] = {
            "recognized_type": routed["recognized_type"],
            "raw_detected_type": routed["raw_detected_type"],
            "is_mixed_input": routed["is_mixed_input"],
            "structure_quality": routed["structure_quality"],
            "source_format": parsed_input.get("source_format", "plain_text"),
            "language": {
                "primary": language["primary"],
                "script": language["script"],
                "dialect": language["dialect"],
            },
            "analysis_confidence": self._repair_confidence(data["input_profile"].get("analysis_confidence", 0.78)),
        }

        data["input_type"] = routed["recognized_type"]
        # Prefer the LLM's richer language detection for Latin-script content
        # (e.g. French, Spanish, Korean romanized) over the fallback "English/Romanized"
        _llm_lang = str(data.get("language") or "").strip()
        _det_primary = language.get("primary", "")
        _det_script = language.get("script", "")
        if _llm_lang and "English/Romanized" in f"{_det_primary} ({_det_script})" and _llm_lang:
            pass  # keep LLM's richer value already in data["language"]
        else:
            data["language"] = f'{_det_primary} ({_det_script})'.strip()
        data["narrative_mode"] = self._ensure_string(data.get("narrative_mode"), "unknown")
        data["location_dna"] = self._ensure_string(
            data.get("location_dna"),
            culture_pack.get("world_defaults", {}).get("cultural_dna")
            or self._language_to_location_dna(language.get("primary", ""))
            or "Universal",
        )
        data["genre_directive"] = genre_cfg["directive"]

        # Pass the vocal gender hint so _repair_speaker uses it as a hard
        # default instead of "Unclear" if the LLM somehow omitted it.
        _vh = (hard_logic.get("pre_analysis") or {}).get("audio_hints") or {}
        _vg = str(_vh.get("vocal_gender_final") or _vh.get("vocal_gender") or "").lower().strip()
        _vg_label = {"male": "Male", "female": "Female", "mixed": "Mixed"}.get(_vg)
        self._repair_speaker(data, culture_pack, language, vocal_gender=_vg_label)
        self._repair_addressee(data)
        self._repair_world_assumptions(data, culture_pack)
        self._repair_core_narrative_fields(data)
        self._repair_emotional_arc(data)
        self._repair_motivation(data)
        data["line_meanings"] = self._repair_line_meanings(data.get("line_meanings"), lines)
        self._repair_entities(data)
        self._repair_literary_devices(data)
        # Stage 2 emits MEANING-LEVEL constraints only — no visual_constraints,
        # no restrictions, no motifs/motif_map. Narrative Intelligence (Stage 3)
        # owns motif strategy; Style (Stage 4) owns visual constraints.
        self._repair_meaning_level_constraints(data, culture_pack)
        # Strip any legacy visual fields the LLM might have emitted by habit.
        for _legacy in ("visual_constraints", "restrictions", "motifs", "motif_map"):
            data.pop(_legacy, None)

        data["surfaced_assumptions"] = self._build_surfaced_assumptions(data)
        data["locked_assumptions"] = self._apply_locked_assumptions(data, locked_assumptions)

        self._repair_ambiguity(data, routed["has_repetition"])
        self._repair_confidence_scores(data)

        data["meta"] = {
            "engine": "Qaivid MetaMind Context Engine Final",
            "version": "final",
            "culture_pack_id": culture_pack_id,
            "source_format": parsed_input.get("source_format", "plain_text"),
            "symbolic_density": routed["symbolic_density"],
            "abstraction_level": routed["abstraction_level"],
        }

        return data

    def _ensure_top_level_defaults(self, data: Dict[str, Any]) -> None:
        # Stage 2 emits MEANING ONLY. No visual_constraints, no restrictions,
        # no motifs/motif_map (Narrative Intelligence owns motif strategy).
        defaults = {
            "input_profile": {},
            "speaker": {},
            "addressee": {},
            "world_assumptions": {},
            "emotional_arc": {},
            "motivation": {},
            "line_meanings": [],
            "entities": [],
            "literary_devices": [],
            "cultural_constraints": [],
            "preservation_rules": [],
            "creative_freedom": [],
            "surfaced_assumptions": [],
            "locked_assumptions": {},
            "ambiguity_flags": [],
            "confidence_scores": {},
        }
        for key, default in defaults.items():
            data.setdefault(key, default)

    def _repair_speaker(self, data: Dict[str, Any], culture_pack: Dict[str, Any], language: Dict[str, str], vocal_gender: Optional[str] = None) -> None:
        # MEANING-ONLY speaker block. No complexion, no wardrobe, no grooming.
        # Those are Materializer (Stage 7) concerns and must be derived
        # downstream from this engine's cultural_background, not pulled here.
        speaker_in = data["speaker"] if isinstance(data.get("speaker"), dict) else {}

        # If we have a high-confidence vocal gender from audio analysis (or user
        # override), it always wins over an LLM-emitted "Unclear"/"Unspecified".
        llm_gender = self._ensure_string(speaker_in.get("gender"), "")
        if vocal_gender and llm_gender.strip().lower() in ("", "unclear", "unspecified", "unknown", "n/a"):
            llm_gender = vocal_gender
        elif not llm_gender:
            llm_gender = vocal_gender or "Unclear"

        # cultural_background is the meaning-level cultural identity. The
        # legacy field name was "ethnicity" — we accept it as a fallback so
        # older context_packets stored in the DB still load, but the new
        # canonical key is "cultural_background".
        cultural_background = self._ensure_string(
            speaker_in.get("cultural_background") or speaker_in.get("ethnicity"),
            culture_pack.get("cultural_dna") or "Inferred from lyric content",
        )

        data["speaker"] = {
            "identity": self._ensure_string(speaker_in.get("identity"), "Unclear speaker"),
            "gender": llm_gender,
            "age_range": self._ensure_string(speaker_in.get("age_range"), "Unclear"),
            "emotional_state": self._ensure_string(speaker_in.get("emotional_state"), "Emotionally charged"),
            "social_role": self._ensure_string(speaker_in.get("social_role"), "Unclear"),
            "relationship_to_addressee": self._ensure_string(speaker_in.get("relationship_to_addressee"), "Unclear"),
            "cultural_background": cultural_background,
        }

    def _repair_addressee(self, data: Dict[str, Any]) -> None:
        addressee_in = data["addressee"] if isinstance(data.get("addressee"), dict) else {}
        data["addressee"] = {
            "identity": self._ensure_string(addressee_in.get("identity"), "Unclear addressee"),
            "relationship": self._ensure_string(addressee_in.get("relationship"), "Unclear"),
            "presence": self._ensure_string(addressee_in.get("presence"), "Unclear"),
        }

    def _repair_world_assumptions(self, data: Dict[str, Any], culture_pack: Dict[str, Any]) -> None:
        # MEANING-ONLY world block. No architecture_style, no
        # characteristic_setting, no characteristic_time — those are visual /
        # creative-brief decisions and must NOT originate here.
        # `timeline_nature` is added: real_time | memory | cyclical | ambiguous.
        # That is a meaning-of-time question, not a clock-time question.
        wa = data["world_assumptions"] if isinstance(data.get("world_assumptions"), dict) else {}
        defaults = culture_pack.get("world_defaults", {})

        timeline_nature_raw = self._ensure_string(wa.get("timeline_nature"), "").strip().lower()
        allowed_timeline = {"real_time", "memory", "cyclical", "ambiguous"}
        timeline_nature = timeline_nature_raw if timeline_nature_raw in allowed_timeline else "ambiguous"

        data["world_assumptions"] = {
            "geography": self._ensure_string(wa.get("geography"), defaults.get("geography", "Unspecified")),
            "era": self._ensure_string(wa.get("era"), "Unspecified"),
            "season": self._ensure_string(wa.get("season"), "Unspecified"),
            "timeline_nature": timeline_nature,
            "social_context": self._ensure_string(wa.get("social_context"), "Unspecified"),
            "economic_context": self._ensure_string(wa.get("economic_context"), "Unspecified"),
        }

    def _repair_core_narrative_fields(self, data: Dict[str, Any]) -> None:
        data["core_theme"] = self._ensure_string(data.get("core_theme"), "Unclear core theme")
        data["dramatic_premise"] = self._ensure_string(
            data.get("dramatic_premise"),
            "A speaker confronts an emotionally significant moment.",
        )
        data["narrative_spine"] = self._ensure_string(
            data.get("narrative_spine"),
            "A speaker expresses an emotionally or narratively significant inner state."
        )

    def _repair_emotional_arc(self, data: Dict[str, Any]) -> None:
        arc = data["emotional_arc"] if isinstance(data.get("emotional_arc"), dict) else {}
        data["emotional_arc"] = {
            "opening": self._ensure_string(arc.get("opening"), "Emotional setup"),
            "development": self._ensure_string(arc.get("development"), "Deepening feeling"),
            "climax": self._ensure_string(arc.get("climax"), "Peak emotional intensity"),
            "resolution": self._ensure_string(arc.get("resolution"), "Lingering aftermath"),
        }

    def _repair_motivation(self, data: Dict[str, Any]) -> None:
        """Fill the WHY block — the missing fifth W of the 5W framework.

        The LLM is instructed to fill this in the system prompt, but we still
        repair it defensively so older context_packets and partial responses
        always have a usable motivation block downstream stages can read.
        """
        m = data["motivation"] if isinstance(data.get("motivation"), dict) else {}
        data["motivation"] = {
            "inciting_cause": self._ensure_string(
                m.get("inciting_cause"),
                "An emotionally significant moment the speaker cannot stay silent about.",
            ),
            "underlying_desire": self._ensure_string(
                m.get("underlying_desire"),
                "To be heard, understood, or reconciled with what has been lost or longed for.",
            ),
            "stakes": self._ensure_string(
                m.get("stakes"),
                "Continued emotional weight — unresolved feeling carried forward.",
            ),
            "obstacle": self._ensure_string(
                m.get("obstacle"),
                "Distance, silence, or time keeping the speaker from resolution.",
            ),
            "confidence": self._repair_confidence(m.get("confidence")),
        }

    def _repair_line_meanings(self, line_meanings: Any, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # MEANING-ONLY per-line block. No expression_mode, no visualization_mode,
        # no visual_suitability, no visual_props. Narrative Intelligence
        # (Stage 3) decides expression channels; Storyboard / Stills (Stage 5/9)
        # decide visual props.
        existing_by_index: Dict[int, Dict[str, Any]] = {}

        if isinstance(line_meanings, list):
            for item in line_meanings:
                if isinstance(item, dict) and isinstance(item.get("line_index"), int):
                    existing_by_index[item["line_index"]] = item

        text_norms = [self._normalize_for_match(x["text"]) for x in lines]
        counts = Counter(text_norms)

        repaired: List[Dict[str, Any]] = []
        for line in lines:
            idx = int(line["line_index"])
            current = existing_by_index.get(idx, {})
            key = self._normalize_for_match(line["text"])
            is_repeat = counts[key] > 1

            literal = self._ensure_string(current.get("literal_meaning"), "")
            implied = self._ensure_string(current.get("implied_meaning"), "")
            emotional = self._ensure_string(current.get("emotional_meaning"), "")
            cultural = self._ensure_string(current.get("cultural_meaning"), "")
            combined = " | ".join(p for p in (implied, emotional, literal) if p) or "Literal or emotional meaning not clearly extracted."

            repaired.append({
                "line_index": idx,
                "text": line["text"],
                "literal_meaning": literal,
                "implied_meaning": implied,
                "emotional_meaning": emotional,
                "cultural_meaning": cultural,
                "meaning": self._ensure_string(current.get("meaning"), combined),
                "function": self._ensure_string(current.get("function"), "emotional_expression"),
                "repeat_status": "repeat" if is_repeat else "original",
                "intensity": self._repair_intensity(current.get("intensity")),
            })

        return repaired

    # NOTE: _repair_motifs was removed. Motifs / motif_map are not a Stage 2
    # concern — Narrative Intelligence (Stage 3) decides motif strategy from
    # the meaning surface this engine produces.

    def _repair_entities(self, data: Dict[str, Any]) -> None:
        if not isinstance(data.get("entities"), list):
            data["entities"] = []
        cleaned_entities = []
        for item in data["entities"]:
            if isinstance(item, dict):
                cleaned_entities.append(
                    {
                        "name": self._ensure_string(item.get("name"), "Unnamed"),
                        "type": self._ensure_string(item.get("type"), "symbol"),
                        "role": self._ensure_string(item.get("role"), "contextual"),
                    }
                )
        data["entities"] = cleaned_entities[:20]

    def _repair_literary_devices(self, data: Dict[str, Any]) -> None:
        if not isinstance(data.get("literary_devices"), list):
            data["literary_devices"] = []
        data["literary_devices"] = [
            str(x).strip() for x in data["literary_devices"] if str(x).strip()
        ][:20]

    def _repair_meaning_level_constraints(
        self,
        data: Dict[str, Any],
        culture_pack: Dict[str, Any],
    ) -> None:
        # Stage 2 produces three meaning-level constraint lists. None of these
        # describe visuals — they describe what meaning MUST be preserved and
        # what downstream stages MAY freely vary.
        #
        # cultural_constraints: cultural truths that MUST not be broken
        #   (e.g. "the beloved in a ghazal may be human, divine, or abstract").
        # preservation_rules: meaning-level facts downstream MUST keep
        #   (e.g. "speaker gender", "emotional truth of the climax").
        # creative_freedom: what downstream stages MAY decide on their own
        #   (e.g. "casting", "architecture", "props", "color", "motion").
        if not isinstance(data.get("cultural_constraints"), list):
            data["cultural_constraints"] = []
        cc = [str(x).strip() for x in data["cultural_constraints"] if str(x).strip()]
        # Pull culture-pack misinterpretation guards in — they are meaning-level
        # cultural rules, not visual ones.
        cc.extend(str(x).strip() for x in (culture_pack.get("common_misinterpretations") or []) if str(x).strip())
        if not cc:
            cc = ["Stay culturally faithful to the literary tradition of the source language."]
        data["cultural_constraints"] = self._dedupe(cc)

        if not isinstance(data.get("preservation_rules"), list):
            data["preservation_rules"] = []
        pr = [str(x).strip() for x in data["preservation_rules"] if str(x).strip()]
        if not pr:
            pr = [
                "Speaker gender as identified.",
                "Cultural identity of the speaker.",
                "Emotional truth of the climax.",
                "The 5W meaning frame (who/what/when/where/why).",
            ]
        data["preservation_rules"] = self._dedupe(pr)

        if not isinstance(data.get("creative_freedom"), list):
            data["creative_freedom"] = []
        cf = [str(x).strip() for x in data["creative_freedom"] if str(x).strip()]
        if not cf:
            cf = [
                "Casting specifics (face, complexion, exact age within range).",
                "Wardrobe and grooming specifics.",
                "Architecture, locations, interiors.",
                "Props and objects.",
                "Era visualization (within meaning-faithful bounds).",
                "Color, lighting, motion, and shot framing.",
            ]
        data["creative_freedom"] = self._dedupe(cf)

    def _apply_locked_assumptions(self, data: Dict[str, Any], locked_assumptions: Dict[str, Any]) -> Dict[str, Any]:
        if not locked_assumptions:
            return {}

        applied: Dict[str, str] = {}
        world = data.get("world_assumptions", {})
        # MEANING-LEVEL world locks only. Visual locks (architecture_style,
        # characteristic_setting, characteristic_time, time_of_day,
        # domestic_setting) are no longer applied here — those decisions
        # belong to the Creative Brief (Stage 6) and are locked there.
        for key in ["geography", "era", "season", "timeline_nature"]:
            if key in locked_assumptions:
                world[key] = str(locked_assumptions[key]).strip()
                applied[key] = world[key]

        if "location_dna" in locked_assumptions:
            data["location_dna"] = str(locked_assumptions["location_dna"]).strip()
            applied["location_dna"] = data["location_dna"]

        speaker = data.get("speaker", {})
        for key in ["identity", "gender", "age_range", "social_role", "emotional_state"]:
            lock_key = f"speaker_{key}"
            if lock_key in locked_assumptions:
                speaker[key] = str(locked_assumptions[lock_key]).strip()
                applied[lock_key] = speaker[key]

        if "narrative_mode" in locked_assumptions:
            data["narrative_mode"] = str(locked_assumptions["narrative_mode"]).strip()
            applied["narrative_mode"] = data["narrative_mode"]

        # WHY block — accept locks like motivation_inciting_cause, etc.
        motivation = data.get("motivation", {}) if isinstance(data.get("motivation"), dict) else {}
        for key in ["inciting_cause", "underlying_desire", "stakes", "obstacle"]:
            lock_key = f"motivation_{key}"
            if lock_key in locked_assumptions:
                motivation[key] = str(locked_assumptions[lock_key]).strip()
                applied[lock_key] = motivation[key]
        data["motivation"] = motivation

        return applied

    def _build_surfaced_assumptions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        assumptions: List[Dict[str, Any]] = []

        def add(field: str, value: str, confidence: float, reason: str):
            if value and value.lower() not in {"unclear", "unspecified"}:
                assumptions.append({
                    "field": field,
                    "value": value,
                    "confidence": confidence,
                    "reason": reason,
                })

        conf = data.get("confidence_scores", {}) if isinstance(data.get("confidence_scores"), dict) else {}
        speaker = data.get("speaker", {}) if isinstance(data.get("speaker"), dict) else {}
        world = data.get("world_assumptions", {}) if isinstance(data.get("world_assumptions"), dict) else {}

        # MEANING-LEVEL surfaced assumptions only. No architecture_style /
        # characteristic_setting — those are downstream concerns now.
        add("speaker_identity", speaker.get("identity", ""), conf.get("speaker", 0.7), "Inferred from voice and address pattern.")
        add("speaker_gender", speaker.get("gender", ""), conf.get("speaker", 0.7), "Inferred from textual cues.")
        add("speaker_cultural_background", speaker.get("cultural_background", ""), conf.get("cultural", 0.7), "Inferred from language and cultural cues.")
        add("geography", world.get("geography", ""), conf.get("cultural", 0.7), "Inferred from cultural and environmental cues.")
        add("timeline_nature", world.get("timeline_nature", ""), conf.get("narrative_mode", 0.7), "Inferred from how time behaves in the meaning of the piece.")
        add("location_dna", data.get("location_dna", ""), conf.get("cultural", 0.7), "Inferred from cultural markers.")
        add("narrative_mode", data.get("narrative_mode", ""), conf.get("narrative_mode", 0.7), "Inferred from semantic mode and structure.")

        # WHY block — surface motivation only when confidence is low enough that
        # the user should review it on the METAMAN Dialogue screen.
        motivation = data.get("motivation", {}) if isinstance(data.get("motivation"), dict) else {}
        m_conf = self._repair_confidence(motivation.get("confidence") or conf.get("motivation"))
        if m_conf < 0.75:
            for key in ("inciting_cause", "underlying_desire", "stakes", "obstacle"):
                add(
                    f"motivation_{key}",
                    motivation.get(key, ""),
                    m_conf,
                    "Inferred from emotional arc, lyrics, and cultural context.",
                )

        return assumptions

    def _repair_ambiguity(self, data: Dict[str, Any], has_repetition: bool) -> None:
        if not isinstance(data.get("ambiguity_flags"), list):
            data["ambiguity_flags"] = []

        cleaned_flags: List[str] = []
        cleaned_details: List[Dict[str, Any]] = []

        for x in data["ambiguity_flags"]:
            if isinstance(x, dict):
                field = self._ensure_string(x.get("field"), "unspecified")
                reason = self._ensure_string(x.get("reason"), "unclear")
                confidence = self._repair_intensity(x.get("confidence"))
                cleaned_details.append({"field": field, "reason": reason, "confidence": confidence})
                cleaned_flags.append(f"{field}: {reason}")
            elif str(x).strip():
                s = str(x).strip()
                cleaned_flags.append(s)
                cleaned_details.append({"field": "general", "reason": s, "confidence": 0.5})

        if has_repetition and not any(
            ("repetition" in s.lower() or "chorus" in s.lower() or "hook" in s.lower())
            for s in cleaned_flags
        ):
            note = "Repetition may indicate chorus/hook emphasis rather than a new narrative event."
            cleaned_flags.append(note)
            cleaned_details.append({"field": "structure", "reason": note, "confidence": 0.6})

        data["ambiguity_flags"] = cleaned_flags
        data["ambiguity_details"] = cleaned_details

    def _repair_confidence_scores(self, data: Dict[str, Any]) -> None:
        data["confidence"] = self._repair_confidence(data.get("confidence"))
        cs_in = data.get("confidence_scores") if isinstance(data.get("confidence_scores"), dict) else {}
        overall = self._repair_confidence(cs_in.get("overall") or data["confidence"])
        # Precedence for motivation confidence:
        #   1. confidence_scores.motivation if explicitly set by the LLM or by
        #      a downstream override (e.g. user resolved motivation in the
        #      METAMAN Dialogue, which bumps confidence_scores.motivation to 0.9)
        #   2. fall back to motivation.confidence stored by _repair_motivation
        #   3. finally fall back to overall confidence
        motivation_block = data.get("motivation") if isinstance(data.get("motivation"), dict) else {}
        motivation_conf = (
            cs_in.get("motivation")
            if cs_in.get("motivation") is not None
            else motivation_block.get("confidence")
        )
        data["confidence_scores"] = {
            "overall": overall,
            "cultural": self._repair_confidence(cs_in.get("cultural") or overall),
            "emotional": self._repair_confidence(cs_in.get("emotional") or overall),
            "speaker": self._repair_confidence(cs_in.get("speaker") or overall),
            "narrative_mode": self._repair_confidence(cs_in.get("narrative_mode") or overall),
            "motivation": self._repair_confidence(motivation_conf or overall),
        }

    def _normalize_by_genre(self, ctx: Dict[str, Any]) -> None:
        # Stage 2 owns MEANING normalization only. Visual / expression-channel
        # normalization (face vs body vs environment vs symbolic vs macro) is
        # the Narrative Intelligence engine's responsibility (Stage 3) — it
        # maps meaning to expression channels using its own strategy.
        # Here we only nudge per-line intensity for genres where the LLM tends
        # to under-rate it (ads).
        genre = (ctx.get("input_type") or "").lower()
        lms = ctx.get("line_meanings", [])

        if genre == "ad":
            for lm in lms:
                lm["intensity"] = max(lm.get("intensity", 0.5), 0.9)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    # ISO 639-1 → (primary, script) for languages we care about
    _LANGDETECT_MAP: Dict[str, tuple] = {
        "en": ("English", "Latin"),
        "fr": ("French", "Latin"),
        "es": ("Spanish", "Latin"),
        "pt": ("Portuguese", "Latin"),
        "de": ("German", "Latin"),
        "it": ("Italian", "Latin"),
        "nl": ("Dutch", "Latin"),
        "ru": ("Russian", "Cyrillic"),
        "uk": ("Ukrainian", "Cyrillic"),
        "pl": ("Polish", "Latin"),
        "sv": ("Swedish", "Latin"),
        "no": ("Norwegian", "Latin"),
        "da": ("Danish", "Latin"),
        "fi": ("Finnish", "Latin"),
        "ro": ("Romanian", "Latin"),
        "hu": ("Hungarian", "Latin"),
        "cs": ("Czech", "Latin"),
        "sk": ("Slovak", "Latin"),
        "hr": ("Croatian", "Latin"),
        "tr": ("Turkish", "Latin"),
        "id": ("Indonesian", "Latin"),
        "ms": ("Malay", "Latin"),
        "tl": ("Tagalog", "Latin"),
        "sw": ("Swahili", "Latin"),
        "so": ("Somali", "Latin"),
        "af": ("Afrikaans", "Latin"),
        "ko": ("Korean", "Hangul"),
        "ja": ("Japanese", "CJK"),
        "zh-cn": ("Mandarin", "CJK"),
        "zh-tw": ("Mandarin", "CJK"),
        "vi": ("Vietnamese", "Latin"),
        "th": ("Thai", "Thai"),
        "ar": ("Arabic", "Arabic"),
        "fa": ("Persian", "Arabic"),
        "ur": ("Urdu", "Arabic/Shahmukhi"),
        "pa": ("Punjabi", "Gurmukhi"),
        "hi": ("Hindi", "Devanagari"),
        "bn": ("Bengali", "Bengali"),
        "ta": ("Tamil", "Tamil"),
        "te": ("Telugu", "Telugu"),
        "am": ("Amharic", "Ethiopic"),
    }

    def _detect_language(self, text: str) -> Dict[str, str]:
        # --- Script-based fast paths (non-Latin scripts are unambiguous) ---
        if re.search(r"[\u0A00-\u0A7F]", text):
            return {"primary": "Punjabi", "script": "Gurmukhi", "dialect": ""}
        if re.search(r"[\u0600-\u06FF]", text):
            # Arabic script — distinguish Urdu/Punjabi from Arabic/Persian
            # (defer to langdetect below; fall back to Urdu/Punjabi if unavailable)
            try:
                from langdetect import detect as _ld_detect
                iso = _ld_detect(text[:400])
                primary, script = self._LANGDETECT_MAP.get(iso, ("Urdu/Punjabi", "Arabic/Shahmukhi"))
                return {"primary": primary, "script": script, "dialect": ""}
            except Exception:
                return {"primary": "Urdu/Punjabi", "script": "Arabic/Shahmukhi", "dialect": ""}
        if re.search(r"[\u0900-\u097F]", text):
            return {"primary": "Hindi", "script": "Devanagari", "dialect": ""}
        if re.search(r"[\u0980-\u09FF]", text):
            return {"primary": "Bengali", "script": "Bengali", "dialect": ""}
        if re.search(r"[\u0E00-\u0E7F]", text):
            return {"primary": "Thai", "script": "Thai", "dialect": ""}
        if re.search(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]", text):
            # CJK / Korean — let langdetect distinguish
            try:
                from langdetect import detect as _ld_detect
                iso = _ld_detect(text[:400])
                primary, script = self._LANGDETECT_MAP.get(iso, ("Unknown", "CJK"))
                return {"primary": primary, "script": script, "dialect": ""}
            except Exception:
                return {"primary": "Unknown", "script": "CJK", "dialect": ""}

        # --- Latin-script: use langdetect to identify the actual language ---
        if re.search(r"[A-Za-z]", text):
            try:
                from langdetect import detect as _ld_detect
                iso = _ld_detect(text[:400])
                primary, script = self._LANGDETECT_MAP.get(iso, ("English/Romanized", "Latin"))
                return {"primary": primary, "script": script, "dialect": ""}
            except Exception:
                return {"primary": "English/Romanized", "script": "Latin", "dialect": ""}

        return {"primary": "Unknown", "script": "", "dialect": ""}

    def _language_to_location_dna(self, primary_language: str) -> str:
        key = (primary_language or "").strip().lower()
        return self.LANGUAGE_LOCATION_DEFAULTS.get(key, "")

    def _normalize_for_match(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u0600-\u06FF\u0A00-\u0A7F]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _repair_confidence(self, value: Any) -> float:
        try:
            num = float(value)
            return max(0.0, min(1.0, num))
        except Exception:
            return self.DEFAULT_CONFIDENCE

    def _repair_intensity(self, value: Any) -> float:
        try:
            num = float(value)
            return max(0.0, min(1.0, num))
        except Exception:
            return 0.5

    def _ensure_string(self, value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        value = str(value).strip()
        return value if value else fallback

    def _dedupe(self, items: List[Any]) -> List[str]:
        seen = set()
        output = []
        for item in items:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                output.append(s)
        return output



# =============================================================================
# BACKWARD-COMPATIBILITY SHIM
# =============================================================================

class UnifiedContextEngine(MetaMindContextEngineFinal):
    """
    Back-compat wrapper around MetaMindContextEngineFinal.

    Old call sites:
        UnifiedContextEngine(api_key).generate(text=..., genre=..., pre_analysis=...)
    """

    async def generate(
        self,
        text: Optional[str] = None,
        genre: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
        raw_input: Any = None,
        hinted_type: Optional[str] = None,
        explicit_culture_pack: Optional[str] = None,
        locked_assumptions: Optional[Dict[str, Any]] = None,
        input_packet: Optional[Dict[str, Any]] = None,
        **_ignored_downstream: Any,
    ) -> Dict[str, Any]:
        # PIPELINE CHAIN RULE: Stage 2 must not consume any downstream
        # parameters (e.g. style_profile from Stage 4). Unexpected kwargs are
        # absorbed so legacy call sites do not crash, and logged loudly so
        # the chain violation gets cleaned up at the call site.
        if _ignored_downstream:
            logger.warning(
                "Context Engine shim received downstream kwargs %s — "
                "chain violation; caller must stop passing these.",
                sorted(_ignored_downstream.keys()),
            )
        if raw_input is None:
            raw_input = text
        if hinted_type is None:
            hinted_type = genre

        return await super().generate(
            raw_input=raw_input,
            hinted_type=hinted_type,
            pre_analysis=pre_analysis,
            explicit_culture_pack=explicit_culture_pack,
            locked_assumptions=locked_assumptions,
            input_packet=input_packet,
        )