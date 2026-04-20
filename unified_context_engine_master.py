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
                "geography": "rural Punjab (South Asia) — village setting, agricultural landscape",
                "architecture_style": (
                    "kuchha (mud-plastered) village architecture — thick bare ochre/sand-toned mud walls, "
                    "flat clay rooftops accessible by exterior stone or mud staircases, small deep-set windows "
                    "with blue or turquoise painted wooden shutters, heavy carved wooden doors painted blue, "
                    "smooth plastered parapet walls, no glass or modern cladding; "
                    "style reference: authentic T-Series Punjabi folk music video locations, not a studio set"
                ),
                "characteristic_setting": (
                    "clean swept earthen vehra (open courtyard) — packed bare mud floor, "
                    "borders of thick kuchha walls, charpai (rope-strung wooden bed) in open air, "
                    "terracotta matkas near the entrance, mustard or wheat fields visible beyond the compound wall, "
                    "open sky overhead, golden or warm afternoon light; "
                    "no concrete or tile, no synthetic materials"
                ),
                "cultural_dna": "Punjabi rural lament",
            },
            "appearance_defaults": {
                "ethnicity": "South Asian (Punjabi)",
                "complexion": "warm wheatish to tan",
                "wardrobe": "phulkari dupatta, salwar-kameez or kurta-pajama; turban for adult men where appropriate",
                "grooming": "kohl-lined eyes, simple traditional jewelry, plaited or covered hair for women",
            },
            "visual_restrictions": [
                "Avoid generic Western suburban visuals.",
                "Do not remove rural material authenticity — no concrete, tile, or modern finishes.",
                "Walls must be bare plastered mud (kuchha), not brick or painted plaster.",
                "Courtyard floor must be packed earth, not tile or concrete.",
                "Do not render speakers with East Asian, European, or African features unless the text explicitly demands it.",
                "Do not use Rajasthani haveli ornamentation — Punjab village aesthetic is plain, massive, and earthen.",
            ],
            "common_misinterpretations": [
                "Do not treat agrarian or seasonal references as generic decoration.",
                "Do not flatten feminine domestic imagery into random rustic props.",
                "Do not substitute a generic 'rural Indian village' — Punjab village architecture has specific ochre mud-wall identity.",
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
            "appearance_defaults": {
                "ethnicity": "South Asian (Punjabi)",
                "complexion": "warm wheatish to tan",
                "wardrobe": "contemporary mixed wardrobe; Punjabi cues in homeland scenes (dupatta, kurta), modern attire abroad",
                "grooming": "contemporary; subtle traditional cues persist",
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
            "appearance_defaults": {
                "ethnicity": "South Asian (Urdu cultural sphere)",
                "complexion": "warm tones",
                "wardrobe": "kurta-shalwar, sherwani, dupatta; subdued elegant cloth",
                "grooming": "restrained classical elegance",
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
            "appearance_defaults": {
                "ethnicity": "South Asian",
                "complexion": "warm tones",
                "wardrobe": "kurta-shalwar, qawwal attire, prayer cap; modest devotional dress",
                "grooming": "traditional, devotional restraint",
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
            "appearance_defaults": {
                "ethnicity": "inferred from lyric content and language — do not default to any single culture",
                "complexion": "inferred from cultural context",
                "wardrobe": "culturally appropriate to the song's world; inferred from language and geography",
                "grooming": "culturally appropriate; inferred from context",
            },
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

    LANGUAGE_APPEARANCE_DEFAULTS = {
        # South Asian
        "punjabi": {
            "ethnicity": "South Asian (Punjabi)",
            "complexion": "warm wheatish to tan",
            "wardrobe": "phulkari dupatta, salwar-kameez or kurta-pajama; turban for adult men where appropriate",
            "grooming": "kohl-lined eyes, simple traditional jewelry, plaited or covered hair for women",
        },
        "urdu": {
            "ethnicity": "South Asian (Urdu cultural sphere)",
            "complexion": "warm tones",
            "wardrobe": "kurta-shalwar, sherwani, dupatta",
            "grooming": "restrained classical elegance",
        },
        "urdu/punjabi": {
            "ethnicity": "South Asian",
            "complexion": "warm tones",
            "wardrobe": "kurta-shalwar, dupatta, regional cloth",
            "grooming": "traditional South Asian",
        },
        "hindi": {
            "ethnicity": "South Asian (North/Central Indian)",
            "complexion": "warm tones",
            "wardrobe": "salwar-kameez, sari, kurta-pajama; regionally faithful",
            "grooming": "traditional South Asian",
        },
        "bengali": {
            "ethnicity": "South Asian (Bengali)",
            "complexion": "warm tones",
            "wardrobe": "sari, dhoti-kurta; regional Bengali cloth",
            "grooming": "traditional South Asian",
        },
        "tamil": {
            "ethnicity": "South Asian (Tamil)",
            "complexion": "warm to deep brown tones",
            "wardrobe": "saree, veshti-jubba; traditional Tamil cloth",
            "grooming": "traditional South Indian, jasmine flowers in hair for women",
        },
        "telugu": {
            "ethnicity": "South Asian (Telugu)",
            "complexion": "warm to deep brown tones",
            "wardrobe": "saree, dhoti; regional Telugu attire",
            "grooming": "traditional South Indian",
        },
        # European
        "english": {
            "ethnicity": "inferred from lyric content (default: Western European)",
            "complexion": "fair to medium tones — defer to lyric evidence",
            "wardrobe": "contemporary casual or period-appropriate Western attire; region-specific if clear from lyrics",
            "grooming": "contemporary Western; defer to lyric evidence",
        },
        "english/romanized": {
            "ethnicity": "inferred from lyric content — do not default to any single ethnicity",
            "complexion": "defer to lyric or cultural evidence",
            "wardrobe": "contemporary or period attire matched to lyric world",
            "grooming": "matched to lyric world",
        },
        "french": {
            "ethnicity": "French / Western European (defer to lyric evidence)",
            "complexion": "fair to medium tones",
            "wardrobe": "contemporary French casual or period-appropriate French attire",
            "grooming": "contemporary French; understated elegance",
        },
        "spanish": {
            "ethnicity": "Spanish / Latin American (defer to lyric evidence for specific region)",
            "complexion": "olive to medium-brown tones",
            "wardrobe": "contemporary casual or traditional regional attire (flamenco cues only if explicit)",
            "grooming": "contemporary; warm and expressive",
        },
        "portuguese": {
            "ethnicity": "Portuguese / Brazilian (defer to lyric evidence)",
            "complexion": "fair to olive to medium-brown tones",
            "wardrobe": "contemporary casual or period attire; region-specific if clear",
            "grooming": "contemporary; warm and natural",
        },
        "german": {
            "ethnicity": "German / Central European (defer to lyric evidence)",
            "complexion": "fair to medium tones",
            "wardrobe": "contemporary casual or period-appropriate attire",
            "grooming": "contemporary Central European; clean and understated",
        },
        "italian": {
            "ethnicity": "Italian / Southern European (defer to lyric evidence)",
            "complexion": "olive to medium tones",
            "wardrobe": "contemporary Italian casual or period attire",
            "grooming": "contemporary; warm Mediterranean style",
        },
        "dutch": {
            "ethnicity": "Dutch / Northern European (defer to lyric evidence)",
            "complexion": "fair tones",
            "wardrobe": "contemporary casual Western attire",
            "grooming": "contemporary Northern European; practical and understated",
        },
        # East Asian
        "korean": {
            "ethnicity": "Korean / East Asian",
            "complexion": "fair to light golden tones",
            "wardrobe": "contemporary Korean casual or hanbok for formal/traditional scenes",
            "grooming": "contemporary Korean; clean and polished",
        },
        "japanese": {
            "ethnicity": "Japanese / East Asian",
            "complexion": "fair to light golden tones",
            "wardrobe": "contemporary Japanese casual or kimono/yukata for traditional scenes",
            "grooming": "contemporary Japanese; clean and restrained",
        },
        "mandarin": {
            "ethnicity": "Chinese / East Asian",
            "complexion": "fair to light golden tones",
            "wardrobe": "contemporary Chinese casual or qipao/hanfu for traditional scenes",
            "grooming": "contemporary Chinese; clean and polished",
        },
        "chinese": {
            "ethnicity": "Chinese / East Asian",
            "complexion": "fair to light golden tones",
            "wardrobe": "contemporary Chinese casual or traditional attire as appropriate",
            "grooming": "contemporary Chinese; clean and polished",
        },
        "cantonese": {
            "ethnicity": "Cantonese / Southern Chinese / East Asian",
            "complexion": "fair to light golden tones",
            "wardrobe": "contemporary Cantonese casual or traditional attire",
            "grooming": "contemporary; clean and polished",
        },
        # Middle Eastern / African
        "arabic": {
            "ethnicity": "Arab / Middle Eastern / North African (defer to lyric evidence for sub-region)",
            "complexion": "olive to warm brown tones",
            "wardrobe": "contemporary casual or traditional thobe/abaya/jalabiya as appropriate to lyric world",
            "grooming": "contemporary; warm and expressive",
        },
        "persian": {
            "ethnicity": "Persian / Iranian",
            "complexion": "olive to warm tones",
            "wardrobe": "contemporary Iranian casual or traditional attire; headscarves where culturally appropriate",
            "grooming": "contemporary; warm and elegant",
        },
        "turkish": {
            "ethnicity": "Turkish / Anatolian",
            "complexion": "olive to warm tones",
            "wardrobe": "contemporary Turkish casual or traditional attire",
            "grooming": "contemporary; warm and expressive",
        },
        "swahili": {
            "ethnicity": "East African (defer to lyric evidence for specific nation)",
            "complexion": "warm to deep brown tones",
            "wardrobe": "contemporary East African casual or kanga/kitenge; regionally faithful",
            "grooming": "contemporary; warm and natural",
        },
        "amharic": {
            "ethnicity": "Ethiopian / East African",
            "complexion": "warm to deep brown tones",
            "wardrobe": "contemporary Ethiopian casual or traditional habesha kemis/netela",
            "grooming": "contemporary; warm and natural",
        },
        # South-East Asian
        "thai": {
            "ethnicity": "Thai / South-East Asian",
            "complexion": "warm golden to tan tones",
            "wardrobe": "contemporary Thai casual or traditional chut thai for formal scenes",
            "grooming": "contemporary Thai; warm and polished",
        },
        "indonesian": {
            "ethnicity": "Indonesian / South-East Asian (defer to island/region from lyrics)",
            "complexion": "warm golden to brown tones",
            "wardrobe": "contemporary Indonesian casual or batik/kebaya; regionally faithful",
            "grooming": "contemporary; warm and natural",
        },
        "tagalog": {
            "ethnicity": "Filipino / South-East Asian",
            "complexion": "warm golden to tan tones",
            "wardrobe": "contemporary Filipino casual or barong tagalog/filipiniana for formal scenes",
            "grooming": "contemporary Filipino; warm and polished",
        },
    }

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
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pre_analysis = pre_analysis or {}
        locked_assumptions = locked_assumptions or {}

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
            "parsed_input": parsed_input,
            "routing": routed,
            "language": language_info,
            "genre_directive": genre_cfg["directive"],
            "genre_default_visual_constraints": genre_cfg["default_visual_constraints"],
            "pre_analysis": pre_analysis,
            "culture_pack_id": culture_pack_id,
            "culture_pack": culture_pack,
            "active_metaphors": active_metaphors,
            "locked_assumptions": locked_assumptions,
            "style_profile": style_profile or {},
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
        style_profile = hard_logic.get("style_profile") or {}

        metaphor_text = "\n".join(
            f"- '{k}' means: {v}" for k, v in active_metaphors.items()
        ) if active_metaphors else "None"

        # Build VISUAL STYLE DIRECTIVE block if a style_profile was chosen
        style_directive_block = ""
        if style_profile:
            prod = style_profile.get("production") or {}
            cin = style_profile.get("cinematic") or {}
            prod_directive = prod.get("context_directive", "")
            cin_directive = cin.get("context_directive", "")
            if prod_directive or cin_directive:
                style_directive_block = f"""
VISUAL STYLE DIRECTIVE (user-selected — must govern all visual interpretation):
- Production Style: {prod.get("label", "")} — {prod_directive}
- Cinematic Style: {cin.get("label", "")} — {cin_directive}
All visual_constraints, shot_type, and mood recommendations must conform to the above style directives.
"""

        # Build VOCAL GENDER DIRECTIVE — hard fact derived from audio analysis
        # (or explicit user override). When present, the LLM MUST set
        # speaker.gender accordingly instead of guessing from the lyrics.
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
You are Qaivid MetaMind — a world-class literary analyst, dramaturg, and cultural interpreter.
Your job is to deeply read expressive input (song lyrics, poem, ghazal, qawwali, script, story,
voiceover, ad copy, documentary text) and produce one structured JSON object for downstream cinematic workflow.

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
{style_directive_block}
{vocal_gender_block}
CORE INTERPRETIVE PRINCIPLES:
1. Treat metaphors as metaphors. Do NOT literalize symbolic language.
2. For each line, separate literal, implied, emotional, and cultural meaning.
3. Classify each line's visualization_mode:
   - direct
   - indirect
   - symbolic
   - absorbed
   - performance_only
4. Be culturally precise. Do not flatten South Asian content into generic Western imagery.
5. Surface assumptions honestly.
6. Respect locked assumptions where provided.
7. line_meanings must cover every line in order.
8. world_assumptions should be inferred from textual evidence, not invented.
9. Return ONLY valid JSON.

5W FRAMEWORK — YOU MUST FILL ALL FIVE Ws:
- WHO  → speaker / addressee
- WHAT → core_theme / dramatic_premise / narrative_spine
- WHEN → world_assumptions.era / season / characteristic_time / emotional_arc
- WHERE → world_assumptions.geography / location_dna
- WHY  → motivation (REQUIRED — do not leave blank)
The "motivation" object captures the emotional engine of the song. You MUST fill:
  - inciting_cause: the concrete event/loss/longing/decision that triggered this expression
  - underlying_desire: what the speaker ultimately wants
  - stakes: what is at risk if the desire is not met
  - obstacle: what stands between the speaker and the desire
  - confidence: 0..1 — how confident you are in the motivation reading
If the lyrics are abstract or symbolic, infer motivation from the emotional arc + cultural context — never leave fields blank.

REQUIRED JSON SHAPE:
{{
  "input_profile": {{
    "recognized_type": "string",
    "raw_detected_type": "string",
    "is_mixed_input": false,
    "structure_quality": "string",
    "source_format": "string",
    "language": {{
      "primary": "string",
      "script": "string",
      "dialect": "string"
    }},
    "analysis_confidence": 0.0
  }},
  "input_type": "string",
  "language": "string",
  "narrative_mode": "string",
  "location_dna": "string",
  "genre_directive": "string",
  "core_theme": "string",
  "dramatic_premise": "string",
  "narrative_spine": "string",
  "speaker": {{
    "identity": "string",
    "gender": "string",
    "age_range": "string",
    "social_role": "string",
    "emotional_state": "string",
    "relationship_to_addressee": "string",
    "ethnicity": "string  (be culturally faithful to the actual language and cultural world of the lyrics — e.g. 'French (Caucasian)', 'Korean', 'South Asian (Punjabi)', 'West African', etc. Do NOT default to any single ethnicity; infer from the lyric language, geography, and cultural cues)",
    "complexion": "string  (culturally faithful to the detected ethnicity — e.g. fair, warm wheatish, warm tan, deep brown, olive, etc.)",
    "wardrobe": "string  (concrete clothing description anchored in the cultural framework)",
    "grooming": "string  (hair, jewelry, facial hair, kohl, etc. — culturally faithful)"
  }},
  "addressee": {{
    "identity": "string",
    "relationship": "string",
    "presence": "string"
  }},
  "world_assumptions": {{
    "geography": "string  (broad geographic anchor for the whole song, e.g. 'Punjab region, South Asia', 'rural France', 'urban Seoul')",
    "era": "string",
    "season": "string",
    "characteristic_time": "string  (general emotional time-feel of the song, e.g. 'dawn', 'dusk', 'late night' — this is a cultural default, NOT the time of day for any specific shot; individual scene times are set by the creative brief)",
    "social_context": "string",
    "economic_context": "string",
    "architecture_style": "string  (MANDATORY: describe in specific physical construction materials and visual form — NOT a generic label. Examples: for Punjabi village: 'kuchha mud-plastered walls, flat clay roof, blue-painted wooden doors and shutters, exterior staircase, smooth ochre plaster'; for French countryside: 'stone farmhouse, timber-frame shutters, slate roof'; for urban Seoul: 'glass and steel high-rise, neon-lit street level'. Never write just 'traditional' or 'rustic' or 'domestic' — always specify the material and construction type.)",
    "characteristic_setting": "string  (MANDATORY: describe with specific physical environment details — NOT a generic category label. Examples: for Punjabi village: 'clean swept earthen courtyard, charpai in open air, terracotta matkas near entrance, mustard fields beyond the compound wall'; for French countryside: 'stone-walled farmyard, gravel path, vegetable garden, hay barn'; for urban Seoul: 'neon-lit alley, convenience store frontage, street stall'. Never write just 'courtyard-oriented home' or 'domestic interior' — always describe what is physically visible.)"
  }},
  "emotional_arc": {{
    "opening": "string",
    "development": "string",
    "climax": "string",
    "resolution": "string"
  }},
  "motivation": {{
    "inciting_cause": "string  (the WHY: what event/loss/longing/decision triggered the speaker to express this)",
    "underlying_desire": "string  (what the speaker ultimately wants — reunion, freedom, forgiveness, recognition, etc.)",
    "stakes": "string  (what is at risk if the desire is not met — loneliness, dishonor, lost love, lost identity, etc.)",
    "obstacle": "string  (what stands between the speaker and the desire — distance, time, social rules, the addressee's silence, etc.)",
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
      "meaning": "string",
      "function": "string",
      "repeat_status": "original|repeat",
      "intensity": 0.5,
      "expression_mode": "face|body|environment|symbolic|macro",
      "visualization_mode": "direct|indirect|symbolic|absorbed|performance_only",
      "visual_suitability": "high|medium|low"
    }}
  ],
  "motifs": ["string"],
  "motif_map": {{}},
  "entities": [
    {{ "name": "string", "type": "person|object|place|symbol", "role": "string" }}
  ],
  "literary_devices": ["string"],
  "visual_constraints": ["string"],
  "restrictions": ["string"],
  "surfaced_assumptions": [
    {{
      "field": "string",
      "value": "string",
      "confidence": 0.0,
      "reason": "string"
    }}
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
        if "punjabi" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Punjabi folk / Sufi / qissa tradition. "
                "Common motifs: separation, beloved, agrarian land, rivers, courtyards, "
                "domestic space, migration, oral-song memory. "
                "VISUAL GROUNDING (mandatory): locations must be authentic Punjabi village — "
                "kuchha mud-plastered houses with thick ochre/sand walls, clean swept earthen courtyards (vehra), "
                "flat clay rooftops with exterior staircases, blue or turquoise painted wooden doors and shutters, "
                "terracotta pots, charpai beds in open courtyards, mustard or wheat fields beyond compound walls. "
                "Style reference: T-Series / Punjabi folk music video aesthetic — no studio backdrops, "
                "no concrete or tile, no Rajasthani ornamentation. "
                "Set world_assumptions.architecture_style and world_assumptions.characteristic_setting accordingly."
            )
        elif "urdu" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Urdu ghazal / nazm / qawwali tradition. "
                "Common motifs: hijr, wasl, mehboob, divine beloved, silence, existential longing."
            )
        elif "hindi" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Hindi film song / bhajan / folk tradition. "
                "Common motifs: love, separation, devotion, nature, seasons, inner longing. "
                "Ground visuals in North/Central Indian cultural world unless lyrics suggest otherwise."
            )
        elif "bengali" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Bengali song tradition (Rabindra Sangeet / Baul / folk). "
                "Common motifs: river, rain, seasons, philosophical longing, nature, rural Bengal. "
                "Ground visuals in Bengali cultural world."
            )
        elif "tamil" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Tamil song / Carnatic / folk tradition. "
                "Common motifs: nature, devotion, love, separation, classical dance. "
                "Ground visuals in South Indian (Tamil) cultural world."
            )
        elif "french" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: French chanson / pop / classical song tradition. "
                "Ground visuals in French or French-speaking cultural world — countryside, city, domestic interiors. "
                "Do not impose South Asian or East Asian aesthetics."
            )
        elif "spanish" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Spanish / Latin American song tradition (bolero, cumbia, flamenco, pop). "
                "Ground visuals in the relevant Spanish-speaking cultural world — identify region from lyric content. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "portuguese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Portuguese / Brazilian song tradition (fado, bossa nova, sertanejo, pop). "
                "Ground visuals in the relevant Portuguese-speaking cultural world. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "korean" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Korean song tradition (K-pop, ballad, trot, folk). "
                "Ground visuals in Korean cultural world — contemporary urban, countryside, seasonal nature. "
                "Do not impose South Asian or Western aesthetics."
            )
        elif "japanese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Japanese song tradition (J-pop, enka, folk, shoegaze). "
                "Ground visuals in Japanese cultural world — seasonal nature, urban Japan, traditional spaces. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "mandarin" in lang or "chinese" in lang or "cantonese" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Chinese song tradition (Mandopop, Cantopop, folk, classical). "
                "Ground visuals in Chinese cultural world — identify mainland / Hong Kong / Taiwan from lyric content. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "arabic" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Arabic song tradition (tarab, khaleeji, pop, mawwal). "
                "Ground visuals in Arabic-speaking cultural world — identify region (Gulf, Levant, Egypt, Maghreb) from lyric content. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "persian" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Persian / Farsi song tradition (classical, pop, ghazal). "
                "Ground visuals in Iranian / Persian cultural world. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "turkish" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Turkish song tradition (Türkü folk, arabesk, pop). "
                "Ground visuals in Turkish / Anatolian cultural world. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "english" in lang or "romanized" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Latin-script content — first identify the ACTUAL language from the lyric text itself "
                "(French, Spanish, Italian, German, English, Korean romanized, etc.). "
                "Then ground ALL visual assumptions (ethnicity, geography, wardrobe, props, locations) "
                "faithfully in that language's cultural world. "
                "Do not default to South Asian, East Asian, or American aesthetics unless the lyrics demand it."
            )
        elif "german" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: German-language song tradition. "
                "Ground visuals in German / Austrian / Swiss cultural world as appropriate. "
                "Do not impose South Asian or other aesthetics."
            )
        elif "italian" in lang:
            cultural_hint = (
                "CULTURAL FRAMEWORK: Italian song tradition (canzone, opera pop, folk). "
                "Ground visuals in Italian cultural world. "
                "Do not impose South Asian or other aesthetics."
            )

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

OPTIONAL PRE-ANALYSIS:
{json.dumps(hard_logic.get("pre_analysis", {}), ensure_ascii=False)}

REQUIREMENTS:
- line_meanings must contain exactly one entry per indexed line, in order.
- Fill literal_meaning, implied_meaning, emotional_meaning, and cultural_meaning for every line.
- Choose visualization_mode honestly.
- Mark repeated lines with repeat_status="repeat".
- Keep narrative_spine compact and storyboard-friendly.
- world_assumptions should be inferred, not invented.
- visual_constraints should help downstream engines stay faithful.
- confidence_scores must be honest.
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
        self._repair_motifs(data)
        self._repair_entities(data)
        self._repair_literary_devices(data)
        self._repair_visual_constraints_and_restrictions(data, genre_cfg, culture_pack, routed["has_repetition"])

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
        defaults = {
            "input_profile": {},
            "speaker": {},
            "addressee": {},
            "world_assumptions": {},
            "emotional_arc": {},
            "motivation": {},
            "line_meanings": [],
            "motif_map": {},
            "entities": [],
            "literary_devices": [],
            "visual_constraints": [],
            "restrictions": [],
            "surfaced_assumptions": [],
            "locked_assumptions": {},
            "ambiguity_flags": [],
            "confidence_scores": {},
        }
        for key, default in defaults.items():
            data.setdefault(key, default)

    def _repair_speaker(self, data: Dict[str, Any], culture_pack: Dict[str, Any], language: Dict[str, str], vocal_gender: Optional[str] = None) -> None:
        speaker_in = data["speaker"] if isinstance(data.get("speaker"), dict) else {}
        appearance_defaults = culture_pack.get("appearance_defaults", {}) or self._language_to_appearance(language.get("primary", "")) or {}

        # If we have a high-confidence vocal gender from audio analysis (or user
        # override), it always wins over an LLM-emitted "Unclear"/"Unspecified".
        llm_gender = self._ensure_string(speaker_in.get("gender"), "")
        if vocal_gender and llm_gender.strip().lower() in ("", "unclear", "unspecified", "unknown", "n/a"):
            llm_gender = vocal_gender
        elif not llm_gender:
            llm_gender = vocal_gender or "Unclear"

        data["speaker"] = {
            "identity": self._ensure_string(speaker_in.get("identity"), "Unclear speaker"),
            "gender": llm_gender,
            "age_range": self._ensure_string(speaker_in.get("age_range"), "Unclear"),
            "emotional_state": self._ensure_string(speaker_in.get("emotional_state"), "Emotionally charged"),
            "social_role": self._ensure_string(speaker_in.get("social_role"), "Unclear"),
            "relationship_to_addressee": self._ensure_string(speaker_in.get("relationship_to_addressee"), "Unclear"),
            "ethnicity": self._ensure_string(speaker_in.get("ethnicity"), appearance_defaults.get("ethnicity", "Unspecified")),
            "complexion": self._ensure_string(speaker_in.get("complexion"), appearance_defaults.get("complexion", "Unspecified")),
            "wardrobe": self._ensure_string(speaker_in.get("wardrobe"), appearance_defaults.get("wardrobe", "Unspecified")),
            "grooming": self._ensure_string(speaker_in.get("grooming"), appearance_defaults.get("grooming", "Unspecified")),
        }

    def _repair_addressee(self, data: Dict[str, Any]) -> None:
        addressee_in = data["addressee"] if isinstance(data.get("addressee"), dict) else {}
        data["addressee"] = {
            "identity": self._ensure_string(addressee_in.get("identity"), "Unclear addressee"),
            "relationship": self._ensure_string(addressee_in.get("relationship"), "Unclear"),
            "presence": self._ensure_string(addressee_in.get("presence"), "Unclear"),
        }

    def _repair_world_assumptions(self, data: Dict[str, Any], culture_pack: Dict[str, Any]) -> None:
        wa = data["world_assumptions"] if isinstance(data.get("world_assumptions"), dict) else {}
        defaults = culture_pack.get("world_defaults", {})
        # Read new field names first; fall back to old names for backward compat
        # with existing projects that have the pre-rename field names stored in DB.
        char_time = (
            wa.get("characteristic_time")
            or wa.get("time_of_day")
        )
        char_setting = (
            wa.get("characteristic_setting")
            or wa.get("domestic_setting")
        )
        # Culture pack is authoritative for visual specifics (architecture_style,
        # characteristic_setting). The LLM always returns vague generics here
        # ("Courtyard-oriented home") which are useless downstream. When the pack
        # has specific values, use them unconditionally — they represent domain
        # knowledge the LLM cannot reliably reproduce.
        pack_arch = defaults.get("architecture_style", "")
        pack_setting = (
            defaults.get("characteristic_setting")
            or defaults.get("domestic_setting", "")
        )
        data["world_assumptions"] = {
            "geography": self._ensure_string(wa.get("geography"), defaults.get("geography", "Unspecified")),
            "era": self._ensure_string(wa.get("era"), "Unspecified"),
            "season": self._ensure_string(wa.get("season"), "Unspecified"),
            "characteristic_time": self._ensure_string(char_time, "Unspecified"),
            "social_context": self._ensure_string(wa.get("social_context"), "Unspecified"),
            "economic_context": self._ensure_string(wa.get("economic_context"), "Unspecified"),
            "architecture_style": (
                pack_arch if pack_arch
                else self._ensure_string(wa.get("architecture_style"), "Unspecified")
            ),
            "characteristic_setting": (
                pack_setting if pack_setting
                else self._ensure_string(char_setting, "Unspecified")
            ),
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
                "expression_mode": self._repair_expression_mode(current.get("expression_mode")),
                "visualization_mode": self._repair_visualization_mode(current.get("visualization_mode")),
                "visual_suitability": self._repair_visual_suitability(current.get("visual_suitability")),
            })

        return repaired

    def _repair_motifs(self, data: Dict[str, Any]) -> None:
        if not isinstance(data.get("motifs"), list):
            data["motifs"] = []
        data["motifs"] = [str(x).strip() for x in data["motifs"] if str(x).strip()][:12]

        mm_in = data.get("motif_map") if isinstance(data.get("motif_map"), dict) else {}
        cleaned_motif_map: Dict[str, List[int]] = {}
        for k, v in mm_in.items():
            if not isinstance(v, list):
                continue
            indices = []
            for x in v:
                try:
                    indices.append(int(x))
                except Exception:
                    pass
            if indices:
                cleaned_motif_map[str(k).strip()] = indices
        data["motif_map"] = cleaned_motif_map

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

    def _repair_visual_constraints_and_restrictions(
        self,
        data: Dict[str, Any],
        genre_cfg: Dict[str, Any],
        culture_pack: Dict[str, Any],
        has_repetition: bool,
    ) -> None:
        if not isinstance(data.get("visual_constraints"), list):
            data["visual_constraints"] = []
        constraints = [str(x).strip() for x in data["visual_constraints"] if str(x).strip()]
        constraints.extend(genre_cfg.get("default_visual_constraints", []))
        constraints.extend(culture_pack.get("visual_restrictions", []))
        data["visual_constraints"] = self._dedupe(constraints) or self._default_visual_constraints(has_repetition, genre_cfg)

        if not isinstance(data.get("restrictions"), list):
            data["restrictions"] = []
        restrictions = [str(x).strip() for x in data["restrictions"] if str(x).strip()]
        restrictions.extend(culture_pack.get("common_misinterpretations", []))
        data["restrictions"] = self._dedupe(restrictions)

    def _apply_locked_assumptions(self, data: Dict[str, Any], locked_assumptions: Dict[str, Any]) -> Dict[str, Any]:
        if not locked_assumptions:
            return {}

        applied: Dict[str, str] = {}
        world = data.get("world_assumptions", {})
        # Accept both old names (time_of_day, domestic_setting) and new names
        # (characteristic_time, characteristic_setting) from locked_assumptions
        # so existing locked projects don't lose their overrides.
        _world_key_aliases = {
            "time_of_day": "characteristic_time",
            "domestic_setting": "characteristic_setting",
        }
        for key in ["geography", "era", "season", "time_of_day", "characteristic_time",
                    "architecture_style", "domestic_setting", "characteristic_setting"]:
            if key in locked_assumptions:
                canonical = _world_key_aliases.get(key, key)
                world[canonical] = str(locked_assumptions[key]).strip()
                applied[canonical] = world[canonical]

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

        add("speaker_identity", speaker.get("identity", ""), conf.get("speaker", 0.7), "Inferred from voice and address pattern.")
        add("speaker_gender", speaker.get("gender", ""), conf.get("speaker", 0.7), "Inferred from textual cues.")
        add("geography", world.get("geography", ""), conf.get("cultural", 0.7), "Inferred from cultural and environmental cues.")
        add("architecture_style", world.get("architecture_style", ""), conf.get("cultural", 0.7), "Inferred from world assumptions.")
        add("location_dna", data.get("location_dna", ""), conf.get("cultural", 0.7), "Inferred from cultural markers.")
        add("narrative_mode", data.get("narrative_mode", ""), conf.get("narrative_mode", 0.7), "Inferred from semantic mode and structure.")

        # WHY block — surface motivation only when confidence is low enough that
        # the user should review it on the JARVIS Dialogue screen.
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
        #      JARVIS Dialogue, which bumps confidence_scores.motivation to 0.9)
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
        genre = (ctx.get("input_type") or "").lower()
        lms = ctx.get("line_meanings", [])

        if genre == "ad":
            for lm in lms:
                lm["intensity"] = max(lm.get("intensity", 0.5), 0.9)
                lm["expression_mode"] = "macro"
        elif genre == "script":
            for lm in lms:
                if lm.get("expression_mode") in {"symbolic", "environment"}:
                    lm["expression_mode"] = "face"
        elif genre in {"song", "poem", "ghazal", "qawwali"}:
            for lm in lms:
                if lm.get("expression_mode") == "macro":
                    lm["expression_mode"] = "symbolic"
        elif genre == "documentary":
            for lm in lms:
                if lm.get("expression_mode") == "symbolic":
                    lm["expression_mode"] = "environment"

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

    def _language_to_appearance(self, primary_language: str) -> Dict[str, str]:
        key = (primary_language or "").strip().lower()
        return self.LANGUAGE_APPEARANCE_DEFAULTS.get(key, {})

    def _normalize_for_match(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u0600-\u06FF\u0A00-\u0A7F]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _repair_visualization_mode(self, value: Any) -> str:
        allowed = {"direct", "indirect", "symbolic", "absorbed", "performance_only"}
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        return "indirect"

    def _repair_visual_suitability(self, value: Any) -> str:
        allowed = {"high", "medium", "low"}
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        return "medium"

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

    def _repair_expression_mode(self, value: Any) -> str:
        allowed = {"face", "body", "environment", "symbolic", "macro"}
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        return "environment"

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

    def _default_visual_constraints(self, has_repetition: bool, genre_cfg: Dict[str, Any]) -> List[str]:
        constraints = [
            "Do not invent major plot events not supported by the text.",
            "Keep emotional interpretation stronger than literal over-expansion.",
            "Preserve regional and cultural grounding where specified.",
        ]
        if has_repetition:
            constraints.append("Repeated lines should generally be treated as emotional emphasis, not as new plot events.")
        constraints.extend(genre_cfg.get("default_visual_constraints", []))
        return self._dedupe(constraints)


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
        style_profile: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
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
            style_profile=style_profile,
        )