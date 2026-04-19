"""
Culture Packs Service
Deterministic cultural enrichment rules. No AI calls.
Each pack contains trigger words, metaphor mappings, social-role logic, visual restrictions, etc.
"""
from typing import Dict, List, Any, Optional

CULTURE_PACKS: Dict[str, Dict[str, Any]] = {
    "punjabi_rural_lament": {
        "id": "punjabi_rural_lament",
        "name": "Punjabi Rural Lament",
        "description": "For songs of separation, longing, and rural Punjabi domestic life",
        "trigger_words": ["charkha", "trinjan", "vichora", "birha", "pardes", "pind", "khet", "lassi",
                          "chunni", "phulkari", "charpai", "kotha", "haveli", "beri", "pipal",
                          "chann", "mahiya", "sajan", "dhol", "ranjha", "heer", "jugni"],
        "metaphor_map": {
            "charkha": "waiting, devotion, feminine patience",
            "pardes": "foreign land, exile, migration separation",
            "beri": "shade tree of meeting, courtship, childhood home",
            "pipal": "village center, gossip, community witness",
            "chann": "beloved, moon, beauty",
            "phulkari": "feminine labor, beauty, bridal preparation",
            "trinjan": "women's spinning circle, female solidarity, storytelling space",
        },
        "social_roles": {"speaker": "young woman or elder woman", "addressee": "absent beloved or self"},
        "visual_settings": {
            "geography": "Punjab plains, rural village",
            "architecture": "haveli, kotha, courtyard with charpai",
            "season_default": "monsoon or winter dawn",
            "colors": ["mustard yellow", "terracotta", "deep green", "white cotton"],
        },
        "restrictions": ["avoid modern urban settings", "avoid western furniture", "maintain modesty in female depiction"],
        "common_misinterpretations": [
            "Do not literalize 'pardes' as just 'foreign country' — it means emotional exile",
            "Charkha is not just a spinning wheel — it symbolizes devotion and waiting",
        ],
    },
    "punjabi_diaspora_memory": {
        "id": "punjabi_diaspora_memory",
        "name": "Punjabi Diaspora Memory",
        "description": "For songs about homeland memory, migration, dual identity",
        "trigger_words": ["yaad", "pardes", "vilayat", "watan", "ghar", "maa", "baapu",
                          "airport", "visa", "dollar", "phone", "video call"],
        "metaphor_map": {
            "pardes": "not just foreign land but identity fracture",
            "ghar": "not just house but emotional anchor of belonging",
            "maa": "homeland embodied in mother figure",
        },
        "social_roles": {"speaker": "migrant, diaspora youth", "addressee": "family back home or self"},
        "visual_settings": {
            "geography": "split between western city and Punjab village",
            "architecture": "urban apartment vs rural courtyard",
            "season_default": "contrasting — grey city winter vs golden Punjab autumn",
            "colors": ["grey concrete", "warm amber", "cool blue", "golden wheat"],
        },
        "restrictions": ["maintain dignity of both worlds", "avoid poverty tourism"],
        "common_misinterpretations": [
            "Nostalgia is not just sadness — it often carries pride and defiance",
        ],
    },
    "urdu_philosophical_ghazal": {
        "id": "urdu_philosophical_ghazal",
        "name": "Urdu Philosophical Ghazal",
        "description": "For ghazals exploring love, existence, divine connection, worldly sorrow",
        "trigger_words": ["ishq", "dard", "sharab", "saqi", "mehfil", "deewar", "aaina",
                          "shama", "parwana", "gul", "bulbul", "bazm", "raqs",
                          "khuda", "duniya", "fanaa", "bekasi"],
        "metaphor_map": {
            "sharab": "not alcohol — spiritual intoxication, truth, ecstasy",
            "saqi": "not bartender — divine guide, beloved, god",
            "shama": "light of truth, attraction, sacrifice",
            "parwana": "the lover, the seeker who burns for truth",
            "aaina": "self-reflection, truth, confrontation with reality",
            "gul": "beauty, beloved, ephemeral life",
            "bulbul": "the poet, the longing voice",
        },
        "social_roles": {"speaker": "poet-philosopher, lover-seeker", "addressee": "beloved or divine or self"},
        "visual_settings": {
            "geography": "Mughal-era aesthetic, old city, garden, mehfil",
            "architecture": "arched doorways, jharokha windows, marble floors, lantern-lit spaces",
            "season_default": "evening or night, autumn",
            "colors": ["deep maroon", "ivory", "gold leaf", "midnight blue"],
        },
        "restrictions": ["never literalize spiritual metaphors as physical drinking",
                         "maintain poetic ambiguity — do not flatten meaning"],
        "common_misinterpretations": [
            "Sharab/wine in ghazal is almost never about alcohol — it is about spiritual ecstasy",
            "Ishq is not romantic love alone — it is existential devotion",
        ],
    },
    "devotional_qawwali": {
        "id": "devotional_qawwali",
        "name": "Devotional / Qawwali",
        "description": "For Sufi qawwali, devotional, spiritual performance content",
        "trigger_words": ["allah", "khwaja", "dargah", "urs", "murshid", "faqir",
                          "maula", "sufi", "raqs", "wajd", "dam", "qalandar", "naat"],
        "metaphor_map": {
            "raqs": "ecstatic devotional dance, not entertainment",
            "wajd": "spiritual trance state",
            "dam": "breath, divine life force, spiritual invocation",
            "qalandar": "mystic wanderer, free spirit in divine love",
        },
        "social_roles": {"speaker": "devotee, faqir, seeker", "addressee": "divine, murshid, saint"},
        "visual_settings": {
            "geography": "dargah courtyard, Sufi shrine, qawwali mehfil",
            "architecture": "shrine with marble, chadar-covered tomb, open courtyard, rose petals",
            "season_default": "night, lit by lanterns and candles",
            "colors": ["deep green", "white", "gold", "rose"],
        },
        "restrictions": ["maintain absolute spiritual respect", "no secular party aesthetics",
                         "no literal depiction of divine figures"],
        "common_misinterpretations": [
            "Raqs in qawwali is spiritual ecstasy, not dance performance",
            "Dam mast qalandar refers to invocating spiritual states",
        ],
    },
    "north_indian_folk_female": {
        "id": "north_indian_folk_female",
        "name": "North Indian Folk Female Voice",
        "description": "For women's folk songs — lori, banna, bidaai, sawan, teej",
        "trigger_words": ["lori", "banna", "bidaai", "sawan", "teej", "mehendi",
                          "doli", "sasural", "maike", "chunariya", "kajra",
                          "kangna", "payal", "sindoor", "bindi"],
        "metaphor_map": {
            "sawan": "monsoon of longing, reunion, fertility",
            "bidaai": "departure from parental home — profound grief and transition",
            "mehendi": "bridal anticipation, love written on hands",
            "payal": "feminine presence, arrival, beauty, movement",
            "doli": "bridal palanquin — transition, departure, new life",
        },
        "social_roles": {"speaker": "bride, mother, sister, young woman", "addressee": "beloved, family, self"},
        "visual_settings": {
            "geography": "north Indian village or small town",
            "architecture": "courtyard, decorated doorway, marigold-draped thresholds",
            "season_default": "monsoon or spring",
            "colors": ["bright red", "turmeric yellow", "green", "pink", "marigold orange"],
        },
        "restrictions": ["maintain feminine dignity", "avoid male gaze framing",
                         "celebratory pain — bidaai is grief but also love"],
        "common_misinterpretations": [
            "Bidaai songs are not just sad — they encode complex social transition",
            "Sawan references are about longing for reunion, not just rain",
        ],
    },
    "modern_urban_alienation": {
        "id": "modern_urban_alienation",
        "name": "Modern Urban Alienation",
        "description": "For contemporary songs about city life, loneliness, digital age disconnect",
        "trigger_words": ["sheher", "raat", "neend", "tanhaai", "phone", "screen",
                          "traffic", "mirror", "coffee", "cigarette", "rain", "window",
                          "3am", "rooftop", "metro", "signal"],
        "metaphor_map": {
            "sheher": "the impersonal city as emotional landscape",
            "raat": "insomnia as existential state",
            "mirror": "self-confrontation, fragmented identity",
            "rooftop": "isolation perch, perspective, escape",
        },
        "social_roles": {"speaker": "young urban individual", "addressee": "self, absent lover, city"},
        "visual_settings": {
            "geography": "Indian metro city — Delhi, Mumbai, Lahore",
            "architecture": "apartment, rooftop, metro station, empty road, chai stall",
            "season_default": "night, late hours, monsoon rain",
            "colors": ["neon blue", "warm yellow streetlight", "grey", "black", "rain-wet surfaces"],
        },
        "restrictions": [],
        "common_misinterpretations": [
            "Urban loneliness in South Asian context includes family pressure, not just romantic loss",
        ],
    },
    "generic_english": {
        "id": "generic_english",
        "name": "Generic English",
        "description": "Default pack for English-language content without strong cultural markers",
        "trigger_words": [],
        "metaphor_map": {},
        "social_roles": {"speaker": "unspecified", "addressee": "unspecified"},
        "visual_settings": {
            "geography": "unspecified",
            "architecture": "unspecified",
            "season_default": "unspecified",
            "colors": [],
        },
        "restrictions": [],
        "common_misinterpretations": [],
    },
}


def detect_culture_pack(text: str, language_hint: str = "auto") -> str:
    text_lower = text.lower()
    scores = {}
    for pack_id, pack in CULTURE_PACKS.items():
        score = sum(1 for tw in pack["trigger_words"] if tw in text_lower)
        scores[pack_id] = score
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    # Fallback by language
    lang_map = {
        "punjabi": "punjabi_rural_lament",
        "urdu": "urdu_philosophical_ghazal",
        "hindi": "north_indian_folk_female",
    }
    if language_hint in lang_map:
        return lang_map[language_hint]
    return "generic_english"


def get_culture_pack(pack_id: str) -> Optional[Dict[str, Any]]:
    return CULTURE_PACKS.get(pack_id)


def apply_culture_enrichment(context_data: Dict[str, Any], pack_id: str) -> Dict[str, Any]:
    pack = CULTURE_PACKS.get(pack_id)
    if not pack:
        return context_data

    # Enrich world assumptions from pack if not already set
    wa = context_data.get("world_assumptions", {})
    vs = pack["visual_settings"]
    if wa.get("geography", "unspecified") == "unspecified" and vs.get("geography"):
        wa["geography"] = vs["geography"]
    if wa.get("architecture_style", "unspecified") == "unspecified" and vs.get("architecture"):
        wa["architecture_style"] = vs["architecture"]
    if wa.get("season", "unspecified") == "unspecified" and vs.get("season_default"):
        wa["season"] = vs["season_default"]
    context_data["world_assumptions"] = wa

    # Enrich cultural setting
    cs = context_data.get("cultural_setting", {})
    cs["culture_pack"] = pack_id
    cs["pack_name"] = pack["name"]
    cs["visual_palette"] = vs.get("colors", [])
    cs["restrictions"] = pack.get("restrictions", [])
    cs["common_misinterpretations"] = pack.get("common_misinterpretations", [])
    context_data["cultural_setting"] = cs

    # Enrich speaker model from pack defaults
    sm = context_data.get("speaker_model", {})
    if sm.get("identity", "unspecified") == "unspecified" and pack["social_roles"].get("speaker"):
        sm["identity"] = pack["social_roles"]["speaker"]
    context_data["speaker_model"] = sm

    am = context_data.get("addressee_model", {})
    if am.get("identity", "unspecified") == "unspecified" and pack["social_roles"].get("addressee"):
        am["identity"] = pack["social_roles"]["addressee"]
    context_data["addressee_model"] = am

    return context_data


def get_metaphor_meanings(text: str, pack_id: str) -> List[Dict[str, str]]:
    pack = CULTURE_PACKS.get(pack_id)
    if not pack:
        return []
    text_lower = text.lower()
    found = []
    for trigger, meaning in pack.get("metaphor_map", {}).items():
        if trigger in text_lower:
            found.append({"trigger": trigger, "cultural_meaning": meaning})
    return found
