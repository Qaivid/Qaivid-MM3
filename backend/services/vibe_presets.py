"""
Vibe Presets — Ported and adapted from Qaivid 1.0 still-engine.ts
These are comprehensive visual style definitions that control every aspect
of the production pipeline: brief, storyboard, references, shot images.

In 2.0, vibe presets work WITH the ContextPacket, not against it.
The context engine provides cultural/emotional intelligence,
the vibe preset provides visual style direction.

Each preset declares production_style_id + cinematic_style_id so Stage 4
can resolve them to the same style_packet path as the AI suggestions and
manual picker, with no separate backend route needed.
"""
from typing import Dict, Any, Optional, List

VIBE_PRESETS: Dict[str, Dict[str, Any]] = {
    "punjab-pulse-cinematic": {
        "id": "punjab-pulse-cinematic",
        "label": "Punjab Pulse Cinematic",
        "tagline": "Premium Punjabi filmmaking — golden hour, mustard fields, haveli courtyards.",
        "production_style_id": "split_narrative_performance",
        "cinematic_style_id": "cinematic_realism",
        "brief_direction": "Premium Punjabi commercial filmmaking. Think Amrinder Gill, Satinder Sartaaj, Qismat. Characters well-dressed and groomed. Male: clean kurta, waistcoat, smart casual. Female: salwar kameez, dupatta, graceful jewellery. Locations: mustard fields, haveli courtyards, diaspora streets — all art-directed. Golden hour dominant. Never poverty aesthetic.",
        "storyboard_direction": "Performance-led narrative hybrid. 60% performance/emotional portraiture, 40% narrative inserts. Face-first framing. Slow push-in, gentle tracking, stabilised follow. Close-ups dominate.",
        "reference_direction": "Professional casting photo quality. Studio or clean outdoor portrait. Shallow depth of field. Characters styled for premium Punjabi production.",
        "shot_direction": "Naturalistic beauty lighting. Warm golden/earth tones. Clean commercial framing. Emotionally readable expressions. Warm earth palette: mustard yellow, deep green, cream, terracotta, maroon.",
        "avoid": ["poverty aesthetic", "tattered clothing", "flooded streets", "disaster scenes", "derelict locations", "harsh neon", "sci-fi"],
    },
    "desi-blockbuster": {
        "id": "desi-blockbuster",
        "label": "Desi Blockbuster",
        "tagline": "High-budget Bollywood/Pollywood glamour — spectacle, jewel tones, star entrances.",
        "production_style_id": "split_narrative_performance",
        "cinematic_style_id": "vibrant_bold",
        "brief_direction": "High-budget glamorous Bollywood/Pollywood. Star-quality presence. Designer outfits, luxury settings. Dramatic poses, hero/heroine entrances. Grand staircases, rooftop parties, expensive cars. Spectacle-driven.",
        "storyboard_direction": "Dramatic hero reveals, slow motion entrances, low-angle power shots. Fashion walk sequences. Dance performance beats. Quick cuts for energy, smooth for romance.",
        "reference_direction": "High-fashion glamour photography. Dramatic studio lighting with rim lights. Bold colours, sharp contrast, aspirational framing.",
        "shot_direction": "Glamorous cinematic lighting. Bold saturated colours. Anamorphic lens feel. Shallow depth of field, lens flares. Larger-than-life spectacle.",
        "avoid": ["documentary realism", "gritty textures", "desaturated palettes", "minimalist staging", "muted colours"],
    },
    "qawwali-sufi": {
        "id": "qawwali-sufi",
        "label": "Qawwali & Sufi",
        "tagline": "Devotional performance — mehfil, dargah, candlelight, spiritual intensity.",
        "production_style_id": "performance",
        "cinematic_style_id": "cinematic_natural",
        "brief_direction": "Devotional music filmmaking. Think Nusrat Fateh Ali Khan mehfils, Abida Parveen, Coke Studio Sufi sessions. Lead qawwal in white kurta with waistcoat, topi, chadar. Mehfil setting with harmonium, tabla. Dargah courtyards, Mughal arches, oil lamps, rose petals.",
        "storyboard_direction": "Devotional performance-led. 65% performance, 35% spiritual imagery. Singer close-ups showing spiritual intensity. Hand details on harmonium/tabla. Slow push-ins for building intensity.",
        "reference_direction": "Dignified portrait of qawwali musician. Traditional attire. Warm amber/candlelit lighting. Spiritual intensity in expression.",
        "shot_direction": "Warm amber, candlelit, oil lamp, golden hour. Incandescent warmth with dramatic shadows. Intimate close-ups. Low-angle looking up at performers. Deep amber, warm gold, cream, rich green, rose pink.",
        "avoid": ["nightclub aesthetics", "cold modern lighting", "Western interiors", "Bollywood dance choreography", "poverty settings", "harsh neon"],
    },
    "golden-hour-romance": {
        "id": "golden-hour-romance",
        "label": "Golden Hour Romance",
        "tagline": "Intimate 35mm romance — soft focus, warm amber, hands touching, eyes meeting.",
        "production_style_id": "narrative",
        "cinematic_style_id": "soft_poetic",
        "brief_direction": "Intimate romantic cinematography. 35mm film feel. Golden light pouring through everything. Characters warm, soft, emotionally open. Linen, cotton, flowing fabrics. Meadows, coastal cliffs, vineyard rows, cafe patios.",
        "storyboard_direction": "65% intimate close-ups, 35% atmospheric wides. Slow gentle pacing. Close-ups of hands touching, eyes meeting. Lingering on tenderness.",
        "reference_direction": "Warm golden hour lighting, soft romantic photography, 35mm film texture, intimate framing, warm skin tones.",
        "shot_direction": "Warm golden hour. Soft focus. 35mm film grain texture. High key lighting. Warm amber tones. Romantic atmosphere.",
        "avoid": ["cold harsh lighting", "dark moody shadows", "aggressive camera moves", "stark contrast"],
    },
    "shadow-smoke": {
        "id": "shadow-smoke",
        "label": "Shadow & Smoke",
        "tagline": "Film noir — chiaroscuro, rain-soaked streets, characters emerging from shadow.",
        "production_style_id": "narrative",
        "cinematic_style_id": "noir_dramatic",
        "brief_direction": "Film noir. Think Blade Runner, Se7en. Enigmatic characters in long dark overcoats. Smoke-filled jazz bars, rain-soaked streets. Single hard source creating dramatic shadows. Venetian blind patterns.",
        "storyboard_direction": "60% tight dramatic close-ups (half-lit faces), 40% atmospheric wides. Slow burn with sharp reveals. Characters emerge from shadow.",
        "reference_direction": "High contrast noir photography. Dramatic chiaroscuro. Low saturation. Deep shadows. Film noir aesthetic.",
        "shot_direction": "High contrast chiaroscuro. Low saturation. Deep blacks, selective highlights. Mystery tension. Practical light sources.",
        "avoid": ["bright cheerful colours", "high-key lighting", "warm romantic tones", "pastoral settings"],
    },
    "neon-dreams": {
        "id": "neon-dreams",
        "label": "Neon Dreams",
        "tagline": "High-energy nightlife — magenta, cyan, neon-slicked streets, electric performance.",
        "production_style_id": "performance",
        "cinematic_style_id": "vibrant_bold",
        "brief_direction": "High-energy nightlife. Neon-drenched club scenes, late-night city energy. Statement streetwear, metallic/holographic fabrics. Neon-lit nightclubs, rain-slicked streets reflecting neon. Magenta, cyan, electric blue.",
        "storyboard_direction": "50% character performance/dance, 30% neon environment, 20% cutaways. Fast cuts on beats, slow-motion on drops.",
        "reference_direction": "Neon-lit portrait photography. Vibrant saturated colours. Bold neon accents. Night photography aesthetic.",
        "shot_direction": "Neon-lit cinematic. Vibrant saturated colours. Bold neon lighting. Night photography. Dynamic energy.",
        "avoid": ["natural daylight", "pastoral landscapes", "muted earth tones", "slow contemplative pacing"],
    },
    "midnight-neon-poetry": {
        "id": "midnight-neon-poetry",
        "label": "Midnight Neon Poetry",
        "tagline": "Melancholic urban night — Wong Kar-wai, rain, neon reflections, solitary figures.",
        "production_style_id": "narrative",
        "cinematic_style_id": "arthouse_minimalist",
        "brief_direction": "Melancholic urban night. Think Wong Kar-wai, Drive, Lost in Translation. Solitary figure on rain-soaked streets. Worn leather jacket, vintage denim. Laundromats, 24-hour diners, empty subway cars. Rain is constant.",
        "storyboard_direction": "50% intimate character study, 50% atmospheric urban landscape. Contemplative, unhurried. Hold shots longer than comfortable.",
        "reference_direction": "Urban night street photography. Film grain. Neon reflections. Rain-soaked melancholic atmosphere.",
        "shot_direction": "Film grain texture. Neon reflections on wet surfaces. Rain-soaked melancholic atmosphere. Motion blur. Lonely night wandering.",
        "avoid": ["bright daylight", "cheerful atmosphere", "luxury glamour", "pastoral countryside"],
    },
    "cold-precision": {
        "id": "cold-precision",
        "label": "Cold Precision Cinema",
        "tagline": "Clinical thriller — Fincher/Villeneuve, desaturated, meticulous framing, controlled tension.",
        "production_style_id": "narrative",
        "cinematic_style_id": "noir_dramatic",
        "brief_direction": "Clinical thriller. Think David Fincher, Denis Villeneuve. Perfectly tailored dark suits. Sterile offices, hospital corridors, server rooms. Cold blue-white fluorescent. Everything controlled.",
        "storyboard_direction": "55% tight controlled close-ups, 45% wide geometric compositions. Metronomic pacing. Cuts land on precise beats.",
        "reference_direction": "Precise clinical photography. Dark tones. Desaturated palette. Sharp detail. Meticulous framing.",
        "shot_direction": "Cold precision. Dark tones. Desaturated. Meticulous framing. Low key lighting. Sharp detail. Controlled tension.",
        "avoid": ["warm romantic tones", "loose handheld camera", "bright colours", "pastoral softness"],
    },
    "autumn-academia": {
        "id": "autumn-academia",
        "label": "Autumn Academia",
        "tagline": "Nostalgic literary warmth — film grain, autumn light, cobblestones, vintage textures.",
        "production_style_id": "narrative",
        "cinematic_style_id": "vintage_grain",
        "brief_direction": "Nostalgic literary cinematography. Think Dead Poets Society, Call Me By Your Name. Tweed blazers, cable-knit sweaters. Old university libraries, autumn landscapes, cobblestone paths. Warm muted overcast golden light.",
        "storyboard_direction": "55% intimate character moments, 45% atmospheric environment. Unhurried, literary pacing. Scenes unfold like turning pages.",
        "reference_direction": "Documentary style portrait. Film grain. Warm earth tones. Natural lighting. Vintage texture. Nostalgic atmosphere.",
        "shot_direction": "Film grain texture. Warm earth tones. Natural lighting. Nostalgic vintage aesthetic. Poetic realism.",
        "avoid": ["synthetic neon", "hyper-modern aesthetics", "bold saturated colours", "clinical framing"],
    },
    "epic-grandeur": {
        "id": "epic-grandeur",
        "label": "Epic Grandeur",
        "tagline": "IMAX spectacle — sweeping landscapes, god-rays, ancient fortresses, hero moments.",
        "production_style_id": "conceptual_abstract",
        "cinematic_style_id": "cinematic_natural",
        "brief_direction": "IMAX spectacle. Sweeping landscapes, powerful hero moments. Warriors, leaders, travellers in flowing coats. Mountain ranges, volcanic terrain, ancient fortresses. Dramatic sky lighting, god-rays.",
        "storyboard_direction": "40% character close-ups for grounding, 60% grand-scale establishing wides. Journey-driven narrative. Summit reveals, cliff edges.",
        "reference_direction": "Realistic IMAX photography. Grand scale. Dramatic landscape lighting. Deep focus. Widescreen composition.",
        "shot_direction": "Widescreen IMAX feel. Grand scale. Dramatic landscapes. Deep focus. Epic composition. Dramatic lighting.",
        "avoid": ["small-scale intimate framing", "soft focus", "pastel colours", "flat lighting"],
    },
    "singer-performance": {
        "id": "singer-performance",
        "label": "Singer Performance",
        "tagline": "Artist as centrepiece — extreme facial detail, lit for camera, lip-sync ready.",
        "production_style_id": "performance",
        "cinematic_style_id": "cinematic_realism",
        "brief_direction": "Performance-driven music video. Singer is visual centrepiece. Extreme facial detail for lip-sync. Camera-ready, styled. At least one dedicated performance space. 35-45% performance shots.",
        "storyboard_direction": "35-45% singer performing (lip-sync close-ups). Face-first framing. Slow push-in on emotional peaks. Avatar shots for lip-sync, normal shots for B-roll.",
        "reference_direction": "Professional portrait with EXTREME facial detail. Lip shape, jaw structure, eye shape, brow arch. Vocally engaged expression.",
        "shot_direction": "Performance-first composition. Singer centred. Face clearly visible. Dramatic but flattering key light, rim light. Shallow depth of field.",
        "avoid": ["performer hidden", "flat lighting", "cluttered backgrounds", "generic stock aesthetic"],
    },
}


def get_vibe_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    return VIBE_PRESETS.get(preset_id)


def list_vibe_presets() -> List[Dict[str, str]]:
    return [{"id": v["id"], "label": v["label"]} for v in VIBE_PRESETS.values()]


def list_vibe_presets_for_ui() -> List[Dict[str, Any]]:
    """Return the fields needed by the Stage 4 style picker UI."""
    return [
        {
            "id":                  v["id"],
            "label":               v["label"],
            "tagline":             v.get("tagline", ""),
            "production_style_id": v["production_style_id"],
            "cinematic_style_id":  v["cinematic_style_id"],
        }
        for v in VIBE_PRESETS.values()
    ]


# ── Valid style IDs (must match style_profile_registry.py) ────────────────────
_VALID_PRODUCTION_IDS = {
    "narrative", "performance", "split_narrative_performance",
    "conceptual_abstract", "single_location", "documentary_candid",
}
_VALID_CINEMATIC_IDS = {
    "cinematic_natural", "noir_dramatic", "vibrant_bold", "soft_poetic",
    "vintage_grain", "monochrome", "arthouse_minimalist",
    "surrealist_dream", "cinematic_realism",
}
_REQUIRED_TEXT_FIELDS = (
    "label", "tagline", "brief_direction",
    "storyboard_direction", "reference_direction", "shot_direction",
)

_CUSTOM_VIBE_SYSTEM_PROMPT = """\
You are a cinematography director and music video vibe designer.
The user will give you:
  1. A plain-language description of the visual vibe they want.
  2. The song's cultural context (geography, genre, mood, era).

Return ONLY a valid JSON object (no markdown, no commentary) with these fields:
{
  "label":                 "Short 3-5 word vibe name (title case)",
  "tagline":               "One evocative sentence capturing the look and feel",
  "production_style_id":   "<one of: narrative | performance | split_narrative_performance | conceptual_abstract | single_location | documentary_candid>",
  "cinematic_style_id":    "<one of: cinematic_natural | noir_dramatic | vibrant_bold | soft_poetic | vintage_grain | monochrome | arthouse_minimalist | surrealist_dream | cinematic_realism>",
  "brief_direction":       "2-4 sentence creative brief direction. Be specific about wardrobe, settings, era, atmosphere.",
  "storyboard_direction":  "2-3 sentences on shot composition ratio, camera movement, pacing.",
  "reference_direction":   "1-2 sentences on lighting, depth, colour, reference plate style.",
  "shot_direction":        "1-2 sentences on colour grade, texture, lighting mood.",
  "avoid":                 ["array", "of", "short", "avoid phrases", "2-6 items"]
}

Enrich from the song's cultural context so the vibe is culturally grounded, not generic.
"""


def build_custom_vibe(description: str, llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise an LLM-generated vibe definition.

    Raises ``ValueError`` with a human-readable message if the LLM response is
    missing required fields or contains unknown style IDs.  Valid responses are
    normalised into a dict shaped like a standard preset (with an added
    ``description`` key holding the original user text).
    """
    # Coerce avoid to list
    avoid = llm_response.get("avoid", [])
    if isinstance(avoid, str):
        avoid = [a.strip() for a in avoid.split(",") if a.strip()]
    if not isinstance(avoid, list):
        avoid = []

    # Validate text fields
    missing = [f for f in _REQUIRED_TEXT_FIELDS if not str(llm_response.get(f) or "").strip()]
    if missing:
        raise ValueError(f"LLM response missing required fields: {', '.join(missing)}")

    prod_id = str(llm_response.get("production_style_id") or "").strip()
    cin_id  = str(llm_response.get("cinematic_style_id") or "").strip()

    if prod_id not in _VALID_PRODUCTION_IDS:
        prod_id = "split_narrative_performance"
    if cin_id not in _VALID_CINEMATIC_IDS:
        cin_id = "cinematic_natural"

    return {
        "id":                    "custom",
        "label":                 str(llm_response["label"]).strip(),
        "tagline":               str(llm_response["tagline"]).strip(),
        "production_style_id":   prod_id,
        "cinematic_style_id":    cin_id,
        "brief_direction":       str(llm_response["brief_direction"]).strip(),
        "storyboard_direction":  str(llm_response["storyboard_direction"]).strip(),
        "reference_direction":   str(llm_response["reference_direction"]).strip(),
        "shot_direction":        str(llm_response["shot_direction"]).strip(),
        "avoid":                 [str(a).strip() for a in avoid if str(a).strip()],
        "description":           description.strip(),
    }
