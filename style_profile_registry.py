"""
Qaivid Style Profile Registry

Registry of Production Styles and Cinematic Styles.

DESIGN PRINCIPLE:
- All style knowledge lives here as pure data dicts.
- No style logic lives in any engine.
- Adding a new style = adding one dict entry here. Zero engine changes needed.

Each PRODUCTION_STYLE entry:
  id, label, description, context_directive, storyboard_modifiers,
  compatible_cinematic_styles

Each CINEMATIC_STYLE entry:
  id, label, description, context_directive, storyboard_modifiers,
  image_generation_suffix, style_preset_mapping, compatible_production_styles
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# PRODUCTION STYLES
# =============================================================================
# Defines the structural/conceptual type of video (what kind of video it is).

PRODUCTION_STYLES: Dict[str, Dict[str, Any]] = {
    "narrative": {
        "id": "narrative",
        "label": "Narrative",
        "description": (
            "Story-driven video. Characters act out scenes. The song's emotional "
            "journey is depicted through story events, environments, and human interaction. "
            "Each shot advances a visual story."
        ),
        "context_directive": (
            "PRODUCTION STYLE — NARRATIVE: "
            "Visualize this song as a story being told. Assign visualization_mode 'direct' or "
            "'indirect' to most lines — the audience should see characters doing things, "
            "being in places, and experiencing events that echo the lyrics. "
            "Minimize 'symbolic' and 'absorbed' modes; keep the imagery grounded in human action."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "purposeful",
            "performance_hint": "Actor in scene, emotionally engaged with surroundings.",
        },
        "compatible_cinematic_styles": [
            "cinematic_natural", "noir_dramatic", "soft_poetic",
            "vintage_grain", "monochrome", "arthouse_minimalist",
        ],
    },

    "performance": {
        "id": "performance",
        "label": "Performance",
        "description": (
            "Artist-focused video. The singer or performer is the primary visual subject. "
            "Shots centre on the performer singing, moving, and emoting directly to camera "
            "or within a styled environment. Energy and presence drive the visual."
        ),
        "context_directive": (
            "PRODUCTION STYLE — PERFORMANCE: "
            "Visualize this song primarily as a performance. Assign visualization_mode "
            "'performance_only' to most lines. The artist is always present in frame. "
            "Environments serve as backdrops for the performer rather than story locations. "
            "Body language, micro-expression, and direct engagement with camera are key."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "intimate_push_ins",
            "performance_hint": "Artist performing: direct camera engagement, high presence.",
        },
        "compatible_cinematic_styles": [
            "cinematic_natural", "vibrant_bold", "noir_dramatic",
            "soft_poetic", "monochrome", "vintage_grain",
        ],
    },

    "split_narrative_performance": {
        "id": "split_narrative_performance",
        "label": "Narrative + Performance",
        "description": (
            "The most common music video format. Story scenes and artist performance "
            "shots are intercut throughout the song. Narrative scenes carry the emotional "
            "content; performance shots ground the viewer in the artist's presence. "
            "The ratio shifts — more narrative in verses, more performance in chorus."
        ),
        "context_directive": (
            "PRODUCTION STYLE — SPLIT NARRATIVE/PERFORMANCE: "
            "Visualize this song as a blend of narrative and performance. "
            "In verses: assign 'direct' or 'indirect' — show story scenes and character moments. "
            "In choruses: shift toward 'performance_only' — the artist performs with energy. "
            "Bridges/outros can go symbolic or absorbed. "
            "Mix feels natural: story and artist presence reinforce each other."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "varied",
            "performance_hint": "Mix: some shots are actor-in-scene, some are artist performing.",
        },
        "compatible_cinematic_styles": [
            "cinematic_natural", "vibrant_bold", "soft_poetic",
            "noir_dramatic", "vintage_grain", "monochrome", "arthouse_minimalist",
        ],
    },

    "conceptual_abstract": {
        "id": "conceptual_abstract",
        "label": "Conceptual / Abstract",
        "description": (
            "Non-literal, mood-driven video. The song's themes and emotions are expressed "
            "through symbolic imagery, visual metaphors, and abstract sequences. "
            "No conventional story or direct narrative. Pure feeling through image."
        ),
        "context_directive": (
            "PRODUCTION STYLE — CONCEPTUAL/ABSTRACT: "
            "Visualize this song through symbolic and abstract imagery. "
            "Assign visualization_mode 'symbolic' or 'absorbed' to most lines. "
            "Avoid literal depiction of events or characters performing. "
            "Instead let objects, environments, light, texture, and motion carry the emotional meaning. "
            "The visual language is poetic and metaphorical throughout."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "slow_contemplative",
            "performance_hint": "No conventional character or artist — symbolic environment and objects.",
        },
        "compatible_cinematic_styles": [
            "surrealist_dream", "arthouse_minimalist", "soft_poetic",
            "monochrome", "vintage_grain", "noir_dramatic",
        ],
    },

    "documentary_candid": {
        "id": "documentary_candid",
        "label": "Documentary / Candid",
        "description": (
            "Real-feeling, observational video. Handheld-style camera, candid moments, "
            "authentic settings. Feels like the camera caught something real rather than staged. "
            "Intimate, honest, unguarded."
        ),
        "context_directive": (
            "PRODUCTION STYLE — DOCUMENTARY/CANDID: "
            "Visualize this song as if a documentary camera is observing real moments. "
            "Assign 'direct' or 'indirect' visualization_mode. Characters and settings feel "
            "unposed, authentic, and observed rather than directed. "
            "Candid body language, available light, real textures. "
            "Avoid anything that feels staged or composed for camera."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "handheld_observational",
            "performance_hint": "Subjects unaware of camera or natural in its presence.",
        },
        "compatible_cinematic_styles": [
            "cinematic_natural", "vintage_grain", "monochrome",
            "arthouse_minimalist", "soft_poetic",
        ],
    },

    "single_location": {
        "id": "single_location",
        "label": "Single Location",
        "description": (
            "Minimalist, contained video. One primary space holds the entire song. "
            "The power comes from depth within that space — detail, light changes, micro-moments. "
            "Claustrophobic or intimate depending on treatment."
        ),
        "context_directive": (
            "PRODUCTION STYLE — SINGLE LOCATION: "
            "Visualize this song as taking place in one primary space throughout. "
            "Assign location_dna as a single, specific, deeply-rendered environment. "
            "visual_constraints should emphasise exploring this one space through changing light, "
            "detail, and emotional register rather than moving between locations. "
            "Depth is achieved through intimacy and attention, not variety of place."
        ),
        "storyboard_modifiers": {
            "camera_movement_bias": "intimate_contained",
            "performance_hint": "Character anchored in the single space throughout.",
        },
        "compatible_cinematic_styles": [
            "arthouse_minimalist", "cinematic_natural", "soft_poetic",
            "monochrome", "noir_dramatic", "vintage_grain",
        ],
    },
}


# =============================================================================
# CINEMATIC STYLES
# =============================================================================
# Defines the visual/aesthetic language of the video (how it looks and feels).

CINEMATIC_STYLES: Dict[str, Dict[str, Any]] = {
    "cinematic_natural": {
        "id": "cinematic_natural",
        "label": "Cinematic Natural",
        "description": (
            "Grounded realism with cinematic quality. Natural light, authentic textures, "
            "warm-to-neutral palette. Feels honest and emotionally present. "
            "The world looks like the world, just more beautifully seen."
        ),
        "context_directive": (
            "CINEMATIC STYLE — CINEMATIC NATURAL: "
            "visual_constraints should emphasise naturalistic, available light. "
            "Palette is warm earth tones, golden hour when emotional peaks arrive. "
            "Avoid artificial or heightened colour treatment. "
            "world_assumptions should reflect grounded, real, lived-in environments."
        ),
        "storyboard_modifiers": {
            "camera_style": "cinematic_naturalism",
            "lighting": "warm naturalistic, golden hour available light",
            "movement": "restrained handheld intimacy",
            "atmosphere_note": "grounded, present, emotionally honest",
        },
        "image_generation_suffix": "cinematic natural light, warm earthy tones, photorealistic",
        "style_preset_mapping": "cinematic_natural",
        "compatible_production_styles": [
            "narrative", "split_narrative_performance", "documentary_candid", "single_location",
        ],
    },

    "noir_dramatic": {
        "id": "noir_dramatic",
        "label": "Noir / Dramatic",
        "description": (
            "High-contrast shadow play, dramatic single-source lighting, deep blacks. "
            "Rain, smoke, hard angles. Morally weighted atmosphere. Visually powerful and tense."
        ),
        "context_directive": (
            "CINEMATIC STYLE — NOIR/DRAMATIC: "
            "visual_constraints must include: high-contrast shadow/light opposition, "
            "single-source dramatic lighting, rain or moisture as atmospheric element, "
            "hard architectural lines and deep shadow pools. "
            "world_assumptions should lean toward dusk, night, or overcast. "
            "Palette is cool-to-silver with deep shadow blacks and limited warm highlights."
        ),
        "storyboard_modifiers": {
            "camera_style": "noir_dramatic",
            "lighting": "single-source dramatic, deep shadow, high contrast",
            "movement": "slow deliberate",
            "atmosphere_note": "morally weighted, tense, shadow-as-emotion",
        },
        "image_generation_suffix": "film noir, dramatic chiaroscuro lighting, deep shadows, moody",
        "style_preset_mapping": "noir",
        "compatible_production_styles": [
            "narrative", "performance", "split_narrative_performance", "single_location",
        ],
    },

    "soft_poetic": {
        "id": "soft_poetic",
        "label": "Soft Poetic",
        "description": (
            "Diffused, gauzy, tender visual language. Shallow depth of field, soft backlight, "
            "muted warmth. Feels like a half-remembered feeling. Gentle and emotionally open."
        ),
        "context_directive": (
            "CINEMATIC STYLE — SOFT POETIC: "
            "visual_constraints should emphasise diffused, soft, backlit or side-lit frames. "
            "Palette is warm pastel, muted golds and dusty pinks, gentle desaturation. "
            "Avoid hard lines, high contrast, or saturated bold colour. "
            "world_assumptions should reflect morning light, late afternoon haze, or soft interiors."
        ),
        "storyboard_modifiers": {
            "camera_style": "soft_poetic",
            "lighting": "diffused backlight, soft warm side-light, atmospheric haze",
            "movement": "slow and breath-paced",
            "atmosphere_note": "tender, gauzy, emotionally open",
        },
        "image_generation_suffix": "soft diffused light, gentle warm tones, ethereal, bokeh",
        "style_preset_mapping": "soft_poetic",
        "compatible_production_styles": [
            "narrative", "performance", "split_narrative_performance",
            "conceptual_abstract", "single_location",
        ],
    },

    "arthouse_minimalist": {
        "id": "arthouse_minimalist",
        "label": "Arthouse / Minimalist",
        "description": (
            "Sparse, contemplative, restrained. Long takes, still frames, deliberate silence "
            "in the composition. Visual grammar borrows from art cinema — Kiarostami, Haneke, "
            "early Nuri Bilge Ceylan. Meaning in patience."
        ),
        "context_directive": (
            "CINEMATIC STYLE — ARTHOUSE/MINIMALIST: "
            "visual_constraints should emphasise sparse composition, long-held frames, "
            "unconventional or off-centre subject placement. "
            "Avoid busy or visually saturated frames. Palette is neutral, desaturated, controlled. "
            "world_assumptions should reflect quiet, uncrowded, contemplative environments. "
            "Empty space in frame is intentional and meaningful."
        ),
        "storyboard_modifiers": {
            "camera_style": "arthouse_minimalist",
            "lighting": "flat natural or overcast, controlled, no dramatic shadow play",
            "movement": "static or imperceptibly slow",
            "atmosphere_note": "contemplative, patient, meaning through restraint",
        },
        "image_generation_suffix": "minimalist composition, sparse, muted tones, arthouse cinema",
        "style_preset_mapping": "monochrome",
        "compatible_production_styles": [
            "conceptual_abstract", "documentary_candid", "single_location", "narrative",
        ],
    },

    "vibrant_bold": {
        "id": "vibrant_bold",
        "label": "Vibrant / Bold",
        "description": (
            "Rich, saturated colour. Dramatic staging, strong costume and set design. "
            "Feels celebratory, grand, or emotionally extravagant. "
            "South Asian cinema register: Bollywood dramatic, Pakistani drama aesthetic."
        ),
        "context_directive": (
            "CINEMATIC STYLE — VIBRANT/BOLD: "
            "visual_constraints should emphasise richly saturated colour, strong costume colour, "
            "dramatic staging with deliberate visual weight. "
            "Palette is jewel tones, deep reds, gold, emerald, and rich warm skin tones. "
            "world_assumptions should reflect ornate or beautifully designed environments. "
            "Lighting is motivated but dramatic — no flat or diffused naturalism."
        ),
        "storyboard_modifiers": {
            "camera_style": "vibrant_dramatic",
            "lighting": "motivated dramatic, rich fill, jewel-tone palette",
            "movement": "purposeful and expressive",
            "atmosphere_note": "celebratory or emotionally extravagant, visually rich",
        },
        "image_generation_suffix": "vibrant saturated colours, rich jewel tones, dramatic lighting",
        "style_preset_mapping": "cinematic_natural",
        "compatible_production_styles": [
            "narrative", "performance", "split_narrative_performance",
        ],
    },

    "vintage_grain": {
        "id": "vintage_grain",
        "label": "Vintage / Film Grain",
        "description": (
            "8mm or 16mm film aesthetic. Warm desaturation, visible grain, light leaks. "
            "Nostalgia as a visual language. Feels like a memory retrieved from film stock. "
            "Intimate, time-worn, irreplaceable."
        ),
        "context_directive": (
            "CINEMATIC STYLE — VINTAGE/GRAIN: "
            "visual_constraints should emphasise visible film grain, warm desaturation, "
            "soft vignette, light leak artifacts, and analogue colour bleed. "
            "Palette is warm amber, faded mustard, dusty rose — like aged colour film. "
            "world_assumptions may reflect a past era or timeless quality. "
            "Avoid clinical sharpness or digital perfection."
        ),
        "storyboard_modifiers": {
            "camera_style": "vintage_analogue",
            "lighting": "warm analogue, soft vignette, imperfect natural",
            "movement": "slightly unsteady, handheld memory quality",
            "atmosphere_note": "nostalgic, time-worn, memory retrieved",
        },
        "image_generation_suffix": "film grain, vintage 16mm, warm analogue tones, nostalgic",
        "style_preset_mapping": "soft_poetic",
        "compatible_production_styles": [
            "narrative", "documentary_candid", "split_narrative_performance",
            "single_location", "performance",
        ],
    },

    "monochrome": {
        "id": "monochrome",
        "label": "Monochrome",
        "description": (
            "Full black and white. The entire tonal spectrum freed from colour. "
            "Emotion is carried by light, shadow, texture, and form alone. "
            "Timeless, forceful, and emotionally direct."
        ),
        "context_directive": (
            "CINEMATIC STYLE — MONOCHROME: "
            "visual_constraints should specify that all shots are black and white. "
            "Emphasise tonal contrast, texture, and form as the primary visual language. "
            "Lighting should exploit the full grey scale — pure blacks, rich mid-tones, bright highlights. "
            "world_assumptions can draw on any era since monochrome is timeless."
        ),
        "storyboard_modifiers": {
            "camera_style": "monochrome_classic",
            "lighting": "full tonal range, black-and-white contrast optimised",
            "movement": "restrained and deliberate",
            "atmosphere_note": "timeless, forceful, tonal purity",
        },
        "image_generation_suffix": "black and white, monochrome, high tonal contrast",
        "style_preset_mapping": "monochrome",
        "compatible_production_styles": [
            "narrative", "performance", "documentary_candid",
            "conceptual_abstract", "single_location", "split_narrative_performance",
        ],
    },

    "surrealist_dream": {
        "id": "surrealist_dream",
        "label": "Surrealist / Dream",
        "description": (
            "Dream logic, impossible imagery, fluid visual sequences. "
            "Reality bends to emotion. Objects behave in unexpected ways. "
            "The song's inner world is made literally visible."
        ),
        "context_directive": (
            "CINEMATIC STYLE — SURREALIST/DREAM: "
            "visual_constraints should embrace impossible or dream-logic imagery. "
            "Objects, people, and environments may defy physical reality when that serves the emotion. "
            "Assign visualization_mode 'symbolic' or 'absorbed' to most lines. "
            "Palette is dreamlike — either hyper-saturated or bleached depending on emotional register. "
            "world_assumptions can be deliberately paradoxical or unresolved."
        ),
        "storyboard_modifiers": {
            "camera_style": "surrealist_dreamlike",
            "lighting": "fluid and emotionally shifting — warm to cold within same scene",
            "movement": "slow drifting, weightless",
            "atmosphere_note": "inner world made visible, dream logic, emotion as reality",
        },
        "image_generation_suffix": "surrealist, dreamlike, impossible imagery, ethereal atmosphere",
        "style_preset_mapping": "dreamy",
        "compatible_production_styles": [
            "conceptual_abstract", "performance", "split_narrative_performance",
        ],
    },
}


# =============================================================================
# REGISTRY CLASS — lookup helpers
# =============================================================================

class StyleProfileRegistry:
    """
    Lookup interface for the style registries.

    All methods are pure lookups — no logic, no side effects.
    Adding a new style = add an entry to PRODUCTION_STYLES or CINEMATIC_STYLES above.
    """

    @staticmethod
    def get_production_style(style_id: str) -> Optional[Dict[str, Any]]:
        return PRODUCTION_STYLES.get(style_id)

    @staticmethod
    def get_cinematic_style(style_id: str) -> Optional[Dict[str, Any]]:
        return CINEMATIC_STYLES.get(style_id)

    @staticmethod
    def all_production_styles() -> List[Dict[str, Any]]:
        return list(PRODUCTION_STYLES.values())

    @staticmethod
    def all_cinematic_styles() -> List[Dict[str, Any]]:
        return list(CINEMATIC_STYLES.values())

    @staticmethod
    def get_style_preset_mapping(cinematic_style_id: str) -> str:
        """Return the StyleGradingEngine preset string for backward compatibility."""
        entry = CINEMATIC_STYLES.get(cinematic_style_id, {})
        return entry.get("style_preset_mapping", "cinematic_natural")

    @staticmethod
    def build_style_profile(production_style_id: str, cinematic_style_id: str) -> Dict[str, Any]:
        """
        Build a complete style_profile dict from two registry IDs.
        This is what gets stored on the project and passed to downstream engines.
        """
        prod = PRODUCTION_STYLES.get(production_style_id) or PRODUCTION_STYLES["split_narrative_performance"]
        cin = CINEMATIC_STYLES.get(cinematic_style_id) or CINEMATIC_STYLES["cinematic_natural"]
        return {
            "production": prod,
            "cinematic": cin,
            "preset": cin.get("style_preset_mapping", "cinematic_natural"),
        }

    @staticmethod
    def default_style_profile() -> Dict[str, Any]:
        """Fallback for projects that pre-date the Style Profile Engine."""
        return StyleProfileRegistry.build_style_profile(
            "narrative", "cinematic_natural"
        )

    @staticmethod
    def suggest_compatible_pairs(
        song_analysis: Optional[Dict[str, Any]] = None,
        max_pairs: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Return up to max_pairs compatible (production_id, cinematic_id) dicts
        based on lightweight heuristics from song analysis data.

        Each returned dict has keys: production_id, cinematic_id.
        This is a non-LLM fallback for when the StyleProfileEngine is unavailable
        and for compatibility-filtering in the UI.

        Heuristic logic:
          - High BPM / high energy → performance or abstract + vibrant/noir
          - Low BPM / emotional → narrative or split + soft_poetic/cinematic_natural
          - Default → split_narrative_performance + cinematic_natural
        """
        sa = song_analysis or {}
        bpm = float(sa.get("bpm") or 0)
        energy_profile = str(sa.get("energy_profile") or "").lower()
        brightness = str(sa.get("brightness_profile") or "").lower()

        high_energy = bpm >= 120 or "high" in energy_profile
        bright = "bright" in brightness or "warm" in brightness

        candidates: List[Dict[str, str]] = []

        if high_energy and bright:
            candidates = [
                {"production_id": "performance", "cinematic_id": "vibrant_bold"},
                {"production_id": "conceptual_abstract", "cinematic_id": "noir_dramatic"},
                {"production_id": "split_narrative_performance", "cinematic_id": "cinematic_natural"},
            ]
        elif high_energy:
            candidates = [
                {"production_id": "performance", "cinematic_id": "noir_dramatic"},
                {"production_id": "split_narrative_performance", "cinematic_id": "vibrant_bold"},
                {"production_id": "conceptual_abstract", "cinematic_id": "arthouse_minimalist"},
            ]
        elif bpm > 0 and bpm < 80:
            candidates = [
                {"production_id": "narrative", "cinematic_id": "soft_poetic"},
                {"production_id": "single_location", "cinematic_id": "cinematic_natural"},
                {"production_id": "split_narrative_performance", "cinematic_id": "vintage_grain"},
            ]
        else:
            candidates = [
                {"production_id": "split_narrative_performance", "cinematic_id": "cinematic_natural"},
                {"production_id": "narrative", "cinematic_id": "soft_poetic"},
                {"production_id": "performance", "cinematic_id": "noir_dramatic"},
            ]

        # Filter out any ID not in the registry (guard for future registry changes)
        valid = [
            c for c in candidates
            if c["production_id"] in PRODUCTION_STYLES and c["cinematic_id"] in CINEMATIC_STYLES
        ]
        return valid[:max_pairs] or [
            {"production_id": "split_narrative_performance", "cinematic_id": "cinematic_natural"}
        ]

    @staticmethod
    def registry_summary_for_llm() -> str:
        """
        Compact text summary of all styles for LLM suggestion prompts.
        """
        lines = ["PRODUCTION STYLES (choose one):"]
        for s in PRODUCTION_STYLES.values():
            lines.append(f'  - id="{s["id"]}" | {s["label"]}: {s["description"][:120]}')

        lines.append("\nCINEMATIC STYLES (choose one):")
        for s in CINEMATIC_STYLES.values():
            compatible = ", ".join(s.get("compatible_production_styles", []))
            lines.append(
                f'  - id="{s["id"]}" | {s["label"]}: {s["description"][:120]}'
                + (f' [works well with: {compatible}]' if compatible else '')
            )

        return "\n".join(lines)
