import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ── Lighting variants per expression mode ────────────────────────────────────
# Cycled by shot_index so a long run of same-mode shots doesn't all collapse
# onto the same lighting recipe (which is what produces the "every face shot
# has identical soft top light" monoculture). Each variant is a complete
# lighting setup that gives the diffusion model a meaningfully different
# direction. Values are deliberately specific (key direction, fill ratio,
# practicals, mood) rather than generic.
LIGHTING_VARIANTS: Dict[str, List[str]] = {
    "face": [
        "soft window key from camera-left, gentle natural fill, eye highlights catching ambient light",
        "low-key chiaroscuro, single warm source from camera-right, deep shadow on opposite cheek, dramatic mood",
        "warm golden-hour rim light around hair and shoulders, soft frontal fill on face",
        "overcast naturalistic top light, even ambient, no harsh shadows, documentary realism",
        "split lighting half-face in shadow, hard key from side, theatrical contrast",
        "warm practical lamp key mixed with cool ambient fill, mixed-temperature mood",
        "backlit silhouette with soft rim glow, face partially in shadow, melancholic atmosphere",
        "high-key open natural light, minimal shadow, breathable and gentle",
    ],
    "body": [
        "wide ambient natural lighting wrapping around figure, subtle motivated key from above-left",
        "warm low sun raking across body, long soft shadows on the ground",
        "cool ambient daylight with neutral fill, naturalistic and observational",
        "directional key from a single window, body half in light half in shadow",
        "twilight blue ambient with warm practical accents in the distance",
        "overcast soft wrap-around, no harsh shadows, calm and grounded",
    ],
    "environment": [
        "expansive golden-hour key with long shadows reaching toward camera, atmospheric backlight",
        "blue-hour ambient with warm practical lights scattered through scene, mood of dusk",
        "high noon hard sunlight with crisp shadows, deep saturation, vivid clarity",
        "overcast diffuse light wrapping the landscape, muted shadows, contemplative",
        "first-light dawn with horizontal warm sunlight grazing surfaces, low contrast",
        "moonlit ambient blue with practical warm accents, nocturnal atmosphere",
        "stormy directional light through breaking clouds, dramatic god-rays",
    ],
    "symbolic": [
        "expressive single-source key with deep negative space, theatrical shaping",
        "soft directional light with poetic atmospheric haze, dreamlike falloff",
        "high-contrast spotlight with crushed surroundings, visual emphasis on subject",
        "diffuse ethereal lighting with gentle bloom, otherworldly mood",
        "low warm key with shadow play across textured surface, evocative shaping",
    ],
    "macro": [
        "precise top-down key with controlled fill, crisp specular separation",
        "raking side light revealing micro-texture and surface detail",
        "soft diffuse omni-light eliminating shadows, clean product clarity",
        "warm focused beam with deep falloff, jewel-like emphasis on subject",
    ],
}

# ── Palette variants per expression mode ─────────────────────────────────────
PALETTE_VARIANTS: Dict[str, List[str]] = {
    "face": [
        "natural skin-faithful palette with warm earth-tone accents",
        "muted blue-green tonal palette with desaturated naturalistic skin",
        "warm sepia-leaning palette with golden highlights and soft amber midtones",
        "rich saturated jewel-toned palette with deep reds and emerald shadows",
        "cool naturalistic palette with pale blue ambience and faithful skin",
        "high-contrast palette with crushed blacks and creamy highlights, classic portraiture",
        "warm terracotta-and-ochre palette with sun-baked midtones",
        "neutral cinematic palette with subtle complementary teal-and-orange grading",
    ],
    "body": [
        "warm grounded palette with earthy midtones and natural shadow tones",
        "cool blue-leaning palette with restrained skin tones and atmospheric haze",
        "muted natural palette emphasizing wardrobe and environmental harmony",
        "rich warm palette with saturated wardrobe accents against neutral surrounds",
        "desaturated documentary palette with faithful color and minimal grading",
    ],
    "environment": [
        "warm golden-hour palette with amber sky and long warm shadows",
        "cool blue-hour palette with cyan sky transitioning to warm horizon",
        "saturated daylight palette with vivid sky and rich foliage",
        "muted overcast palette with soft greys and gentle color separation",
        "warm dawn palette with peach sky and dewy desaturated landscape",
        "moonlit palette with deep blues and warm practical highlights",
        "stormy palette with dramatic grey-greens and golden break-light accents",
    ],
    "symbolic": [
        "poetic restrained palette with one symbolic accent color emerging from the muted field",
        "monochromatic blue-leaning palette with selective saturated accent",
        "warm amber palette with controlled symbolic red or gold accent",
        "cool desaturated palette with one warm focal element",
        "high-contrast palette with crushed black surrounds and luminous subject color",
    ],
    "macro": [
        "clean premium palette with refined product separation and neutral surrounds",
        "rich saturated palette emphasizing material color and surface tone",
        "muted minimalist palette with subtle product highlights",
        "warm catalog palette with creamy backgrounds and faithful product hues",
    ],
}


def _hash_index(shot_index: int, modulo: int, salt: int = 0) -> int:
    """Deterministic but non-trivial cycle so consecutive shots of the same
    mode don't always rotate in lockstep. The salt lets palette and lighting
    cycle on different offsets so they don't covary."""
    if modulo <= 0:
        return 0
    # Multiply by a small prime to spread adjacent indices.
    return (int(shot_index) * 7 + salt) % modulo


# Framing-keyword overrides — when present in the framing_directive they
# take priority over the cycling, because the cue is genuinely specific.
_LIGHTING_FRAMING_OVERRIDES: List[tuple[tuple[str, ...], str]] = [
    (("tear-line", "extreme close-up", "eyes and brow"),
     "intimate close lighting with single soft key, catchlight in eyes, deep shadow falloff on edges"),
    (("silhouette", "backlit"),
     "strong backlight rendering subject in silhouette, rim glow only, deep front shadow"),
    (("wide", "establishing", "landscape", "vista"),
     "expansive natural lighting filling the entire frame, atmospheric depth across distance"),
    (("low angle", "hero shot"),
     "low warm key from below, heroic uplighting with controlled fill"),
    (("high angle", "top-down", "overhead"),
     "soft top-down ambient lighting with even fill"),
]


def pick_lighting_variant(*, expression_mode: str, shot_index: int,
                          framing_directive: str = "", meaning: str = "",
                          intensity: float = 0.5,
                          global_lighting: str = "") -> str:
    """Select a lighting variant for one shot.

    Resolution order:
    1. Specific framing-directive override (extreme close-up → intimate
       single-source; silhouette → strong backlight; etc.).
    2. Mode-specific cycling through ``LIGHTING_VARIANTS`` by shot_index.
    3. Global lighting family (final fallback)."""
    fd = (framing_directive or "").lower()
    for kws, lighting in _LIGHTING_FRAMING_OVERRIDES:
        if any(kw in fd for kw in kws):
            return lighting
    variants = LIGHTING_VARIANTS.get((expression_mode or "").lower())
    if variants:
        idx = _hash_index(shot_index, len(variants), salt=0)
        return variants[idx]
    return global_lighting or "soft cinematic natural lighting"


def pick_palette_variant(*, expression_mode: str, shot_index: int,
                         framing_directive: str = "", meaning: str = "",
                         global_palette: str = "") -> str:
    """Select a palette variant for one shot. Cycles on a different salt
    than lighting so the two don't covary into 'identical pairs'."""
    variants = PALETTE_VARIANTS.get((expression_mode or "").lower())
    if variants:
        idx = _hash_index(shot_index, len(variants), salt=3)
        return variants[idx]
    return global_palette or "balanced cinematic natural palette"


class StyleGradingEngine:
    """
    Qaivid Style Grading Engine
    """

    DEFAULT_STYLE_PRESET = "cinematic_natural"
    DEFAULT_LOOK_STRENGTH = 0.72

    def __init__(self):
        pass

    def apply_style(
        self,
        timeline: List[Dict[str, Any]],
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        validated_timeline = self._validate_timeline(timeline)
        validated_style_profile = self._validate_style_profile(style_profile or {})

        global_style = self._build_global_style_plan(
            timeline=validated_timeline,
            style_profile=validated_style_profile,
        )

        styled_timeline: List[Dict[str, Any]] = []

        for shot in validated_timeline:
            shot_style = self._build_shot_style(
                shot=shot,
                global_style=global_style,
                style_profile=validated_style_profile,
            )

            styled_prompt = self._build_styled_prompt(
                base_prompt=shot.get("visual_prompt", ""),
                shot_style=shot_style,
                framing_directive=shot.get("framing_directive", ""),
                composition_note=shot.get("composition_note", ""),
            )

            styled_shot = dict(shot)
            styled_shot["style_preset"] = shot_style["style_preset"]
            styled_shot["style_strength"] = shot_style["style_strength"]
            styled_shot["color_palette"] = shot_style["color_palette"]
            styled_shot["lighting_style"] = shot_style["lighting_style"]
            styled_shot["contrast_profile"] = shot_style["contrast_profile"]
            styled_shot["texture_profile"] = shot_style["texture_profile"]
            styled_shot["atmosphere_profile"] = shot_style["atmosphere_profile"]
            styled_shot["lens_feel"] = shot_style["lens_feel"]
            styled_shot["style_notes"] = shot_style["style_notes"]
            styled_shot["styled_visual_prompt"] = styled_prompt

            styled_timeline.append(styled_shot)

        return styled_timeline

    def _validate_timeline(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(timeline, list) or not timeline:
            raise ValueError("Timeline must be a non-empty list.")

        repaired: List[Dict[str, Any]] = []
        for i, shot in enumerate(timeline, start=1):
            if not isinstance(shot, dict):
                raise ValueError(f"Timeline shot at position {i} must be a dictionary.")

            repaired.append(
                {
                    "timeline_index": shot.get("timeline_index", i),
                    "shot_index": shot.get("shot_index", i),
                    "shot_id": shot.get("shot_id", f"shot_{i}"),
                    "start_time": self._coerce_float(shot.get("start_time", 0.0), 0.0),
                    "duration": self._coerce_float(shot.get("duration", 2.0), 2.0),
                    "end_time": self._coerce_float(shot.get("end_time", 2.0), 2.0),
                    "visual_prompt": str(shot.get("visual_prompt", "")).strip(),
                    "meaning": str(shot.get("meaning", "")).strip(),
                    "function": str(shot.get("function", "emotional_expression")).strip(),
                    "repeat_status": str(shot.get("repeat_status", "original")).strip().lower(),
                    "intensity": self._clamp_01(shot.get("intensity", 0.5), 0.5),
                    "expression_mode": self._repair_expression_mode(
                        shot.get("expression_mode", "environment")
                    ),
                    "transition": str(shot.get("transition", "straight_cut")).strip(),
                    "motion_scale": str(shot.get("motion_scale", "standard cinematic drift")).strip(),
                    "reference_image": shot.get("reference_image"),
                    "fidelity_lock": self._clamp_01(shot.get("fidelity_lock", 0.72), 0.72),
                    "camera_profile": shot.get("camera_profile", {}),
                    "environment_profile": shot.get("environment_profile", {}),
                    "continuity_anchor": shot.get("continuity_anchor", {}),
                    "rendering_notes": shot.get("rendering_notes", []),
                    # Cinematic variety fields (Task #50)
                    "motion_prompt": str(shot.get("motion_prompt", "")).strip(),
                    "framing_directive": str(shot.get("framing_directive", "")).strip(),
                    "composition_note": str(shot.get("composition_note", "")).strip(),
<<<<<<< HEAD
                    # Cinematography rig block (Task #69) — pass-through so
                    # styled_timeline keeps the structured rig/lens info that
                    # the storyboard UI badges + video generator depend on.
                    "cinematography": shot.get("cinematography"),
=======
>>>>>>> a59a5ef (Task #50 — Storyboard Cinematic Quality Overhaul (with code review fixes))
                }
            )

        return repaired

    def _validate_style_profile(self, style_profile: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(style_profile, dict):
            style_profile = {}

        return {
            "preset": str(style_profile.get("preset", self.DEFAULT_STYLE_PRESET)).strip(),
            "look_strength": self._clamp_01(
                style_profile.get("look_strength", self.DEFAULT_LOOK_STRENGTH),
                self.DEFAULT_LOOK_STRENGTH,
            ),
            "color_bias": self._safe_string(style_profile.get("color_bias")),
            "lighting_bias": self._safe_string(style_profile.get("lighting_bias")),
            "contrast_bias": self._safe_string(style_profile.get("contrast_bias")),
            "texture_bias": self._safe_string(style_profile.get("texture_bias")),
            "atmosphere_bias": self._safe_string(style_profile.get("atmosphere_bias")),
            "lens_bias": self._safe_string(style_profile.get("lens_bias")),
        }

    def _build_global_style_plan(
        self,
        timeline: List[Dict[str, Any]],
        style_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        avg_intensity = sum(shot["intensity"] for shot in timeline) / len(timeline)

        genre_hint = self._infer_primary_genre_from_timeline(timeline)
        preset = style_profile["preset"] or self.DEFAULT_STYLE_PRESET

        base_plan = {
            "preset": preset,
            "genre_hint": genre_hint,
            "avg_intensity": avg_intensity,
            "palette_family": self._choose_palette_family(preset, genre_hint, avg_intensity),
            "lighting_family": self._choose_lighting_family(preset, genre_hint, avg_intensity),
            "contrast_family": self._choose_contrast_family(preset, genre_hint, avg_intensity),
            "texture_family": self._choose_texture_family(preset, genre_hint),
            "atmosphere_family": self._choose_atmosphere_family(preset, genre_hint, avg_intensity),
            "lens_family": self._choose_lens_family(preset, genre_hint),
        }

        return base_plan

    def _infer_primary_genre_from_timeline(self, timeline: List[Dict[str, Any]]) -> str:
        styles = []
        for shot in timeline:
            camera_profile = shot.get("camera_profile", {})
            style = str(camera_profile.get("style", "")).strip().lower()
            if style:
                styles.append(style)

        if not styles:
            return "cinematic"

        if any("commercial" in s for s in styles):
            return "ad"
        if any("observational" in s for s in styles):
            return "documentary"
        if any("performance" in s for s in styles):
            return "script"
        if any("poetic" in s for s in styles):
            return "song_poem"

        return "cinematic"

    def _build_shot_style(
        self,
        shot: Dict[str, Any],
        global_style: Dict[str, Any],
        style_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        intensity = shot["intensity"]
        expression_mode = shot["expression_mode"]
        repeat_status = shot["repeat_status"]
        meaning = shot.get("meaning", "")
        environment_profile = shot.get("environment_profile", {})
        regional_required = bool(environment_profile.get("regional_grounding_required", False))

        style_preset = global_style["preset"]
        style_strength = self._derive_style_strength(
            base_strength=style_profile["look_strength"],
            intensity=intensity,
            repeat_status=repeat_status,
        )

        shot_idx = int(shot.get("shot_index") or shot.get("timeline_index") or 0)
        framing_directive = str(shot.get("framing_directive") or "")

        color_palette = self._derive_color_palette(
            global_style=global_style,
            intensity=intensity,
            expression_mode=expression_mode,
            repeat_status=repeat_status,
            regional_required=regional_required,
            color_bias=style_profile.get("color_bias"),
            shot_index=shot_idx,
            framing_directive=framing_directive,
            meaning=meaning,
        )

        lighting_style = self._derive_lighting_style(
            global_style=global_style,
            intensity=intensity,
            expression_mode=expression_mode,
            lighting_bias=style_profile.get("lighting_bias"),
            shot_index=shot_idx,
            framing_directive=framing_directive,
            meaning=meaning,
        )

        contrast_profile = self._derive_contrast_profile(
            global_style=global_style,
            intensity=intensity,
            contrast_bias=style_profile.get("contrast_bias"),
        )

        texture_profile = self._derive_texture_profile(
            global_style=global_style,
            expression_mode=expression_mode,
            regional_required=regional_required,
            texture_bias=style_profile.get("texture_bias"),
        )

        atmosphere_profile = self._derive_atmosphere_profile(
            global_style=global_style,
            meaning=meaning,
            intensity=intensity,
            repeat_status=repeat_status,
            atmosphere_bias=style_profile.get("atmosphere_bias"),
        )

        lens_feel = self._derive_lens_feel(
            global_style=global_style,
            expression_mode=expression_mode,
            lens_bias=style_profile.get("lens_bias"),
        )

        style_notes = self._build_style_notes(
            intensity=intensity,
            expression_mode=expression_mode,
            repeat_status=repeat_status,
            regional_required=regional_required,
        )

        return {
            "style_preset": style_preset,
            "style_strength": style_strength,
            "color_palette": color_palette,
            "lighting_style": lighting_style,
            "contrast_profile": contrast_profile,
            "texture_profile": texture_profile,
            "atmosphere_profile": atmosphere_profile,
            "lens_feel": lens_feel,
            "style_notes": style_notes,
        }

    def _derive_style_strength(
        self,
        base_strength: float,
        intensity: float,
        repeat_status: str,
    ) -> float:
        strength = base_strength

        if intensity > 0.8:
            strength += 0.08
        elif intensity < 0.3:
            strength -= 0.05

        if repeat_status == "repeat":
            strength += 0.03

        return round(max(0.0, min(1.0, strength)), 3)

    def _derive_color_palette(
        self,
        global_style: Dict[str, Any],
        intensity: float,
        expression_mode: str,
        repeat_status: str,
        regional_required: bool,
        color_bias: str,
        shot_index: int = 0,
        framing_directive: str = "",
        meaning: str = "",
    ) -> str:
        palette = pick_palette_variant(
            expression_mode=expression_mode,
            shot_index=shot_index,
            framing_directive=framing_directive,
            meaning=meaning,
            global_palette=global_style.get("palette_family", ""),
        )

        if intensity > 0.8:
            palette += ", slightly heightened saturation and emotional emphasis"
        elif intensity < 0.3:
            palette += ", softened saturation and subdued tonal restraint"

        if repeat_status == "repeat":
            palette += ", preserve previous palette memory with subtle variation"

        if regional_required:
            palette += ", maintain regionally authentic material and environmental tones"

        if color_bias:
            palette += f", bias toward {color_bias}"

        return palette

    def _derive_lighting_style(
        self,
        global_style: Dict[str, Any],
        intensity: float,
        expression_mode: str,
        lighting_bias: str,
        shot_index: int = 0,
        framing_directive: str = "",
        meaning: str = "",
    ) -> str:
        lighting = pick_lighting_variant(
            expression_mode=expression_mode,
            shot_index=shot_index,
            framing_directive=framing_directive,
            meaning=meaning,
            intensity=intensity,
            global_lighting=global_style.get("lighting_family", ""),
        )

        if intensity > 0.8:
            lighting += ", slightly more dramatic emphasis"
        elif intensity < 0.3:
            lighting += ", quiet restrained softness"

        if lighting_bias:
            lighting += f", bias toward {lighting_bias}"

        return lighting

    def _derive_contrast_profile(
        self,
        global_style: Dict[str, Any],
        intensity: float,
        contrast_bias: str,
    ) -> str:
        contrast = global_style["contrast_family"]

        if intensity > 0.82:
            contrast += ", with stronger tonal separation"
        elif intensity < 0.3:
            contrast += ", with softer rolloff"

        if contrast_bias:
            contrast += f", bias toward {contrast_bias}"

        return contrast

    def _derive_texture_profile(
        self,
        global_style: Dict[str, Any],
        expression_mode: str,
        regional_required: bool,
        texture_bias: str,
    ) -> str:
        texture = global_style["texture_family"]

        if expression_mode == "macro":
            texture = "clean premium surface detail with precise micro-texture rendering"

        if regional_required:
            texture += ", preserve authentic material textures and lived-in surfaces"

        if texture_bias:
            texture += f", bias toward {texture_bias}"

        return texture

    def _derive_atmosphere_profile(
        self,
        global_style: Dict[str, Any],
        meaning: str,
        intensity: float,
        repeat_status: str,
        atmosphere_bias: str,
    ) -> str:
        atmosphere = global_style["atmosphere_family"]

        if intensity > 0.8:
            atmosphere += ", emotionally charged"
        elif intensity < 0.3:
            atmosphere += ", spacious and restrained"

        if repeat_status == "repeat":
            atmosphere += ", carrying emotional memory from earlier beats"

        if meaning:
            atmosphere += f", aligned with the emotional meaning of {meaning}"

        if atmosphere_bias:
            atmosphere += f", bias toward {atmosphere_bias}"

        return atmosphere

    def _derive_lens_feel(
        self,
        global_style: Dict[str, Any],
        expression_mode: str,
        lens_bias: str,
    ) -> str:
        lens_feel = global_style["lens_family"]

        if expression_mode == "face":
            lens_feel = "intimate portrait-oriented cinematic lens feel"
        elif expression_mode == "symbolic":
            lens_feel = "poetic cinematic lens feel with expressive depth and atmosphere"
        elif expression_mode == "macro":
            lens_feel = "high-detail premium macro-capable lens feel"

        if lens_bias:
            lens_feel += f", bias toward {lens_bias}"

        return lens_feel

    def _build_style_notes(
        self,
        intensity: float,
        expression_mode: str,
        repeat_status: str,
        regional_required: bool,
    ) -> List[str]:
        notes: List[str] = []

        if repeat_status == "repeat":
            notes.append("Maintain visual continuity with prior repeated emotional beat.")

        if expression_mode == "face":
            notes.append("Protect facial readability and eye detail.")
        elif expression_mode == "symbolic":
            notes.append("Keep symbolism emotionally grounded, not random or surreal.")
        elif expression_mode == "macro":
            notes.append("Protect clarity, polish, and focal precision.")

        if regional_required:
            notes.append("Preserve regional authenticity in color, material, and atmosphere.")

        if intensity > 0.8:
            notes.append("Allow stronger visual emphasis without breaking continuity.")
        elif intensity < 0.3:
            notes.append("Favor restraint over excessive stylization.")

        return notes

    def _build_styled_prompt(
        self,
        base_prompt: str,
        shot_style: Dict[str, Any],
        framing_directive: str = "",
        composition_note: str = "",
    ) -> str:
        parts = [base_prompt.strip()]

        if framing_directive:
            parts.append(f"Framing directive: {framing_directive}.")
        if composition_note:
            parts.append(f"Composition note: {composition_note}.")

        parts.extend([
            f"Style preset: {shot_style['style_preset']}.",
            f"Style strength: {shot_style['style_strength']}.",
            f"Color palette: {shot_style['color_palette']}.",
            f"Lighting style: {shot_style['lighting_style']}.",
            f"Contrast profile: {shot_style['contrast_profile']}.",
            f"Texture profile: {shot_style['texture_profile']}.",
            f"Atmosphere profile: {shot_style['atmosphere_profile']}.",
            f"Lens feel: {shot_style['lens_feel']}.",
        ])

        if shot_style["style_notes"]:
            parts.append(f"Style notes: {' '.join(shot_style['style_notes'])}")

        return " ".join(parts).strip()

    def _choose_palette_family(self, preset: str, genre_hint: str, avg_intensity: float) -> str:
        if preset == "monochrome":
            return "refined monochrome tonal palette"
        if genre_hint == "ad":
            return "clean premium commercial palette"
        if genre_hint == "documentary":
            return "natural restrained documentary palette"
        if genre_hint == "song_poem":
            return "cinematic lyrical palette"
        if avg_intensity > 0.75:
            return "rich cinematic palette with controlled emotional emphasis"
        return "balanced cinematic natural palette"

    def _choose_lighting_family(self, preset: str, genre_hint: str, avg_intensity: float) -> str:
        if preset == "noir":
            return "directional moody lighting with dramatic shaping"
        if genre_hint == "documentary":
            return "naturalistic observational lighting"
        if genre_hint == "ad":
            return "clean premium controlled lighting"
        if avg_intensity > 0.75:
            return "dramatic cinematic lighting"
        return "soft cinematic natural lighting"

    def _choose_contrast_family(self, preset: str, genre_hint: str, avg_intensity: float) -> str:
        if preset == "soft_poetic":
            return "gentle low-contrast rolloff"
        if preset == "noir":
            return "deep contrast with bold separation"
        if genre_hint == "documentary":
            return "measured natural contrast"
        if avg_intensity > 0.75:
            return "moderately heightened cinematic contrast"
        return "balanced contrast with filmic restraint"

    def _choose_texture_family(self, preset: str, genre_hint: str) -> str:
        if genre_hint == "ad":
            return "clean polished texture rendering"
        if genre_hint == "documentary":
            return "natural lived-in texture rendering"
        if preset == "dreamy":
            return "soft textured rendering with gentle bloom"
        return "filmic texture rendering with grounded detail"

    def _choose_atmosphere_family(self, preset: str, genre_hint: str, avg_intensity: float) -> str:
        if genre_hint == "documentary":
            return "observational authentic atmosphere"
        if genre_hint == "song_poem":
            return "lyrical atmospheric tone"
        if preset == "dreamy":
            return "soft suspended dreamlike atmosphere"
        if avg_intensity > 0.75:
            return "tense emotionally heightened atmosphere"
        return "grounded cinematic atmosphere"

    def _choose_lens_family(self, preset: str, genre_hint: str) -> str:
        if genre_hint == "ad":
            return "premium commercial lens feel"
        if genre_hint == "documentary":
            return "natural observational lens feel"
        if preset == "dreamy":
            return "soft cinematic lens feel with gentle depth"
        return "cinematic film lens feel"

    def _repair_expression_mode(self, value: Any) -> str:
        allowed = {"face", "body", "environment", "symbolic", "macro"}
        if isinstance(value, str):
            value = value.strip().lower()
            if value in allowed:
                return value
        return "environment"

    def _coerce_float(self, value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _clamp_01(self, value: Any, fallback: float) -> float:
        try:
            num = float(value)
            return max(0.0, min(1.0, num))
        except Exception:
            return fallback

    def _safe_string(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
