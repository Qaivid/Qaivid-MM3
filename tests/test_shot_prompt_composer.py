"""Tests for the shot prompt composer.

The composer's job is to turn the verbose, often-truncated
styled_visual_prompt into a tight ~1100-char prompt the diffusion
model can actually attend to — while preserving cross-shot continuity
through explicit reference-image cues.
"""

from shot_prompt_composer import (
    DEFAULT_NEGATIVE,
    QUALITY_BOOSTERS,
    compose_image_prompt,
)


_CHAR = {
    "name": "Arjun",
    "role": "protagonist",
    "age_range": "30s",
    "gender": "male",
    "ethnicity": "Punjabi",
    "wardrobe": "weathered cream linen kurta, indigo shawl",
    "grooming": "salt-and-pepper beard, shoulder-length hair tied back",
}

_LOC = {
    "name": "Old Lahore courtyard",
    "description": "sandstone arches, jasmine vines, brass lanterns",
    "mood": "melancholic dusk",
}

_SHOT = {
    "shot_index": 4,
    "meaning": "Arjun pauses at the threshold, weighing whether to step "
               "back into the home he abandoned.",
    "framing_directive": "Medium close-up, slight low angle, shallow depth of field",
    "motion_prompt": "slow dolly-in 12mm, breathing handheld",
    "lighting_style": "soft golden-hour rim light, deep shadows",
    "color_palette": "warm ochre and indigo, muted teal accents",
    "expression_mode": "face",
    "cinematography": {
        "rig": "Arri Alexa Mini LF, 50mm Cooke S4",
        "lens": "50mm anamorphic, T1.4, shallow DOF",
    },
}


def test_composer_leads_with_meaning():
    prompt, _ = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    # Story beat must lead the prompt (most-attended position).
    assert prompt.lower().startswith(("arjun pauses", "50mm")) or \
        "arjun pauses" in prompt[:500].lower()
    # Concrete subject — never the dreaded "Adult Unspecified".
    assert "unspecified" not in prompt.lower()
    assert "punjabi" in prompt.lower() and "man" in prompt.lower()


def test_composer_includes_wardrobe_for_continuity():
    prompt, _ = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    assert "kurta" in prompt.lower()
    assert "beard" in prompt.lower()


def test_composer_includes_location_description():
    prompt, _ = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    assert "lahore" in prompt.lower() or "sandstone" in prompt.lower()


def test_composer_adds_continuity_cue_when_char_ref_present():
    prompt, _ = compose_image_prompt(
        _SHOT, character=_CHAR, location=_LOC,
        has_character_ref=True,
    )
    # The continuity cue is the key piece for cross-shot consistency.
    assert "same person" in prompt.lower() or "reference image" in prompt.lower()


def test_composer_adds_env_continuity_cue_when_env_ref_present():
    prompt, _ = compose_image_prompt(
        _SHOT, character=_CHAR, location=_LOC,
        has_environment_ref=True,
    )
    assert "environment reference" in prompt.lower() or "established" in prompt.lower()


def test_composer_appends_quality_boosters():
    prompt, _ = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    assert "arri alexa" in prompt.lower()
    assert "photorealistic" in prompt.lower()


def test_composer_returns_negative_prompt():
    _, neg = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    assert neg == DEFAULT_NEGATIVE
    assert "blurry" in neg and "watermark" in neg


def test_composer_trims_to_under_1200_chars():
    prompt, _ = compose_image_prompt(
        _SHOT, character=_CHAR, location=_LOC,
        has_character_ref=True, has_environment_ref=True,
        cine_prefix="50mm anamorphic Arri Alexa Mini LF, T1.4, shallow DOF",
    )
    # Plenty of headroom under FAL's effective attention window.
    assert len(prompt) <= 1200
    # But still substantial — not a one-liner.
    assert len(prompt) >= 200


def test_composer_strips_director_instructions_from_user_override():
    """If the user pastes director-style instructions into their edit,
    the composer should still strip them — the model can't render them."""
    user_text = (
        "A man at a doorway in golden light. "
        "Maintain strict character continuity with the reference plate. "
        "Performance: contemplative, weighted. "
        "Hard restrictions: no logos, no text overlays."
    )
    prompt, _ = compose_image_prompt(
        _SHOT, character=_CHAR, user_override=user_text,
    )
    assert "maintain strict character continuity" not in prompt.lower()
    assert "performance:" not in prompt.lower()
    assert "hard restrictions" not in prompt.lower()
    assert "doorway in golden light" in prompt.lower()


def test_composer_user_override_keeps_quality_boosters():
    prompt, _ = compose_image_prompt(
        _SHOT, character=_CHAR,
        user_override="A wide shot of a temple courtyard at dawn.",
        cine_prefix="35mm anamorphic",
    )
    assert "temple courtyard at dawn" in prompt.lower()
    assert "arri alexa" in prompt.lower()  # boosters reattached
    assert "35mm anamorphic" in prompt.lower()  # cine prefix attached


def test_composer_handles_missing_character_gracefully():
    prompt, _ = compose_image_prompt(_SHOT, character=None, location=None)
    assert "unspecified" not in prompt.lower()
    assert "person" in prompt.lower() or "arjun pauses" in prompt.lower()
    # Still has quality boosters even when fields are sparse.
    assert "photorealistic" in prompt.lower()


def test_composer_falls_back_to_environment_profile_when_no_location():
    shot = {
        **_SHOT,
        "environment_profile": {
            "location_dna": "narrow alley with hanging laundry lines",
            "world_assumptions": {
                "time_of_day": "predawn",
                "season": "monsoon",
            },
        },
    }
    prompt, _ = compose_image_prompt(shot, character=_CHAR, location=None)
    assert "alley" in prompt.lower() or "predawn" in prompt.lower()


def test_composer_no_director_instructions_leak_through():
    """Regression: the verbose styled_visual_prompt was 95% identical
    across all 70 shots because director-instruction sentences dominated
    its first 1500 chars. None of those phrases should appear in the
    composed prompt."""
    prompt, _ = compose_image_prompt(_SHOT, character=_CHAR, location=_LOC)
    forbidden = [
        "maintain strict character continuity",
        "performance:",
        "function:",
        "ambiguity handling",
        "spine anchor",
        "central metaphor",
        "treatment:",
        "repeat status",
    ]
    lower = prompt.lower()
    for phrase in forbidden:
        assert phrase not in lower, f"Director instruction leaked: {phrase!r}"


def test_quality_boosters_constant_is_nonempty():
    assert QUALITY_BOOSTERS and "alexa" in QUALITY_BOOSTERS.lower()
