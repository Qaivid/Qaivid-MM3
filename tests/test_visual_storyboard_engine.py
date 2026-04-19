"""
pytest coverage for VisualStoryboardEngine cinematic quality logic (Task #50).

Covers:
- Framing rotation: no two consecutive face shots share the same framing_directive
- Chorus escalation: second chorus block starts at a different framing rotation than the first
- Body language composition note: "tears"/"despair"/"longing" keywords yield a non-empty composition_note
- Motion prompt length: every shot's motion_prompt is <= 250 characters
- Mandatory environment cutaway: after 4+ consecutive face/body shots the next
  environment/symbolic shot receives the "wide establishing cutaway" prefix override
"""

import pytest
from visual_storyboard_engine import VisualStoryboardEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(
    line_index: int,
    expression_mode: str = "face",
    repeat_status: str = "original",
    emotional_meaning: str = "",
    implied_meaning: str = "",
    meaning: str = "",
    intensity: float = 0.5,
) -> dict:
    """Return a minimal but valid line_meanings entry."""
    return {
        "line_index": line_index,
        "text": f"line {line_index}",
        "literal_meaning": "",
        "implied_meaning": implied_meaning,
        "emotional_meaning": emotional_meaning,
        "cultural_meaning": "",
        "meaning": meaning,
        "function": "emotional_expression",
        "repeat_status": repeat_status,
        "intensity": intensity,
        "expression_mode": expression_mode,
        "visualization_mode": "default",
        "visual_suitability": "suitable",
    }


def _make_context(shots: list) -> dict:
    """Return a minimal valid context packet containing the given shots."""
    return {"line_meanings": shots}


def _build(shots: list) -> list:
    """Instantiate a fresh engine and build a storyboard from the given shots."""
    engine = VisualStoryboardEngine()
    return engine.build_storyboard(_make_context(shots))


# ---------------------------------------------------------------------------
# Test 1 – No two consecutive face shots share the same framing_directive
# ---------------------------------------------------------------------------

def test_consecutive_face_shots_have_different_framing_directives():
    """
    Build a run of consecutive face-mode shots and verify that each pair of
    adjacent shots carries a distinct framing_directive.
    """
    num_shots = 7  # equals the length of _FACE_FRAMES, hitting every entry once
    shots = [_make_shot(i, expression_mode="face") for i in range(1, num_shots + 1)]
    storyboard = _build(shots)

    assert len(storyboard) == num_shots

    for idx in range(len(storyboard) - 1):
        current = storyboard[idx]["framing_directive"]
        nxt = storyboard[idx + 1]["framing_directive"]
        assert current != nxt, (
            f"Shot {idx} and shot {idx + 1} share the same framing_directive: '{current}'"
        )


# ---------------------------------------------------------------------------
# Test 2 – Chorus repeat blocks start at a different framing rotation
# ---------------------------------------------------------------------------

def test_second_chorus_starts_at_different_framing_than_first_chorus():
    """
    The engine applies a +2 counter offset when a second repeat block begins,
    ensuring the opening framing_directive of chorus 2 differs from that of
    chorus 1.
    """
    shots = [
        # Two original shots advance the face counter to 2
        _make_shot(1, expression_mode="face", repeat_status="original"),
        _make_shot(2, expression_mode="face", repeat_status="original"),
        # Chorus 1: first repeat block (chorus_count becomes 1, no offset yet)
        _make_shot(3, expression_mode="face", repeat_status="repeat"),
        _make_shot(4, expression_mode="face", repeat_status="repeat"),
        # Two more originals to advance the face counter further
        _make_shot(5, expression_mode="face", repeat_status="original"),
        _make_shot(6, expression_mode="face", repeat_status="original"),
        # Chorus 2: second repeat block starts — +2 offset is applied now
        _make_shot(7, expression_mode="face", repeat_status="repeat"),
    ]

    storyboard = _build(shots)

    first_chorus_opening_directive = storyboard[2]["framing_directive"]
    second_chorus_opening_directive = storyboard[6]["framing_directive"]

    assert first_chorus_opening_directive != second_chorus_opening_directive, (
        "First and second chorus blocks started with the same framing_directive; "
        f"both used: '{first_chorus_opening_directive}'"
    )


# ---------------------------------------------------------------------------
# Test 3 – Emotional keywords produce a non-empty composition_note on body shots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keyword", ["tears", "despair", "longing"])
def test_body_shot_emotional_keywords_produce_composition_note(keyword: str):
    """
    Body-mode shots whose emotional_meaning contains one of the tracked
    emotional keywords must carry a non-empty composition_note.
    """
    shots = [
        _make_shot(
            1,
            expression_mode="body",
            emotional_meaning=f"A deep sense of {keyword} overwhelms the moment",
        )
    ]
    storyboard = _build(shots)

    composition_note = storyboard[0]["composition_note"]
    assert composition_note, (
        f"Expected a non-empty composition_note for body shot with emotional keyword "
        f"'{keyword}', but got an empty string."
    )


# ---------------------------------------------------------------------------
# Test 4 – Every shot's motion_prompt is at most 250 characters
# ---------------------------------------------------------------------------

def test_all_motion_prompts_within_250_chars():
    """
    motion_prompt must never exceed 250 characters for any expression mode or
    intensity level, including the high-intensity word-replacement path.
    """
    modes = ["face", "body", "environment", "symbolic", "macro"]
    intensities = [0.0, 0.3, 0.5, 0.8, 1.0]

    shots = []
    idx = 1
    for mode in modes:
        for intensity in intensities:
            shots.append(
                _make_shot(idx, expression_mode=mode, intensity=intensity)
            )
            idx += 1

    storyboard = _build(shots)

    for shot_result in storyboard:
        prompt = shot_result["motion_prompt"]
        assert len(prompt) <= 250, (
            f"motion_prompt for shot {shot_result['shot_index']} exceeds 250 chars "
            f"({len(prompt)} chars): '{prompt}'"
        )


# ---------------------------------------------------------------------------
# Test 5 – Wide establishing cutaway override after 4+ consecutive face/body shots
# ---------------------------------------------------------------------------

def test_cutaway_override_after_four_consecutive_character_shots():
    """
    After 4 or more consecutive face/body shots, the next environment or
    symbolic shot must have its framing_directive prefixed with the
    "wide establishing cutaway" instruction.
    """
    shots = [
        # 4 consecutive face shots to trigger the pending_cutaway flag
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="face"),
        _make_shot(3, expression_mode="body"),
        _make_shot(4, expression_mode="face"),
        # 5th shot is environment — should get the wide cutaway override
        _make_shot(5, expression_mode="environment"),
    ]
    storyboard = _build(shots)

    env_shot = storyboard[4]
    assert env_shot["expression_mode"] == "environment"
    assert env_shot["framing_directive"].startswith("wide establishing cutaway"), (
        f"Expected framing_directive to start with 'wide establishing cutaway', "
        f"but got: '{env_shot['framing_directive']}'"
    )


def test_cutaway_override_not_applied_before_four_consecutive_shots():
    """
    Fewer than 4 consecutive face/body shots must NOT trigger the cutaway override.
    """
    shots = [
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="body"),
        _make_shot(3, expression_mode="face"),
        # Only 3 consecutive — no override expected
        _make_shot(4, expression_mode="environment"),
    ]
    storyboard = _build(shots)

    env_shot = storyboard[3]
    assert not env_shot["framing_directive"].startswith("wide establishing cutaway"), (
        "Unexpected cutaway override applied after only 3 consecutive face/body shots."
    )


# ---------------------------------------------------------------------------
# Test 6 – Macro shots rotate _FACE_FRAMES independently of the face counter
# ---------------------------------------------------------------------------

def test_macro_frame_rotation_is_independent_of_face_counter():
    """
    Macro shots borrow the _FACE_FRAMES table for framing directives, but they
    must rotate through that table using their own counter — completely
    independent of the face-mode counter. Interleaved face/macro shots must
    each advance only their own counter.
    """
    face_frames = VisualStoryboardEngine._FACE_FRAMES

    # Two face shots, then two macro shots, then another face shot, then macro.
    # Expected indices:
    #   face → 0, 1, _, _, 2, _
    #   macro → _, _, 0, 1, _, 2
    shots = [
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="face"),
        _make_shot(3, expression_mode="macro"),
        _make_shot(4, expression_mode="macro"),
        _make_shot(5, expression_mode="face"),
        _make_shot(6, expression_mode="macro"),
    ]
    storyboard = _build(shots)

    assert storyboard[0]["framing_directive"] == face_frames[0]
    assert storyboard[1]["framing_directive"] == face_frames[1]
    assert storyboard[2]["framing_directive"] == face_frames[0]
    assert storyboard[3]["framing_directive"] == face_frames[1]
    assert storyboard[4]["framing_directive"] == face_frames[2]
    assert storyboard[5]["framing_directive"] == face_frames[2]


def test_consecutive_macro_shots_have_different_framing_directives():
    """
    Build a run of consecutive macro-mode shots covering the full _FACE_FRAMES
    table and verify each adjacent pair carries a distinct framing_directive.
    """
    num_shots = len(VisualStoryboardEngine._FACE_FRAMES)
    shots = [_make_shot(i, expression_mode="macro") for i in range(1, num_shots + 1)]
    storyboard = _build(shots)

    assert len(storyboard) == num_shots
    for idx in range(len(storyboard) - 1):
        current = storyboard[idx]["framing_directive"]
        nxt = storyboard[idx + 1]["framing_directive"]
        assert current != nxt, (
            f"Macro shot {idx} and shot {idx + 1} share the same framing_directive: '{current}'"
        )


# ---------------------------------------------------------------------------
# Test 7 – Macro motion_prompt stays within 250 chars under all intensities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intensity", [0.0, 0.1, 0.29, 0.3, 0.5, 0.79, 0.8, 0.95, 1.0])
def test_macro_build_motion_prompt_length_within_250_chars(intensity: float):
    """
    Directly exercise `_build_motion_prompt` for macro mode at every intensity
    band (low, mid, high) and across every entry in the macro template table,
    bypassing any cinematography-engine override path.
    """
    engine = VisualStoryboardEngine()
    macro_templates = engine._MOTION_TEMPLATES["macro"]

    for frame_idx in range(len(macro_templates)):
        shot = _make_shot(frame_idx + 1, expression_mode="macro", intensity=intensity)
        prompt = engine._build_motion_prompt("macro", frame_idx, shot)

        assert prompt, f"Empty macro motion prompt at frame_idx={frame_idx}, intensity={intensity}"
        assert len(prompt) <= 250, (
            f"Macro motion_prompt exceeds 250 chars (frame_idx={frame_idx}, "
            f"intensity={intensity}, len={len(prompt)}): '{prompt}'"
        )


def test_macro_motion_prompt_in_storyboard_within_250_chars_all_intensities():
    """
    End-to-end: every macro shot built via `build_storyboard` carries a
    motion_prompt of <= 250 chars at every intensity band.
    """
    intensities = [0.0, 0.2, 0.5, 0.8, 1.0]
    shots = [
        _make_shot(i + 1, expression_mode="macro", intensity=intensity)
        for i, intensity in enumerate(intensities)
    ]
    storyboard = _build(shots)

    for shot_result in storyboard:
        prompt = shot_result["motion_prompt"]
        assert prompt
        assert len(prompt) <= 250, (
            f"Macro motion_prompt for shot {shot_result['shot_index']} exceeds 250 chars "
            f"({len(prompt)} chars): '{prompt}'"
        )


# ---------------------------------------------------------------------------
# Test 8 – Macro shots do not inadvertently trigger the cutaway override
# ---------------------------------------------------------------------------

def test_macro_shot_does_not_receive_cutaway_override():
    """
    The cutaway override only applies to environment/symbolic shots. A macro
    shot following 4+ consecutive face/body shots must NOT receive the
    "wide establishing cutaway" prefix on its framing_directive.
    """
    shots = [
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="body"),
        _make_shot(3, expression_mode="face"),
        _make_shot(4, expression_mode="body"),
        # 5th shot is macro — must NOT receive the cutaway override
        _make_shot(5, expression_mode="macro"),
    ]
    storyboard = _build(shots)

    macro_shot = storyboard[4]
    assert macro_shot["expression_mode"] == "macro"
    assert not macro_shot["framing_directive"].startswith("wide establishing cutaway"), (
        f"Macro shot unexpectedly received cutaway override: "
        f"'{macro_shot['framing_directive']}'"
    )


def test_macro_shot_resets_consecutive_face_body_run_for_cutaway():
    """
    A macro shot is neither face nor body, so it must reset the consecutive
    face/body counter that drives the cutaway override. After a macro shot
    interrupts the run, a subsequent environment shot should NOT receive the
    override unless 4+ new consecutive face/body shots have accumulated.
    """
    shots = [
        # Build up 4 consecutive face/body shots — would normally arm the cutaway
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="face"),
        _make_shot(3, expression_mode="body"),
        _make_shot(4, expression_mode="face"),
        # Macro interrupts and resets the counter
        _make_shot(5, expression_mode="macro"),
        # Only one face shot follows — not enough to re-arm the cutaway
        _make_shot(6, expression_mode="face"),
        _make_shot(7, expression_mode="environment"),
    ]
    storyboard = _build(shots)

    env_shot = storyboard[6]
    assert env_shot["expression_mode"] == "environment"
    assert not env_shot["framing_directive"].startswith("wide establishing cutaway"), (
        "Environment shot received cutaway override even though a macro shot "
        f"reset the consecutive face/body run: '{env_shot['framing_directive']}'"
    )


def test_cutaway_override_also_triggers_for_symbolic_shot():
    """
    The wide establishing cutaway override must also be applied to symbolic
    shots (not just environment shots) after 4+ consecutive face/body shots.
    """
    shots = [
        _make_shot(1, expression_mode="face"),
        _make_shot(2, expression_mode="face"),
        _make_shot(3, expression_mode="body"),
        _make_shot(4, expression_mode="body"),
        _make_shot(5, expression_mode="symbolic"),
    ]
    storyboard = _build(shots)

    symbolic_shot = storyboard[4]
    assert symbolic_shot["expression_mode"] == "symbolic"
    assert symbolic_shot["framing_directive"].startswith("wide establishing cutaway"), (
        f"Expected cutaway override on symbolic shot, "
        f"but got: '{symbolic_shot['framing_directive']}'"
    )
