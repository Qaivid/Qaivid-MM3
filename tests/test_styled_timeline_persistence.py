"""
End-to-end test: cinematic_beat + shot_event fields survive assembly and style.

Verifies the MM3.1 requirement that every cinematic beat field — including
`subject_action` and `environment_usage` — travels from the storyboard through
RhythmicAssemblyEngine.assemble_timeline() → StyleGradingEngine.apply_style()
without being dropped or overwritten.

This is the 'stage 2 persistence' check requested by the code reviewer.
"""
import pytest

from rhythmic_assembly_engine import RhythmicAssemblyEngine
from style_grading_engine import StyleGradingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(idx: int) -> dict:
    """Build a minimal storyboard shot that carries all MM3.1 cinematic fields."""
    return {
        "line_index": idx,
        "expression_mode": "body",
        "intensity": 0.6,
        "repeat_status": "original",
        "meaning": "longing",
        "emotional_meaning": "longing",
        "arc_position": idx,
        "framing_directive": "wide body shot",
        "composition_note": "subject faces door",
        "visual_prompt": "Character walks toward the doorway as light shifts across the threshold.",
        "location_dna": "interior_corridor",
        "cinematic_beat": {
            "subject_action": "walks toward the doorway with careful purpose",
            "trigger_event": "door opens ahead",
            "environment_usage": "corridor lit from one end, shadow trailing behind",
            "object_usage": "hand brushes the doorframe",
            "camera_motive": "track forward with subject",
            "lyric_relation_type": "literal",
        },
        "shot_event": {
            "action": "walks toward the doorway with careful purpose",
            "environment_usage": "corridor lit from one end",
            "camera_motivation": "track forward with subject",
            "shot_type": "body_movement",
            "is_generic": False,
            "is_valid": True,
        },
        "shot_type": "body_movement",
        "shot_validation": {"is_generic": False, "is_valid": True},
    }


_AUDIO_DATA = {
    "bpm": 120,
    "beats_per_bar": 4,
    "intensity_curve": [0.6, 0.7, 0.5, 0.8],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cinematic_beat_survives_assembly():
    """cinematic_beat dict must be present on every assembled timeline entry."""
    engine = RhythmicAssemblyEngine()
    storyboard = [_make_shot(i) for i in range(4)]
    timeline = engine.assemble_timeline(storyboard, _AUDIO_DATA)

    assert len(timeline) == 4, "Timeline length must match storyboard length"
    for entry in timeline:
        assert "cinematic_beat" in entry, "cinematic_beat must survive assembly"
        beat = entry["cinematic_beat"]
        assert beat is not None, "cinematic_beat must not be None"


def test_subject_action_survives_assembly():
    """cinematic_beat.subject_action must survive assembly unchanged."""
    engine = RhythmicAssemblyEngine()
    storyboard = [_make_shot(i) for i in range(4)]
    timeline = engine.assemble_timeline(storyboard, _AUDIO_DATA)

    for entry in timeline:
        beat = entry.get("cinematic_beat") or {}
        assert beat.get("subject_action") == "walks toward the doorway with careful purpose", (
            f"subject_action must survive assembly: got {beat.get('subject_action')!r}"
        )


def test_environment_usage_survives_assembly():
    """cinematic_beat.environment_usage must survive assembly unchanged."""
    engine = RhythmicAssemblyEngine()
    storyboard = [_make_shot(i) for i in range(4)]
    timeline = engine.assemble_timeline(storyboard, _AUDIO_DATA)

    for entry in timeline:
        beat = entry.get("cinematic_beat") or {}
        assert beat.get("environment_usage"), (
            f"environment_usage must survive assembly: got {beat.get('environment_usage')!r}"
        )


def test_shot_event_and_shot_type_survive_assembly():
    """shot_event and shot_type must both be present after assembly."""
    engine = RhythmicAssemblyEngine()
    storyboard = [_make_shot(i) for i in range(4)]
    timeline = engine.assemble_timeline(storyboard, _AUDIO_DATA)

    for entry in timeline:
        assert "shot_event" in entry, "shot_event must survive assembly"
        assert entry.get("shot_type") == "body_movement", (
            f"shot_type must survive assembly: got {entry.get('shot_type')!r}"
        )


def test_cinematic_beat_survives_style_grading():
    """After full assembly + style grading, cinematic_beat.subject_action and
    environment_usage must still be present — this is the MM3.1 stage 2
    end-to-end persistence guarantee."""
    asm_engine = RhythmicAssemblyEngine()
    sge = StyleGradingEngine()

    storyboard = [_make_shot(i) for i in range(4)]
    timeline = asm_engine.assemble_timeline(storyboard, _AUDIO_DATA)
    styled = sge.apply_style(timeline, {})

    assert len(styled) == 4
    for entry in styled:
        beat = entry.get("cinematic_beat") or {}
        assert beat.get("subject_action"), (
            "cinematic_beat.subject_action must survive style grading"
        )
        assert beat.get("environment_usage"), (
            "cinematic_beat.environment_usage must survive style grading"
        )
        assert entry.get("shot_type"), (
            "shot_type must survive style grading"
        )
