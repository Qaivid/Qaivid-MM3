"""Tests for the deterministic Cinematography Engine (Task #69)."""
import pytest

from cinematography_engine import (
    EMOTION_TO_RIG,
    RIGS,
    derive,
    motion_prompt_from_block,
    lens_clause,
)


_BRIEF_CTX = {"creative_brief": {"chosen": {"director_note": ""}}}


def _ctx(extra=None):
    """Return a context dict with a locked Creative Brief (required for derive)."""
    base = {"creative_brief": {"chosen": {"director_note": ""}}}
    if extra:
        base.update(extra)
    return base


def test_returns_none_for_legacy_shot_without_expression_mode():
    assert derive({}, _ctx(), {}) is None
    assert derive({"expression_mode": None}, _ctx(), {}) is None
    assert derive({"expression_mode": "unknown_mode"}, _ctx(), {}) is None


def test_returns_none_when_no_creative_brief_locked():
    """Backward-compat: legacy projects without a locked brief get no rig."""
    shot = {"expression_mode": "face", "intensity": 0.6, "meaning": "longing"}
    assert derive(shot, {}, {}) is None
    assert derive(shot, {"creative_brief": {}}, {}) is None
    assert derive(shot, {"creative_brief": {"chosen": {}}}, {}) is None


def test_block_shape_has_canonical_fields():
    block = derive(
        {"expression_mode": "face", "intensity": 0.6, "meaning": "longing"},
        _ctx({"speaker": {"emotional_state": "longing"}}),
        {"cinematic": {"id": "cinematic_natural"}, "preset": "cinematic_natural"},
    )
    assert block is not None
    for key in ("rig", "direction", "speed", "lens", "intensity", "justification"):
        assert key in block, f"missing key {key}"
    assert block["rig"] in RIGS


def test_determinism_same_inputs_same_output():
    shot = {"expression_mode": "body", "intensity": 0.7, "meaning": "kinetic chase"}
    ctx = {"speaker": {"emotional_state": "panic"}}
    sp = {"cinematic": {"id": "vibrant_bold"}}
    assert derive(shot, ctx, sp) == derive(shot, ctx, sp)


@pytest.mark.parametrize("emotion", list(EMOTION_TO_RIG.keys()))
def test_emotion_to_rig_coverage_returns_a_canonical_rig(emotion):
    block = derive(
        {"expression_mode": "environment", "intensity": 0.5, "meaning": emotion},
        _ctx({"speaker": {"emotional_state": emotion}}),
        {},
    )
    assert block is not None
    assert block["rig"] in RIGS


def test_director_note_boost_overrides_default():
    # With "tarkovsky / locked" director note we expect tripod even on a
    # body shot that would otherwise prefer steadicam/gimbal.
    shot = {"expression_mode": "body", "intensity": 0.5, "meaning": "calm walk"}
    ctx = {
        "speaker": {"emotional_state": "calm"},
        "creative_brief": {"chosen": {"director_note": "Tarkovsky locked frames"}},
    }
    block = derive(shot, ctx, {})
    assert block["rig"] == "tripod"


def test_motion_prompt_short_and_well_formed():
    block = {"rig": "dolly", "direction": "slow push-in",
             "speed": "slow", "lens": "85mm portrait, shallow depth of field",
             "intensity": "low", "justification": "x"}
    out = motion_prompt_from_block(block)
    assert "dolly" in out and "push-in" in out and "85mm" in out
    assert len(out) <= 240

    assert motion_prompt_from_block(None) == ""
    assert motion_prompt_from_block({}) == ""


def test_lens_clause_handles_none():
    assert lens_clause(None) == ""
    assert "Cinematography" in lens_clause(
        {"rig": "dolly", "direction": "slow push-in", "lens": "85mm"}
    )


def test_intensity_label_thresholds():
    low = derive({"expression_mode": "face", "intensity": 0.1, "meaning": "calm"}, _ctx(), {})
    med = derive({"expression_mode": "face", "intensity": 0.6, "meaning": "calm"}, _ctx(), {})
    high = derive({"expression_mode": "face", "intensity": 0.9, "meaning": "calm"}, _ctx(), {})
    assert low["intensity"] == "low"
    assert med["intensity"] == "medium"
    assert high["intensity"] == "high"
