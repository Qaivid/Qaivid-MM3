"""Tests for the deterministic Cinematography Engine (Task #69 and #79)."""
import pytest

from cinematography_engine import (
    ACTION_TO_RIG,
    CAMERA_PLAN_TO_RIG,
    EMOTION_TO_RIG,
    RIGS,
    _DEFAULT_RANK,
    _emotion_rank,
    _event_rank,
    _mode_rank,
    _pick_rig,
    _style_rank,
    derive,
    lens_clause,
    motion_prompt_from_block,
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


# ---------------------------------------------------------------------------
# _pick_rig justification-branch tests (Task #79)
# ---------------------------------------------------------------------------

_INTERNAL_TAGS = (
    "emotion_top", "style_top", "mode_top", "event_top",
    "emotion match", "style match", "mode match",
    "matched_emotion", "matched_plan", "matched_action",
)


def _assert_justification_quality(justification: str) -> None:
    """Re-usable quality gate: non-empty, ends with punctuation, no raw tags."""
    assert justification, "justification must not be empty"
    assert justification[-1] in ".!?", (
        f"justification must end with punctuation, got: {justification!r}"
    )
    for tag in _INTERNAL_TAGS:
        assert tag not in justification, (
            f"raw internal tag {tag!r} leaked into justification: {justification!r}"
        )


def _default_ranks():
    """Return four default rank lists (no signal biasing)."""
    dr = list(_DEFAULT_RANK)
    return dr, dr, dr, dr


def test_pick_rig_director_note_branch():
    """Branch 1: director note keyword fires director justification sentence.

    The director note adds a +6 boost rather than a hard override, so we use
    ranks that already favour the boosted rig (dolly) and only verify that the
    justification text mentions the director's note.
    """
    ev, em, st, mo = _default_ranks()
    rig, justification = _pick_rig(
        ev, em, st, mo,
        director_note="dolly push-in intimate slow",
    )
    assert rig == "dolly"
    assert "director" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_camera_plan_branch():
    """Branch 2: matched camera-plan keyword + event at top → camera-plan sentence."""
    event_signal = "locked"
    ev = _event_rank(event_signal)
    em = list(_DEFAULT_RANK)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    rig, justification = _pick_rig(
        ev, em, st, mo,
        event_signal=event_signal,
    )
    assert rig == "tripod"
    assert "camera" in justification.lower() or "behaviour" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_action_branch():
    """Branch 3: matched action keyword + event at top → action sentence."""
    event_signal = "walk"
    ev = _event_rank(event_signal)
    em = list(_DEFAULT_RANK)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    rig, justification = _pick_rig(
        ev, em, st, mo,
        event_signal=event_signal,
    )
    assert rig == "steadicam"
    assert "action" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_emotion_branch():
    """Branch 4: emotion keyword present + emotion rank wins → emotion sentence."""
    emotion_signal = "nostalg"
    ev = list(_DEFAULT_RANK)
    em = _emotion_rank(emotion_signal)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    rig, justification = _pick_rig(
        ev, em, st, mo,
        emotion_signal=emotion_signal,
    )
    assert rig == "dolly"
    assert "nostalgic" in justification.lower()
    assert "tone" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_style_branch():
    """Branch 5: no emotion signal, style rank tops → style sentence."""
    gimbal_first = ["gimbal", "steadicam", "dolly", "drone", "crane", "handheld", "vehicle", "tripod"]
    ev = list(_DEFAULT_RANK)
    em = gimbal_first
    st = gimbal_first
    mo = list(_DEFAULT_RANK)
    rig, justification = _pick_rig(
        ev, em, st, mo,
        emotion_signal="",
    )
    assert rig == "gimbal"
    assert "style" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_mode_branch():
    """Branch 6: no emotion signal, style doesn't top, mode rank wins → mode sentence."""
    crane_first_em = ["crane", "drone", "gimbal", "handheld", "vehicle", "steadicam", "tripod", "dolly"]
    ev = list(_DEFAULT_RANK)
    em = crane_first_em
    st = ["dolly", "steadicam", "tripod"]
    mo = ["crane", "drone", "dolly"]
    rig, justification = _pick_rig(
        ev, em, st, mo,
        emotion_signal="",
        mode="environment",
    )
    assert rig == "crane"
    assert "environment" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_default_branch():
    """Branch 7: no signal matches any ranked top → balanced default sentence."""
    ev = list(_DEFAULT_RANK)
    em = list(_DEFAULT_RANK)
    st = ["crane", "drone", "gimbal"]
    mo = ["handheld", "gimbal", "steadicam"]
    rig, justification = _pick_rig(
        ev, em, st, mo,
        emotion_signal="",
    )
    assert rig == "dolly"
    assert "balanced" in justification.lower()
    _assert_justification_quality(justification)


def test_pick_rig_no_emotion_signal_never_produces_empty_justification():
    """Edge case: completely empty emotion signal still yields a readable sentence."""
    ev = list(_DEFAULT_RANK)
    em = list(_DEFAULT_RANK)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    _, justification = _pick_rig(ev, em, st, mo, emotion_signal="")
    _assert_justification_quality(justification)


def test_pick_rig_unknown_emotion_falls_back_gracefully():
    """Edge case: an unrecognised emotion keyword must still produce a clean sentence."""
    ev = list(_DEFAULT_RANK)
    em = _emotion_rank("xyzunknownemotion")
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    _, justification = _pick_rig(
        ev, em, st, mo,
        emotion_signal="xyzunknownemotion",
    )
    _assert_justification_quality(justification)
    assert "xyzunknownemotion" not in justification


def test_pick_rig_director_note_keyword_variants():
    """Edge case: every director-note keyword family triggers the director branch."""
    keyword_families = [
        "aerial drone",
        "crane epic majestic",
        "locked tarkovsky static",
        "music video fluid kinetic",
        "floating dreamlike smooth",
        "car chase tracking vehicle",
        "push-in dolly intimate",
    ]
    ev, em, st, mo = _default_ranks()
    for note in keyword_families:
        _, justification = _pick_rig(ev, em, st, mo, director_note=note)
        assert "director" in justification.lower(), (
            f"director note {note!r} did not fire director branch; got: {justification!r}"
        )
        _assert_justification_quality(justification)


@pytest.mark.parametrize("emotion_kw", list(EMOTION_TO_RIG.keys()))
def test_pick_rig_every_mapped_emotion_produces_clean_justification(emotion_kw):
    """Every EMOTION_TO_RIG keyword should produce a non-empty, punctuated sentence."""
    ev = list(_DEFAULT_RANK)
    em = _emotion_rank(emotion_kw)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    _, justification = _pick_rig(ev, em, st, mo, emotion_signal=emotion_kw)
    _assert_justification_quality(justification)


@pytest.mark.parametrize("action_kw", list(ACTION_TO_RIG.keys()))
def test_pick_rig_every_action_keyword_produces_clean_justification(action_kw):
    """Every ACTION_TO_RIG keyword should produce a non-empty, punctuated sentence."""
    ev = _event_rank(action_kw)
    em = list(_DEFAULT_RANK)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    _, justification = _pick_rig(ev, em, st, mo, event_signal=action_kw)
    _assert_justification_quality(justification)


@pytest.mark.parametrize("plan_kw", list(CAMERA_PLAN_TO_RIG.keys()))
def test_pick_rig_every_camera_plan_keyword_produces_clean_justification(plan_kw):
    """Every CAMERA_PLAN_TO_RIG keyword should produce a non-empty, punctuated sentence."""
    ev = _event_rank(plan_kw)
    em = list(_DEFAULT_RANK)
    st = list(_DEFAULT_RANK)
    mo = list(_DEFAULT_RANK)
    _, justification = _pick_rig(ev, em, st, mo, event_signal=plan_kw)
    _assert_justification_quality(justification)


def test_justification_via_derive_is_always_present_and_readable():
    """Integration: every derive() call returns a non-empty, punctuated justification."""
    cases = [
        ({"expression_mode": "face", "intensity": 0.5, "meaning": "calm"}, _ctx(), {}),
        ({"expression_mode": "body", "intensity": 0.8, "meaning": "kinetic"}, _ctx(), {}),
        ({"expression_mode": "environment", "intensity": 0.3, "meaning": "awe"}, _ctx(), {}),
        (
            {"expression_mode": "face", "intensity": 0.6, "meaning": "grief"},
            _ctx({"speaker": {"emotional_state": "grief"}}),
            {"cinematic": {"id": "soft_poetic"}},
        ),
        (
            {"expression_mode": "body", "intensity": 0.5, "meaning": "calm walk"},
            {"creative_brief": {"chosen": {"director_note": "Tarkovsky locked frames"}}},
            {},
        ),
    ]
    for shot, ctx, sp in cases:
        block = derive(shot, ctx, sp)
        assert block is not None
        _assert_justification_quality(block["justification"])
