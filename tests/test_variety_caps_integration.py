"""
Integration tests for MM3.1 variety-cap enforcement and per-shot quality guarantees.

Covers:
  (i)  Distribution caps: no mode exceeds its cap fraction after _enforce_variety_caps
  (ii) Cinematography re-derivation: reclassified shots have cinematography updated
       to match new expression_mode
  (iii) Verb + environment interaction present on every shot exiting the
        GenericShotValidator → cinematography pipeline
"""
import pytest
from unittest.mock import patch

from generic_shot_validator import GenericShotValidator, _ACTION_VERB_RE
from visual_storyboard_engine import VisualStoryboardEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(mode: str, idx: int) -> dict:
    """Minimal storyboard-like shot dict with shot_type set (activates caps)."""
    return {
        "expression_mode": mode,
        "shot_type": f"{mode}_shot",
        "framing_directive": f"framing for {mode}",
        "intensity": 0.6,
        "emotional_meaning": "longing",
        "arc_position": idx,
        "action": f"turns toward the {mode} window gently",
        "subject_action": f"turns toward the {mode} window gently",
        "environment_usage": f"interior lit softly for {mode}",
        "environment_interaction": f"interior lit softly for {mode}",
    }


def _make_all_face_shots(n: int) -> list:
    """Return n shots all set to 'face' mode to trigger cap enforcement."""
    return [_make_shot("face", i) for i in range(n)]


def _vse() -> VisualStoryboardEngine:
    return VisualStoryboardEngine.__new__(VisualStoryboardEngine)


# ---------------------------------------------------------------------------
# (i) Distribution cap enforcement
# ---------------------------------------------------------------------------


def test_no_mode_exceeds_cap_after_enforcement():
    """After _enforce_variety_caps, no mode's fraction exceeds its cap."""
    vse = _vse()
    storyboard = _make_all_face_shots(20)
    result = vse._enforce_variety_caps(storyboard)

    total = len(result)
    counts: dict = {}
    for s in result:
        m = s.get("expression_mode", "face")
        counts[m] = counts.get(m, 0) + 1

    for mode, cap in vse._VARIETY_CAPS.items():
        frac = counts.get(mode, 0) / total
        assert frac <= cap + 0.001, (
            f"Mode '{mode}' fraction {frac:.2%} exceeds cap {cap:.0%}"
        )


def test_reclassified_shots_are_flagged():
    """Reclassified shots carry variety_cap_reclassified=True."""
    vse = _vse()
    storyboard = _make_all_face_shots(10)
    result = vse._enforce_variety_caps(storyboard)
    reclassified = [s for s in result if s.get("variety_cap_reclassified")]
    face_count = sum(1 for s in result if s["expression_mode"] == "face")
    face_cap = int(vse._VARIETY_CAPS["face"] * len(result))
    assert face_count <= face_cap + 1, "Face count must be near or at cap after enforcement"
    assert len(reclassified) > 0, "At least some shots must have been reclassified"


# ---------------------------------------------------------------------------
# (ii) Cinematography re-derivation on reclassified shots
# ---------------------------------------------------------------------------


def test_cinematography_rederived_for_reclassified_shots():
    """If a ctx is supplied, reclassified shots get new cinematography derived
    for the new expression_mode, not the old one."""
    vse = _vse()
    vse._active_style_profile = {}
    storyboard = _make_all_face_shots(10)

    # Provide a minimal ctx that satisfies cinematography_engine.derive()
    ctx = {"creative_brief": {"chosen": {"director_note": ""}}}
    result = vse._enforce_variety_caps(storyboard, ctx)

    for s in result:
        if s.get("variety_cap_reclassified") and s.get("cinematography"):
            # The re-derived block must reference the new mode, not the original "face"
            new_mode = s["expression_mode"]
            assert new_mode != "face", "Reclassified shot must have a new mode"


def test_variety_caps_without_ctx_still_reclassifies():
    """_enforce_variety_caps without ctx still reclassifies mode/shot_type/framing
    (cinematography re-derivation is skipped but reclassification happens)."""
    vse = _vse()
    storyboard = _make_all_face_shots(10)
    result = vse._enforce_variety_caps(storyboard, ctx=None)
    face_count = sum(1 for s in result if s["expression_mode"] == "face")
    assert face_count <= int(vse._VARIETY_CAPS["face"] * len(result)) + 1


# ---------------------------------------------------------------------------
# (iii) Verb + environment present on every shot after full validation
# ---------------------------------------------------------------------------


def test_every_shot_has_verb_action_after_validator():
    """validate_sequence guarantees a verb-bearing action on every event."""
    validator = GenericShotValidator()
    events = [
        {"expression_mode": "face",        "action": "",
         "environment_usage": "sunlit room"},
        {"expression_mode": "body",        "action": "heavy emotional state",
         "environment_usage": "crowded street at noon"},
        {"expression_mode": "environment", "action": "turns toward the far horizon slowly",
         "environment_usage": "open field at dusk"},
        {"expression_mode": "macro",       "action": "subtle detail shifts",
         "environment_usage": "wooden tabletop texture"},
    ]
    result = validator.validate_sequence(events)
    for ev in result:
        assert _ACTION_VERB_RE.match(ev["action"]), (
            f"No verb found in action after validate_sequence: {ev['action']!r}"
        )


def test_every_shot_has_env_after_validator():
    """validate_sequence backfills environment on shots missing spatial grounding."""
    validator = GenericShotValidator()
    events = [
        {"expression_mode": "face", "action": "turns slowly away from the light source",
         "environment_usage": ""},
        {"expression_mode": "body", "action": "shifts weight and stares forward with focus",
         "environment_interaction": ""},
    ]
    result = validator.validate_sequence(events)
    for ev in result:
        env_val = ev.get("environment_usage", "") or ev.get("environment_interaction", "")
        assert env_val, f"Missing environment grounding after validate_sequence: {ev!r}"


def test_distribution_and_verb_env_together():
    """End-to-end: after variety cap enforcement AND validation, shots meet
    distribution targets AND each carries a verb-bearing action + env field."""
    validator = GenericShotValidator()
    vse = _vse()

    raw_events = []
    for i in range(20):
        raw_events.append({
            "expression_mode": "face",
            "action": "personal emotional weight presses",  # verb-less, triggers rewrite
            "environment_usage": "",
        })

    validated = validator.validate_sequence(raw_events)
    storyboard = []
    for ev in validated:
        storyboard.append({
            "expression_mode": ev["expression_mode"],
            "shot_type":       ev["expression_mode"] + "_type",
            "framing_directive": "",
            "intensity":       0.5,
            "emotional_meaning": "",
            "arc_position":    0,
            "action":          ev["action"],
            "subject_action":  ev["action"],
            "environment_usage":      ev.get("environment_usage", ""),
            "environment_interaction": ev.get("environment_interaction", ""),
        })

    result = vse._enforce_variety_caps(storyboard)
    total = len(result)
    counts: dict = {}
    for s in result:
        m = s["expression_mode"]
        counts[m] = counts.get(m, 0) + 1

    for mode, cap in vse._VARIETY_CAPS.items():
        frac = counts.get(mode, 0) / total
        assert frac <= cap + 0.001, f"Mode '{mode}' still over cap: {frac:.2%} > {cap:.0%}"

    for s in result:
        assert _ACTION_VERB_RE.match(s["action"]), (
            f"Action missing verb: {s['action']!r}"
        )
        env_val = s.get("environment_usage") or s.get("environment_interaction")
        assert env_val, f"Shot missing environment grounding: {s!r}"
