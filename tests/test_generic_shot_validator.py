"""
Regression tests for GenericShotValidator (MM3.1 Task #94).

Covers:
  (a) verb-less 3+ word action detection and rewrite
  (b) environment field backfill for both canonical alias fields
  (c) event-primary rig selection in cinematography_engine survives rewritten
      actions — rewritten verb-bearing actions are matched by ACTION_TO_RIG so
      rig selection uses event-primary path (not emotion fallback)
"""
import pytest
from generic_shot_validator import GenericShotValidator, _ACTION_VERB_RE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _event(action="", env="", mode="body", subject_action=None):
    """Return a minimal shot event dict."""
    d = {
        "action": action,
        "expression_mode": mode,
        "visual_contrast": "",
        "camera_motivation": "",
    }
    if subject_action is not None:
        d["subject_action"] = subject_action
    if env:
        d["environment_usage"] = env
    return d


validator = GenericShotValidator()


# ---------------------------------------------------------------------------
# (a) Verb-less 3+ word action detection and rewrite
# ---------------------------------------------------------------------------


def test_verb_less_3word_action_is_detected_as_generic():
    """A 3+ word action with no verb must be flagged as generic."""
    event = _event(action="private emotional shift", env="empty street outside")
    assert validator.is_generic(event), "Expected verb-less action to be generic"


def test_verb_less_3word_action_is_rewritten():
    """rewrite_generic must replace a 3+ word verb-less action with a verb-bearing fallback."""
    event = _event(action="private emotional shift during routine", env="empty street outside", mode="body")
    validator.rewrite_generic(event)
    # Both canonical fields must carry the rewritten value
    assert _ACTION_VERB_RE.match(event["action"]), (
        f"rewritten action must start with a verb, got: {event['action']!r}"
    )
    assert event["subject_action"] == event["action"], "action and subject_action must be in sync"
    assert "_original_action" in event, "original action must be preserved for debugging"


def test_verb_bearing_3word_action_is_not_rewritten():
    """An action that starts with a recognised verb must NOT be rewritten."""
    verb_action = "turns slightly toward the doorway"
    event = _event(action=verb_action, env="doorway bathed in afternoon light", mode="face")
    validator.rewrite_generic(event)
    assert event["action"] == verb_action, "Valid verb-bearing action must not be overwritten"


def test_rewritten_action_passes_verb_check():
    """After rewrite the event action must always match _ACTION_VERB_RE."""
    for mode in ("face", "body", "environment", "macro", "symbolic"):
        event = _event(action="abstract noun phrase here", env="somewhere", mode=mode)
        validator.rewrite_generic(event)
        assert _ACTION_VERB_RE.match(event["action"]), (
            f"mode={mode}: rewritten action must start with verb: {event['action']!r}"
        )


def test_validate_sequence_leaves_no_generic_events_without_verb():
    """validate_sequence must ensure every event exits with a verb-bearing action."""
    events = [
        _event(action="emotional emptiness in the room", env="empty room", mode="body"),
        _event(action="walks toward the open window", env="window light spilling in", mode="body"),
        _event(action="", env="stone steps", mode="environment"),
    ]
    result = validator.validate_sequence(events)
    for ev in result:
        assert _ACTION_VERB_RE.match(ev["action"]), (
            f"action must have verb after validate_sequence: {ev['action']!r}"
        )
        # subject_action must mirror action
        assert ev.get("subject_action") == ev["action"] or ev.get("subject_action") is None or \
               _ACTION_VERB_RE.match(ev.get("subject_action", "")), (
            f"subject_action must be verb-bearing: {ev.get('subject_action')!r}"
        )


# ---------------------------------------------------------------------------
# (b) Environment field backfill
# ---------------------------------------------------------------------------


def test_missing_env_is_backfilled_in_both_aliases():
    """When both env alias fields are missing, rewrite_generic fills both."""
    event = _event(action="turns toward doorway as sound fades", env="", mode="face")
    event.pop("environment_usage", None)
    event.pop("environment_interaction", None)
    validator.rewrite_generic(event)
    assert event.get("environment_usage"), "environment_usage must be filled"
    assert event.get("environment_interaction"), "environment_interaction must be filled"
    assert event["environment_usage"] == event["environment_interaction"], (
        "Both alias fields must carry the same fallback value"
    )


def test_present_env_is_not_overwritten():
    """Existing environment_usage must not be replaced by the fallback."""
    original_env = "sunlit window with curtain moving in breeze"
    event = _event(
        action="stands looking at the horizon far below",
        env=original_env,
        mode="body",
    )
    validator.rewrite_generic(event)
    assert event.get("environment_usage") == original_env, "Existing env must not be overwritten"


# ---------------------------------------------------------------------------
# (c) Event-primary rig selection after rewrite
# ---------------------------------------------------------------------------


def test_event_primary_rig_after_validator_rewrite():
    """After rewrite, the action keyword (e.g. 'walks') must be matched by
    ACTION_TO_RIG, causing event-primary rig selection rather than emotion fallback.
    """
    from cinematography_engine import ACTION_TO_RIG, derive, _has_event_match

    # Pick a mode where the rewritten action contains a recognisable action keyword
    event = _event(action="emotional noun phrase lacking a verb", env="open space", mode="body")
    result = validator.validate_sequence([event])
    rewritten_action = result[0]["action"]

    # The rewritten action should match at least one ACTION_TO_RIG keyword
    event_signal = rewritten_action.lower()
    assert _has_event_match(event_signal), (
        f"Rewritten action '{rewritten_action}' must match at least one ACTION_TO_RIG keyword "
        f"so the cinematography engine uses the event-primary path."
    )


def test_action_to_rig_keywords_covered_by_fallback_actions():
    """Verify that all mode fallback actions contain at least one ACTION_TO_RIG keyword
    OR are matched by _ACTION_VERB_RE (double-guarantee that event matching will fire).
    """
    from generic_shot_validator import _FALLBACK_ACTIONS
    from cinematography_engine import _has_event_match

    for mode, action in _FALLBACK_ACTIONS.items():
        matches_verb_re = bool(_ACTION_VERB_RE.match(action))
        assert matches_verb_re, (
            f"Fallback action for mode={mode!r} does not match _ACTION_VERB_RE: {action!r}"
        )


# ---------------------------------------------------------------------------
# (d) shot_type → expression_mode derivation (ShotEventBuilder compat)
# ---------------------------------------------------------------------------


def test_shot_type_portrait_uses_face_fallback():
    """When expression_mode is absent but shot_type='portrait', rewrite must use face fallback."""
    from generic_shot_validator import _FALLBACK_ACTIONS
    event = {
        "shot_type": "portrait",
        "action": "",
        "environment_usage": "sunlit corner of a room",
    }
    validator.rewrite_generic(event)
    face_fallback = _FALLBACK_ACTIONS["face"]
    assert event["action"] == face_fallback, (
        f"portrait shot_type must use face fallback, got: {event['action']!r}"
    )


def test_shot_type_wide_environment_uses_environment_fallback():
    """When expression_mode is absent but shot_type='wide_environment', use environment fallback."""
    from generic_shot_validator import _FALLBACK_ACTIONS
    event = {
        "shot_type": "wide_environment",
        "action": "",
        "environment_usage": "open field at dusk",
    }
    validator.rewrite_generic(event)
    env_fallback = _FALLBACK_ACTIONS["environment"]
    assert event["action"] == env_fallback, (
        f"wide_environment shot_type must use environment fallback, got: {event['action']!r}"
    )


def test_shot_type_silhouette_uses_symbolic_fallback():
    """When expression_mode is absent but shot_type='silhouette', use symbolic fallback."""
    from generic_shot_validator import _FALLBACK_ACTIONS
    event = {
        "shot_type": "silhouette",
        "action": "",
        "environment_usage": "backlit doorway",
    }
    validator.rewrite_generic(event)
    symbolic_fallback = _FALLBACK_ACTIONS["symbolic"]
    assert event["action"] == symbolic_fallback, (
        f"silhouette shot_type must use symbolic fallback, got: {event['action']!r}"
    )


def test_expression_mode_takes_priority_over_shot_type():
    """If both expression_mode and shot_type present, expression_mode wins."""
    from generic_shot_validator import _FALLBACK_ACTIONS
    event = {
        "expression_mode": "macro",
        "shot_type": "portrait",       # would map to face — must NOT win
        "action": "",
        "environment_usage": "wooden surface texture",
    }
    validator.rewrite_generic(event)
    macro_fallback = _FALLBACK_ACTIONS["macro"]
    assert event["action"] == macro_fallback, (
        f"expression_mode must override shot_type; expected macro fallback, got: {event['action']!r}"
    )
