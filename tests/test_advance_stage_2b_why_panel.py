"""
Tests for advance_stage_2b — the WHY-panel dialogue submission route.

Route: POST /project/<id>/advance/2b
WHY action fields: why_action_inciting_cause, why_action_underlying_desire,
                   why_action_stakes, why_action_obstacle

Each why_action_{key} field accepts "accept" | "override" | "reject".
When a key is absent entirely the route skips that field and leaves whatever
lock already existed for it completely untouched.

Covered scenarios
-----------------
1. No why_action_* fields posted at all.
   - Route must still redirect successfully.
   - Pre-existing motivation values in the motivation block must survive.
   - Pre-existing motivation_* entries in locked_assumptions must survive.
   - confidence must not be bumped (no user action taken).

2. Only some why_action_* fields posted ("partial form post").
   - Absent keys leave their corresponding locks untouched.
   - Present "override" actions update motivation + locked_assumptions.
   - Present "accept" actions leave the existing lock value unchanged.
   - confidence is bumped when at least one change is recorded.
"""

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub psycopg.connect so importing app.py does not need a live DB.
# This follows the same pattern used by every other test in this suite.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-why-panel")
os.environ.setdefault("FLASK_ENV", "development")


class _StubCursor:
    def __init__(self):
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        pass

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _StubConn:
    def cursor(self, *a, **kw):
        return _StubCursor()

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


import psycopg  # noqa: E402

psycopg.connect = lambda *a, **kw: _StubConn()  # type: ignore[assignment]

import app as app_module  # noqa: E402

app_module.app.config["WTF_CSRF_ENABLED"] = False
app_module.app.config["TESTING"] = True


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_PRESET_MOTIVATION = {
    "inciting_cause":    "She lost her voice after the accident",
    "underlying_desire": "To be seen and heard again",
    "stakes":            "Her entire career",
    "obstacle":          "The label wants a different sound",
    "confidence":        0.6,
}

# Matching locked_assumptions entries for the same four motivation fields.
_PRESET_LOCKS = {
    "motivation_inciting_cause":    "She lost her voice after the accident",
    "motivation_underlying_desire": "To be seen and heard again",
    "motivation_stakes":            "Her entire career",
    "motivation_obstacle":          "The label wants a different sound",
}


def _make_project(motivation=None, locked_assumptions=None):
    """Return a minimal project dict suitable for patching into _get_project."""
    return {
        "id":      "proj-why-2b",
        "user_id": 99,
        "name":    "Why_Panel_Project",
        "stage":   "context_review",
        "status":  "awaiting_review",
        "context_packet": {
            "motivation": dict(
                motivation if motivation is not None else _PRESET_MOTIVATION
            ),
            "locked_assumptions": dict(
                locked_assumptions if locked_assumptions is not None else _PRESET_LOCKS
            ),
            "surfaced_assumptions": [],
            "confidence_scores": {"motivation": 0.6},
        },
        "styled_timeline": None,
    }


def _run_advance_2b(client, project, form_data, fake_user):
    """
    POST /project/<id>/advance/2b with all side-effects patched.

    Returns (saved_cp, response).

    saved_cp is the context_packet dict passed to the DB UPDATE, captured by
    intercepting app.Json before it is handed to psycopg.
    """
    captured: dict = {}
    _real_Json = app_module.Json

    def _capturing_Json(obj):
        captured["cp"] = dict(obj)
        return _real_Json(obj)

    patches = [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=fake_user),
        patch("auth.current_user", return_value=fake_user),
        patch.object(app_module, "db", lambda: _StubConn()),
        patch.object(app_module, "Json", side_effect=_capturing_Json),
    ]

    for p in patches:
        p.start()
    try:
        resp = client.post(
            f"/project/{project['id']}/advance/2b",
            data=form_data,
        )
    finally:
        for p in patches:
            p.stop()

    return captured.get("cp"), resp


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture
def fake_user():
    return {"id": 99, "email": "why@example.com"}


# ---------------------------------------------------------------------------
# Scenario 1 — No why_action_* fields submitted at all
# ---------------------------------------------------------------------------


def test_no_why_actions_redirect(client, fake_user):
    """Route must redirect even when no why_action_* fields are present."""
    project = _make_project()
    _, resp = _run_advance_2b(client, project, {}, fake_user)
    assert resp.status_code in (302, 303), (
        "advance_stage_2b must redirect on success even with no why_action_* fields"
    )


def test_no_why_actions_motivation_locks_survive(client, fake_user):
    """
    When no why_action_* keys are in the form the pre-existing motivation_*
    entries in locked_assumptions must be present and unchanged in the saved
    context_packet.
    """
    project = _make_project()
    saved_cp, _ = _run_advance_2b(client, project, {}, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    locked = saved_cp.get("locked_assumptions")
    assert locked is not None, "saved context_packet must contain locked_assumptions"

    for lock_key, expected_val in _PRESET_LOCKS.items():
        assert locked.get(lock_key) == expected_val, (
            f"locked_assumptions[{lock_key!r}] must survive unchanged when its "
            f"why_action field is absent; expected {expected_val!r}, "
            f"got {locked.get(lock_key)!r}"
        )


def test_no_why_actions_motivation_values_survive(client, fake_user):
    """
    With no why_action_* keys the motivation block in the saved context_packet
    must carry the original motivation values verbatim.
    """
    project = _make_project()
    saved_cp, _ = _run_advance_2b(client, project, {}, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    saved_motivation = saved_cp.get("motivation")
    assert saved_motivation is not None, "saved context_packet must contain a motivation block"

    for key in ("inciting_cause", "underlying_desire", "stakes", "obstacle"):
        assert saved_motivation.get(key) == _PRESET_MOTIVATION[key], (
            f"motivation[{key!r}] must not change when why_action_{key} is absent"
        )


def test_no_why_actions_confidence_not_bumped(client, fake_user):
    """
    Without any why_action_* submissions that cause a change, confidence must
    remain at its original value (not bumped to 0.9).
    """
    project = _make_project(locked_assumptions={})
    saved_cp, _ = _run_advance_2b(client, project, {}, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    saved_motivation = saved_cp.get("motivation", {})
    confidence = float(saved_motivation.get("confidence", 0.6))
    assert confidence < 0.9, (
        f"confidence must not be bumped when no why_action_* fields are submitted; "
        f"got {confidence}"
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Only some why_action_* fields submitted ("partial form post")
# ---------------------------------------------------------------------------


def test_partial_why_actions_absent_locks_survive(client, fake_user):
    """
    When only why_action_inciting_cause and why_action_stakes are posted the
    other two motivation locks (underlying_desire, obstacle) must remain in
    locked_assumptions with their original values.
    """
    project = _make_project()
    form_data = {
        "why_action_inciting_cause": "override",
        "motivation_inciting_cause": "She walked away from everything",
        "why_action_stakes":         "accept",
    }
    saved_cp, resp = _run_advance_2b(client, project, form_data, fake_user)

    assert resp.status_code in (302, 303), "advance_stage_2b must redirect with partial actions"
    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    locked = saved_cp.get("locked_assumptions", {})

    assert locked.get("motivation_underlying_desire") == _PRESET_LOCKS["motivation_underlying_desire"], (
        "motivation_underlying_desire lock must survive because why_action_underlying_desire was absent"
    )
    assert locked.get("motivation_obstacle") == _PRESET_LOCKS["motivation_obstacle"], (
        "motivation_obstacle lock must survive because why_action_obstacle was absent"
    )


def test_partial_why_actions_override_updates_lock(client, fake_user):
    """
    A posted why_action_{key}='override' with a non-empty motivation_{key}
    must update both locked_assumptions and the motivation block.
    """
    project = _make_project()
    form_data = {
        "why_action_inciting_cause": "override",
        "motivation_inciting_cause": "She walked away from everything",
    }
    saved_cp, _ = _run_advance_2b(client, project, form_data, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    locked = saved_cp.get("locked_assumptions", {})
    saved_motivation = saved_cp.get("motivation", {})

    assert locked.get("motivation_inciting_cause") == "She walked away from everything", (
        "locked_assumptions must be updated for an 'override' action"
    )
    assert saved_motivation.get("inciting_cause") == "She walked away from everything", (
        "motivation block must be updated for an 'override' action"
    )


def test_partial_why_actions_accept_leaves_lock_unchanged(client, fake_user):
    """
    A posted why_action_{key}='accept' must leave the existing locked value
    unchanged (accept means the user confirmed the current value as correct).
    """
    project = _make_project()
    form_data = {
        "why_action_stakes": "accept",
    }
    saved_cp, _ = _run_advance_2b(client, project, form_data, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    locked = saved_cp.get("locked_assumptions", {})
    assert locked.get("motivation_stakes") == _PRESET_LOCKS["motivation_stakes"], (
        "motivation_stakes lock must be unchanged for an 'accept' action"
    )


def test_partial_why_actions_confidence_bumped_on_override(client, fake_user):
    """
    A successful 'override' action must bump confidence to at least 0.9.
    """
    project = _make_project(locked_assumptions={})
    form_data = {
        "why_action_obstacle": "override",
        "motivation_obstacle": "The industry machine",
    }
    saved_cp, _ = _run_advance_2b(client, project, form_data, fake_user)

    assert saved_cp is not None, (
        "Json() must be called once — context_packet was not saved to the DB"
    )
    saved_motivation = saved_cp.get("motivation", {})
    confidence = float(saved_motivation.get("confidence", 0))
    assert confidence >= 0.9, (
        f"confidence must be bumped to at least 0.9 after an 'override' action; got {confidence}"
    )
