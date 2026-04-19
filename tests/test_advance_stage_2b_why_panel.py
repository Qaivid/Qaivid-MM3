"""
Tests for the WHY — Motivation panel in the JARVIS Dialogue (Task #76).

Covers POST /project/<id>/advance/2b and verifies the four behaviours
introduced by Task #68:

1. override  → writes motivation_<field> into locked_assumptions AND mirrors
               the value into context_packet.motivation; confidence >= 0.9.
2. reject    → removes any existing lock for that field and appends an entry
               to ambiguity_flags.
3. accept    → takes the AI-engine value and locks it; confidence >= 0.9.
4. motivation_* entries in surfaced_assumptions are NOT double-processed by
               the generic action_<i> loop.

The Flask app and its psycopg dependency are patched at import time so the
suite can run without a live database.
"""

import os
import sys
import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub psycopg.connect so importing app.py doesn't need a real DB.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-why")
os.environ.setdefault("FLASK_ENV", "development")


class _StubCursor:
    def __init__(self):
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        return None

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
# Recording fakes — capture every SQL call so we can assert on the saved
# context_packet without a live database.
# ---------------------------------------------------------------------------


class RecordingCursor:
    """Cursor that records every execute() call and always reports rowcount=1."""

    def __init__(self, sink):
        self.sink = sink
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sink.append((sql, params))


class RecordingConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self, *a, **kw):
        return RecordingCursor(self.sink)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHY_FIELDS = ("inciting_cause", "underlying_desire", "stakes", "obstacle")


def _make_project(extra_cp=None):
    """Return a project dict in the state expected by advance_stage_2b."""
    cp = {
        "motivation": {
            "inciting_cause": "engine-cause",
            "underlying_desire": "engine-desire",
            "stakes": "engine-stakes",
            "obstacle": "engine-obstacle",
            "confidence": 0.5,
        },
        "locked_assumptions": {},
        "ambiguity_flags": [],
        "surfaced_assumptions": [],
        "confidence_scores": {"motivation": 0.5},
        "world_assumptions": {},
        "speaker": {},
    }
    if extra_cp:
        cp.update(extra_cp)
    return {
        "id": "proj-why",
        "user_id": 7,
        "stage": "assumptions_review",
        "status": "awaiting_review",
        "context_packet": cp,
    }


def _patch_request(project, sink, user):
    """Return the patches needed for a single advance_stage_2b request."""
    return [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=user),
        patch("auth.current_user", return_value=user),
        patch.object(app_module, "db", lambda: RecordingConn(sink)),
        patch.object(app_module, "kick_stage_brief", return_value=None),
    ]


def _run_with_patches(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in patches:
            p.stop()


def _extract_saved_cp(sink):
    """Pull the context_packet out of the recorded UPDATE statement."""
    # The UPDATE params are: (Json(cp), "queued", "queued", project_id)
    assert sink, "No SQL was recorded — did the route abort early?"
    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects SET context_packet" in sql]
    assert update_calls, f"No context_packet UPDATE found. Recorded calls: {sink}"
    _, params = update_calls[0]
    raw = params[0]
    # psycopg.types.json.Json wraps the dict in a .obj attribute
    return raw.obj if hasattr(raw, "obj") else raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture
def fake_user():
    return {"id": 7, "email": "director@test.com"}


# ---------------------------------------------------------------------------
# Test 1: override writes lock and mirrors into motivation, confidence >= 0.9
# ---------------------------------------------------------------------------


def test_override_writes_lock_and_mirrors_into_motivation(client, fake_user):
    project = _make_project()
    sink = []
    patches = _patch_request(project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-why/advance/2b",
            data={
                "why_action_inciting_cause": "override",
                "why_value_inciting_cause": "she lost her voice",
                "why_action_underlying_desire": "accept",
                "why_value_underlying_desire": "",
                "why_action_stakes": "accept",
                "why_value_stakes": "",
                "why_action_obstacle": "accept",
                "why_value_obstacle": "",
            },
        )

    resp = _run_with_patches(patches, go)
    # Route redirects on success
    assert resp.status_code in (302, 303)

    cp = _extract_saved_cp(sink)

    # Lock was written
    assert cp["locked_assumptions"]["motivation_inciting_cause"] == "she lost her voice"

    # Mirrored into motivation block
    assert cp["motivation"]["inciting_cause"] == "she lost her voice"

    # Confidence bumped to at least 0.9
    assert cp["motivation"]["confidence"] >= 0.9
    assert cp["confidence_scores"]["motivation"] >= 0.9


# ---------------------------------------------------------------------------
# Test 2: reject clears the lock and appends an ambiguity_flag
# ---------------------------------------------------------------------------


def test_reject_clears_lock_and_appends_ambiguity_flag(client, fake_user):
    # Pre-seed a lock for the stakes field to confirm it gets cleared.
    project = _make_project(
        extra_cp={"locked_assumptions": {"motivation_stakes": "old-stakes-value"}}
    )
    sink = []
    patches = _patch_request(project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-why/advance/2b",
            data={
                "why_action_inciting_cause": "accept",
                "why_value_inciting_cause": "",
                "why_action_underlying_desire": "accept",
                "why_value_underlying_desire": "",
                "why_action_stakes": "reject",
                "why_value_stakes": "",
                "why_action_obstacle": "accept",
                "why_value_obstacle": "",
            },
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code in (302, 303)

    cp = _extract_saved_cp(sink)

    # Pre-existing lock must be removed
    assert "motivation_stakes" not in cp["locked_assumptions"]

    # An ambiguity flag must have been appended
    flag_fields = [f["field"] for f in cp["ambiguity_flags"]]
    assert "motivation_stakes" in flag_fields

    # The rejected field must not appear in motivation
    assert cp["motivation"].get("stakes") != "old-stakes-value"


# ---------------------------------------------------------------------------
# Test 3: accept uses the AI-engine value; confidence >= 0.9
# ---------------------------------------------------------------------------


def test_accept_locks_engine_value_and_bumps_confidence(client, fake_user):
    project = _make_project()
    sink = []
    patches = _patch_request(project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-why/advance/2b",
            data={
                "why_action_inciting_cause": "accept",
                "why_value_inciting_cause": "",
                "why_action_underlying_desire": "accept",
                "why_value_underlying_desire": "",
                "why_action_stakes": "accept",
                "why_value_stakes": "",
                "why_action_obstacle": "accept",
                "why_value_obstacle": "",
            },
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code in (302, 303)

    cp = _extract_saved_cp(sink)

    # Each accepted field gets the engine value locked
    assert cp["locked_assumptions"]["motivation_inciting_cause"] == "engine-cause"
    assert cp["locked_assumptions"]["motivation_underlying_desire"] == "engine-desire"
    assert cp["locked_assumptions"]["motivation_stakes"] == "engine-stakes"
    assert cp["locked_assumptions"]["motivation_obstacle"] == "engine-obstacle"

    # Values are mirrored back into motivation
    assert cp["motivation"]["inciting_cause"] == "engine-cause"
    assert cp["motivation"]["underlying_desire"] == "engine-desire"
    assert cp["motivation"]["stakes"] == "engine-stakes"
    assert cp["motivation"]["obstacle"] == "engine-obstacle"

    # Confidence was bumped
    assert cp["motivation"]["confidence"] >= 0.9
    assert cp["confidence_scores"]["motivation"] >= 0.9


# ---------------------------------------------------------------------------
# Test 4: motivation_* entries in surfaced_assumptions are NOT double-processed
# ---------------------------------------------------------------------------


def test_motivation_surfaced_assumption_is_not_double_processed(client, fake_user):
    """
    If a motivation_* field was also surfaced as a low-confidence assumption
    (via action_<i> in the generic loop), the WHY panel must own it and the
    generic loop must skip it.  The final locked value must come from the WHY
    panel input, not from the generic action_0 input.
    """
    project = _make_project(
        extra_cp={
            "surfaced_assumptions": [
                {
                    "field": "motivation_inciting_cause",
                    "value": "surfaced-engine-cause",
                    "confidence": 0.4,
                }
            ]
        }
    )
    sink = []
    patches = _patch_request(project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-why/advance/2b",
            data={
                # WHY panel sets an override for the same field
                "why_action_inciting_cause": "override",
                "why_value_inciting_cause": "why-panel-cause",
                "why_action_underlying_desire": "accept",
                "why_value_underlying_desire": "",
                "why_action_stakes": "accept",
                "why_value_stakes": "",
                "why_action_obstacle": "accept",
                "why_value_obstacle": "",
                # Generic loop action for surfaced item 0 — should be ignored
                # because motivation_inciting_cause belongs to the WHY panel.
                "action_0": "override",
                "value_0": "generic-loop-cause",
            },
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code in (302, 303)

    cp = _extract_saved_cp(sink)

    # The WHY panel value wins; the generic loop value must NOT override it.
    assert cp["locked_assumptions"]["motivation_inciting_cause"] == "why-panel-cause"
    assert cp["motivation"]["inciting_cause"] == "why-panel-cause"
    assert cp["locked_assumptions"].get("motivation_inciting_cause") != "generic-loop-cause"


# ---------------------------------------------------------------------------
# Test 5: _apply_locked_assumptions_inplace bumps confidence for any override
# ---------------------------------------------------------------------------


def test_apply_locked_assumptions_inplace_bumps_confidence_directly():
    """
    Unit test for _apply_locked_assumptions_inplace: calling it with a
    motivation_* key in locked should raise the confidence score to >= 0.9
    even if the packet started with a lower value.
    """
    cp = {
        "motivation": {"inciting_cause": "old", "confidence": 0.3},
        "confidence_scores": {"motivation": 0.3},
        "world_assumptions": {},
        "speaker": {},
    }
    locked = {"motivation_inciting_cause": "brand new cause"}
    app_module._apply_locked_assumptions_inplace(cp, locked)

    assert cp["motivation"]["inciting_cause"] == "brand new cause"
    assert cp["motivation"]["confidence"] >= 0.9
    assert cp["confidence_scores"]["motivation"] >= 0.9


# ---------------------------------------------------------------------------
# Test 6: all four motivation fields can be handled in a single submission
# ---------------------------------------------------------------------------


def test_all_four_fields_mixed_actions(client, fake_user):
    """
    Submitting a mix of accept / override / reject for all four WHY fields in
    one request produces the correct locked_assumptions state.
    """
    project = _make_project(
        extra_cp={"locked_assumptions": {"motivation_obstacle": "pre-existing-obstacle"}}
    )
    sink = []
    patches = _patch_request(project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-why/advance/2b",
            data={
                "why_action_inciting_cause": "override",
                "why_value_inciting_cause": "new cause",
                "why_action_underlying_desire": "accept",
                "why_value_underlying_desire": "",
                "why_action_stakes": "reject",
                "why_value_stakes": "",
                "why_action_obstacle": "reject",
                "why_value_obstacle": "",
            },
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code in (302, 303)

    cp = _extract_saved_cp(sink)
    la = cp["locked_assumptions"]

    # override
    assert la["motivation_inciting_cause"] == "new cause"
    # accept → engine value
    assert la["motivation_underlying_desire"] == "engine-desire"
    # reject → key removed
    assert "motivation_stakes" not in la
    # reject → pre-existing key also removed
    assert "motivation_obstacle" not in la

    # Both rejected fields produced ambiguity flags
    flag_fields = {f["field"] for f in cp["ambiguity_flags"]}
    assert "motivation_stakes" in flag_fields
    assert "motivation_obstacle" in flag_fields

    # At least two fields were locked → confidence bumped
    assert cp["motivation"]["confidence"] >= 0.9
