"""
Tests for reset_brief and regenerate_brief routes (Task #77).

Covers POST /project/<id>/reset_brief and POST /project/<id>/regenerate_brief
introduced in Task #70:

1. reset_brief happy path (from storyboard_review)  — clears chosen, sets
   styled_timeline=NULL, stage=creative_brief_review, status=awaiting_review.
2. reset_brief happy path (from creative_brief_review) — same outcome.
3. reset_brief rejection — wrong stage/status combo is rejected before DB hit.
4. regenerate_brief happy path — kicks kick_stage_brief with the correct
   pending overrides built from creative_brief._pending_overrides.
5. regenerate_brief pending overrides are normalised — missing style_preset
   falls back to "cinematic_natural".
6. regenerate_brief rejection — wrong stage/status combo is rejected before
   DB hit, kick_stage_brief is never called.

The Flask app and its psycopg dependency are patched at import time so the
suite runs without a live database.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub psycopg.connect so importing app.py doesn't need a real DB.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-brief-reset")
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
# Recording fakes — capture every SQL call for assertions without a live DB.
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


def _make_project(stage="storyboard_review", status="awaiting_review", extra_cb=None):
    """Return a minimal project dict suitable for reset_brief / regenerate_brief."""
    cb = {
        "variants": [{"id": "v1", "title": "Option A"}],
        "chosen": {"variant_id": "v1", "title": "Option A"},
        "_pending_overrides": {
            "speaker_name": "Aiko",
            "location": "Rooftop",
            "era": "Contemporary",
            "style_preset": "music_video_vibrant",
        },
    }
    if extra_cb:
        cb.update(extra_cb)
    return {
        "id": "proj-brief",
        "name": "Test Project",
        "user_id": 42,
        "stage": stage,
        "status": status,
        "context_packet": {
            "creative_brief": cb,
        },
        "styled_timeline": [{"shot": 1}],
    }


def _patches_for(project, sink, user, kick_mock):
    """Return the list of patches needed for one route call."""
    return [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=user),
        patch("auth.current_user", return_value=user),
        patch.object(app_module, "db", lambda: RecordingConn(sink)),
        patch.object(app_module, "kick_stage_brief", kick_mock),
    ]


def _run(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in patches:
            p.stop()


def _extract_update_call(sink, keyword="UPDATE projects SET context_packet"):
    """Return the first (sql, params) tuple whose sql contains keyword."""
    hits = [(sql, p) for sql, p in sink if keyword in sql]
    assert hits, f"No matching UPDATE found. Recorded calls:\n{sink}"
    return hits[0]


def _unwrap_cp(raw):
    """Unwrap a psycopg Json wrapper to a plain dict."""
    return raw.obj if hasattr(raw, "obj") else raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture
def fake_user():
    return {"id": 42, "email": "director@test.com"}


# ---------------------------------------------------------------------------
# reset_brief tests
# ---------------------------------------------------------------------------


def test_reset_brief_from_storyboard_review_clears_chosen_and_sets_stage(client, fake_user):
    """Happy path from storyboard_review: chosen removed, styled_timeline nulled, stage/status set correctly."""
    project = _make_project(stage="storyboard_review")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/reset_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303), f"Expected redirect, got {resp.status_code}"

    sql, params = _extract_update_call(sink)

    # styled_timeline must be cleared in the SQL
    assert "styled_timeline=NULL" in sql.replace(" ", "").replace("\n", ""), (
        "reset_brief SQL must null styled_timeline"
    )

    # Stage and status values in params
    assert "creative_brief_review" in params, "stage should be set to creative_brief_review"
    assert params.count("awaiting_review") >= 1, "status should be set to awaiting_review"

    # context_packet no longer contains chosen
    cp = _unwrap_cp(params[0])
    assert "chosen" not in cp.get("creative_brief", {}), "chosen should be cleared"

    # kick_stage_brief must NOT be called for a simple reset
    kick_mock.assert_not_called()


def test_reset_brief_from_creative_brief_review_also_succeeds(client, fake_user):
    """reset_brief also works when the project is parked at creative_brief_review."""
    project = _make_project(stage="creative_brief_review")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/reset_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    sql, params = _extract_update_call(sink)

    # styled_timeline must be cleared here too
    assert "styled_timeline=NULL" in sql.replace(" ", "").replace("\n", ""), (
        "reset_brief SQL must null styled_timeline"
    )

    cp = _unwrap_cp(params[0])
    assert "chosen" not in cp.get("creative_brief", {}), "chosen should be cleared"
    assert "creative_brief_review" in params
    kick_mock.assert_not_called()


def test_reset_brief_rejected_when_stage_is_wrong(client, fake_user):
    """reset_brief must not touch the DB when the project is at a disallowed stage."""
    project = _make_project(stage="queued", status="awaiting_review")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/reset_brief")

    resp = _run(patches, go)
    # Route redirects with a flash error — still a redirect, not a 200 or 5xx
    assert resp.status_code in (302, 303)

    # No UPDATE should have been issued
    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched when stage is not allowed"
    kick_mock.assert_not_called()


def test_reset_brief_rejected_when_status_is_not_awaiting_review(client, fake_user):
    """reset_brief must not touch the DB when status is not awaiting_review."""
    project = _make_project(stage="storyboard_review", status="queued")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/reset_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched when status is not awaiting_review"
    kick_mock.assert_not_called()


# ---------------------------------------------------------------------------
# regenerate_brief tests
# ---------------------------------------------------------------------------


def test_regenerate_brief_kicks_with_correct_pending_overrides(client, fake_user):
    """Happy path: kick_stage_brief is called with overrides from _pending_overrides."""
    project = _make_project(stage="storyboard_review")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/regenerate_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    # DB must have been updated to queued
    sql, params = _extract_update_call(sink)
    assert "queued" in params, "stage should be flipped to queued"

    # styled_timeline must be nulled so stale shots don't survive
    assert "styled_timeline=NULL" in sql.replace(" ", "").replace("\n", ""), (
        "regenerate_brief SQL must null styled_timeline"
    )

    # chosen is cleared so the new review starts clean
    cp = _unwrap_cp(params[0])
    assert "chosen" not in cp.get("creative_brief", {}), "chosen should be cleared before regeneration"

    # kick_stage_brief called once with the project id and the correct overrides
    kick_mock.assert_called_once()
    call_args = kick_mock.call_args
    assert call_args[0][0] == "proj-brief", "project_id must be passed as first positional arg"
    overrides = call_args[0][1]
    assert overrides["speaker_name"] == "Aiko"
    assert overrides["location"] == "Rooftop"
    assert overrides["era"] == "Contemporary"
    assert overrides["style_preset"] == "music_video_vibrant"


def test_regenerate_brief_style_preset_defaults_to_cinematic_natural(client, fake_user):
    """When _pending_overrides has no style_preset, it defaults to cinematic_natural."""
    project = _make_project(
        stage="creative_brief_review",
        extra_cb={
            "_pending_overrides": {
                "speaker_name": "Sam",
                "location": None,
                "era": None,
                # style_preset deliberately absent
            }
        },
    )
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/regenerate_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    kick_mock.assert_called_once()
    overrides = kick_mock.call_args[0][1]
    assert overrides["style_preset"] == "cinematic_natural", (
        "style_preset must default to 'cinematic_natural' when absent from _pending_overrides"
    )


def test_regenerate_brief_rejected_when_stage_is_wrong(client, fake_user):
    """regenerate_brief must not touch the DB or call kick when stage is wrong."""
    project = _make_project(stage="context_review", status="awaiting_review")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/regenerate_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched when stage is not allowed"
    kick_mock.assert_not_called()


def test_regenerate_brief_rejected_when_status_is_not_awaiting_review(client, fake_user):
    """regenerate_brief must not touch the DB or call kick when status is wrong."""
    project = _make_project(stage="storyboard_review", status="processing")
    sink = []
    kick_mock = MagicMock()
    patches = _patches_for(project, sink, fake_user, kick_mock)

    def go():
        return client.post("/project/proj-brief/regenerate_brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched when status is not awaiting_review"
    kick_mock.assert_not_called()


# ---------------------------------------------------------------------------
# advance_brief race-condition tests (Task #92)
# ---------------------------------------------------------------------------


def _patches_for_advance(project, sink, user, materializer_mock):
    """Patches needed to exercise the advance_brief route without a live DB."""
    return [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=user),
        patch("auth.current_user", return_value=user),
        patch.object(app_module, "db", lambda: RecordingConn(sink)),
        patch.object(app_module, "kick_stage_materializer", materializer_mock),
        # File upload helper — returns None for missing files (no uploads in tests)
        patch.object(app_module, "_save_upload_to_r2", return_value=None),
    ]


def test_advance_brief_blocked_when_stage_is_queued(client, fake_user):
    """advance_brief must refuse to lock while regeneration is in progress (stage=queued)."""
    project = _make_project(stage="queued", status="queued")
    sink = []
    materializer_mock = MagicMock()
    patches = _patches_for_advance(project, sink, fake_user, materializer_mock)

    def go():
        return client.post("/project/proj-brief/advance/brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    # No DB write should happen — we must bail before touching the DB
    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched while regeneration is in progress"

    # Materializer must never be kicked
    materializer_mock.assert_not_called()


def test_advance_brief_blocked_when_only_status_is_queued(client, fake_user):
    """advance_brief must refuse to lock when status alone is queued (regeneration race window)."""
    project = _make_project(stage="creative_brief_review", status="queued")
    sink = []
    materializer_mock = MagicMock()
    patches = _patches_for_advance(project, sink, fake_user, materializer_mock)

    def go():
        return client.post("/project/proj-brief/advance/brief")

    resp = _run(patches, go)
    assert resp.status_code in (302, 303)

    update_calls = [(sql, p) for sql, p in sink if "UPDATE projects" in sql]
    assert not update_calls, "DB must not be touched when status is queued"
    materializer_mock.assert_not_called()
