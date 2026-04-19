"""
Tests for the inline director override endpoints (Task #75).

Covers POST /project/<id>/shot/<idx>/motion_prompt and .../framing_directive:
- happy-path save updates the right shot in styled_timeline and persists it
- requests outside storyboard_review stage are rejected
- unknown shot indices return 404
- payloads longer than 500 chars are trimmed before persistence

The Flask app and its psycopg dependency are patched at import time so the
suite can run without a live database.
"""

import os
import sys
import types
import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub psycopg.connect so importing app.py does not require a DB.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
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
# Recording fakes that the route's `with db() as conn, conn.cursor() as cur:`
# block can drive. We capture every UPDATE so tests can assert persistence.
# ---------------------------------------------------------------------------


class RecordingCursor:
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
        self.commits = 0

    def cursor(self, *a, **kw):
        return RecordingCursor(self.sink)

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture
def fake_user():
    return {"id": 42, "email": "director@example.com"}


@pytest.fixture
def base_project():
    return {
        "id": "proj-abc",
        "user_id": 42,
        "stage": "storyboard_review",
        "styled_timeline": [
            {"shot_index": 0, "timeline_index": 0, "motion_prompt": "old0",
             "framing_directive": "frame0"},
            {"shot_index": 1, "timeline_index": 1, "motion_prompt": "old1",
             "framing_directive": "frame1"},
            {"timeline_index": 2, "motion_prompt": "old2",
             "framing_directive": "frame2"},
        ],
    }


def _patch_request(project, sink, user):
    """Patch the request-time dependencies for both endpoints."""
    return [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=user),
        # `login_required` calls auth.current_user from its own module scope.
        patch("auth.current_user", return_value=user),
        patch.object(app_module, "db", lambda: RecordingConn(sink)),
    ]


def _run_with_patches(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# motion_prompt
# ---------------------------------------------------------------------------


def test_motion_prompt_happy_path_persists_only_target_shot(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/1/motion_prompt",
            data=json.dumps({"motion_prompt": "slow dolly in on her hands"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}

    # Exactly one UPDATE was issued.
    assert len(sink) == 1
    sql, params = sink[0]
    assert "UPDATE projects SET styled_timeline" in sql
    new_timeline, pid = params
    assert pid == "proj-abc"

    # Json wrapper holds the timeline payload.
    payload = new_timeline.obj if hasattr(new_timeline, "obj") else new_timeline
    assert payload[0]["motion_prompt"] == "old0"  # untouched
    assert payload[1]["motion_prompt"] == "slow dolly in on her hands"
    assert payload[2]["motion_prompt"] == "old2"  # untouched
    # Framing directives must not be disturbed by a motion_prompt save.
    assert [s["framing_directive"] for s in payload] == ["frame0", "frame1", "frame2"]


def test_motion_prompt_rejects_wrong_stage(client, fake_user, base_project):
    base_project["stage"] = "stills_review"
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/1/motion_prompt",
            data=json.dumps({"motion_prompt": "x"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
    # Nothing must have been written.
    assert sink == []


def test_motion_prompt_unknown_shot_index_returns_404(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/99/motion_prompt",
            data=json.dumps({"motion_prompt": "x"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 404
    assert resp.get_json()["ok"] is False
    assert sink == []


def test_motion_prompt_oversized_payload_is_trimmed_to_500(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)
    huge = "z" * 1200

    def go():
        return client.post(
            "/project/proj-abc/shot/0/motion_prompt",
            data=json.dumps({"motion_prompt": huge}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 200
    assert len(sink) == 1
    _, params = sink[0]
    new_timeline, _pid = params
    payload = new_timeline.obj if hasattr(new_timeline, "obj") else new_timeline
    saved = payload[0]["motion_prompt"]
    assert len(saved) == 500
    assert saved == "z" * 500


# ---------------------------------------------------------------------------
# framing_directive
# ---------------------------------------------------------------------------


def test_framing_directive_happy_path_persists_only_target_shot(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/2/framing_directive",
            data=json.dumps({"framing_directive": "tight over-shoulder"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}

    assert len(sink) == 1
    sql, params = sink[0]
    assert "UPDATE projects SET styled_timeline" in sql
    new_timeline, pid = params
    assert pid == "proj-abc"
    payload = new_timeline.obj if hasattr(new_timeline, "obj") else new_timeline

    # Only the shot whose timeline_index == 2 is rewritten (it has no shot_index).
    assert payload[2]["framing_directive"] == "tight over-shoulder"
    assert payload[0]["framing_directive"] == "frame0"
    assert payload[1]["framing_directive"] == "frame1"
    # Motion prompts on every shot must remain untouched.
    assert [s["motion_prompt"] for s in payload] == ["old0", "old1", "old2"]


def test_framing_directive_rejects_wrong_stage(client, fake_user, base_project):
    base_project["stage"] = "queued"
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/0/framing_directive",
            data=json.dumps({"framing_directive": "x"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 400
    assert sink == []


def test_framing_directive_unknown_shot_index_returns_404(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)

    def go():
        return client.post(
            "/project/proj-abc/shot/77/framing_directive",
            data=json.dumps({"framing_directive": "x"}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 404
    assert sink == []


def test_framing_directive_oversized_payload_is_trimmed_to_500(client, fake_user, base_project):
    sink = []
    patches = _patch_request(base_project, sink, fake_user)
    huge = "q" * 999

    def go():
        return client.post(
            "/project/proj-abc/shot/1/framing_directive",
            data=json.dumps({"framing_directive": huge}),
            content_type="application/json",
        )

    resp = _run_with_patches(patches, go)
    assert resp.status_code == 200
    _, params = sink[0]
    new_timeline, _pid = params
    payload = new_timeline.obj if hasattr(new_timeline, "obj") else new_timeline
    saved = payload[1]["framing_directive"]
    assert len(saved) == 500
    assert saved == "q" * 500
