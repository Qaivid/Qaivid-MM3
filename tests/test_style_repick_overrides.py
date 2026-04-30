"""
Tests that a style re-pick never wipes speaker, location, or era edits (Task #73 / #89).

The advance_imagination route is the approval gate that fires kick_stage_2
(Storyboard) after the Director's Vision is confirmed.  It reads the project's
context_packet from the Brain and builds an overrides dict so those user-
confirmed values survive through the storyboard stage.

Two scenarios are verified here:

1. Re-pick path — context_packet already holds speaker/location/era values that
   the user approved on a prior run.  The overrides dict handed to kick_stage_2
   must carry those values (not None) so the Storyboard Engine does not regress.

2. First-time path — context_packet is empty (brand-new project).  The overrides
   dict must be all-None for speaker_name / location / era so the engine still
   generates fresh values and is not handed stale content.

The Flask app and psycopg are patched at import time so the suite runs without
a live database.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub psycopg.connect so importing app.py does not need a real DB.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql://test/test")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-style-repick")
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
# Helpers
# ---------------------------------------------------------------------------


class _ApprovingCursor:
    """Cursor whose first execute() always reports rowcount=1 (UPDATE matched)."""

    def __init__(self):
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        return None


class _ApprovingConn:
    def cursor(self, *a, **kw):
        return _ApprovingCursor()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_brain(context_packet=None, style_packet=None):
    """Return a mock ProjectBrain whose .read() returns the supplied packets."""
    brain = MagicMock()
    packets = {
        "context_packet": context_packet or {},
        "style_packet":   style_packet   or {},
    }
    brain.read.side_effect = lambda ns: packets.get(ns, {})
    return brain


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture
def fake_user():
    return {"id": 7, "email": "director@example.com"}


@pytest.fixture
def repick_project():
    """Project that already has a storyboard (i.e., a re-pick scenario)."""
    return {
        "id": "proj-repick",
        "user_id": 7,
        "name": "My_Music_Video",
        "stage": "imagination_review",
        "status": "awaiting_review",
        "styled_timeline": [{"shot_index": 0}],
    }


@pytest.fixture
def firsttime_project():
    """Brand-new project with no prior storyboard."""
    return {
        "id": "proj-firsttime",
        "user_id": 7,
        "name": "Fresh_Project",
        "stage": "imagination_review",
        "status": "awaiting_review",
        "styled_timeline": None,
    }


def _run_advance_imagination(client, project, brain, fake_user):
    """
    Drive POST /project/<id>/advance/imagination with all dependencies patched.

    Returns the list of (args, kwargs) captured from kick_stage_2 calls.
    """
    kick_calls = []

    def _fake_kick(pid, name, overrides):
        kick_calls.append({"project_id": pid, "name": name, "overrides": overrides})

    patches = [
        patch.object(app_module, "_get_project", return_value=project),
        patch.object(app_module, "current_user", return_value=fake_user),
        patch("auth.current_user", return_value=fake_user),
        patch.object(app_module, "db", lambda: _ApprovingConn()),
        patch.object(app_module.ProjectBrain, "load", return_value=brain),
        patch.object(app_module, "kick_stage_2", side_effect=_fake_kick),
    ]

    for p in patches:
        p.start()
    try:
        resp = client.post(f"/project/{project['id']}/advance/imagination")
    finally:
        for p in patches:
            p.stop()

    return kick_calls, resp


# ---------------------------------------------------------------------------
# Test 1 — Re-pick path: context_packet carries confirmed edits
# ---------------------------------------------------------------------------


def test_repick_overrides_carry_speaker_location_era(client, fake_user, repick_project):
    """
    When context_packet already has a speaker name, location_dna, and an era
    inside world_assumptions, the overrides dict passed to kick_stage_2 must
    contain those exact non-None values.
    """
    context_packet = {
        "speaker": {"name": "Amara", "gender": "female"},
        "location_dna": "Lagos waterfront, Nigeria",
        "world_assumptions": {"era": "contemporary"},
    }
    style_packet = {"preset": "cinematic_natural"}
    brain = _make_brain(context_packet=context_packet, style_packet=style_packet)

    kick_calls, resp = _run_advance_imagination(client, repick_project, brain, fake_user)

    assert resp.status_code in (302, 303), "advance_imagination must redirect on success"
    assert len(kick_calls) == 1, "kick_stage_2 must be called exactly once"
    overrides = kick_calls[0]["overrides"]

    assert overrides["speaker_name"] == "Amara", (
        "speaker_name in overrides must equal the confirmed name from context_packet"
    )
    assert overrides["location"] == "Lagos waterfront, Nigeria", (
        "location in overrides must equal location_dna from context_packet"
    )
    assert overrides["era"] == "contemporary", (
        "era in overrides must equal world_assumptions.era from context_packet"
    )


def test_repick_overrides_carry_era_from_top_level_fallback(client, fake_user, repick_project):
    """
    If era lives at the top level of context_packet rather than inside
    world_assumptions, the overrides dict must still pick it up.
    """
    context_packet = {
        "speaker": {"name": "Jonah", "gender": "male"},
        "location_dna": "Chicago rooftop",
        "era": "1990s",
    }
    brain = _make_brain(context_packet=context_packet)

    kick_calls, resp = _run_advance_imagination(client, repick_project, brain, fake_user)

    assert resp.status_code in (302, 303), "advance_imagination must redirect on success"
    overrides = kick_calls[0]["overrides"]
    assert overrides["era"] == "1990s", (
        "era in overrides must fall back to top-level era when world_assumptions is absent"
    )
    assert overrides["speaker_name"] == "Jonah"
    assert overrides["location"] == "Chicago rooftop"


# ---------------------------------------------------------------------------
# Test 2 — First-time path: empty context_packet yields all-None user fields
# ---------------------------------------------------------------------------


def test_firsttime_overrides_are_all_none_for_user_fields(client, fake_user, firsttime_project):
    """
    On a first-time run the context_packet is empty (the engine has not yet
    been called).  The overrides dict passed to kick_stage_2 must have
    speaker_name, location, and era all set to None so the Storyboard Engine
    generates fresh values instead of inheriting stale strings.
    """
    brain = _make_brain(context_packet={}, style_packet={})

    kick_calls, resp = _run_advance_imagination(client, firsttime_project, brain, fake_user)

    assert resp.status_code in (302, 303), "advance_imagination must redirect on success"
    assert len(kick_calls) == 1, "kick_stage_2 must be called exactly once"
    overrides = kick_calls[0]["overrides"]

    assert overrides["speaker_name"] is None, (
        "speaker_name must be None when context_packet has no speaker entry"
    )
    assert overrides["location"] is None, (
        "location must be None when context_packet has no location_dna"
    )
    assert overrides["era"] is None, (
        "era must be None when context_packet has no era entry"
    )


def test_firsttime_style_preset_falls_back_to_default(client, fake_user, firsttime_project):
    """
    When style_packet is also absent, style_preset in overrides should fall
    back to 'cinematic_natural' so downstream stages always have a preset value.
    """
    brain = _make_brain(context_packet={}, style_packet={})

    kick_calls, resp = _run_advance_imagination(client, firsttime_project, brain, fake_user)
    assert resp.status_code in (302, 303), "advance_imagination must redirect on success"

    overrides = kick_calls[0]["overrides"]
    assert overrides["style_preset"] == "cinematic_natural", (
        "style_preset must default to 'cinematic_natural' when style_packet is empty"
    )
