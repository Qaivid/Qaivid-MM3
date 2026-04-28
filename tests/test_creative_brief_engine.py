"""Tests for creative_brief_engine variant generation and coercion (Task #69).

Covers:
- Fallback variants have the required schema shape (all spec fields present).
- coerce_chosen parses comma-separated cast_roster correctly.
- coerce_chosen preserves an intentionally blank cast_roster.
- coerce_chosen handles list-mode cast_roster from future multi-input UI.
- Variant id uniqueness in fallback set.
- generate_variants sets used_fallback=True when LLM returns <2 valid variants (Task #72).
- generate_variants sets used_fallback=True on LLM exception (Task #72).
- generate_variants sets used_fallback=False on a successful LLM response (Task #72).
- stage_creative_brief.html renders #fallback-warning when used_fallback is True (Task #81).
- stage_creative_brief.html omits #fallback-warning when used_fallback is False/None (Task #81).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from creative_brief_engine import _fallback_variants, coerce_chosen, generate_variants

_REQUIRED_VARIANT_KEYS = (
    "id", "title", "pitch", "treatment", "scenes",
    "cast_roster", "central_metaphor", "justification", "self_critique",
)


def _minimal_context():
    return {
        "speaker": {"identity": "a child"},
        "motivation": {
            "inciting_cause": "a fight at home",
            "underlying_desire": "to be held",
        },
        "location_dna": "a small bedroom",
    }


# ---------------------------------------------------------------------------
# Fallback variant schema
# ---------------------------------------------------------------------------

def test_fallback_variants_have_required_keys():
    variants = _fallback_variants(_minimal_context(), ["Mum", "Dad"])
    assert len(variants) >= 2, "Need at least 2 fallback variants"
    for v in variants:
        for key in _REQUIRED_VARIANT_KEYS:
            assert key in v, f"Variant missing key: {key!r}"


def test_fallback_variants_self_critique_shape():
    variants = _fallback_variants(_minimal_context(), [])
    for v in variants:
        sc = v["self_critique"]
        assert isinstance(sc, dict)
        assert 1 <= sc["score"] <= 10
        assert isinstance(sc["rationale"], str)


def test_fallback_variants_have_unique_ids():
    variants = _fallback_variants(_minimal_context(), [])
    ids = [v["id"] for v in variants]
    assert len(ids) == len(set(ids)), "Variant ids must be unique"


def test_fallback_variants_justification_non_empty():
    variants = _fallback_variants(_minimal_context(), [])
    for v in variants:
        assert v["justification"].strip(), "justification should be non-empty in fallback"


# ---------------------------------------------------------------------------
# coerce_chosen
# ---------------------------------------------------------------------------

def test_coerce_chosen_comma_separated_cast():
    payload = {"variant_id": "v1", "cast_roster": "Alice, Bob, Carol"}
    result = coerce_chosen(payload)
    assert result["cast_roster"] == ["Alice", "Bob", "Carol"]


def test_coerce_chosen_blank_cast_roster_stays_empty():
    """User intentionally cleared the cast field — must remain []."""
    payload = {"variant_id": "v2", "cast_roster": ""}
    result = coerce_chosen(payload)
    assert result["cast_roster"] == []


def test_coerce_chosen_list_cast_roster_passthrough():
    """List-mode input (future multi-select) each element kept as-is."""
    payload = {"variant_id": "v1", "cast_roster": ["Alice", "Bob"]}
    result = coerce_chosen(payload)
    assert result["cast_roster"] == ["Alice", "Bob"]


def test_coerce_chosen_required_keys_present():
    payload = {"variant_id": "v1"}
    result = coerce_chosen(payload)
    for key in ("variant_id", "title", "pitch", "treatment",
                "central_metaphor", "director_note", "justification", "cast_roster"):
        assert key in result, f"coerce_chosen missing key: {key!r}"


def test_coerce_chosen_truncates_long_title():
    payload = {"variant_id": "v1", "title": "x" * 500}
    result = coerce_chosen(payload)
    assert len(result["title"]) <= 120


def test_coerce_chosen_cast_roster_max_ten():
    payload = {"variant_id": "v1", "cast_roster": ", ".join([f"Person{i}" for i in range(20)])}
    result = coerce_chosen(payload)
    assert len(result["cast_roster"]) <= 10


# ---------------------------------------------------------------------------
# generate_variants — used_fallback flag (Task #72)
# ---------------------------------------------------------------------------

def _make_llm_response(variants_payload):
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = json.dumps({"variants": variants_payload})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _good_variant(vid="v1"):
    return {
        "id": vid, "title": "T", "pitch": "P", "treatment": "TR",
        "scenes": [{"name": "S", "beat_range": "0-10", "summary": "x"}],
        "cast_roster": [], "central_metaphor": "M",
        "justification": "J",
        "self_critique": {"score": 7, "rationale": "R"},
    }


def test_generate_variants_success_returns_used_fallback_false():
    """When LLM returns ≥2 valid variants, used_fallback should be False."""
    good_variants = [_good_variant("v1"), _good_variant("v2"), _good_variant("v3")]
    mock_resp = _make_llm_response(good_variants)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("creative_brief_engine.AsyncOpenAI", return_value=mock_client):
        variants, used_fallback = asyncio.run(generate_variants(
            api_key="test-key",
            context_packet=_minimal_context(),
        ))

    assert used_fallback is False
    assert len(variants) >= 2


def test_generate_variants_insufficient_variants_sets_fallback_true():
    """When LLM returns <2 coerceable variants, used_fallback should be True."""
    mock_resp = _make_llm_response([])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("creative_brief_engine.AsyncOpenAI", return_value=mock_client):
        variants, used_fallback = asyncio.run(generate_variants(
            api_key="test-key",
            context_packet=_minimal_context(),
        ))

    assert used_fallback is True
    assert len(variants) >= 2


def test_generate_variants_llm_exception_sets_fallback_true():
    """When the LLM call raises an exception, used_fallback should be True."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

    with patch("creative_brief_engine.AsyncOpenAI", return_value=mock_client):
        variants, used_fallback = asyncio.run(generate_variants(
            api_key="test-key",
            context_packet=_minimal_context(),
        ))

    assert used_fallback is True
    assert len(variants) >= 2


# ---------------------------------------------------------------------------
# Template banner rendering — used_fallback flag (Task #81)
#
# These tests render the *real* templates/stage_creative_brief.html through
# Jinja2's FileSystemLoader so that any future change to that file is caught.
# Flask-specific globals (url_for, csrf_token) are stubbed out; base.html and
# the small include partials are replaced with minimal pass-through stubs so
# the render stays fast and self-contained.
# ---------------------------------------------------------------------------

import os
import types
from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")

_STUB_TEMPLATES = {
    "base.html": "{% block content %}{% endblock %}",
    "_stage_header.html": "",
    "_redo_panel.html": "",
}


def _make_jinja_env():
    """Build a Jinja2 environment that loads the real templates directory but
    overrides inherited/included partials with lightweight stubs."""
    loader = ChoiceLoader([
        DictLoader(_STUB_TEMPLATES),
        FileSystemLoader(_TEMPLATES_DIR),
    ])
    env = Environment(loader=loader)
    env.globals["url_for"] = lambda *a, **kw: "#"
    env.globals["csrf_token"] = lambda: "test-csrf"
    return env


def _make_project(used_fallback):
    """Return a minimal project-like namespace matching the template's data model."""
    project = types.SimpleNamespace(
        name="Test Project",
        id="proj-1",
        _is_review_only=True,
        context_packet={
            "creative_brief": {
                "used_fallback": used_fallback,
                "chosen_variant": None,
                "title": "",
                "pitch": "",
                "treatment": "",
                "scenes": [],
                "cast_roster": [],
                "central_metaphor": "",
                "justification": "",
                "self_critique": {"score": 7, "rationale": ""},
                "world_assumptions": {},
            }
        },
    )
    return project


def _render_brief_template(used_fallback):
    """Render stage_creative_brief.html with the given used_fallback value."""
    env = _make_jinja_env()
    tmpl = env.get_template("stage_creative_brief.html")
    return tmpl.render(project=_make_project(used_fallback))


def test_brief_template_banner_present_when_used_fallback_true():
    """stage_creative_brief.html must render #fallback-warning when used_fallback is True."""
    html = _render_brief_template(True)
    assert 'id="fallback-warning"' in html, (
        "Expected #fallback-warning banner in rendered stage_creative_brief.html"
    )


def test_brief_template_banner_absent_when_used_fallback_false():
    """stage_creative_brief.html must NOT render #fallback-warning when used_fallback is False."""
    html = _render_brief_template(False)
    assert 'id="fallback-warning"' not in html, (
        "Banner should be hidden in stage_creative_brief.html when used_fallback is False"
    )


def test_brief_template_banner_absent_when_used_fallback_none():
    """stage_creative_brief.html must NOT render #fallback-warning when used_fallback is None."""
    html = _render_brief_template(None)
    assert 'id="fallback-warning"' not in html, (
        "Banner should be hidden in stage_creative_brief.html when used_fallback is None/missing"
    )
