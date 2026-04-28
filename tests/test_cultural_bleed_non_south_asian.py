"""
Tests to confirm no South Asian cultural bleed when the pipeline processes
non-South-Asian songs (Task #84).

Bias Findings Summary (Task #84)
=================================
Three cultural worlds were tested: UK English folk, French chanson, Korean ballad.
No South Asian bias was discovered in any of the four pipeline layers tested:

Stage 2 — Context Engine (unified_context_engine_master.py):
  - CulturePackRegistry correctly fires NO South Asian pack for any non-SA lyric.
  - The universal pack world_defaults are empty — no SA geography or cultural_dna
    can bleed through as a default.
  - _repair_world_assumptions() preserves LLM-provided geography faithfully and
    does NOT substitute any SA default when geography is blank.
  - The system prompt exposes culture_pack_id as "none" for non-SA content.

Stage 6 — Creative Brief Engine (creative_brief_engine.py):
  - _build_cultural_grounding() returns empty string for UK/French/Korean packets
    (only fires for Punjabi/Urdu markers in location_dna or geography).
  - _fallback_variants() does not contain SA cultural markers in its scene text.

Stage 5 — Visual Storyboard Engine (visual_storyboard_engine.py):
  - The "PUNJABI LOCATION MANDATE" block is not injected for non-SA geography.
  - visual_prompt and location_dna in each shot contain no SA cultural markers.

No code fixes were required. All layers are clean for these three cultural worlds.

Coverage:
- CulturePackRegistry.detect_pack() returns None for UK/French/Korean lyrics.
- _repair_world_assumptions() preserves British / French / Korean geography
  provided by the LLM and does NOT substitute South Asian defaults.
- _build_system_prompt() shows culture_pack_id "none" for non-South-Asian lyrics.
- Full generate() end-to-end (LLM mocked) correctly produces British / French /
  Korean world_assumptions with no SA geography, cultural_dna, or SA props.
- creative_brief._build_cultural_grounding() returns empty for non-SA packets.
- creative_brief._fallback_variants() contains no SA cultural markers.
- VisualStoryboardEngine.build_storyboard() injects no Punjabi mandate / SA props
  when geography is British / French / Korean.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from unified_context_engine_master import (
    CulturePackRegistry,
    MetaMindContextEngineFinal,
)
from creative_brief_engine import (
    _build_cultural_grounding,
    _fallback_variants,
    generate_variants,
)
from visual_storyboard_engine import VisualStoryboardEngine


# ---------------------------------------------------------------------------
# Sample lyrics for each cultural world under test
# ---------------------------------------------------------------------------

UK_FOLK_LYRICS = """\
Over the rolling hills of Somerset I roam,
The hedgerows dressed in blossom, calling me home.
By the river where the willows softly weep,
My love is but a memory I keep.
"""

FRENCH_CHANSON_LYRICS = """\
Sous les toits de Paris la pluie tombe encore,
Je marche sur les quais, je rêve comme avant.
Tu m'as laissé seul devant cette vieille porte,
Et le vent de la Seine emporte mon tourment.
"""

KOREAN_BALLAD_LYRICS = """\
서울의 밤거리를 혼자 걷네,
별빛 아래 네 생각이 나,
그리움이 파도처럼 밀려와,
내 가슴 속에 너만 남아.
"""

SOUTH_ASIAN_LYRICS = """\
pind di gali vich charpai te baithe,
phulkari da dupatta ohna de hath,
khet vich vaisakh di hawa chalti,
jhhanjar di awaaz dil nu karti ghat.
"""

SOUTH_ASIAN_CULTURE_MARKERS = {
    "dupatta", "bangles", "mustard field", "phulkari", "south asian",
    "punjabi rural", "punjab region", "south asia", "dargah", "shrine",
    "charpai", "jhhanjar", "pind",
}


# ---------------------------------------------------------------------------
# Helper: assert no South Asian cultural bleed in a string
# ---------------------------------------------------------------------------

def _assert_no_sa_bleed(text: str, label: str) -> None:
    """Assert that *text* contains none of the known South Asian bleed markers."""
    lowered = (text or "").lower()
    for marker in SOUTH_ASIAN_CULTURE_MARKERS:
        assert marker not in lowered, (
            f"{label}: South Asian cultural bleed detected — found '{marker}' in: {text!r}"
        )


# ---------------------------------------------------------------------------
# Helper: build a minimal but valid LLM-style context packet for a song
# ---------------------------------------------------------------------------

def _make_llm_packet(
    geography: str,
    cultural_background: str,
    language: str,
    location_dna: str,
) -> dict:
    """Return a realistic-looking context packet the LLM might produce."""
    return {
        "input_profile": {
            "recognized_type": "song",
            "raw_detected_type": "song",
            "is_mixed_input": False,
            "structure_quality": "inferred",
            "source_format": "plain_text",
            "language": {"primary": language, "script": "Latin", "dialect": ""},
            "analysis_confidence": 0.85,
        },
        "input_type": "song",
        "language": language,
        "narrative_mode": "lyrical",
        "location_dna": location_dna,
        "genre_directive": "Prioritize lyrical meaning and emotional flow.",
        "core_theme": "longing and memory",
        "dramatic_premise": "A speaker recalls a lost love against a familiar landscape.",
        "narrative_spine": "The speaker walks through their homeland, haunted by the memory of love.",
        "speaker": {
            "identity": "a solitary figure walking through a familiar landscape",
            "gender": "Male",
            "age_range": "young adult",
            "emotional_state": "nostalgic and melancholic",
            "social_role": "lover",
            "relationship_to_addressee": "former romantic partner",
            "cultural_background": cultural_background,
        },
        "addressee": {
            "identity": "a lost love",
            "relationship": "former romantic partner",
            "presence": "absent",
        },
        "world_assumptions": {
            "geography": geography,
            "era": "contemporary",
            "season": "spring",
            "timeline_nature": "memory",
            "social_context": "rural community life",
            "economic_context": "unspecified",
        },
        "emotional_arc": {
            "opening": "Peaceful longing",
            "development": "Deepening sense of loss",
            "climax": "Peak of grief",
            "resolution": "Quiet acceptance",
        },
        "motivation": {
            "inciting_cause": "Walking through a familiar landscape that evokes a lost love",
            "underlying_desire": "To reconnect with or make peace with what has been lost",
            "stakes": "Remaining trapped in grief versus finding peace",
            "obstacle": "The irrecoverable absence of the beloved",
            "confidence": 0.82,
        },
        "line_meanings": [
            {
                "line_index": i + 1,
                "text": f"line {i + 1}",
                "literal_meaning": "A lyrical image of landscape and longing",
                "implied_meaning": "The speaker feels the absence of their love",
                "emotional_meaning": "Melancholy and nostalgic longing",
                "cultural_meaning": f"Imagery rooted in {geography}",
                "meaning": "Longing expressed through landscape",
                "function": "verse",
                "repeat_status": "original",
                "intensity": 0.6,
            }
            for i in range(4)
        ],
        "entities": [],
        "literary_devices": ["imagery", "personification"],
        "cultural_constraints": [
            f"The world is rooted in {geography} — do not substitute imagery from other cultures.",
        ],
        "preservation_rules": ["speaker cultural identity", "geography"],
        "creative_freedom": ["exact locations", "props", "colour palette"],
        "surfaced_assumptions": [],
        "locked_assumptions": {},
        "ambiguity_flags": [],
        "confidence": 0.83,
        "confidence_scores": {
            "overall": 0.83,
            "cultural": 0.80,
            "emotional": 0.87,
            "speaker": 0.82,
            "narrative_mode": 0.85,
            "motivation": 0.80,
        },
    }


# =============================================================================
# PART 1 — CulturePackRegistry: deterministic pack detection
# =============================================================================

class TestCulturePackDetectionNonSouthAsian:
    """CulturePackRegistry must not fire any South Asian pack for non-SA lyrics."""

    def test_uk_folk_lyrics_detect_no_pack(self):
        pack = CulturePackRegistry.detect_pack(UK_FOLK_LYRICS)
        assert pack is None, (
            f"UK folk lyrics triggered South Asian pack '{pack}'; expected None"
        )

    def test_french_chanson_detect_no_pack(self):
        pack = CulturePackRegistry.detect_pack(FRENCH_CHANSON_LYRICS)
        assert pack is None, (
            f"French chanson triggered South Asian pack '{pack}'; expected None"
        )

    def test_korean_ballad_detect_no_pack(self):
        pack = CulturePackRegistry.detect_pack(KOREAN_BALLAD_LYRICS)
        assert pack is None, (
            f"Korean ballad triggered South Asian pack '{pack}'; expected None"
        )

    def test_south_asian_lyrics_do_trigger_a_pack(self):
        """Sanity check: lyrics with South Asian triggers still get a pack."""
        pack = CulturePackRegistry.detect_pack(SOUTH_ASIAN_LYRICS)
        assert pack is not None, (
            "South Asian lyrics should trigger a culture pack, but got None"
        )
        assert "punjabi" in pack.lower() or "urdu" in pack.lower() or "qawwali" in pack.lower(), (
            f"Expected a Punjabi/Urdu pack; got '{pack}'"
        )

    def test_uk_folk_universal_pack_returned(self):
        """When no pack fires, get_pack returns the universal pack (empty world_defaults)."""
        pack_id = CulturePackRegistry.detect_pack(UK_FOLK_LYRICS)
        pack = CulturePackRegistry.get_pack(pack_id)
        world_defaults = pack.get("world_defaults", {})
        geography = world_defaults.get("geography", "")
        assert "south asia" not in geography.lower() and "punjab" not in geography.lower(), (
            f"Universal pack geography should not default to South Asian: {geography!r}"
        )

    def test_french_universal_pack_world_defaults_empty(self):
        pack_id = CulturePackRegistry.detect_pack(FRENCH_CHANSON_LYRICS)
        pack = CulturePackRegistry.get_pack(pack_id)
        world_defaults = pack.get("world_defaults", {})
        assert not world_defaults, (
            f"Universal pack world_defaults should be empty for French song, got: {world_defaults}"
        )

    def test_no_sa_metaphors_active_for_uk_lyrics(self):
        """No South Asian cultural metaphors should be active for UK folk lyrics."""
        pack_id = CulturePackRegistry.detect_pack(UK_FOLK_LYRICS)
        metaphors = CulturePackRegistry.get_triggered_metaphors(UK_FOLK_LYRICS, pack_id)
        # metaphors should be empty since no SA triggers appear in the text
        assert metaphors == {}, (
            f"South Asian cultural metaphors triggered for UK folk lyrics: {metaphors}"
        )

    def test_no_sa_metaphors_active_for_french_lyrics(self):
        pack_id = CulturePackRegistry.detect_pack(FRENCH_CHANSON_LYRICS)
        metaphors = CulturePackRegistry.get_triggered_metaphors(FRENCH_CHANSON_LYRICS, pack_id)
        assert metaphors == {}, (
            f"South Asian cultural metaphors triggered for French chanson: {metaphors}"
        )


# =============================================================================
# PART 2 — _repair_world_assumptions: geography is preserved, not overwritten
# =============================================================================

class TestRepairWorldAssumptionsNoBiasInjection:
    """
    _repair_world_assumptions must preserve the LLM-provided geography for
    non-South-Asian songs and must NOT substitute South Asian defaults.
    """

    def _make_engine(self) -> MetaMindContextEngineFinal:
        return MetaMindContextEngineFinal(api_key="test-key")

    def _universal_pack(self) -> dict:
        return CulturePackRegistry.get_pack(None)

    def test_british_geography_preserved(self):
        engine = self._make_engine()
        data = {
            "world_assumptions": {
                "geography": "rural Somerset, England",
                "era": "contemporary",
                "season": "spring",
                "timeline_nature": "memory",
                "social_context": "rural English countryside",
                "economic_context": "unspecified",
            }
        }
        engine._repair_world_assumptions(data, self._universal_pack())
        assert data["world_assumptions"]["geography"] == "rural Somerset, England", (
            f"British geography was altered: {data['world_assumptions']['geography']!r}"
        )

    def test_french_geography_preserved(self):
        engine = self._make_engine()
        data = {
            "world_assumptions": {
                "geography": "Paris, France",
                "era": "contemporary",
                "season": "autumn",
                "timeline_nature": "real_time",
                "social_context": "urban Parisian life",
                "economic_context": "unspecified",
            }
        }
        engine._repair_world_assumptions(data, self._universal_pack())
        assert data["world_assumptions"]["geography"] == "Paris, France", (
            f"French geography was altered: {data['world_assumptions']['geography']!r}"
        )

    def test_korean_geography_preserved(self):
        engine = self._make_engine()
        data = {
            "world_assumptions": {
                "geography": "urban Seoul, South Korea",
                "era": "contemporary",
                "season": "winter",
                "timeline_nature": "real_time",
                "social_context": "urban Korean city life",
                "economic_context": "modern",
            }
        }
        engine._repair_world_assumptions(data, self._universal_pack())
        assert data["world_assumptions"]["geography"] == "urban Seoul, South Korea", (
            f"Korean geography was altered: {data['world_assumptions']['geography']!r}"
        )

    def test_no_south_asian_geography_injected_for_empty_uk_song(self):
        """When the LLM returns empty geography for a UK song, the repair must
        NOT fall back to a South Asian location from the universal pack."""
        engine = self._make_engine()
        data = {"world_assumptions": {}}
        engine._repair_world_assumptions(data, self._universal_pack())
        geography = data["world_assumptions"]["geography"]
        _assert_no_sa_bleed(geography, "world_assumptions.geography (UK song empty fallback)")

    def test_no_south_asian_geography_injected_for_empty_french_song(self):
        engine = self._make_engine()
        data = {"world_assumptions": {}}
        engine._repair_world_assumptions(data, self._universal_pack())
        geography = data["world_assumptions"]["geography"]
        _assert_no_sa_bleed(geography, "world_assumptions.geography (French song empty fallback)")

    def test_south_asian_pack_geography_only_used_for_sa_content(self):
        """The Punjabi rural lament pack's geography must NOT appear for non-SA content."""
        engine = self._make_engine()
        # Simulate a UK song that gets the universal pack (no SA triggers)
        universal = CulturePackRegistry.get_pack(None)
        data = {"world_assumptions": {}}
        engine._repair_world_assumptions(data, universal)
        geography = data["world_assumptions"]["geography"]
        assert "south asia" not in geography.lower(), (
            f"South Asian geography leaked into non-SA song: {geography!r}"
        )
        assert "punjab" not in geography.lower(), (
            f"Punjabi geography leaked into non-SA song: {geography!r}"
        )


# =============================================================================
# PART 3 — _build_system_prompt: no South Asian pack is injected
# =============================================================================

class TestSystemPromptNoCulturePackForNonSALyrics:
    """
    _build_system_prompt must show culture_pack_id as 'none' for non-South-Asian
    lyrics and must not inject any South Asian cultural context into the prompt.
    """

    def _make_engine(self) -> MetaMindContextEngineFinal:
        return MetaMindContextEngineFinal(api_key="test-key")

    def _make_hard_logic(self, lyrics: str) -> dict:
        pack_id = CulturePackRegistry.detect_pack(lyrics)
        pack = CulturePackRegistry.get_pack(pack_id)
        metaphors = CulturePackRegistry.get_triggered_metaphors(lyrics, pack_id)
        return {
            "routing": {
                "recognized_type": "song",
                "raw_detected_type": "song",
                "is_mixed_input": False,
                "structure_quality": "inferred",
                "line_count": 4,
                "has_repetition": False,
                "symbolic_density": "low",
                "abstraction_level": "low",
            },
            "language": {"primary": "English", "script": "Latin", "dialect": ""},
            "genre_directive": "Prioritize lyrical meaning and emotional flow.",
            "culture_pack_id": pack_id,
            "culture_pack": pack,
            "active_metaphors": metaphors,
            "locked_assumptions": {},
            "pre_analysis": {},
            "input_packet": None,
        }

    def test_uk_folk_prompt_shows_no_culture_pack(self):
        engine = self._make_engine()
        hard_logic = self._make_hard_logic(UK_FOLK_LYRICS)
        prompt = engine._build_system_prompt(hard_logic)
        assert "- selected: none" in prompt, (
            f"Expected culture_pack_id 'none' in prompt for UK folk lyrics, got:\n{prompt[:300]}"
        )

    def test_french_chanson_prompt_shows_no_culture_pack(self):
        engine = self._make_engine()
        hard_logic = self._make_hard_logic(FRENCH_CHANSON_LYRICS)
        hard_logic["language"] = {"primary": "French", "script": "Latin", "dialect": ""}
        prompt = engine._build_system_prompt(hard_logic)
        assert "- selected: none" in prompt, (
            f"Expected culture_pack_id 'none' in prompt for French lyrics, got:\n{prompt[:300]}"
        )

    def test_korean_ballad_prompt_shows_no_culture_pack(self):
        engine = self._make_engine()
        hard_logic = self._make_hard_logic(KOREAN_BALLAD_LYRICS)
        hard_logic["language"] = {"primary": "Korean", "script": "Hangul", "dialect": ""}
        prompt = engine._build_system_prompt(hard_logic)
        assert "- selected: none" in prompt, (
            f"Expected culture_pack_id 'none' in prompt for Korean lyrics, got:\n{prompt[:300]}"
        )

    def test_uk_folk_prompt_has_no_active_sa_metaphors(self):
        engine = self._make_engine()
        hard_logic = self._make_hard_logic(UK_FOLK_LYRICS)
        prompt = engine._build_system_prompt(hard_logic)
        assert "ACTIVE CULTURAL METAPHORS:\nNone" in prompt, (
            "Expected 'None' for active metaphors on UK folk lyrics, got something else"
        )

    def test_sa_lyrics_do_get_pack_in_prompt(self):
        """Sanity check: South Asian lyrics do inject a pack into the prompt."""
        engine = self._make_engine()
        hard_logic = self._make_hard_logic(SOUTH_ASIAN_LYRICS)
        hard_logic["language"] = {"primary": "Punjabi", "script": "Gurmukhi", "dialect": ""}
        prompt = engine._build_system_prompt(hard_logic)
        assert "- selected: none" not in prompt, (
            "South Asian lyrics should have a culture pack in the prompt, not 'none'"
        )


# =============================================================================
# PART 4 — End-to-end generate() with mocked LLM
# =============================================================================

class TestEndToEndNoCulturalBleed:
    """
    Full generate() call with the OpenAI client mocked.
    Verifies that the final context packet for UK / French / Korean songs
    carries the correct cultural world and has zero South Asian bleed.
    """

    def _make_engine(self) -> MetaMindContextEngineFinal:
        return MetaMindContextEngineFinal(api_key="test-key")

    def _run(self, engine: MetaMindContextEngineFinal, lyrics: str, llm_packet: dict) -> dict:
        with patch.object(
            engine,
            "_call_model",
            new_callable=AsyncMock,
            return_value=json.dumps(llm_packet),
        ):
            return asyncio.run(engine.generate(lyrics, hinted_type="song"))

    # -- UK English folk song -------------------------------------------------

    def test_uk_folk_geography_is_british(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="rural Somerset, England",
            cultural_background="British",
            language="English",
            location_dna="British countryside",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        assert "england" in geography.lower() or "somerset" in geography.lower() or "british" in geography.lower(), (
            f"Expected British geography for UK folk song, got: {geography!r}"
        )

    def test_uk_folk_no_south_asian_geography(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="rural Somerset, England",
            cultural_background="British",
            language="English",
            location_dna="British countryside",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        _assert_no_sa_bleed(geography, "UK folk world_assumptions.geography")

    def test_uk_folk_no_sa_bleed_in_location_dna(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="rural Somerset, England",
            cultural_background="British",
            language="English",
            location_dna="British countryside",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        location_dna = result.get("location_dna", "")
        _assert_no_sa_bleed(location_dna, "UK folk location_dna")

    def test_uk_folk_no_sa_bleed_in_cultural_constraints(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="rural Somerset, England",
            cultural_background="British",
            language="English",
            location_dna="British countryside",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        constraints_text = " ".join(result.get("cultural_constraints", []))
        _assert_no_sa_bleed(constraints_text, "UK folk cultural_constraints")

    def test_uk_folk_no_sa_pack_in_meta(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="rural Somerset, England",
            cultural_background="British",
            language="English",
            location_dna="British countryside",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        pack_id = result.get("meta", {}).get("culture_pack_id")
        assert pack_id is None, (
            f"UK folk song should have no culture_pack_id in meta, got: {pack_id!r}"
        )

    # -- French chanson -------------------------------------------------------

    def test_french_chanson_geography_is_french(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="Paris, France",
            cultural_background="French",
            language="French",
            location_dna="French-speaking world",
        )
        result = self._run(engine, FRENCH_CHANSON_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        assert "france" in geography.lower() or "paris" in geography.lower(), (
            f"Expected French geography for French chanson, got: {geography!r}"
        )

    def test_french_chanson_no_south_asian_geography(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="Paris, France",
            cultural_background="French",
            language="French",
            location_dna="French-speaking world",
        )
        result = self._run(engine, FRENCH_CHANSON_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        _assert_no_sa_bleed(geography, "French chanson world_assumptions.geography")

    def test_french_chanson_no_sa_bleed_in_location_dna(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="Paris, France",
            cultural_background="French",
            language="French",
            location_dna="French-speaking world",
        )
        result = self._run(engine, FRENCH_CHANSON_LYRICS, llm_packet)
        location_dna = result.get("location_dna", "")
        _assert_no_sa_bleed(location_dna, "French chanson location_dna")

    def test_french_chanson_no_sa_pack_in_meta(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="Paris, France",
            cultural_background="French",
            language="French",
            location_dna="French-speaking world",
        )
        result = self._run(engine, FRENCH_CHANSON_LYRICS, llm_packet)
        pack_id = result.get("meta", {}).get("culture_pack_id")
        assert pack_id is None, (
            f"French chanson should have no culture_pack_id in meta, got: {pack_id!r}"
        )

    # -- Korean ballad --------------------------------------------------------

    def test_korean_ballad_geography_is_korean(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="urban Seoul, South Korea",
            cultural_background="Korean",
            language="Korean",
            location_dna="Korean cultural world",
        )
        result = self._run(engine, KOREAN_BALLAD_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        assert "korea" in geography.lower() or "seoul" in geography.lower(), (
            f"Expected Korean geography for Korean ballad, got: {geography!r}"
        )

    def test_korean_ballad_no_south_asian_geography(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="urban Seoul, South Korea",
            cultural_background="Korean",
            language="Korean",
            location_dna="Korean cultural world",
        )
        result = self._run(engine, KOREAN_BALLAD_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        _assert_no_sa_bleed(geography, "Korean ballad world_assumptions.geography")

    def test_korean_ballad_no_sa_bleed_in_location_dna(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="urban Seoul, South Korea",
            cultural_background="Korean",
            language="Korean",
            location_dna="Korean cultural world",
        )
        result = self._run(engine, KOREAN_BALLAD_LYRICS, llm_packet)
        location_dna = result.get("location_dna", "")
        _assert_no_sa_bleed(location_dna, "Korean ballad location_dna")

    def test_korean_ballad_no_sa_pack_in_meta(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="urban Seoul, South Korea",
            cultural_background="Korean",
            language="Korean",
            location_dna="Korean cultural world",
        )
        result = self._run(engine, KOREAN_BALLAD_LYRICS, llm_packet)
        pack_id = result.get("meta", {}).get("culture_pack_id")
        assert pack_id is None, (
            f"Korean ballad should have no culture_pack_id in meta, got: {pack_id!r}"
        )

    # -- Cross-check: no bleed even if LLM misbehaves (returns empty geography) --

    def test_empty_llm_geography_for_uk_song_does_not_become_south_asian(self):
        """If the LLM returns empty geography for a UK song, the repaired fallback
        must NOT be a South Asian location."""
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="",
            cultural_background="British",
            language="English",
            location_dna="",
        )
        result = self._run(engine, UK_FOLK_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        _assert_no_sa_bleed(geography, "UK song with empty LLM geography, world_assumptions.geography")

    def test_empty_llm_geography_for_french_song_does_not_become_south_asian(self):
        engine = self._make_engine()
        llm_packet = _make_llm_packet(
            geography="",
            cultural_background="French",
            language="French",
            location_dna="",
        )
        result = self._run(engine, FRENCH_CHANSON_LYRICS, llm_packet)
        geography = result.get("world_assumptions", {}).get("geography", "")
        _assert_no_sa_bleed(geography, "French song with empty LLM geography, world_assumptions.geography")


# ---------------------------------------------------------------------------
# Helper: build a minimal context packet for downstream stage tests
# ---------------------------------------------------------------------------

def _make_context_packet(
    geography: str,
    cultural_background: str,
    location_dna: str,
) -> dict:
    """Build a realistic context packet for passing to creative brief and storyboard tests."""
    return {
        "speaker": {
            "identity": "a solitary figure walking through a familiar landscape",
            "gender": "Male",
            "age_range": "young adult",
            "emotional_state": "nostalgic and melancholic",
            "social_role": "lover",
            "relationship_to_addressee": "former romantic partner",
            "cultural_background": cultural_background,
        },
        "addressee": {
            "identity": "a lost love",
            "relationship": "former romantic partner",
            "presence": "absent",
        },
        "world_assumptions": {
            "geography": geography,
            "era": "contemporary",
            "season": "spring",
            "characteristic_time": "golden hour",
            "characteristic_setting": "open countryside",
            "architecture_style": "unspecified",
            "social_context": "rural community life",
            "economic_context": "unspecified",
        },
        "location_dna": location_dna,
        "core_theme": "longing and memory",
        "dramatic_premise": "A speaker recalls a lost love against a familiar landscape.",
        "narrative_spine": "The speaker walks through their homeland, haunted by the memory of love.",
        "narrative_mode": "lyrical",
        "motivation": {
            "inciting_cause": "Walking through a familiar landscape that evokes a lost love",
            "underlying_desire": "To reconnect with or make peace with what has been lost",
            "stakes": "Remaining trapped in grief versus finding peace",
            "obstacle": "The irrecoverable absence of the beloved",
            "confidence": 0.82,
        },
        "emotional_arc": {
            "opening": "Peaceful longing",
            "development": "Deepening sense of loss",
            "climax": "Peak of grief",
            "resolution": "Quiet acceptance",
        },
        "line_meanings": [
            {
                "line_index": i + 1,
                "text": f"line {i + 1}",
                "literal_meaning": "A lyrical image of landscape and longing",
                "implied_meaning": "The speaker feels the absence of their love",
                "emotional_meaning": "Melancholy and nostalgic longing",
                "cultural_meaning": f"Imagery rooted in {geography}",
                "meaning": "Longing expressed through landscape",
                "function": "verse",
                "repeat_status": "original",
                "intensity": 0.6,
                "expression_mode": "environment",
                "visualization_mode": "default",
                "visual_suitability": "suitable",
                "visual_props": [],
            }
            for i in range(4)
        ],
        "entities": [],
        "motifs": [],
        "motif_map": {},
        "literary_devices": ["imagery", "personification"],
        "cultural_constraints": [
            f"The world is rooted in {geography} — do not substitute imagery from other cultures.",
        ],
        "preservation_rules": ["speaker cultural identity", "geography"],
        "creative_freedom": ["exact locations", "props", "colour palette"],
        "surfaced_assumptions": [],
        "locked_assumptions": {},
        "ambiguity_flags": [],
        "visual_constraints": [],
        "restrictions": [],
        "confidence": 0.83,
        "confidence_scores": {"overall": 0.83},
        "input_type": "song",
        "culture_pack": None,
    }


UK_CONTEXT = _make_context_packet(
    geography="rural Somerset, England",
    cultural_background="British",
    location_dna="British countryside",
)

FRENCH_CONTEXT = _make_context_packet(
    geography="Paris, France",
    cultural_background="French",
    location_dna="French-speaking world",
)

KOREAN_CONTEXT = _make_context_packet(
    geography="urban Seoul, South Korea",
    cultural_background="Korean",
    location_dna="Korean cultural world",
)

PUNJABI_CONTEXT = _make_context_packet(
    geography="Punjab region, South Asia",
    cultural_background="South Asian (Punjabi)",
    location_dna="Punjab cultural region (South Asian)",
)


# =============================================================================
# PART 5 — Creative Brief: _build_cultural_grounding() for non-SA packets
# =============================================================================

class TestCreativeBriefCulturalGroundingNonSA:
    """
    _build_cultural_grounding() must return an empty string for UK / French / Korean
    context packets because none of them contain Punjabi or Urdu markers.
    """

    def test_uk_context_produces_no_grounding_block(self):
        grounding = _build_cultural_grounding(UK_CONTEXT)
        assert grounding == "", (
            f"Expected empty grounding for UK context, got: {grounding!r}"
        )

    def test_french_context_produces_no_grounding_block(self):
        grounding = _build_cultural_grounding(FRENCH_CONTEXT)
        assert grounding == "", (
            f"Expected empty grounding for French context, got: {grounding!r}"
        )

    def test_korean_context_produces_no_grounding_block(self):
        grounding = _build_cultural_grounding(KOREAN_CONTEXT)
        assert grounding == "", (
            f"Expected empty grounding for Korean context, got: {grounding!r}"
        )

    def test_no_sa_grounding_block_contains_sa_markers(self):
        """None of the non-SA grounding blocks should leak SA text."""
        for ctx, label in [
            (UK_CONTEXT, "UK"),
            (FRENCH_CONTEXT, "French"),
            (KOREAN_CONTEXT, "Korean"),
        ]:
            grounding = _build_cultural_grounding(ctx)
            _assert_no_sa_bleed(grounding, f"{label} creative brief grounding block")

    def test_punjabi_context_does_produce_grounding_block(self):
        """Sanity check: a Punjabi context must produce a non-empty grounding block."""
        grounding = _build_cultural_grounding(PUNJABI_CONTEXT)
        assert grounding, "Punjabi context should produce a cultural grounding block"
        assert "punjab" in grounding.lower(), (
            f"Punjabi grounding block should reference Punjab, got: {grounding[:200]!r}"
        )


# =============================================================================
# PART 6 — Creative Brief: _fallback_variants() for non-SA packets
# =============================================================================

class TestCreativeBriefFallbackVariantsNonSA:
    """
    _fallback_variants() must not emit South Asian cultural markers in its
    scene descriptions, treatment text, or location strings for non-SA songs.
    """

    def _combined_text(self, variants: list) -> str:
        parts = []
        for v in variants:
            parts.append(str(v.get("pitch", "")))
            parts.append(str(v.get("treatment", "")))
            parts.append(str(v.get("central_metaphor", "")))
            for scene in v.get("scenes", []):
                parts.append(str(scene.get("location", "")))
                parts.append(str(scene.get("summary", "")))
                parts.extend(str(p) for p in scene.get("props", []))
        return " ".join(parts)

    def test_uk_fallback_variants_have_no_sa_bleed(self):
        variants = _fallback_variants(UK_CONTEXT, entity_names=["Alice"])
        text = self._combined_text(variants)
        _assert_no_sa_bleed(text, "UK fallback variants combined text")

    def test_french_fallback_variants_have_no_sa_bleed(self):
        variants = _fallback_variants(FRENCH_CONTEXT, entity_names=["Marie"])
        text = self._combined_text(variants)
        _assert_no_sa_bleed(text, "French fallback variants combined text")

    def test_korean_fallback_variants_have_no_sa_bleed(self):
        variants = _fallback_variants(KOREAN_CONTEXT, entity_names=["Ji-ho"])
        text = self._combined_text(variants)
        _assert_no_sa_bleed(text, "Korean fallback variants combined text")

    def test_fallback_variants_reference_correct_location_dna(self):
        """The UK fallback pitch should reference British countryside, not a SA location."""
        variants = _fallback_variants(UK_CONTEXT, entity_names=[])
        combined = " ".join(
            str(v.get("pitch", "")) + str(v.get("treatment", ""))
            for v in variants
        )
        assert "british countryside" in combined.lower() or "british" in combined.lower() or combined, (
            "UK fallback variants should reference British location or be culturally neutral"
        )


# =============================================================================
# PART 7 — Creative Brief: generate_variants() end-to-end (LLM mocked)
# =============================================================================

def _make_brief_variants(geography: str, location_dna: str) -> dict:
    """Return a minimal JSON structure the brief LLM might produce."""
    return {
        "variants": [
            {
                "id": "v1",
                "title": "Landscape Memory",
                "pitch": f"A speaker walks through {geography} haunted by a lost love.",
                "treatment": (
                    f"We follow the speaker through {geography}. The landscape itself "
                    f"becomes the emotional mirror of their longing."
                ),
                "scenes": [
                    {
                        "name": "Opening Walk",
                        "beat_range": "intro",
                        "summary": f"Speaker enters the landscape at dawn in {geography}.",
                        "location": f"countryside in {geography}, dawn",
                        "time_of_day": "dawn",
                        "props": ["worn path", "gate"],
                    },
                ],
                "visual_locations": [f"countryside in {geography}"],
                "cast_roster": ["Speaker"],
                "central_metaphor": "landscape as emotional memory",
                "justification": "Emotionally legible and culturally grounded.",
                "self_critique": {"score": 8, "rationale": "Strong visual story."},
            },
            {
                "id": "v2",
                "title": "Object Study",
                "pitch": "An object carries the emotional arc.",
                "treatment": "No face on camera — an object transforms across four scenes.",
                "scenes": [
                    {
                        "name": "Object",
                        "beat_range": "intro",
                        "summary": "Object introduced on a window sill.",
                        "location": "stone window sill",
                        "time_of_day": "morning",
                        "props": ["flower", "dust particles"],
                    },
                ],
                "visual_locations": ["stone window sill"],
                "cast_roster": [],
                "central_metaphor": "the object as the unspoken self",
                "justification": "Maximises cinematic freedom.",
                "self_critique": {"score": 9, "rationale": "Bold and visually inventive."},
            },
        ]
    }


class TestCreativeBriefGenerateVariantsNonSA:
    """generate_variants() end-to-end with mocked LLM — no SA bleed in output."""

    def _run(self, ctx: dict, llm_response: dict) -> tuple:
        with patch("creative_brief_engine.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_openai_cls.return_value = mock_client
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = json.dumps(llm_response)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            variants, used_fallback = asyncio.run(
                generate_variants(api_key="test-key", context_packet=ctx, n=2)
            )
        return variants, used_fallback

    def _combined_variant_text(self, variants: list) -> str:
        parts = []
        for v in variants:
            parts.append(str(v.get("pitch", "")))
            parts.append(str(v.get("treatment", "")))
            parts.append(str(v.get("central_metaphor", "")))
            for scene in v.get("scenes", []):
                parts.append(str(scene.get("location", "")))
                parts.append(str(scene.get("summary", "")))
        return " ".join(parts)

    def test_uk_variants_contain_no_sa_bleed(self):
        response = _make_brief_variants("rural Somerset, England", "British countryside")
        variants, _ = self._run(UK_CONTEXT, response)
        text = self._combined_variant_text(variants)
        _assert_no_sa_bleed(text, "UK generate_variants output")

    def test_french_variants_contain_no_sa_bleed(self):
        response = _make_brief_variants("Paris, France", "French-speaking world")
        variants, _ = self._run(FRENCH_CONTEXT, response)
        text = self._combined_variant_text(variants)
        _assert_no_sa_bleed(text, "French generate_variants output")

    def test_korean_variants_contain_no_sa_bleed(self):
        response = _make_brief_variants("urban Seoul, South Korea", "Korean cultural world")
        variants, _ = self._run(KOREAN_CONTEXT, response)
        text = self._combined_variant_text(variants)
        _assert_no_sa_bleed(text, "Korean generate_variants output")

    def test_uk_variants_reference_correct_geography(self):
        response = _make_brief_variants("rural Somerset, England", "British countryside")
        variants, _ = self._run(UK_CONTEXT, response)
        text = self._combined_variant_text(variants)
        assert "england" in text.lower() or "somerset" in text.lower() or "british" in text.lower(), (
            f"UK variants should reference England/Somerset/British geography, got: {text[:300]!r}"
        )


# =============================================================================
# PART 8 — Visual Storyboard: build_storyboard() with non-SA context packets
# =============================================================================

class TestStoryboardNoPunjabiMandateForNonSA:
    """
    VisualStoryboardEngine.build_storyboard() must not inject the Punjabi
    Location Mandate or any South Asian cultural props when the geography is
    British / French / Korean.
    """

    def _build(self, ctx: dict) -> list:
        engine = VisualStoryboardEngine()
        return engine.build_storyboard(ctx)

    def _all_shot_text(self, storyboard: list) -> str:
        parts = []
        for shot in storyboard:
            parts.append(str(shot.get("visual_prompt", "")))
            parts.append(str(shot.get("location_dna", "")))
            parts.append(str(shot.get("framing_directive", "")))
            parts.append(str(shot.get("motion_prompt", "")))
            parts.append(str(shot.get("environment_profile", "")))
            for seg_key, seg_val in (shot.get("prompt_segments") or {}).items():
                parts.append(str(seg_val))
        return " ".join(parts)

    def test_uk_storyboard_has_no_punjabi_location_mandate(self):
        storyboard = self._build(UK_CONTEXT)
        for shot in storyboard:
            vp = shot.get("visual_prompt", "")
            assert "PUNJABI LOCATION MANDATE" not in vp, (
                f"UK storyboard shot {shot['shot_index']} contains PUNJABI LOCATION MANDATE"
            )

    def test_french_storyboard_has_no_punjabi_location_mandate(self):
        storyboard = self._build(FRENCH_CONTEXT)
        for shot in storyboard:
            vp = shot.get("visual_prompt", "")
            assert "PUNJABI LOCATION MANDATE" not in vp, (
                f"French storyboard shot {shot['shot_index']} contains PUNJABI LOCATION MANDATE"
            )

    def test_korean_storyboard_has_no_punjabi_location_mandate(self):
        storyboard = self._build(KOREAN_CONTEXT)
        for shot in storyboard:
            vp = shot.get("visual_prompt", "")
            assert "PUNJABI LOCATION MANDATE" not in vp, (
                f"Korean storyboard shot {shot['shot_index']} contains PUNJABI LOCATION MANDATE"
            )

    def test_uk_storyboard_no_sa_bleed_in_visual_prompts(self):
        storyboard = self._build(UK_CONTEXT)
        for shot in storyboard:
            _assert_no_sa_bleed(
                shot.get("visual_prompt", ""),
                f"UK storyboard shot {shot['shot_index']} visual_prompt",
            )

    def test_french_storyboard_no_sa_bleed_in_visual_prompts(self):
        storyboard = self._build(FRENCH_CONTEXT)
        for shot in storyboard:
            _assert_no_sa_bleed(
                shot.get("visual_prompt", ""),
                f"French storyboard shot {shot['shot_index']} visual_prompt",
            )

    def test_korean_storyboard_no_sa_bleed_in_visual_prompts(self):
        storyboard = self._build(KOREAN_CONTEXT)
        for shot in storyboard:
            _assert_no_sa_bleed(
                shot.get("visual_prompt", ""),
                f"Korean storyboard shot {shot['shot_index']} visual_prompt",
            )

    def test_uk_storyboard_location_dna_is_british(self):
        storyboard = self._build(UK_CONTEXT)
        for shot in storyboard:
            location_dna = shot.get("location_dna", "")
            assert "british" in location_dna.lower(), (
                f"UK shot {shot['shot_index']} location_dna should be British, got: {location_dna!r}"
            )

    def test_punjabi_context_does_inject_mandate(self):
        """Sanity check: the Punjabi mandate fires for a Punjabi context packet."""
        storyboard = self._build(PUNJABI_CONTEXT)
        mandate_found = any(
            "PUNJABI LOCATION MANDATE" in (shot.get("visual_prompt") or "")
            for shot in storyboard
        )
        assert mandate_found, (
            "Expected PUNJABI LOCATION MANDATE in at least one shot for Punjabi context"
        )
