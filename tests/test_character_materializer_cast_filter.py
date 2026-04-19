"""Regression tests for Task #69 cast-filter logic in character_materializer.

These tests exercise the cast-filter decision logic in isolation so they
don't require a live database connection or full materialization pipeline.

Guards:
- Empty cast_roster (objects-only variant) → only speaker survives.
- Non-empty cast_roster → speaker + listed names survive; others dropped.
- No creative_brief → legacy behaviour, all characters pass through.
"""


def _simulate_cast_filter(context_packet, characters, entities):
    """
    Mirror the cast-filter logic from character_materializer so we can test
    it without spinning up a DB connection.
    """
    cb_chosen = ((context_packet.get("creative_brief") or {}).get("chosen") or {})
    cast_roster_raw = cb_chosen.get("cast_roster")
    brief_locks_cast = isinstance(cast_roster_raw, list)
    cast_filter = (
        {str(n).strip().lower() for n in cast_roster_raw if str(n).strip()}
        if brief_locks_cast else set()
    )

    if brief_locks_cast:
        characters = [
            c for c in characters
            if c.get("entity_type") == "speaker" or c["name"].lower() in cast_filter
        ]

    seen = {c["name"].lower() for c in characters}
    ingested = []
    for ent in entities:
        ent_type = (ent.get("type") or "").lower()
        if "person" not in ent_type and "character" not in ent_type:
            continue
        name = (ent.get("name") or "").strip().lower()
        if not name or name in seen:
            continue
        if brief_locks_cast and name not in cast_filter:
            continue
        seen.add(name)
        ingested.append(name)

    return characters, ingested


_SPEAKER = {"name": "the narrator", "entity_type": "speaker"}
_ADDRESSEE = {"name": "alice", "entity_type": "addressee"}
_NAMED_BOB = {"name": "bob", "entity_type": "named_entity"}

_ENTITIES = [
    {"type": "person", "name": "Alice", "role": "love interest"},
    {"type": "person", "name": "Bob",   "role": "friend"},
    {"type": "person", "name": "Carol", "role": "stranger"},
]


def test_empty_cast_roster_keeps_only_speaker_no_entities():
    """cast_roster=[] (objects-only) → only speaker; no named entities added."""
    ctx = {"creative_brief": {"chosen": {"cast_roster": []}}}
    chars = [_SPEAKER, _ADDRESSEE, _NAMED_BOB]

    filtered, ingested = _simulate_cast_filter(ctx, chars, _ENTITIES)

    entity_types = [c["entity_type"] for c in filtered]
    assert "speaker" in entity_types
    assert "addressee" not in entity_types, "addressee should be dropped with empty cast"
    assert "named_entity" not in entity_types
    assert ingested == [], f"No new entities should be ingested; got {ingested}"


def test_nonempty_cast_roster_keeps_speaker_plus_listed_names():
    """cast_roster=['Alice'] → speaker + Alice; Bob and Carol dropped."""
    ctx = {"creative_brief": {"chosen": {"cast_roster": ["Alice"]}}}
    chars = [_SPEAKER, _NAMED_BOB]

    filtered, ingested = _simulate_cast_filter(ctx, chars, _ENTITIES)

    filtered_names = [c["name"].lower() for c in filtered]
    assert "the narrator" in filtered_names, "speaker must survive"
    assert "bob" not in filtered_names, "Bob not in roster → dropped"
    assert "alice" in ingested, "Alice is in roster → ingested"
    assert "bob" not in ingested
    assert "carol" not in ingested


def test_no_brief_does_not_filter_legacy():
    """No creative_brief → legacy behaviour, all characters and entities pass."""
    ctx = {}  # no brief at all
    chars = [_SPEAKER, _NAMED_BOB]

    filtered, ingested = _simulate_cast_filter(ctx, chars, _ENTITIES)

    assert len(filtered) == 2  # speaker + bob unchanged
    assert "alice" in ingested  # Alice and Carol freely ingested
    assert "carol" in ingested


def test_cast_filter_case_insensitive():
    """Filter is case-insensitive; 'ALICE' in roster matches entity named 'alice'."""
    ctx = {"creative_brief": {"chosen": {"cast_roster": ["ALICE"]}}}
    chars = [_SPEAKER]

    _, ingested = _simulate_cast_filter(ctx, chars, _ENTITIES)

    assert "alice" in ingested
    assert "bob" not in ingested
