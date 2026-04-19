"""Character Materializer (MetaMind v3.1) — reads a context_packet and writes
concrete character rows to the `characters` table.

Idempotent: rows are matched by (project_id, name). Existing rows have their
NULL fields back-filled; non-NULL fields are left untouched so user edits
survive a pipeline re-run.

Sources:
  - `speaker` dict   → PRIMARY character (the main vocal subject), now carrying
                       ethnicity / complexion / wardrobe / grooming.
  - `addressee` dict → SECONDARY character if it has a non-trivial identity.
  - `entities` list  → type contains "person" / "character" → named characters.
  - `location_dna` + `world_assumptions.cultural_dna` → cultural grounding.
"""
from __future__ import annotations

import logging
import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

logger = logging.getLogger(__name__)


_VAGUE = {"", "unclear", "unspecified", "unknown", "none", "n/a"}


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def _norm(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_vague(value) -> bool:
    return _norm(value).lower() in _VAGUE


def _upsert_character(cur, project_id: str, ch: dict) -> int:
    """Atomic UPSERT (idempotent under concurrency); returns the row id."""
    cur.execute(
        """
        INSERT INTO characters
            (project_id, name, role, entity_type,
             appearance, age_range, cultural_notes, emotional_notes,
             gender, ethnicity, complexion, wardrobe, grooming, relationship,
             metadata)
        VALUES (%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s)
        ON CONFLICT (project_id, name) DO UPDATE
           SET role            = COALESCE(characters.role,            EXCLUDED.role),
               entity_type     = COALESCE(characters.entity_type,     EXCLUDED.entity_type),
               appearance      = COALESCE(characters.appearance,      EXCLUDED.appearance),
               age_range       = COALESCE(characters.age_range,       EXCLUDED.age_range),
               cultural_notes  = COALESCE(characters.cultural_notes,  EXCLUDED.cultural_notes),
               emotional_notes = COALESCE(characters.emotional_notes, EXCLUDED.emotional_notes),
               gender          = COALESCE(characters.gender,          EXCLUDED.gender),
               ethnicity       = COALESCE(characters.ethnicity,       EXCLUDED.ethnicity),
               complexion      = COALESCE(characters.complexion,      EXCLUDED.complexion),
               wardrobe        = COALESCE(characters.wardrobe,        EXCLUDED.wardrobe),
               grooming        = COALESCE(characters.grooming,        EXCLUDED.grooming),
               relationship    = COALESCE(characters.relationship,    EXCLUDED.relationship)
        RETURNING id
        """,
        (
            project_id,
            ch["name"],
            ch.get("role"),
            ch["entity_type"],
            ch.get("appearance"),
            ch.get("age_range"),
            ch.get("cultural_notes"),
            ch.get("emotional_notes"),
            ch.get("gender"),
            ch.get("ethnicity"),
            ch.get("complexion"),
            ch.get("wardrobe"),
            ch.get("grooming"),
            ch.get("relationship"),
            ch.get("metadata"),
        ),
    )
    return cur.fetchone()["id"]


def _build_appearance(
    gender: str,
    age_range: str,
    ethnicity: str,
    complexion: str,
    wardrobe: str,
    grooming: str,
    location_dna: str,
) -> str | None:
    parts = []
    # Lead with the gender constraint so downstream image-generation prompts
    # always see "<gender>-presenting, ..." as the first descriptor — this
    # prevents the image model from drifting to the wrong-gender protagonist.
    g = _norm(gender).lower()
    if g in ("male", "female"):
        parts.append(f"{g}-presenting")
    elif g == "mixed":
        # Mixed/duet projects spawn a separate row per gender; the per-row
        # gender will be the concrete one. This branch is just defensive.
        parts.append("mixed-gender ensemble")
    elif not _is_vague(gender):
        parts.append(gender.capitalize())
    if not _is_vague(age_range):
        parts.append(age_range)
    if not _is_vague(ethnicity):
        parts.append(ethnicity)
    if not _is_vague(complexion):
        parts.append(f"{complexion} complexion")
    if not _is_vague(wardrobe):
        parts.append(f"wardrobe: {wardrobe}")
    if not _is_vague(grooming):
        parts.append(f"grooming: {grooming}")
    if not parts and not _is_vague(location_dna) and location_dna.lower() != "universal":
        parts.append(f"{location_dna} features")
    return ", ".join(parts) if parts else None


def _resolve_speaker_gender(speaker: dict, vocal_gender: str | None) -> str:
    """Resolve the final speaker gender by precedence:
       1. vocal_gender ("male" / "female" / "mixed") — hard fact from audio analysis
          (or the user's explicit override at the Audio Review screen).
       2. vocal_gender == "instrumental" — user explicitly said the track has no
          singer, so do NOT fall back to the LLM's lyric-only gender guess.
       3. speaker.gender from the context_packet (LLM-inferred from lyrics).
       4. "" (caller treats as Unspecified).
    """
    vg = _norm(vocal_gender).lower()
    if vg in ("male", "female", "mixed"):
        return vg.capitalize() if vg != "mixed" else "Mixed"
    if vg == "instrumental":
        return ""
    sp = _norm(speaker.get("gender") if isinstance(speaker, dict) else "")
    if sp and not _is_vague(sp):
        return sp
    return ""


def materialize_characters(
    project_id: str,
    context_packet: dict,
    vocal_gender: str | None = None,
) -> list[dict]:
    """Derive character rows from context_packet and persist them."""
    if not isinstance(context_packet, dict):
        context_packet = {}
    location_dna = _norm(context_packet.get("location_dna")) or "Universal"
    raw_world = context_packet.get("world_assumptions")
    world: dict = raw_world if isinstance(raw_world, dict) else {}
    cultural_dna = _norm(world.get("cultural_dna"))

    cultural_grounding = (
        location_dna if location_dna.lower() != "universal" else (cultural_dna or None)
    )

    raw_speaker = context_packet.get("speaker")
    speaker: dict = raw_speaker if isinstance(raw_speaker, dict) else {}
    raw_addressee = context_packet.get("addressee")
    addressee: dict = raw_addressee if isinstance(raw_addressee, dict) else {}
    raw_entities = context_packet.get("entities")
    entities: list = raw_entities if isinstance(raw_entities, list) else []

    characters: list[dict] = []

    # ---- PRIMARY: speaker ----
    if speaker:
        identity = _norm(speaker.get("identity")) or "Primary Speaker"
        # Vocal-gender (audio analysis or user override) wins over the LLM's
        # text-only inference. Falls back cleanly when no audio info exists.
        gender = _resolve_speaker_gender(speaker, vocal_gender)
        age_range = _norm(speaker.get("age_range"))
        emotional_state = _norm(speaker.get("emotional_state"))
        social_role = _norm(speaker.get("social_role"))
        ethnicity = _norm(speaker.get("ethnicity"))
        complexion = _norm(speaker.get("complexion"))
        wardrobe = _norm(speaker.get("wardrobe"))
        grooming = _norm(speaker.get("grooming"))
        relationship = _norm(speaker.get("relationship_to_addressee"))

        appearance = _build_appearance(
            gender, age_range, ethnicity, complexion, wardrobe, grooming, location_dna
        )

        characters.append({
            "name": identity,
            "role": social_role or "Primary speaker / protagonist",
            "entity_type": "speaker",
            "appearance": appearance,
            "age_range": age_range or None,
            "cultural_notes": cultural_grounding,
            "emotional_notes": emotional_state or None,
            "gender": gender or None,
            "ethnicity": ethnicity or None,
            "complexion": complexion or None,
            "wardrobe": wardrobe or None,
            "grooming": grooming or None,
            "relationship": relationship or None,
            "metadata": Json({"source": "speaker", "raw_speaker": speaker}),
        })

    # ---- SECONDARY: addressee (only if non-trivial) ----
    if addressee:
        ad_identity = _norm(addressee.get("identity"))
        ad_relationship = _norm(addressee.get("relationship"))
        ad_presence = _norm(addressee.get("presence"))
        if ad_identity and not _is_vague(ad_identity) and ad_identity.lower() not in {
            c["name"].lower() for c in characters
        }:
            characters.append({
                "name": ad_identity,
                "role": ad_relationship or "Addressee",
                "entity_type": "addressee",
                "appearance": None,
                "age_range": None,
                "cultural_notes": cultural_grounding,
                "emotional_notes": ad_presence or None,
                "gender": None,
                "ethnicity": None,
                "complexion": None,
                "wardrobe": None,
                "grooming": None,
                "relationship": ad_relationship or None,
                "metadata": Json({"source": "addressee", "raw_addressee": addressee}),
            })

    # ---- TERTIARY: named entities of type person/character ----
    # Task #69 — if the user locked a Creative Brief variant, honor its
    # cast_roster authoritatively:
    #   * Non-empty list → keep speaker + named entries in the roster.
    #   * Empty list     → keep only the speaker (objects-only treatment).
    #   * Key absent     → no filter, legacy behaviour unchanged.
    cb_chosen = ((context_packet.get("creative_brief") or {}).get("chosen") or {})
    cast_roster_raw = cb_chosen.get("cast_roster")
    brief_locks_cast: bool = isinstance(cast_roster_raw, list)
    cast_filter: set[str] = (
        {str(n).strip().lower() for n in cast_roster_raw if str(n).strip()}
        if brief_locks_cast else set()
    )

    if brief_locks_cast:
        # Always keep the speaker; only keep additional characters that the
        # chosen treatment explicitly names in its cast roster.  An empty
        # roster means the treatment calls for no people on screen (e.g. an
        # object-study variant) — discard all non-speaker entries.
        characters = [
            c for c in characters
            if c.get("entity_type") == "speaker" or c["name"].lower() in cast_filter
        ]

    seen_names: set[str] = {c["name"].lower() for c in characters}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ent_type = _norm(ent.get("type")).lower()
        if "person" not in ent_type and "character" not in ent_type:
            continue
        name = _norm(ent.get("name"))
        if not name or name.lower() in seen_names:
            continue
        # Task #69 — when a Creative Brief variant is locked, only ingest
        # named entities that the chosen cast roster calls for.  An empty
        # roster (objects-only treatment) blocks ALL new named entities here.
        if brief_locks_cast and name.lower() not in cast_filter:
            continue
        seen_names.add(name.lower())
        characters.append({
            "name": name,
            "role": _norm(ent.get("role")) or "Supporting character",
            "entity_type": "named_entity",
            "appearance": None,
            "age_range": None,
            "cultural_notes": cultural_grounding,
            "emotional_notes": None,
            "gender": None,
            "ethnicity": None,
            "complexion": None,
            "wardrobe": None,
            "grooming": None,
            "relationship": None,
            "metadata": Json({"source": "entity_map", "raw_entity": ent}),
        })

    if not characters:
        logger.info("No characters extracted for project %s", project_id)
        return []

    with _db() as conn, conn.cursor() as cur:
        for ch in characters:
            _upsert_character(cur, project_id, ch)
        conn.commit()
    logger.info("Upserted %d character(s) for project %s", len(characters), project_id)
    return characters
