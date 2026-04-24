"""Location Materializer (MetaMind v3.1) — reads a context_packet and writes
concrete location rows to the `locations` table.

Idempotent: rows are matched by (project_id, name). Existing rows have their
NULL fields back-filled; non-NULL fields are left untouched so user edits
survive a pipeline re-run.

Sources:
  - `location_dna`         → PRIMARY world DNA row (e.g. "Punjab cultural region").
  - `world_assumptions`    → enriches PRIMARY with geography, time_period,
                              architecture_style, weather_or_atmosphere,
                              social_layer, cultural_dna.
  - `entities` list        → type contains "place" / "location" → named locations.
  - `motifs` / `motif_map` → atmospheric notes on the primary location.
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


def _upsert_location(cur, project_id: str, loc: dict) -> int:
    """Atomic UPSERT (idempotent under concurrency); returns the row id."""
    cur.execute(
        """
        INSERT INTO locations
            (project_id, name, description, time_of_day,
             mood, cultural_notes, visual_details, entity_type,
             geography, time_period, architecture_style,
             weather_or_atmosphere, social_layer, cultural_dna,
             metadata)
        VALUES (%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s)
        ON CONFLICT (project_id, name) DO UPDATE
           SET description           = COALESCE(locations.description,           EXCLUDED.description),
               time_of_day           = COALESCE(locations.time_of_day,           EXCLUDED.time_of_day),
               mood                  = COALESCE(locations.mood,                  EXCLUDED.mood),
               cultural_notes        = COALESCE(locations.cultural_notes,        EXCLUDED.cultural_notes),
               visual_details        = COALESCE(locations.visual_details,        EXCLUDED.visual_details),
               entity_type           = COALESCE(locations.entity_type,           EXCLUDED.entity_type),
               geography             = COALESCE(locations.geography,             EXCLUDED.geography),
               time_period           = COALESCE(locations.time_period,           EXCLUDED.time_period),
               architecture_style    = COALESCE(locations.architecture_style,    EXCLUDED.architecture_style),
               weather_or_atmosphere = COALESCE(locations.weather_or_atmosphere, EXCLUDED.weather_or_atmosphere),
               social_layer          = COALESCE(locations.social_layer,          EXCLUDED.social_layer),
               cultural_dna          = COALESCE(locations.cultural_dna,          EXCLUDED.cultural_dna)
        RETURNING id
        """,
        (
            project_id,
            loc["name"],
            loc.get("description"),
            loc.get("time_of_day"),
            loc.get("mood"),
            loc.get("cultural_notes"),
            loc.get("visual_details"),
            loc.get("entity_type"),
            loc.get("geography"),
            loc.get("time_period"),
            loc.get("architecture_style"),
            loc.get("weather_or_atmosphere"),
            loc.get("social_layer"),
            loc.get("cultural_dna"),
            loc.get("metadata"),
        ),
    )
    return cur.fetchone()["id"]


def _build_world_description(location_dna: str, world: dict) -> str:
    """Build a brief world-context description — geography and cultural DNA only.

    architecture_style and characteristic_setting are intentionally excluded:
    they should not be baked into every location row as a visual prescription.
    The storyboard and image engines derive visual detail from story context.
    """
    bits = [f"World DNA: {location_dna}"] if not _is_vague(location_dna) else []
    for key, label in [
        ("geography",   "geography"),
        ("time_period", "time period"),
        ("social_layer", "social layer"),
        ("cultural_dna", "cultural DNA"),
    ]:
        v = _norm(world.get(key))
        if v and not _is_vague(v):
            bits.append(f"{label}: {v}")
    return ". ".join(bits) if bits else (location_dna or "Universal")


def _build_visual_details(location_dna: str, world: dict) -> str:
    """Build a minimal visual anchor — location name only.

    architecture_style is excluded: it was causing every location row to carry
    the same architectural template regardless of the actual space type.
    The image engine uses story context + geographic anchor instead.
    """
    if not _is_vague(location_dna) and location_dna.lower() != "universal":
        return location_dna
    return ""


def _world_field(world: dict, *keys: str) -> str:
    """Pick the first non-vague value from any of the given world_assumptions keys."""
    for key in keys:
        v = _norm(world.get(key))
        if v and not _is_vague(v):
            return v
    return ""


def materialize_locations(project_id: str, context_packet: dict) -> list[dict]:
    """Derive location rows from context_packet and persist them."""
    if not isinstance(context_packet, dict):
        context_packet = {}

    location_dna = _norm(context_packet.get("location_dna")) or "Universal"
    raw_world = context_packet.get("world_assumptions")
    world: dict = raw_world if isinstance(raw_world, dict) else {}
    raw_entities = context_packet.get("entities")
    entities: list = raw_entities if isinstance(raw_entities, list) else []
    raw_motifs = context_packet.get("motifs")
    motifs: list = raw_motifs if isinstance(raw_motifs, list) else []
    raw_mm = context_packet.get("motif_map")
    motif_map: dict = raw_mm if isinstance(raw_mm, dict) else {}

    # Alias-aware reads (engine emits era/time_of_day/social_context/etc.;
    # we also accept v3.1 alternate names so we never drop data).
    geography = _world_field(world, "geography")
    time_period = _world_field(world, "time_period", "era")
    # Accept both new canonical name and pre-rename alias
    time_of_day = _world_field(world, "characteristic_time", "time_of_day")
    architecture_style = _world_field(world, "architecture_style")
    weather_or_atmosphere = _world_field(
        world, "weather_or_atmosphere", "season",
        "characteristic_setting", "domestic_setting"
    )
    social_layer = _world_field(
        world, "social_layer", "social_context", "economic_context"
    )
    cultural_dna = _world_field(world, "cultural_dna")

    locations: list[dict] = []

    # ---- PRIMARY: world DNA + world_assumptions ----
    has_world_signal = (
        (location_dna and location_dna.lower() != "universal")
        or any(not _is_vague(v) for v in (geography, time_period, architecture_style,
                                          weather_or_atmosphere, social_layer, cultural_dna))
    )
    if has_world_signal:
        motif_notes_parts = []
        for name, payload in list(motif_map.items())[:5]:
            if not isinstance(payload, dict):
                motif_notes_parts.append(str(name))
                continue
            visual_form = _norm(payload.get("visual_form"))
            significance = _norm(payload.get("significance"))
            bit = str(name)
            if visual_form:
                bit += f" ({visual_form})"
            if significance:
                bit += f" — {significance}"
            motif_notes_parts.append(bit)
        if not motif_notes_parts and motifs:
            motif_notes_parts = motifs[:5]
        cultural_notes = "; ".join(motif_notes_parts) if motif_notes_parts else None

        locations.append({
            "name": location_dna,
            "description": _build_world_description(location_dna, world),
            "time_of_day": time_of_day or time_period or None,
            "mood": weather_or_atmosphere or None,
            "cultural_notes": cultural_notes,
            "visual_details": _build_visual_details(location_dna, world),
            "entity_type": "world_dna",
            "geography": geography or None,
            "time_period": time_period or None,
            "architecture_style": architecture_style or None,
            "weather_or_atmosphere": weather_or_atmosphere or None,
            "social_layer": social_layer or None,
            "cultural_dna": cultural_dna or None,
            "metadata": Json({
                "source": "location_dna+world_assumptions",
                "motifs": motifs,
                "motif_map": motif_map,
                "world_assumptions": world,
            }),
        })

    # ---- SECONDARY: named places from entity map ----
    seen_names: set[str] = {loc["name"].lower() for loc in locations}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ent_type = _norm(ent.get("type")).lower()
        if "place" not in ent_type and "location" not in ent_type:
            continue
        name = _norm(ent.get("name"))
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        role_desc = _norm(ent.get("role"))
        locations.append({
            "name": name,
            "description": role_desc or name,
            "time_of_day": time_period or None,
            "mood": role_desc or weather_or_atmosphere or None,
            "cultural_notes": cultural_dna or (location_dna if location_dna.lower() != "universal" else None),
            "visual_details": (
                f"{name} in {location_dna}" if location_dna.lower() != "universal" else name
            ),
            "entity_type": "named_place",
            "geography": geography or None,
            "time_period": time_period or None,
            "architecture_style": architecture_style or None,
            "weather_or_atmosphere": weather_or_atmosphere or None,
            "social_layer": social_layer or None,
            "cultural_dna": cultural_dna or None,
            "metadata": Json({"source": "entity_map", "raw_entity": ent}),
        })

    # ---- TERTIARY: creative brief scene locations ----
    # Each scene in the chosen creative brief specifies a distinct visual
    # setting.  Materialize a location row for each so the pipeline can
    # later link shot_assets rows to the correct scene plate.
    cb = context_packet.get("creative_brief")
    chosen = (cb or {}).get("chosen") if isinstance(cb, dict) else None
    if isinstance(chosen, dict):
        cb_scenes = chosen.get("scenes") or []
        if isinstance(cb_scenes, list):
            for scene in cb_scenes:
                if not isinstance(scene, dict):
                    continue
                scene_loc = _norm(scene.get("location"))
                if not scene_loc or _is_vague(scene_loc):
                    continue
                if scene_loc.lower() in seen_names:
                    continue
                seen_names.add(scene_loc.lower())
                tod = _norm(scene.get("time_of_day"))
                props = scene.get("props") or []
                props_str = ", ".join(str(p) for p in props[:6] if p) if isinstance(props, list) else ""
                summary = _norm(scene.get("summary") or "")
                locations.append({
                    "name": scene_loc,
                    "description": summary or scene_loc,
                    "time_of_day": tod or time_of_day or None,
                    "mood": weather_or_atmosphere or None,
                    "cultural_notes": cultural_dna or (location_dna if location_dna.lower() != "universal" else None),
                    "visual_details": (
                        f"{scene_loc}; {props_str}" if props_str else scene_loc
                    ),
                    "entity_type": "creative_brief_scene",
                    "geography": geography or None,
                    "time_period": time_period or None,
                    "architecture_style": architecture_style or None,
                    "weather_or_atmosphere": weather_or_atmosphere or None,
                    "social_layer": social_layer or None,
                    "cultural_dna": cultural_dna or None,
                    "metadata": Json({
                        "source": "creative_brief_scene",
                        "scene_name": scene.get("name"),
                        "beat_range": scene.get("beat_range"),
                        "props": props,
                    }),
                })
        # Also materialize any top-level visual_locations entries
        for vl in (chosen.get("visual_locations") or []):
            vl_name = _norm(vl)
            if not vl_name or _is_vague(vl_name) or vl_name.lower() in seen_names:
                continue
            seen_names.add(vl_name.lower())
            locations.append({
                "name": vl_name,
                "description": vl_name,
                "time_of_day": time_of_day or None,
                "mood": weather_or_atmosphere or None,
                "cultural_notes": cultural_dna or None,
                "visual_details": vl_name,
                "entity_type": "creative_brief_scene",
                "geography": geography or None,
                "time_period": time_period or None,
                "architecture_style": architecture_style or None,
                "weather_or_atmosphere": weather_or_atmosphere or None,
                "social_layer": social_layer or None,
                "cultural_dna": cultural_dna or None,
                "metadata": Json({"source": "creative_brief_visual_locations"}),
            })

    if not locations:
        logger.info("No locations extracted for project %s", project_id)
        return []

    with _db() as conn, conn.cursor() as cur:
        for loc in locations:
            _upsert_location(cur, project_id, loc)
        conn.commit()
    logger.info("Upserted %d location(s) for project %s", len(locations), project_id)
    return locations
