"""
One-time migration: rename legacy world_assumptions field names inside
styled_timeline shot data stored in the projects table.

Renames applied inside each shot's environment_profile.world_assumptions:
  domestic_setting  →  characteristic_setting
  time_of_day       →  characteristic_time

Run once:
  python migrate_world_assumptions.py

Safe to re-run — already-canonical records are left untouched.
"""

import json
import logging
import os
import sys

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    sys.exit("DATABASE_URL is not set.")

_RENAMES = {
    "domestic_setting": "characteristic_setting",
    "time_of_day": "characteristic_time",
}


def _migrate_world_assumptions(wa: dict) -> tuple[dict, bool]:
    """Return (updated_wa, changed)."""
    changed = False
    for old_key, new_key in _RENAMES.items():
        if old_key in wa:
            if not wa.get(new_key):
                wa[new_key] = wa[old_key]
            del wa[old_key]
            changed = True
    return wa, changed


def _migrate_timeline(timeline: list) -> tuple[list, int]:
    """Walk every shot in the timeline and migrate world_assumptions fields.
    Returns (updated_timeline, shots_changed_count)."""
    shots_changed = 0
    for shot in timeline:
        env = shot.get("environment_profile")
        if not isinstance(env, dict):
            continue
        wa = env.get("world_assumptions")
        if not isinstance(wa, dict):
            continue
        updated_wa, changed = _migrate_world_assumptions(wa)
        if changed:
            env["world_assumptions"] = updated_wa
            shot["environment_profile"] = env
            shots_changed += 1
    return timeline, shots_changed


def run_migration() -> None:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, styled_timeline FROM projects "
                "WHERE styled_timeline IS NOT NULL"
            )
            projects = cur.fetchall()

        log.info("Found %d projects with styled_timeline.", len(projects))
        total_projects_updated = 0
        total_shots_updated = 0

        for project in projects:
            project_id = project["id"]
            raw_timeline = project["styled_timeline"]

            if isinstance(raw_timeline, str):
                try:
                    timeline = json.loads(raw_timeline)
                except json.JSONDecodeError:
                    log.warning("Project %s: could not parse styled_timeline JSON — skipping.", project_id)
                    continue
            elif isinstance(raw_timeline, list):
                timeline = raw_timeline
            else:
                continue

            updated_timeline, shots_changed = _migrate_timeline(timeline)
            if shots_changed == 0:
                continue

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE projects SET styled_timeline=%s, updated_at=NOW() WHERE id=%s",
                    (Jsonb(updated_timeline), project_id),
                )
            conn.commit()
            log.info(
                "Project %s: migrated %d shot(s).",
                project_id,
                shots_changed,
            )
            total_projects_updated += 1
            total_shots_updated += shots_changed

        log.info(
            "Migration complete. Projects updated: %d. Shots updated: %d.",
            total_projects_updated,
            total_shots_updated,
        )


if __name__ == "__main__":
    run_migration()
