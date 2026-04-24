"""Background pipeline worker for Qaivid MetaMind.

Runs the full two-pass production:
  Pass 1: context-only (so we know the speaker + location_dna).
  Pass 2: build / accept reference images → stored in R2.
  Pass 3: full production with user_image_url injected.
  Pass 4: per-shot still rendering, face-locked to the character ref.
  Pass 5: per-shot video clips via Kling image-to-video.

Status is streamed into the `projects.progress` JSONB column and the
`refs` / `shot_assets` / `video_assets` tables so the UI can poll for
live updates.

All file assets are stored in Cloudflare R2; `file_path` columns now hold
public R2 URLs rather than local filesystem paths.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from audio_processor import AudioProcessor
from input_processor import InputProcessor
from narrative_engine import generate_narrative_intelligence
from project_brain import ProjectBrain
from production_orchestrator import ProductionOrchestrator
from asset_export_module import AssetExportModule
from image_generator import (
    ImageGenerationError,
    build_character_plate_prompt,
    build_location_plate_prompt,
    generate_character_plate,
    generate_character_ref,
    generate_environment_ref,
    generate_location_plate,
    generate_shot_still,
)
from style_profile_engine import StyleProfileEngine
from style_profile_registry import StyleProfileRegistry
from video_generator import VideoGenerationError, generate_shot_video
from character_materializer import materialize_characters
from location_materializer import materialize_locations
from motif_materializer import materialize_motifs
from materializer_engine_v2 import run_materializer_sync
import r2_storage
import dataset_collector

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PROJECTS_ROOT = ROOT / "projects"
PROJECTS_ROOT.mkdir(exist_ok=True)

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qaivid-pipeline")
_SHOT_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="qaivid-shot")
_REF_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qaivid-ref")
_VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qaivid-video")

_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: Set[Tuple[str, str]] = set()


def _try_acquire(key: Tuple[str, str]) -> bool:
    with _INFLIGHT_LOCK:
        if key in _INFLIGHT:
            return False
        _INFLIGHT.add(key)
        return True


def _release(key: Tuple[str, str]) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(key)


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def ensure_schema() -> None:
    """Idempotent bootstrap of every table this worker reads/writes."""
    try:
        from system_config import ensure_schema as _ensure_settings_schema
        _ensure_settings_schema()
    except Exception as exc:
        logger.warning("system_config schema bootstrap failed: %s", exc)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              SERIAL PRIMARY KEY,
                email           TEXT NOT NULL UNIQUE,
                password_hash   TEXT NOT NULL,
                is_admin        BOOLEAN NOT NULL DEFAULT FALSE,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'free';")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMPTZ;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS credits INTEGER NOT NULL DEFAULT 0;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_interval TEXT NOT NULL DEFAULT 'monthly';")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verify_token TEXT;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verify_sent_at TIMESTAMPTZ;")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS users_email_verify_token_idx
            ON users (email_verify_token) WHERE email_verify_token IS NOT NULL;
        """)
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token TEXT;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token_sent_at TIMESTAMPTZ;")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS users_reset_token_idx
            ON users (reset_token) WHERE reset_token IS NOT NULL;
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id              TEXT PRIMARY KEY,
                user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
                name            TEXT NOT NULL DEFAULT 'Untitled',
                genre           TEXT NOT NULL DEFAULT 'song',
                text            TEXT NOT NULL,
                audio_filename  TEXT,
                status          TEXT NOT NULL DEFAULT 'queued',
                error           TEXT,
                summary         JSONB,
                context_packet  JSONB,
                styled_timeline JSONB,
                progress        JSONB,
                export_path     TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS progress JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS input_packet JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS lyrics_timed JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS shared_at TIMESTAMPTZ;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;")
        cur.execute(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS user_id INTEGER "
            "REFERENCES users(id) ON DELETE CASCADE;"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS projects_user_id_idx ON projects(user_id);"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS refs (
                id          SERIAL PRIMARY KEY,
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'generated',
                file_path   TEXT,
                prompt      TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',
                error       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT refs_project_role_unique UNIQUE (project_id, role)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shot_assets (
                id          SERIAL PRIMARY KEY,
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                shot_index  INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                file_path   TEXT,
                prompt      TEXT,
                error       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT shot_assets_project_idx_unique UNIQUE (project_id, shot_index)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_assets (
                id          SERIAL PRIMARY KEY,
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                shot_index  INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'queued',
                file_path   TEXT,
                error       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT video_assets_project_idx_unique UNIQUE (project_id, shot_index)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id              SERIAL PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name            TEXT NOT NULL,
                role            TEXT,
                entity_type     TEXT NOT NULL DEFAULT 'speaker',
                appearance      TEXT,
                age_range       TEXT,
                cultural_notes  TEXT,
                emotional_notes TEXT,
                metadata        JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS characters_project_idx ON characters(project_id);"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS characters_project_name_unique "
            "ON characters(project_id, name);"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id              SERIAL PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name            TEXT NOT NULL,
                description     TEXT,
                time_of_day     TEXT,
                mood            TEXT,
                cultural_notes  TEXT,
                visual_details  TEXT,
                entity_type     TEXT NOT NULL DEFAULT 'world_dna',
                metadata        JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS locations_project_idx ON locations(project_id);"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS locations_project_name_unique "
            "ON locations(project_id, name);"
        )
        cur.execute(
            "ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS character_id INTEGER "
            "REFERENCES characters(id) ON DELETE SET NULL;"
        )
        cur.execute(
            "ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS location_id INTEGER "
            "REFERENCES locations(id) ON DELETE SET NULL;"
        )
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS stage TEXT DEFAULT 'new';")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS audio_data JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS transcript TEXT;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS final_video_url TEXT;")

        # Task #50 — motion_prompt column for per-shot Kling-optimised motion cue
        cur.execute("ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS motion_prompt TEXT;")

        # Wardrobe Engine — scene-aware per-shot wardrobe override
        # Replaces the single characters.wardrobe global with a context-specific
        # outfit description per scene cluster so the character's look varies
        # authentically between scenes (casual home → formal wedding, etc.).
        cur.execute("ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS wardrobe_context TEXT;")

        # Task — prompt composer: track whether the prompt was hand-edited
        # by the user so the composer knows to use it verbatim instead of
        # rebuilding from structured fields.
        cur.execute("ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS prompt_user_edited BOOLEAN NOT NULL DEFAULT FALSE;")

        # MetaMind v3.1 — character appearance fields
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS gender TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ethnicity TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS complexion TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS wardrobe TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS grooming TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS relationship TEXT;")

        # Reference Engine — per-character locked plates
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ref_image_url TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ref_prompt TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ref_status TEXT NOT NULL DEFAULT 'pending';")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ref_source TEXT;")
        cur.execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ref_error TEXT;")

        # MetaMind v3.1 — location world_assumptions fields
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS geography TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS time_period TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS architecture_style TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS weather_or_atmosphere TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS social_layer TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS cultural_dna TEXT;")

        # Reference Engine — per-location locked plates
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS ref_image_url TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS ref_prompt TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS ref_status TEXT NOT NULL DEFAULT 'pending';")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS ref_source TEXT;")
        cur.execute("ALTER TABLE locations ADD COLUMN IF NOT EXISTS ref_error TEXT;")

        # MetaMind v3.1 — motifs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS motifs (
                id              SERIAL PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name            TEXT NOT NULL,
                motif_type      TEXT,
                significance    TEXT,
                visual_form     TEXT,
                metadata        JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT motifs_project_name_unique UNIQUE (project_id, name)
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS motifs_project_idx ON motifs(project_id);"
        )

        # Task #57 — Style Profile Engine columns
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS style_profile JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS style_suggestions JSONB;")

        # Character Looks — one styled plate per (character × scene cluster).
        # Generated after the base identity plates in the reference engine stage,
        # so the Reference Images review shows all looks before stills begin.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS character_looks (
                id              SERIAL PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                character_id    INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
                cluster_id      TEXT NOT NULL,
                cluster_label   TEXT,
                wardrobe_text   TEXT,
                ref_image_url   TEXT,
                ref_status      TEXT NOT NULL DEFAULT 'pending',
                ref_prompt      TEXT,
                ref_error       TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT character_looks_unique UNIQUE (project_id, character_id, cluster_id)
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS character_looks_project_idx "
            "ON character_looks(project_id);"
        )
        # Track which scene cluster each shot belongs to so the stills renderer
        # can pick the right styled look plate for that shot.
        cur.execute(
            "ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS look_cluster_id TEXT;"
        )

        # Post Production Stage — Quick Video (Task #100)
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS postprod_config JSONB;")
        cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS quick_video_url TEXT;")

        # Project Brain — single JSONB accumulator for all pipeline stage outputs
        cur.execute(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS project_brain JSONB;"
        )
        # Narrative Engine output — also kept as a top-level JSONB column for
        # legacy reads (Style fallback, etc.). Lazily added by _stage_narrative_job
        # historically; bootstrap here so any earlier-running stage can SELECT it.
        cur.execute(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS narrative_packet JSONB;"
        )

        # Outpaint / wide-format expansion columns for shot stills
        cur.execute("ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS outpaint_url TEXT;")
        cur.execute("ALTER TABLE shot_assets ADD COLUMN IF NOT EXISTS outpaint_status TEXT;")

        # Credit ledger — append-only audit log of credit transactions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS credit_ledger (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                project_id  TEXT REFERENCES projects(id) ON DELETE SET NULL,
                credits     INTEGER NOT NULL DEFAULT 0,
                label       TEXT NOT NULL,
                deducted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS credit_ledger_user_idx ON credit_ledger(user_id);"
        )

        conn.commit()


def cleanup_project_assets(project_id: str) -> None:
    """Delete all R2 assets for this project and any leftover local files."""
    if r2_storage.r2_available():
        try:
            deleted = r2_storage.delete_prefix(f"projects/{project_id}/")
            logger.info("Removed %d R2 objects for project %s", deleted, project_id)
        except Exception:
            logger.exception("R2 cleanup failed for project %s", project_id)

    target = (PROJECTS_ROOT / project_id).resolve()
    if target.parent == PROJECTS_ROOT.resolve() and target.is_dir():
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        logger.info("Removed local folder %s", target)


def _set_status(project_id: str, status: str, progress: dict, error: Optional[str] = None,
                stage: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        if stage is not None:
            cur.execute(
                """
                UPDATE projects
                   SET status = %s,
                       progress = %s,
                       error = %s,
                       stage = %s,
                       updated_at = NOW()
                 WHERE id = %s
                """,
                (status, Json(progress), error, stage, project_id),
            )
        else:
            cur.execute(
                """
                UPDATE projects
                   SET status = %s,
                       progress = %s,
                       error = %s,
                       updated_at = NOW()
                 WHERE id = %s
                """,
                (status, Json(progress), error, project_id),
            )
        conn.commit()


def _upsert_ref(project_id: str, role: str, source: str, status: str,
                file_path: Optional[str] = None, prompt: Optional[str] = None,
                error: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO refs (project_id, role, source, file_path, prompt, status, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id, role) DO UPDATE
               SET source = EXCLUDED.source,
                   file_path = COALESCE(EXCLUDED.file_path, refs.file_path),
                   prompt = COALESCE(EXCLUDED.prompt, refs.prompt),
                   status = EXCLUDED.status,
                   error = EXCLUDED.error
            """,
            (project_id, role, source, file_path, prompt, status, error),
        )
        conn.commit()


def _get_ref(project_id: str, role: str) -> Optional[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM refs WHERE project_id = %s AND role = %s", (project_id, role))
        return cur.fetchone()


def _seed_shot_rows(project_id: str, shot_indices: list[int]) -> None:
    with _db() as conn, conn.cursor() as cur:
        for idx in shot_indices:
            cur.execute(
                """
                INSERT INTO shot_assets (project_id, shot_index, status)
                VALUES (%s, %s, 'pending')
                ON CONFLICT (project_id, shot_index) DO UPDATE
                   SET status = 'pending', error = NULL, updated_at = NOW()
                """,
                (project_id, idx),
            )
        conn.commit()


def _update_shot(project_id: str, shot_index: int, status: str,
                 file_path: Optional[str] = None, prompt: Optional[str] = None,
                 error: Optional[str] = None,
                 motion_prompt: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO shot_assets (project_id, shot_index, status, file_path, prompt, error, motion_prompt)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id, shot_index) DO UPDATE
               SET status = EXCLUDED.status,
                   file_path = COALESCE(EXCLUDED.file_path, shot_assets.file_path),
                   prompt = COALESCE(EXCLUDED.prompt, shot_assets.prompt),
                   motion_prompt = COALESCE(EXCLUDED.motion_prompt, shot_assets.motion_prompt),
                   error = EXCLUDED.error,
                   updated_at = NOW()
            """,
            (project_id, shot_index, status, file_path, prompt, error, motion_prompt),
        )
        conn.commit()


def _seed_video_rows(project_id: str, shot_indices: list[int]) -> None:
    with _db() as conn, conn.cursor() as cur:
        for idx in shot_indices:
            cur.execute(
                """
                INSERT INTO video_assets (project_id, shot_index, status)
                VALUES (%s, %s, 'queued')
                ON CONFLICT (project_id, shot_index) DO UPDATE
                   SET status = 'queued', error = NULL, updated_at = NOW()
                """,
                (project_id, idx),
            )
        conn.commit()


def _update_video(project_id: str, shot_index: int, status: str,
                  file_path: Optional[str] = None,
                  error: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_assets (project_id, shot_index, status, file_path, error)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (project_id, shot_index) DO UPDATE
               SET status = EXCLUDED.status,
                   file_path = COALESCE(EXCLUDED.file_path, video_assets.file_path),
                   error = EXCLUDED.error,
                   updated_at = NOW()
            """,
            (project_id, shot_index, status, file_path, error),
        )
        conn.commit()


def _get_ready_still_url(project_id: str, shot_index: int) -> Optional[str]:
    """Return the R2 public URL of a ready still for this shot, or None.

    Prefers the outpainted version (outpaint_url) when available so that
    video generation receives a true 16:9 / 9:16 frame.
    Falls back to the original file_path if no outpaint exists.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_path, outpaint_url, outpaint_status FROM shot_assets "
            "WHERE project_id=%s AND shot_index=%s AND status='ready'",
            (project_id, shot_index),
        )
        row = cur.fetchone()
    if not row:
        return None
    if row.get("outpaint_status") == "ready" and row.get("outpaint_url"):
        logger.debug("Using outpainted still for project=%s shot=%s", project_id, shot_index)
        return row["outpaint_url"]
    return row.get("file_path") or None


def _get_shot_motion_prompt(project_id: str, shot_index: int) -> Optional[str]:
    """Return the stored motion_prompt for a shot, or None."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT motion_prompt FROM shot_assets WHERE project_id=%s AND shot_index=%s",
            (project_id, shot_index),
        )
        row = cur.fetchone()
    if row:
        return row.get("motion_prompt") or None
    return None


def _render_video(project_id: str, shot: dict) -> None:
    """Animate one still into a video clip, enriched with Project Brain data.

    Brain reads (non-fatal fallback to legacy on any error):
      materializer_packet  → identity_seed, continuity_rules, character db_id
      reference_assets     → (not used for video conditioning but logged)
      style_packet         → cinematic_style, lighting_logic
      narrative_packet     → motion_philosophy
    """
    from motion_render_prompt_builder import build_video_clip_prompt

    idx = shot.get("shot_index") or shot.get("timeline_index")
    still_url = _get_ready_still_url(project_id, idx)
    if not still_url:
        _update_video(project_id, idx, "failed",
                      error="No ready still image found; run stills first.")
        return
    _update_video(project_id, idx, "rendering")

    # ── Brain load (non-fatal) ────────────────────────────────────────────────
    identity_seed:      Optional[str]       = None
    continuity_rules:   Optional[List[str]] = None
    cinematic_style:    Optional[str]       = None
    lighting_logic:     Optional[str]       = None
    motion_philosophy:  str                 = "mixed"
    character_db_id:    Optional[int]       = None
    location_db_id:     Optional[int]       = None
    link_status:        str                 = "unresolved"
    char_ref_url:       Optional[str]       = None

    try:
        with _db() as conn:
            _vbrain = ProjectBrain.load(project_id, conn)

        # materializer_packet — identity + continuity
        _mp     = _vbrain.read("materializer_packet") or {}
        _cb     = (_mp.get("character_bible") or {}).get("characters") or []
        _lb     = (_mp.get("location_bible")  or {}).get("locations")  or []
        _rules  = _mp.get("continuity_rules") or []
        continuity_rules = [str(r) for r in _rules] if _rules else None

        # Build exact-FK lookup dicts (no alias guessing)
        _char_by_dbid: dict = {
            int(c["db_id"]): c
            for c in _cb if isinstance(c, dict) and isinstance(c.get("db_id"), int)
        }
        _loc_by_dbid: dict = {
            int(l["db_id"]): l
            for l in _lb if isinstance(l, dict) and isinstance(l.get("db_id"), int)
        }

        # Per-shot entity resolution: read character_id / location_id from shot_assets
        # (written by _link_shots_to_entities at stills stage — exact FK, no guessing)
        _shot_char_id: Optional[int] = None
        _shot_loc_id:  Optional[int] = None
        try:
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT character_id, location_id FROM shot_assets "
                    "WHERE project_id=%s AND shot_index=%s",
                    (project_id, idx),
                )
                _sa = cur.fetchone() or {}
            _shot_char_id = _sa.get("character_id")
            _shot_loc_id  = _sa.get("location_id")
        except Exception:
            logger.debug(
                "_render_video: could not read shot entity IDs from shot_assets "
                "for project=%s shot=%s (non-fatal)", project_id, idx,
            )

        # Resolve matched bible entries by exact FK — no alias/primary fallback.
        # IDs stay null if shot_assets has no FK for this shot.
        _char_entry: Optional[dict] = None
        _loc_entry:  Optional[dict] = None
        if _shot_char_id is not None:
            _char_entry = _char_by_dbid.get(_shot_char_id)
            if _char_entry is not None:
                character_db_id = _shot_char_id
        if _shot_loc_id is not None:
            _loc_entry = _loc_by_dbid.get(_shot_loc_id)
            if _loc_entry is not None:
                location_db_id = _shot_loc_id

        if _char_entry:
            identity_seed = str(_char_entry.get("identity_seed") or "").strip() or None

        # link_status — resolved only when BOTH FKs are exact matches from shot_assets
        if character_db_id is not None and location_db_id is not None:
            link_status = "resolved"
        elif character_db_id is not None or location_db_id is not None:
            link_status = "semantic_only"
        else:
            link_status = "unresolved"

        # reference_assets — read character ref URL for this shot's matched character
        _ra = _vbrain.read("reference_assets") or {}
        _char_refs = _ra.get("character_references") or []
        if character_db_id is not None:
            _matched_ref = next(
                (r for r in _char_refs
                 if isinstance(r, dict) and r.get("character_id") == character_db_id),
                None,
            )
            if _matched_ref:
                char_ref_url = _matched_ref.get("ref_image_url") or None

        # style_packet — cinematic look + lighting
        _sp             = _vbrain.read("style_packet") or {}
        cinematic_style = ((_sp.get("cinematic") or {}).get("look") or "").strip() or None
        lighting_logic  = (str(_sp.get("lighting_logic") or "")).strip() or None

        # narrative_packet — motion philosophy
        _np = _vbrain.read("narrative_packet") or {}
        motion_philosophy = str(_np.get("motion_philosophy") or "mixed").strip() or "mixed"

    except Exception:
        logger.debug(
            "_render_video: brain load failed for project=%s shot=%s (non-fatal, using legacy path)",
            project_id, idx,
        )

    try:
        prompt = shot.get("styled_visual_prompt") or shot.get("visual_prompt") or ""
        duration = shot.get("duration")

        # Build a brain-aware motion prompt; fall back to stored/legacy value if empty
        brain_motion_prompt = build_video_clip_prompt(
            shot,
            motion_philosophy=motion_philosophy,
            identity_seed=identity_seed,
            cinematic_style=cinematic_style,
            lighting_logic=lighting_logic,
            continuity_rules=continuity_rules,
        )

        # If brain prompt is empty, fall back to shot dict or stored DB value
        motion_prompt = brain_motion_prompt or (shot.get("motion_prompt") or "").strip() or \
            _get_shot_motion_prompt(project_id, idx)

        logger.debug(
            "_render_video: project=%s shot=%s link_status=%s motion_philosophy=%s "
            "char_db_id=%s loc_db_id=%s char_ref=%s prompt_len=%d",
            project_id, idx, link_status, motion_philosophy,
            character_db_id, location_db_id,
            "yes" if char_ref_url else "no",
            len(motion_prompt),
        )

        public_url = generate_shot_video(project_id, idx, still_url, prompt, duration,
                                         motion_prompt=motion_prompt)
        _update_video(project_id, idx, "ready", file_path=public_url)

    except (VideoGenerationError, Exception) as exc:
        logger.exception("Video render failed for project=%s shot=%s", project_id, idx)
        _update_video(project_id, idx, "failed", error=str(exc))


def _store_pipeline_result(project_id: str, name: str, result: dict) -> str:
    """Store pipeline result in the DB; export JSON goes to R2."""
    exporter = AssetExportModule(name or "Qaivid_Project")
    export_json = exporter.export_to_json(result["styled_timeline"])
    r2_key = f"projects/{project_id}/exports/{project_id}.json"
    try:
        export_url = r2_storage.upload_bytes(
            export_json.encode("utf-8"), r2_key, content_type="application/json"
        )
    except Exception:
        logger.exception("Export JSON upload to R2 failed; storing path as key")
        export_url = r2_key

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE projects
               SET summary = %s,
                   context_packet = %s,
                   styled_timeline = %s,
                   export_path = %s,
                   updated_at = NOW()
             WHERE id = %s
            """,
            (
                Json(result["summary"]),
                Json(result["context_packet"]),
                Json(result["styled_timeline"]),
                export_url,
                project_id,
            ),
        )
        conn.commit()
    return export_url


def _ensure_character_ref(project_id: str, context_packet: dict) -> Optional[str]:
    """Return public R2 URL of character ref (generates if missing)."""
    existing = _get_ref(project_id, "character")
    if existing and existing["status"] == "ready" and existing.get("file_path"):
        return existing["file_path"]

    key = (project_id, "ref:character")
    if not _try_acquire(key):
        logger.info("Character ref for %s already rendering; skipping duplicate", project_id)
        return None
    try:
        _upsert_ref(project_id, "character", "generated", "rendering")
        try:
            url = generate_character_ref(
                speaker=context_packet.get("speaker") or {},
                location_dna=context_packet.get("location_dna") or "Universal",
                project_id=project_id,
            )
            _upsert_ref(project_id, "character", "generated", "ready", file_path=url)
            return url
        except Exception as exc:
            logger.exception("Character ref generation failed")
            _upsert_ref(project_id, "character", "generated", "failed", error=str(exc))
            return None
    finally:
        _release(key)


def _ensure_env_ref(project_id: str, context_packet: dict) -> Optional[str]:
    """Return public R2 URL of environment ref (generates if missing)."""
    existing = _get_ref(project_id, "environment")
    if existing and existing["status"] == "ready" and existing.get("file_path"):
        return existing["file_path"]

    key = (project_id, "ref:environment")
    if not _try_acquire(key):
        logger.info("Environment ref for %s already rendering; skipping duplicate", project_id)
        return None
    try:
        _upsert_ref(project_id, "environment", "generated", "rendering")
        try:
            url = generate_environment_ref(
                location_dna=context_packet.get("location_dna") or "Universal",
                motifs=context_packet.get("motifs") or [],
                project_id=project_id,
            )
            _upsert_ref(project_id, "environment", "generated", "ready", file_path=url)
            return url
        except Exception as exc:
            logger.exception("Environment ref generation failed")
            _upsert_ref(project_id, "environment", "generated", "failed", error=str(exc))
            return None
    finally:
        _release(key)


def _resolve_shot_refs(project_id: str, shot_index) -> tuple[Optional[str], Optional[str]]:
    """Backward-compat shim: returns (char_url, env_url) only."""
    char_url, env_url, _, _ = _resolve_shot_refs_full(project_id, shot_index)
    return char_url, env_url


def _resolve_shot_refs_full(
    project_id: str, shot_index
) -> tuple[Optional[str], Optional[str], Optional[dict], Optional[dict]]:
    """Look up the per-shot character + environment ref URLs *and* the
    full character/location records (so the prompt composer can describe
    the same subject consistently across all shots).

    Backfills missing appearance fields on the linked character from the
    project's most-detailed character record. The character_materializer
    sometimes creates a sparse record for the on-screen subject (e.g.
    "Beloved" — the addressee) while a parallel record (e.g. "Female
    protagonist") carries all the gender/ethnicity/wardrobe data. Without
    this backfill the composer would describe the subject as a generic
    "person" and every shot would render identically.

    Returns ``(char_url, env_url, character_dict, location_dict)``.
    Either dict may be None if the shot isn't linked to an entity.
    """
    char_url: Optional[str] = None
    env_url: Optional[str] = None
    character: Optional[dict] = None
    location: Optional[dict] = None
    _APPEARANCE_FIELDS = (
        "age_range", "gender", "ethnicity", "complexion",
        "wardrobe", "grooming", "appearance",
    )
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT character_id, location_id, wardrobe_context, look_cluster_id "
            "  FROM shot_assets "
            " WHERE project_id=%s AND shot_index=%s",
            (project_id, shot_index),
        )
        row = cur.fetchone() or {}
        char_id = row.get("character_id")
        loc_id = row.get("location_id")
        shot_wardrobe_context = (row.get("wardrobe_context") or "").strip()
        if char_id:
            cur.execute(
                "SELECT id, name, role, age_range, gender, ethnicity, "
                "       complexion, wardrobe, grooming, appearance, "
                "       ref_image_url, ref_status "
                "  FROM characters WHERE id=%s",
                (char_id,),
            )
            r = cur.fetchone() or {}
            if r:
                character = dict(r)
                if r.get("ref_status") == "ready" and r.get("ref_image_url"):
                    char_url = r["ref_image_url"]  # base identity plate

            # Prefer a scene-specific styled look plate over the base plate.
            # The look plate shows the same character in the outfit for THIS
            # scene cluster — stronger visual signal than text description alone.
            look_cluster_id = row.get("look_cluster_id")
            if look_cluster_id:
                cur.execute(
                    "SELECT ref_image_url FROM character_looks "
                    " WHERE project_id=%s AND character_id=%s "
                    "   AND cluster_id=%s AND ref_status='ready'",
                    (project_id, char_id, look_cluster_id),
                )
                look_row = cur.fetchone()
                if look_row and look_row.get("ref_image_url"):
                    char_url = look_row["ref_image_url"]  # override with look plate

        # Backfill empty appearance fields from the project's other
        # characters. We pick the donor with the most filled fields.
        if character:
            missing = [f for f in _APPEARANCE_FIELDS
                       if not (character.get(f) or "").strip()]
            if missing:
                cur.execute(
                    "SELECT id, name, age_range, gender, ethnicity, "
                    "       complexion, wardrobe, grooming, appearance "
                    "  FROM characters WHERE project_id=%s AND id<>%s",
                    (project_id, character["id"]),
                )
                donors = list(cur.fetchall() or [])
                def _filled_count(d):
                    return sum(1 for f in _APPEARANCE_FIELDS
                               if (d.get(f) or "").strip())
                donors.sort(key=_filled_count, reverse=True)
                if donors and _filled_count(donors[0]) > 0:
                    donor = donors[0]
                    for f in missing:
                        v = (donor.get(f) or "").strip()
                        if v:
                            character[f] = v

        # Inject per-shot wardrobe context (from wardrobe_engine.diversify_wardrobe).
        # When present this overrides characters.wardrobe so the prompt composer
        # uses the scene-appropriate outfit rather than the single global default.
        if character and shot_wardrobe_context:
            character["wardrobe_override"] = shot_wardrobe_context

        if loc_id:
            cur.execute(
                "SELECT id, name, description, mood, ref_image_url, ref_status "
                "  FROM locations WHERE id=%s",
                (loc_id,),
            )
            r = cur.fetchone() or {}
            if r:
                location = dict(r)
                if r.get("ref_status") == "ready" and r.get("ref_image_url"):
                    env_url = r["ref_image_url"]

        # Same backfill for location: when no shot-linked location exists,
        # fall back to the project's only / first location so the env clause
        # still has a concrete description.
        if location is None:
            cur.execute(
                "SELECT id, name, description, mood "
                "  FROM locations WHERE project_id=%s ORDER BY id LIMIT 1",
                (project_id,),
            )
            r = cur.fetchone()
            if r:
                location = dict(r)

    # Legacy fallback: read the project-wide character/environment ref.
    if char_url is None:
        legacy = _get_ref(project_id, "character")
        if legacy and legacy.get("status") == "ready" and legacy.get("file_path"):
            char_url = legacy["file_path"]
    if env_url is None:
        legacy = _get_ref(project_id, "environment")
        if legacy and legacy.get("status") == "ready" and legacy.get("file_path"):
            env_url = legacy["file_path"]
    return char_url, env_url, character, location


def _resolve_user_edited_flag(project_id: str, shot_index) -> bool:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(prompt_user_edited, FALSE) AS f "
            "  FROM shot_assets WHERE project_id=%s AND shot_index=%s",
            (project_id, shot_index),
        )
        r = cur.fetchone() or {}
        return bool(r.get("f"))


def _render_shot(project_id: str, shot: dict,
                 character_ref_url: Optional[str] = None,
                 environment_ref_url: Optional[str] = None) -> None:
    idx = shot.get("shot_index") or shot.get("timeline_index")
    key = (project_id, f"shot:{idx}")
    if not _try_acquire(key):
        logger.info("Shot %s for project %s already rendering; skipping duplicate", idx, project_id)
        return
    try:
        # Always resolve the full character + location records so the
        # prompt composer can describe the same subject consistently
        # across every shot. The per-shot ref URL args are only used
        # when callers supply them explicitly (legacy paths).
        char_url_db, env_url_db, character, location = _resolve_shot_refs_full(project_id, idx)
        if character_ref_url is None and environment_ref_url is None:
            character_ref_url, environment_ref_url = char_url_db, env_url_db

        # ── Brain identity data — look up matched char/loc entries by db_id ──
        # brain_chars_by_dbid / brain_locs_by_dbid are built here per shot
        # (brain load is fast — single DB row read + JSON parse).
        brain_char: Optional[dict] = None
        brain_loc:  Optional[dict] = None
        try:
            with _db() as conn:
                _shot_brain = ProjectBrain.load(project_id, conn)
            _mp = _shot_brain.read("materializer_packet") or {}
            _bc_list = list((_mp.get("character_bible") or {}).get("characters") or [])
            _bl_list = list((_mp.get("location_bible")  or {}).get("locations")  or [])
            # Build db_id indexes
            _bc_idx = {
                b["db_id"]: b for b in _bc_list
                if isinstance(b, dict) and isinstance(b.get("db_id"), int)
            }
            _bl_idx = {
                b["db_id"]: b for b in _bl_list
                if isinstance(b, dict) and isinstance(b.get("db_id"), int)
            }
            if character and isinstance(character.get("id"), int):
                brain_char = _bc_idx.get(character["id"])
            if location and isinstance(location.get("id"), int):
                brain_loc = _bl_idx.get(location["id"])
        except Exception:
            logger.debug(
                "_render_shot: brain load failed for project=%s shot=%s (non-fatal)",
                project_id, idx,
            )

        # Did the user hand-edit the prompt? If yes we use it verbatim.
        user_edited = _resolve_user_edited_flag(project_id, idx)
        user_override = None
        if user_edited:
            user_override = (shot.get("styled_visual_prompt")
                             or shot.get("visual_prompt") or "").strip() or None

        _update_shot(project_id, idx, "rendering",
                     prompt=(shot.get("styled_visual_prompt") or shot.get("visual_prompt") or "")[:4000],
                     motion_prompt=(shot.get("motion_prompt") or "")[:400] or None)
        try:
            url = generate_shot_still(
                shot, character_ref_url, project_id,
                environment_ref_url=environment_ref_url,
                character=character,
                location=location,
                user_override=user_override,
                brain_char=brain_char,
                brain_loc=brain_loc,
            )
            _update_shot(project_id, idx, "ready", file_path=url)
        except Exception as exc:
            logger.exception("Shot %s render failed", idx)
            _update_shot(project_id, idx, "failed", error=str(exc))
    finally:
        _release(key)


def _set_entity_ref(project_id: str, kind: str, entity_id: int, *,
                    status: str, file_path: Optional[str] = None,
                    prompt: Optional[str] = None,
                    source: Optional[str] = None,
                    error: Optional[str] = None) -> None:
    """Persist plate state on a single character/location row."""
    table = "characters" if kind == "character" else "locations"
    sets = ["ref_status=%s", "ref_error=%s"]
    args: list = [status, error]
    if file_path is not None:
        sets.append("ref_image_url=%s"); args.append(file_path)
    if prompt is not None:
        sets.append("ref_prompt=%s"); args.append(prompt)
    if source is not None:
        sets.append("ref_source=%s"); args.append(source)
    args.extend([project_id, entity_id])
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET {', '.join(sets)} WHERE project_id=%s AND id=%s",
            tuple(args),
        )
        conn.commit()


def _render_character_plate(project_id: str, character_id: int,
                             location_dna: str = "Universal",
                             prompt_override: Optional[str] = None,
                             style_suffix: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM characters WHERE id=%s AND project_id=%s",
                    (character_id, project_id))
        character = cur.fetchone()
    if not character:
        logger.warning("Character %s not found for project %s", character_id, project_id)
        return
    key = (project_id, f"plate:char:{character_id}")
    if not _try_acquire(key):
        return
    try:
        _set_entity_ref(project_id, "character", character_id,
                        status="rendering", source="generated", error=None)
        try:
            if style_suffix:
                if prompt_override:
                    prompt_override = f"{prompt_override} {style_suffix}".strip()
                else:
                    base = build_character_plate_prompt(character, location_dna)
                    prompt_override = f"{base} {style_suffix}".strip()
            url, used_prompt = generate_character_plate(
                character, project_id, location_dna=location_dna,
                prompt_override=prompt_override,
            )
            _set_entity_ref(project_id, "character", character_id,
                            status="ready", file_path=url, prompt=used_prompt,
                            source="generated", error=None)
        except Exception as exc:
            logger.exception("Character plate failed for id=%s", character_id)
            _set_entity_ref(project_id, "character", character_id,
                            status="failed", error=str(exc))
    finally:
        _release(key)


def _render_location_plate(project_id: str, location_id: int,
                            prompt_override: Optional[str] = None,
                            style_suffix: Optional[str] = None) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM locations WHERE id=%s AND project_id=%s",
                    (location_id, project_id))
        location = cur.fetchone()
    if not location:
        logger.warning("Location %s not found for project %s", location_id, project_id)
        return
    key = (project_id, f"plate:loc:{location_id}")
    if not _try_acquire(key):
        return
    try:
        _set_entity_ref(project_id, "location", location_id,
                        status="rendering", source="generated", error=None)
        try:
            if style_suffix:
                if prompt_override:
                    prompt_override = f"{prompt_override} {style_suffix}".strip()
                else:
                    base = build_location_plate_prompt(location)
                    prompt_override = f"{base} {style_suffix}".strip()
            url, used_prompt = generate_location_plate(
                location, project_id, prompt_override=prompt_override,
            )
            _set_entity_ref(project_id, "location", location_id,
                            status="ready", file_path=url, prompt=used_prompt,
                            source="generated", error=None)
        except Exception as exc:
            logger.exception("Location plate failed for id=%s", location_id)
            _set_entity_ref(project_id, "location", location_id,
                            status="failed", error=str(exc))
    finally:
        _release(key)


def _link_shots_to_entities(project_id: str, styled_timeline: list[dict]) -> None:
    """Update shot_assets.character_id / location_id based on expression_mode.

    Brain-first: reads materializer_packet from the project brain and uses the
    character/location descriptors (identity_seed, archetype, world label) as
    the primary name signals for shot-text matching.  Falls back to plain DB
    character/location names when the brain is absent or empty.

    FK resolution is done within entity-type buckets (speakers → speaker DB
    rows; world locations → world_dna DB rows) using positional alignment
    within that bucket — never across buckets and never by fuzzy name
    similarity across different entity types.

    expression_mode governs which FK column is written per shot:
    - "face" / "body" → character_id set; location_id NULL
    - anything else   → character_id NULL; location_id set
    """
    if not styled_timeline:
        return

    # ── Brain-first: load materializer_packet ────────────────────────────
    brain_chars: list[dict] = []
    brain_locs:  list[dict] = []
    try:
        with _db() as conn:
            _lb = ProjectBrain.load(project_id, conn)
        _mat = _lb.read("materializer_packet") or {}
        brain_chars = list((_mat.get("character_bible") or {}).get("characters") or [])
        brain_locs  = list((_mat.get("location_bible")  or {}).get("locations")  or [])
    except Exception:
        logger.debug("_link_shots_to_entities: brain load failed (non-fatal)")

    # ── DB: authoritative source for integer FK IDs ───────────────────────
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, entity_type FROM characters WHERE project_id = %s ORDER BY id",
            (project_id,),
        )
        chars = cur.fetchall()
        cur.execute(
            "SELECT id, name, entity_type FROM locations WHERE project_id = %s ORDER BY id",
            (project_id,),
        )
        locs = cur.fetchall()

    if not chars and not locs:
        return

    primary_char_id = next((c["id"] for c in chars if c["entity_type"] == "speaker"),   None)
    primary_loc_id  = next((l["id"] for l in locs  if l["entity_type"] == "world_dna"), None)

    # ── Build lookup lists: brain db_id first, DB names as fallback ──────
    # The brain materializer_packet entries now carry a `db_id` field (written
    # by _stage_refs_job after legacy materializers populate the DB tables).
    # When present, db_id is the exact integer FK — zero guessing required.
    # Brain descriptors (identity_seed, archetype, world) paired with db_id
    # are placed first so they take precedence in shot-text matching.
    # DB character/location names are appended after as fallback for projects
    # that predate this enrichment or where brain load fails.
    char_lookup: list[tuple[str, int]] = []
    loc_lookup:  list[tuple[str, int]] = []

    for bch in brain_chars:
        if not isinstance(bch, dict):
            continue
        fk = bch.get("db_id")
        if not isinstance(fk, int):
            continue
        for field in ("identity_seed", "archetype"):
            val = str(bch.get(field) or "").strip().lower()
            if val and len(val) > 2 and (val, fk) not in char_lookup:
                char_lookup.append((val, fk))

    for bloc in brain_locs:
        if not isinstance(bloc, dict):
            continue
        fk = bloc.get("db_id")
        if not isinstance(fk, int):
            continue
        for field in ("world", "location_id"):
            val = str(bloc.get(field) or "").strip().lower()
            if val and len(val) > 2 and (val, fk) not in loc_lookup:
                loc_lookup.append((val, fk))

    # DB names appended as fallback (covers legacy projects + brain-load failures).
    for c in chars:
        name = (c.get("name") or "").lower()
        if name and (name, c["id"]) not in char_lookup:
            char_lookup.append((name, c["id"]))
    for l in locs:
        name = (l.get("name") or "").lower()
        if name and (name, l["id"]) not in loc_lookup:
            loc_lookup.append((name, l["id"]))

    def _best_char(shot_text: str) -> Optional[int]:
        for name, cid in char_lookup:
            if name and name in shot_text:
                return cid
        return primary_char_id

    def _best_loc(shot_text: str) -> Optional[int]:
        for name, lid in loc_lookup:
            if name and name in shot_text:
                return lid
        return primary_loc_id

    human_modes = {"face", "body"}
    with _db() as conn, conn.cursor() as cur:
        for shot in styled_timeline:
            idx  = shot.get("shot_index") or shot.get("timeline_index")
            mode = shot.get("expression_mode", "environment")
            shot_text = " ".join(filter(None, [
                str(shot.get("visual_prompt")   or ""),
                str(shot.get("shot_meaning")    or ""),
                str(shot.get("character_name")  or ""),
            ])).lower()

            if mode in human_modes:
                char_id = _best_char(shot_text)
                loc_id  = None
            else:
                char_id = None
                loc_id  = _best_loc(shot_text)

            cur.execute(
                """
                UPDATE shot_assets
                   SET character_id = %s, location_id = %s
                 WHERE project_id = %s AND shot_index = %s
                """,
                (char_id, loc_id, project_id, idx),
            )
        conn.commit()
    logger.info("Linked shots to entities for project=%s", project_id)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── STAGED PIPELINE ──────────────────────────────────────────────────────────
# Each stage runs as a background job, saves its output, then STOPS.
# The user reviews the result and clicks "Continue" to advance to the next stage.

def _get_audio_duration(audio_path: Path) -> float | None:
    """Return exact audio duration in seconds using mutagen, or None on failure."""
    try:
        import mutagen
        f = mutagen.File(str(audio_path))
        if f and hasattr(f, "info") and hasattr(f.info, "length"):
            dur = float(f.info.length)
            logger.info("Audio duration from mutagen: %.3fs for %s", dur, audio_path.name)
            return dur
    except Exception as exc:
        logger.warning("mutagen duration read failed (%s)", exc)
    return None


def _merge_words_to_lines(lines: list[str], words: list[dict]) -> list[dict]:
    """Map Gemini lyric lines onto Whisper word-level timestamps proportionally.

    Each line i is mapped to the Whisper word at proportional position
    i / (N-1) × (W-1).  Because Whisper only emits words during actual vocals
    (it skips silence and instrumental sections), word[0].start naturally
    reflects the first vocal moment — preserving any instrumental intro offset
    without any special-casing.

    End timestamps are filled as the next line's start; the last line extends
    to the last word's end time.
    """
    n = len(lines)
    w = len(words)
    if n == 0 or w == 0:
        return [{"start": 0.0, "end": 0.0, "text": l} for l in lines]

    out: list[dict] = []
    for i, line in enumerate(lines):
        ratio = i / max(n - 1, 1)
        idx = min(round(ratio * (w - 1)), w - 1)
        out.append({
            "start": round(words[idx]["start"], 3),
            "end":   0.0,
            "text":  line,
        })

    # Fill end timestamps
    for i in range(len(out) - 1):
        out[i]["end"] = out[i + 1]["start"]
    out[-1]["end"] = round(words[-1]["end"], 3)

    return out


def _transcribe_hybrid(audio_path: Path, openai_key: str, project_id: str) -> tuple[str, list[dict]]:
    """Hybrid Gemini-timed + Whisper-anchored transcription.

    Strategy:
    1. Measure exact audio duration with mutagen.
    2. Gemini (PRIMARY): transcribe lyrics WITH per-line timestamps.
       Gemini hears the actual audio so its per-line timing is accurate —
       respecting varying durations (e.g. a 34-second sustained note vs a
       3-second rapid-fire line).  Prompt includes exact duration as a hard
       constraint to prevent timestamp compression.
    3. Whisper word-level (ANCHOR): run in parallel to detect the true vocal
       start time.  Because Whisper skips silence, word[0].start = first
       actual vocal moment.  If Gemini's first timestamp drifts by >3 s from
       Whisper's first word, shift ALL Gemini timestamps to match.
    4. Duration guard: if Gemini's last timestamp is <80 % of actual song
       length, rescale all timestamps proportionally.
    5. Fallbacks:
       - If Gemini timed call fails: fall back to Gemini plain-text + Whisper
         proportional word mapping.
       - If Whisper also fails: use Gemini plain-text with no timing.

    Returns (plain_text, timed_lines).  On failure returns ("", []).
    """
    # Step 1 — exact duration (used as constraint for Gemini and rescaling guard)
    exact_dur = _get_audio_duration(audio_path)

    # Step 2 — Gemini timed transcription (primary path)
    timed_from_gemini: list[dict] | None = _transcribe_gemini_timed(
        audio_path, project_id, exact_dur
    )

    # Step 3 — Whisper word-level timestamps (anchor + fallback timing)
    whisper_words: list[dict] = []
    whisper_segs: list[dict] = []
    try:
        from openai import OpenAI as _OpenAI
        _oai = _OpenAI(api_key=openai_key)
        with open(audio_path, "rb") as _af:
            _tr = _oai.audio.transcriptions.create(
                model="whisper-1",
                file=_af,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        for w in (getattr(_tr, "words", None) or []):
            whisper_words.append({
                "start": float(getattr(w, "start", 0.0) or 0.0),
                "end":   float(getattr(w, "end",   0.0) or 0.0),
            })
        for seg in (getattr(_tr, "segments", None) or []):
            t = (getattr(seg, "text", "") or "").strip()
            if t:
                whisper_segs.append({
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end":   float(getattr(seg, "end",   0.0) or 0.0),
                    "text":  t,
                })
        logger.info("Whisper: %d words, %d segs for project=%s",
                    len(whisper_words), len(whisper_segs), project_id)
    except Exception as exc:
        logger.warning("Whisper extraction failed (%s)", exc)

    # ── PRIMARY PATH: Gemini produced timed lines ─────────────────────────
    if timed_from_gemini is not None and len(timed_from_gemini) > 0:
        timed = timed_from_gemini

        # Step 3a — Instrumental-intro offset correction
        # If Gemini's first timestamp is too early vs Whisper's first vocal word
        if whisper_words:
            whisper_first = whisper_words[0]["start"]
            gemini_first  = timed[0]["start"]
            offset = whisper_first - gemini_first
            # Apply shift only when the discrepancy is significant (>3 s)
            if abs(offset) > 3.0:
                logger.info(
                    "Offset correction: shifting Gemini timestamps by %.2fs "
                    "(Gemini first=%.2fs, Whisper first=%.2fs) for project=%s",
                    offset, gemini_first, whisper_first, project_id,
                )
                timed = [
                    {
                        "start": round(max(0.0, t["start"] + offset), 3),
                        "end":   round(max(0.0, t["end"]   + offset), 3),
                        "text":  t["text"],
                    }
                    for t in timed
                ]

        # Step 3b — Duration compression guard
        if exact_dur and exact_dur > 0 and timed:
            last_start = timed[-1]["start"]
            if last_start > 0 and last_start < exact_dur * 0.80:
                # Rescale start times proportionally; keep end = next start
                scale = exact_dur / last_start
                logger.info(
                    "Rescaling Gemini timestamps ×%.3f "
                    "(last lyric at %.1fs, actual %.1fs) for project=%s",
                    scale, last_start, exact_dur, project_id,
                )
                timed = [
                    {"start": round(t["start"] * scale, 3),
                     "end":   round(t["end"]   * scale, 3),
                     "text":  t["text"]}
                    for t in timed
                ]
                # Clamp last end to actual duration
                timed[-1]["end"] = round(exact_dur, 3)

        text = "\n".join(t["text"] for t in timed if t["text"])
        logger.info(
            "Hybrid (Gemini-timed): %d lines, first=%.2fs, last=%.2fs for project=%s",
            len(timed), timed[0]["start"], timed[-1]["start"], project_id,
        )
        return text, timed

    # ── FALLBACK: Gemini plain-text + Whisper proportional word mapping ───
    logger.warning("Gemini timed failed — falling back to plain-text + Whisper merge for project=%s", project_id)

    gemini_lines: list[str] = []
    gtxt = _transcribe_with_gemini(audio_path, project_id)
    if gtxt:
        gemini_lines = [l.strip() for l in gtxt.splitlines() if l.strip()]

    if not gemini_lines and not whisper_words and not whisper_segs:
        return "", []

    if not gemini_lines:
        if whisper_segs:
            return "\n".join(s["text"] for s in whisper_segs), whisper_segs
        return "", []

    if not whisper_words and not whisper_segs:
        timed = [{"start": 0.0, "end": 0.0, "text": l} for l in gemini_lines]
        return "\n".join(gemini_lines), timed

    if whisper_words:
        timed = _merge_words_to_lines(gemini_lines, whisper_words)
    else:
        timed = _align_lines_to_segments(gemini_lines, whisper_segs)

    text = "\n".join(t["text"] for t in timed if t["text"])
    logger.info(
        "Hybrid (fallback word-map): %d lines for project=%s", len(timed), project_id,
    )
    return text, timed


def _word_bag(text: str) -> set:
    """Return a set of lowercase unicode words from text.

    Works for any script (Gurmukhi, Devanagari, Latin, etc.) because
    re.UNICODE is used and \\w matches characters in all Unicode word classes.
    """
    import re
    return set(re.findall(r'\w+', text.lower(), re.UNICODE))


def _overlap_score(a: set, b: set) -> float:
    """Jaccard word-overlap between two word bags. Returns 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _align_lines_to_segments(lines: list[str], segs: list[dict]) -> list[dict]:
    """Map each Gemini lyric line onto Whisper's timestamped segments using
    forward-greedy word-overlap alignment.

    Why: Whisper gives correct timestamps for every segment including the
    pre-lyric instrumental offset (e.g. first vocal at 45s, not 0s).
    Gemini may produce a different number of lines. The old proportional
    fallback smeared lines evenly, ignoring actual word positions. This
    replacement finds the Whisper segment where each Gemini line's words
    actually appear and borrows its timestamps directly.

    Algorithm:
    1. Fast path: counts match → 1:1 assignment (unchanged).
    2. Otherwise: forward-greedy Jaccard matching.
       For each Gemini line, scan Whisper segments from the current
       pointer forward (monotone — time never goes backwards). Pick the
       segment with the highest word overlap above a minimum threshold.
       Advance the pointer past that segment.
    3. Unmatched lines (humming, pure melody, no word overlap found):
       timestamps are interpolated linearly between the nearest matched
       neighbours, bounded by the Whisper timeline endpoints.
    """
    n_lines = len(lines)
    n_segs = len(segs)
    if n_lines == 0 or n_segs == 0:
        return [{"start": 0.0, "end": 0.0, "text": l} for l in lines]

    # Fast path: perfect count match — Gemini text + Whisper timing 1:1
    if n_lines == n_segs:
        return [
            {"start": segs[i]["start"], "end": segs[i]["end"], "text": lines[i]}
            for i in range(n_lines)
        ]

    # Pre-compute word bags for all Whisper segments and all Gemini lines
    seg_bags = [_word_bag(s["text"]) for s in segs]
    line_bags = [_word_bag(l) for l in lines]

    # Forward-greedy matching — matched[i] = index into segs, or None
    matched: list = [None] * n_lines
    seg_ptr = 0  # monotone pointer — never goes backwards

    for i, lb in enumerate(line_bags):
        if not lb:
            continue  # empty / humming line — will interpolate
        best_score = 0.0
        best_seg_idx = None
        for j in range(seg_ptr, n_segs):
            score = _overlap_score(lb, seg_bags[j])
            if score > best_score:
                best_score = score
                best_seg_idx = j
        if best_score >= 0.1 and best_seg_idx is not None:
            matched[i] = best_seg_idx
            seg_ptr = best_seg_idx + 1  # consume this segment, advance

    # Build output list; unmatched entries carry None timestamps for now
    t_first = segs[0]["start"]
    t_last = segs[-1]["end"]

    out: list[dict] = []
    for i, line in enumerate(lines):
        if matched[i] is not None:
            seg = segs[matched[i]]
            out.append({"start": seg["start"], "end": seg["end"], "text": line})
        else:
            out.append({"start": None, "end": None, "text": line})

    # Build anchor list from matched entries plus boundary sentinels
    anchors = [(-1, t_first, t_first)]
    for i in range(n_lines):
        if out[i]["start"] is not None:
            anchors.append((i, out[i]["start"], out[i]["end"]))
    anchors.append((n_lines, t_last, t_last))

    # Interpolate each run of consecutive unmatched lines between its anchors
    for k in range(len(anchors) - 1):
        a_idx, _, a_end = anchors[k]
        b_idx, b_start, _ = anchors[k + 1]
        gap = [i for i in range(a_idx + 1, b_idx) if out[i]["start"] is None]
        if not gap:
            continue
        span = max(b_start - a_end, 0.001)
        n_gap = len(gap)
        for pos, i in enumerate(gap):
            s = a_end + span * (pos / n_gap)
            e = a_end + span * ((pos + 1) / n_gap)
            out[i]["start"] = round(s, 3)
            out[i]["end"] = round(e, 3)

    return out


def _parse_gemini_ts(ts: str) -> float:
    """Parse M:SS.mmm timestamp string → seconds. Returns 0.0 on failure."""
    import re
    clean = ts.replace("[", "").replace("]", "").strip()
    m = re.match(r"^(\d+):(\d{2})(?:[.,](\d+))?$", clean)
    if not m:
        return 0.0
    mins = int(m.group(1))
    secs = int(m.group(2))
    frac = (m.group(3) or "0").ljust(3, "0")[:3]
    return mins * 60 + secs + int(frac) / 1000.0


def _transcribe_gemini_timed(
    audio_path: Path,
    project_id: str,
    exact_dur: float | None = None,
) -> list[dict] | None:
    """Ask Gemini to transcribe the song AND timestamp each lyric line.

    Gemini listens to the actual audio so it can hear when each line is sung.
    This gives accurate per-line timing that respects varying line durations
    (e.g. a 34-second sustained note vs a 3-second rapid-fire line).

    Returns a list of {"start": float, "end": float, "text": str} dicts,
    or None on failure (caller can fall back to plain-text path).
    """
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logger.warning("GEMINI_API_KEY not set — skipping Gemini timed transcription")
            return None

        from google import genai
        import json, re

        client = genai.Client(api_key=gemini_key)
        uploaded = client.files.upload(file=str(audio_path))

        dur_constraint = ""
        if exact_dur and exact_dur > 0:
            mins = int(exact_dur // 60)
            secs = int(exact_dur % 60)
            pct80 = exact_dur * 0.80
            dur_constraint = (
                f"\nAUDIO LENGTH CONSTRAINT (measured from file — this is exact):\n"
                f"- Track is EXACTLY {exact_dur:.3f} seconds ({mins}m {secs:02d}s)\n"
                f"- totalDurationSeconds MUST equal {exact_dur:.3f}\n"
                f"- Your LAST lyric timestamp MUST be after {pct80:.1f}s\n"
            )

        prompt = (
            "You are a professional audio transcriptionist specialising in music lyrics.\n"
            "An audio file is attached. Listen to the ENTIRE track from start to finish and "
            "transcribe every lyric line with accurate timestamps.\n"
            f"{dur_constraint}\n"
            "Return ONLY this JSON (no markdown, no explanation):\n"
            "{\n"
            '  "lines": [\n'
            '    {"timestamp": "M:SS.mmm", "text": "<exact lyric in original script>"}\n'
            "  ],\n"
            '  "totalDurationSeconds": <true full length>,\n'
            '  "language": "<e.g. Punjabi>",\n'
            '  "isInstrumental": false\n'
            "}\n\n"
            "RULES:\n"
            "- timestamp format: M:SS.mmm  e.g. \"0:15.000\" or \"1:23.450\"\n"
            "- One object per lyric line — do NOT group multiple lines\n"
            "- Preserve ALL repeats (chorus, hook, ad-libs)\n"
            "- Timestamps must be spread across the FULL duration\n"
            "- Use the ORIGINAL script (Gurmukhi, Devanagari, Arabic, etc.) — no transliteration\n"
            "- If purely instrumental with no vocals: isInstrumental=true, lines=[]\n"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, uploaded],
        )

        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        raw = (response.text or "").strip()
        if not raw:
            logger.warning("Gemini timed: empty response for project=%s", project_id)
            return None

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
        raw = raw.strip()
        # Extract JSON object
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            raw = raw[start_idx:end_idx + 1]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Light repair: fix trailing commas
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                parsed = json.loads(raw)
            except Exception as e2:
                logger.warning("Gemini timed: JSON parse failed (%s) for project=%s", e2, project_id)
                return None

        if parsed.get("isInstrumental"):
            logger.info("Gemini timed: track is instrumental for project=%s", project_id)
            return []

        lines_raw = parsed.get("lines") or []
        if not lines_raw:
            logger.warning("Gemini timed: no lines returned for project=%s", project_id)
            return None

        # Parse timestamps into seconds
        timed: list[dict] = []
        for item in lines_raw:
            ts_str = str(item.get("timestamp") or "0:00.000")
            text = (item.get("text") or item.get("line") or "").strip()
            if not text:
                continue
            timed.append({"start": _parse_gemini_ts(ts_str), "end": 0.0, "text": text})

        if not timed:
            return None

        # Fill end timestamps (each line ends where the next begins)
        for i in range(len(timed) - 1):
            timed[i]["end"] = timed[i + 1]["start"]
        # Last line: use totalDurationSeconds if available, else exact_dur, else +5s
        total = float(parsed.get("totalDurationSeconds") or 0)
        timed[-1]["end"] = total or exact_dur or (timed[-1]["start"] + 5.0)

        logger.info(
            "Gemini timed: %d lines, first=%.2fs, last=%.2fs for project=%s",
            len(timed), timed[0]["start"], timed[-1]["start"], project_id,
        )
        return timed

    except Exception as exc:
        logger.warning("Gemini timed transcription failed (%s)", exc)
        return None


def _transcribe_with_gemini(audio_path: Path, project_id: str) -> str:
    """Plain-text Gemini transcription fallback (no timestamps).

    Used when timed transcription is unavailable. Returns "" on failure.
    """
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return ""

        from google import genai
        client = genai.Client(api_key=gemini_key)

        uploaded = client.files.upload(file=str(audio_path))

        prompt = (
            "Transcribe the lyrics of this song faithfully in the original language and script. "
            "Output the lyrics as ONE lyrical line per row, separated by newlines. "
            "Each row should be a single natural lyrical phrase or vocal line as it is sung. "
            "Preserve repeats (do not deduplicate). "
            "Do NOT include timestamps, line numbers, brackets, translations, or commentary. "
            "Output ONLY the raw lyric lines, nothing else. "
            "If the audio is purely instrumental with no vocals, output exactly: INSTRUMENTAL"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, uploaded],
        )

        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        text = (response.text or "").strip()
        if not text or text.upper().startswith("INSTRUMENTAL"):
            return ""

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cleaned = "\n".join(lines)
        logger.info("Gemini plain: %d lines for project=%s", len(lines), project_id)
        return cleaned

    except Exception as exc:
        logger.warning("Gemini plain transcription failed (%s)", exc)
        return ""


def _transcribe_with_whisper(audio_path: Path, openai_key: str, project_id: str) -> str:
    """Whisper fallback. Uses verbose_json so we can join segments with newlines."""
    try:
        from openai import OpenAI as _OpenAI
        _oai = _OpenAI(api_key=openai_key)
        with open(audio_path, "rb") as _af:
            _tr = _oai.audio.transcriptions.create(
                model="whisper-1", file=_af, response_format="verbose_json",
            )
        segs = getattr(_tr, "segments", None) or []
        if segs:
            lines = [(getattr(s, "text", "") or "").strip() for s in segs]
            text = "\n".join(l for l in lines if l)
        else:
            text = (getattr(_tr, "text", "") or "").strip()
        logger.info("Whisper: %d chars, %d segments for project=%s",
                    len(text), len(segs), project_id)
        return text
    except Exception as exc:
        logger.warning("Whisper failed (%s)", exc)
        return ""


def _normalize_lyrics_for_context(text: str) -> str:
    """Ensure lyrics have line breaks so the context engine produces multiple shots.

    The context engine treats each line of text as one beat of meaning. If the user
    pastes a song as one giant paragraph (or Whisper returns no segments), we end
    up with line_meanings = [1 entry] and the entire pipeline collapses to 1 shot.

    Strategy:
    - If the text already has 3+ newlines, leave it alone (user formatted it).
    - Otherwise, split on sentence punctuation (. ? ! ।) and on the common
      Devanagari/Punjabi virama-style breaks.
    - Fall back to splitting every ~80 characters on word boundaries so even
      punctuation-free lyrics get broken up.
    """
    import re
    if not text:
        return text
    text = text.strip()
    # Already has structure
    if text.count("\n") >= 3:
        return text

    # First pass: split on sentence/clause punctuation (Latin + Devanagari danda)
    parts = re.split(r"(?<=[.!?।])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    # If still effectively one big chunk, fall back to soft wrapping at ~80 chars.
    if len(parts) <= 2 and len(text) > 200:
        words = text.split()
        parts = []
        line = []
        line_len = 0
        for w in words:
            if line_len + len(w) + 1 > 80 and line:
                parts.append(" ".join(line))
                line = [w]
                line_len = len(w)
            else:
                line.append(w)
                line_len += len(w) + 1
        if line:
            parts.append(" ".join(line))

    return "\n".join(parts) if parts else text


def _stage0_job(project_id: str, audio_path: Optional[Path], text: str, genre: str) -> None:
    """Stage 0 — Acoustic Audit: extract audio features + Whisper transcription."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        _set_status(project_id, "running",
                    {"stage": "audio", "label": "Reading audio file…"}, stage="running_0")

        audio_data: dict = {}
        pre_analysis: dict = {}
        transcript = text

        if audio_path and audio_path.is_file():
            proc = AudioProcessor()
            audio_data = proc.extract_features(str(audio_path))
            pre_analysis = proc.build_context_pre_analysis(audio_data)

        timed_lyrics: list[dict] = []
        if not transcript and audio_path and audio_path.is_file():
            _set_status(project_id, "running",
                        {"stage": "audio", "label": "Transcribing lyrics (Gemini + Whisper)…"},
                        stage="running_0")
            transcript, timed_lyrics = _transcribe_hybrid(audio_path, api_key, project_id)

            # Last-ditch fallback if hybrid totally failed
            if not transcript:
                transcript = _transcribe_with_whisper(audio_path, api_key, project_id)

        if not transcript:
            transcript = "Instrumental audio track. Cinematic, atmospheric, beat-driven."

        # Ensure the lyrics have line breaks so the context engine can produce
        # multiple shots. A paragraph with no newlines collapses into 1 shot.
        transcript = _normalize_lyrics_for_context(transcript)

        # ── Stage 1: Input Processing ──────────────────────────────────────
        # Run InputProcessor to produce a clean, structured input_packet.
        # This is purely structural — no meaning, no visuals.
        ip = InputProcessor()
        input_packet = ip.process(
            raw_text=transcript,
            genre_hint=genre,
            audio_meta=audio_data if audio_data else {},
            timed_segments=timed_lyrics if timed_lyrics else [],
        )
        logger.info(
            "InputProcessor: type=%s sub_type=%s sections=%d units=%d langs=%s",
            input_packet.get("input_type"),
            input_packet.get("sub_type"),
            len(input_packet.get("sections") or []),
            len(input_packet.get("units") or []),
            input_packet.get("languages"),
        )

        # Persist audio data + transcript + input_packet (+ timed lyrics if available)
        # Also write Stage 0 and Stage 1-Input namespaces into the Project Brain.
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE projects ADD COLUMN IF NOT EXISTS lyrics_timed JSONB;"
            )
            cur.execute(
                "ALTER TABLE projects ADD COLUMN IF NOT EXISTS input_packet JSONB;"
            )
            cur.execute(
                "UPDATE projects SET audio_data=%s, transcript=%s, text=%s, "
                "lyrics_timed=%s, input_packet=%s, updated_at=NOW() WHERE id=%s",
                (Json({**audio_data, "_pre_analysis": pre_analysis}),
                 transcript, transcript,
                 Json(timed_lyrics) if timed_lyrics else None,
                 Json(input_packet),
                 project_id),
            )
            conn.commit()

        # ── Project Brain: write Stage 0 + Stage 1-Input namespaces ───────────
        # Also write project_settings so every downstream stage (including
        # Narrative Engine) can read platform/duration/genre without touching DB.
        _audio_duration = float(audio_data.get("duration_seconds") or 0) or None
        with _db() as conn:
            brain = ProjectBrain(project_id)
            brain.write("raw_input", {
                "text":         transcript,
                "genre":        genre,
                "audio_meta":   audio_data,
                "pre_analysis": pre_analysis,
                "timed_lyrics": timed_lyrics,
            })
            brain.write("input_structure", input_packet)
            brain.write("project_settings", {
                "genre":            genre,
                "duration_seconds": _audio_duration,
                "platform":         None,
                "style_preset":     None,
            })
            brain.save(conn)
            conn.commit()
        logger.info(
            "ProjectBrain: wrote raw_input + input_structure + project_settings "
            "for project=%s", project_id
        )

        _set_status(project_id, "awaiting_review",
                    {"stage": "audio", "label": "Acoustic audit complete. Review and continue."},
                    stage="audio_review")

    except Exception as exc:
        logger.exception("Stage 0 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Stage 0 failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage_style_job(project_id: str) -> None:
    """Style Profile Engine (Stage 4) — suggest 2-3 style profiles, park at style_review.

    Reads from Project Brain (per master spec):
      • context_packet  (Stage 2) — locked meaning, world, speaker, motivation
      • narrative_packet (Stage 3) — storytelling mode, perspective, motifs
    These are passed to the engine so suggestions are grounded in the locked
    story decisions, not inferred from lyrics + audio alone.

    Falls back to legacy text/genre/audio-only mode for pre-brain projects.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        # ── Read raw row + brain in parallel ─────────────────────────────────
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT text, genre, audio_data FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        text = row["text"] or ""
        genre = row["genre"] or "song"
        audio_data_blob = dict(row.get("audio_data") or {})
        audio_data_blob.pop("_pre_analysis", None)

        # ── Brain reads (defensive — empty dicts if pre-brain project) ──────
        try:
            with _db() as conn:
                brain = ProjectBrain.load(project_id, conn)
            context_packet   = brain.read("context_packet") or {}
            narrative_packet = brain.read("narrative_packet") or {}
        except Exception:
            logger.exception(
                "Stage style: failed to load ProjectBrain for project=%s — "
                "falling back to lyrics+audio only.", project_id
            )
            context_packet, narrative_packet = {}, {}

        # Legacy fallback: if brain has no context_packet, try the JSONB column.
        if not context_packet:
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT context_packet, narrative_packet FROM projects WHERE id=%s",
                    (project_id,),
                )
                _legacy = cur.fetchone()
            if _legacy:
                context_packet   = dict(_legacy.get("context_packet") or {})
                if not narrative_packet:
                    narrative_packet = dict(_legacy.get("narrative_packet") or {})

        if context_packet:
            logger.info(
                "Stage style: using context_packet (%d keys) + narrative_packet "
                "(%d keys) from brain for project=%s",
                len(context_packet), len(narrative_packet), project_id,
            )
        else:
            logger.info(
                "Stage style: no upstream packets available for project=%s — "
                "running in lyrics+audio fallback mode.", project_id
            )

        # Surface the user-confirmed (or auto-detected) singer gender so the
        # style engine can tailor its suggestions to female-led / male-led /
        # mixed framings instead of guessing from lyrics alone.
        # `vocal_gender_final` (set by the Audio Review screen) wins. If the
        # user explicitly chose "instrumental", we strip vocal_gender entirely
        # so the prompt does not leak the auto-detected gender.
        _vg_final = str(audio_data_blob.get("vocal_gender_final") or "").strip().lower()
        _vg_detected = str(audio_data_blob.get("vocal_gender") or "").strip().lower()
        if _vg_final in ("male", "female", "mixed"):
            audio_data_blob["vocal_gender"] = _vg_final
        elif _vg_final == "instrumental":
            audio_data_blob.pop("vocal_gender", None)
        elif _vg_detected not in ("male", "female", "mixed"):
            audio_data_blob.pop("vocal_gender", None)

        _set_status(project_id, "running",
                    {"stage": "style", "label": "Analysing content for style suggestions…"},
                    stage="running_style")

        engine = StyleProfileEngine(api_key)
        suggestions = _run_async(engine.suggest(
            text=text,
            genre=genre,
            audio_analytics=audio_data_blob,
            context_packet=context_packet,
            narrative_packet=narrative_packet,
        ))

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET style_suggestions=%s, updated_at=NOW() WHERE id=%s",
                (Json(suggestions), project_id),
            )
            conn.commit()

        _set_status(project_id, "awaiting_review",
                    {"stage": "style",
                     "label": f"{len(suggestions)} style options ready. Choose your visual direction."},
                    stage="style_review")

    except Exception as exc:
        logger.exception("Style stage failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Style analysis failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage1_job(project_id: str) -> None:
    """Stage 1 — Context Engine: 5W Audit → context_packet."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        # ── Read from Project Brain (Stage 2 reads Stage 1 output) ──────────
        with _db() as conn:
            brain = ProjectBrain.load(project_id, conn)

        input_structure = brain.read("input_structure")  # Stage 1 output
        raw_input       = brain.read("raw_input")        # Stage 0 output

        # Fallback: if brain is empty (project predates brain), read from columns
        if not input_structure:
            logger.warning(
                "Stage 1: brain has no input_structure for project=%s — "
                "falling back to legacy columns.", project_id
            )
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT text, genre, audio_data, input_packet FROM projects WHERE id=%s",
                    (project_id,),
                )
                row = cur.fetchone()
            if not row:
                raise RuntimeError("Project not found.")
            text       = row["text"] or ""
            genre      = row["genre"] or "song"
            audio_blob = dict(row.get("audio_data") or {})
            pre_analysis = audio_blob.pop("_pre_analysis", {})
            input_structure = dict(row.get("input_packet") or {})
        else:
            text         = raw_input.get("text", "")
            genre        = raw_input.get("genre", "song")
            pre_analysis = raw_input.get("pre_analysis", {})

        _set_status(project_id, "running",
                    {"stage": "context", "label": "Running 5W Context Audit…"}, stage="running_1")

        orchestrator = ProductionOrchestrator(api_key)
        context_packet = _run_async(
            orchestrator.run_context_only(
                text=text,
                genre=genre,
                pre_analysis=pre_analysis,
                input_packet=input_structure or None,
            )
        )

        # ── Write context_packet to brain + keep legacy column in sync ───────
        with _db() as conn:
            brain = ProjectBrain.load(project_id, conn)
            brain.write("context_packet", context_packet)
            brain.save(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE projects SET context_packet=%s, updated_at=NOW() WHERE id=%s",
                    (Json(context_packet), project_id),
                )
            conn.commit()
        logger.info("ProjectBrain: wrote context_packet for project=%s", project_id)

        _set_status(project_id, "awaiting_review",
                    {"stage": "context", "label": "Context analysis complete. Review and continue."},
                    stage="context_review")

    except Exception as exc:
        logger.exception("Stage 1 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Stage 1 failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage_narrative_job(project_id: str) -> None:
    """Stage 3 — Narrative Intelligence: reads context_packet from brain,
    calls generate_narrative_intelligence(), writes narrative_packet to brain,
    parks at narrative_review."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        # ── Read from Project Brain ──────────────────────────────────────────
        # Stage 3 reads: context_packet (Stage 2), input_structure (Stage 1),
        # raw_input (Stage 0), and project_settings (Stage 0).
        # All must be written to the brain by upstream stages before this runs.
        with _db() as conn:
            brain = ProjectBrain.load(project_id, conn)

        context_packet   = brain.read("context_packet")
        input_structure  = brain.read("input_structure")
        project_settings = brain.read("project_settings")
        # raw_input available for nuance verification if needed
        raw_input_data   = brain.read("raw_input")

        # Fallback: if brain has no context_packet (pre-brain project), read
        # from legacy columns. input_structure and project_settings will be
        # empty dicts — narrative engine handles both gracefully.
        if not context_packet:
            logger.warning(
                "Stage narrative: brain has no context_packet for project=%s — "
                "falling back to legacy column.", project_id
            )
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT context_packet, genre, audio_data FROM projects WHERE id=%s",
                    (project_id,),
                )
                row = cur.fetchone()
            if not row:
                raise RuntimeError("Project not found.")
            context_packet = dict(row.get("context_packet") or {})
            _audio = dict(row.get("audio_data") or {})
            project_settings = {
                "genre":            row.get("genre") or "song",
                "duration_seconds": _audio.get("duration_seconds"),
                "platform":         None,
                "style_preset":     None,
            }

        # Merge style_preset into project_settings if brain project_settings
        # was written before style was chosen (style_preset=None at Stage 0).
        if not project_settings.get("style_preset"):
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT style_profile FROM projects WHERE id=%s",
                    (project_id,),
                )
                sp_row = cur.fetchone()
            if sp_row:
                _sp = dict(sp_row.get("style_profile") or {})
                if _sp.get("preset"):
                    project_settings = dict(project_settings)
                    project_settings["style_preset"] = _sp["preset"]

        _set_status(project_id, "running",
                    {"stage": "narrative", "label": "Running Narrative Intelligence…"},
                    stage="running_narrative")

        narrative_packet = _run_async(
            generate_narrative_intelligence(
                api_key=api_key,
                context_packet=context_packet,
                input_structure=input_structure or {},
                project_settings=project_settings or {},
            )
        )

        # ── Write narrative_packet to brain + legacy column ──────────────────
        with _db() as conn:
            brain = ProjectBrain.load(project_id, conn)
            brain.write("narrative_packet", narrative_packet)
            brain.save(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS "
                    "narrative_packet JSONB;"
                )
                cur.execute(
                    "UPDATE projects SET narrative_packet=%s, updated_at=NOW() "
                    "WHERE id=%s",
                    (Json(narrative_packet), project_id),
                )
            conn.commit()
        logger.info("ProjectBrain: wrote narrative_packet for project=%s", project_id)

        _set_status(project_id, "awaiting_review",
                    {"stage": "narrative", "label": "Narrative intelligence ready. Review and continue."},
                    stage="narrative_review")

    except Exception as exc:
        logger.exception("Stage narrative failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Narrative stage failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage_brief_job(project_id: str, overrides: dict) -> None:
    """Creative Brief Engine v2 (Stage 6) — controlled selection + variation anchoring.

    The FIRST commitment layer. Reads the storyboard's possibilities and
    locks ONE direction per scene with full per-scene execution intent
    (subject_focus, environment_type, character_presence, key_elements,
    emotional_state/intensity, lighting_condition, movement_type,
    timeline_mode, motion_density, repetition_handling, motif_usage,
    continuity_hooks). Parks at creative_brief_review for user yes/regenerate.

    Reads from Project Brain (per master spec):
      • storyboard_packet (Stage 5) — scenes + valid_realizations  ← REQUIRED
      • narrative_packet  (Stage 3) — story logic to honor
      • context_packet    (Stage 2) — locked meaning, world, speaker
      • style_packet      (Stage 4) — high-level visual guidance only
      • input_structure   (Stage 1) — section type, repetition_map (light)
      • project_settings  (Stage 0) — duration, platform, prefs (optional)
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        # ── Brain reads (defensive — empty dicts if pre-brain project) ──────
        try:
            with _db() as conn:
                brain = ProjectBrain.load(project_id, conn)
            brain_storyboard = brain.read("storyboard_packet") or {}
            brain_narrative  = brain.read("narrative_packet")  or {}
            brain_context    = brain.read("context_packet")    or {}
            brain_style      = brain.read("style_packet")      or {}
            brain_input      = brain.read("input_structure")   or {}
            brain_settings   = brain.read("project_settings")  or {}
        except Exception:
            logger.exception(
                "Stage brief v2: failed to load ProjectBrain for project=%s",
                project_id,
            )
            brain_storyboard = {}
            brain_narrative  = {}
            brain_context    = {}
            brain_style      = {}
            brain_input      = {}
            brain_settings   = {}

        # Legacy column fallbacks for context/narrative/style only —
        # storyboard_packet must come from the brain (v2 intent layer).
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT context_packet, narrative_packet, style_profile "
                "FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        context_packet   = brain_context   or dict(row.get("context_packet")   or {})
        narrative_packet = brain_narrative or dict(row.get("narrative_packet") or {})
        style_profile    = (brain_style
                            or dict(row.get("style_profile") or {})
                            or StyleProfileRegistry.default_style_profile())

        scenes = list(brain_storyboard.get("scenes") or [])

        logger.info(
            "Stage brief v2: brain inputs storyboard.scenes=%d narrative=%s "
            "context=%s style=%s for project=%s",
            len(scenes),
            "yes" if brain_narrative else ("fallback" if narrative_packet else "none"),
            "yes" if brain_context   else ("fallback" if context_packet   else "none"),
            "yes" if brain_style     else "fallback",
            project_id,
        )

        _set_status(project_id, "running",
                    {"stage": "brief",
                     "label": "Locking ONE direction per scene…"},
                    stage="running_brief")

        from creative_brief_engine_v2 import generate_creative_brief_v2
        scene_briefs, used_fallback = _run_async(generate_creative_brief_v2(
            api_key=api_key,
            storyboard_scenes=scenes,
            narrative_packet=narrative_packet,
            context_packet=context_packet,
            style_profile=style_profile,
            input_structure=brain_input,
            project_settings=brain_settings,
        ))

        # ── Write creative_briefs to brain (Stage 6 namespace) ──────────────
        creative_brief_packet = {
            "schema_version": 2,
            "scenes":         scene_briefs,
            "used_fallback":  used_fallback,
            "approved":       False,
        }
        try:
            with _db() as conn:
                brain = ProjectBrain.load(project_id, conn)
                brain.write("creative_briefs", creative_brief_packet)
                brain.save(conn)
                conn.commit()
            logger.info(
                "ProjectBrain: wrote creative_briefs v2 for project=%s "
                "(%d scenes, fallback=%s)",
                project_id, len(scene_briefs), used_fallback,
            )
        except Exception:
            logger.exception(
                "Stage brief v2: failed to write creative_briefs to brain for project=%s",
                project_id,
            )

        # ── Mirror into legacy context_packet.creative_brief for the UI ─────
        # Templates that haven't migrated to brain reads still pull from here.
        creative_brief_legacy = dict(context_packet.get("creative_brief") or {})
        creative_brief_legacy["schema_version"] = 2
        creative_brief_legacy["scenes"]         = scene_briefs
        creative_brief_legacy["used_fallback"]  = used_fallback
        # Drop legacy v1 keys — they no longer apply under v2.
        creative_brief_legacy.pop("variants", None)
        creative_brief_legacy.pop("chosen", None)
        creative_brief_legacy.pop("_pending_overrides", None)
        context_packet["creative_brief"] = creative_brief_legacy

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET context_packet=%s, updated_at=NOW() WHERE id=%s",
                (Json(context_packet), project_id),
            )
            conn.commit()

        _set_status(project_id, "awaiting_review",
                    {"stage": "brief",
                     "label": (f"Creative Brief ready ({len(scene_briefs)} scenes "
                               f"locked). Review and approve."
                               if scene_briefs else
                               "Creative Brief produced no scenes — please regenerate.")},
                    stage="creative_brief_review")

    except Exception as exc:
        logger.exception("Stage Brief v2 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Creative Brief stage failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage2_job(project_id: str, name: str, overrides: dict) -> None:
    """Storyboard (Stage 5) — VisualStoryboard → RhythmicAssembly → StyleGrading.

    Reads from Project Brain (per master spec):
      • context_packet   (Stage 2)
      • narrative_packet (Stage 3)
      • style_packet     (Stage 4)
      • input_structure / project_settings (root-level intent inputs)
    Storyboard is the POSSIBILITIES layer — it MUST NOT read
    creative_briefs (Stage 6 = SELECTION). Brief runs after this stage
    and picks one realization per scene.
    Falls back to legacy JSONB columns per-namespace for pre-brain projects.
    On success, writes brain.storyboard_packet (Stage 5 namespace).
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Replit secrets.")

        # ── Brain reads (defensive — empty if pre-brain project) ───────────
        try:
            with _db() as conn:
                brain = ProjectBrain.load(project_id, conn)
            brain_context   = brain.read("context_packet") or {}
            brain_narrative = brain.read("narrative_packet") or {}
            brain_style     = brain.read("style_packet") or {}
            brain_input     = brain.read("input_structure") or {}
            brain_settings  = brain.read("project_settings") or {}
        except Exception:
            logger.exception(
                "Stage storyboard: failed to load ProjectBrain for project=%s — "
                "falling back to legacy columns.", project_id
            )
            brain_context, brain_narrative, brain_style = {}, {}, {}
            brain_input, brain_settings = {}, {}

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT text, genre, audio_data, context_packet, narrative_packet,"
                " style_profile, lyrics_timed, input_packet FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        text = row["text"] or ""
        genre = row["genre"] or "song"
        audio_data_blob = dict(row.get("audio_data") or {})
        audio_data_blob.pop("_pre_analysis", None)
        timed_lyrics: list[dict] = list(row.get("lyrics_timed") or [])

        # Resolve each legacy packet independently so per-namespace fallback
        # never depends on whether a different namespace fell back or not.
        _legacy_context = dict(row.get("context_packet") or {})
        _legacy_narr    = dict(row.get("narrative_packet") or {})
        _legacy_sp      = dict(row.get("style_profile") or {})
        _legacy_input   = dict(row.get("input_packet") or {})
        # Resolved packets used by the v2 storyboard intent layer.
        input_structure  = brain_input    or _legacy_input
        project_settings = brain_settings or {}

        # Brain wins, legacy fills gaps — pre-brain projects keep working.
        context_packet   = brain_context or _legacy_context
        # narrative_packet read but not directly used here — orchestrator's
        # storyboard engine consumes it indirectly via context. Logged for trace.
        narrative_packet = brain_narrative or _legacy_narr
        style_profile    = brain_style or _legacy_sp or StyleProfileRegistry.default_style_profile()
        # NOTE: Storyboard MUST NOT read brain.creative_briefs — under the
        # master spec, Storyboard runs BEFORE Creative Brief (storyboard is
        # POSSIBILITIES, brief is SELECTION). Reading the brief here would
        # leak stale data from a prior run/rerun and constrain storyboard.

        # Apply user overrides
        if overrides.get("speaker_name"):
            speaker = dict(context_packet.get("speaker") or {})
            speaker["name"] = overrides["speaker_name"]
            context_packet["speaker"] = speaker
        if overrides.get("location"):
            context_packet["location_dna"] = overrides["location"]
        if overrides.get("era"):
            context_packet["era"] = overrides["era"]

        # Derive style_preset from the resolved style_profile (always populated above),
        # allowing the override param to trump if explicitly set.
        style_preset = overrides.get("style_preset") or style_profile.get("preset") or "cinematic_natural"

        logger.info(
            "Stage storyboard: brain inputs context=%s narrative=%s style=%s for project=%s",
            "yes" if brain_context else "fallback",
            "yes" if brain_narrative else ("fallback" if narrative_packet else "none"),
            "yes" if brain_style else "fallback",
            project_id,
        )

        _set_status(project_id, "running",
                    {"stage": "storyboard", "label": "Building visual storyboard…"}, stage="running_2")

        # ── Storyboard Engine v2 (master spec — pure intent layer) ─────────
        # Produces scenes with valid_realizations from brain inputs. Output
        # is written to brain.storyboard_packet.scenes alongside the legacy
        # locked timeline below — downstream stages (assembly/refs/stills/
        # videos) still consume the legacy timeline until they're migrated.
        #
        # Run v2 in a background thread so its LLM latency overlaps with the
        # legacy orchestrator's work. v2 failure NEVER blocks legacy: the
        # holder dict captures success or fallback, and the join below is
        # bounded so a hung v2 thread cannot stall this stage.
        import threading
        _v2_holder: dict = {"scenes": [], "used_fallback": False, "ok": False}

        def _run_v2() -> None:
            try:
                from storyboard_engine_v2 import generate_storyboard_v2
                scenes, used_fallback = _run_async(generate_storyboard_v2(
                    api_key=api_key,
                    input_structure=input_structure,
                    context_packet=context_packet,
                    narrative_packet=narrative_packet,
                    style_profile=style_profile,
                    project_settings=project_settings,
                ))
                _v2_holder["scenes"]        = scenes
                _v2_holder["used_fallback"] = used_fallback
                _v2_holder["ok"]            = True
            except Exception:
                logger.exception(
                    "[StoryboardV2] thread failed for project=%s — "
                    "intent layer will be empty in brain.",
                    project_id,
                )

        _v2_thread = threading.Thread(
            target=_run_v2,
            name=f"storyboard-v2-{project_id}",
            daemon=True,
        )
        _v2_thread.start()

        orchestrator = ProductionOrchestrator(api_key)

        # Run storyboard + assembly + style using the orchestrator's staged methods.
        # MM3.1 Cinematic Beat Engine integration point:
        #   ProductionOrchestrator.run_to_timeline()
        #     → VisualStoryboardEngine.build_storyboard()
        #       → _attach_optional_cinematic_layers()
        #           → CinematicBeatEngine.generate_beats()      [emotion→behaviour]
        #           → BehaviourMapper + ShotEventBuilder        [behaviour→events]
        #           → GenericShotValidator.validate_sequence()  [mark+rewrite weak shots]
        #           → ShotVarietyEngine.assign_shot_types()     [enforce target distribution]
        #           → _enforce_variety_caps() post-pass         [hard caps face≤25%, body≤35%]
        # Task #105 — ensure timed_lyrics always carries valid timestamps before
        # passing to the orchestrator.  Two fallback cases:
        # (a) timed_lyrics is empty/None (Whisper was skipped, or pre-#105 project):
        #     derive lyric lines from the stored transcript text and distribute
        #     them evenly over audio_data duration_seconds.
        # (b) timed_lyrics is non-empty but all have start==end==0 (Gemini-only
        #     transcription, Whisper segments failed): fill timestamps evenly.
        _audio_dur = float(audio_data_blob.get("duration_seconds") or 0)
        if _audio_dur > 0:
            if not timed_lyrics:
                _lines = [l.strip() for l in text.splitlines() if l.strip()]
                if _lines:
                    _n = len(_lines)
                    _step = _audio_dur / _n
                    timed_lyrics = [
                        {
                            "text": _lines[_i],
                            "start": round(_i * _step, 3),
                            "end": round((_i + 1) * _step, 3),
                        }
                        for _i in range(_n)
                    ]
                    logger.info(
                        "Stage 2: no lyrics_timed in DB; approximated timestamps "
                        "for %d transcript lines (%.1fs / %d = %.2fs each)",
                        _n, _audio_dur, _n, _step,
                    )
                    # Persist so the Context Engine review page can show them.
                    try:
                        with _db() as _conn, _conn.cursor() as _cur:
                            _cur.execute(
                                "ALTER TABLE projects ADD COLUMN IF NOT EXISTS "
                                "lyrics_timed JSONB;"
                            )
                            _cur.execute(
                                "UPDATE projects SET lyrics_timed=%s WHERE id=%s",
                                (Json(timed_lyrics), project_id),
                            )
                            _conn.commit()
                    except Exception as _e:
                        logger.warning(
                            "Stage 2: could not persist fallback lyrics_timed (%s)", _e
                        )
            else:
                def _safe_float(v: object) -> float:
                    try:
                        return float(v or 0)
                    except (TypeError, ValueError):
                        return 0.0
                has_real_ts = any(
                    _safe_float(t.get("end")) > _safe_float(t.get("start"))
                    for t in timed_lyrics
                )
                if not has_real_ts:
                    _n = len(timed_lyrics)
                    _step = _audio_dur / _n
                    for _i, _t in enumerate(timed_lyrics):
                        _t["start"] = round(_i * _step, 3)
                        _t["end"] = round((_i + 1) * _step, 3)
                    logger.info(
                        "Stage 2: timed_lyrics had no real timestamps; "
                        "distributed %d lines evenly (%.1fs / %d = %.2fs each)",
                        _n, _audio_dur, _n, _step,
                    )
                    try:
                        with _db() as _conn2, _conn2.cursor() as _cur2:
                            _cur2.execute(
                                "UPDATE projects SET lyrics_timed=%s WHERE id=%s",
                                (Json(timed_lyrics), project_id),
                            )
                            _conn2.commit()
                    except Exception as _e:
                        logger.warning(
                            "Stage 2: could not persist repaired lyrics_timed (%s)", _e
                        )

        # Task #69 — pass the user-locked Creative Brief so the orchestrator
        # splices it into its freshly-regenerated context_packet (otherwise
        # director_note/central_metaphor would be lost when the storyboard
        # rebuilds context from raw text).
        pre_result = _run_async(orchestrator.run_to_timeline(
            text=text, genre=genre,
            audio_analytics=audio_data_blob,
            style_profile=style_profile,
            # Brief runs AFTER storyboard now (master spec) — never pass it.
            creative_brief=None,
            timed_lyrics=timed_lyrics or None,
        ))
        raw_storyboard = pre_result.get("storyboard") or []
        raw_timeline = pre_result.get("timeline") or []

        # MM3.1 beat-engine verification — log how many shots received beat enrichment.
        # Shots without cinematic_beat fall back to legacy emotion-only prompting.
        _n_beats = sum(1 for s in raw_timeline if s.get("cinematic_beat"))
        _n_variety = sum(1 for s in raw_timeline if s.get("shot_type"))
        if _n_beats or _n_variety:
            logger.info(
                "[MM3.1] project=%s storyboard=%d shots | beats=%d variety=%d",
                project_id, len(raw_timeline), _n_beats, _n_variety,
            )
        else:
            logger.warning(
                "[MM3.1] project=%s: no cinematic beats or shot_types found — "
                "running in legacy mode (cinematic_beat_engine modules may be missing)",
                project_id,
            )

        _set_status(project_id, "running",
                    {"stage": "storyboard", "label": "Applying style grading…"}, stage="running_2")

        from style_grading_engine import StyleGradingEngine
        style_engine = StyleGradingEngine()
        styled_timeline = style_engine.apply_style(
            timeline=raw_timeline,
            style_profile={"preset": style_preset},
        )

        # The StyleGradingEngine rebuilds each shot from a strict whitelist
        # and silently drops the rhythmic-sync fields the assembly engine
        # produced. Merge them back so beat/bar/audio_intensity survive into
        # the final styled_timeline that downstream stages + the UI consume.
        _RHYTHM_FIELDS = (
            "start_beat", "bar_index", "audio_intensity", "raw_shot_intensity",
        )
        raw_by_idx = {r.get("shot_index"): r for r in raw_timeline}
        for styled in styled_timeline:
            src = raw_by_idx.get(styled.get("shot_index"))
            if not src:
                continue
            for k in _RHYTHM_FIELDS:
                if k in src and styled.get(k) is None:
                    styled[k] = src[k]

        # Entity materialization (characters / locations / motifs) has moved to
        # the start of _stage_refs_job — it now runs AFTER the Creative Brief
        # is locked so it can read brain.creative_briefs for richer identity
        # anchors. Shot rows are seeded here so the UI can show progress, but
        # entity linking (_link_shots_to_entities) also happens in refs once
        # the entity rows exist.
        shot_indices = [s.get("shot_index") or s.get("timeline_index") for s in styled_timeline]
        _seed_shot_rows(project_id, shot_indices)

        # Export JSON
        from asset_export_module import AssetExportModule
        exporter = AssetExportModule(name or "Qaivid_Project")
        export_json = exporter.export_to_json(styled_timeline)
        r2_key = f"projects/{project_id}/exports/{project_id}.json"
        try:
            export_url = r2_storage.upload_bytes(
                export_json.encode(), r2_key, content_type="application/json"
            )
        except Exception:
            export_url = r2_key

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE projects
                   SET context_packet=%s, summary=%s, styled_timeline=%s, export_path=%s, updated_at=NOW()
                 WHERE id=%s
                """,
                (
                    Json(context_packet),
                    Json({"storyboard_shot_count": len(raw_storyboard),
                          "styled_timeline_shot_count": len(styled_timeline),
                          "total_duration": round(sum(x.get("duration", 0) for x in raw_timeline), 2),
                          "style_preset": style_preset}),
                    Json(styled_timeline),
                    export_url,
                    project_id,
                ),
            )
            conn.commit()

        # ── Join v2 thread (bounded) before writing brain ──────────────────
        # Bounded join so a hung LLM call cannot stall the stage. If v2 is
        # still running after the timeout we proceed with empty intent —
        # downstream is unaffected because it consumes the legacy timeline.
        _v2_thread.join(timeout=120)
        if _v2_thread.is_alive():
            logger.warning(
                "[StoryboardV2] join timed out for project=%s — proceeding "
                "without intent layer; thread will finish in background.",
                project_id,
            )
        v2_scenes        = list(_v2_holder.get("scenes") or [])
        v2_used_fallback = bool(_v2_holder.get("used_fallback"))
        if _v2_holder.get("ok"):
            logger.info(
                "[StoryboardV2] project=%s scenes=%d fallback=%s",
                project_id, len(v2_scenes), v2_used_fallback,
            )

        # ── Write storyboard_packet to brain (Stage 6 namespace) ───────────
        # schema_version 2 layout:
        #   scenes              — v2 intent layer (purpose, valid_realizations…)
        #   used_fallback_v2    — true if v2 LLM call fell back to defaults
        # Legacy-bridge keys (kept until downstream stages migrate to scenes):
        #   raw_storyboard, styled_timeline, shot_count, total_duration,
        #   style_preset
        # Per-shot inline edits during storyboard_review stay in the legacy
        # styled_timeline column; the final approved snapshot is mirrored back
        # at advance_stage_3.
        try:
            with _db() as conn:
                brain = ProjectBrain.load(project_id, conn)
                brain.write("storyboard_packet", {
                    "schema_version":   2,
                    "scenes":           v2_scenes,
                    "used_fallback_v2": v2_used_fallback,
                    "raw_storyboard":   raw_storyboard,
                    "styled_timeline":  styled_timeline,
                    "shot_count":       len(styled_timeline),
                    "total_duration":   round(sum(x.get("duration", 0) for x in raw_timeline), 2),
                    "style_preset":     style_preset,
                })
                brain.save(conn)
                conn.commit()
            logger.info(
                "ProjectBrain: wrote storyboard_packet v2 for project=%s "
                "(intent: %d scenes, legacy: %d shots / %d raw, preset=%s)",
                project_id, len(v2_scenes), len(styled_timeline),
                len(raw_storyboard), style_preset,
            )
        except Exception:
            logger.exception(
                "Stage storyboard: failed to write storyboard_packet to brain for project=%s",
                project_id,
            )

        _set_status(project_id, "awaiting_review",
                    {"stage": "storyboard", "label": f"Storyboard ready — {len(styled_timeline)} shots. Review and continue."},
                    stage="storyboard_review")

    except Exception as exc:
        logger.exception("Stage 2 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Stage 2 failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage_refs_job(project_id: str,
                    uploaded_character_ref_url: Optional[str] = None,
                    uploaded_env_ref_url: Optional[str] = None) -> None:
    """Reference Engine — generate ONE identity plate per character + ONE
    environment plate per location. Parks at `references_review` so the user
    can fix any plate (regenerate / upload) before stills are rendered.

    Legacy projects with no character/location rows skip straight to stills_review.

    Stage 7 — Materializer v2 runs first (reads creative_briefs, context,
    narrative, style, settings from brain → produces materializer_packet →
    populates DB chars/locs/motifs for the UI and the entity-linking step).
    """
    try:
        # ── Stage 7: Materializer v2 ─────────────────────────────────────
        # Must run before querying chars/locs because it creates those rows.
        # Non-fatal: if it fails, refs stage continues with whatever legacy
        # rows already exist (or empty, which triggers the skip-to-stills path).
        try:
            with _db() as conn:
                _mat_brain = ProjectBrain.load(project_id, conn)
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT audio_data, styled_timeline FROM projects WHERE id=%s",
                    (project_id,),
                )
                _mat_prj = cur.fetchone() or {}
            _mat_audio = dict(_mat_prj.get("audio_data") or {})
            _vg_final  = str(_mat_audio.get("vocal_gender_final") or "").strip().lower()
            _vg_for_mat = (
                _vg_final if _vg_final in ("male", "female", "mixed", "instrumental")
                else (str(_mat_audio.get("vocal_gender") or "").strip().lower() or None)
            )
            if _vg_for_mat not in ("male", "female", "mixed", "instrumental"):
                _vg_for_mat = None
            mat_packet = run_materializer_sync(project_id, _mat_brain, vocal_gender=_vg_for_mat)
            # Save brain FIRST — brain is authoritative before DB population.
            with _db() as conn:
                _mat_brain.save(conn)
                conn.commit()
            # Now run legacy materializers to populate DB tables (chars/locs/motifs)
            # in the exact ordered sequence specified by the stage contract.
            _mat_ctx = dict(_mat_brain.read("context_packet") or {})
            try:
                materialize_characters(project_id, _mat_ctx, vocal_gender=_vg_for_mat)
                materialize_locations(project_id, _mat_ctx)
                materialize_motifs(project_id, _mat_ctx)
            except Exception:
                logger.exception(
                    "_stage_refs_job: legacy materializers failed (non-fatal) for project=%s",
                    project_id,
                )

            # ── Enrich brain with DB integer IDs ─────────────────────────
            # After DB rows are created, read their integer PKs and store them
            # directly in the materializer_packet entries (brain.character_bible
            # characters[i].db_id / location_bible.locations[i].db_id).
            # This gives _link_shots_to_entities exact IDs without any guessing.
            try:
                with _db() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, entity_type FROM characters WHERE project_id=%s ORDER BY id",
                        (project_id,),
                    )
                    _enriched_chars = cur.fetchall() or []
                    cur.execute(
                        "SELECT id, entity_type FROM locations WHERE project_id=%s ORDER BY id",
                        (project_id,),
                    )
                    _enriched_locs = cur.fetchall() or []

                _ep = dict(_mat_brain.read("materializer_packet") or mat_packet)
                _ep_cb = (_ep.get("character_bible") or {}).get("characters") or []
                _ep_lb = (_ep.get("location_bible")  or {}).get("locations")  or []
                _db_spk = [c for c in _enriched_chars if c["entity_type"] == "speaker"]
                _db_wdn = [l for l in _enriched_locs  if l["entity_type"] == "world_dna"]

                for _i, _bch in enumerate(_ep_cb):
                    if isinstance(_bch, dict) and _i < len(_db_spk):
                        _bch["db_id"] = _db_spk[_i]["id"]
                for _i, _bloc in enumerate(_ep_lb):
                    if isinstance(_bloc, dict) and _i < len(_db_wdn):
                        _bloc["db_id"] = _db_wdn[_i]["id"]

                _mat_brain.write("materializer_packet", _ep)
                _mat_brain.write("character_bible", _ep.get("character_bible") or {})
                _mat_brain.write("location_bible",  _ep.get("location_bible")  or {})
                with _db() as conn:
                    _mat_brain.save(conn)
                    conn.commit()
                logger.info(
                    "_stage_refs_job: enriched materializer_packet with DB IDs for project=%s",
                    project_id,
                )
            except Exception:
                logger.exception(
                    "_stage_refs_job: failed to enrich materializer_packet with DB IDs (non-fatal) for project=%s",
                    project_id,
                )

            # Link shots to the freshly created entity rows.
            _raw_timeline = list(_mat_prj.get("styled_timeline") or [])
            if _raw_timeline:
                _link_shots_to_entities(project_id, _raw_timeline)
            logger.info(
                "Materializer v2 completed for project=%s: %d chars, %d locs",
                project_id,
                len((mat_packet.get("character_bible") or {}).get("characters") or []),
                len((mat_packet.get("location_bible") or {}).get("locations") or []),
            )
        except Exception:
            logger.exception(
                "_stage_refs_job: Materializer v2 failed (non-fatal) for project=%s — "
                "refs will use any legacy entity rows that exist",
                project_id,
            )

        # ── Read chars/locs (now populated by the Materializer above) ────
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT context_packet, style_profile FROM projects WHERE id=%s",
                (project_id,),
            )
            prj = cur.fetchone()
            cur.execute(
                "SELECT id, name, entity_type, ref_status, ref_image_url "
                "  FROM characters WHERE project_id=%s ORDER BY entity_type, id",
                (project_id,),
            )
            chars = cur.fetchall() or []
            cur.execute(
                "SELECT id, name, entity_type, ref_status, ref_image_url "
                "  FROM locations WHERE project_id=%s ORDER BY entity_type, id",
                (project_id,),
            )
            locs = cur.fetchall() or []
        if not prj:
            raise RuntimeError("Project not found.")
        context_packet = dict(prj.get("context_packet") or {})
        location_dna = context_packet.get("location_dna") or "Universal"
        _raw_sp = dict(prj.get("style_profile") or {})
        _style_profile = _raw_sp if _raw_sp else StyleProfileRegistry.default_style_profile()
        _cin = _style_profile.get("cinematic") or {}
        style_image_suffix = (_cin.get("image_generation_suffix") or "").strip() or None

        # Apply user-uploaded fallbacks from the storyboard upload form to the
        # primary speaker / world_dna entity rows (back-compat with old form).
        if uploaded_character_ref_url and chars:
            primary = next((c for c in chars if c["entity_type"] == "speaker"), chars[0])
            _set_entity_ref(project_id, "character", primary["id"],
                            status="ready", source="uploaded",
                            file_path=uploaded_character_ref_url, error=None)
            primary["ref_status"] = "ready"
            primary["ref_image_url"] = uploaded_character_ref_url
        if uploaded_env_ref_url and locs:
            primary = next((l for l in locs if l["entity_type"] == "world_dna"), locs[0])
            _set_entity_ref(project_id, "location", primary["id"],
                            status="ready", source="uploaded",
                            file_path=uploaded_env_ref_url, error=None)
            primary["ref_status"] = "ready"
            primary["ref_image_url"] = uploaded_env_ref_url

        # Legacy back-compat: also persist into the project-wide refs table so
        # the older stills viewer keeps working.
        if uploaded_character_ref_url:
            _upsert_ref(project_id, "character", "uploaded", "ready",
                        file_path=uploaded_character_ref_url)
        if uploaded_env_ref_url:
            _upsert_ref(project_id, "environment", "uploaded", "ready",
                        file_path=uploaded_env_ref_url)

        # If the storyboard never produced any cast/location rows (very old
        # projects), there's nothing to gate on — fall back to the legacy
        # global refs path and skip straight to stills.
        if not chars and not locs:
            logger.info("Project %s has no characters/locations rows — skipping refs gate", project_id)
            _ensure_character_ref(project_id, context_packet)
            _ensure_env_ref(project_id, context_packet)
            _set_status(project_id, "queued",
                        {"stage": "stills", "label": "No cast/locations to gate — going straight to stills."},
                        stage="queued")
            kick_stage_3(project_id)
            return

        total = len(chars) + len(locs)
        _set_status(project_id, "running",
                    {"stage": "refs",
                     "label": f"Building prompts for {len(chars)} character(s) + {len(locs)} location(s)…",
                     "total": total},
                    stage="running_refs")

        # ── AI-written prompts ────────────────────────────────────────────────
        # Fetch full entity rows so the LLM has all fields available, then ask
        # GPT to write purpose-built cinematic prompts for every character and
        # location in one call.  Falls back silently to template prompts if the
        # call fails, so the pipeline is never blocked by this step.
        try:
            from image_generator import ai_build_ref_prompts
            with _db() as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM characters WHERE project_id=%s ORDER BY id",
                            (project_id,))
                chars_full = cur.fetchall() or []
                cur.execute("SELECT * FROM locations WHERE project_id=%s ORDER BY id",
                            (project_id,))
                locs_full = cur.fetchall() or []
            # Read materializer_packet and creative_briefs from the brain so
            # prompts include the full identity rules (identity_seed, archetype,
            # emotional_baseline, continuity_rules, motif_anchors, world DNA)
            # and creative direction (lighting, emotional state, key elements).
            _ref_mat_packet = dict(_mat_brain.read("materializer_packet") or {})
            _ref_briefs     = dict(_mat_brain.read("creative_briefs")     or {})
            ai_prompts = ai_build_ref_prompts(
                chars_full, locs_full, context_packet, _style_profile,
                materializer_packet=_ref_mat_packet,
                creative_briefs=_ref_briefs,
            )
        except Exception:
            logger.exception("AI ref-prompt generation failed — using templates")
            ai_prompts = {}

        char_prompts = ai_prompts.get("characters", {})
        loc_prompts  = ai_prompts.get("locations",  {})

        # Save AI-generated prompts to DB so they are visible and editable on
        # the references_review page. Plates are NOT auto-generated here — the
        # user reviews the prompts, edits them if needed, then generates or
        # uploads each plate manually on the review page.
        for c in chars:
            if c.get("ref_status") == "ready":
                continue  # already uploaded/ready — preserve as-is
            ai_prompt = char_prompts.get(str(c["id"]))
            if ai_prompt:
                _set_entity_ref(project_id, "character", c["id"],
                                status="pending", prompt=ai_prompt)
        for l in locs:
            if l.get("ref_status") == "ready":
                continue
            ai_prompt = loc_prompts.get(str(l["id"]))
            if ai_prompt:
                _set_entity_ref(project_id, "location", l["id"],
                                status="pending", prompt=ai_prompt)

        # ── Wardrobe diversification + styled look plates ─────────────────
        # Run the wardrobe engine here (before references_review) so the
        # director can see all looks — base plates AND per-scene styled
        # look clusters to shots so the stills stage knows which wardrobe look
        # to reference per shot. Look plate images are NOT generated here —
        # they will be rendered on-demand when the user generates/uploads plates.
        try:
            from wardrobe_engine import diversify_wardrobe
            n_wardrobe = diversify_wardrobe(project_id)
            logger.info("Refs stage: wardrobe engine updated %d shots for project=%s",
                        n_wardrobe, project_id)
        except Exception:
            logger.exception("Refs stage: wardrobe diversification failed "
                             "(non-fatal) for project=%s", project_id)

        # ── Write reference_assets_packet to brain (Stage 8) ────────────────
        # Index final plate URLs by character_id and location_id so the stills
        # stage can look them up semantically rather than by positional DB join.
        try:
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, entity_type, ref_image_url, ref_status "
                    "  FROM characters WHERE project_id=%s AND ref_status='ready'",
                    (project_id,),
                )
                _ra_chars = cur.fetchall() or []
                cur.execute(
                    "SELECT id, name, entity_type, ref_image_url, ref_status "
                    "  FROM locations WHERE project_id=%s AND ref_status='ready'",
                    (project_id,),
                )
                _ra_locs = cur.fetchall() or []

            _ra_mat = dict(_mat_brain.read("materializer_packet") or {})
            _ra_sp  = dict(_mat_brain.read("style_packet") or {})

            _ref_assets_packet = {
                "character_references": [
                    {
                        "character_id": c["id"],
                        "name": c.get("name") or "",
                        "entity_type": c.get("entity_type") or "",
                        "ref_image_url": c.get("ref_image_url") or "",
                    }
                    for c in _ra_chars if c.get("ref_image_url")
                ],
                "environment_plates": [
                    {
                        "location_id": l["id"],
                        "name": l.get("name") or "",
                        "entity_type": l.get("entity_type") or "",
                        "ref_image_url": l.get("ref_image_url") or "",
                    }
                    for l in _ra_locs if l.get("ref_image_url")
                ],
                "style_frames": {
                    "cinematic_style":  (_ra_sp.get("cinematic") or {}).get("look") or "",
                    "color_psychology": _ra_sp.get("color_psychology") or "",
                    "texture_profile":  _ra_sp.get("texture_profile") or "",
                    "lighting_logic":   _ra_sp.get("lighting_logic") or "",
                },
                "motif_references": list(
                    (_ra_mat.get("motif_anchors") or [])
                ),
            }
            _mat_brain.write("reference_assets", _ref_assets_packet)
            with _db() as conn:
                _mat_brain.save(conn)
                conn.commit()
            logger.info(
                "_stage_refs_job: wrote reference_assets_packet to brain "
                "(%d char refs, %d env plates) for project=%s",
                len(_ref_assets_packet["character_references"]),
                len(_ref_assets_packet["environment_plates"]),
                project_id,
            )
        except Exception:
            logger.exception(
                "_stage_refs_job: failed to write reference_assets_packet to brain "
                "(non-fatal) for project=%s",
                project_id,
            )

        _set_status(project_id, "awaiting_review",
                    {"stage": "references",
                     "label": "Prompts ready. Review and edit, then generate or upload each reference image before continuing."},
                    stage="references_review")

    except Exception as exc:
        logger.exception("References stage failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "References stage failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _stage3_job(project_id: str,
                uploaded_character_ref_url: Optional[str] = None,
                uploaded_env_ref_url: Optional[str] = None) -> None:
    """Stage 3 — Stills only. Per-shot refs are resolved inside `_render_shot`
    from the locked character/location plates produced by the references stage.
    """
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT styled_timeline FROM projects WHERE id=%s", (project_id,))
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        styled_timeline = list(row.get("styled_timeline") or [])

        # Back-compat: if a caller still passes uploads (old direct kick path),
        # honor them by overriding the project-wide refs table.
        if uploaded_character_ref_url:
            _upsert_ref(project_id, "character", "uploaded", "ready",
                        file_path=uploaded_character_ref_url)
        if uploaded_env_ref_url:
            _upsert_ref(project_id, "environment", "uploaded", "ready",
                        file_path=uploaded_env_ref_url)

        _set_status(project_id, "running",
                    {"stage": "stills",
                     "label": "Assigning scene-appropriate wardrobe to shots…"},
                    stage="running_3")

        # Wardrobe engine already ran during the reference engine stage so each
        # shot already has wardrobe_context + look_cluster_id written.  The
        # _resolve_shot_refs_full helper picks the scene-specific look plate.
        _set_status(project_id, "running",
                    {"stage": "stills",
                     "label": f"Rendering {len(styled_timeline)} stills…",
                     "total": len(styled_timeline)},
                    stage="running_3")

        futures = [
            _SHOT_EXECUTOR.submit(_render_shot, project_id, shot, None, None)
            for shot in styled_timeline
        ]
        for f in futures:
            try:
                f.result()
            except Exception:
                logger.exception("Shot render raised")

        _set_status(project_id, "awaiting_review",
                    {"stage": "stills", "label": "Stills complete. Review and continue."},
                    stage="stills_review")

    except Exception as exc:
        logger.exception("Stage 3 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Stage 3 failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


def _camera_to_transition(camera_movement: str) -> str:
    """Map a shot camera movement string to a clip transition type.

    Transitions are directional cues — pans suggest wipes, zooms suggest
    dissolves, and static shots fall back to a clean cut.
    """
    mv = (camera_movement or "").lower()
    if "zoom" in mv:
        return "cross_dissolve"
    if "pan" in mv or "tilt" in mv or "drift" in mv:
        return "wipe"
    return "cut"


def _build_video_sequence_packet(
    styled_timeline: list,
    materializer_packet: dict,
    narrative_packet: dict,
    shot_entity_map: Optional[dict] = None,
) -> dict:
    """Build the video_sequence_packet brain entry from the shot timeline.

    Per shot: shot_id, duration, motion_applied (camera_plan movement → spec mode),
    transition_in/out (derived from camera movement), character_id, location_id,
    link_status.

    Args:
        shot_entity_map: optional {shot_index: (char_db_id|None, loc_db_id|None)}
            pre-resolved from shot_assets; falls back to primary bible entries.
    """
    from motion_render_prompt_builder import _intensity_float, _motion_mode_for_intensity

    motion_philosophy = str(narrative_packet.get("motion_philosophy") or "mixed").strip()
    _cb = (materializer_packet.get("character_bible") or {}).get("characters") or []
    _lb = (materializer_packet.get("location_bible")  or {}).get("locations")  or []

    shots_plan = []
    for shot in styled_timeline:
        idx      = shot.get("shot_index") or shot.get("timeline_index")
        duration = shot.get("duration")

        # Resolve character/location IDs from shot_entity_map only — no primary fallback.
        # IDs stay null if shot_assets had no FK for this shot.
        char_db_id: Optional[int] = None
        loc_db_id:  Optional[int] = None
        if shot_entity_map and idx in shot_entity_map:
            char_db_id, loc_db_id = shot_entity_map[idx]

        # link_status
        if char_db_id is not None and loc_db_id is not None:
            link_status = "resolved"
        elif char_db_id is not None or loc_db_id is not None:
            link_status = "semantic_only"
        else:
            link_status = "unresolved"

        # Derive motion from camera_plan.movement first, fall back to intensity scaling
        camera = shot.get("camera_plan") or {}
        cam_movement = (camera.get("movement") or "").strip()
        raw_int = shot.get("emotional_intensity") or "medium"
        if cam_movement:
            # Use camera_plan movement as primary source; intensity modifies quality
            from motion_render_prompt_builder import _MOTION_MODES
            mapped = _MOTION_MODES.get(cam_movement)
            if mapped:
                motion_applied = mapped
            else:
                # Camera movement string not a spec mode — use it directly
                motion_applied = cam_movement
        else:
            motion_applied = _motion_mode_for_intensity(
                _intensity_float(raw_int), motion_philosophy
            )

        # Transitions derived from camera movement
        transition = _camera_to_transition(cam_movement)

        shots_plan.append({
            "shot_id":        idx,
            "duration":       duration,
            "motion_applied": motion_applied,
            "transition_in":  transition,
            "transition_out": transition,
            "character_id":   char_db_id,
            "location_id":    loc_db_id,
            "link_status":    link_status,
        })

    return {
        "motion_philosophy": motion_philosophy,
        "total_shots":       len(shots_plan),
        "shots":             shots_plan,
    }


def _stage4_job(project_id: str) -> None:
    """Stage 4 — Video clips: animate each still into a short clip.

    Brain writes:
        video_sequence_packet — per-shot render plan (before rendering)
        video_render_packet   — per-shot clip manifest (after rendering)
    """
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT styled_timeline FROM projects WHERE id=%s", (project_id,))
            row = cur.fetchone()
        if not row or not row.get("styled_timeline"):
            raise RuntimeError("No storyboard found for project.")

        styled_timeline = list(row["styled_timeline"])
        shot_indices = [s.get("shot_index") or s.get("timeline_index") for s in styled_timeline]
        _seed_video_rows(project_id, shot_indices)

        # Initialise so the render_packet builder can reference it even if
        # the sequence_packet write fails (non-fatal path).
        _vsp: dict = {}

        # ── Write video_sequence_packet to brain (before rendering) ──────────
        try:
            with _db() as conn:
                _s4brain = ProjectBrain.load(project_id, conn)
            _s4_mat = dict(_s4brain.read("materializer_packet") or {})
            _s4_nar = dict(_s4brain.read("narrative_packet")     or {})

            # Pre-load per-shot entity FKs from shot_assets so _build_video_sequence_packet
            # can resolve character_id / location_id per shot (not just primary global).
            _s4_shot_entity_map: dict = {}
            try:
                with _db() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT shot_index, character_id, location_id "
                        "  FROM shot_assets WHERE project_id=%s",
                        (project_id,),
                    )
                    for _sa_row in (cur.fetchall() or []):
                        _s4_shot_entity_map[_sa_row.get("shot_index")] = (
                            _sa_row.get("character_id"),
                            _sa_row.get("location_id"),
                        )
            except Exception:
                logger.debug(
                    "_stage4_job: could not pre-load shot entity map (non-fatal) for project=%s",
                    project_id,
                )

            _vsp = _build_video_sequence_packet(
                styled_timeline, _s4_mat, _s4_nar,
                shot_entity_map=_s4_shot_entity_map or None,
            )
            _s4brain.write("video_sequence_packet", _vsp)
            with _db() as conn:
                _s4brain.save(conn)
                conn.commit()
            logger.info(
                "_stage4_job: wrote video_sequence_packet (%d shots, philosophy=%s) for project=%s",
                _vsp.get("total_shots", 0), _vsp.get("motion_philosophy"), project_id,
            )
        except Exception:
            logger.exception(
                "_stage4_job: failed to write video_sequence_packet (non-fatal) for project=%s",
                project_id,
            )

        _set_status(project_id, "running",
                    {"stage": "videos",
                     "label": f"Animating {len(styled_timeline)} clips…",
                     "total": len(styled_timeline)},
                    stage="running_4")

        video_futures = [
            _VIDEO_EXECUTOR.submit(_render_video, project_id, shot)
            for shot in styled_timeline
        ]
        for vf in video_futures:
            try:
                vf.result()
            except Exception:
                logger.exception("Video render raised")

        # ── Write video_render_packet to brain (after all clips complete) ────
        try:
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT shot_index, file_path, status, error "
                    "  FROM video_assets WHERE project_id=%s ORDER BY shot_index",
                    (project_id,),
                )
                _va_rows = cur.fetchall() or []

                # Also grab still image URLs (input_image per clip)
                cur.execute(
                    "SELECT shot_index, file_path AS still_url "
                    "  FROM shot_assets WHERE project_id=%s AND status='ready'",
                    (project_id,),
                )
                _still_by_idx: dict = {
                    r.get("shot_index"): r.get("still_url") or ""
                    for r in (cur.fetchall() or [])
                }

            _clips = []
            _render_warnings = []
            for _va in _va_rows:
                _si    = _va.get("shot_index")
                _st    = _va.get("status") or "unknown"
                _fpath = _va.get("file_path") or ""
                _err   = _va.get("error") or ""
                if _st == "failed" and _err:
                    _render_warnings.append(f"shot_{_si}: {_err}")
                # Corresponding sequence plan entry
                _seq_shot = next(
                    (sp for sp in (_vsp.get("shots") or []) if sp.get("shot_id") == _si),
                    {},
                )
                _clips.append({
                    "clip_id":        f"clip_{_si}",
                    "shot_id":        _si,
                    "mode":           "image_to_video",
                    "input_image":    _still_by_idx.get(_si) or "",
                    "duration":       _seq_shot.get("duration"),
                    "motion_applied": _seq_shot.get("motion_applied"),
                    "transition_in":  _seq_shot.get("transition_in", "cut"),
                    "transition_out": _seq_shot.get("transition_out", "cut"),
                    "character_id":   _seq_shot.get("character_id"),
                    "location_id":    _seq_shot.get("location_id"),
                    "link_status":    _seq_shot.get("link_status", "unresolved"),
                    "video_url":      _fpath,
                    "render_status":  _st,
                })

            _vrp = {
                "total_clips":      len(_clips),
                "clips":            _clips,
                "render_warnings":  _render_warnings,
            }

            with _db() as conn:
                _vrp_brain = ProjectBrain.load(project_id, conn)
            _vrp_brain.write("video_render_packet", _vrp)
            with _db() as conn:
                _vrp_brain.save(conn)
                conn.commit()
            logger.info(
                "_stage4_job: wrote video_render_packet (%d clips, %d warnings) for project=%s",
                len(_clips), len(_render_warnings), project_id,
            )
        except Exception:
            logger.exception(
                "_stage4_job: failed to write video_render_packet (non-fatal) for project=%s",
                project_id,
            )

        _set_status(project_id, "awaiting_review",
                    {"stage": "videos", "label": "Video clips complete. Ready for final assembly."},
                    stage="videos_review")

    except Exception as exc:
        logger.exception("Stage 4 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Stage 4 failed."},
                    stage="failed", error=f"{exc}\n{traceback.format_exc(limit=4)}")


# ── Public kick functions ─────────────────────────────────────────────────────

def kick_stage_0(project_id: str, audio_path: Optional[Path], text: str, genre: str) -> None:
    _set_status(project_id, "queued", {"stage": "queued", "label": "Queued…"}, stage="new")
    _EXECUTOR.submit(_stage0_job, project_id, audio_path, text, genre)


def kick_stage_style(project_id: str) -> None:
    _EXECUTOR.submit(_stage_style_job, project_id)


def kick_stage_1(project_id: str) -> None:
    _EXECUTOR.submit(_stage1_job, project_id)


def kick_stage_narrative(project_id: str) -> None:
    """Stage 3 — Narrative Intelligence. Reads context_packet from brain."""
    _EXECUTOR.submit(_stage_narrative_job, project_id)


def kick_stage_2(project_id: str, name: str, overrides: dict) -> None:
    _EXECUTOR.submit(_stage2_job, project_id, name, overrides)


def kick_stage_brief(project_id: str, overrides: dict) -> None:
    """Task #69 — Creative Brief variant generation (parks at creative_brief_review)."""
    _EXECUTOR.submit(_stage_brief_job, project_id, overrides)


def kick_stage_refs(project_id: str,
                    uploaded_character_ref_url: Optional[str] = None,
                    uploaded_env_ref_url: Optional[str] = None) -> None:
    _EXECUTOR.submit(_stage_refs_job, project_id,
                     uploaded_character_ref_url, uploaded_env_ref_url)


def kick_stage_3(project_id: str,
                 uploaded_character_ref_url: Optional[str] = None,
                 uploaded_env_ref_url: Optional[str] = None) -> None:
    _EXECUTOR.submit(_stage3_job, project_id, uploaded_character_ref_url, uploaded_env_ref_url)


def regenerate_entity_plate(project_id: str, kind: str, entity_id: int,
                              prompt_override: Optional[str] = None,
                              location_dna: str = "Universal") -> None:
    """Public helper used by the references_review page to retry one plate.

    Synchronously flips ref_status to 'rendering' BEFORE the worker is queued
    so that a concurrent /references/approve cannot observe the row as 'ready'
    while a regeneration is in flight (the atomic NOT EXISTS gate in the
    approve route relies on this)."""
    if kind not in ("character", "location"):
        raise ValueError(f"Unknown plate kind: {kind!r}")
    _set_entity_ref(project_id, kind, entity_id,
                    status="rendering", source="ai", error=None)
    if kind == "character":
        _SHOT_EXECUTOR.submit(_render_character_plate, project_id, entity_id,
                              location_dna, prompt_override)
    else:
        _SHOT_EXECUTOR.submit(_render_location_plate, project_id, entity_id,
                              prompt_override)


def set_entity_uploaded_plate(project_id: str, kind: str, entity_id: int,
                                file_url: str) -> None:
    """Public helper used by the references_review page when the user uploads
    a custom plate. Marks the row as user-supplied so the next stage uses it."""
    _set_entity_ref(project_id, kind, entity_id,
                    status="ready", source="uploaded",
                    file_path=file_url, error=None)


# ── Stills Control helpers (Task: per-shot generation gate) ──────────────────

def _project_cine_context(project_id: str, upto_idx: Optional[int] = None
                          ) -> tuple[dict, dict, list]:
    """Load (context_packet, style_profile, prior_cine_blocks) for the
    cinematography engine. ``prior_cine_blocks`` is the list of blocks
    that would have been derived for shots before ``upto_idx`` so the
    anti-repeat penalty stays consistent across the timeline preview.
    Falls back gracefully when fields are missing."""
    from psycopg.rows import dict_row
    with _db() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT styled_timeline, context_packet, style_profile "
            "FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone() or {}
    tl = row.get("styled_timeline") or []
    ctx = row.get("context_packet") or {}
    sp = row.get("style_profile") or {}
    if not tl or upto_idx is None:
        return ctx, sp, []
    try:
        from cinematography_engine import derive as _cine_derive
    except Exception:
        return ctx, sp, []
    history: list = []
    for s in tl:
        sidx = s.get("shot_index") or s.get("timeline_index")
        if sidx is None or sidx >= upto_idx:
            break
        block = _cine_derive(
            s, ctx, sp,
            prev_block=(history[-1] if history else None),
            recent_blocks=history[-4:],
        )
        if block:
            history.append(block)
    return ctx, sp, history


def composed_prompts_for_project(project_id: str) -> dict:
    """Precompute composed prompts for every shot in one pass.

    Avoids the O(n²) re-derive that happens when ``composed_prompt_preview``
    is called in a loop (each call would otherwise re-derive every
    prior shot's cinematography to maintain anti-repeat consistency).

    Returns ``{shot_index: composed_prompt}``."""
    from psycopg.rows import dict_row
    from shot_prompt_composer import compose_image_prompt
    try:
        from cinematography_engine import derive as _cine_derive, lens_clause as _lens_clause
    except Exception:
        _cine_derive = None
        _lens_clause = None

    with _db() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT styled_timeline, context_packet, style_profile "
            "FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone() or {}
    tl = row.get("styled_timeline") or []
    ctx = row.get("context_packet") or {}
    sp = row.get("style_profile") or {}

    try:
        from style_grading_engine import (
            pick_lighting_variant as _pick_lighting,
            pick_palette_variant as _pick_palette,
        )
    except Exception:
        _pick_lighting = None
        _pick_palette = None

    out: dict = {}
    history: list = []
    for s in tl:
        idx = s.get("shot_index") or s.get("timeline_index")
        if idx is None:
            continue
        char_url, env_url, character, location = _resolve_shot_refs_full(project_id, idx)
        expression_mode = (s.get("expression_mode") or "environment").lower()
        has_human_focus = expression_mode in {"face", "body"}

        cine_block = None
        if _cine_derive:
            try:
                cine_block = _cine_derive(
                    s, ctx, sp,
                    prev_block=(history[-1] if history else None),
                    recent_blocks=history[-4:],
                )
            except Exception:
                cine_block = None
        if not cine_block and isinstance(s.get("cinematography"), dict):
            cine_block = s.get("cinematography")
        if cine_block:
            history.append(cine_block)

        cine_prefix = ""
        if cine_block and cine_block.get("rig") and _lens_clause:
            try:
                cine_prefix = (_lens_clause(cine_block) or "").strip()
            except Exception:
                cine_prefix = ""

        # Re-derive lighting/palette on the fly so existing projects get
        # the variant cycling immediately — without re-running the
        # style_grading_engine across the whole timeline. Mutate a copy
        # of the shot dict so we don't touch the persisted styled_timeline.
        # Only override when the stored value matches one of the known
        # monoculture strings the old engine produced, so any hand-tuned
        # styled_timeline lighting/palette is preserved verbatim.
        shot_for_compose = s
        if _pick_lighting and _pick_palette:
            try:
                stored_light = (s.get("lighting_style") or "").strip().lower()
                stored_palette = (s.get("color_palette") or "").strip().lower()
                light_is_stale = (
                    not stored_light
                    or stored_light.startswith("soft controlled facial lighting")
                    or stored_light.startswith("expressive atmospheric lighting")
                    or stored_light.startswith("precise premium lighting")
                    or stored_light.startswith("soft cinematic natural lighting")
                    or stored_light.startswith("naturalistic observational lighting")
                    or stored_light.startswith("clean premium controlled lighting")
                    or stored_light.startswith("dramatic cinematic lighting")
                    or stored_light.startswith("directional moody lighting")
                )
                palette_is_stale = (
                    not stored_palette
                    or stored_palette.startswith("poetic tonal palette")
                    or stored_palette.startswith("natural skin-faithful palette")
                    or stored_palette.startswith("clean premium palette")
                    or stored_palette.startswith("balanced cinematic natural palette")
                    or stored_palette.startswith("rich cinematic palette")
                    or stored_palette.startswith("cinematic lyrical palette")
                    or stored_palette.startswith("clean premium commercial palette")
                    or stored_palette.startswith("natural restrained documentary palette")
                    or stored_palette.startswith("refined monochrome tonal palette")
                )
                if light_is_stale or palette_is_stale:
                    fd = str(s.get("framing_directive") or "")
                    meaning = str(s.get("meaning") or "")
                    intensity = float(s.get("intensity") or 0.5)
                    shot_for_compose = dict(s)
                    if light_is_stale:
                        shot_for_compose["lighting_style"] = _pick_lighting(
                            expression_mode=expression_mode,
                            shot_index=int(idx),
                            framing_directive=fd,
                            meaning=meaning,
                            intensity=intensity,
                        )
                    if palette_is_stale:
                        shot_for_compose["color_palette"] = _pick_palette(
                            expression_mode=expression_mode,
                            shot_index=int(idx),
                            framing_directive=fd,
                            meaning=meaning,
                        )
            except Exception:
                shot_for_compose = s

        try:
            prompt, _ = compose_image_prompt(
                shot_for_compose,
                character=character,
                location=location,
                has_character_ref=bool(char_url) and has_human_focus,
                has_environment_ref=bool(env_url),
                cine_prefix=cine_prefix,
            )
            out[idx] = prompt
        except Exception:
            continue
    return out


def composed_prompt_preview(project_id: str, shot: dict) -> str:
    """Return the prompt that will actually be sent to the image model
    for this shot, so the stills_control UI can show users the real
    text instead of the legacy 4000-char styled_visual_prompt that was
    seeded into shot_assets.prompt before the composer existed.

    Mirrors what generate_shot_still does: resolves the linked
    character/location (with backfill), prepends the cinematography
    rig clause, and runs the composer. Does NOT include the negative
    prompt (UI shows positive only)."""
    from shot_prompt_composer import compose_image_prompt
    idx = shot.get("shot_index") or shot.get("timeline_index")
    char_url, env_url, character, location = _resolve_shot_refs_full(project_id, idx)

    expression_mode = (shot.get("expression_mode") or "environment").lower()
    has_human_focus = expression_mode in {"face", "body"}

    # Always re-derive cinematography from the current engine so the UI
    # preview reflects the latest rig/lens/direction logic — not the
    # stale block baked into styled_timeline at the time of last
    # storyboard generation. Falls back to the saved block if the engine
    # returns None (e.g., legacy projects without a chosen Creative Brief).
    cine_block = None
    _lens_clause = None
    try:
        from cinematography_engine import derive as _cine_derive, lens_clause as _lens_clause
        ctx, sp, recent = _project_cine_context(project_id, upto_idx=idx)
        cine_block = _cine_derive(
            shot, ctx, sp,
            prev_block=(recent[-1] if recent else None),
            recent_blocks=recent[-4:],
        )
    except Exception:
        cine_block = None
    if not cine_block and isinstance(shot.get("cinematography"), dict):
        cine_block = shot.get("cinematography")
    cine_prefix = ""
    if cine_block and cine_block.get("rig") and _lens_clause:
        try:
            cine_prefix = (_lens_clause(cine_block) or "").strip()
        except Exception:
            cine_prefix = ""

    prompt, _ = compose_image_prompt(
        shot,
        character=character,
        location=location,
        has_character_ref=bool(char_url) and has_human_focus,
        has_environment_ref=bool(env_url),
        cine_prefix=cine_prefix,
    )
    return prompt


def seed_shot_rows_with_prompts(project_id: str, styled_timeline: list) -> None:
    """Seed shot_assets rows with prompts from the styled_timeline so the
    stills_control page can display editable prompts before generation starts.
    Does NOT overwrite rows that are already 'ready' (user-uploaded or previously generated).
    """
    with _db() as conn, conn.cursor() as cur:
        for shot in styled_timeline:
            idx = shot.get("shot_index") or shot.get("timeline_index")
            if idx is None:
                continue
            prompt = (shot.get("styled_visual_prompt") or shot.get("visual_prompt") or "")[:4000]
            motion = (shot.get("motion_prompt") or "")[:400] or None  # safety cap — builder already sizes to model limit
            cur.execute(
                """
                INSERT INTO shot_assets (project_id, shot_index, status, prompt, motion_prompt)
                VALUES (%s, %s, 'pending', %s, %s)
                ON CONFLICT (project_id, shot_index) DO UPDATE
                   SET prompt        = COALESCE(EXCLUDED.prompt, shot_assets.prompt),
                       motion_prompt = COALESCE(EXCLUDED.motion_prompt, shot_assets.motion_prompt),
                       status        = CASE WHEN shot_assets.status IN ('ready') THEN shot_assets.status
                                            ELSE 'pending' END,
                       updated_at    = NOW()
                """,
                (project_id, idx, prompt, motion),
            )
        conn.commit()


def update_shot_prompt(project_id: str, shot_index: int, prompt: str) -> None:
    """Persist an edited prompt for a shot without changing its status.
    Marks the row as ``prompt_user_edited=TRUE`` so the renderer uses
    the prompt verbatim instead of re-composing from structured fields.
    Leaves ready/rendering shots untouched — the new prompt is used on next generate."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO shot_assets (project_id, shot_index, status, prompt, prompt_user_edited)
            VALUES (%s, %s, 'pending', %s, TRUE)
            ON CONFLICT (project_id, shot_index) DO UPDATE
               SET prompt             = EXCLUDED.prompt,
                   prompt_user_edited = TRUE,
                   updated_at         = NOW()
            """,
            (project_id, shot_index, prompt[:4000]),
        )
        conn.commit()


def set_shot_uploaded_image(project_id: str, shot_index: int, file_url: str) -> None:
    """Mark a shot as user-uploaded (ready) so it is skipped during generate-all."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO shot_assets (project_id, shot_index, status, file_path, source)
            VALUES (%s, %s, 'ready', %s, 'uploaded')
            ON CONFLICT (project_id, shot_index) DO UPDATE
               SET status = 'ready', file_path = EXCLUDED.file_path,
                   source = 'uploaded', error = NULL, updated_at = NOW()
            """,
            (project_id, shot_index, file_url),
        )
        conn.commit()


def kick_single_shot(project_id: str, shot_index: int) -> None:
    """Generate a single shot still in the background.
    Looks up the shot from the project's styled_timeline so it always has
    the latest prompt data.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT styled_timeline FROM projects WHERE id=%s",
            (project_id,),
        )
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        logger.warning("kick_single_shot: no timeline for project %s", project_id)
        return
    timeline = list(row["styled_timeline"])
    shot = next(
        (s for s in timeline
         if (s.get("shot_index") or s.get("timeline_index")) == shot_index),
        None,
    )
    if shot is None:
        logger.warning("kick_single_shot: shot_index %s not in timeline", shot_index)
        return

    # Honour any user-edited prompt stored in shot_assets
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT prompt FROM shot_assets WHERE project_id=%s AND shot_index=%s",
            (project_id, shot_index),
        )
        asset_row = cur.fetchone()
    if asset_row and asset_row.get("prompt"):
        shot = {**shot, "styled_visual_prompt": asset_row["prompt"]}

    _SHOT_EXECUTOR.submit(_render_shot, project_id, shot, None, None)


def kick_all_pending_shots(project_id: str, force: bool = False) -> None:
    """Generate shots in the background.

    With force=False (default) skips shots already marked ``ready``.
    With force=True regenerates every shot in the timeline, including
    already-ready ones — used for the bulk "Regenerate All" action.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT styled_timeline FROM projects WHERE id=%s",
            (project_id,),
        )
        row = cur.fetchone()
        cur.execute(
            "SELECT shot_index, status, prompt FROM shot_assets WHERE project_id=%s",
            (project_id,),
        )
        asset_rows = {r["shot_index"]: r for r in (cur.fetchall() or [])}

    if not row or not row.get("styled_timeline"):
        return
    timeline = list(row["styled_timeline"])

    for shot in timeline:
        idx = shot.get("shot_index") or shot.get("timeline_index")
        asset = asset_rows.get(idx, {})
        if not force and asset.get("status") == "ready":
            continue
        merged = {**shot}
        if asset.get("prompt"):
            merged["styled_visual_prompt"] = asset["prompt"]
        _SHOT_EXECUTOR.submit(_render_shot, project_id, merged, None, None)


def seed_video_rows(project_id: str, shot_indices: list) -> None:
    """Public: seed video_assets rows as 'queued' without starting rendering."""
    _seed_video_rows(project_id, [i for i in shot_indices if i is not None])


def render_shot_videos(project_id: str, shot_indices: set) -> None:
    """Submit specific shots (by index) to the video render executor."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT styled_timeline FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        return
    for shot in row["styled_timeline"]:
        idx = shot.get("shot_index") or shot.get("timeline_index")
        if idx in shot_indices:
            _VIDEO_EXECUTOR.submit(_render_video, project_id, shot)


def render_failed_videos(project_id: str) -> int:
    """Re-queue all failed video clips. Returns count submitted."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT shot_index FROM video_assets WHERE project_id=%s AND status='failed'",
            (project_id,),
        )
        failed_idxs = {r["shot_index"] for r in cur.fetchall()}
    if not failed_idxs:
        return 0
    render_shot_videos(project_id, failed_idxs)
    return len(failed_idxs)


def kick_stage_4(project_id: str) -> None:
    _EXECUTOR.submit(_stage4_job, project_id)


def _stage5_job(project_id: str) -> None:
    """Stage 5 — Final cut: concat all video clips + mux original audio."""
    import shutil
    import subprocess
    import tempfile
    import requests

    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, audio_filename, styled_timeline "
                "  FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")
        if not row.get("styled_timeline"):
            raise RuntimeError("No storyboard found for project.")

        timeline = list(row["styled_timeline"])
        timeline.sort(key=lambda s: (s.get("start_time") or 0,
                                     s.get("shot_index") or s.get("timeline_index") or 0))
        ordered_indices = [s.get("shot_index") or s.get("timeline_index") for s in timeline]

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT shot_index, file_path, status FROM video_assets "
                " WHERE project_id=%s",
                (project_id,),
            )
            vrows = cur.fetchall() or []
        clip_by_idx = {v["shot_index"]: v for v in vrows}

        ordered_clips: list[dict] = []
        missing: list[int] = []
        for idx in ordered_indices:
            v = clip_by_idx.get(idx)
            if not v or v.get("status") != "ready" or not v.get("file_path"):
                missing.append(idx)
                continue
            ordered_clips.append(v)

        if missing:
            raise RuntimeError(
                f"Cannot assemble: {len(missing)} video clip(s) not ready "
                f"(shots {missing}). Retry them on the Videos step first."
            )
        if not ordered_clips:
            raise RuntimeError("No ready video clips to assemble.")

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found on system.")

        _set_status(project_id, "running",
                    {"stage": "final",
                     "label": f"Downloading {len(ordered_clips)} clips…",
                     "total": len(ordered_clips)},
                    stage="running_5")

        work_dir = Path(tempfile.mkdtemp(prefix=f"qaivid_final_{project_id}_"))
        try:
            local_clips: list[Path] = []
            for i, v in enumerate(ordered_clips):
                dst = work_dir / f"clip_{i:04d}.mp4"
                resp = requests.get(v["file_path"], timeout=600, stream=True)
                resp.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                local_clips.append(dst)

            # Resolve audio source: prefer local upload; fall back to R2.
            audio_local: Optional[Path] = None
            audio_filename = row.get("audio_filename")
            if audio_filename:
                candidate = PROJECTS_ROOT / project_id / "uploads" / audio_filename
                if candidate.is_file():
                    audio_local = candidate
                elif r2_storage.r2_available():
                    try:
                        data = r2_storage.download_bytes(
                            f"projects/{project_id}/uploads/{audio_filename}"
                        )
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        candidate.write_bytes(data)
                        audio_local = candidate
                    except Exception:
                        logger.exception("Could not pull audio from R2 for project %s",
                                         project_id)

            _set_status(project_id, "running",
                        {"stage": "final",
                         "label": "Stitching clips…",
                         "total": len(ordered_clips)},
                        stage="running_5")

            # Concat list file (concat demuxer requires identical codecs; we
            # re-encode below to be safe across mixed FAL outputs).
            concat_list = work_dir / "concat.txt"
            with open(concat_list, "w") as f:
                for p in local_clips:
                    f.write(f"file '{p.as_posix()}'\n")

            silent_cut = work_dir / "stitched.mp4"
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-r", "24",
                "-an",
                str(silent_cut),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            final_out = work_dir / "final.mp4"
            if audio_local:
                _set_status(project_id, "running",
                            {"stage": "final", "label": "Muxing audio…"},
                            stage="running_5")
                cmd = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(silent_cut),
                    "-i", str(audio_local),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    str(final_out),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                logger.warning("No source audio for project %s; final cut will be silent.",
                               project_id)
                final_out = silent_cut

            _set_status(project_id, "running",
                        {"stage": "final", "label": "Uploading final cut…"},
                        stage="running_5")

            safe_name = (row.get("name") or "qaivid").strip().replace(" ", "_")
            r2_key = f"projects/{project_id}/final/{safe_name}_final.mp4"
            final_url = r2_storage.upload_file(final_out, r2_key)

            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE projects SET final_video_url=%s, updated_at=NOW() "
                    " WHERE id=%s",
                    (final_url, project_id),
                )
                conn.commit()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        # Snapshot project to training dataset
        try:
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT text, genre, name, context_packet, styled_timeline, summary "
                    "FROM projects WHERE id=%s",
                    (project_id,),
                )
                ds_row = cur.fetchone() or {}
            dataset_collector.save_project_dataset(project_id, ds_row)
        except Exception:
            logger.warning("Dataset snapshot pre-fetch failed for project %s", project_id)

        _set_status(project_id, "awaiting_review",
                    {"stage": "final", "label": "Final cut ready."},
                    stage="final_review")

    except subprocess.CalledProcessError as exc:
        logger.exception("ffmpeg failed for project=%s", project_id)
        stderr = (exc.stderr or b"").decode("utf-8", "ignore")[:1500]
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Final assembly failed."},
                    stage="failed",
                    error=f"ffmpeg failed:\n{stderr}")
    except Exception as exc:
        logger.exception("Stage 5 failed for project=%s", project_id)
        _set_status(project_id, "failed",
                    {"stage": "error", "label": "Final assembly failed."},
                    stage="failed",
                    error=f"{exc}\n{traceback.format_exc(limit=4)}")


def kick_stage_5(project_id: str) -> None:
    _EXECUTOR.submit(_stage5_job, project_id)


# Legacy alias kept so existing tests/scripts don't break
def kick_pipeline(project_id: str, name: str, text: str, genre: str,
                  audio_path: Optional[Path],
                  uploaded_character_ref: Optional[str],
                  uploaded_env_ref: Optional[str]) -> None:
    kick_stage_0(project_id, audio_path, text, genre)


def retry_all_failed_shots(project_id: str) -> int:
    """Reset all failed shots to pending and resubmit them for rendering.

    Returns the number of shots queued.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT styled_timeline FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        raise RuntimeError("Project has no styled_timeline yet.")

    timeline = row["styled_timeline"]
    timeline_by_idx = {
        (s.get("shot_index") or s.get("timeline_index")): s for s in timeline
    }

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT shot_index FROM shot_assets "
            " WHERE project_id=%s AND status='failed'",
            (project_id,),
        )
        failed_rows = cur.fetchall() or []

    if not failed_rows:
        return 0

    def _ref_url(role: str) -> Optional[str]:
        ref = _get_ref(project_id, role)
        if not ref or not ref.get("file_path"):
            return None
        return ref["file_path"]

    char_url = _ref_url("character")
    env_url  = _ref_url("environment")

    queued = 0
    for r in failed_rows:
        idx = r["shot_index"]
        shot = timeline_by_idx.get(idx)
        if not shot:
            continue
        # Reset to pending so _render_shot can update it
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE shot_assets SET status='pending', error=NULL "
                " WHERE project_id=%s AND shot_index=%s",
                (project_id, idx),
            )
            conn.commit()
        _SHOT_EXECUTOR.submit(_render_shot, project_id, shot, char_url, env_url)
        queued += 1

    logger.info("retry_all_failed_shots: queued %d shots for project=%s", queued, project_id)
    return queued


def retry_shot(project_id: str, shot_index: int) -> None:
    """Re-render a single shot using the stored styled_timeline + character ref."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT styled_timeline FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        raise RuntimeError("Project has no styled_timeline yet.")

    timeline = row["styled_timeline"]
    target = next(
        (s for s in timeline if (s.get("shot_index") or s.get("timeline_index")) == shot_index),
        None,
    )
    if not target:
        raise RuntimeError(f"Shot {shot_index} not found.")

    def _ref_url(role: str) -> Optional[str]:
        ref = _get_ref(project_id, role)
        if not ref or not ref.get("file_path"):
            return None
        return ref["file_path"]

    _SHOT_EXECUTOR.submit(
        _render_shot, project_id, target,
        _ref_url("character"), _ref_url("environment"),
    )


def retry_video(project_id: str, shot_index: int) -> None:
    """Re-render a single video clip from the stored still."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT styled_timeline FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        raise RuntimeError("Project has no styled_timeline yet.")

    timeline = row["styled_timeline"]
    target = next(
        (s for s in timeline if (s.get("shot_index") or s.get("timeline_index")) == shot_index),
        None,
    )
    if not target:
        raise RuntimeError(f"Shot {shot_index} not found in timeline.")

    _VIDEO_EXECUTOR.submit(_render_video, project_id, target)


def generate_all_videos(project_id: str) -> None:
    """Trigger video generation for all shots in a completed project."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT styled_timeline FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("styled_timeline"):
        raise RuntimeError("Project has no styled_timeline yet.")

    timeline = row["styled_timeline"]
    shot_indices = [(s.get("shot_index") or s.get("timeline_index")) for s in timeline]
    _seed_video_rows(project_id, shot_indices)

    for shot in timeline:
        _VIDEO_EXECUTOR.submit(_render_video, project_id, shot)


def retry_all_failed_refs(project_id: str) -> dict:
    """Re-queue all failed character plates, location plates, and look plates.

    Returns counts: {"characters": n, "locations": n, "looks": n}
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT context_packet FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("context_packet"):
        raise RuntimeError("Project context not ready yet.")

    location_dna = (row.get("context_packet") or {}).get("location_dna") or "Universal"

    counts = {"characters": 0, "locations": 0, "looks": 0}

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM characters WHERE project_id=%s AND ref_status='failed'",
            (project_id,),
        )
        failed_chars = [r["id"] for r in (cur.fetchall() or [])]

        cur.execute(
            "SELECT id FROM locations WHERE project_id=%s AND ref_status='failed'",
            (project_id,),
        )
        failed_locs = [r["id"] for r in (cur.fetchall() or [])]

        cur.execute(
            "SELECT id FROM character_looks WHERE project_id=%s AND ref_status='failed'",
            (project_id,),
        )
        failed_look_ids = [r["id"] for r in (cur.fetchall() or [])]

    for char_id in failed_chars:
        _set_entity_ref(project_id, "character", char_id, status="rendering",
                        source="ai", error=None)
        _SHOT_EXECUTOR.submit(_render_character_plate, project_id, char_id,
                              location_dna, None)
        counts["characters"] += 1

    for loc_id in failed_locs:
        _set_entity_ref(project_id, "location", loc_id, status="rendering",
                        source="ai", error=None)
        _SHOT_EXECUTOR.submit(_render_location_plate, project_id, loc_id,
                              None, None)
        counts["locations"] += 1

    if failed_look_ids:
        # Reset look plates to pending and let generate_look_plates handle them
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE character_looks SET ref_status='pending', ref_error=NULL "
                " WHERE project_id=%s AND ref_status='failed'",
                (project_id,),
            )
            conn.commit()
        # Trigger look plate generation in a background thread
        def _regen_looks():
            try:
                from wardrobe_engine import generate_look_plates
                generate_look_plates(project_id, location_dna, None)
            except Exception:
                logger.exception("retry_all_failed_refs: look plate regen failed "
                                 "for project=%s", project_id)
        _SHOT_EXECUTOR.submit(_regen_looks)
        counts["looks"] = len(failed_look_ids)

    logger.info("retry_all_failed_refs: queued %s for project=%s", counts, project_id)
    return counts


def retry_ref(project_id: str, role: str) -> None:
    """Re-render a missing/failed generated reference."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT context_packet FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
    if not row or not row.get("context_packet"):
        raise RuntimeError("Project context not ready yet.")

    ctx = row["context_packet"]

    def _job():
        try:
            if role == "character":
                _ensure_character_ref(project_id, ctx)
            elif role == "environment":
                _ensure_env_ref(project_id, ctx)
        except Exception:
            logger.exception("Ref retry failed")

    _REF_EXECUTOR.submit(_job)


# ── Post Production — Quick Video assembly (Task #100) ────────────────────────

_POSTPROD_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qaivid-postprod")

_KB_MODES = {
    "zoom-in":    "zoompan=z='min(zoom+0.0015,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "zoom-out":   "zoompan=z='if(eq(on,1),1.3,max(zoom-0.0015,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "pan-left":   "zoompan=z=1.1:x='iw/2-(iw/zoom/2)+t*(iw*0.04/d)':y='ih/2-(ih/zoom/2)'",
    "pan-right":  "zoompan=z=1.1:x='iw/2-(iw/zoom/2)-t*(iw*0.04/d)':y='ih/2-(ih/zoom/2)'",
    "pan-up":     "zoompan=z=1.1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)+t*(ih*0.04/d)'",
    "pan-down":   "zoompan=z=1.1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)-t*(ih*0.04/d)'",
    "static":     "zoompan=z=1.0:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "random":     None,
}

_KB_RANDOM_POOL = ["zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down"]

_COLOUR_FILTERS = {
    "original":       "",
    "film-grain":     "noise=alls=8:allf=t+u,curves=all='0/0 0.4/0.38 0.7/0.68 1/1'",
    "vintage":        "curves=r='0/0.05 1/0.9':g='0/0.02 1/0.88':b='0/0.08 1/0.85',hue=s=0.7",
    "bleach-bypass":  "colorchannelmixer=rr=0.5:rg=0.25:rb=0.25:gr=0.25:gg=0.5:gb=0.25:br=0.25:bg=0.25:bb=0.5,curves=all='0/0 0.5/0.42 1/1'",
    "teal-orange":    "curves=r='0/0.08 0.5/0.6 1/0.95':g='0/0 0.5/0.5 1/1':b='0/0.05 0.4/0.55 1/0.7',hue=s=1.3",
    "desaturated":    "hue=s=0.25,curves=all='0/0.05 1/0.95'",
    "high-contrast":  "curves=all='0/0 0.3/0.18 0.7/0.85 1/1'",
    "warm-glow":      "curves=r='0/0.1 1/1':g='0/0.02 1/0.95':b='0/0 1/0.75',hue=s=1.1",
    "cool-tone":      "curves=r='0/0 1/0.85':g='0/0 1/0.95':b='0/0.05 1/1',hue=s=1.05",
}

_XFADE_NAMES = {
    "cut":          None,
    "crossfade":    "fade",
    "dip-to-black": "fade",
    "dissolve":     "dissolve",
    "wipe":         "wipeleft",
}

# Aspect ratio → (w, h) for FFmpeg scale+pad
_ASPECT_DIMS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "4:3":  (1440, 1080),
}

# Quality presets → (short_side_pixels, crf, ffmpeg_preset)
# Output resolution = aspect_ratio_dims scaled so that min(w,h) == short_side
_QUALITY_PRESETS = {
    "480p":    (480,  26, "veryfast"),
    "720p":    (720,  23, "fast"),
    "1080p":   (1080, 20, "medium"),
    "1080p-hq":(1080, 18, "slow"),
    "4k":      (2160, 18, "medium"),
    # Legacy keys for backwards-compat (some old postprod_configs use these)
    "9:16":    (1080, 20, "medium"),
    "1:1":     (1080, 20, "medium"),
}


def _kb_expr(mode: Optional[str], shot_fps: int, shot_dur: float,
             out_w: int = 1920, out_h: int = 1080) -> str:
    """Return a zoompan filter string for the given Ken Burns mode and output size."""
    import random as _rnd
    if not mode or mode == "random":
        mode = _rnd.choice(_KB_RANDOM_POOL)
    base = _KB_MODES.get(mode, _KB_MODES["zoom-in"])
    n_frames = max(1, int(shot_fps * shot_dur))
    return f"{base}:d={n_frames}:fps={shot_fps}:s={out_w}x{out_h}"


def _srt_time_to_cs(t: str) -> int:
    """Convert SRT timestamp string to centiseconds."""
    import re as _re
    m = _re.match(r'(\d+):(\d+):(\d+),(\d+)', t)
    if not m:
        return 0
    h, mi, s, ms = m.groups()
    return (int(h) * 3600 + int(mi) * 60 + int(s)) * 100 + int(ms) // 10


def _srt_time_to_ass(t: str) -> str:
    """Convert SRT timestamp to ASS time format (H:MM:SS.cc)."""
    import re as _re
    m = _re.match(r'(\d+):(\d+):(\d+),(\d+)', t)
    if not m:
        return "0:00:00.00"
    h, mi, s, ms = m.groups()
    return f"{int(h)}:{mi}:{s}.{ms[:2]}"


def _srt_to_ass_karaoke(srt_bytes: bytes, style: dict) -> str:
    """Convert an SRT file to ASS format with word-by-word karaoke fill timing."""
    import re as _re
    content = srt_bytes.decode("utf-8", errors="replace")

    font       = style.get("font", "Arial")
    font_size  = int(style.get("font_size", 24))
    primary_c  = style.get("primary_colour", "&H00FFFFFF")
    outline    = float(style.get("outline", 1.5))
    shadow     = float(style.get("shadow", 0))
    margin_v   = int(style.get("margin_v", 40))
    margin_l   = int(style.get("margin_l", 10))
    margin_r   = int(style.get("margin_r", 10))
    alignment  = int(style.get("alignment", 2))

    # Strip the leading &H from colour to get BBGGRR hex
    def _c(col: str) -> str:
        return col.lstrip("&H").lstrip("&h")

    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{font_size},"
        f"&H{_c(primary_c)},&H0000FFFF,&H00000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,{outline:.1f},{shadow:.1f},{alignment},{margin_l},{margin_r},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    blocks = _re.split(r'\n\n+', content.strip())
    event_lines: list[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parts = block.split("\n", 2)
        if len(parts) < 3:
            continue
        _, timecode, text_raw = parts[0], parts[1], parts[2]

        m = _re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            timecode,
        )
        if not m:
            continue

        dur_cs = max(1, _srt_time_to_cs(m.group(2)) - _srt_time_to_cs(m.group(1)))
        clean  = _re.sub(r'<[^>]+>', '', text_raw.replace("\n", " ")).strip()
        words  = clean.split()
        if not words:
            continue

        per_word_cs = max(1, dur_cs // len(words))
        kar_text = "".join(f"{{\\kf{per_word_cs}}}{w} " for w in words).rstrip()

        s_time = _srt_time_to_ass(m.group(1))
        e_time = _srt_time_to_ass(m.group(2))
        event_lines.append(f"Dialogue: 0,{s_time},{e_time},Default,,0,0,0,,{kar_text}")

    return ass_header + "\n".join(event_lines) + "\n"


def _assemble_quick_video_job(project_id: str, settings: dict) -> None:
    """Background job: assembles approved stills into a polished quick video MP4."""
    import shutil
    import subprocess
    import tempfile
    import requests
    import json as _json
    import random as _rnd

    try:
        ffmpeg = shutil.which("ffmpeg") or (
            "/nix/store/ynlnyy6rn70kvzamy3b40bp3qlz70mn0-ffmpeg-full-7.1.1-bin/bin/ffmpeg"
        )
        if not ffmpeg or not Path(ffmpeg).exists():
            raise RuntimeError("ffmpeg binary not found.")

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, audio_filename, styled_timeline, postprod_config "
                "FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        timeline = row.get("styled_timeline") or []
        if not timeline:
            raise RuntimeError("No styled timeline — run the storyboard pipeline first.")

        # ── Settings ──────────────────────────────────────────────────────────
        global_kb    = settings.get("ken_burns_mode", "zoom-in")
        per_shot_kb  = settings.get("per_shot_kb") or {}           # {str(idx): mode}
        transition   = settings.get("transition", "crossfade")
        trans_dur    = float(settings.get("transition_duration", 0.8))
        colour_filter    = settings.get("colour_filter", "original").lower().replace(" ", "-")
        filter_intensity = max(0, min(100, int(settings.get("filter_intensity", 100))))
        aspect       = settings.get("aspect_ratio", "16:9")
        quality      = settings.get("quality", "1080p").lower().replace(" ", "-")
        logo_slots   = settings.get("logos") or {}                  # {slot: {r2_key, opacity, ...}}
        srt_r2_key   = settings.get("srt_r2_key") or ""
        per_shot_dur = settings.get("per_shot_duration") or {}      # {str(idx): float}

        # ── Work dir ──────────────────────────────────────────────────────────
        work_dir = Path(tempfile.mkdtemp(prefix=f"qv_quick_{project_id}_"))
        try:
            # ── Collect shots ─────────────────────────────────────────────────
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT shot_index, file_path FROM shot_assets "
                    " WHERE project_id=%s AND status='ready' ORDER BY shot_index",
                    (project_id,),
                )
                ready_rows = cur.fetchall()

            if not ready_rows:
                raise RuntimeError("No ready stills found — generate stills first.")

            timeline_map = {
                (s.get("shot_index") or s.get("timeline_index")): s for s in timeline
            }

            # ── Output dimensions ─────────────────────────────────────────────
            # Quality preset may override aspect ratio (9:16 and 1:1 presets)
            if quality in ("9:16", "1:1"):
                aspect = quality
            aw, ah = _ASPECT_DIMS.get(aspect, (1920, 1080))
            qshort, crf, enc_preset = _QUALITY_PRESETS.get(quality, (1080, 20, "medium"))
            # Scale aspect ratio dims so the short side matches the quality target
            native_short = min(aw, ah)
            _scale = qshort / native_short
            out_w = max(2, int(aw * _scale) // 2 * 2)  # ensure even number of pixels
            out_h = max(2, int(ah * _scale) // 2 * 2)

            shot_fps = 25
            local_stills: list[tuple[int, Path, float]] = []

            for r in ready_rows:
                idx = r["shot_index"]
                url = r["file_path"]
                shot = timeline_map.get(idx) or {}
                if str(idx) in per_shot_dur:
                    dur = float(per_shot_dur[str(idx)])
                else:
                    dur = float(shot.get("duration") or 4.0)
                dur = max(2.0, dur)  # WAN 2.6 minimum billable duration

                dst = work_dir / f"still_{idx:04d}.jpg"
                try:
                    resp = requests.get(url, timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in resp.iter_content(1 << 16):
                            f.write(chunk)
                    local_stills.append((idx, dst, dur))
                except Exception:
                    logger.warning("Quick video: could not download still %s for project %s", idx, project_id)

            if not local_stills:
                raise RuntimeError("Could not download any ready stills.")

            # ── Fetch audio early to measure its duration ──────────────────────
            audio_local: Optional[Path] = None
            audio_filename = row.get("audio_filename")
            if audio_filename:
                candidate = PROJECTS_ROOT / project_id / "uploads" / audio_filename
                if candidate.is_file():
                    audio_local = candidate
                elif r2_storage.r2_available():
                    try:
                        data = r2_storage.download_bytes(
                            f"projects/{project_id}/uploads/{audio_filename}"
                        )
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        candidate.write_bytes(data)
                        audio_local = candidate
                    except Exception:
                        logger.warning("Quick video: could not fetch audio for project %s", project_id)

            # ── Measure audio duration via ffprobe ─────────────────────────────
            audio_dur: float = 0.0
            if audio_local:
                _ffprobe = Path(ffmpeg).with_name("ffprobe")
                try:
                    _probe = subprocess.run(
                        [str(_ffprobe), "-v", "quiet", "-show_entries",
                         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                         str(audio_local)],
                        capture_output=True, text=True, check=True,
                    )
                    audio_dur = float(_probe.stdout.strip())
                except Exception:
                    logger.warning("Quick video: ffprobe duration failed — stills will not be extended",
                                   exc_info=True)

            # ── Legacy fallback: proportional scale for old projects ──────────
            # New storyboards produced by the fixed RhythmicAssemblyEngine
            # already store correct durations (sum == audio_duration_seconds),
            # so this block should be a no-op for those.  For existing projects
            # whose styled_timeline was built before the fix, we still scale so
            # the quick video covers the full audio track.
            if audio_dur > 0.1:
                total_stills_dur = sum(d for _, _, d in local_stills)
                if total_stills_dur > 0 and abs(total_stills_dur - audio_dur) > 0.5:
                    scale = audio_dur / total_stills_dur
                    local_stills = [(idx, path, max(0.5, dur * scale))
                                    for idx, path, dur in local_stills]
                    logger.warning(
                        "Quick video: legacy duration mismatch — scaled stills %.1fs → %.1fs "
                        "(audio) ×%.3f for project %s. Re-run storyboard stage to fix permanently.",
                        total_stills_dur, audio_dur, scale, project_id,
                    )

            # ── Build per-shot processed clips ────────────────────────────────
            processed_clips: list[Path] = []
            colour_expr = _COLOUR_FILTERS.get(colour_filter, "")

            for i, (idx, still_path, dur) in enumerate(local_stills):
                shot_mode = per_shot_kb.get(str(idx)) or global_kb
                kb_expr   = _kb_expr(shot_mode, shot_fps, dur, out_w, out_h)
                n_frames  = max(1, int(shot_fps * dur))
                out_clip  = work_dir / f"clip_{i:04d}.mp4"

                # scale + fill to chosen aspect ratio, then apply Ken Burns
                fill_mode = settings.get("fill_mode", "crop")
                if fill_mode == "blur-fill":
                    # Two-layer composite: blurred crop background + fitted foreground
                    alpha = filter_intensity / 100.0
                    if colour_expr and 0.0 < alpha < 1.0:
                        # Partial intensity: blend after compose
                        colour_node = (
                            f"[_composed]split=2[_co][_cf];"
                            f"[_cf]{colour_expr}[_cfilt];"
                            f"[_co][_cfilt]blend=all_expr='A*{1-alpha:.6f}+B*{alpha:.6f}'[vkb];"
                            f"[vkb]"
                        )
                        fc = (
                            f"[0:v]split=2[_bg][_fg];"
                            f"[_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                            f"crop={out_w}:{out_h},gblur=sigma=25[_blurred];"
                            f"[_fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2[_padded];"
                            f"[_blurred][_padded]overlay=0:0[_composed];"
                            f"{colour_node}{kb_expr}[vout]"
                        )
                    elif colour_expr and alpha > 0:
                        colour_node = f",{colour_expr}"
                        fc = (
                            f"[0:v]split=2[_bg][_fg];"
                            f"[_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                            f"crop={out_w}:{out_h},gblur=sigma=25[_blurred];"
                            f"[_fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2[_padded];"
                            f"[_blurred][_padded]overlay=0:0[_composed];"
                            f"[_composed]{kb_expr}{colour_node}[vout]"
                        )
                    else:
                        fc = (
                            f"[0:v]split=2[_bg][_fg];"
                            f"[_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                            f"crop={out_w}:{out_h},gblur=sigma=25[_blurred];"
                            f"[_fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2[_padded];"
                            f"[_blurred][_padded]overlay=0:0[_composed];"
                            f"[_composed]{kb_expr}[vout]"
                        )
                    cmd = [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-loop", "1", "-i", str(still_path),
                        "-filter_complex", fc,
                        "-map", "[vout]",
                        "-t", str(dur),
                        "-r", str(shot_fps),
                        "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-an",
                        str(out_clip),
                    ]
                else:
                    # Default: crop to fill
                    scale_pad = (
                        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                        f"crop={out_w}:{out_h}"
                    )
                    alpha = filter_intensity / 100.0
                    if colour_expr and 0.0 < alpha < 1.0:
                        # Partial intensity: use filter_complex blend
                        fc_crop = (
                            f"[0:v]{scale_pad},{kb_expr}[zoomed];"
                            f"[zoomed]split=2[_orig][_tofilt];"
                            f"[_tofilt]{colour_expr}[_filt];"
                            f"[_orig][_filt]blend=all_expr='A*{1-alpha:.6f}+B*{alpha:.6f}'[vout]"
                        )
                        cmd = [
                            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-loop", "1", "-i", str(still_path),
                            "-filter_complex", fc_crop,
                            "-map", "[vout]",
                            "-t", str(dur),
                            "-r", str(shot_fps),
                            "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                            "-pix_fmt", "yuv420p", "-an",
                            str(out_clip),
                        ]
                    else:
                        vf_chain = f"{scale_pad},{kb_expr}"
                        if colour_expr and alpha > 0:
                            vf_chain = f"{vf_chain},{colour_expr}"
                        cmd = [
                            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-loop", "1", "-i", str(still_path),
                            "-vf", vf_chain,
                            "-t", str(dur),
                            "-r", str(shot_fps),
                            "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                            "-pix_fmt", "yuv420p", "-an",
                            str(out_clip),
                        ]
                subprocess.run(cmd, check=True, capture_output=True)
                processed_clips.append(out_clip)

            # ── Concatenate clips (with or without xfade transitions) ─────────
            xfade_effect = _XFADE_NAMES.get(transition)
            use_dip_black = (transition == "dip-to-black")

            if len(processed_clips) == 1 or xfade_effect is None:
                # Hard cut or single clip: use concat demuxer
                concat_list = work_dir / "concat.txt"
                with open(concat_list, "w") as f:
                    for p in processed_clips:
                        f.write(f"file '{p.as_posix()}'\n")
                stitched = work_dir / "stitched.mp4"
                cmd = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c", "copy", str(stitched),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # xfade filter_complex chain
                td = min(trans_dur, min(d for _, _, d in local_stills) * 0.5)
                td = max(0.1, td)

                if use_dip_black:
                    # For dip-to-black: fade each clip out then in using fade filter
                    faded_clips: list[Path] = []
                    for i, clip in enumerate(processed_clips):
                        faded = work_dir / f"faded_{i:04d}.mp4"
                        dur_i = local_stills[i][2]
                        vf = ""
                        if i > 0:
                            vf += f"fade=t=in:st=0:d={td}:color=black"
                        if i < len(processed_clips) - 1:
                            fade_out_st = max(0, dur_i - td)
                            if vf:
                                vf += f",fade=t=out:st={fade_out_st:.3f}:d={td}:color=black"
                            else:
                                vf = f"fade=t=out:st={fade_out_st:.3f}:d={td}:color=black"
                        if vf:
                            cmd = [
                                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                "-i", str(clip),
                                "-vf", vf, "-c:v", "libx264",
                                "-preset", enc_preset, "-crf", str(crf),
                                "-pix_fmt", "yuv420p", "-an", str(faded),
                            ]
                            subprocess.run(cmd, check=True, capture_output=True)
                            faded_clips.append(faded)
                        else:
                            faded_clips.append(clip)

                    concat_list = work_dir / "concat.txt"
                    with open(concat_list, "w") as f:
                        for p in faded_clips:
                            f.write(f"file '{p.as_posix()}'\n")
                    stitched = work_dir / "stitched.mp4"
                    cmd = [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(concat_list),
                        "-c", "copy", str(stitched),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                else:
                    # xfade filter_complex
                    inputs = []
                    for clip in processed_clips:
                        inputs += ["-i", str(clip)]

                    n = len(processed_clips)
                    filter_parts = []
                    offsets: list[float] = []
                    cumulative = 0.0
                    for i in range(n - 1):
                        cumulative += local_stills[i][2] - td
                        offsets.append(round(cumulative, 3))

                    prev_label = "[0:v]"
                    for i in range(n - 1):
                        next_label = f"[v{i+1}]" if i < n - 2 else "[vout]"
                        filter_parts.append(
                            f"{prev_label}[{i+1}:v]xfade=transition={xfade_effect}"
                            f":duration={td}:offset={offsets[i]}{next_label}"
                        )
                        prev_label = f"[v{i+1}]"

                    filter_complex = ";".join(filter_parts)
                    stitched = work_dir / "stitched.mp4"
                    cmd = (
                        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
                        + inputs
                        + ["-filter_complex", filter_complex,
                           "-map", "[vout]",
                           "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                           "-pix_fmt", "yuv420p", "-an", str(stitched)]
                    )
                    subprocess.run(cmd, check=True, capture_output=True)

            # ── Logo overlays ─────────────────────────────────────────────────
            current_video = stitched
            if logo_slots and r2_storage.r2_available():
                logo_filter_parts = []
                logo_inputs = []
                logo_idx = 1

                valid_logos: list[tuple[str, dict, str]] = []
                for slot, logo_cfg in logo_slots.items():
                    r2_key = (logo_cfg or {}).get("r2_key", "")
                    if not r2_key:
                        continue
                    try:
                        logo_data = r2_storage.download_bytes(r2_key)
                        logo_path = work_dir / f"logo_{slot}.png"
                        logo_path.write_bytes(logo_data)
                        valid_logos.append((slot, logo_cfg, str(logo_path)))
                    except Exception:
                        logger.warning("Quick video: could not download logo for slot %s", slot)

                if valid_logos:
                    logo_out = work_dir / "with_logos.mp4"
                    cmd_parts = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", str(current_video)]
                    for _slot, _cfg, logo_path in valid_logos:
                        cmd_parts += ["-i", logo_path]

                    # Per-slot corner base positions (expressions use overlay_w/overlay_h after scale)
                    _corner_x = {
                        "top-left":     "20",
                        "top-right":    "main_w-overlay_w-20",
                        "bottom-left":  "20",
                        "bottom-right": "main_w-overlay_w-20",
                    }
                    _corner_y = {
                        "top-left":     "20",
                        "top-right":    "20",
                        "bottom-left":  "main_h-overlay_h-20",
                        "bottom-right": "main_h-overlay_h-20",
                    }
                    overlay_filter = ""
                    prev = "[0:v]"
                    for li, (_slot, logo_cfg, _lp) in enumerate(valid_logos):
                        opacity   = float((logo_cfg or {}).get("opacity", 0.9))
                        width     = int((logo_cfg or {}).get("width", 120))
                        height    = int((logo_cfg or {}).get("height", -1))  # -1 = auto
                        x_offset  = int((logo_cfg or {}).get("x_offset", 0))
                        y_offset  = int((logo_cfg or {}).get("y_offset", 0))
                        cx        = _corner_x.get(_slot, "20")
                        cy        = _corner_y.get(_slot, "20")
                        # Build offset-adjusted position expression
                        ox = f"({cx})+({x_offset})" if x_offset else cx
                        oy = f"({cy})+({y_offset})" if y_offset else cy
                        lbl_in  = f"[{li+1}:v]"
                        lbl_out = f"[ov{li}]" if li < len(valid_logos) - 1 else "[vfinal]"
                        overlay_filter += (
                            f"{lbl_in}scale={width}:{height}[ls{li}];"
                            f"[ls{li}]format=rgba,colorchannelmixer=aa={opacity:.2f}[la{li}];"
                            f"{prev}[la{li}]overlay={ox}:{oy}{lbl_out};"
                        )
                        prev = lbl_out

                    overlay_filter = overlay_filter.rstrip(";")
                    cmd_parts += [
                        "-filter_complex", overlay_filter,
                        "-map", "[vfinal]", "-map", "0:a?",
                        "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-an", str(logo_out),
                    ]
                    try:
                        subprocess.run(cmd_parts, check=True, capture_output=True)
                        current_video = logo_out
                    except subprocess.CalledProcessError as e:
                        logger.warning("Quick video: logo overlay failed — skipping (%s)",
                                       (e.stderr or b"").decode("utf-8", "ignore")[:300])

            # ── Audio mux ─────────────────────────────────────────────────────
            # audio_local already fetched above; video duration now matches audio.
            muxed = work_dir / "muxed.mp4"
            if audio_local:
                cmd = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(current_video),
                    "-i", str(audio_local),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    str(muxed),           # no -shortest: video already matches audio
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                current_video = muxed

            # ── Subtitle burn-in ──────────────────────────────────────────────
            if srt_r2_key and r2_storage.r2_available():
                try:
                    srt_data = r2_storage.download_bytes(srt_r2_key)
                    sub_style  = settings.get("subtitle_style") or {}
                    animation  = sub_style.get("animation", "none")

                    if animation == "karaoke":
                        # Convert SRT → ASS with karaoke word-by-word fill timing
                        ass_content = _srt_to_ass_karaoke(srt_data, sub_style)
                        sub_path = work_dir / "subtitles.ass"
                        sub_path.write_text(ass_content, encoding="utf-8")
                        sub_filter = f"ass={sub_path.as_posix()}"
                    else:
                        srt_path = work_dir / "subtitles.srt"
                        srt_path.write_bytes(srt_data)

                        font_name  = sub_style.get("font", "Arial")
                        font_size  = int(sub_style.get("font_size", 24))
                        primary_c  = sub_style.get("primary_colour", "&H00FFFFFF")
                        outline_c  = sub_style.get("outline_colour", "&H00000000")
                        back_c     = sub_style.get("back_colour", "&H80000000")
                        bold       = int(sub_style.get("bold", 0))
                        outline    = float(sub_style.get("outline", 1.5))
                        shadow     = float(sub_style.get("shadow", 0))
                        alignment  = int(sub_style.get("alignment", 2))
                        margin_v   = int(sub_style.get("margin_v", 40))
                        margin_l   = int(sub_style.get("margin_l", 10))
                        margin_r   = int(sub_style.get("margin_r", 10))

                        force_style = (
                            f"FontName={font_name},FontSize={font_size},"
                            f"PrimaryColour={primary_c},OutlineColour={outline_c},"
                            f"BackColour={back_c},Bold={bold},"
                            f"Outline={outline},Shadow={shadow},"
                            f"Alignment={alignment},"
                            f"MarginL={margin_l},MarginR={margin_r},MarginV={margin_v}"
                        )
                        sub_filter = (
                            f"subtitles={srt_path.as_posix()}:force_style='{force_style}'"
                        )

                    subtitled = work_dir / "subtitled.mp4"
                    cmd = [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(current_video),
                        "-vf", sub_filter,
                        "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                        "-c:a", "copy", str(subtitled),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    current_video = subtitled
                except Exception:
                    logger.warning("Quick video: subtitle burn-in failed (non-fatal)", exc_info=True)

            # ── Upload to R2 ──────────────────────────────────────────────────
            safe_name = (row.get("name") or "qaivid").strip().replace(" ", "_")
            r2_key = f"projects/{project_id}/quick/{safe_name}_quick.mp4"
            final_url = r2_storage.upload_file(current_video, r2_key)

            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE projects SET quick_video_url=%s, "
                    "postprod_config=COALESCE(postprod_config,'{}')::jsonb || %s::jsonb, "
                    "updated_at=NOW() WHERE id=%s",
                    (final_url, _json.dumps({"generating": False, "quick_video_error": None}), project_id),
                )
                conn.commit()

            logger.info("Quick video assembled for project %s → %s", project_id, final_url)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "ignore")[:800]
        logger.exception("Quick video ffmpeg error for project %s: %s", project_id, stderr)
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET postprod_config=postprod_config || %s::jsonb, updated_at=NOW() "
                "WHERE id=%s",
                (_json.dumps({"generating": False, "quick_video_error": f"ffmpeg failed: {stderr}"}), project_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Quick video assembly failed for project %s", project_id)
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET postprod_config=postprod_config || %s::jsonb, updated_at=NOW() "
                "WHERE id=%s",
                (_json.dumps({"generating": False, "quick_video_error": str(exc)[:400]}), project_id),
            )
            conn.commit()


def kick_quick_video(project_id: str, settings: dict) -> None:
    """Queue an async quick-video assembly job."""
    import json as _json
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET quick_video_url=NULL, "
            "postprod_config=COALESCE(postprod_config,'{}')::jsonb || %s::jsonb, "
            "updated_at=NOW() WHERE id=%s",
            (_json.dumps({"generating": True, "quick_video_error": None}), project_id),
        )
        conn.commit()
    _POSTPROD_EXECUTOR.submit(_assemble_quick_video_job, project_id, settings)


# ── AI Post-Production Export ──────────────────────────────────────────────────


def _assemble_ai_postprod_job(project_id: str, settings: dict) -> None:
    """Background job: applies post-production effects to the AI-rendered stitched video."""
    import shutil
    import subprocess
    import tempfile
    import requests
    import json as _json

    try:
        ffmpeg = shutil.which("ffmpeg") or (
            "/nix/store/ynlnyy6rn70kvzamy3b40bp3qlz70mn0-ffmpeg-full-7.1.1-bin/bin/ffmpeg"
        )
        if not ffmpeg or not Path(ffmpeg).exists():
            raise RuntimeError("ffmpeg binary not found.")

        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, final_video_url, postprod_config FROM projects WHERE id=%s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("Project not found.")

        final_video_url = row.get("final_video_url") or ""
        if not final_video_url:
            raise RuntimeError("AI video has not been rendered yet.")

        # ── Settings ──────────────────────────────────────────────────────────
        colour_filter    = settings.get("colour_filter", "original").lower().replace(" ", "-")
        filter_intensity = max(0, min(100, int(settings.get("filter_intensity", 100))))
        aspect       = settings.get("aspect_ratio", "16:9")
        quality      = settings.get("quality", "1080p").lower().replace(" ", "-")
        fill_mode    = settings.get("fill_mode", "crop")
        logo_slots   = settings.get("logos") or {}
        srt_r2_key   = settings.get("srt_r2_key") or ""

        # ── Output dimensions ─────────────────────────────────────────────────
        if quality in ("9:16", "1:1"):
            aspect = quality
        aw, ah = _ASPECT_DIMS.get(aspect, (1920, 1080))
        qshort, crf, enc_preset = _QUALITY_PRESETS.get(quality, (1080, 20, "medium"))
        native_short = min(aw, ah)
        _scale = qshort / native_short
        out_w = max(2, int(aw * _scale) // 2 * 2)
        out_h = max(2, int(ah * _scale) // 2 * 2)

        work_dir = Path(tempfile.mkdtemp(prefix=f"qv_ai_{project_id}_"))
        try:
            # ── Download the stitched AI video ────────────────────────────────
            ai_video_local = work_dir / "ai_source.mp4"
            resp = requests.get(final_video_url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(ai_video_local, "wb") as f:
                for chunk in resp.iter_content(1 << 16):
                    f.write(chunk)

            current_video = ai_video_local

            # ── Colour filter + scale/resize ───────────────────────────────────
            colour_expr = _COLOUR_FILTERS.get(colour_filter, "")
            alpha = filter_intensity / 100.0

            # Build scale/pad filter for target aspect ratio
            if fill_mode == "blur-fill":
                scale_filter = (
                    f"split=2[_bg][_fg];"
                    f"[_bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                    f"crop={out_w}:{out_h},gblur=sigma=25[_blurred];"
                    f"[_fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2[_padded];"
                    f"[_blurred][_padded]overlay=0:0[_scaled]"
                )
            else:
                scale_filter = (
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                    f"crop={out_w}:{out_h}[_scaled]"
                )

            # Combine with colour filter
            if colour_expr and 0.0 < alpha < 1.0:
                vf_filter = (
                    f"[0:v]{scale_filter};"
                    f"[_scaled]split=2[_co][_cf];"
                    f"[_cf]{colour_expr}[_cfilt];"
                    f"[_co][_cfilt]blend=all_expr='A*{1-alpha:.6f}+B*{alpha:.6f}'[vout]"
                )
            elif colour_expr and alpha > 0:
                vf_filter = f"[0:v]{scale_filter};[_scaled]{colour_expr}[vout]"
            else:
                vf_filter = f"[0:v]{scale_filter};[_scaled]copy[vout]"

            colour_out = work_dir / "colour_scaled.mp4"
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(current_video),
                "-filter_complex", vf_filter,
                "-map", "[vout]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-c:a", "copy",
                str(colour_out),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            current_video = colour_out

            # ── Logo overlays ──────────────────────────────────────────────────
            if logo_slots and r2_storage.r2_available():
                valid_logos: list[tuple[str, dict, str]] = []
                for slot, logo_cfg in logo_slots.items():
                    r2_key = (logo_cfg or {}).get("r2_key", "")
                    if not r2_key:
                        continue
                    try:
                        logo_data = r2_storage.download_bytes(r2_key)
                        logo_path = work_dir / f"logo_{slot}.png"
                        logo_path.write_bytes(logo_data)
                        valid_logos.append((slot, logo_cfg, str(logo_path)))
                    except Exception:
                        logger.warning("AI postprod: could not download logo for slot %s", slot)

                if valid_logos:
                    logo_out = work_dir / "with_logos.mp4"
                    cmd_parts = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", str(current_video)]
                    for _slot, _cfg, logo_path in valid_logos:
                        cmd_parts += ["-i", logo_path]

                    _corner_x = {
                        "top-left":     "20",
                        "top-right":    "main_w-overlay_w-20",
                        "bottom-left":  "20",
                        "bottom-right": "main_w-overlay_w-20",
                    }
                    _corner_y = {
                        "top-left":     "20",
                        "top-right":    "20",
                        "bottom-left":  "main_h-overlay_h-20",
                        "bottom-right": "main_h-overlay_h-20",
                    }
                    overlay_filter = ""
                    prev = "[0:v]"
                    for li, (_slot, logo_cfg, _lp) in enumerate(valid_logos):
                        opacity  = float((logo_cfg or {}).get("opacity", 0.9))
                        width    = int((logo_cfg or {}).get("width", 120))
                        height   = int((logo_cfg or {}).get("height", -1))
                        x_offset = int((logo_cfg or {}).get("x_offset", 0))
                        y_offset = int((logo_cfg or {}).get("y_offset", 0))
                        cx       = _corner_x.get(_slot, "20")
                        cy       = _corner_y.get(_slot, "20")
                        ox = f"({cx})+({x_offset})" if x_offset else cx
                        oy = f"({cy})+({y_offset})" if y_offset else cy
                        lbl_in  = f"[{li+1}:v]"
                        lbl_out = f"[ov{li}]" if li < len(valid_logos) - 1 else "[vfinal]"
                        overlay_filter += (
                            f"{lbl_in}scale={width}:{height}[ls{li}];"
                            f"[ls{li}]format=rgba,colorchannelmixer=aa={opacity:.2f}[la{li}];"
                            f"{prev}[la{li}]overlay={ox}:{oy}{lbl_out};"
                        )
                        prev = lbl_out
                    overlay_filter = overlay_filter.rstrip(";")
                    cmd_parts += [
                        "-filter_complex", overlay_filter,
                        "-map", "[vfinal]", "-map", "0:a?",
                        "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(logo_out),
                    ]
                    try:
                        subprocess.run(cmd_parts, check=True, capture_output=True)
                        current_video = logo_out
                    except subprocess.CalledProcessError as e:
                        logger.warning("AI postprod: logo overlay failed (%s)",
                                       (e.stderr or b"").decode("utf-8", "ignore")[:300])

            # ── Subtitle burn-in ───────────────────────────────────────────────
            if srt_r2_key and r2_storage.r2_available():
                try:
                    srt_data = r2_storage.download_bytes(srt_r2_key)
                    sub_style = settings.get("subtitle_style") or {}
                    animation = sub_style.get("animation", "none")

                    if animation == "karaoke":
                        ass_content = _srt_to_ass_karaoke(srt_data, sub_style)
                        sub_path = work_dir / "subtitles.ass"
                        sub_path.write_text(ass_content, encoding="utf-8")
                        sub_filter = f"ass={sub_path.as_posix()}"
                    else:
                        srt_path = work_dir / "subtitles.srt"
                        srt_path.write_bytes(srt_data)
                        font_name  = sub_style.get("font", "Arial")
                        font_size  = int(sub_style.get("font_size", 24))
                        primary_c  = sub_style.get("primary_colour", "&H00FFFFFF")
                        outline_c  = sub_style.get("outline_colour", "&H00000000")
                        back_c     = sub_style.get("back_colour", "&H80000000")
                        bold       = int(sub_style.get("bold", 0))
                        outline    = float(sub_style.get("outline", 1.5))
                        shadow     = float(sub_style.get("shadow", 0))
                        alignment  = int(sub_style.get("alignment", 2))
                        margin_v   = int(sub_style.get("margin_v", 40))
                        margin_l   = int(sub_style.get("margin_l", 10))
                        margin_r   = int(sub_style.get("margin_r", 10))
                        force_style = (
                            f"FontName={font_name},FontSize={font_size},"
                            f"PrimaryColour={primary_c},OutlineColour={outline_c},"
                            f"BackColour={back_c},Bold={bold},"
                            f"Outline={outline},Shadow={shadow},"
                            f"Alignment={alignment},"
                            f"MarginL={margin_l},MarginR={margin_r},MarginV={margin_v}"
                        )
                        sub_filter = (
                            f"subtitles={srt_path.as_posix()}:force_style='{force_style}'"
                        )

                    subtitled = work_dir / "subtitled.mp4"
                    cmd = [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(current_video),
                        "-vf", sub_filter,
                        "-c:v", "libx264", "-preset", enc_preset, "-crf", str(crf),
                        "-c:a", "copy", str(subtitled),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    current_video = subtitled
                except Exception:
                    logger.warning("AI postprod: subtitle burn-in failed (non-fatal)", exc_info=True)

            # ── Upload to R2 ───────────────────────────────────────────────────
            safe_name = (row.get("name") or "qaivid").strip().replace(" ", "_")
            r2_key = f"projects/{project_id}/quick/{safe_name}_ai_export.mp4"
            export_url = r2_storage.upload_file(current_video, r2_key)

            with _db() as conn, conn.cursor() as cur:
                # Read-modify-write to store export_url inside the ai sub-config
                cur.execute("SELECT postprod_config FROM projects WHERE id=%s FOR UPDATE", (project_id,))
                _row = cur.fetchone()
                _cfg = (_row[0] if _row and _row[0] else {}) if _row else {}
                if isinstance(_cfg, str):
                    _cfg = _json.loads(_cfg)
                _cfg["ai_generating"] = False
                _cfg["ai_error"] = None
                if "ai" not in _cfg or not isinstance(_cfg["ai"], dict):
                    _cfg["ai"] = {}
                _cfg["ai"]["export_url"] = export_url
                cur.execute(
                    "UPDATE projects SET postprod_config=%s, updated_at=NOW() WHERE id=%s",
                    (_json.dumps(_cfg), project_id),
                )
                conn.commit()

            logger.info("AI postprod export done for project %s → %s", project_id, export_url)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "ignore")[:800]
        logger.exception("AI postprod ffmpeg error for project %s: %s", project_id, stderr)
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET postprod_config=postprod_config || %s::jsonb, updated_at=NOW() "
                "WHERE id=%s",
                (_json.dumps({"ai_generating": False, "ai_error": f"ffmpeg failed: {stderr}"}), project_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("AI postprod export failed for project %s", project_id)
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET postprod_config=postprod_config || %s::jsonb, updated_at=NOW() "
                "WHERE id=%s",
                (_json.dumps({"ai_generating": False, "ai_error": str(exc)[:400]}), project_id),
            )
            conn.commit()


def kick_ai_postprod(project_id: str, settings: dict) -> None:
    """Queue an async AI-video post-production export job."""
    import json as _json
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET "
            "postprod_config=COALESCE(postprod_config,'{}')::jsonb || %s::jsonb, "
            "updated_at=NOW() WHERE id=%s",
            (_json.dumps({"ai_generating": True, "ai_error": None}), project_id),
        )
        conn.commit()
    _POSTPROD_EXECUTOR.submit(_assemble_ai_postprod_job, project_id, settings)
