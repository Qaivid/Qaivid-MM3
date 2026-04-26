"""Lightweight key/value settings backed by the existing PostgreSQL database.

Three image generation modes, two independent slots:

  * ``image_mode_ref``  — character / location reference plates
  * ``image_mode_shot`` — per-shot stills

Shot mode values:
  * ``cheap``    — gpt-image-1 low ($0.009–$0.013), img2img with char+env refs.
                   Good for fast prompt testing with face retention via img2img.
  * ``standard`` — FAL FLUX/schnell (~$0.003–0.005), no face-lock.
                   Cinematic-quality stills, same engine as reference plates.
                   Best for nature, landscape, abstract, or wide/drone shots
                   where character face consistency is less critical.
  * ``quality``  — FAL FLUX/dev + PuLID (~$0.025–0.05), true hard face-lock.
                   Identity embeddings injected from the character plate —
                   same face across every shot. Required for character-driven
                   narrative music videos.

  Deprecated alias: ``sdxl_face`` → normalised to ``standard``
  (fal-ai/ip-adapter-face-id-plus was removed from fal.ai).

Ref mode values:
  * ``cheap``   — gpt-image-1 low ($0.009–$0.013)
  * ``quality`` — FAL FLUX/schnell (~$0.003)  ← recommended

A short in-process cache prevents hammering the DB on every render step.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

VALID_MODES = ("quality", "standard", "cheap")
# sdxl_face is a legacy alias — normalised to standard at read time
_LEGACY_ALIASES = {"sdxl_face": "standard"}
DEFAULT_MODE = "quality"

KEY_REF = "image_mode_ref"
KEY_SHOT = "image_mode_shot"

_CACHE_TTL_SEC = 5.0
_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def _db():
    """Lazy import of the worker's connection helper to avoid an import cycle
    (pipeline_worker imports many heavy deps at module load)."""
    from pipeline_worker import _db as _open
    return _open()


def ensure_schema() -> None:
    """Idempotent. Called from pipeline_worker.ensure_schema()."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS system_config (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )


def _normalize(value: Optional[str]) -> str:
    v = (value or "").strip().lower()
    v = _LEGACY_ALIASES.get(v, v)   # resolve legacy aliases first
    return v if v in VALID_MODES else DEFAULT_MODE


def get_setting(key: str, default: str = DEFAULT_MODE) -> str:
    """Read a setting with a 5-second in-process cache."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and (now - cached[0]) < _CACHE_TTL_SEC:
            return cached[1]
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = %s", (key,))
            row = cur.fetchone()
            value = row["value"] if row else default
    except Exception as exc:
        logger.warning("system_config read failed for %s: %s — using default", key, exc)
        value = default
    value = _normalize(value)
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def set_setting(key: str, value: str) -> str:
    """Persist a setting and return the normalized value actually stored."""
    norm = _normalize(value)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO system_config (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, norm),
        )
    with _cache_lock:
        _cache[key] = (time.monotonic(), norm)
    return norm


def get_image_modes() -> dict[str, str]:
    """Return both image modes in one shot, applying env overrides if set.

    Environment variables ``QAIVID_IMAGE_MODE_REF`` and
    ``QAIVID_IMAGE_MODE_SHOT`` take precedence over the DB value — useful for
    a developer running locally without touching shared admin state."""
    ref = os.environ.get("QAIVID_IMAGE_MODE_REF") or get_setting(KEY_REF)
    shot = os.environ.get("QAIVID_IMAGE_MODE_SHOT") or get_setting(KEY_SHOT)
    return {"ref": _normalize(ref), "shot": _normalize(shot)}


def invalidate_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ── Generic raw helpers (no normalization) ────────────────────────────────

def get_raw(key: str, default: str = "") -> str:
    """Read any key without normalization."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and (now - cached[0]) < _CACHE_TTL_SEC:
            return cached[1]
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = %s", (key,))
            row = cur.fetchone()
            value = row["value"] if row else default
    except Exception as exc:
        logger.warning("system_config get_raw failed for %s: %s", key, exc)
        value = default
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def set_raw(key: str, value: str) -> None:
    """Persist any setting without normalization."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO system_config (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, value),
        )
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)


def get_all_site_settings() -> dict:
    """Return all site_* keys as a plain dict."""
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT key, value FROM system_config WHERE key LIKE 'site_%'")
            return {r["key"]: r["value"] for r in cur.fetchall()}
    except Exception as exc:
        logger.warning("get_all_site_settings failed: %s", exc)
        return {}
