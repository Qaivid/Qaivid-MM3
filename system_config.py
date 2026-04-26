"""Lightweight key/value settings backed by the existing PostgreSQL database.

Two independent image-generation slots:

  * ``image_mode_ref``  — character / location reference plates
  * ``image_mode_shot`` — per-shot stills

GPT Image 2.0 tiers (native 1920×1080 landscape):
  * ``gpt_low``    — $0.01/image  — fast testing
  * ``gpt_medium`` — $0.04/image  — recommended
  * ``gpt_high``   — $0.16/image  — highest fidelity

FAL tiers:
  * ``standard`` — FAL FLUX/schnell (~$0.003–0.005), no face-lock.
                   Cinematic-quality stills. Best for landscape/nature/abstract.
  * ``quality``  — FAL FLUX/dev + PuLID (~$0.025–0.05), true hard face-lock.
                   Required for character-driven narrative music videos.

Legacy aliases:
  * ``cheap``    → ``gpt_low``    (gpt-image-1.5 low; now routes to GPT Image 2.0 low)
  * ``sdxl_face``→ ``standard``   (ip-adapter removed from fal.ai)

A short in-process cache prevents hammering the DB on every render step.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

VALID_MODES = ("quality", "standard", "gpt_low", "gpt_medium", "gpt_high")
# Legacy aliases — normalised at read time
# cheap → gpt_low  (old gpt-image-1.5 "low" path, now GPT Image 2.0 low)
# sdxl_face → standard  (ip-adapter-face-id-plus removed from fal.ai)
_LEGACY_ALIASES = {"sdxl_face": "standard", "cheap": "gpt_low"}
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
