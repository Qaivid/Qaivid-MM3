"""Motif Materializer (MetaMind v3.1) — reads a context_packet and writes
concrete motif rows to the `motifs` table.

Idempotent: rows are matched by (project_id, name). Existing rows have their
NULL fields back-filled; non-NULL fields are left untouched so user edits
survive a pipeline re-run.

Sources:
  - `motif_map` dict   → PRIMARY: per-motif type / significance / visual_form.
  - `motifs` list      → name-only fallback for any motifs not present in motif_map.
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


def _upsert_motif(cur, project_id: str, motif: dict) -> int:
    """Atomic UPSERT (idempotent under concurrency); returns the row id."""
    cur.execute(
        """
        INSERT INTO motifs
            (project_id, name, motif_type, significance, visual_form, metadata)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (project_id, name) DO UPDATE
           SET motif_type   = COALESCE(motifs.motif_type,   EXCLUDED.motif_type),
               significance = COALESCE(motifs.significance, EXCLUDED.significance),
               visual_form  = COALESCE(motifs.visual_form,  EXCLUDED.visual_form)
        RETURNING id
        """,
        (
            project_id,
            motif["name"],
            motif.get("motif_type"),
            motif.get("significance"),
            motif.get("visual_form"),
            motif.get("metadata"),
        ),
    )
    return cur.fetchone()["id"]


def materialize_motifs(project_id: str, context_packet: dict) -> list[dict]:
    """Derive motif rows from context_packet and persist them."""
    if not isinstance(context_packet, dict):
        context_packet = {}
    raw_mm = context_packet.get("motif_map")
    motif_map = raw_mm if isinstance(raw_mm, dict) else {}
    raw_list = context_packet.get("motifs")
    motifs_list = raw_list if isinstance(raw_list, list) else []

    motifs: list[dict] = []
    seen: set[str] = set()

    if isinstance(motif_map, dict):
        for raw_name, payload in motif_map.items():
            name = _norm(raw_name)
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            if isinstance(payload, dict):
                motif_type = _norm(payload.get("type"))
                significance = _norm(payload.get("significance"))
                visual_form = _norm(payload.get("visual_form"))
            else:
                motif_type = ""
                significance = ""
                visual_form = _norm(payload)

            motifs.append({
                "name": name,
                "motif_type": motif_type or None,
                "significance": significance or None,
                "visual_form": visual_form or None,
                "metadata": Json({
                    "source": "motif_map",
                    "raw_payload": payload if isinstance(payload, dict) else {"value": payload},
                }),
            })

    if isinstance(motifs_list, list):
        for raw_name in motifs_list:
            name = _norm(raw_name)
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            motifs.append({
                "name": name,
                "motif_type": None,
                "significance": None,
                "visual_form": None,
                "metadata": Json({"source": "motifs_list"}),
            })

    if not motifs:
        logger.info("No motifs extracted for project %s", project_id)
        return []

    with _db() as conn, conn.cursor() as cur:
        for motif in motifs:
            _upsert_motif(cur, project_id, motif)
        conn.commit()
    logger.info("Upserted %d motif(s) for project %s", len(motifs), project_id)
    return motifs
