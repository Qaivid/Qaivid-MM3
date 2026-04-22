"""Automatic training-dataset collector.

Every time a project finishes the full pipeline it automatically saves a
structured JSON snapshot to R2 under ``dataset/projects/<project_id>.json``.

The file is fire-and-forget — failures are logged but never raise so the
pipeline is never blocked.

R2 layout
---------
dataset/
  projects/
    <project_id>.json      ← one file per completed project (overwritten on retry)
  index.jsonl              ← append-only log; one JSON line per entry

Schema (v1)
-----------
{
  "schema_version": "1",
  "project_id": "...",
  "captured_at": "2026-04-22T12:00:00Z",
  "input": {
    "text": "raw lyrics / script",
    "genre": "song",
    "name": "project name",
    "language": "en"          # from context_packet if available
  },
  "output": {
    "summary":        { ... },
    "context_packet": { ... },
    "styled_timeline":{ ... }
  }
}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


def _upload_bytes_safe(data: bytes, r2_key: str, content_type: str) -> bool:
    """Upload to R2, returning True on success. Never raises."""
    try:
        import r2_storage
        r2_storage.upload_bytes(data, r2_key, content_type)
        return True
    except Exception as exc:
        logger.warning("Dataset upload to R2 failed (%s): %s", r2_key, exc)
        return False


def _append_to_index(entry: dict) -> None:
    """Append one line to the rolling JSONL index in R2."""
    import boto3
    from botocore.config import Config

    try:
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        bucket     = os.environ.get("R2_BUCKET_NAME", "")
        if not all([account_id, access_key, secret_key, bucket]):
            return

        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        index_key = "dataset/index.jsonl"

        existing = b""
        try:
            resp = client.get_object(Bucket=bucket, Key=index_key)
            existing = resp["Body"].read()
        except client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        updated = existing + line.encode("utf-8")
        client.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=updated,
            ContentType="application/x-ndjson",
        )
    except Exception as exc:
        logger.warning("Dataset index append failed: %s", exc)


def _collect(project_id: str, db_row: dict) -> None:
    """Build the dataset entry and push to R2. Runs in a background thread."""
    try:
        text   = db_row.get("text") or ""
        genre  = db_row.get("genre") or "unknown"
        name   = db_row.get("name") or ""

        ctx    = db_row.get("context_packet") or {}
        sty    = db_row.get("styled_timeline") or {}
        summ   = db_row.get("summary") or {}

        language = (
            ctx.get("language")
            or ctx.get("detected_language")
            or "unknown"
        )

        entry = {
            "schema_version": SCHEMA_VERSION,
            "project_id":     project_id,
            "captured_at":    datetime.now(timezone.utc).isoformat(),
            "input": {
                "text":     text,
                "genre":    genre,
                "name":     name,
                "language": language,
            },
            "output": {
                "summary":         summ,
                "context_packet":  ctx,
                "styled_timeline": sty,
            },
        }

        payload = json.dumps(entry, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        r2_key  = f"dataset/projects/{project_id}.json"

        ok = _upload_bytes_safe(payload, r2_key, "application/json")
        if ok:
            logger.info("Dataset snapshot saved for project %s", project_id)
            _append_to_index(entry)
        else:
            logger.warning("Dataset snapshot skipped for project %s (upload failed)", project_id)

    except Exception as exc:
        logger.exception("Dataset collection failed for project %s: %s", project_id, exc)


def save_project_dataset(project_id: str, db_row: dict) -> None:
    """Fire-and-forget: snapshot this project to the R2 training dataset.

    Call this once a project reaches a rich enough state (styled_timeline
    + context_packet populated).  Safe to call multiple times — each call
    overwrites the per-project JSON with the latest data.
    """
    if not os.environ.get("R2_BUCKET_NAME"):
        logger.debug("R2 not configured — dataset collection skipped")
        return
    Thread(
        target=_collect,
        args=(project_id, db_row),
        daemon=True,
        name=f"dataset-{project_id[:8]}",
    ).start()
