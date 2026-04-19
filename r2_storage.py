"""Cloudflare R2 storage layer for Qaivid MetaMind.

All generated and uploaded assets are stored in R2.
The public URL (R2_PUBLIC_URL) is used for serving assets to browsers and
passing image URLs to FAL AI models (no more REPLIT_DEV_DOMAIN hack).

Environment secrets required:
  R2_ACCOUNT_ID        — Cloudflare account ID
  R2_ACCESS_KEY_ID     — R2 API token access key
  R2_SECRET_ACCESS_KEY — R2 API token secret key
  R2_BUCKET_NAME       — bucket name
  R2_PUBLIC_URL        — public base URL, e.g. https://pub-xxxx.r2.dev
"""
from __future__ import annotations

import io
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    _client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    return _client


def _bucket() -> str:
    return os.environ["R2_BUCKET_NAME"]


def _public_url() -> str:
    return os.environ["R2_PUBLIC_URL"].rstrip("/")


def upload_file(local_path: Path, r2_key: str) -> str:
    """Upload a local file to R2 and return its public URL."""
    mime, _ = mimetypes.guess_type(str(local_path))
    mime = mime or "application/octet-stream"
    client = _get_client()
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=_bucket(),
            Key=r2_key,
            Body=f,
            ContentType=mime,
        )
    url = f"{_public_url()}/{r2_key}"
    logger.info("Uploaded %s → %s", local_path, url)
    return url


def upload_bytes(data: bytes, r2_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to R2 and return its public URL."""
    _get_client().put_object(
        Bucket=_bucket(),
        Key=r2_key,
        Body=data,
        ContentType=content_type,
    )
    url = f"{_public_url()}/{r2_key}"
    logger.info("Uploaded bytes → %s", url)
    return url


def upload_from_url(remote_url: str, r2_key: str, content_type: str | None = None,
                    stream: bool = False) -> str:
    """Download from *remote_url* and stream-upload to R2. Returns public URL."""
    resp = requests.get(remote_url, timeout=300, stream=True)
    resp.raise_for_status()
    mime = content_type or resp.headers.get("content-type", "application/octet-stream").split(";")[0]

    if stream:
        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buf.write(chunk)
        buf.seek(0)
        _get_client().upload_fileobj(buf, _bucket(), r2_key, ExtraArgs={"ContentType": mime})
    else:
        _get_client().put_object(
            Bucket=_bucket(),
            Key=r2_key,
            Body=resp.content,
            ContentType=mime,
        )
    url = f"{_public_url()}/{r2_key}"
    logger.info("Streamed %s → %s", remote_url, url)
    return url


def upload_fileobj(fileobj: BinaryIO, r2_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload a file-like object to R2 and return its public URL."""
    _get_client().upload_fileobj(
        fileobj, _bucket(), r2_key,
        ExtraArgs={"ContentType": content_type},
    )
    url = f"{_public_url()}/{r2_key}"
    logger.info("Uploaded fileobj → %s", url)
    return url


def delete_prefix(prefix: str) -> int:
    """Delete all R2 objects under *prefix*. Returns count deleted."""
    client = _get_client()
    bucket = _bucket()
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
        )
        deleted += len(objects)
    logger.info("Deleted %d objects under prefix %s", deleted, prefix)
    return deleted


def public_url_for(r2_key: str) -> str:
    """Return the public URL for a given R2 key (no network call)."""
    return f"{_public_url()}/{r2_key}"


def presigned_url(r2_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL valid for *expires_in* seconds."""
    return _get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": r2_key},
        ExpiresIn=expires_in,
    )


def download_bytes(r2_key: str) -> bytes:
    """Download an object from R2 and return its raw bytes."""
    buf = io.BytesIO()
    _get_client().download_fileobj(_bucket(), r2_key, buf)
    return buf.getvalue()


def r2_available() -> bool:
    """Return True if all R2 env vars are set."""
    return all(os.getenv(k) for k in [
        "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_PUBLIC_URL",
    ])
