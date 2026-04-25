"""Runware FLUX.1 Fill Dev outpainting for Qaivid MetaMind.

Extends shot stills to true 16:9 (1920×1080) or 9:16 (1080×1920) using
AI-powered content-aware fill — ~$0.006 per image.

API: https://api.runware.ai/v1
Model: runware:102@1  (FLUX.1 Fill Dev — open-source inpaint/outpaint)
"""
from __future__ import annotations

import io
import logging
import os
import struct
import time
import uuid
from typing import Optional

import boto3
import requests
from PIL import Image

import r2_storage

logger = logging.getLogger(__name__)

RUNWARE_API_URL = "https://api.runware.ai/v1"
OUTPAINT_MODEL  = "runware:102@1"   # FLUX.1 Fill Dev

TARGET_RESOLUTIONS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1":  (1024, 1024),
}

OUTPAINT_STEPS   = 30
REQUEST_TIMEOUT  = 60
POLL_TIMEOUT_S   = 120
POLL_INTERVAL_S  = 2


class OutpaintError(RuntimeError):
    pass


# ── Auth ─────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.getenv("RUNWARE_API_KEY")
    if not key:
        raise OutpaintError("RUNWARE_API_KEY is not set in secrets.")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


# ── Runware REST helper ───────────────────────────────────────────────────────

def _runware_post(tasks: list[dict]) -> list[dict]:
    resp = requests.post(
        RUNWARE_API_URL,
        headers=_headers(),
        json=tasks,
        timeout=REQUEST_TIMEOUT,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise OutpaintError(
            f"Runware API error ({resp.status_code}): {resp.text[:400]}"
        ) from exc
    body = resp.json()
    if body.get("errors"):
        raise OutpaintError(f"Runware API errors: {body['errors']}")
    return body.get("data") or []


# ── Image dimension detection ─────────────────────────────────────────────────

def _get_image_dimensions_from_url(url: str) -> tuple[int, int]:
    """Download just enough bytes to read width/height from a PNG or JPEG."""
    try:
        # PNG: width at bytes 16-19, height at 20-23
        r = requests.get(url, headers={"Range": "bytes=0-23"}, timeout=15)
        data = r.content
        if len(data) >= 24 and data[:4] == b'\x89PNG':
            w = struct.unpack(">I", data[16:20])[0]
            h = struct.unpack(">I", data[20:24])[0]
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass

    # Fallback: download full image and detect
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = r.content

    # JPEG: scan for SOF marker
    if raw[:2] == b'\xff\xd8':
        i = 2
        while i < len(raw) - 8:
            if raw[i] != 0xFF:
                break
            marker = raw[i + 1]
            length = struct.unpack(">H", raw[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):
                h = struct.unpack(">H", raw[i + 5:i + 7])[0]
                w = struct.unpack(">H", raw[i + 7:i + 9])[0]
                return w, h
            i += 2 + length

    # WebP
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        if raw[12:16] == b'VP8 ':
            w = struct.unpack("<H", raw[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", raw[28:30])[0] & 0x3FFF
            return w, h

    raise OutpaintError(f"Could not determine image dimensions from URL: {url[:80]}")


# ── Outpaint geometry ─────────────────────────────────────────────────────────

def _compute_extension(
    orig_w: int, orig_h: int, aspect: str
) -> Optional[tuple[int, int, int, int, int, int]]:
    """Return (left, right, top, bottom, final_w, final_h) or None if no extension needed."""

    def round64(n: int) -> int:
        return round(n / 64) * 64

    target_w, target_h = TARGET_RESOLUTIONS.get(aspect, (1920, 1080))

    if aspect == "16:9":
        final_w = round64(int(orig_h * 16 / 9))
        total_ext = final_w - orig_w
        if total_ext <= 0:
            return None
        half = total_ext // 2
        return half, total_ext - half, 0, 0, final_w, orig_h

    if aspect == "9:16":
        final_h = round64(int(orig_w * 16 / 9))
        total_ext = final_h - orig_h
        if total_ext <= 0:
            return None
        half = total_ext // 2
        return 0, 0, half, total_ext - half, orig_w, final_h

    if aspect == "1:1":
        return None

    return None


# ── Presigned URL for private R2 objects ──────────────────────────────────────

def _presigned_url(r2_url: str, expires: int = 600) -> str:
    try:
        from urllib.parse import urlparse
        parsed  = urlparse(r2_url)
        raw_key = parsed.path.lstrip("/")
        bucket  = os.getenv("R2_BUCKET_NAME", "")
        r2_key  = raw_key[len(bucket) + 1:] if bucket and raw_key.startswith(bucket + "/") else raw_key
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": r2_key},
            ExpiresIn=expires,
        )
    except Exception as exc:
        logger.warning("Presigned URL failed (%s); using original", exc)
        return r2_url


def _accessible_url(url: str) -> str:
    try:
        r = requests.head(url, timeout=6)
        if r.status_code < 400:
            return url
    except Exception:
        pass
    return _presigned_url(url)


# ── Runware upload ────────────────────────────────────────────────────────────

def _upload_to_runware(image_url: str) -> str:
    """Upload a source image to Runware and return its imageUUID."""
    task_uuid = str(uuid.uuid4())
    results = _runware_post([{
        "taskType":  "imageUpload",
        "taskUUID":  task_uuid,
        "image":     image_url,
    }])
    result = next((r for r in results if r.get("taskUUID") == task_uuid), None) or (results[0] if results else {})
    image_uuid = result.get("imageUUID")
    if not image_uuid:
        raise OutpaintError(f"Runware imageUpload returned no imageUUID: {results}")
    logger.info("Runware imageUpload OK — imageUUID=%s", image_uuid)
    return image_uuid


# ── R2 key ────────────────────────────────────────────────────────────────────

def _r2_key(project_id: str, shot_index: int) -> str:
    return f"projects/{project_id}/stills/outpainted_{shot_index}_{uuid.uuid4().hex[:6]}.webp"


# ── Public interface ──────────────────────────────────────────────────────────

def outpaint_shot_still(
    project_id: str,
    shot_index: int,
    still_url: str,
    prompt: str,
    aspect_ratio: str = "16:9",
) -> str:
    """Outpaint *still_url* to the target aspect ratio using Runware FLUX Fill Dev.

    Args:
        still_url:    Public R2 URL of the generated still (1:1 or landscape).
        prompt:       Visual prompt describing the scene (guides content fill).
        aspect_ratio: "16:9" or "9:16" (default "16:9").

    Returns a new public R2 URL of the outpainted image (JPEG, true 16:9 or 9:16).
    Raises OutpaintError on failure.
    """
    logger.info(
        "Outpainting shot %s for project %s → %s",
        shot_index, project_id, aspect_ratio,
    )

    accessible = _accessible_url(still_url)

    # Detect dimensions
    orig_w, orig_h = _get_image_dimensions_from_url(accessible)
    logger.info("Source dimensions: %dx%d", orig_w, orig_h)

    target_w, target_h = TARGET_RESOLUTIONS.get(aspect_ratio, (1920, 1080))

    # Check if the image already IS the target size
    if orig_w == target_w and orig_h == target_h:
        logger.info("Image already at target resolution — skipping outpaint")
        return still_url

    ext = _compute_extension(orig_w, orig_h, aspect_ratio)
    if ext is None:
        logger.info("No extension needed (image is wider/taller than target) — returning original")
        return still_url

    left, right, top, bottom, final_w, final_h = ext
    logger.info(
        "Extension: left=%d right=%d top=%d bottom=%d → %dx%d",
        left, right, top, bottom, final_w, final_h,
    )

    # Upload source image to Runware
    image_uuid = _upload_to_runware(accessible)

    # Submit outpaint inference
    task_uuid = str(uuid.uuid4())
    task = {
        "taskType":       "imageInference",
        "taskUUID":       task_uuid,
        "model":          OUTPAINT_MODEL,
        "positivePrompt": "  ",
        "seedImage":      image_uuid,
        "outpaint": {
            "left":   left,
            "right":  right,
            "top":    top,
            "bottom": bottom,
        },
        "width":         final_w,
        "height":        final_h,
        "steps":         OUTPAINT_STEPS,
        "numberResults": 1,
    }

    logger.info("Submitting Runware imageInference task %s", task_uuid)
    submit_results = _runware_post([task])

    # Check for synchronous result first
    def _extract_url(r: dict) -> Optional[str]:
        return r.get("imageURL") or r.get("imageUrl") or r.get("url")

    result_url: Optional[str] = None
    sync = next((r for r in submit_results if r.get("taskUUID") == task_uuid), None) or (submit_results[0] if submit_results else None)
    if sync:
        result_url = _extract_url(sync)
        if result_url:
            logger.info("Got synchronous result from Runware")

    # Poll if not synchronous
    if not result_url:
        deadline = time.time() + POLL_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_S)
            poll_results = _runware_post([{"taskType": "getResponse", "taskUUID": task_uuid}])
            if poll_results:
                pr = next((r for r in poll_results if r.get("taskUUID") == task_uuid), None) or poll_results[0]
                result_url = _extract_url(pr)
                if result_url:
                    logger.info("Runware poll succeeded")
                    break
                status = (pr.get("status") or "").lower()
                if status in ("failed", "error"):
                    raise OutpaintError(f"Runware outpaint failed: {pr.get('error') or pr}")

        if not result_url:
            raise OutpaintError(f"Runware outpaint timed out after {POLL_TIMEOUT_S}s")

    logger.info("Runware result URL: %s", result_url[:80])

    # Download result
    dl = requests.get(result_url, timeout=60)
    dl.raise_for_status()
    image_bytes = dl.content

    # Scale to exact canonical resolution so video generation receives a true
    # 16:9 / 9:16 frame.  The round64 geometry means Runware may return e.g.
    # 1792×1024 instead of exactly 1920×1080; this final resize corrects that.
    canonical = TARGET_RESOLUTIONS.get(aspect_ratio)
    if canonical:
        target_w, target_h = canonical
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.size != (target_w, target_h):
                logger.info(
                    "Scaling Runware output %dx%d → %dx%d",
                    img.width, img.height, target_w, target_h,
                )
                img = img.resize((target_w, target_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=90)
            image_bytes = buf.getvalue()
        except Exception as exc:
            logger.warning("Post-scale failed (%s) — using raw Runware output", exc)

    # Upload to R2
    r2_key    = _r2_key(project_id, shot_index)
    public_url = r2_storage.upload_bytes(image_bytes, r2_key, content_type="image/webp")
    logger.info("Outpainted still saved to R2: %s", public_url)
    return public_url
