"""AtlasCloud WAN 2.6 video generation for Qaivid MetaMind.

Animates each shot still into a short video clip using WAN 2.6 image-to-video.
Video clips are stored in Cloudflare R2 (r2_storage.py).
Returns public R2 URLs.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

import boto3
import requests

import r2_storage

logger = logging.getLogger(__name__)

ATLAS_BASE_URL          = "https://api.atlascloud.ai/api/v1/model"
VIDEO_MODEL             = "alibaba/wan-2.6/image-to-video"
ASPECT_RATIO            = "16:9"
DEFAULT_DURATION        = 5       # seconds
MAX_DURATION            = 10      # seconds (AtlasCloud I2V cap)
MOTION_PROMPT_MAX_CHARS = 400     # WAN 2.6 sweet spot — change here when switching models
FPS              = 24
RESOLUTION       = "1080p"
POLL_INTERVAL_S  = 5
VIDEO_TIMEOUT_S  = 480     # 8 min — WAN 2.6 is slower than Kling


class VideoGenerationError(RuntimeError):
    pass


# ── Auth ──────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.getenv("ATLAS_CLOUD_API_KEY") or os.getenv("ATLASCLOUD_API_KEY")
    if not key:
        raise VideoGenerationError(
            "ATLAS_CLOUD_API_KEY is not set in secrets."
        )
    return key


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key()}",
    }


# ── Image accessibility ───────────────────────────────────────────────────────

def _presigned_url(r2_url: str, expires: int = 600) -> str:
    """Generate a time-limited presigned URL for a private R2 object so that
    AtlasCloud can fetch it.  Falls back to returning the original URL if
    credentials are missing (public bucket case)."""
    try:
        from urllib.parse import urlparse
        parsed   = urlparse(r2_url)
        raw_path = parsed.path.lstrip("/")
        bucket   = os.getenv("R2_BUCKET_NAME", "")
        if bucket and raw_path.startswith(bucket + "/"):
            r2_key = raw_path[len(bucket) + 1:]
        else:
            r2_key = raw_path

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
        logger.warning("Could not generate presigned URL (%s); using original", exc)
        return r2_url


def _accessible_url(r2_url: str) -> str:
    """Return a URL that AtlasCloud can fetch.

    Tries the URL anonymously first.  If it returns 4xx, generate a
    presigned R2 URL valid for 10 minutes.
    """
    try:
        resp = requests.head(r2_url, timeout=6)
        if resp.status_code < 400:
            return r2_url
    except Exception:
        pass
    logger.info("R2 URL not publicly accessible — generating presigned URL")
    return _presigned_url(r2_url)


# ── R2 storage ────────────────────────────────────────────────────────────────

def _r2_key(project_id: str, shot_index: int) -> str:
    return f"projects/{project_id}/videos/shot_{shot_index}_{uuid.uuid4().hex[:6]}.mp4"


# ── AtlasCloud API ────────────────────────────────────────────────────────────

def _submit_job(image_url: str, prompt: str, duration: int) -> str:
    """POST a generation job and return the prediction ID."""
    payload = {
        "model": VIDEO_MODEL,
        "input": {
            "prompt": prompt,
            "image_url": image_url,
            "resolution": RESOLUTION,
            "duration": duration,
            "fps": FPS,
            "aspect_ratio": ASPECT_RATIO,
        },
    }
    resp = requests.post(
        f"{ATLAS_BASE_URL}/generateVideo",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise VideoGenerationError(
            f"AtlasCloud submit failed ({resp.status_code}): {resp.text[:400]}"
        ) from exc

    data = resp.json().get("data", {})
    prediction_id = data.get("id")
    if not prediction_id:
        raise VideoGenerationError(
            f"AtlasCloud did not return a prediction ID: {resp.text[:400]}"
        )
    logger.info("AtlasCloud job submitted — prediction_id=%s", prediction_id)
    return prediction_id


def _poll_job(prediction_id: str, timeout_s: int = VIDEO_TIMEOUT_S) -> str:
    """Poll until the job is done and return the video URL."""
    poll_url = f"{ATLAS_BASE_URL}/prediction/{prediction_id}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = requests.get(poll_url, headers=_headers(), timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise VideoGenerationError(
                f"AtlasCloud poll failed ({resp.status_code}): {resp.text[:400]}"
            ) from exc

        data   = resp.json().get("data", {})
        status = data.get("status", "")
        logger.debug("AtlasCloud poll — id=%s status=%s", prediction_id, status)

        if status in ("completed", "succeeded"):
            outputs = data.get("outputs") or []
            if outputs:
                return outputs[0]
            raise VideoGenerationError(
                f"AtlasCloud job succeeded but outputs list is empty: {data}"
            )

        if status in ("failed", "error", "cancelled"):
            raise VideoGenerationError(
                f"AtlasCloud job {status}: {data.get('error') or data}"
            )

        time.sleep(POLL_INTERVAL_S)

    raise VideoGenerationError(
        f"AtlasCloud video timed out after {timeout_s}s (prediction_id={prediction_id})"
    )


# ── Public interface ──────────────────────────────────────────────────────────

def generate_shot_video(
    project_id: str,
    shot_index: int,
    still_url: str,
    prompt: str,
    duration_s: float | None = None,
    motion_prompt: str | None = None,
) -> str:
    """Animate *still_url* into a short video clip stored in R2.

    Args:
        still_url:     Public (or R2) URL of the still image.
        prompt:        Visual prompt for the shot.
        duration_s:    Shot duration hint (clips are capped at MAX_DURATION).
        motion_prompt: Concise WAN-optimised camera/motion description.
                       When provided, used as the video prompt instead of the
                       full visual prompt (which is often too long).

    Returns a public R2 URL string.
    Raises VideoGenerationError on failure.
    """
    # Determine clip duration
    if duration_s is not None and duration_s >= 8:
        clip_duration = MAX_DURATION
    else:
        clip_duration = DEFAULT_DURATION

    # Choose prompt — motion_prompt is concise and WAN-optimised
    effective_prompt = (motion_prompt or "").strip()
    if not effective_prompt:
        effective_prompt = (prompt or "cinematic camera motion, smooth dolly").strip()
    # WAN 2.6 handles longer prompts well, but trim to safe limit
    effective_prompt = effective_prompt[:400]

    # Ensure the image URL is reachable by AtlasCloud
    accessible_image_url = _accessible_url(still_url)

    logger.info(
        "Generating WAN 2.6 video — project=%s shot=%s dur=%ss",
        project_id, shot_index, clip_duration,
    )

    prediction_id = _submit_job(accessible_image_url, effective_prompt, clip_duration)
    video_url     = _poll_job(prediction_id)

    # Download from AtlasCloud CDN and persist to our R2 bucket
    r2_key     = _r2_key(project_id, shot_index)
    public_url = r2_storage.upload_from_url(
        video_url, r2_key, content_type="video/mp4", stream=True
    )
    logger.info("Saved WAN 2.6 clip to R2: %s", public_url)
    return public_url
