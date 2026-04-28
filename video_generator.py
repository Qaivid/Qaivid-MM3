"""AtlasCloud WAN 2.6 Flash video generation for Qaivid MetaMind.

Animates each shot still into a short video clip using WAN 2.6 Flash image-to-video.
Flash is the speed-optimised, cost-effective tier of WAN 2.6 — same API shape as
the standard model; switch VIDEO_MODEL to alibaba/wan-2.6/image-to-video to go back.
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
VIDEO_MODEL             = "alibaba/wan-2.6/image-to-video-flash"
RESOLUTION              = "720p"          # Flash sweet spot; change to "1080p" for standard
MIN_DURATION            = 2               # seconds (WAN 2.6 minimum)
DEFAULT_DURATION        = 5               # seconds — fallback when no duration hint
MAX_DURATION            = 15              # seconds (WAN 2.6 maximum)
MOTION_PROMPT_MAX_CHARS = 2000            # WAN 2.6 I2V limit; change here when switching models
POLL_INTERVAL_S         = 4
VIDEO_TIMEOUT_S         = 360             # 6 min — Flash is faster than standard


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
    AtlasCloud can fetch it.  Falls back to the original URL on any error."""
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
    """Return a URL AtlasCloud can fetch.

    Tries the URL anonymously first.  If it returns 4xx, generates a
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
    """POST a generation job and return the prediction ID.

    AtlasCloud uses a flat payload — NOT a nested "input" object.
    The image field is "image", not "image_url".
    """
    payload = {
        "model":                    VIDEO_MODEL,
        "image":                    image_url,    # flat, not input.image_url
        "prompt":                   prompt,
        "resolution":               RESOLUTION,
        "duration":                 duration,
        "shot_type":                "single",     # one camera per clip
        "enable_prompt_expansion":  False,        # use our precise prompts as-is
        "generate_audio":           False,        # audio assembled separately in post
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
    wan_video_prompt: str | None = None,
) -> str:
    """Animate *still_url* into a short video clip stored in R2.

    Args:
        still_url:        Public (or R2) URL of the still image.
        prompt:           Visual prompt for the shot.
        duration_s:       Shot duration hint (clips are capped at MAX_DURATION).
        motion_prompt:    Full WAN-optimised motion description (rich, for Frame 0
                          derivation). Used as fallback when wan_video_prompt is absent.
        wan_video_prompt: Lean start-frame continuation prompt (Camera/Action/Lighting/
                          Mood/Micro-detail/Avoid format, ≤600 chars). When present,
                          preferred over motion_prompt for WAN submission because it
                          omits scene re-description (the still already shows the scene).

    Returns a public R2 URL string.
    Raises VideoGenerationError on failure.
    """
    # Use the shot's actual duration, clamped to WAN 2.6's accepted range [2, 15].
    # Round to nearest integer — AtlasCloud takes whole seconds.
    if duration_s is not None and duration_s > 0:
        clip_duration = max(MIN_DURATION, min(MAX_DURATION, round(duration_s)))
    else:
        clip_duration = DEFAULT_DURATION

    # Prefer the lean WAN continuation prompt; fall back to full motion_prompt,
    # then to the visual prompt as a last resort.
    effective_prompt = (wan_video_prompt or "").strip()
    if not effective_prompt:
        effective_prompt = (motion_prompt or "").strip()
    if not effective_prompt:
        effective_prompt = (prompt or "cinematic camera motion, smooth dolly").strip()
    effective_prompt = effective_prompt[:MOTION_PROMPT_MAX_CHARS]

    accessible_image_url = _accessible_url(still_url)

    logger.info(
        "Generating WAN 2.6 Flash video — project=%s shot=%s dur=%ss",
        project_id, shot_index, clip_duration,
    )

    prediction_id = _submit_job(accessible_image_url, effective_prompt, clip_duration)
    video_url     = _poll_job(prediction_id)

    r2_key     = _r2_key(project_id, shot_index)
    public_url = r2_storage.upload_from_url(
        video_url, r2_key, content_type="video/mp4", stream=True
    )
    logger.info("Saved WAN 2.6 Flash clip to R2: %s", public_url)
    return public_url
