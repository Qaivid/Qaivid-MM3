"""FAL-backed video generation for Qaivid MetaMind.

Animates each shot still into a short video clip using Kling image-to-video.
Video clips are stored in Cloudflare R2 (r2_storage.py).
Returns public R2 URLs instead of local Paths.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

import fal_client
import requests

import r2_storage

logger = logging.getLogger(__name__)

VIDEO_MODEL = "fal-ai/kling-video/v1/standard/image-to-video"
MAX_DURATION = 5
ASPECT_RATIO = "16:9"
VIDEO_TIMEOUT_S = 360


class VideoGenerationError(RuntimeError):
    pass


def _ensure_key() -> None:
    if not os.getenv("FAL_KEY") and os.getenv("FAL_API_KEY"):
        os.environ["FAL_KEY"] = os.environ["FAL_API_KEY"]
    if not os.getenv("FAL_KEY"):
        raise VideoGenerationError("FAL_API_KEY (or FAL_KEY) is not set in secrets.")


def _r2_key(project_id: str, shot_index: int) -> str:
    return f"projects/{project_id}/videos/shot_{shot_index}_{uuid.uuid4().hex[:6]}.mp4"


def _fal_accessible_url(r2_url: str) -> str:
    """Ensure Kling can access the image URL.

    Tries the URL anonymously first.  If not accessible, downloads from R2 via
    authenticated boto3 and uploads to FAL's transient CDN.
    """
    try:
        resp = requests.head(r2_url, timeout=6)
        if resp.status_code < 400:
            return r2_url
    except Exception:
        pass

    from urllib.parse import urlparse
    parsed = urlparse(r2_url)
    raw_path = parsed.path.lstrip("/")
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if bucket and raw_path.startswith(bucket + "/"):
        r2_key = raw_path[len(bucket) + 1:]
    else:
        r2_key = raw_path

    _ensure_key()
    img_bytes = r2_storage.download_bytes(r2_key)
    fal_url = fal_client.upload(img_bytes, "image/jpeg")
    logger.info("Re-hosted private R2 still on FAL CDN: %s", fal_url)
    return fal_url


def _run_fal_video(payload: dict, timeout_s: int = VIDEO_TIMEOUT_S) -> dict:
    _ensure_key()
    deadline = time.time() + timeout_s
    handler = fal_client.submit(VIDEO_MODEL, arguments=payload)
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            result = handler.get()
            if result is not None:
                return result
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if any(sig in err_str for sig in ("422", "400", "ValidationError",
                                               "loc", "field required", "value_error")):
                raise VideoGenerationError(f"FAL validation error on {VIDEO_MODEL}: {exc}") from exc
        time.sleep(3.0)
    raise VideoGenerationError(
        f"FAL video timed out after {timeout_s}s: {last_exc}"
    ) from last_exc


def _extract_video_url(result: dict) -> str:
    video = result.get("video")
    if isinstance(video, dict):
        url = video.get("url")
        if url:
            return url
    if isinstance(video, str):
        return video
    raise VideoGenerationError(f"Could not extract video URL from FAL response: {result}")


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
        still_url: Public URL of the still image (R2 URL).
        motion_prompt: Concise Kling-optimised camera/motion description.
            When provided, used as the video prompt instead of the full
            still-image visual_prompt (which is far too long for Kling).

    Returns a public R2 URL string (previously returned a local Path).
    Raises VideoGenerationError on failure.
    """
    _ensure_key()

    clip_duration = "5"
    if duration_s is not None and duration_s >= 8:
        clip_duration = "10"

    effective_prompt = (motion_prompt or "").strip()
    if not effective_prompt:
        effective_prompt = (prompt or "cinematic camera motion, smooth dolly").strip()[:200]

    short_prompt = effective_prompt[:250]

    # Ensure Kling can access the still image (private R2 → FAL CDN if needed)
    fal_still_url = _fal_accessible_url(still_url)

    payload = {
        "image_url": fal_still_url,
        "prompt": short_prompt,
        "duration": clip_duration,
        "aspect_ratio": ASPECT_RATIO,
    }

    logger.info(
        "Generating video for project=%s shot=%s dur=%ss",
        project_id, shot_index, clip_duration,
    )

    result = _run_fal_video(payload)
    video_fal_url = _extract_video_url(result)

    r2_key = _r2_key(project_id, shot_index)
    public_url = r2_storage.upload_from_url(
        video_fal_url, r2_key, content_type="video/mp4", stream=True
    )
    logger.info("Saved video clip to R2: %s", public_url)
    return public_url
