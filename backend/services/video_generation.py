"""
Video Generation — AtlasCloud (WAN 2.6, Kling 1.6, Kling 2.0)
Image-to-video using AtlasCloud's unified API. Provider and model selected from
platform settings so admin can switch without code changes.

Architecture: Submit + Store + Frontend-polls-status.
No background threads. Prediction IDs stored in MongoDB.
Frontend calls /render-status which checks AtlasCloud and downloads completed videos.
"""
import os
import uuid
import logging
import httpx
from typing import Optional
from models import now_utc

logger = logging.getLogger(__name__)

VIDEO_DIR = "/app/backend/generated_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

BASE_URL = "https://api.atlascloud.ai"
# Defaults — overridden at call time by the active provider config
PRIMARY_MODEL = "alibaba/wan-2.6/image-to-video-flash"
FALLBACK_MODEL = "alibaba/wan-2.6/image-to-video"

TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def _auth_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ─── Upload local image to AtlasCloud ─────────────────────

async def upload_image_to_atlas(filepath: str, api_key: str) -> str:
    """Upload a local image file to AtlasCloud, return hosted URL."""
    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        with open(filepath, "rb") as f:
            files = {"file": (f"image.{ext}", f, mime)}
            resp = await client.post(
                f"{BASE_URL}/api/v1/model/upload",
                headers=_auth_headers(api_key),
                files=files,
            )

    if resp.status_code != 200:
        raise ValueError(f"AtlasCloud image upload failed (HTTP {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    url = data.get("data", {}).get("url") or data.get("url")
    if not url:
        raise ValueError(f"AtlasCloud upload returned no URL: {resp.text[:300]}")

    logger.info(f"[AtlasCloud] Image uploaded: {url}")
    return url


# ─── Resolve image to a public URL AtlasCloud can fetch ───

async def resolve_image_url(image_path: str, api_key: str) -> str:
    """If the image is a local file, upload it. If it's already a URL, pass through."""
    if not image_path or not image_path.strip():
        raise FileNotFoundError("No image path provided")

    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path

    # Local file — upload to AtlasCloud
    local = image_path if os.path.isabs(image_path) else os.path.join("/app/backend/generated_images", os.path.basename(image_path))
    if not os.path.exists(local) or os.path.isdir(local):
        raise FileNotFoundError(f"Image not found: {local}")

    return await upload_image_to_atlas(local, api_key)


# ─── Submit video generation ──────────────────────────────

async def submit_video_generation(
    image_url: str,
    prompt: str,
    api_key: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    primary_model: str = None,
    fallback_model: str = None,
) -> str:
    """
    Submit a video generation job to AtlasCloud.
    Model is determined by provider config passed in (admin-configurable).
    Returns prediction_id. Does NOT wait for completion.
    """
    duration = max(2, min(duration, 15))
    primary = primary_model or PRIMARY_MODEL
    fallback = fallback_model or FALLBACK_MODEL

    # Enhance prompt to avoid slow-motion unless intended
    if not any(kw in prompt.lower() for kw in ("slow motion", "slow-motion", "slowmo", "timelapse")):
        prompt = f"{prompt}. Dynamic movement, natural real-time speed, no slow motion."

    body = {
        "model": primary,
        "image": image_url,
        "prompt": prompt,
        "negative_prompt": "blurry, distorted, low quality, watermark, text overlay, artifacts, jitter, slow motion, unnaturally slow movement",
        "duration": duration,
        "resolution": "720p",
        "aspect_ratio": aspect_ratio,
        "seed": -1,
        "shot_type": "multi" if duration >= 8 else "single",
        "generate_audio": False,
        "enable_prompt_expansion": False,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_URL}/api/v1/model/generateVideo",
            headers={**_auth_headers(api_key), "Content-Type": "application/json"},
            json=body,
        )

        if not resp.is_success:
            # Try fallback model
            logger.warning(f"[AtlasCloud] Primary model failed (HTTP {resp.status_code}), trying fallback: {fallback}")
            body["model"] = fallback
            resp = await client.post(
                f"{BASE_URL}/api/v1/model/generateVideo",
                headers={**_auth_headers(api_key), "Content-Type": "application/json"},
                json=body,
            )
            if not resp.is_success:
                raise ValueError(f"AtlasCloud generation failed (HTTP {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    prediction_id = data.get("data", {}).get("id")
    if not prediction_id:
        raise ValueError(f"AtlasCloud returned no prediction ID: {resp.text[:300]}")

    logger.info(f"[AtlasCloud] Submitted: {prediction_id} (model={body['model']}, {duration}s)")
    return prediction_id


# ─── Check status of a single prediction ──────────────────

async def check_prediction_status(prediction_id: str, api_key: str) -> dict:
    """
    Check status of a video generation prediction.
    Returns: {"status": "processing|completed|failed", "video_url": "...", "error": "..."}
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{BASE_URL}/api/v1/model/prediction/{prediction_id}",
            headers=_auth_headers(api_key),
        )

    if not resp.is_success:
        if resp.status_code == 404 or "not found" in resp.text.lower():
            return {"status": "failed", "error": f"Prediction {prediction_id} not found"}
        # Transient error — treat as still processing
        return {"status": "processing"}

    data = resp.json()
    inner = data.get("data", data)
    status = (inner.get("status") or "").lower()

    if status in ("completed", "succeeded"):
        video_url = None
        if isinstance(inner.get("outputs"), list) and inner["outputs"]:
            video_url = inner["outputs"][0]
        else:
            video_url = inner.get("output", {}).get("video_url") or inner.get("output_url") or inner.get("video_url")

        if not video_url:
            return {"status": "failed", "error": "Completed but no video URL returned"}

        return {"status": "completed", "video_url": video_url}

    elif status in ("failed", "error", "cancelled"):
        error = inner.get("error") or inner.get("message") or "Generation failed"
        return {"status": "failed", "error": str(error)[:300]}

    else:
        # processing, pending, queued, starting
        return {"status": "processing"}


# ─── Download a completed video to local storage ──────────

async def download_video(remote_url: str, video_id: str) -> dict:
    """Download a completed video from AtlasCloud to local storage."""
    filename = f"{video_id}.mp4"
    filepath = os.path.join(VIDEO_DIR, filename)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), follow_redirects=True) as client:
        resp = await client.get(remote_url)
        if not resp.is_success:
            raise ValueError(f"Failed to download video: HTTP {resp.status_code}")

        with open(filepath, "wb") as f:
            f.write(resp.content)

    if os.path.getsize(filepath) < 1000:
        os.unlink(filepath)
        raise ValueError("Downloaded video too small, likely corrupt")

    logger.info(f"[AtlasCloud] Video downloaded: {filepath} ({os.path.getsize(filepath)} bytes)")
    return {
        "video_id": video_id,
        "filepath": filepath,
        "filename": filename,
        "video_url": f"/api/videos/{filename}",
        "downloaded_at": now_utc(),
    }


# ─── High-level: Submit a render job for a shot ───────────

async def submit_shot_render(render_job: dict, shot: dict, api_key: str, provider_config: dict = None) -> dict:
    """
    Submit a single shot for video generation using the configured provider.
    provider_config comes from the provider_registry (admin-selected).
    Returns updated render_job with prediction_id and status='submitted'.
    """
    cfg = provider_config or {}
    primary = cfg.get("model_slug") or PRIMARY_MODEL
    fallback = cfg.get("fallback_slug") or FALLBACK_MODEL
    provider_label = cfg.get("label", "WAN 2.6")

    # Build the still image path — field is `input_image_url` from production_pipeline
    still_image = render_job.get("input_image_url") or render_job.get("still_image_url") or render_job.get("image_url", "")
    if not still_image:
        return {**render_job, "status": "failed", "error": "No still image available for this shot"}

    if still_image.startswith("/api/images/"):
        still_image = os.path.join("/app/backend/generated_images", still_image.split("/")[-1])

    # Build prompt from shot data
    action = shot.get("subject_action", "")
    shot_type = shot.get("shot_type", "medium")
    camera = shot.get("camera_behavior", "static")
    light = shot.get("light_description", "")
    emotion = shot.get("emotional_micro_state", "")
    motion = render_job.get("motion_prompt", "")

    prompt = f"Cinematic {shot_type} shot. {action}. Camera: {camera}. {motion}. Lighting: {light}. Mood: {emotion}."

    # Resolve duration
    hint = render_job.get("duration_sec", shot.get("duration_hint", 5))
    duration = max(2, min(int(hint), 15))
    aspect = render_job.get("aspect_ratio", "16:9")

    try:
        image_url = await resolve_image_url(still_image, api_key)
        prediction_id = await submit_video_generation(
            image_url, prompt, api_key, duration, aspect,
            primary_model=primary, fallback_model=fallback,
        )

        return {
            **render_job,
            "status": "submitted",
            "prediction_id": prediction_id,
            "submitted_at": now_utc(),
            "provider": "atlascloud",
            "provider_label": provider_label,
            "model": primary,
        }
    except Exception as e:
        logger.error(f"[AtlasCloud/{provider_label}] Submit failed for shot {render_job.get('shot_number', '?')}: {e}")
        return {
            **render_job,
            "status": "failed",
            "error": str(e)[:300],
        }


# ─── High-level: Check status and download if ready ───────

async def check_and_download(render_job: dict, api_key: str) -> dict:
    """
    Check a submitted render job's status. If completed, download the video.
    Returns updated render_job dict.
    """
    prediction_id = render_job.get("prediction_id")
    if not prediction_id:
        return {**render_job, "status": "failed", "error": "No prediction ID"}

    result = await check_prediction_status(prediction_id, api_key)

    if result["status"] == "completed":
        vid_id = render_job.get("id", str(uuid.uuid4()))
        try:
            dl = await download_video(result["video_url"], vid_id)
            return {
                **render_job,
                "status": "completed",
                "output_video_url": dl["video_url"],
                "filepath": dl["filepath"],
                "actual_duration": render_job.get("duration_sec", 5),
                "generated_at": dl["downloaded_at"],
            }
        except Exception as e:
            logger.error(f"[AtlasCloud] Download failed: {e}")
            return {**render_job, "status": "failed", "error": f"Download failed: {e}"}

    elif result["status"] == "failed":
        return {**render_job, "status": "failed", "error": result.get("error", "Unknown error")}

    else:
        return {**render_job, "status": "processing"}
