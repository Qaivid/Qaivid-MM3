"""
Image Generation — Multi-Provider
Supports: GPT Image 1, DALL-E 3, Flux.1 Dev, Flux.1 Schnell
Active provider is determined by platform settings (admin-configurable).
"""
import os
import base64
import uuid
import logging
import httpx
from openai import AsyncOpenAI
from models import now_utc

logger = logging.getLogger(__name__)

UPLOAD_DIR = "/app/backend/generated_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FAL_API_BASE = "https://fal.run"
TIMEOUT = httpx.Timeout(120.0, connect=15.0)


# ─── Provider dispatch ────────────────────────────────────

async def generate_image(
    prompt: str,
    api_key: str,
    image_id: str = None,
    provider: str = "gpt_image_1",
    negative_prompt: str = "",
    size: str = "landscape",
) -> dict:
    """
    Generate one image with the specified provider.
    Returns: {image_id, filepath, filename, image_url, generated_at}
    """
    img_id = image_id or str(uuid.uuid4())

    if provider in ("gpt_image_1", "dall_e_3"):
        return await _generate_openai(prompt, api_key, img_id, provider, size)
    elif provider in ("flux_schnell", "flux_dev"):
        return await _generate_fal(prompt, api_key, img_id, provider, negative_prompt, size)
    else:
        raise ValueError(f"Unknown image provider: {provider}")


# ─── OpenAI (GPT Image 1 + DALL-E 3) ─────────────────────

async def _generate_openai(
    prompt: str, api_key: str, img_id: str, provider: str, size: str
) -> dict:
    if not api_key:
        raise ValueError("OpenAI API key not configured.")

    client = AsyncOpenAI(api_key=api_key)
    model = "gpt-image-1" if provider == "gpt_image_1" else "dall-e-3"

    # GPT Image 1 returns base64; DALL-E 3 returns URL by default
    if model == "gpt-image-1":
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1536x1024",
        )
        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise ValueError("No image data returned from GPT Image 1")
        image_bytes = base64.b64decode(image_b64)
    else:
        # DALL-E 3 — request base64
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1792x1024",
            response_format="b64_json",
        )
        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise ValueError("No image data returned from DALL-E 3")
        image_bytes = base64.b64decode(image_b64)

    filename = f"{img_id}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return {
        "image_id": img_id,
        "filepath": filepath,
        "filename": filename,
        "image_url": f"/api/images/{filename}",
        "generated_at": now_utc(),
        "provider": provider,
    }


# ─── fal.ai (Flux Schnell + Flux Dev) ────────────────────

FAL_MODEL_MAP = {
    "flux_schnell": "fal-ai/flux/schnell",
    "flux_dev":     "fal-ai/flux/dev",
}

async def _generate_fal(
    prompt: str, api_key: str, img_id: str, provider: str,
    negative_prompt: str = "", size: str = "landscape"
) -> dict:
    if not api_key:
        raise ValueError("fal.ai API key not configured. Set FAL_API_KEY in Settings.")

    model_path = FAL_MODEL_MAP.get(provider)
    if not model_path:
        raise ValueError(f"Unknown Flux provider: {provider}")

    # fal.ai image dimensions
    size_map = {
        "landscape": {"width": 1344, "height": 768},
        "portrait":  {"width": 768,  "height": 1344},
        "square":    {"width": 1024, "height": 1024},
    }
    dims = size_map.get(size, size_map["landscape"])

    payload = {
        "prompt": prompt,
        "image_size": dims,
        "num_inference_steps": 4 if provider == "flux_schnell" else 28,
        "num_images": 1,
        "enable_safety_checker": True,
    }
    if provider == "flux_dev" and negative_prompt:
        payload["negative_prompt"] = negative_prompt

    api_key = api_key.strip()
    url = f"{FAL_API_BASE}/{model_path}"
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    logger.info(f"[fal.ai] POST {url} | key_len={len(api_key)} | key_prefix={api_key[:8]}...")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if not resp.is_success:
        raise ValueError(f"fal.ai generation failed (HTTP {resp.status_code}): {resp.text[:400]}")

    data = resp.json()
    images = data.get("images") or []
    if not images:
        raise ValueError(f"fal.ai returned no images: {resp.text[:400]}")

    image_url = images[0].get("url")
    if not image_url:
        raise ValueError("fal.ai returned image without URL")

    # Download the image to local storage
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        dl_resp = await client.get(image_url)
    if not dl_resp.is_success:
        raise ValueError(f"Failed to download fal.ai image: HTTP {dl_resp.status_code}")

    filename = f"{img_id}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(dl_resp.content)

    return {
        "image_id": img_id,
        "filepath": filepath,
        "filename": filename,
        "image_url": f"/api/images/{filename}",
        "generated_at": now_utc(),
        "provider": provider,
    }


# ─── High-level wrappers (used by server.py) ─────────────

async def generate_reference_image(ref_prompt: dict, api_key: str, provider: str = "gpt_image_1") -> dict:
    try:
        result = await generate_image(
            prompt=ref_prompt.get("prompt", ""),
            api_key=api_key,
            image_id=ref_prompt.get("id"),
            provider=provider,
            size="portrait",
        )
        return {**ref_prompt, "status": "completed", "image_url": result["image_url"], "filepath": result["filepath"], "generated_at": result["generated_at"], "provider": provider}
    except Exception as e:
        logger.error(f"Reference image failed [{provider}]: {e}")
        return {**ref_prompt, "status": "failed", "error": str(e)}


async def generate_still_image(still_prompt: dict, api_key: str, provider: str = "gpt_image_1") -> dict:
    prompt = still_prompt.get("positive_prompt", "")
    neg = still_prompt.get("negative_prompt", "")
    full_prompt = f"{prompt}. Avoid: {neg}" if neg and provider in ("gpt_image_1", "dall_e_3") else prompt
    try:
        result = await generate_image(
            prompt=full_prompt,
            api_key=api_key,
            image_id=still_prompt.get("id"),
            provider=provider,
            negative_prompt=neg,
            size="landscape",
        )
        return {**still_prompt, "status": "completed", "image_url": result["image_url"], "filepath": result["filepath"], "generated_at": result["generated_at"], "provider": provider}
    except Exception as e:
        logger.error(f"Still image failed [{provider}]: {e}")
        return {**still_prompt, "status": "failed", "error": str(e)}
