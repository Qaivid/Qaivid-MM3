"""
Provider Registry — Qaivid Core 2.0
Single source of truth for all image and video generation providers.
Admin selects the active provider per purpose; generation services read from here.
"""
from typing import Dict, Any, Optional

IMAGE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gpt_image_1": {
        "label": "GPT Image 1",
        "vendor": "OpenAI",
        "quality": "highest",
        "speed": "medium",
        "cost_per_image_usd": 0.040,
        "secret_key": "OPENAI_API_KEY",
        "notes": "Best quality. Highest cost. Recommended for reference portraits.",
    },
    "dall_e_3": {
        "label": "DALL-E 3",
        "vendor": "OpenAI",
        "quality": "high",
        "speed": "medium",
        "cost_per_image_usd": 0.040,
        "secret_key": "OPENAI_API_KEY",
        "notes": "Strong instruction following. Good for shot stills.",
    },
    "flux_dev": {
        "label": "Flux.1 Dev",
        "vendor": "fal.ai",
        "quality": "high",
        "speed": "medium",
        "cost_per_image_usd": 0.025,
        "secret_key": "FAL_API_KEY",
        "notes": "High quality, lower cost. Good balance for shot stills.",
    },
    "flux_schnell": {
        "label": "Flux.1 Schnell",
        "vendor": "fal.ai",
        "quality": "good",
        "speed": "fast",
        "cost_per_image_usd": 0.003,
        "secret_key": "FAL_API_KEY",
        "notes": "Fastest and cheapest. Best for shot stills at scale.",
    },
}

VIDEO_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "wan_2_6": {
        "label": "WAN 2.6",
        "vendor": "Atlas Cloud",
        "model_slug": "alibaba/wan-2.6/image-to-video-flash",
        "fallback_slug": "alibaba/wan-2.6/image-to-video",
        "quality": "high",
        "speed": "medium",
        "secret_key": "ATLAS_CLOUD_API_KEY",
        "notes": "Cinematic quality. Recommended for most productions.",
    },
    "kling_1_6": {
        "label": "Kling 1.6",
        "vendor": "Atlas Cloud",
        "model_slug": "kling-ai/kling-1.6/image-to-video",
        "fallback_slug": "kling-ai/kling-1.6/image-to-video",
        "quality": "highest",
        "speed": "slow",
        "secret_key": "ATLAS_CLOUD_API_KEY",
        "notes": "Highest quality motion. Slower render time. Best for hero shots.",
    },
    "kling_2_0": {
        "label": "Kling 2.0 Master",
        "vendor": "Atlas Cloud",
        "model_slug": "kling-ai/kling-2.0/image-to-video-master",
        "fallback_slug": "kling-ai/kling-2.0/image-to-video",
        "quality": "highest",
        "speed": "slow",
        "secret_key": "ATLAS_CLOUD_API_KEY",
        "notes": "Flagship Kling. Best cinematic output. Premium cost.",
    },
    "kling_3_0": {
        "label": "Kling 3.0",
        "vendor": "Atlas Cloud",
        "model_slug": "kling-ai/kling-3.0/image-to-video",
        "fallback_slug": "kling-ai/kling-2.0/image-to-video-master",
        "quality": "highest",
        "speed": "slow",
        "secret_key": "ATLAS_CLOUD_API_KEY",
        "notes": "Latest Kling. Most advanced motion intelligence. Falls back to Kling 2.0 Master.",
    },
}

DEFAULT_SETTINGS = {
    "image_provider_references": "flux_schnell",
    "image_provider_stills": "flux_schnell",
    "video_provider": "wan_2_6",
}


def get_image_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    return IMAGE_PROVIDERS.get(provider_id)


def get_video_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    return VIDEO_PROVIDERS.get(provider_id)


def list_image_providers():
    return [{"id": k, **v} for k, v in IMAGE_PROVIDERS.items()]


def list_video_providers():
    return [{"id": k, **v} for k, v in VIDEO_PROVIDERS.items()]
