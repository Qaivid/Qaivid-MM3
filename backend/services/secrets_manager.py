"""
Secrets Manager
Secure storage for user API keys. Keys are stored in MongoDB,
loaded at runtime, never hardcoded, never logged.
"""
import os
import logging
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# In-memory cache to avoid DB reads on every API call
_cache: Dict[str, str] = {}


async def get_secret(db, key: str) -> Optional[str]:
    """Get a secret value. Checks cache first, then DB."""
    if key in _cache and _cache[key]:
        return _cache[key]
    doc = await db.secrets.find_one({"key": key}, {"_id": 0})
    if doc and doc.get("value"):
        _cache[key] = doc["value"]
        return doc["value"]
    # Fallback to env var
    env_val = os.environ.get(key, "").strip()
    if env_val:
        _cache[key] = env_val
        return env_val
    return None


async def set_secret(db, key: str, value: str):
    """Set a secret value. Stores in DB, clears ALL cache to prevent stale reads."""
    await db.secrets.update_one(
        {"key": key},
        {"$set": {"key": key, "value": value}},
        upsert=True,
    )
    # Clear entire cache to ensure no stale keys remain
    _cache.clear()
    _cache[key] = value


async def delete_secret(db, key: str):
    """Delete a secret."""
    await db.secrets.delete_one({"key": key})
    _cache.pop(key, None)


async def list_secret_keys(db):
    """List all secret keys (without values)."""
    docs = await db.secrets.find({}, {"_id": 0, "key": 1}).to_list(50)
    return [d["key"] for d in docs]


async def get_secrets_status(db):
    """Get which keys are configured (without revealing values)."""
    keys = await list_secret_keys(db)
    env_keys = []
    for k in REQUIRED_SECRETS:
        if os.environ.get(k):
            env_keys.append(k)

    return {
        s["key"]: {
            "configured": s["key"] in keys or s["key"] in env_keys,
            "source": "database" if s["key"] in keys else ("environment" if s["key"] in env_keys else "not set"),
            "label": s["label"],
            "description": s["description"],
            "required": s["required"],
        }
        for s in REQUIRED_SECRETS_META
    }


def clear_cache():
    """Clear the in-memory cache (e.g., after key update)."""
    _cache.clear()


# Define all secrets the system needs
REQUIRED_SECRETS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ATLAS_CLOUD_API_KEY",
    "FAL_API_KEY",
]

REQUIRED_SECRETS_META = [
    {
        "key": "OPENAI_API_KEY",
        "label": "OpenAI API Key",
        "description": "Powers GPT (intelligence + brief), GPT Image 1 (references + stills), Whisper-1 (audio timing). Get from platform.openai.com/api-keys",
        "required": True,
    },
    {
        "key": "GEMINI_API_KEY",
        "label": "Google Gemini API Key",
        "description": "Powers audio transcription (lyrics + segment detection). Gemini understands Punjabi, Urdu, Hindi audio natively. Get from aistudio.google.com/apikey",
        "required": True,
    },
    {
        "key": "ATLAS_CLOUD_API_KEY",
        "label": "AtlasCloud API Key",
        "description": "Powers video generation using Wan 2.6 (image-to-video). Fast, cost-effective clips from shot stills. Get from atlascloud.ai",
        "required": True,
    },
    {
        "key": "FAL_API_KEY",
        "label": "fal.ai API Key",
        "description": "Powers Flux Schnell and Flux Dev image generation (shot stills + reference images). 93% cheaper than DALL-E. Get from fal.ai/dashboard/keys",
        "required": False,
    },
]
