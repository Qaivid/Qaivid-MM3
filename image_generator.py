"""FAL-backed image generation for Qaivid MetaMind.

Two roles:
  * Reference images (character, environment) — text-to-image with FLUX schnell.
  * Per-shot stills — face-locked using FLUX + PuLID when a character ref exists,
    otherwise plain FLUX.

All generated images are stored in Cloudflare R2 (r2_storage.py).
Returns public R2 URLs instead of local Path objects.
"""
from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Optional

import fal_client
import requests

import r2_storage

logger = logging.getLogger(__name__)

# ── FAL model identifiers (quality mode) ─────────────────────────────────────
REF_MODEL = "fal-ai/flux/schnell"
SHOT_MODEL_FACE = "fal-ai/flux-pulid"
SHOT_MODEL_NO_FACE = "fal-ai/flux/dev"
SHOT_MODEL_ENV_I2I = "fal-ai/flux/dev/image-to-image"

# ── Standard mode (FLUX/schnell, no face-lock) ───────────────────────────────
# Middle tier: ~$0.003–0.005/image, cinematic-quality stills, no face-lock.
# Same FLUX backbone as reference plate generation — fast and affordable.
# Good for wide shots, landscapes, drone shots, or projects where character
# face consistency is less critical (env-heavy or abstract videos).
# All shot types use FLUX/schnell text-to-image; no img2img conditioning.
# Deprecated alias sdxl_face → standard (ip-adapter-face-id-plus was removed).
STANDARD_SHOT_MODEL = "fal-ai/flux/schnell"     # standard mode — FLUX schnell for all shot types

# ── OpenAI gpt-image-1.5 cheap-mode constants ────────────────────────────────
# Cheap mode routes ALL generation (refs + stills) through gpt-image-1.5 at
# "low" quality. Cost: $0.009–$0.013/image (1024×1024 → 1536×1024)
# vs ~$0.025–0.05 for FAL flux-pulid quality mode.
# Labelled "GPT Image 1.5 · Low" in the admin UI.
# gpt-image-1.5 supports image-to-image and multi-image references —
# face identity + scene atmosphere preserved in a single edit call.
# Sizes: 1024x1536 (portrait) for character refs, 1536x1024 for everything
# else (landscape, nearest to 16:9 that gpt-image-1.5 supports).
OPENAI_IMAGE_MODEL = "gpt-image-1.5"
OPENAI_CHEAP_QUALITY = "low"
OPENAI_SIZE_LANDSCAPE = "1536x1024"
OPENAI_SIZE_PORTRAIT = "1024x1536"


def _resolve_ref_mode() -> str:
    """Returns 'cheap' or 'quality' for reference plates."""
    try:
        from system_config import get_image_modes
        return get_image_modes().get("ref", "quality")
    except Exception:
        return "quality"


def _resolve_shot_mode() -> str:
    """Returns 'cheap' or 'quality' for per-shot stills."""
    try:
        from system_config import get_image_modes
        return get_image_modes().get("shot", "quality")
    except Exception:
        return "quality"


# ── OpenAI image helpers ──────────────────────────────────────────────────────

def _openai_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ImageGenerationError("OPENAI_API_KEY is not set in secrets.")
    return OpenAI(api_key=key)


def ai_build_ref_prompts(
    characters: list,
    locations: list,
    context_packet: dict,
    style_profile: dict,
) -> dict:
    """Ask GPT to write purpose-built image prompts for every character and location.

    Returns::
        {
            "characters": {"<id>": "<prompt>"},
            "locations":  {"<id>": "<prompt>"},
        }

    Falls back to an empty dict on any failure — callers then use the
    template-built prompts as a safe fallback.
    """
    if not characters and not locations:
        return {}

    # ── Summarise story context for the LLM ─────────────────────────────────
    cp = context_packet or {}
    sp = style_profile or {}
    cin = sp.get("cinematic") or {}

    story_summary = "\n".join(filter(None, [
        f"World: {cp.get('location_dna', '')}",
        f"Era / setting: {cp.get('world_assumptions', {}).get('era', '') if isinstance(cp.get('world_assumptions'), dict) else ''}",
        f"Emotional themes: {cp.get('emotional_core', '')}",
        f"Visual style: {cin.get('look', '')} — {cin.get('palette', '')}",
        f"Genre: {sp.get('genre', '')}",
    ]))

    # ── Serialise entities ───────────────────────────────────────────────────
    def _char_summary(c: dict) -> str:
        parts = [
            f"ID:{c['id']} Name:{c.get('name','')}",
            f"Role:{c.get('role','')} Age:{c.get('age_range','')} Gender:{c.get('gender','')}",
            f"Ethnicity:{c.get('ethnicity','')} Complexion:{c.get('complexion','')}",
            f"Wardrobe:{c.get('wardrobe','')} Grooming:{c.get('grooming','')}",
            f"Cultural:{c.get('cultural_notes','')}",
        ]
        return " | ".join(p for p in parts if p.split(":",1)[-1].strip())

    def _loc_summary(l: dict) -> str:
        parts = [
            f"ID:{l['id']} Name:{l.get('name','')}",
            f"Desc:{l.get('description','')}",
            f"Geography:{l.get('geography','')} Time:{l.get('time_of_day','')}",
            f"Weather:{l.get('weather_or_atmosphere','')} Mood:{l.get('mood','')}",
            f"Props:{l.get('visual_details','')} Cultural:{l.get('cultural_notes','')}",
        ]
        return " | ".join(p for p in parts if p.split(":",1)[-1].strip())

    chars_block = "\n".join(_char_summary(c) for c in characters)
    locs_block  = "\n".join(_loc_summary(l)  for l in locations)

    system_msg = (
        "You are a world-class cinematic image prompt writer for AI image generation models. "
        "Your prompts read like a director's vision — flowing, evocative, and precise. "
        "You never use field labels like 'Wardrobe:' or 'Architecture:'. "
        "You always weave visual details into natural prose that gives the image model "
        "creative room while anchoring it firmly to the cultural and emotional world of the story."
    )

    user_msg = f"""You are creating reference image prompts for a music video project.

STORY CONTEXT:
{story_summary}

CHARACTERS (write a PORTRAIT prompt for each):
{chars_block}

LOCATIONS (write a SCENE prompt for each):
{locs_block}

INSTRUCTIONS:
- CHARACTER prompts: 2-3 sentences. Describe who the person is, their physical presence, emotional bearing, cultural world. Weave wardrobe and grooming in naturally. End with: "Photorealistic portrait, cinematic lighting, sharp facial detail, culturally authentic appearance, regionally accurate clothing and features."
- LOCATION prompts: 3-4 sentences. Open with time-of-day + cultural setting. Describe the place cinematically. Mention key props naturally. State the scene is empty (no people). End with: "Establishing wide-angle shot, photorealistic, cinematic, geographically and architecturally authentic, atmospheric lighting, no people."
- Do NOT copy field labels into prompts. Write freely.
- Each prompt should feel unique to that specific character or location — not generic.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "characters": {{"<id as string>": "<portrait prompt>"}},
  "locations":  {{"<id as string>": "<scene prompt>"}}
}}"""

    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        import json as _json
        raw = response.choices[0].message.content or "{}"
        result = _json.loads(raw)
        chars_prompts = {str(k): v for k, v in (result.get("characters") or {}).items()}
        locs_prompts  = {str(k): v for k, v in (result.get("locations")  or {}).items()}
        logger.info(
            "ai_build_ref_prompts: got %d char + %d loc prompts",
            len(chars_prompts), len(locs_prompts),
        )
        return {"characters": chars_prompts, "locations": locs_prompts}
    except Exception as exc:
        logger.warning("ai_build_ref_prompts failed (%s) — falling back to templates", exc)
        return {}


def _openai_generate(prompt: str, size: str = OPENAI_SIZE_LANDSCAPE,
                     quality: str = OPENAI_CHEAP_QUALITY) -> bytes:
    """Text-to-image via gpt-image-1.5. Returns raw PNG bytes."""
    client = _openai_client()
    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt[:4000],
        size=size,
        quality=quality,
        n=1,
    )
    import base64 as _b64
    b64 = response.data[0].b64_json
    if not b64:
        raise ImageGenerationError("gpt-image-1.5 returned no image data.")
    return _b64.b64decode(b64)


def _download_image_bytes(url: str) -> bytes:
    """Download image bytes from any URL — public or private R2.

    Tries a plain HTTP GET first.  If the server returns 4xx (e.g. private R2
    bucket returns 400/403), extracts the R2 key from the URL and downloads
    via authenticated boto3 — the same fallback used by _fal_accessible_url.
    This makes all OpenAI img2img calls work regardless of whether R2_PUBLIC_URL
    points to a public CDN domain or the private r2.cloudflarestorage.com endpoint.
    """
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code < 400:
            return resp.content
    except Exception:
        pass

    # Fallback: authenticated R2 download using boto3
    from urllib.parse import urlparse
    parsed = urlparse(url)
    raw_path = parsed.path.lstrip("/")
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if bucket and raw_path.startswith(bucket + "/"):
        r2_key = raw_path[len(bucket) + 1:]
    else:
        r2_key = raw_path
    logger.info("_download_image_bytes: falling back to boto3 for key=%s", r2_key)
    return r2_storage.download_bytes(r2_key)


def _openai_edit(prompt: str, ref_url: str, size: str = OPENAI_SIZE_LANDSCAPE,
                 quality: str = OPENAI_CHEAP_QUALITY) -> bytes:
    """Image-to-image edit via gpt-image-1.5 (single reference).
    Downloads ref from R2/URL, passes to the edits endpoint, returns raw PNG bytes."""
    return _openai_edit_multi(prompt, [ref_url], size=size, quality=quality)


def _openai_edit_multi(prompt: str, ref_urls: list, size: str = OPENAI_SIZE_LANDSCAPE,
                       quality: str = OPENAI_CHEAP_QUALITY) -> bytes:
    """Multi-image-to-image edit via gpt-image-1.5.

    gpt-image-1.5 accepts multiple image references simultaneously — it uses
    all supplied images to condition the output, preserving:
      * face identity (from character plate)
      * scene atmosphere / lighting (from environment plate)
      * wardrobe / colour palette (from look plate)

    Passes up to 3 ref images as a list to the edits endpoint.
    Falls back to single-image edit if only one URL is supplied.
    Returns raw PNG bytes.
    """
    import io
    import base64 as _b64

    valid_urls = [u for u in (ref_urls or []) if u]
    if not valid_urls:
        raise ImageGenerationError("_openai_edit_multi called with no valid ref URLs.")

    client = _openai_client()

    if len(valid_urls) == 1:
        img_bytes = _download_image_bytes(valid_urls[0])
        image_arg = ("reference.png", io.BytesIO(img_bytes), "image/png")
    else:
        images = []
        for i, url in enumerate(valid_urls[:3]):  # gpt-image-1.5 cap: 3 refs
            img_bytes = _download_image_bytes(url)
            images.append((f"ref_{i}.png", io.BytesIO(img_bytes), "image/png"))
        image_arg = images

    response = client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=image_arg,
        prompt=prompt[:4000],
        size=size,
        quality=quality,
        n=1,
    )
    b64 = response.data[0].b64_json
    if not b64:
        raise ImageGenerationError("gpt-image-1.5 edit returned no image data.")
    return _b64.b64decode(b64)


def _save_bytes_to_r2(img_bytes: bytes, r2_key: str) -> str:
    """Upload raw image bytes to R2 and return the public URL."""
    return r2_storage.upload_bytes(img_bytes, r2_key, content_type="image/png")

CHARACTER_REF_HINT = (
    "photorealistic portrait, cinematic lighting, sharp facial detail, "
    "natural skin texture, head-and-shoulders composition, "
    "culturally authentic appearance, regionally accurate clothing and features"
)

ENV_REF_HINT = (
    "establishing shot, photorealistic, cinematic wide angle, atmospheric lighting, "
    "no people, rich environmental detail, naturalistic color, "
    "geographically and architecturally authentic, true to local culture and heritage, "
    "accurate regional architecture and materials"
)


class ImageGenerationError(RuntimeError):
    pass


def _ensure_key() -> None:
    if not os.getenv("FAL_KEY") and os.getenv("FAL_API_KEY"):
        os.environ["FAL_KEY"] = os.environ["FAL_API_KEY"]
    if not os.getenv("FAL_KEY"):
        raise ImageGenerationError("FAL_API_KEY (or FAL_KEY) is not set in secrets.")


def _r2_key(project_id: str, sub: str, name: str) -> str:
    return f"projects/{project_id}/{sub}/{name}"


def _new_ref_key(project_id: str, role: str, ext: str = "jpg") -> str:
    return _r2_key(project_id, "refs", f"{role}_{uuid.uuid4().hex[:6]}.{ext}")


def _new_shot_key(project_id: str, shot_index, ext: str = "jpg") -> str:
    return _r2_key(project_id, "shots", f"shot_{shot_index}_{uuid.uuid4().hex[:6]}.{ext}")


def _save_to_r2(fal_url: str, r2_key: str) -> str:
    """Download image from FAL and upload to R2. Returns public R2 URL."""
    return r2_storage.upload_from_url(fal_url, r2_key, content_type="image/jpeg")


def _run_fal(model: str, payload: dict, timeout_s: int = 240) -> dict:
    _ensure_key()
    deadline = time.time() + timeout_s
    handler = fal_client.submit(model, arguments=payload)
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            result = handler.get()
            if result is not None:
                return result
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # Validation / client errors (4xx-equivalent) — fail immediately
            if any(sig in err_str for sig in ("422", "400", "ValidationError",
                                               "loc", "field required", "value_error")):
                raise ImageGenerationError(f"FAL validation error on {model}: {exc}") from exc
        time.sleep(2.0)
    raise ImageGenerationError(
        f"FAL timed out after {timeout_s}s on {model}: {last_exc}"
    ) from last_exc


def _extract_image_url(result: dict) -> str:
    if not isinstance(result, dict):
        raise ImageGenerationError(f"Unexpected FAL response type: {type(result)}")
    images = result.get("images") or result.get("image")
    if isinstance(images, dict):
        images = [images]
    if not images:
        raise ImageGenerationError(f"FAL returned no images: {result}")
    first = images[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        url = first.get("url")
        if url:
            return url
    raise ImageGenerationError(f"Could not extract image URL from FAL response: {result}")


def _fal_accessible_url(r2_url: str) -> str:
    """Ensure FAL can access the image.

    If the R2 bucket is public the URL works as-is.  If not (HTTP 4xx), we
    extract the R2 key from the URL, download the bytes via authenticated boto3,
    and re-upload them to FAL's transient CDN.
    """
    try:
        resp = requests.head(r2_url, timeout=6)
        if resp.status_code < 400:
            return r2_url
    except Exception:
        pass

    # Not publicly accessible — extract R2 key and download via authenticated boto3
    from urllib.parse import urlparse
    parsed = urlparse(r2_url)
    # R2 path is either /<key> or /<bucket>/<key> depending on R2_PUBLIC_URL config
    # Strip leading slash then remove bucket prefix if present
    raw_path = parsed.path.lstrip("/")
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if bucket and raw_path.startswith(bucket + "/"):
        r2_key = raw_path[len(bucket) + 1:]
    else:
        # R2_PUBLIC_URL already includes bucket; raw path IS the key
        r2_key = raw_path

    _ensure_key()
    img_bytes = r2_storage.download_bytes(r2_key)
    fal_url = fal_client.upload(img_bytes, "image/jpeg")
    logger.info("Re-hosted private R2 image on FAL CDN: %s", fal_url)
    return fal_url


def generate_character_ref(speaker: dict, location_dna: str, project_id: str) -> str:
    """Generate a character reference portrait and store it in R2.

    Returns a public R2 URL string (previously returned a local Path).
    """
    identity = (speaker.get("identity") or "a person").strip()
    gender = (speaker.get("gender") or "").strip()
    age = (speaker.get("age_range") or "").strip()
    role = (speaker.get("social_role") or "").strip()
    emotion = (speaker.get("emotional_state") or "").strip()

    descriptors = [d for d in [age, gender, role] if d and d.lower() not in {"unclear", "unknown"}]
    descriptor_str = ", ".join(descriptors) if descriptors else "adult"
    region_hint = f", consistent with {location_dna}" if location_dna and location_dna.lower() != "universal" else ""

    prompt = (
        f"Reference portrait of a single {descriptor_str} character{region_hint}. "
        f"Identity context: {identity}. Emotional tone: {emotion}. "
        f"{CHARACTER_REF_HINT}."
    )
    logger.info("Generating character ref for project=%s mode=%s", project_id, _resolve_ref_mode())

    if _resolve_ref_mode() == "cheap":
        img_bytes = _openai_generate(prompt, size=OPENAI_SIZE_PORTRAIT)
        r2_key = _new_ref_key(project_id, "character", ext="png")
        return _save_bytes_to_r2(img_bytes, r2_key)

    result = _run_fal(REF_MODEL, {
        "prompt": prompt,
        "image_size": "portrait_4_3",
        "num_inference_steps": 8,
        "num_images": 1,
        "seed": random.randint(1, 2**32 - 1),
        "enable_safety_checker": False,
    })
    fal_url = _extract_image_url(result)
    r2_key = _new_ref_key(project_id, "character")
    return _save_to_r2(fal_url, r2_key)


def build_character_plate_prompt(character: dict, location_dna: str = "Universal") -> str:
    """Build a director-style casting-brief prompt for a character.

    Uses flowing prose rather than a labelled checklist so the image model
    can interpret the character naturally.  The `appearance` field is skipped
    because it is typically an auto-generated concatenation of the other
    fields, which would just repeat information already present.
    """
    name     = (character.get("name")    or "").strip()
    role     = (character.get("role")    or character.get("entity_type") or "").strip()
    age      = (character.get("age_range")  or "").strip()
    gender   = (character.get("gender")     or "").strip()
    ethnicity= (character.get("ethnicity")  or "").strip()
    complexion=(character.get("complexion") or "").strip()
    wardrobe = (character.get("wardrobe")   or "").strip()
    grooming = (character.get("grooming")   or "").strip()
    cultural = (character.get("cultural_notes") or "").strip()

    # Pronoun based on gender
    gender_low = gender.lower()
    if gender_low in {"female", "woman", "girl"}:
        pronoun = "She"
    elif gender_low in {"male", "man", "boy"}:
        pronoun = "He"
    else:
        pronoun = "They"

    # ── Who they are ────────────────────────────────────────────────────────
    identity_parts = [p for p in [age, ethnicity, gender_low if gender_low not in {"male","female"} else ("woman" if gender_low=="female" else "man")]
                      if p and p.lower() not in {"unclear","unknown","any"}]
    identity = ", ".join(identity_parts) if identity_parts else "adult"
    role_clause = f", carrying the emotional weight of a {role.lower()}" if role and role.lower() not in {"unclear","unknown"} else ""
    who_line = f"A {identity}{role_clause}."

    # ── Complexion ──────────────────────────────────────────────────────────
    complexion_line = (
        f"{pronoun} has a {complexion.lower()} complexion." if complexion else ""
    )

    # ── Wardrobe and grooming in natural prose ───────────────────────────────
    if wardrobe and grooming:
        look_line = f"{pronoun} wears {wardrobe.lower()}, with {grooming.lower()}."
    elif wardrobe:
        look_line = f"{pronoun} wears {wardrobe.lower()}."
    elif grooming:
        look_line = f"{pronoun} has {grooming.lower()}."
    else:
        look_line = ""

    # ── Cultural world ───────────────────────────────────────────────────────
    world_parts = [p for p in [cultural, location_dna] if p and p.lower() not in {"universal",""}]
    world = world_parts[0] if world_parts else ""
    world_line = f"{pronoun} belongs to the world of {world}." if world else ""

    parts = [who_line, complexion_line, look_line, world_line, CHARACTER_REF_HINT + "."]
    return " ".join(" ".join(p.split()) for p in parts if p)


def generate_character_plate(character: dict, project_id: str,
                              location_dna: str = "Universal",
                              prompt_override: Optional[str] = None) -> tuple[str, str]:
    """Generate one identity plate for a character row.

    Returns (public_r2_url, prompt_used).
    """
    prompt = (prompt_override or build_character_plate_prompt(character, location_dna)).strip()
    if not prompt:
        raise ImageGenerationError("Empty character plate prompt.")
    safe = "".join(ch if ch.isalnum() else "_" for ch in (character.get("name") or "char"))[:32]
    logger.info("Generating character plate for project=%s name=%s mode=%s",
                project_id, character.get("name"), _resolve_ref_mode())

    if _resolve_ref_mode() == "cheap":
        img_bytes = _openai_generate(prompt[:4000], size=OPENAI_SIZE_PORTRAIT)
        r2_key = _new_ref_key(project_id, f"char_{character.get('id','x')}_{safe}", ext="png")
        return _save_bytes_to_r2(img_bytes, r2_key), prompt

    result = _run_fal(REF_MODEL, {
        "prompt": prompt[:1800],
        "image_size": "portrait_4_3",
        "num_inference_steps": 8,
        "num_images": 1,
        "seed": random.randint(1, 2**32 - 1),
        "enable_safety_checker": False,
    })
    fal_url = _extract_image_url(result)
    r2_key = _new_ref_key(project_id, f"char_{character.get('id','x')}_{safe}")
    return _save_to_r2(fal_url, r2_key), prompt


_TOD_PHRASES = {
    "golden_hour": "a golden-hour evening",
    "dawn":        "early morning",
    "morning":     "a gentle morning",
    "afternoon":   "a quiet afternoon",
    "dusk":        "a dusky evening",
    "evening":     "a warm evening",
    "night":       "a still night",
    "midnight":    "deep night",
}


def build_location_plate_prompt(location: dict) -> str:
    """Build a director-style cinematic prompt for a location.

    Rather than a structured field checklist (which over-constrains image
    models and produces identical-looking scenes), this writes flowing prose
    that gives the model a cultural/geographic anchor and scene mood, then
    lets it compose cinematically.  The architecture_style field is
    intentionally omitted — it is the same world-DNA string for every
    location in a project, so including it forces identical architectural
    elements into every scene.
    """
    name     = (location.get("name")                  or "").strip()
    desc     = (location.get("description")            or "").strip()
    geo      = (location.get("geography")              or "").strip()
    weather  = (location.get("weather_or_atmosphere")  or "").strip()
    tod_raw  = (location.get("time_of_day")            or "").strip()
    cultural = (location.get("cultural_notes")         or "").strip()
    social   = (location.get("social_layer")           or "").strip()

    # Clean props: visual_details often starts with the location name as a
    # copy-pasted prefix — strip it so only the actual prop list remains.
    raw_visual = (location.get("visual_details") or "").strip()
    if not raw_visual or raw_visual.lower() == name.lower():
        props = ""
    elif name and raw_visual.lower().startswith(name.lower()):
        props = raw_visual[len(name):].lstrip(";., ").strip()
    else:
        props = raw_visual

    # ── Opening line: time + cultural anchor ────────────────────────────────
    tod_phrase = _TOD_PHRASES.get(tod_raw.lower(), tod_raw or "a timeless moment")
    anchor_parts = [p for p in [cultural, geo] if p]
    anchor = ", ".join(anchor_parts)
    opening = f"{tod_phrase.capitalize()} in {anchor}." if anchor else f"{tod_phrase.capitalize()}."

    # ── Scene identity ───────────────────────────────────────────────────────
    # Use the narrative desc if available; fall back to the location name.
    scene_sentence = desc if desc else (f"{name}." if name else "")

    # ── Props woven in naturally ─────────────────────────────────────────────
    props_sentence = (
        f"The scene carries traces of everyday life — {props} — "
        f"present but unhurried, without people in frame."
        if props else
        "The space feels lived-in yet quietly empty, without people in frame."
    )

    # ── Atmospheric cue ──────────────────────────────────────────────────────
    if weather and weather.lower() not in {"unknown", "clear"}:
        atmo_sentence = (
            f"The air carries the quality of {weather.lower()}: "
            f"gentle, still, shaped by light and season."
        )
    else:
        atmo_sentence = "Natural light shapes the depth and mood of the scene."

    # ── Cinematic framing ────────────────────────────────────────────────────
    framing = (
        "The frame is an establishing wide-angle view, grounded and immersive, "
        "with natural light carving texture and depth into the environment."
    )

    # ── Assemble ─────────────────────────────────────────────────────────────
    parts = [opening, scene_sentence, props_sentence, atmo_sentence,
             framing, ENV_REF_HINT + "."]
    return " ".join(" ".join(p.split()) for p in parts if p)


def generate_location_plate(location: dict, project_id: str,
                             prompt_override: Optional[str] = None) -> tuple[str, str]:
    """Generate one environment plate for a location row.

    Returns (public_r2_url, prompt_used).
    """
    prompt = (prompt_override or build_location_plate_prompt(location)).strip()
    if not prompt:
        raise ImageGenerationError("Empty location plate prompt.")
    safe = "".join(ch if ch.isalnum() else "_" for ch in (location.get("name") or "loc"))[:32]
    logger.info("Generating location plate for project=%s name=%s mode=%s",
                project_id, location.get("name"), _resolve_ref_mode())

    if _resolve_ref_mode() == "cheap":
        img_bytes = _openai_generate(prompt[:4000], size=OPENAI_SIZE_LANDSCAPE)
        r2_key = _new_ref_key(project_id, f"loc_{location.get('id','x')}_{safe}", ext="png")
        return _save_bytes_to_r2(img_bytes, r2_key), prompt

    result = _run_fal(REF_MODEL, {
        "prompt": prompt[:1800],
        "image_size": "landscape_16_9",
        "num_inference_steps": 8,
        "num_images": 1,
        "seed": random.randint(1, 2**32 - 1),
        "enable_safety_checker": False,
    })
    fal_url = _extract_image_url(result)
    r2_key = _new_ref_key(project_id, f"loc_{location.get('id','x')}_{safe}")
    return _save_to_r2(fal_url, r2_key), prompt


def generate_environment_ref(location_dna: str, motifs: list, project_id: str) -> str:
    """Generate an environment plate and store it in R2.

    Returns a public R2 URL string (previously returned a local Path).
    """
    location = (location_dna or "Universal").strip()
    motif_hint = ", ".join((motifs or [])[:5])
    motif_part = f" Recurring motifs: {motif_hint}." if motif_hint else ""

    prompt = (
        f"Establishing environment reference plate for setting: {location}. "
        f"{motif_part} {ENV_REF_HINT}."
    )
    logger.info("Generating env ref for project=%s mode=%s", project_id, _resolve_ref_mode())

    if _resolve_ref_mode() == "cheap":
        img_bytes = _openai_generate(prompt, size=OPENAI_SIZE_LANDSCAPE)
        r2_key = _new_ref_key(project_id, "environment", ext="png")
        return _save_bytes_to_r2(img_bytes, r2_key)

    result = _run_fal(REF_MODEL, {
        "prompt": prompt,
        "image_size": "landscape_16_9",
        "num_inference_steps": 8,
        "num_images": 1,
        "seed": random.randint(1, 2**32 - 1),
        "enable_safety_checker": False,
    })
    fal_url = _extract_image_url(result)
    r2_key = _new_ref_key(project_id, "environment")
    return _save_to_r2(fal_url, r2_key)


def generate_shot_still(
    shot: dict,
    character_ref_url: Optional[str],
    project_id: str,
    environment_ref_url: Optional[str] = None,
    *,
    character: Optional[dict] = None,
    location: Optional[dict] = None,
    user_override: Optional[str] = None,
) -> str:
    """Generate a per-shot still and store it in R2.

    The model receives a tight, composer-built prompt (60-180 words)
    derived from the shot's structured fields plus the linked
    character/location records — NOT the verbose styled_visual_prompt,
    which mixes director's notes with visual prose and was getting
    truncated mid-sentence at 1800 chars.

    When ``user_override`` is provided (the user hand-edited the prompt
    in the UI) it is used verbatim with only the cinematography prefix
    and quality boosters reattached.

    Args:
        character_ref_url: Public R2 URL of character reference (or None).
        environment_ref_url: Public R2 URL of environment reference (or None).
        character: Linked character record (drives subject description
            and identity-consistent wardrobe/grooming cues).
        location: Linked location record (drives setting description).
        user_override: User-edited prompt to use verbatim, or None.

    Returns a public R2 URL string.
    """
    from shot_prompt_composer import compose_image_prompt

    # Task #69 — locked cinematography rig prefix.
    cine_block = shot.get("cinematography") if isinstance(shot.get("cinematography"), dict) else None
    cine_prefix = ""
    if cine_block and cine_block.get("rig"):
        try:
            from cinematography_engine import lens_clause
            cine_prefix = (lens_clause(cine_block) or "").strip()
        except Exception:
            cine_prefix = ""

    expression_mode = (shot.get("expression_mode") or "environment").lower()
    has_human_focus = expression_mode in {"face", "body"}
    has_env = bool(environment_ref_url)
    has_char_ref = bool(character_ref_url) and has_human_focus

    prompt, negative_prompt = compose_image_prompt(
        shot,
        character=character,
        location=location,
        has_character_ref=has_char_ref,
        has_environment_ref=has_env,
        user_override=user_override,
        cine_prefix=cine_prefix,
    )

    if not prompt.strip():
        raise ImageGenerationError(f"Shot {shot.get('shot_index')} has no prompt to render.")

    logger.info(
        "Composed prompt for shot %s (%d chars, char_ref=%s, env_ref=%s, user_edit=%s)",
        shot.get("shot_index"), len(prompt), has_char_ref, has_env, bool(user_override),
    )

    shot_idx = shot.get("shot_index") or shot.get("timeline_index") or "x"
    shot_mode = _resolve_shot_mode()
    logger.info("Rendering shot %s mode=%s char_ref=%s env_ref=%s",
                shot_idx, shot_mode, has_char_ref, has_env)

    # ── Cheap mode: gpt-image-1.5 low quality ($0.009–$0.013/image) ───────────
    # gpt-image-1.5 img2img: character ref preserves face identity, env ref
    # grounds the scene atmosphere. Multi-image edit passes both simultaneously.
    if shot_mode == "cheap":
        r2_key = _new_shot_key(project_id, shot_idx, ext="png")
        if has_char_ref and has_env:
            # Multi-image edit: pass character plate (face identity) +
            # environment plate (scene atmosphere) simultaneously.
            # gpt-image-1.5 conditions on both in a single generation call.
            img_bytes = _openai_edit_multi(
                prompt,
                [character_ref_url, environment_ref_url],
                size=OPENAI_SIZE_LANDSCAPE,
            )
        elif has_char_ref:
            img_bytes = _openai_edit(prompt, character_ref_url, size=OPENAI_SIZE_LANDSCAPE)
        elif has_env:
            img_bytes = _openai_edit(prompt, environment_ref_url, size=OPENAI_SIZE_LANDSCAPE)
        else:
            img_bytes = _openai_generate(prompt, size=OPENAI_SIZE_LANDSCAPE)
        return _save_bytes_to_r2(img_bytes, r2_key)

    # ── Standard mode: FLUX Schnell (~$0.003–0.005/image) ────────────────────
    # Cinematic-quality stills using the same FLUX/schnell backbone as the
    # reference plate engine.  No face-lock — every shot is text-to-image with
    # the full styled prompt.  Fast (4 steps) and very cost-effective.
    # Best for landscape-heavy, nature, or abstract videos where face
    # consistency is less critical.
    # Note: sdxl_face is a legacy alias that normalises here at runtime.
    if shot_mode in ("standard", "sdxl_face"):
        result = _run_fal(STANDARD_SHOT_MODEL, {
            "prompt": prompt,
            "image_size": "landscape_16_9",
            "num_inference_steps": 4,
            "num_images": 1,
            "enable_safety_checker": False,
        })
        fal_url = _extract_image_url(result)
        r2_key = _new_shot_key(project_id, shot_idx)
        return _save_to_r2(fal_url, r2_key)

    # ── Quality mode: FAL FLUX + PuLID (current production pipeline) ─────────
    if has_char_ref:
        fal_char_url = _fal_accessible_url(character_ref_url)
        result = _run_fal(SHOT_MODEL_FACE, {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "reference_image_url": fal_char_url,
            "image_size": "landscape_16_9",
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "num_images": 1,
            "true_cfg": 1.0,
            "id_weight": 1.0,
            "enable_safety_checker": False,
        })
    elif has_env:
        fal_env_url = _fal_accessible_url(environment_ref_url)
        result = _run_fal(SHOT_MODEL_ENV_I2I, {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_url": fal_env_url,
            "strength": 0.85,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": False,
        })
    else:
        result = _run_fal(SHOT_MODEL_NO_FACE, {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": "landscape_16_9",
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": False,
        })

    fal_url = _extract_image_url(result)
    r2_key = _new_shot_key(project_id, shot_idx)
    return _save_to_r2(fal_url, r2_key)
