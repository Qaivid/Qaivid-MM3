from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import asyncio
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / '.env')

from models import (
    Project, ProjectCreate, SourceInput, SourceInputCreate,
    ContextPacket, UserOverride, UserOverrideCreate,
    Scene, Shot, PromptVariant, ValidationResult, now_utc,
    CharacterProfile, CharacterProfileCreate,
    EnvironmentProfile, EnvironmentProfileCreate,
)
from services.input_parser import parse_input
from services.culture_packs import detect_culture_pack, get_culture_pack, CULTURE_PACKS
from services.context_engine import build_context_packet
from services.scene_engine import build_scenes
from services.shot_engine import build_shots_for_scene
from services.prompt_engine import build_all_prompts, get_available_models
from services.validation_engine import validate_project
from services.export_service import export_json, export_csv_shots, export_prompt_list, export_storyboard
from services.continuity_engine import build_continuity_report
from services.pre_enrichment import pre_enrich_lines, build_pre_enrichment_context
from services.creative_brief import generate_creative_brief
from services.vibe_presets import get_vibe_preset, list_vibe_presets, VIBE_PRESETS
from services.production_pipeline import build_reference_prompts, build_still_prompts, build_render_plan, build_timeline
from services.image_generation import generate_reference_image, generate_still_image
from services.audio_transcription import transcribe_audio
from services.video_generation import submit_shot_render, check_and_download
from services.provider_registry import (
    DEFAULT_SETTINGS, get_image_provider, get_video_provider,
    list_image_providers, list_video_providers,
)
from services.assembly_engine import assemble_video
from services.secrets_manager import get_secret
from services.auth import (
    hash_password, verify_password, create_access_token, create_refresh_token,
    set_auth_cookies, clear_auth_cookies, get_current_user, require_admin,
    check_brute_force, record_failed_attempt, clear_failed_attempts,
    seed_admin, format_user,
)
from services.billing import (
    OPERATION_COSTS, PLAN_CREDIT_LIMITS, PLAN_LABELS,
    get_user_credits, charge_credits, add_credits, reset_credits,
    get_credit_ledger, get_plan_limit,
)

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI(title="Qaivid 2.0 API", version="2.0.0")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    await seed_admin(db)


# ─── Health ──────────────────────────────────────────────
@api.get("/")
async def root():
    return {"message": "Qaivid 2.0 API", "version": "2.0.0"}


# ─── Auth ────────────────────────────────────────────────
@api.post("/auth/register")
async def register(request: Request):
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password", "")
    name = body.get("name", "")
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(409, "Email already registered")

    plan = "free"
    doc = {
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "role": "user",
        "plan": plan,
        "credit_balance": get_plan_limit(plan),
        "video_generation_enabled": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = await db.users.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    user_out = format_user(doc)

    access = create_access_token(user_out["id"], email, "user")
    refresh = create_refresh_token(user_out["id"])
    response = JSONResponse(content=user_out)
    set_auth_cookies(response, access, refresh)
    return response


@api.post("/auth/login")
async def login(request: Request):
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password", "")
    if not email or not password:
        raise HTTPException(400, "Email and password required")

    ip = request.client.host if request.client else "unknown"
    identifier = f"{ip}:{email}"
    await check_brute_force(db, identifier)

    user = await db.users.find_one({"email": email})
    if not user or not verify_password(password, user.get("password_hash", "")):
        await record_failed_attempt(db, identifier)
        raise HTTPException(401, "Invalid email or password")

    await clear_failed_attempts(db, identifier)
    user_id = str(user["_id"])
    user_out = format_user(user)

    access = create_access_token(user_id, email, user.get("role", "user"))
    refresh = create_refresh_token(user_id)
    response = JSONResponse(content=user_out)
    set_auth_cookies(response, access, refresh)
    return response


@api.get("/auth/me")
async def auth_me(request: Request):
    user = await get_current_user(request, db)
    credits = await get_user_credits(db, user["id"])
    return {**user, **credits}


@api.post("/auth/logout")
async def logout():
    response = JSONResponse(content={"logged_out": True})
    clear_auth_cookies(response)
    return response


@api.post("/auth/refresh")
async def refresh_token(request: Request):
    import jwt as pyjwt
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(401, "No refresh token")
    try:
        payload = pyjwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(401, "User not found")
        user_id = str(user["_id"])
        access = create_access_token(user_id, user["email"], user.get("role", "user"))
        response = JSONResponse(content={"refreshed": True})
        response.set_cookie(key="access_token", value=access, httponly=True, secure=True, samesite="none", max_age=3600, path="/")
        return response
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid refresh token")


# ─── Admin Routes ────────────────────────────────────────
@api.get("/admin/stats")
async def admin_stats(request: Request):
    await require_admin(request, db)
    user_count = await db.users.count_documents({})
    project_count = await db.projects.count_documents({})
    plan_counts = {}
    for plan in PLAN_CREDIT_LIMITS:
        plan_counts[plan] = await db.users.count_documents({"plan": plan})
    return {"users": user_count, "projects": project_count, "plans": plan_counts}


@api.get("/admin/users")
async def admin_list_users(request: Request):
    await require_admin(request, db)
    users = await db.users.find({}, {"password_hash": 0}).to_list(500)
    for u in users:
        u["id"] = str(u["_id"])
        del u["_id"]
        u["project_count"] = await db.projects.count_documents({"user_id": u["id"]})
    return users


@api.get("/admin/users/{user_id}")
async def admin_get_user(request: Request, user_id: str):
    await require_admin(request, db)
    user = await db.users.find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})
    if not user:
        raise HTTPException(404, "User not found")
    user["id"] = str(user["_id"])
    del user["_id"]
    projects = await db.projects.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    ledger = await get_credit_ledger(db, user_id, 50)
    return {"user": user, "projects": projects, "credit_ledger": ledger}


@api.put("/admin/users/{user_id}")
async def admin_update_user(request: Request, user_id: str):
    await require_admin(request, db)
    body = await request.json()
    allowed = {"plan", "credit_balance", "video_generation_enabled", "name", "role"}
    update = {k: v for k, v in body.items() if k in allowed}
    if "plan" in update and update["plan"] not in PLAN_CREDIT_LIMITS:
        raise HTTPException(400, f"Invalid plan. Must be one of: {list(PLAN_CREDIT_LIMITS.keys())}")
    if not update:
        raise HTTPException(400, "No valid fields to update")
    result = await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(404, "User not found")
    user = await db.users.find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})
    user["id"] = str(user["_id"])
    del user["_id"]
    return user


@api.delete("/admin/users/{user_id}")
async def admin_delete_user(request: Request, user_id: str):
    await require_admin(request, db)
    await db.users.delete_one({"_id": ObjectId(user_id)})
    await db.projects.delete_many({"user_id": user_id})
    return {"deleted": True}


@api.post("/admin/users/{user_id}/reset-credits")
async def admin_reset_credits(request: Request, user_id: str):
    await require_admin(request, db)
    result = await reset_credits(db, user_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@api.post("/admin/users/{user_id}/add-credits")
async def admin_add_credits(request: Request, user_id: str):
    await require_admin(request, db)
    body = await request.json()
    amount = body.get("amount", 0)
    if amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    await add_credits(db, user_id, amount, "admin-add")
    credits = await get_user_credits(db, user_id)
    return credits


@api.get("/admin/projects")
async def admin_list_projects(request: Request):
    await require_admin(request, db)
    projects = await db.projects.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    user_ids = list(set(p.get("user_id", "") for p in projects if p.get("user_id")))
    users = {}
    for uid in user_ids:
        try:
            u = await db.users.find_one({"_id": ObjectId(uid)}, {"_id": 0, "email": 1, "name": 1})
            if u:
                users[uid] = u
        except Exception:
            pass
    for p in projects:
        uid = p.get("user_id", "")
        p["owner_email"] = users.get(uid, {}).get("email", "")
        p["owner_name"] = users.get(uid, {}).get("name", "")
    return projects


@api.get("/admin/billing-config")
async def admin_billing_config(request: Request):
    await require_admin(request, db)
    return {"plans": PLAN_CREDIT_LIMITS, "plan_labels": PLAN_LABELS, "operation_costs": OPERATION_COSTS}


# ─── Admin: Platform Settings (provider selection) ───────

async def _get_platform_settings() -> dict:
    """Load platform settings from DB, merging with defaults."""
    doc = await db.platform_settings.find_one({"_id": "global"}, {"_id": 0})
    return {**DEFAULT_SETTINGS, **(doc or {})}


@api.get("/admin/settings")
async def get_admin_settings(request: Request):
    await require_admin(request, db)
    settings = await _get_platform_settings()
    return {
        "settings": settings,
        "image_providers": list_image_providers(),
        "video_providers": list_video_providers(),
    }


@api.patch("/admin/settings")
async def update_admin_settings(request: Request):
    await require_admin(request, db)
    body = await request.json()
    allowed_keys = {"image_provider_references", "image_provider_stills", "video_provider"}
    updates = {k: v for k, v in body.items() if k in allowed_keys}
    if not updates:
        raise HTTPException(400, "No valid settings keys provided")

    # Validate providers exist
    for key, val in updates.items():
        if "video" in key:
            if not get_video_provider(val):
                raise HTTPException(400, f"Unknown video provider: {val}")
        else:
            if not get_image_provider(val):
                raise HTTPException(400, f"Unknown image provider: {val}")

    await db.platform_settings.update_one(
        {"_id": "global"},
        {"$set": {**updates, "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )
    return {"message": "Settings saved", "settings": await _get_platform_settings()}


# ─── Projects ────────────────────────────────────────────
@api.post("/projects")
async def create_project(request: Request, data: ProjectCreate):
    user = await get_current_user(request, db)
    project = Project(
        name=data.name, description=data.description,
        input_mode=data.input_mode, language=data.language,
        culture_pack=data.culture_pack, settings=data.settings,
    )
    doc = project.model_dump()
    doc["user_id"] = user["id"]
    await db.projects.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.get("/projects")
async def list_projects(request: Request):
    user = await get_current_user(request, db)
    projects = await db.projects.find({"user_id": user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return projects


@api.get("/projects/{project_id}")
async def get_project(project_id: str, request: Request):
    project = await _require_project_access(request, project_id)
    return project


@api.put("/projects/{project_id}")
async def update_project(project_id: str, data: ProjectCreate, request: Request):
    await _require_project_access(request, project_id)
    update = data.model_dump()
    update["updated_at"] = now_utc()
    result = await db.projects.update_one({"id": project_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(404, "Project not found")
    return await db.projects.find_one({"id": project_id}, {"_id": 0})


@api.delete("/projects/{project_id}")
async def delete_project(project_id: str, request: Request):
    await _require_project_access(request, project_id)
    await db.projects.delete_one({"id": project_id})
    for col in ["source_inputs", "context_packets", "user_overrides", "scenes", "shots", "prompt_variants", "character_profiles", "environment_profiles"]:
        await db[col].delete_many({"project_id": project_id})
    return {"deleted": True}


# ─── Source Input ────────────────────────────────────────
@api.post("/projects/{project_id}/input")
async def add_input(project_id: str, data: SourceInputCreate, request: Request):
    project = await _require_project_access(request, project_id)

    # Parse input deterministically
    parsed = parse_input(data.raw_text)

    si = SourceInput(
        project_id=project_id,
        raw_text=data.raw_text,
        cleaned_text=parsed["cleaned_text"],
        detected_language=parsed["detected_language"],
        detected_script=parsed["detected_script"],
        detected_type=parsed["detected_type"],
        sections=parsed["sections"],
        lines=parsed["lines"],
        line_count=parsed["line_count"],
        metadata=parsed["metadata"],
    )
    doc = si.model_dump()

    # Upsert — one input per project
    await db.source_inputs.delete_many({"project_id": project_id})
    await db.source_inputs.insert_one(doc)
    doc.pop("_id", None)

    # Auto-detect culture pack
    culture = data.culture_hint if data.culture_hint != "auto" else detect_culture_pack(
        parsed["cleaned_text"], parsed["detected_language"]
    )
    lang = data.language_hint if data.language_hint != "auto" else parsed["detected_language"]

    await db.projects.update_one({"id": project_id}, {"$set": {
        "status": "input_added", "updated_at": now_utc(),
        "culture_pack": culture, "language": lang,
    }})

    return doc


@api.get("/projects/{project_id}/input")
async def get_input(project_id: str, request: Request):
    await _require_project_access(request, project_id)
    si = await db.source_inputs.find_one({"project_id": project_id}, {"_id": 0})
    if not si:
        raise HTTPException(404, "No input found")
    return si


# ─── Auto-Brief Helper ───────────────────────────────────
async def _auto_generate_brief(project_id: str, ctx: dict, project: dict, api_key: str):
    """
    Fires automatically after interpret succeeds.
    Generates the Creative Brief (characters + locations) from the ContextPacket
    so the Production Pipeline always has a visual cast ready.
    """
    try:
        vibe_preset = project.get("settings", {}).get("vibe_preset", "")
        brief = await generate_creative_brief(project_id, ctx, project, vibe_preset or None, api_key=api_key)
        await db.creative_briefs.delete_many({"project_id": project_id})
        await db.creative_briefs.insert_one({**brief})
        brief.pop("_id", None)
        from models import CharacterProfile, EnvironmentProfile
        for char in brief.get("characters", []):
            existing = await db.character_profiles.find_one({"project_id": project_id, "name": char.get("name", "")})
            if not existing:
                cp = CharacterProfile(
                    project_id=project_id, name=char.get("name", ""),
                    role=char.get("role", ""), description=char.get("emotional_arc", ""),
                    appearance=char.get("physical_description", char.get("physicalDescription", "")),
                    wardrobe=char.get("wardrobe", ""), age_range=char.get("age", ""),
                )
                await db.character_profiles.insert_one(cp.model_dump())
        for loc in brief.get("locations", []):
            existing = await db.environment_profiles.find_one({"project_id": project_id, "name": loc.get("name", "")})
            if not existing:
                ep = EnvironmentProfile(
                    project_id=project_id, name=loc.get("name", ""),
                    description=loc.get("description", ""),
                    time_of_day=loc.get("time_of_day", loc.get("timeOfDay", "")),
                    mood=loc.get("mood", ""),
                    visual_details=loc.get("visual_details", loc.get("visualDetails", "")),
                )
                await db.environment_profiles.insert_one(ep.model_dump())
        logger.info(f"Auto-generated creative brief for project {project_id}: {len(brief.get('characters', []))} chars, {len(brief.get('locations', []))} locs")
    except Exception as e:
        logger.error(f"Auto-brief generation failed for project {project_id}: {e}")


async def _auto_build_scenes(project_id: str, ctx: dict, si: dict):
    """
    Fires automatically after interpret succeeds — in parallel with _auto_generate_brief.
    build_scenes is deterministic (no LLM), so this completes in ~1-2s.
    By the time the user finishes reading the Creative Brief, scenes are ready.
    """
    try:
        scenes = build_scenes(project_id, ctx, si)
        await db.scenes.delete_many({"project_id": project_id})
        if scenes:
            await db.scenes.insert_many([{**s} for s in scenes])
        await db.projects.update_one({"id": project_id}, {"$set": {"status": "scenes_built", "updated_at": now_utc()}})
        logger.info(f"Auto-built {len(scenes)} scenes for project {project_id}")
    except Exception as e:
        logger.error(f"Auto-scene build failed for project {project_id}: {e}")


# ─── Context / Interpretation ────────────────────────────
@api.post("/projects/{project_id}/interpret")
async def interpret_project(project_id: str, request: Request, background_tasks: BackgroundTasks):
    project = await _require_project_access(request, project_id)

    si = await db.source_inputs.find_one({"project_id": project_id}, {"_id": 0})
    if not si:
        raise HTTPException(400, "No input text found. Add input first.")

    await db.projects.update_one({"id": project_id}, {"$set": {"status": "interpreting", "updated_at": now_utc()}})

    try:
        # Pre-enrich lines deterministically before LLM call
        enriched_lines = pre_enrich_lines(si.get("lines", []), project.get("culture_pack", ""))
        pre_hints = build_pre_enrichment_context(enriched_lines)

        # Fetch user's API key
        api_key = await get_secret(db, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured. Go to Settings to add your key.")

        # Pull audio-derived hints (vocal gender, age) if this project came from an audio upload
        audio_hints = None
        audio_doc = await db.audio_transcriptions.find_one({"project_id": project_id}, {"_id": 0})
        if audio_doc:
            audio_hints = {
                "vocal_gender": audio_doc.get("vocal_gender"),
                "vocal_age_range": audio_doc.get("vocal_age_range"),
            }

        context_data = await build_context_packet(
            project_id=project_id,
            cleaned_text=si["cleaned_text"],
            lines=enriched_lines,
            detected_type=si.get("detected_type", project.get("input_mode", "song")),
            detected_language=project.get("language", "auto"),
            culture_pack_id=project.get("culture_pack", "generic_english"),
            user_settings=project.get("settings", {}),
            pre_enrichment_hints=pre_hints,
            api_key=api_key,
            audio_hints=audio_hints,
        )

        # Apply existing user overrides
        overrides = await db.user_overrides.find({"project_id": project_id}, {"_id": 0}).to_list(100)
        for ov in overrides:
            if ov.get("locked"):
                _apply_override(context_data, ov["field_path"], ov["override_value"])

        await db.context_packets.delete_many({"project_id": project_id})
        await db.context_packets.insert_one(context_data)
        context_data.pop("_id", None)

        await db.projects.update_one({"id": project_id}, {"$set": {"status": "interpreted", "updated_at": now_utc()}})

        # Return the context packet. Creative brief and scene generation are
        # triggered manually by the user from their respective tabs.
        return context_data

    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        await db.projects.update_one({"id": project_id}, {"$set": {"status": "input_added", "updated_at": now_utc()}})
        raise HTTPException(500, f"Interpretation failed: {str(e)}")


@api.get("/projects/{project_id}/context")
async def get_context(project_id: str, request: Request):
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    if not ctx:
        raise HTTPException(404, "No context packet. Run interpretation first.")
    return ctx


# ─── User Overrides ──────────────────────────────────────
@api.post("/projects/{project_id}/overrides")
async def add_override(project_id: str, data: UserOverrideCreate, request: Request):
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    if not ctx:
        raise HTTPException(400, "No context packet. Run interpretation first.")

    original = _get_nested(ctx, data.field_path)
    ov = UserOverride(
        project_id=project_id, field_path=data.field_path,
        original_value=original, override_value=data.override_value,
        locked=data.locked,
    )
    doc = ov.model_dump()
    await db.user_overrides.insert_one(doc)
    doc.pop("_id", None)

    # Apply to context packet
    if data.locked:
        _apply_override(ctx, data.field_path, data.override_value)
        ctx["locked_assumptions"][data.field_path] = data.override_value
        ctx["updated_at"] = now_utc()
        await db.context_packets.update_one({"project_id": project_id}, {"$set": ctx})

    return doc


@api.get("/projects/{project_id}/overrides")
async def get_overrides(project_id: str, request: Request):
    overrides = await db.user_overrides.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    return overrides


@api.delete("/projects/{project_id}/overrides/{override_id}")
async def delete_override(project_id: str, override_id: str, request: Request):
    await db.user_overrides.delete_one({"id": override_id, "project_id": project_id})
    return {"deleted": True}


# ─── Scenes ──────────────────────────────────────────────
@api.post("/projects/{project_id}/scenes/build")
async def build_project_scenes(project_id: str, request: Request):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    if not ctx:
        raise HTTPException(400, "No context packet. Run interpretation first.")

    si = await db.source_inputs.find_one({"project_id": project_id}, {"_id": 0})
    if not si:
        raise HTTPException(400, "No source input found.")

    scenes = build_scenes(project_id, ctx, si)

    await db.scenes.delete_many({"project_id": project_id})
    if scenes:
        await db.scenes.insert_many([{**s} for s in scenes])

    # Clean _id from returned data
    for s in scenes:
        s.pop("_id", None)

    await db.projects.update_one({"id": project_id}, {"$set": {"status": "scenes_built", "updated_at": now_utc()}})
    return scenes


@api.get("/projects/{project_id}/scenes")
async def get_scenes(project_id: str, request: Request):
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).sort("scene_number", 1).to_list(100)
    return scenes


@api.put("/projects/{project_id}/scenes/{scene_id}")
async def update_scene(project_id: str, scene_id: str, data: dict, request: Request):
    allowed_fields = ["purpose", "location", "time_of_day", "emotional_temperature",
                      "temporal_status", "objects_of_significance", "character_blocking",
                      "visual_motif_priority"]
    update = {k: v for k, v in data.items() if k in allowed_fields}
    if not update:
        raise HTTPException(400, "No valid fields to update")
    result = await db.scenes.update_one({"id": scene_id, "project_id": project_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(404, "Scene not found")
    scene = await db.scenes.find_one({"id": scene_id}, {"_id": 0})
    return scene


@api.put("/projects/{project_id}/scenes/reorder")
async def reorder_scenes(project_id: str, data: dict, request: Request):
    """Reorder scenes by providing ordered list of scene IDs."""
    scene_ids = data.get("scene_ids", [])
    if not scene_ids:
        raise HTTPException(400, "scene_ids required")
    for i, sid in enumerate(scene_ids):
        await db.scenes.update_one(
            {"id": sid, "project_id": project_id},
            {"$set": {"scene_number": i + 1}}
        )
    return await db.scenes.find({"project_id": project_id}, {"_id": 0}).sort("scene_number", 1).to_list(100)


# ─── Shots ───────────────────────────────────────────────
@api.post("/projects/{project_id}/shots/build")
async def build_project_shots(project_id: str, request: Request):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    if not ctx:
        raise HTTPException(400, "No context packet. Run interpretation first.")

    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).sort("scene_number", 1).to_list(100)
    if not scenes:
        raise HTTPException(400, "No scenes found. Build scenes first.")

    all_shots = []
    content_type = project.get("input_mode", "song")
    for scene in scenes:
        shots = build_shots_for_scene(scene, ctx, project.get("settings", {}), content_type)
        all_shots.extend(shots)

    await db.shots.delete_many({"project_id": project_id})
    if all_shots:
        await db.shots.insert_many([{**s} for s in all_shots])

    for s in all_shots:
        s.pop("_id", None)

    await db.projects.update_one({"id": project_id}, {"$set": {"status": "shots_built", "updated_at": now_utc()}})
    return all_shots


@api.get("/projects/{project_id}/shots")
async def get_shots(project_id: str, request: Request):
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).sort("shot_number", 1).to_list(500)
    return shots


@api.put("/projects/{project_id}/shots/{shot_id}")
async def update_shot(project_id: str, shot_id: str, data: dict, request: Request):
    allowed = ["visual_priority", "shot_type", "camera_height", "camera_behavior",
               "subject_action", "emotional_micro_state", "light_description",
               "secondary_objects", "motion_constraints", "negative_constraints",
               "duration_hint"]
    update = {k: v for k, v in data.items() if k in allowed}
    if not update:
        raise HTTPException(400, "No valid fields to update")
    result = await db.shots.update_one({"id": shot_id, "project_id": project_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(404, "Shot not found")
    shot = await db.shots.find_one({"id": shot_id}, {"_id": 0})
    return shot


@api.put("/projects/{project_id}/shots/reorder")
async def reorder_shots(project_id: str, data: dict, request: Request):
    """Reorder shots by providing ordered list of shot IDs."""
    shot_ids = data.get("shot_ids", [])
    if not shot_ids:
        raise HTTPException(400, "shot_ids required")
    for i, sid in enumerate(shot_ids):
        await db.shots.update_one(
            {"id": sid, "project_id": project_id},
            {"$set": {"shot_number": i + 1}}
        )
    return await db.shots.find({"project_id": project_id}, {"_id": 0}).sort("shot_number", 1).to_list(500)


# ─── Prompts ─────────────────────────────────────────────
@api.post("/projects/{project_id}/prompts/build")
async def build_project_prompts(project_id: str, request: Request, model_target: str = "generic"):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)

    if not shots:
        raise HTTPException(400, "No shots found. Build shots first.")

    # Load character and environment profiles for prompt injection
    characters = await db.character_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    environments = await db.environment_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)

    # Load creative brief and vibe preset for inheritance
    brief = await db.creative_briefs.find_one({"project_id": project_id}, {"_id": 0})
    vibe_id = project.get("settings", {}).get("vibe_preset", "")
    vibe = get_vibe_preset(vibe_id) if vibe_id else None

    prompts = build_all_prompts(
        shots, scenes, ctx or {}, project.get("settings", {}), model_target,
        characters=characters if characters else None,
        environments=environments if environments else None,
        project=project,
        creative_brief=brief,
        vibe_preset=vibe,
    )

    await db.prompt_variants.delete_many({"project_id": project_id})
    if prompts:
        await db.prompt_variants.insert_many([{**p} for p in prompts])

    for p in prompts:
        p.pop("_id", None)

    await db.projects.update_one({"id": project_id}, {"$set": {"status": "prompts_ready", "updated_at": now_utc()}})
    return prompts


@api.get("/projects/{project_id}/prompts")
async def get_prompts(project_id: str, request: Request):
    prompts = await db.prompt_variants.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    return prompts


# ─── Validation ──────────────────────────────────────────
@api.get("/projects/{project_id}/validate")
async def validate(project_id: str, request: Request):
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0}) or {}
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    prompts = await db.prompt_variants.find({"project_id": project_id}, {"_id": 0}).to_list(500)

    result = validate_project(ctx, scenes, shots, prompts)
    result["project_id"] = project_id
    result["validated_at"] = now_utc()
    return result


# ─── Export ──────────────────────────────────────────────
@api.get("/projects/{project_id}/export/{format}")
async def export_project(project_id: str, format: str, request: Request):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0}) or {}
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0}) or {}
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    prompts = await db.prompt_variants.find({"project_id": project_id}, {"_id": 0}).to_list(500)

    if format == "json":
        return PlainTextResponse(export_json(project, ctx, scenes, shots, prompts), media_type="application/json")
    elif format == "csv":
        return PlainTextResponse(export_csv_shots(shots, prompts), media_type="text/csv")
    elif format == "prompts":
        return PlainTextResponse(export_prompt_list(prompts), media_type="text/plain")
    elif format == "storyboard":
        return PlainTextResponse(export_storyboard(scenes, shots), media_type="text/plain")
    else:
        raise HTTPException(400, f"Unknown format: {format}. Use: json, csv, prompts, storyboard")


# ─── Culture Packs ───────────────────────────────────────
@api.get("/culture-packs")
async def list_culture_packs():
    return [{"id": k, "name": v["name"], "description": v["description"]} for k, v in CULTURE_PACKS.items()]


@api.get("/culture-packs/{pack_id}")
async def get_culture_pack_detail(pack_id: str):
    pack = get_culture_pack(pack_id)
    if not pack:
        raise HTTPException(404, "Culture pack not found")
    return pack


# ─── Character Profiles ──────────────────────────────────
@api.post("/projects/{project_id}/characters")
async def create_character(project_id: str, data: CharacterProfileCreate, request: Request):
    char = CharacterProfile(project_id=project_id, **data.model_dump())
    doc = char.model_dump()
    await db.character_profiles.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.get("/projects/{project_id}/characters")
async def list_characters(project_id: str, request: Request):
    chars = await db.character_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    return chars


@api.put("/projects/{project_id}/characters/{char_id}")
async def update_character(project_id: str, char_id: str, data: dict, request: Request):
    allowed = ["name", "role", "description", "appearance", "age_range", "wardrobe", "emotional_range", "reference_image_url"]
    update = {k: v for k, v in data.items() if k in allowed}
    if not update:
        raise HTTPException(400, "No valid fields")
    await db.character_profiles.update_one({"id": char_id, "project_id": project_id}, {"$set": update})
    char = await db.character_profiles.find_one({"id": char_id}, {"_id": 0})
    return char


@api.delete("/projects/{project_id}/characters/{char_id}")
async def delete_character(project_id: str, char_id: str, request: Request):
    await db.character_profiles.delete_one({"id": char_id, "project_id": project_id})
    return {"deleted": True}


@api.post("/projects/{project_id}/characters/{char_id}/upload-reference")
async def upload_character_reference(project_id: str, char_id: str, request: Request, file: UploadFile = File(...)):
    upload_dir = "/app/backend/reference_images"
    os.makedirs(upload_dir, exist_ok=True)
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    filename = f"char_{char_id}.{ext}"
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    image_url = f"/api/reference-images/{filename}"
    await db.character_profiles.update_one({"id": char_id, "project_id": project_id}, {"$set": {"reference_image_url": image_url}})
    char = await db.character_profiles.find_one({"id": char_id}, {"_id": 0})
    return char


@api.post("/projects/{project_id}/characters/{char_id}/variants")
async def add_look_variant(project_id: str, char_id: str, data: dict, request: Request):
    """Add a look variant (wardrobe/appearance change) to a character."""
    variant = {
        "id": str(__import__('uuid').uuid4()),
        "label": data.get("label", "Variant"),
        "wardrobe": data.get("wardrobe", ""),
        "appearance_notes": data.get("appearance_notes", ""),
        "scene_ids": data.get("scene_ids", []),
        "reference_image_url": data.get("reference_image_url", ""),
    }
    result = await db.character_profiles.update_one(
        {"id": char_id, "project_id": project_id},
        {"$push": {"look_variants": variant}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Character not found")
    char = await db.character_profiles.find_one({"id": char_id}, {"_id": 0})
    return char


@api.delete("/projects/{project_id}/characters/{char_id}/variants/{variant_id}")
async def delete_look_variant(project_id: str, char_id: str, variant_id: str, request: Request):
    await db.character_profiles.update_one(
        {"id": char_id, "project_id": project_id},
        {"$pull": {"look_variants": {"id": variant_id}}}
    )
    char = await db.character_profiles.find_one({"id": char_id}, {"_id": 0})
    return char


# ─── Environment Profiles ────────────────────────────────
@api.post("/projects/{project_id}/environments")
async def create_environment(project_id: str, data: EnvironmentProfileCreate, request: Request):
    env = EnvironmentProfile(project_id=project_id, **data.model_dump())
    doc = env.model_dump()
    await db.environment_profiles.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api.get("/projects/{project_id}/environments")
async def list_environments(project_id: str, request: Request):
    envs = await db.environment_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    return envs


@api.put("/projects/{project_id}/environments/{env_id}")
async def update_environment(project_id: str, env_id: str, data: dict, request: Request):
    allowed = ["name", "description", "time_of_day", "mood", "visual_details", "architecture", "reference_image_url"]
    update = {k: v for k, v in data.items() if k in allowed}
    if not update:
        raise HTTPException(400, "No valid fields")
    await db.environment_profiles.update_one({"id": env_id, "project_id": project_id}, {"$set": update})
    env = await db.environment_profiles.find_one({"id": env_id}, {"_id": 0})
    return env


@api.delete("/projects/{project_id}/environments/{env_id}")
async def delete_environment(project_id: str, env_id: str, request: Request):
    await db.environment_profiles.delete_one({"id": env_id, "project_id": project_id})
    return {"deleted": True}


@api.post("/projects/{project_id}/environments/{env_id}/upload-reference")
async def upload_environment_reference(project_id: str, env_id: str, request: Request, file: UploadFile = File(...)):
    upload_dir = "/app/backend/reference_images"
    os.makedirs(upload_dir, exist_ok=True)
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    filename = f"env_{env_id}.{ext}"
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    image_url = f"/api/reference-images/{filename}"
    await db.environment_profiles.update_one({"id": env_id, "project_id": project_id}, {"$set": {"reference_image_url": image_url}})
    env = await db.environment_profiles.find_one({"id": env_id}, {"_id": 0})
    return env


# ─── Continuity ──────────────────────────────────────────
@api.get("/projects/{project_id}/continuity")
async def get_continuity(project_id: str, request: Request):
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0}) or {}
    characters = await db.character_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    environments = await db.environment_profiles.find({"project_id": project_id}, {"_id": 0}).to_list(50)

    if not scenes:
        raise HTTPException(400, "No scenes found. Build scenes first.")

    report = build_continuity_report(project_id, scenes, shots, ctx, characters, environments)
    return report


# ─── Model Adapters ──────────────────────────────────────
@api.get("/models")
async def list_models():
    return get_available_models()


# ─── Vibe Presets ────────────────────────────────────────
@api.get("/vibe-presets")
async def get_vibe_presets():
    return list_vibe_presets()


@api.get("/vibe-presets/{preset_id}")
async def get_vibe_detail(preset_id: str):
    vp = get_vibe_preset(preset_id)
    if not vp:
        raise HTTPException(404, "Vibe preset not found")
    return vp


# ─── Creative Brief ──────────────────────────────────────
@api.post("/projects/{project_id}/brief")
async def create_brief(project_id: str, request: Request, vibe_preset: str = ""):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0})
    if not ctx:
        raise HTTPException(400, "No context packet. Run intelligence first.")

    try:
        api_key = await get_secret(db, "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured. Go to Settings to add your key.")

        brief = await generate_creative_brief(project_id, ctx, project, vibe_preset or None, api_key=api_key)
        await db.creative_briefs.delete_many({"project_id": project_id})
        await db.creative_briefs.insert_one({**brief})
        brief.pop("_id", None)

        # Auto-create character and environment profiles from brief
        for char in brief.get("characters", []):
            existing = await db.character_profiles.find_one({"project_id": project_id, "name": char.get("name", "")})
            if not existing:
                from models import CharacterProfile
                cp = CharacterProfile(
                    project_id=project_id, name=char.get("name", ""),
                    role=char.get("role", ""), description=char.get("emotional_arc", ""),
                    appearance=char.get("physical_description", char.get("physicalDescription", "")),
                    wardrobe=char.get("wardrobe", ""), age_range=char.get("age", ""),
                )
                doc = cp.model_dump()
                await db.character_profiles.insert_one(doc)

        for loc in brief.get("locations", []):
            existing = await db.environment_profiles.find_one({"project_id": project_id, "name": loc.get("name", "")})
            if not existing:
                from models import EnvironmentProfile
                ep = EnvironmentProfile(
                    project_id=project_id, name=loc.get("name", ""),
                    description=loc.get("description", ""),
                    time_of_day=loc.get("time_of_day", loc.get("timeOfDay", "")),
                    mood=loc.get("mood", ""),
                    visual_details=loc.get("visual_details", loc.get("visualDetails", "")),
                )
                doc = ep.model_dump()
                await db.environment_profiles.insert_one(doc)

        return brief
    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        raise HTTPException(500, f"Brief generation failed: {str(e)}")


@api.get("/projects/{project_id}/brief")
async def get_brief(project_id: str, request: Request):
    brief = await db.creative_briefs.find_one({"project_id": project_id}, {"_id": 0})
    if not brief:
        raise HTTPException(404, "No creative brief. Generate one first.")
    return brief


# ─── Production Pipeline ─────────────────────────────────
@api.post("/projects/{project_id}/reference-prompts")
async def create_reference_prompts(project_id: str, request: Request):
    brief = await db.creative_briefs.find_one({"project_id": project_id}, {"_id": 0})
    if not brief:
        raise HTTPException(400, "No creative brief. Generate brief first.")
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0}) or {}
    project = await db.projects.find_one({"id": project_id}, {"_id": 0}) or {}
    vibe = get_vibe_preset(project.get("settings", {}).get("vibe_preset", "")) if project.get("settings") else None
    prompts = build_reference_prompts(brief, ctx, vibe)
    await db.reference_prompts.delete_many({"project_id": project_id})
    for p in prompts:
        p["project_id"] = project_id
    if prompts:
        await db.reference_prompts.insert_many([{**p} for p in prompts])
    for p in prompts:
        p.pop("_id", None)
    return prompts


@api.get("/projects/{project_id}/reference-prompts")
async def get_reference_prompts(project_id: str, request: Request):
    return await db.reference_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(50)


@api.post("/projects/{project_id}/still-prompts")
async def create_still_prompts(project_id: str, request: Request):
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    ctx = await db.context_packets.find_one({"project_id": project_id}, {"_id": 0}) or {}
    brief = await db.creative_briefs.find_one({"project_id": project_id}, {"_id": 0}) or {}
    project = await db.projects.find_one({"id": project_id}, {"_id": 0}) or {}
    vibe = get_vibe_preset(project.get("settings", {}).get("vibe_preset", "")) if project.get("settings") else None
    refs = await db.reference_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    prompts = build_still_prompts(shots, scenes, ctx, brief, vibe, refs)
    await db.still_prompts.delete_many({"project_id": project_id})
    for p in prompts:
        p["project_id"] = project_id
    if prompts:
        await db.still_prompts.insert_many([{**p} for p in prompts])
    for p in prompts:
        p.pop("_id", None)
    return prompts


@api.get("/projects/{project_id}/still-prompts")
async def get_still_prompts(project_id: str, request: Request):
    return await db.still_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(500)


@api.post("/projects/{project_id}/render-plan")
async def create_render_plan(project_id: str, request: Request, model: str = "wan_2_6"):
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    stills = await db.still_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    plan = build_render_plan(shots, stills, model)
    await db.render_plans.delete_many({"project_id": project_id})
    for p in plan:
        p["project_id"] = project_id
    if plan:
        await db.render_plans.insert_many([{**p} for p in plan])
    for p in plan:
        p.pop("_id", None)
    return plan


@api.get("/projects/{project_id}/render-plan")
async def get_render_plan(project_id: str, request: Request):
    return await db.render_plans.find({"project_id": project_id}, {"_id": 0}).to_list(500)


@api.post("/projects/{project_id}/timeline")
async def create_timeline(project_id: str, request: Request):
    scenes = await db.scenes.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    renders = await db.render_plans.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    tl = build_timeline(scenes, shots, renders)
    await db.timelines.delete_many({"project_id": project_id})
    tl["project_id"] = project_id
    await db.timelines.insert_one({**tl})
    tl.pop("_id", None)
    return tl


@api.get("/projects/{project_id}/timeline")
async def get_timeline(project_id: str, request: Request):
    tl = await db.timelines.find_one({"project_id": project_id}, {"_id": 0})
    if not tl:
        raise HTTPException(404, "No timeline.")
    return tl


# ─── Full Pipeline Status ────────────────────────────────
@api.get("/projects/{project_id}/pipeline")
async def get_pipeline_status(project_id: str, request: Request):
    """Get the full pipeline status showing which engines have output."""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    has_input = await db.source_inputs.count_documents({"project_id": project_id}) > 0
    has_context = await db.context_packets.count_documents({"project_id": project_id}) > 0
    has_brief = await db.creative_briefs.count_documents({"project_id": project_id}) > 0
    has_scenes = await db.scenes.count_documents({"project_id": project_id}) > 0
    has_shots = await db.shots.count_documents({"project_id": project_id}) > 0
    has_refs = await db.reference_prompts.count_documents({"project_id": project_id}) > 0
    has_prompts = await db.prompt_variants.count_documents({"project_id": project_id}) > 0
    has_stills = await db.still_prompts.count_documents({"project_id": project_id}) > 0
    has_renders = await db.render_plans.count_documents({"project_id": project_id}) > 0
    has_timeline = await db.timelines.count_documents({"project_id": project_id}) > 0

    engines = [
        {"key": "source", "label": "Source Content", "complete": has_input},
        {"key": "intelligence", "label": "Context Intelligence", "complete": has_context},
        {"key": "brief", "label": "Creative Brief", "complete": has_brief},
        {"key": "storyboard", "label": "Storyboard", "complete": has_scenes},
        {"key": "references", "label": "References", "complete": has_refs},
        {"key": "shots", "label": "Shot Plan", "complete": has_shots},
        {"key": "prompts", "label": "Prompts", "complete": has_prompts},
        {"key": "stills", "label": "Shot Stills", "complete": has_stills},
        {"key": "render", "label": "Video Render", "complete": has_renders},
        {"key": "assembly", "label": "Assembly", "complete": has_timeline},
    ]

    completed = sum(1 for e in engines if e["complete"])
    return {
        "project_id": project_id,
        "engines": engines,
        "completed": completed,
        "total": len(engines),
        "progress": round((completed / len(engines)) * 100),
    }


# ─── Audio Upload & Transcription ────────────────────────
@api.post("/projects/{project_id}/audio")
async def upload_audio(project_id: str, request: Request, file: UploadFile = File(...)):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")

    upload_dir = "/app/backend/uploaded_audio"
    os.makedirs(upload_dir, exist_ok=True)
    ext = file.filename.split(".")[-1] if "." in file.filename else "mp3"
    filepath = os.path.join(upload_dir, f"{project_id}.{ext}")

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Transcribe using dual-engine pipeline (Gemini + Whisper)
    try:
        gemini_key = await get_secret(db, "GEMINI_API_KEY")
        openai_key = await get_secret(db, "OPENAI_API_KEY")
        if not gemini_key:
            raise ValueError("Gemini API key not configured. Go to Settings to add your key.")

        result = await transcribe_audio(filepath, gemini_key, openai_key or "", project.get("language", "auto"))
        result["project_id"] = project_id
        result["audio_filepath"] = filepath
        result["audio_filename"] = file.filename
        await db.audio_transcriptions.delete_many({"project_id": project_id})
        await db.audio_transcriptions.insert_one({**result})
        result.pop("_id", None)

        # Helper: produce timing-enriched lines from parsed.lines + transcribed lines
        def _attach_timing(parsed_lines, transcribed_lines, total_dur):
            from services.audio_transcription import parse_ts as _parse_ts
            timed = []
            ti = 0
            for ln in parsed_lines:
                ln_out = dict(ln)
                if ln.get("text", "").strip() and ti < len(transcribed_lines):
                    t_line = transcribed_lines[ti]
                    start = _parse_ts(t_line.get("timestamp", "0:00.000"))
                    end = None
                    if ti + 1 < len(transcribed_lines):
                        end = _parse_ts(transcribed_lines[ti + 1].get("timestamp", "0:00.000"))
                    elif total_dur:
                        end = float(total_dur)
                    ln_out["start_time"] = start
                    if end is not None and end > start:
                        ln_out["end_time"] = end
                        ln_out["duration"] = round(end - start, 3)
                    ti += 1
                timed.append(ln_out)
            return timed

        existing_input = await db.source_inputs.find_one({"project_id": project_id}, {"_id": 0})
        transcribed = result.get("lines") or []
        total_dur = result.get("total_duration")

        if existing_input and transcribed:
            # Enrich existing source_input lines with timing (does not change the text)
            updated_lines = _attach_timing(existing_input.get("lines", []), transcribed, total_dur)
            new_meta = {**(existing_input.get("metadata") or {}), "has_audio_timing": True, "audio_total_duration": total_dur}
            await db.source_inputs.update_one(
                {"project_id": project_id},
                {"$set": {"lines": updated_lines, "metadata": new_meta}},
            )
        elif not existing_input and result.get("text"):
            # No source input yet — auto-create from transcribed text
            from services.input_parser import parse_input
            from models import SourceInput
            parsed = parse_input(result["text"])
            timed_lines = _attach_timing(parsed["lines"], transcribed, total_dur)
            si = SourceInput(
                project_id=project_id, raw_text=result["text"],
                cleaned_text=parsed["cleaned_text"],
                detected_language=parsed["detected_language"],
                detected_script=parsed["detected_script"],
                detected_type=parsed["detected_type"],
                sections=parsed["sections"], lines=timed_lines,
                line_count=parsed["line_count"],
                metadata={**parsed["metadata"], "has_audio_timing": True, "audio_total_duration": total_dur},
            )
            doc = si.model_dump()
            await db.source_inputs.insert_one(doc)
            await db.projects.update_one({"id": project_id}, {"$set": {"status": "input_added", "updated_at": now_utc()}})

        return result
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@api.get("/projects/{project_id}/audio")
async def get_audio(project_id: str, request: Request):
    t = await db.audio_transcriptions.find_one({"project_id": project_id}, {"_id": 0})
    if not t:
        raise HTTPException(404, "No audio transcription")
    return t


# ─── Helper: resolve API key for an image provider ───────

_IMAGE_PROVIDER_SECRET = {
    "gpt_image_1": "OPENAI_API_KEY",
    "dall_e_3":    "OPENAI_API_KEY",
    "flux_dev":    "FAL_API_KEY",
    "flux_schnell":"FAL_API_KEY",
}

async def _resolve_image_api_key(provider_id: str) -> str:
    secret_name = _IMAGE_PROVIDER_SECRET.get(provider_id, "OPENAI_API_KEY")
    key = await get_secret(db, secret_name)
    if not key:
        label = {"OPENAI_API_KEY": "OpenAI", "FAL_API_KEY": "fal.ai"}.get(secret_name, secret_name)
        raise HTTPException(400, f"{label} API key not configured. Go to Admin → Settings to add your key.")
    return key


# ─── Generate Reference Images ────────────────────────────
@api.post("/projects/{project_id}/generate-references")
async def generate_references(project_id: str, background_tasks: BackgroundTasks, request: Request):
    refs = await db.reference_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(50)
    if not refs:
        raise HTTPException(400, "No reference prompts. Build reference prompts first.")

    pending = [r for r in refs if r.get("status") != "completed"]
    if not pending:
        return {"message": "All references already generated", "total": len(refs)}

    platform = await _get_platform_settings()
    provider_id = platform.get("image_provider_references", "gpt_image_1")
    api_key = await _resolve_image_api_key(provider_id)

    sem = asyncio.Semaphore(5)

    async def _gen_ref(ref):
        async with sem:
            updated = await generate_reference_image(ref, api_key, provider=provider_id)
            await db.reference_prompts.update_one(
                {"id": ref["id"], "project_id": project_id},
                {"$set": {"status": updated["status"], "image_url": updated.get("image_url"), "error": updated.get("error"), "provider": provider_id}}
            )
            return updated

    results = await asyncio.gather(*[_gen_ref(r) for r in pending])
    return {
        "generated": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "total": len(refs),
        "provider": provider_id,
    }


# ─── Generate Shot Stills ─────────────────────────────────
@api.post("/projects/{project_id}/generate-stills")
async def generate_stills(project_id: str, request: Request):
    stills = await db.still_prompts.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    if not stills:
        raise HTTPException(400, "No still prompts. Build still prompts first.")

    pending = [s for s in stills if s.get("status") != "completed"]
    if not pending:
        return {"message": "All stills already generated", "total": len(stills)}

    platform = await _get_platform_settings()
    provider_id = platform.get("image_provider_stills", "gpt_image_1")
    api_key = await _resolve_image_api_key(provider_id)

    sem = asyncio.Semaphore(8)

    async def _gen_still(still):
        async with sem:
            updated = await generate_still_image(still, api_key, provider=provider_id)
            await db.still_prompts.update_one(
                {"id": still["id"], "project_id": project_id},
                {"$set": {"status": updated["status"], "image_url": updated.get("image_url"), "error": updated.get("error"), "provider": provider_id}}
            )
            return updated

    results = await asyncio.gather(*[_gen_still(s) for s in pending])
    return {
        "generated": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "total": len(stills),
        "provider": provider_id,
    }


# ─── Render Video Clips ───────────────────────────────────
@api.post("/projects/{project_id}/render-videos")
async def render_videos(project_id: str, request: Request):
    """Submit all pending shots for video generation concurrently. Returns immediately."""
    renders = await db.render_plans.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    shots = await db.shots.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    if not renders:
        raise HTTPException(400, "No render plan. Create render plan first.")

    api_key = await get_secret(db, "ATLAS_CLOUD_API_KEY")
    if not api_key:
        raise HTTPException(400, "AtlasCloud API key not configured. Go to Settings to add your key.")

    platform = await _get_platform_settings()
    video_provider_id = platform.get("video_provider", "wan_2_6")
    video_provider_cfg = get_video_provider(video_provider_id) or {}

    shot_map = {s["id"]: s for s in shots}
    pending = [r for r in renders if r.get("status") not in ("completed", "submitted", "processing")]
    if not pending:
        return {"message": "All videos already submitted or completed", "total": len(renders)}

    sem = asyncio.Semaphore(20)

    async def _submit(render):
        async with sem:
            shot = shot_map.get(render.get("shot_id", ""), {})
            updated = await submit_shot_render(render, shot, api_key, provider_config=video_provider_cfg)
            update_fields = {"status": updated["status"]}
            if updated.get("prediction_id"):
                update_fields["prediction_id"] = updated["prediction_id"]
                update_fields["submitted_at"] = updated.get("submitted_at")
                update_fields["provider"] = updated.get("provider")
                update_fields["provider_label"] = updated.get("provider_label")
                update_fields["model"] = updated.get("model")
            if updated.get("error"):
                update_fields["error"] = updated["error"]
            await db.render_plans.update_one(
                {"id": render["id"], "project_id": project_id},
                {"$set": update_fields}
            )
            return updated["status"]

    statuses = await asyncio.gather(*[_submit(r) for r in pending])
    return {
        "submitted": sum(1 for s in statuses if s == "submitted"),
        "failed": sum(1 for s in statuses if s == "failed"),
        "total": len(renders),
        "provider": video_provider_id,
    }


@api.get("/projects/{project_id}/render-status")
async def get_render_status(project_id: str, request: Request):
    """Check status of all submitted renders. Downloads completed videos."""
    renders = await db.render_plans.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    if not renders:
        raise HTTPException(404, "No render plans found.")

    api_key = await get_secret(db, "ATLAS_CLOUD_API_KEY")

    in_progress = [r for r in renders if r.get("status") in ("submitted", "processing")]
    completed_count = sum(1 for r in renders if r.get("status") == "completed")
    failed_count = sum(1 for r in renders if r.get("status") == "failed")
    pending_count = sum(1 for r in renders if r.get("status") not in ("completed", "failed", "submitted", "processing"))

    newly_completed = 0
    newly_failed = 0
    for render in in_progress:
        if not api_key:
            continue
        updated = await check_and_download(render, api_key)
        if updated["status"] != render.get("status"):
            update_fields = {"status": updated["status"]}
            if updated.get("output_video_url"):
                update_fields["output_video_url"] = updated["output_video_url"]
                update_fields["generated_at"] = updated.get("generated_at")
            if updated.get("error"):
                update_fields["error"] = updated["error"]
            await db.render_plans.update_one(
                {"id": render["id"], "project_id": project_id},
                {"$set": update_fields}
            )
            if updated["status"] == "completed":
                newly_completed += 1
                completed_count += 1
            elif updated["status"] == "failed":
                newly_failed += 1
                failed_count += 1

    still_processing = len(in_progress) - newly_completed - newly_failed

    return {
        "total": len(renders),
        "completed": completed_count,
        "processing": still_processing,
        "failed": failed_count,
        "pending": pending_count,
        "all_done": still_processing == 0 and pending_count == 0,
    }


# ─── Assemble Final Video ────────────────────────────────
@api.post("/projects/{project_id}/assemble")
async def assemble_final_video(project_id: str, request: Request):
    timeline = await db.timelines.find_one({"project_id": project_id}, {"_id": 0})
    if not timeline:
        raise HTTPException(400, "No timeline. Build timeline first.")

    # Check for audio
    audio = await db.audio_transcriptions.find_one({"project_id": project_id}, {"_id": 0})
    audio_path = audio.get("audio_filepath") if audio else None

    # Update timeline clips with rendered video URLs
    renders = await db.render_plans.find({"project_id": project_id}, {"_id": 0}).to_list(500)
    render_map = {r.get("shot_id", ""): r for r in renders}
    for clip in timeline.get("clips", []):
        render = render_map.get(clip.get("shot_id", ""))
        if render and render.get("output_video_url"):
            clip["video_url"] = render["output_video_url"]
            clip["status"] = "ready"

    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    project_name = (project.get("name", "qaivid") if project else "qaivid").replace(" ", "_").lower()

    try:
        result = assemble_video(timeline, audio_path, f"{project_name}_{project_id[:8]}.mp4")
        result["project_id"] = project_id
        await db.assemblies.delete_many({"project_id": project_id})
        await db.assemblies.insert_one({**result})
        result.pop("_id", None)
        await db.projects.update_one({"id": project_id}, {"$set": {"status": "complete", "updated_at": now_utc()}})
        return result
    except Exception as e:
        logger.error(f"Assembly failed: {e}")
        raise HTTPException(500, f"Assembly failed: {str(e)}")


@api.get("/projects/{project_id}/assembly")
async def get_assembly(project_id: str, request: Request):
    a = await db.assemblies.find_one({"project_id": project_id}, {"_id": 0})
    if not a:
        raise HTTPException(404, "No assembly")
    return a


# ─── Static File Serving ─────────────────────────────────
@api.get("/images/{filename}")
async def serve_image(filename: str):
    path = os.path.join("/app/backend/generated_images", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


@api.get("/reference-images/{filename}")
async def serve_reference_image(filename: str):
    path = os.path.join("/app/backend/reference_images", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    media = "image/png"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media = "image/jpeg"
    elif filename.endswith(".webp"):
        media = "image/webp"
    return FileResponse(path, media_type=media)


@api.get("/videos/{filename}")
async def serve_video(filename: str):
    path = os.path.join("/app/backend/generated_videos", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@api.get("/exports/{filename}")
async def serve_export(filename: str):
    path = os.path.join("/app/backend/assembled", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Export not found")
    return FileResponse(path, media_type="video/mp4")


# ─── Helpers ─────────────────────────────────────────────
async def _require_project_access(request: Request, project_id: str) -> dict:
    """Verify user is authenticated and owns the project (or is admin)."""
    user = await get_current_user(request, db)
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(404, "Project not found")
    if user.get("role") != "admin" and project.get("user_id") != user.get("id"):
        raise HTTPException(403, "Access denied")
    return project


def _get_nested(d: dict, path: str):
    keys = path.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def _apply_override(d: dict, path: str, value):
    keys = path.split(".")
    target = d
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value


# ─── App Config ──────────────────────────────────────────
app.include_router(api)

frontend_url = os.environ.get('REACT_APP_FRONTEND_URL', os.environ.get('CORS_ORIGINS', '*'))
origins = [o.strip() for o in frontend_url.split(',') if o.strip()]
if '*' in origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
