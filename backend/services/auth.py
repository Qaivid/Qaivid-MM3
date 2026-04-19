"""
Auth Service — JWT + bcrypt for Qaivid 2.0
Email/password registration + login. Admin seeding. Brute force protection.
"""
import os
import bcrypt
import jwt
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import Request, HTTPException
from bson import ObjectId

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = timedelta(minutes=60)
REFRESH_TOKEN_EXPIRE = timedelta(days=7)
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


def get_jwt_secret() -> str:
    return os.environ["JWT_SECRET"]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str, role: str = "user") -> str:
    payload = {
        "sub": user_id, "email": email, "role": role,
        "exp": datetime.now(timezone.utc) + ACCESS_TOKEN_EXPIRE,
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + REFRESH_TOKEN_EXPIRE,
        "type": "refresh",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def set_auth_cookies(response, access_token: str, refresh_token: str):
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="none", max_age=3600, path="/")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=True, samesite="none", max_age=604800, path="/")


def clear_auth_cookies(response):
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/")


async def get_current_user(request: Request, db) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(401, "Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(401, "User not found")
        user["id"] = str(user["_id"])
        del user["_id"]
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


async def get_optional_user(request: Request, db) -> Optional[dict]:
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None


async def require_admin(request: Request, db) -> dict:
    user = await get_current_user(request, db)
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user


# ─── Brute Force Protection ──────────────────────────────

async def check_brute_force(db, identifier: str):
    doc = await db.login_attempts.find_one({"identifier": identifier})
    if doc and doc.get("attempts", 0) >= MAX_LOGIN_ATTEMPTS:
        locked_until = doc.get("locked_until")
        if locked_until and datetime.now(timezone.utc) < locked_until:
            remaining = int((locked_until - datetime.now(timezone.utc)).total_seconds() / 60) + 1
            raise HTTPException(429, f"Too many failed attempts. Try again in {remaining} minutes.")
        else:
            await db.login_attempts.delete_one({"identifier": identifier})


async def record_failed_attempt(db, identifier: str):
    doc = await db.login_attempts.find_one({"identifier": identifier})
    attempts = (doc.get("attempts", 0) if doc else 0) + 1
    update = {"attempts": attempts, "last_attempt": datetime.now(timezone.utc)}
    if attempts >= MAX_LOGIN_ATTEMPTS:
        update["locked_until"] = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
    await db.login_attempts.update_one(
        {"identifier": identifier}, {"$set": update}, upsert=True
    )


async def clear_failed_attempts(db, identifier: str):
    await db.login_attempts.delete_one({"identifier": identifier})


# ─── Admin Seeding ────────────────────────────────────────

async def seed_admin(db):
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@qaivid.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Qaivid@Admin2025")

    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one({
            "email": admin_email,
            "password_hash": hash_password(admin_password),
            "name": "Admin",
            "role": "admin",
            "plan": "studio",
            "credit_balance": 20000,
            "video_generation_enabled": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[Auth] Admin seeded: {admin_email}")
    elif not verify_password(admin_password, existing["password_hash"]):
        await db.users.update_one(
            {"email": admin_email},
            {"$set": {"password_hash": hash_password(admin_password)}}
        )
        print(f"[Auth] Admin password updated: {admin_email}")

    # Indexes
    await db.users.create_index("email", unique=True)
    await db.login_attempts.create_index("identifier")


def format_user(user: dict) -> dict:
    """Format a MongoDB user document for API response."""
    u = {**user}
    if "_id" in u:
        u["id"] = str(u["_id"])
        del u["_id"]
    u.pop("password_hash", None)
    return u
