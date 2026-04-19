"""
Credit / Billing System — Ported from Qaivid 1.0
1 credit = 1 cent. Plans define monthly limits.
"""
from typing import Dict, List, Optional
from datetime import datetime, timezone


# ─── Plans ────────────────────────────────────────────────

PLAN_CREDIT_LIMITS: Dict[str, int] = {
    "free": 0,
    "starter": 1500,
    "pro": 4000,
    "studio": 20000,
}

PLAN_LABELS: Dict[str, str] = {
    "free": "Free",
    "starter": "Starter ($14.99/mo)",
    "pro": "Pro ($39.99/mo)",
    "studio": "Studio ($199.99/mo)",
}


# ─── Flat operation costs (credits) ───────────────────────

OPERATION_COSTS: Dict[str, int] = {
    "transcription": 5,       # ~2.5¢ API cost → 5¢ charged
    "creative_brief": 10,     # ~5¢ → 10¢
    "interpretation": 10,     # ~5¢ → 10¢
    "scene_generation": 10,   # ~5¢ → 10¢
    "image_per_shot": 6,      # ~3¢ → 6¢
    "image_per_ref": 3,       # ~1.5¢ → 3¢
}


# ─── Video credit rates per minute ────────────────────────

VIDEO_CREDIT_RATES: Dict[str, int] = {
    "normal": 500,       # 500 credits/min ($5/min)
    "avatar": 500,
    "animation": 100,    # 100 credits/min ($1/min) — Ken Burns animatic
}


def credits_for_video(duration_sec: float, category: str = "normal") -> float:
    rate = VIDEO_CREDIT_RATES.get(category.lower(), VIDEO_CREDIT_RATES["normal"])
    return (duration_sec / 60) * rate


def get_plan_limit(plan: str) -> int:
    return PLAN_CREDIT_LIMITS.get(plan, 0)


def format_credits(credits: float) -> str:
    r = round(credits)
    return f"{r:,} credit{'s' if r != 1 else ''}"


# ─── DB Operations ────────────────────────────────────────

async def get_user_credits(db, user_id: str) -> dict:
    """Get user's credit balance and plan info."""
    from bson import ObjectId
    user = await db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "plan": 1, "credit_balance": 1})
    if not user:
        return {"plan": "free", "credit_balance": 0, "plan_limit": 0}
    plan = user.get("plan", "free")
    return {
        "plan": plan,
        "credit_balance": user.get("credit_balance", 0),
        "plan_limit": get_plan_limit(plan),
    }


async def charge_credits(db, user_id: str, amount: float, operation: str, project_id: str = "") -> bool:
    """Deduct credits from user. Returns False if insufficient."""
    from bson import ObjectId
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return False
    balance = user.get("credit_balance", 0)
    if balance < amount:
        return False

    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$inc": {"credit_balance": -amount}}
    )

    # Log to ledger
    await db.credit_ledger.insert_one({
        "user_id": user_id,
        "project_id": project_id,
        "amount": -amount,
        "operation": operation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return True


async def add_credits(db, user_id: str, amount: float, reason: str = "admin-add"):
    """Add credits to user (admin action)."""
    from bson import ObjectId
    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$inc": {"credit_balance": amount}}
    )
    await db.credit_ledger.insert_one({
        "user_id": user_id,
        "project_id": "",
        "amount": amount,
        "operation": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def reset_credits(db, user_id: str) -> dict:
    """Reset user credits to their plan limit."""
    from bson import ObjectId
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return {"error": "User not found"}
    plan = user.get("plan", "free")
    limit = get_plan_limit(plan)
    old_balance = user.get("credit_balance", 0)

    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"credit_balance": limit}}
    )
    await db.credit_ledger.insert_one({
        "user_id": user_id,
        "project_id": "",
        "amount": limit - old_balance,
        "operation": "admin-credit-reset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {"plan": plan, "old_balance": old_balance, "new_balance": limit}


async def get_credit_ledger(db, user_id: str, limit: int = 50) -> List[dict]:
    """Get credit history for a user."""
    entries = await db.credit_ledger.find(
        {"user_id": user_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(limit)
    return entries
