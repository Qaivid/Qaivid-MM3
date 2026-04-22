"""Authentication helpers: password hashing, current-user lookup, decorators.

Session strategy: Flask's signed-cookie session stores only `user_id`. Every
request that needs the user re-reads the row from Postgres so a deleted user is
immediately logged out, and so admin flips take effect on the next request.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Optional

import psycopg
from flask import flash, g, redirect, request, session, url_for
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from werkzeug.security import check_password_hash, generate_password_hash


class DuplicateEmailError(ValueError):
    """Raised when an email is already registered (covers concurrent inserts)."""


logger = logging.getLogger(__name__)


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


# --- user CRUD ---------------------------------------------------------------

def get_user_by_id(user_id: int) -> Optional[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, is_admin, created_at, "
            "       plan, stripe_customer_id, plan_expires_at, "
            "       credits, plan_interval "
            "FROM users WHERE id = %s",
            (user_id,),
        )
        return cur.fetchone()


def update_user_plan(
    user_id: int,
    plan: str,
    stripe_customer_id: Optional[str] = None,
    plan_expires_at=None,
    plan_interval: Optional[str] = None,
    credits_to_grant: int = 0,
) -> None:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE users
               SET plan = %s,
                   stripe_customer_id = COALESCE(%s, stripe_customer_id),
                   plan_expires_at = %s,
                   plan_interval = COALESCE(%s, plan_interval),
                   credits = credits + %s
             WHERE id = %s
            """,
            (plan, stripe_customer_id, plan_expires_at, plan_interval, credits_to_grant, user_id),
        )
        if credits_to_grant > 0:
            cur.execute(
                """INSERT INTO credit_ledger (user_id, credits, label)
                   VALUES (%s, %s, %s)""",
                (user_id, credits_to_grant, f"Plan grant: {plan}"),
            )
        conn.commit()


def grant_monthly_credits(user_id: int, plan: str, plan_interval: str = "monthly") -> int:
    """Grant monthly credit allocation for a plan. Returns credits granted."""
    from billing import PLANS  # avoid circular at module level
    plan_def = PLANS.get(plan, {})
    credits = plan_def.get("credits_monthly", 0)
    if credits <= 0:
        return 0
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET credits = %s WHERE id = %s",
            (credits, user_id),
        )
        cur.execute(
            "INSERT INTO credit_ledger (user_id, credits, label) VALUES (%s, %s, %s)",
            (user_id, credits, f"Monthly reset: {plan}"),
        )
        conn.commit()
    return credits


def deduct_credits(user_id: int, amount: float, label: str, project_id: Optional[str] = None) -> bool:
    """Deduct credits from user balance. Returns True if successful, False if insufficient."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT credits FROM users WHERE id = %s FOR UPDATE", (user_id,))
        row = cur.fetchone()
        if not row or (row.get("credits") or 0) < amount:
            return False
        cur.execute(
            "UPDATE users SET credits = credits - %s WHERE id = %s",
            (amount, user_id),
        )
        cur.execute(
            "INSERT INTO credit_ledger (user_id, project_id, credits, label) VALUES (%s, %s, %s, %s)",
            (user_id, project_id, -amount, label),
        )
        conn.commit()
    return True


def get_credit_balance(user_id: int) -> int:
    """Return the current credit balance for a user."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT credits FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return int((row or {}).get("credits") or 0)


def get_user_by_stripe_customer(customer_id: str) -> Optional[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, is_admin, plan, stripe_customer_id, plan_expires_at "
            "FROM users WHERE stripe_customer_id = %s",
            (customer_id,),
        )
        return cur.fetchone()


def count_user_projects(user_id: int) -> int:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS c FROM projects WHERE user_id = %s",
            (user_id,),
        )
        return (cur.fetchone() or {}).get("c", 0)


def get_user_by_email(email: str) -> Optional[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, is_admin, created_at, email_verified "
            "FROM users WHERE LOWER(email) = LOWER(%s)",
            (email,),
        )
        return cur.fetchone()


def create_user(email: str, password: str, is_admin: bool = False, pre_verified: bool = False) -> dict:
    """Insert a new user. Raises DuplicateEmailError on a unique-constraint race.

    Regular signups are created with email_verified=FALSE and a random token.
    Admin bootstrap and pre_verified=True skip the verification requirement.
    Returns the new user row including email_verify_token.
    """
    pw_hash = generate_password_hash(password)
    token: Optional[str] = None if (is_admin or pre_verified) else secrets.token_urlsafe(32)
    verified: bool = is_admin or pre_verified
    now = datetime.now(timezone.utc)
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users
                  (email, password_hash, is_admin,
                   email_verified, email_verify_token, email_verify_sent_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, email, is_admin, created_at,
                          email_verified, email_verify_token
                """,
                (email.strip(), pw_hash, is_admin, verified, token, now if token else None),
            )
            row = cur.fetchone()
            conn.commit()
            return row
    except UniqueViolation as exc:
        raise DuplicateEmailError(email) from exc


def get_user_by_verify_token(token: str) -> Optional[dict]:
    """Look up a user by their email verification token."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, email_verified, email_verify_token, email_verify_sent_at "
            "FROM users WHERE email_verify_token = %s",
            (token,),
        )
        return cur.fetchone()


def confirm_email_verification(user_id: int) -> None:
    """Mark a user as verified and clear the token."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE users
               SET email_verified = TRUE,
                   email_verify_token = NULL,
                   email_verify_sent_at = NULL
             WHERE id = %s""",
            (user_id,),
        )
        conn.commit()


def refresh_verify_token(user_id: int) -> str:
    """Generate a new verification token (for resend requests)."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET email_verify_token = %s, email_verify_sent_at = %s WHERE id = %s",
            (token, now, user_id),
        )
        conn.commit()
    return token


EMAIL_VERIFY_EXPIRY_HOURS = 24
RESET_TOKEN_EXPIRY_HOURS = 1


def is_verify_token_expired(sent_at) -> bool:
    """Return True if the token is older than EMAIL_VERIFY_EXPIRY_HOURS."""
    if not sent_at:
        return True
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - sent_at > timedelta(hours=EMAIL_VERIFY_EXPIRY_HOURS)


def is_reset_token_expired(sent_at) -> bool:
    """Return True if the reset token is older than RESET_TOKEN_EXPIRY_HOURS."""
    if not sent_at:
        return True
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - sent_at > timedelta(hours=RESET_TOKEN_EXPIRY_HOURS)


def create_reset_token(user_id: int) -> str:
    """Generate and store a password-reset token. Returns the token."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET reset_token = %s, reset_token_sent_at = %s WHERE id = %s",
            (token, now, user_id),
        )
        conn.commit()
    return token


def get_user_by_reset_token(token: str) -> Optional[dict]:
    """Look up a user by their password-reset token."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, reset_token, reset_token_sent_at FROM users WHERE reset_token = %s",
            (token,),
        )
        return cur.fetchone()


def consume_reset_token(user_id: int, new_password: str) -> None:
    """Set the new password and clear the reset token atomically."""
    pw_hash = generate_password_hash(new_password)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET password_hash = %s, reset_token = NULL, reset_token_sent_at = NULL WHERE id = %s",
            (pw_hash, user_id),
        )
        conn.commit()


def verify_password(user: dict, password: str) -> bool:
    return check_password_hash(user.get("password_hash") or "", password)


def get_user_password_hash(user_id: int) -> Optional[str]:
    """Return only the password_hash for the given user (for current-password verification)."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return row["password_hash"] if row else None


def update_user_password(user_id: int, new_password: str) -> None:
    pw_hash = generate_password_hash(new_password)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (pw_hash, user_id))
        conn.commit()


def delete_user(user_id: int) -> None:
    """Permanently delete a user. Cascade on projects/assets is handled by FK constraints."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()


# --- session glue ------------------------------------------------------------

def login_user(user: dict) -> None:
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True


def logout_user() -> None:
    session.clear()


def current_user() -> Optional[dict]:
    """Return the current logged-in user dict, or None. Cached on `g` per request."""
    if "current_user" in g:
        return g.current_user
    uid = session.get("user_id")
    user = get_user_by_id(uid) if uid else None
    if uid and not user:
        # Stale session pointing at a deleted user — clear it.
        session.clear()
    g.current_user = user
    return user


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        if not user.get("is_admin"):
            flash("Admin access required.", "error")
            return redirect(url_for("index"))
        return view(*args, **kwargs)
    return wrapper


# --- bootstrap admin ---------------------------------------------------------

def bootstrap_admin() -> None:
    """Create an admin from ADMIN_EMAIL/ADMIN_PASSWORD on first boot.

    Idempotent: if a user with that email already exists, only the is_admin
    flag is promoted (password is left alone so rotations don't surprise anyone).
    """
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")
    if not email or not password:
        return
    existing = get_user_by_email(email)
    if existing:
        if not existing.get("is_admin"):
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_admin = TRUE WHERE id = %s",
                    (existing["id"],),
                )
                conn.commit()
            logger.info("Promoted existing user %s to admin", email)
        return
    create_user(email, password, is_admin=True)
    logger.info("Bootstrapped admin user %s", email)
