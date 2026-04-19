"""Authentication helpers: password hashing, current-user lookup, decorators.

Session strategy: Flask's signed-cookie session stores only `user_id`. Every
request that needs the user re-reads the row from Postgres so a deleted user is
immediately logged out, and so admin flips take effect on the next request.
"""

from __future__ import annotations

import logging
import os
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
            "SELECT id, email, is_admin, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        return cur.fetchone()


def get_user_by_email(email: str) -> Optional[dict]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, is_admin, created_at "
            "FROM users WHERE LOWER(email) = LOWER(%s)",
            (email,),
        )
        return cur.fetchone()


def create_user(email: str, password: str, is_admin: bool = False) -> dict:
    """Insert a new user. Raises DuplicateEmailError on a unique-constraint race
    (the route handler also pre-checks, but two concurrent signups can both pass
    the pre-check and only one wins the insert)."""
    pw_hash = generate_password_hash(password)
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (email, password_hash, is_admin)
                VALUES (%s, %s, %s)
                RETURNING id, email, is_admin, created_at
                """,
                (email.strip(), pw_hash, is_admin),
            )
            row = cur.fetchone()
            conn.commit()
            return row
    except UniqueViolation as exc:
        raise DuplicateEmailError(email) from exc


def verify_password(user: dict, password: str) -> bool:
    return check_password_hash(user.get("password_hash") or "", password)


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
