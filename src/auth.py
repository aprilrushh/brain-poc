"""
Brain — OAuth authentication module.

Google OAuth 2.0 via Authlib. Handles login/callback/logout endpoints,
session management, and email allowlist check for closed beta.

Routes mounted under /auth/* in app/server.py.
"""
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.config import Config

# --- Config ---------------------------------------------------------------

DB_PATH = os.environ.get("BRAIN_DB_PATH", "data/brain.db")

# Authlib needs Starlette Config object, not just env vars
config = Config(environ=os.environ)

oauth = OAuth(config)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Determine redirect URI: prefer env override, else infer from request host
def _redirect_uri(request: Request) -> str:
    override = os.environ.get("OAUTH_REDIRECT_URI")
    if override:
        return override
    # Infer from request — works for both localhost dev and brain.umparumpa.com
    return str(request.url_for("auth_callback"))


# --- DB helpers -----------------------------------------------------------

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def is_email_allowed(email: str) -> bool:
    if not email:
        return False
    with _conn() as c:
        row = c.execute("SELECT 1 FROM allowed_emails WHERE email = ?", (email,)).fetchone()
        return row is not None


def get_or_create_user_by_google(email: str, name: str, google_sub: str, picture_url: Optional[str]) -> dict:
    """Upsert user by google_sub (preferred) or email. Returns user dict."""
    with _conn() as c:
        # 1) Try by google_sub (most stable identity)
        row = c.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)).fetchone()
        if row:
            c.execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP, picture_url = ?, display_name = ? WHERE id = ?",
                (picture_url, name, row["id"]),
            )
            c.commit()
            return dict(row)

        # 2) Try by email (legacy token-based user upgrading to Google)
        row = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            c.execute(
                "UPDATE users SET google_sub = ?, picture_url = ?, display_name = ?, "
                "auth_provider = 'google', last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
                (google_sub, picture_url, name, row["id"]),
            )
            c.commit()
            return dict(c.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone())

        # 3) Create new user
        token = secrets.token_urlsafe(24)
        cur = c.execute(
            "INSERT INTO users (token, email, display_name, google_sub, picture_url, auth_provider, last_login_at) "
            "VALUES (?, ?, ?, ?, ?, 'google', CURRENT_TIMESTAMP)",
            (token, email, name, google_sub, picture_url),
        )
        c.commit()
        return dict(c.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone())


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# --- Dependency: get_current_user ----------------------------------------

def get_current_user(request: Request) -> Optional[dict]:
    """FastAPI dependency. Returns user dict or None if not logged in."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def require_login(request: Request) -> dict:
    """FastAPI dependency. 401 if not logged in."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user


# --- Router --------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
async def login(request: Request):
    redirect_uri = _redirect_uri(request)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        return HTMLResponse(f"<h1>OAuth error</h1><pre>{e.error}: {e.description}</pre>", status_code=400)

    userinfo = token.get("userinfo")
    if not userinfo:
        return HTMLResponse("<h1>OAuth error</h1><p>No userinfo returned from Google.</p>", status_code=400)

    email = userinfo.get("email", "").lower().strip()
    name = userinfo.get("name") or email.split("@")[0]
    google_sub = userinfo.get("sub")
    picture_url = userinfo.get("picture")

    if not (email and google_sub):
        return HTMLResponse("<h1>OAuth error</h1><p>Missing email or sub claim.</p>", status_code=400)

    if False:  # OPEN ACCESS — closed beta via obscurity. is_email_allowed() bypassed. allowed_emails table preserved for future re-enable. (2026-05-03)
        return HTMLResponse(
            f"<h1>Access denied</h1>"
            f"<p>Your email <b>{email}</b> is not on the closed beta allowlist.</p>"
            f"<p>If you believe this is a mistake, contact <a href='mailto:andy@umparumpa.com'>andy@umparumpa.com</a>.</p>",
            status_code=403,
        )

    user = get_or_create_user_by_google(email, name, google_sub, picture_url)
    request.session["user_id"] = user["id"]
    return RedirectResponse(url="/", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)


@router.get("/me")
async def me(request: Request):
    user = get_current_user(request)
    if not user:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "picture_url": user.get("picture_url"),
        "is_admin": bool(user.get("is_admin", 0)),
    }
