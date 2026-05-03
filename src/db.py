"""
SQLite DB layer for Brain.

5 tables:
  - users        : Token-based identity (email, display_name)
  - projects     : User's projects (= corpus container)
  - files        : Uploaded files within a project
  - chats        : Chat sessions within a project
  - messages     : User queries + AI answers within a chat

Phase 1 D4: Token auth (URL ?token=abc → cookie).
Phase 2: Migrate to Google OAuth (preserve email mapping).
"""
from __future__ import annotations
import os
import sqlite3
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(os.environ.get("BRAIN_DB_PATH", "data/brain.db"))


# ============================================================
# Connection management
# ============================================================

@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Context manager for SQLite connections.
    Ensures DB_PATH parent directory exists.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# Schema
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token           TEXT UNIQUE NOT NULL,
    email           TEXT,
    display_name    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at   TIMESTAMP,
    is_admin        INTEGER DEFAULT 0,
    google_sub      TEXT,
    picture_url     TEXT,
    auth_provider   TEXT DEFAULT 'token'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub) WHERE google_sub IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email) WHERE email IS NOT NULL;

CREATE TABLE IF NOT EXISTS allowed_emails (
    email       TEXT PRIMARY KEY,
    note        TEXT,
    invited_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    invited_by  INTEGER REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_users_token ON users(token);

CREATE TABLE IF NOT EXISTS projects (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    description         TEXT,
    brain_index_path    TEXT,
    n_chunks            INTEGER DEFAULT 0,
    is_shared           INTEGER DEFAULT 0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_shared ON projects(is_shared);

CREATE TABLE IF NOT EXISTS files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    size_bytes      INTEGER,
    n_chunks        INTEGER,
    uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);

CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title       TEXT DEFAULT 'New chat',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chats_project ON chats(project_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    answer_general  TEXT,
    sources_json    TEXT,
    timings_json    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
"""


def init_db() -> None:
    """Create schema if not exists. Idempotent."""
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


# ============================================================
# User management — token-based auth (Phase 1 D4)
# ============================================================

def generate_token() -> str:
    """Generate a URL-safe random token (32 chars)."""
    return secrets.token_urlsafe(24)


def create_user(email: Optional[str] = None,
                display_name: Optional[str] = None,
                is_admin: bool = False) -> dict:
    """Create a new user with a fresh token. Returns user dict."""
    token = generate_token()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (token, email, display_name, is_admin) "
            "VALUES (?, ?, ?, ?)",
            (token, email, display_name, 1 if is_admin else 0),
        )
        user_id = cur.lastrowid
    return {
        "id": user_id,
        "token": token,
        "email": email,
        "display_name": display_name,
        "is_admin": is_admin,
    }


def get_user_by_token(token: str) -> Optional[dict]:
    """Lookup user by token. Returns dict or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        # Update last_login
        conn.execute(
            "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        return dict(row)


def list_users() -> list[dict]:
    """For admin debugging."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, email, display_name, is_admin, created_at, last_login_at "
            "FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# Project CRUD
# ============================================================

def create_project(user_id: int,
                   name: str,
                   description: Optional[str] = None,
                   is_shared: bool = False) -> dict:
    """Create a new project for a user."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects (user_id, name, description, is_shared) "
            "VALUES (?, ?, ?, ?)",
            (user_id, name, description, 1 if is_shared else 0),
        )
        project_id = cur.lastrowid
    return get_project(project_id)


def get_project(project_id: int) -> Optional[dict]:
    """Lookup project by id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return dict(row) if row else None


def list_projects(user_id: int, include_shared: bool = True) -> list[dict]:
    """List user's projects + (optionally) shared projects (e.g., Wiki demo).
    Sorted by updated_at DESC.
    """
    with get_conn() as conn:
        if include_shared:
            sql = ("SELECT * FROM projects "
                   "WHERE user_id = ? OR is_shared = 1 "
                   "ORDER BY updated_at DESC")
            rows = conn.execute(sql, (user_id,)).fetchall()
        else:
            sql = ("SELECT * FROM projects WHERE user_id = ? "
                   "ORDER BY updated_at DESC")
            rows = conn.execute(sql, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def update_project(project_id: int, **fields) -> bool:
    """Update project fields. Whitelist: name, description,
    brain_index_path, n_chunks, is_shared. updated_at auto-bumped.
    Returns True if a row was updated.
    """
    allowed = {"name", "description", "brain_index_path", "n_chunks", "is_shared"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values())
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE projects SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
            f"WHERE id = ?",
            (*values, project_id),
        )
        return cur.rowcount > 0


def delete_project(project_id: int) -> bool:
    """Delete project. Cascade deletes files/chats/messages via FK."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cur.rowcount > 0


# ============================================================
# File CRUD
# ============================================================

def create_file(project_id: int,
                filename: str,
                size_bytes: Optional[int] = None,
                n_chunks: Optional[int] = None,
                chat_id: Optional[int] = None) -> dict:
    """Register an uploaded file. If chat_id is None, scope = project knowledge.
    If chat_id is set, scope = chat-only attachment (Claude-style 📎)."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO files (project_id, filename, size_bytes, n_chunks, chat_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, filename, size_bytes, n_chunks, chat_id),
        )
        file_id = cur.lastrowid
    return get_file(file_id)


def get_file(file_id: int) -> Optional[dict]:
    """Lookup file by id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return dict(row) if row else None


def list_files(project_id: int) -> list[dict]:
    """List PROJECT KNOWLEDGE files only (chat_id IS NULL). Oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE project_id = ? AND chat_id IS NULL "
            "ORDER BY uploaded_at",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_chat_files(chat_id: int) -> list[dict]:
    """List chat-scoped attachments (Claude-style 📎). Oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE chat_id = ? ORDER BY uploaded_at",
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_file(file_id: int) -> bool:
    """Delete a file row."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return cur.rowcount > 0


# ============================================================
# Chat CRUD
# ============================================================

def create_chat(project_id: int, title: str = "New chat") -> dict:
    """Create a new chat session under a project."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chats (project_id, title) VALUES (?, ?)",
            (project_id, title),
        )
        chat_id = cur.lastrowid
    return get_chat(chat_id)


def get_chat(chat_id: int) -> Optional[dict]:
    """Lookup chat by id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None


def list_chats(project_id: int) -> list[dict]:
    """List chats under a project, newest activity first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE project_id = ? "
            "ORDER BY updated_at DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_chat_title(chat_id: int, title: str) -> bool:
    """Rename a chat (e.g., auto-title from first user message)."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (title, chat_id),
        )
        return cur.rowcount > 0


def delete_chat(chat_id: int) -> bool:
    """Delete chat. Cascade deletes messages via FK."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return cur.rowcount > 0


# ============================================================
# Message CRUD
# ============================================================

def create_message(chat_id: int,
                   role: str,
                   content: str,
                   answer_general: Optional[str] = None,
                   sources_json: Optional[str] = None,
                   timings_json: Optional[str] = None) -> dict:
    """Append a message to a chat. Bumps the chat's updated_at
    so the sidebar reorders to show recent activity first.

    role: 'user' or 'assistant'
    answer_general/sources_json/timings_json: dual mode payload (assistant only)
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages "
            "(chat_id, role, content, answer_general, sources_json, timings_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, answer_general, sources_json, timings_json),
        )
        message_id = cur.lastrowid
        # Bump chat updated_at for sidebar reordering
        conn.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,),
        )
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        return dict(row)


def list_messages(chat_id: int) -> list[dict]:
    """List messages in a chat in chronological order (oldest first)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at, id",
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]
