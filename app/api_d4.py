"""Phase 1 D4.5b — Project/Chat/Message API endpoints.

user_id=1 hardcoded (auth deferred to D4.5e — token cookie middleware).
Brain retrieval + LLM integration deferred to D4.5d (currently mock answer).
"""
from __future__ import annotations
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from src.auth import require_login
from pydantic import BaseModel
from src.db import (
    init_db,
    list_projects, create_project, get_project, delete_project,
    list_chats, create_chat, get_chat, update_chat_title, delete_chat,
    list_messages, create_message,
    list_files, list_chat_files,
    create_file, get_file, delete_file,
)

# Idempotent schema init on import.
init_db()

router = APIRouter(prefix="/api", tags=["d4"])

# HARDCODE_USER_ID = 1  # D4.5e-5 (2026-05-03): replaced by Depends(require_login). Each endpoint now uses current_user["id"] from OAuth session.


# ---- Pydantic schemas ----
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ChatCreate(BaseModel):
    title: Optional[str] = None


class MessageCreate(BaseModel):
    content: str
    chat_attached_filenames: Optional[list[str]] = None  # D4.5c persist


# ---- Project endpoints ----
@router.get("/projects")
def api_list_projects(current_user: dict = Depends(require_login)):
    projects = list_projects(current_user["id"])
    for p in projects:
        p["chat_count"] = len(list_chats(p["id"]))
        p["file_count"] = len(list_files(p["id"]))
    return {"projects": projects}


@router.post("/projects")
def api_create_project(body: ProjectCreate, current_user: dict = Depends(require_login)):
    return create_project(
        user_id=current_user["id"], name=body.name, description=body.description
    )


@router.get("/projects/{project_id}")
def api_get_project(project_id: int, current_user: dict = Depends(require_login)):
    p = get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    p["files"] = list_files(project_id)
    p["chats"] = list_chats(project_id)
    return p


@router.delete("/projects/{project_id}")
def api_delete_project(project_id: int, current_user: dict = Depends(require_login)):
    if not delete_project(project_id):
        raise HTTPException(404, "Project not found")
    try:
        _drop_project_session(project_id)
    except Exception:
        pass
    return {"deleted": project_id}


# ---- Chat endpoints ----
@router.get("/projects/{project_id}/chats")
def api_list_chats(project_id: int, current_user: dict = Depends(require_login)):
    return {"chats": list_chats(project_id)}


@router.post("/projects/{project_id}/chats")
def api_create_chat(project_id: int, body: ChatCreate, current_user: dict = Depends(require_login)):
    return create_chat(project_id=project_id, title=body.title or "New chat")


@router.delete("/chats/{chat_id}")
def api_delete_chat(chat_id: int, current_user: dict = Depends(require_login)):
    if not delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")
    return {"deleted": chat_id}


# ---- Message endpoints ----
@router.get("/chats/{chat_id}/messages")
def api_list_messages(chat_id: int, current_user: dict = Depends(require_login)):
    return {"messages": list_messages(chat_id)}


@router.post("/chats/{chat_id}/messages")
def api_post_message(chat_id: int, body: MessageCreate, current_user: dict = Depends(require_login)):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    user_msg = create_message(chat_id=chat_id, role="user", content=body.content)
    # Auto-title from first user message
    msgs = list_messages(chat_id)
    if chat["title"] == "New chat" and len([m for m in msgs if m["role"] == "user"]) == 1:
        new_title = body.content.strip()[:40] or "New chat"
        update_chat_title(chat_id, new_title)
    # D4.5d: Real Brain retrieval + dual mode LLM via RAGOrchestrator
    project_id = chat["project_id"]
    try:
        orch = _get_orchestrator(project_id)
        # D4.5d-bonus: chat-scoped 📎 files → LLM context
        chat_attached_text = ""
        try:
            chat_files = list_chat_files(chat_id)
            for cf in chat_files:
                fpath = _file_disk_path(cf)
                if fpath.exists():
                    chunks = _document_load_file(fpath, fname=cf["filename"])
                    parts = []
                    for c in chunks:
                        if c.text:
                            parts.append(c.text)
                    nl = chr(10)
                    file_text = nl.join(parts)
                    if file_text:
                        fname = cf["filename"]
                        chat_attached_text += nl + "--- " + fname + " ---" + nl + file_text[:50000] + nl
        except Exception as e:
            print(f"[D4.5d-bonus] chat file load failed: {e}")
        result = orch.query(body.content, extra_context=chat_attached_text or None)
        # Compose visible answer = doc-grounded + general knowledge (separator marker)
        answer_text = result["answer"]
        if result.get("answer_general"):
            answer_text = (
                result["answer"] + "\n\n— — —\n[General knowledge]\n" + result["answer_general"]
            )
        sources_payload = result.get("retrieved", [])
        timings_payload = result.get("timings", {})
    except HTTPException as he:
        # Project has no Brain index yet → graceful fallback message
        answer_text = (
            f"⚠️ {he.detail}\n\n"
            "Upload at least one PDF/TXT file to this project's knowledge "
            "before asking questions. Use the \"Add files\" button above."
        )
        sources_payload, timings_payload = [], {}
    except Exception as e:
        answer_text = f"[Error: {type(e).__name__}: {e}]"
        sources_payload, timings_payload = [], {}

    assistant_msg = create_message(
        chat_id=chat_id,
        role="assistant",
        content=answer_text,
        sources_json=json.dumps(sources_payload),
        timings_json=json.dumps(timings_payload),
    )
    return {"user": user_msg, "assistant": assistant_msg}


# ============================================================
# Phase 1 D4.5c — File upload (project knowledge + chat-scoped)
# ============================================================
import re
from pathlib import Path
from fastapi import UploadFile, File

PROJECT_FILE_LIMIT = 100 * 1024 * 1024  # 100 MB per project knowledge file
CHAT_FILE_LIMIT = 5 * 1024 * 1024        # 5 MB per chat attachment


def _safe_filename(name: str) -> str:
    """Sanitize filename for disk use. Keeps alnum/dot/dash/underscore."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]', '_', name or '')[:128]
    return cleaned or 'file'


def _file_disk_path(file_row: dict) -> Path:
    """Compute disk path for a file row (project knowledge or chat attachment)."""
    safe = _safe_filename(file_row.get('filename') or 'file')
    pid = file_row['project_id']
    fid = file_row['id']
    if file_row.get('chat_id'):
        return Path(f"data/projects/{pid}/chats/{file_row['chat_id']}/files/{fid}_{safe}")
    return Path(f"data/projects/{pid}/files/{fid}_{safe}")


@router.post("/projects/{project_id}/files")
async def api_upload_project_file(project_id: int, file: UploadFile = File(...), current_user: dict = Depends(require_login)):
    """Upload a file into project knowledge (joins Brain index in D4.5d)."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    contents = await file.read()
    size = len(contents)
    if size > PROJECT_FILE_LIMIT:
        raise HTTPException(413, f"File too large ({size} bytes); project limit is {PROJECT_FILE_LIMIT}")
    f_row = create_file(project_id=project_id, filename=file.filename or 'file', size_bytes=size)
    path = _file_disk_path(f_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    # D4.5d: encode chunks + add to project's BrainMemory (sync)
    try:
        n_chunks = _encode_and_index_file(project_id, file.filename or 'file', contents)
        f_row["n_chunks"] = n_chunks
    except Exception as e:
        f_row["n_chunks"] = 0
        f_row["index_error"] = str(e)
    return f_row


@router.post("/chats/{chat_id}/files")
async def api_upload_chat_file(chat_id: int, file: UploadFile = File(...), current_user: dict = Depends(require_login)):
    """Upload a file scoped to one chat only (Claude-style attachment)."""
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    contents = await file.read()
    size = len(contents)
    if size > CHAT_FILE_LIMIT:
        raise HTTPException(413, f"Chat attachment too large ({size} bytes); limit is 5MB. Use project knowledge for larger files.")
    f_row = create_file(
        project_id=chat['project_id'],
        filename=file.filename or 'file',
        size_bytes=size,
        chat_id=chat_id,
    )
    path = _file_disk_path(f_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return f_row


@router.get("/chats/{chat_id}/files")
def api_list_chat_files(chat_id: int, current_user: dict = Depends(require_login)):
    return {"files": list_chat_files(chat_id)}


@router.delete("/files/{file_id}")
def api_delete_file(file_id: int, current_user: dict = Depends(require_login)):
    f = get_file(file_id)
    if not f:
        raise HTTPException(404, "File not found")
    path = _file_disk_path(f)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass  # don't block DB delete on disk cleanup failure
    delete_file(file_id)
    return {"deleted": file_id}


# ============================================================
# Phase 1 D4.5d — Brain index per project
# project_id (DB) ↔ session_id (SessionManager in-memory) mapping
# Reuses src/orchestrator.py (D3 dual mode) + src/session_manager.py (D2)
# ============================================================
import torch as _torch
from src.document_loader import load_file as _load_file
from src.session_manager import get_session_manager as _get_sm
from src.orchestrator import RAGOrchestrator as _RAGOrchestrator

# Module-level mapping (in-memory; cleared on uvicorn reload)
_pid_to_sid: dict[int, str] = {}


def _get_state():
    """Lazy import of server.STATE — avoids circular import at module load."""
    from app.server import STATE, load_globals
    if not STATE["encoder_loaded"]:
        load_globals()
    return STATE


def _get_or_create_session_id(project_id: int) -> str:
    """Return SessionManager session_id for a project. Create new if missing.
    After uvicorn reload _pid_to_sid is empty → user must re-upload files.
    D4.5d4 (deferred) will add disk-cached `data/projects/{pid}/index.pt`."""
    state = _get_state()
    sm = state["session_manager"]
    if project_id in _pid_to_sid:
        sid = _pid_to_sid[project_id]
        if sm.get_session(sid) is not None:
            return sid
        del _pid_to_sid[project_id]
    new_session = sm.create_session()
    _pid_to_sid[project_id] = new_session.session_id
    return new_session.session_id


def _encode_and_index_file(project_id: int, filename: str, file_bytes: bytes) -> int:
    """Parse → chunks → BGE-M3 encode → add to project's BrainMemory.
    Sync (blocks the upload response). Returns chunk count.
    """
    state = _get_state()
    encoder = state["encoder"]
    chunks = _load_file(file_bytes, filename)
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    embeddings = encoder.encode(
        texts, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).float()
    sid = _get_or_create_session_id(project_id)
    sm = state["session_manager"]
    sm.add_file_to_session(
        session_id=sid,
        filename=filename,
        file_bytes_count=len(file_bytes),
        chunks=chunks,
        embeddings=embeddings,
    )
    return len(chunks)


def _get_orchestrator(project_id: int) -> _RAGOrchestrator:
    """Build a RAGOrchestrator pointing at the project's BrainMemory."""
    state = _get_state()
    sm = state["session_manager"]
    sid = _pid_to_sid.get(project_id)
    if not sid:
        raise HTTPException(400, f"Project {project_id} has no Brain index yet (upload at least one file).")
    session = sm.get_session(sid)
    if not session:
        del _pid_to_sid[project_id]
        raise HTTPException(410, f"Project {project_id} index expired (TTL/LRU). Please re-upload files.")
    return _RAGOrchestrator(brain=session.brain, encoder=state["encoder"])


def _drop_project_session(project_id: int):
    """Free GPU memory + remove mapping when a project is deleted."""
    state = _get_state()
    sm = state["session_manager"]
    sid = _pid_to_sid.pop(project_id, None)
    if sid:
        sm.delete_session(sid)
