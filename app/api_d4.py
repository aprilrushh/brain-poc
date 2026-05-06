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
def api_post_message(chat_id: int, body: MessageCreate, stream: int = 0, current_user: dict = Depends(require_login)):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    if stream == 1:
        return _api_post_message_stream(chat_id, body, chat)
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
                    chunks = _load_file(fpath.read_bytes(), cf["filename"])
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
        # Option B (D5.0): build general_extra_context = project knowledge + chat attached
        # full file content (size-capped). General LLM gets reference docs so its answer
        # is grounded in same material as Doc-grounded answer (Brain retrieval).
        general_attached_text = ""
        try:
            _GEN_PER_FILE = 200 * 1024
            _GEN_TOTAL = 800 * 1024
            _gen_total_size = 0
            _proj_files = list_files(project_id)
            _chat_files_for_gen = list_chat_files(chat_id)
            _all_files = [(_f, "project knowledge") for _f in _proj_files] + [(_f, "chat-attached") for _f in _chat_files_for_gen]
            for _fr, _scope in _all_files:
                if _gen_total_size >= _GEN_TOTAL:
                    break
                _fpath = _file_disk_path(_fr)
                if not _fpath.exists():
                    continue
                try:
                    _chunks_g = _load_file(_fpath.read_bytes(), _fr["filename"])
                    _file_text_g = chr(10).join(_c.text for _c in _chunks_g if _c.text)
                    if not _file_text_g:
                        continue
                    _file_text_g = _file_text_g[:_GEN_PER_FILE]
                    _remaining = _GEN_TOTAL - _gen_total_size
                    _file_text_g = _file_text_g[:_remaining]
                    general_attached_text += chr(10) + "--- " + _fr["filename"] + " (" + _scope + ") ---" + chr(10) + _file_text_g + chr(10)
                    _gen_total_size += len(_file_text_g)
                except Exception as _ee:
                    print("[OptionB] file load failed: " + str(_ee))
        except Exception as _e:
            print("[OptionB] general_extra_context build failed: " + str(_e))
        result = orch.query(body.content, extra_context=chat_attached_text or None, general_extra_context=general_attached_text or None)
        # Compose visible answer = doc-grounded + general knowledge (separator marker)
        answer_text = result["answer"]
        if result.get("answer_general"):
            doc_ans = result["answer"].strip()
            gen_ans = result["answer_general"].strip()
            if doc_ans.startswith("I don't have information"):
                answer_text = (
                    "**📄 본 문서 기반:** 해당 내용 없음.\n\n"
                    "**🌐 일반 지식 답변:**\n\n" + gen_ans
                )
            else:
                answer_text = (
                    "**📄 본 문서 기반:**\n\n" + doc_ans
                    + "\n\n---\n\n**🌐 일반 지식 답변:**\n\n" + gen_ans
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
from fastapi.responses import StreamingResponse as _SSEResponse
import json as _sse_json

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
    # D4.5g: persist Brain index to disk (uvicorn reload safe)
    try:
        session = sm.get_session(sid)
        save_dir = f"data/projects/{project_id}/brain_index"
        session.brain.save(save_dir)
        from src.db import update_project
        update_project(project_id, brain_index_path=save_dir, n_chunks=len(session.brain.docs))
    except Exception as e:
        print(f"[D4.5g] persist save failed for project {project_id}: {e}")
    return len(chunks)


def _get_orchestrator(project_id: int) -> _RAGOrchestrator:
    """Build a RAGOrchestrator pointing at the project's BrainMemory.
    D4.5g: Lazy reload from disk if session evicted (uvicorn reload / TTL / LRU)."""
    state = _get_state()
    sm = state["session_manager"]
    sid = _pid_to_sid.get(project_id)
    session = sm.get_session(sid) if sid else None
    if session is None:
        # Try lazy reload from disk
        from src.db import get_project
        from src.brain import BrainMemory
        from pathlib import Path as _P
        proj = get_project(project_id)
        idx_path = (proj or {}).get("brain_index_path")
        if not idx_path or not _P(idx_path).exists():
            _pid_to_sid.pop(project_id, None)
            raise HTTPException(400, f"Project {project_id} has no Brain index yet (upload at least one file).")
        try:
            new_session = sm.create_session()
            new_session.brain = BrainMemory.load(idx_path, device=state["device"])
            _pid_to_sid[project_id] = new_session.session_id
            session = new_session
            print(f"[D4.5g] lazy-reloaded Brain index for project {project_id} from {idx_path}")
        except Exception as e:
            _pid_to_sid.pop(project_id, None)
            raise HTTPException(500, f"Failed to reload Brain index for project {project_id}: {e}")
    return _RAGOrchestrator(brain=session.brain, encoder=state["encoder"])


def _drop_project_session(project_id: int):
    """Free GPU memory + remove mapping when a project is deleted."""
    state = _get_state()
    sm = state["session_manager"]
    sid = _pid_to_sid.pop(project_id, None)
    if sid:
        sm.delete_session(sid)


# ============================================================
# Second Brain — chat events + manual G1 trigger
# Step 2 production wire (ledger v0.7/v0.8)
# ============================================================
from src.db_second_brain import (
    list_brain_events as _sb_list_events,
    mark_events_seen as _sb_mark_events_seen,
    count_unread_events as _sb_count_unread,
    get_chat_extraction_by_chat as _sb_get_extraction,
    list_mechanisms_by_extraction as _sb_list_mechs,
    list_recent_brain_events_by_user as _sb_list_recent_by_user,
)
import json as _sb_json
from src.g1_extractor import run_g1_extraction as _sb_run_g1


@router.get("/chats/{chat_id}/brain-events")
def api_list_brain_events(chat_id: int, current_user: dict = Depends(require_login)):
    """Brain timeline events + latest extraction detail for chat tail render.
    v0.16: enrich discovery_generated events with paper + discovery JOIN
           so frontend can render abstract / reasoning / DOI link on expand.
    """
    raw_events = _sb_list_events(chat_id, only_visible=True, only_unread=False)
    # Enrich discovery_generated events with paper + discovery metadata
    events = []
    import sqlite3 as _sqlite3
    from src.db import DB_PATH as _SB_DB_PATH
    _conn = _sqlite3.connect(_SB_DB_PATH)
    _conn.row_factory = _sqlite3.Row
    _cur = _conn.cursor()
    try:
        for ev in raw_events:
            ev_dict = dict(ev) if not isinstance(ev, dict) else ev
            if ev_dict.get('event_type') == 'discovery_generated' and ev_dict.get('ref_table') == 'discoveries':
                ref_id = ev_dict.get('ref_id')
                if ref_id:
                    drow = _cur.execute("SELECT type, evidence_chain_json, confidence_raw, paper_id FROM discoveries WHERE id = ?", (ref_id,)).fetchone()
                    if drow:
                        ev_dict['type'] = drow['type']
                        ev_dict['evidence_chain_json'] = drow['evidence_chain_json']
                        ev_dict['confidence_raw'] = drow['confidence_raw']
                        if drow['paper_id']:
                            prow = _cur.execute("SELECT id, doi, title, authors_json, year, abstract_text, url, source FROM papers WHERE id = ?", (drow['paper_id'],)).fetchone()
                            if prow:
                                ev_dict['paper'] = dict(prow)
            events.append(ev_dict)
    finally:
        _conn.close()
    unread = _sb_count_unread(chat_id)
    extraction = _sb_get_extraction(chat_id)
    detail = None
    if extraction:
        mechs = _sb_list_mechs(extraction['id'])
        def _parse(col):
            v = extraction[col]
            return _sb_json.loads(v) if v else []
        detail = {
            'extraction_id': extraction['id'],
            'mechanisms':    [{'id': m['id'], 'statement': m['statement'], 'position_in_chat': m['position_in_chat']} for m in mechs],
            'open_questions': _parse('open_questions_json'),
            'methods':        _parse('methods_json'),
            'frameworks':     _parse('frameworks_json'),
            'vocabulary':     _parse('vocabulary_json'),
        }
    return {"events": events, "unread_count": unread, "extraction_detail": detail}


@router.post("/chats/{chat_id}/brain-events/mark-seen")
def api_mark_events_seen(chat_id: int, current_user: dict = Depends(require_login)):
    """Mark all unread events as seen (called on chat enter)."""
    n = _sb_mark_events_seen(chat_id)
    return {"marked_seen": n}


@router.post("/chats/{chat_id}/extract-now")
def api_extract_now(chat_id: int, current_user: dict = Depends(require_login)):
    """Manual G1 trigger — bypass 30min idle wait. Closed beta debug + web validation."""
    result = _sb_run_g1(chat_id)
    return {
        "status": result.status,
        "extraction_id": result.extraction_id,
        "mechanism_count": len(result.mechanism_ids),
        "event_id": result.event_id,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }

@router.get("/brain-events/recent")
def api_recent_brain_events(limit: int = 5, current_user: dict = Depends(require_login)):
    """Cross-chat recent visible Brain events for sidebar Second Brain section."""
    limit = max(1, min(20, limit))
    events = _sb_list_recent_by_user(current_user["id"], limit)
    return {"events": events, "count": len(events)}


# ============================================================
# Streaming chat endpoint (ledger v0.11) — SSE event flow
# Reuses RAGOrchestrator.query_stream() generator.
# 4 SSE event types: start / brain_chunk / brain_done / general / meta / done / error
# DB save happens once at end (full assistant message, no partial).
# ============================================================

def _api_post_message_stream(chat_id: int, body, chat):
    user_msg = create_message(chat_id=chat_id, role="user", content=body.content)
    msgs = list_messages(chat_id)
    if chat["title"] == "New chat" and len([m for m in msgs if m["role"] == "user"]) == 1:
        new_title = body.content.strip()[:40] or "New chat"
        update_chat_title(chat_id, new_title)
    project_id = chat["project_id"]

    def _event(name: str, data: dict) -> bytes:
        return ("event: " + name + "\n" + "data: " + _sse_json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")

    def _generate():
        try:
            print(f"[stream {chat_id}] generate start", flush=True)
            yield _event("user_msg", {"id": user_msg["id"], "content": user_msg["content"]})
            print(f"[stream {chat_id}] user_msg yielded", flush=True)

            try:
                orch = _get_orchestrator(project_id)
                has_brain = True
                print(f"[stream {chat_id}] orchestrator ready", flush=True)
            except HTTPException as he:
                if he.status_code == 400 and "no Brain index" in str(he.detail):
                    has_brain = False
                    orch = None
                    print(f"[stream {chat_id}] no brain index -> general-only mode", flush=True)
                else:
                    raise

            # chat-attached 📎 files (same as non-streaming path)
            chat_attached_text = ""
            print(f"[stream {chat_id}] loading chat files...", flush=True)
            try:
                chat_files = list_chat_files(chat_id)
                print(f"[stream {chat_id}] chat_files count={len(chat_files)}", flush=True)
                for cf in chat_files:
                    fpath = _file_disk_path(cf)
                    if fpath.exists():
                        chunks = _load_file(fpath.read_bytes(), cf["filename"])
                        nl = chr(10)
                        file_text = nl.join(c.text for c in chunks if c.text)
                        if file_text:
                            chat_attached_text += nl + "--- " + cf["filename"] + " ---" + nl + file_text[:50000] + nl
            except Exception as e:
                print("[stream] chat file load failed:", e)

            # general_extra_context build (same OptionB pattern)
            general_attached_text = ""
            print(f"[stream {chat_id}] building general_extra_context...", flush=True)
            try:
                _GEN_PER_FILE = 200 * 1024
                _GEN_TOTAL = 800 * 1024
                _gen_total_size = 0
                _all_files = (
                    [(_f, "project knowledge") for _f in list_files(project_id)]
                    + [(_f, "chat-attached") for _f in list_chat_files(chat_id)]
                )
                for _fr, _scope in _all_files:
                    if _gen_total_size >= _GEN_TOTAL:
                        break
                    _fpath = _file_disk_path(_fr)
                    if not _fpath.exists():
                        continue
                    try:
                        _chunks_g = _load_file(_fpath.read_bytes(), _fr["filename"])
                        _file_text_g = chr(10).join(_c.text for _c in _chunks_g if _c.text)
                        if not _file_text_g:
                            continue
                        _file_text_g = _file_text_g[:_GEN_PER_FILE]
                        _remaining = _GEN_TOTAL - _gen_total_size
                        _file_text_g = _file_text_g[:_remaining]
                        general_attached_text += chr(10) + "--- " + _fr["filename"] + " (" + _scope + ") ---" + chr(10) + _file_text_g + chr(10)
                        _gen_total_size += len(_file_text_g)
                    except Exception as _ee:
                        print("[stream OptionB] file load failed:", _ee)
            except Exception as _e:
                print("[stream OptionB] build failed:", _e)

            brain_text_acc = []
            general_text_final = ""
            meta_final = {}
            stop_reason = None
            model_used = None
            assistant_msg_id = None  # v0.17 L1: brain_done partial save -> done final update 분기 위해 init

            retrieved_count_for_compose = 0
            _kind_counts = {"start":0, "brain_chunk":0, "brain_done":0, "general":0, "meta":0}

            if not has_brain:
                # v0.17 Issue C: 0-file project -> general-only stream
                # paradigm intent #2 (ledger v0.11): 문서 없으면 General mode = 일반 지식 답변
                print(f"[stream {chat_id}] general-only: starting general LLM call", flush=True)
                yield _event("start", {"retrieved": [], "thinking_enabled": False, "thinking_reason": "no_brain_index", "retrieval_pattern": None, "no_brain_mode": True})
                _kind_counts["start"] = 1
                from src.llm_client import get_llm_client
                from src.orchestrator import GENERAL_KNOWLEDGE_PROMPT
                _adapter = get_llm_client()
                if general_attached_text:
                    _gen_user_msg = "Reference documents (the user has provided these files):\n\n" + general_attached_text + "\n\nQuestion: " + body.content
                else:
                    _gen_user_msg = body.content
                _gen_result = _adapter.chat_complete(
                    messages=[
                        {"role": "system", "content": GENERAL_KNOWLEDGE_PROMPT},
                        {"role": "user", "content": _gen_user_msg},
                    ],
                    max_tokens=4096,
                )
                if isinstance(_gen_result, dict):
                    general_text_final = _gen_result.get("text", "") or ""
                    _gen_payload = {"text": general_text_final}
                    for _k in ("input_tokens", "output_tokens", "stop_reason", "model"):
                        if _k in _gen_result:
                            _gen_payload[_k] = _gen_result[_k]
                else:
                    general_text_final = str(_gen_result)
                    _gen_payload = {"text": general_text_final}
                print(f"[stream {chat_id}] general-only: len={len(general_text_final)}", flush=True)
                yield _event("general", _gen_payload)
                _kind_counts["general"] = 1
                stop_reason = _gen_payload.get("stop_reason")
                model_used = _gen_payload.get("model")
                meta_final = {"timings": {}, "model_used": model_used, "stop_reason": stop_reason}
                yield _event("meta", meta_final)
                _kind_counts["meta"] = 1
            else:
                print(f"[stream {chat_id}] starting query_stream chat_attached={len(chat_attached_text)} general={len(general_attached_text)}", flush=True)
                for kind, val in orch.query_stream(
                    body.content,
                    extra_context=chat_attached_text or None,
                    general_extra_context=general_attached_text or None,
                ):
                    _kind_counts[kind] = _kind_counts.get(kind, 0) + 1
                    if kind == "start":
                        retrieved_count_for_compose = len(val.get("retrieved", []) or [])
                        print(f"[stream {chat_id}] start: retrieved={retrieved_count_for_compose}", flush=True)
                        yield _event("start", val)
                    elif kind == "brain_chunk":
                        brain_text_acc.append(val)
                        yield _event("brain_chunk", {"text": val})
                    elif kind == "brain_done":
                        yield _event("brain_done", val)
                        # v0.17 L1: Brain answer 끝나면 즉시 DB partial save (General hang/Cloudflare timeout 시 답변 손실 방지)
                        if not assistant_msg_id:
                            _brain_partial = "**ð 본 문서 기반:**\n\n" + "".join(brain_text_acc).strip()
                            try:
                                _msg = create_message(chat_id=chat_id, role="assistant", content=_brain_partial)
                                assistant_msg_id = _msg["id"]
                                print(f"[stream {chat_id}] L1 partial save msg_id={assistant_msg_id} len={len(_brain_partial)}", flush=True)
                            except Exception as _e:
                                print(f"[stream {chat_id}] L1 partial save failed: {_e}", flush=True)
                    elif kind == "general":
                        general_text_final = val.get("text", "")
                        print(f"[stream {chat_id}] general: len={len(general_text_final)}", flush=True)
                        yield _event("general", val)
                    elif kind == "meta":
                        meta_final = val
                        stop_reason = val.get("stop_reason")
                        model_used = val.get("model_used")
                        print(f"[stream {chat_id}] meta: timings={val.get('timings')}", flush=True)
                        yield _event("meta", val)

            # Compose final visible answer (ledger v0.11: retrieved=0 → General only, no prefix)
            doc_ans = "".join(brain_text_acc).strip()
            gen_ans = (general_text_final or "").strip()
            _has_sources = bool(retrieved_count_for_compose)
            if not _has_sources:
                # No documents uploaded — Big Brain answers with general knowledge only
                final_text = gen_ans or doc_ans
            elif any(marker in doc_ans[:200] for marker in [
                "I don't have information", "I do not have information",
                "해당 내용 없음", "제공된 소스에는", "관련 정보가 없",
                "정보가 포함되어 있지 않", "정보를 찾을 수 없",
            ]):
                # ledger v0.11: Brain says "no info" — show General only (no 📄 prefix)
                final_text = gen_ans or doc_ans
            else:
                final_text = "**📄 본 문서 기반:**\n\n" + doc_ans
                if gen_ans:
                    final_text += "\n\n---\n\n**🌐 일반 지식 답변:**\n\n" + gen_ans

            print(f"[stream {chat_id}] kind_counts={_kind_counts} final_text_len={len(final_text)}", flush=True)
            # v0.17 L1: assistant_msg_id 있으면 (brain_done partial save 후) UPDATE, 없으면 create (기존 path)
            if assistant_msg_id:
                from src.db import get_conn
                with get_conn() as conn:
                    conn.execute("UPDATE messages SET content = ? WHERE id = ?", (final_text, assistant_msg_id))
                print(f"[stream {chat_id}] L1 final update msg_id={assistant_msg_id}", flush=True)
                assistant_msg = {"id": assistant_msg_id}
            else:
                assistant_msg = create_message(chat_id=chat_id, role="assistant", content=final_text)
            yield _event("done", {
                "assistant_msg_id": assistant_msg["id"],
                "user_msg_id": user_msg["id"],
                "stop_reason": stop_reason,
                "model_used": model_used,
                "timings": meta_final.get("timings", {}),
                "usage": meta_final.get("usage", {}),
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("[stream] ERROR:", tb)
            yield _event("error", {"message": str(e), "type": type(e).__name__})

    return _SSEResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx/cloudflared buffering
            "Connection": "keep-alive",
        },
    )

