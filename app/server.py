"""
Brain RAG Web Server (Phase 1 D2 — File Upload mode).

Endpoints:
- GET  /                              -> index.html
- GET  /api/health                    -> {ready, encoder_loaded, n_sessions, ...}
- POST /api/sessions                  -> {session_id}
- GET  /api/sessions/{sid}            -> session summary
- DELETE /api/sessions/{sid}          -> delete session
- POST /api/sessions/{sid}/upload     -> multipart file upload
- POST /api/sessions/{sid}/query      -> {query} -> answer + sources
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.brain import auto_device
from src.document_loader import load_file, estimate_encoding_seconds
from src.session_manager import (
    get_session_manager,
    MAX_FILE_BYTES,
    MAX_SESSION_BYTES,
)


app = FastAPI(title="Brain RAG (File Upload)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals: encoder + LLM client are loaded once and shared across sessions
STATE = {
    "device": None,
    "encoder": None,
    "llm_client": None,
    "llm_model": None,
    "system_prompt": None,
    "session_manager": None,
    "encoder_loaded": False,
    "load_error": None,
}


def load_globals():
    """Load encoder + LLM client once. Called on first request."""
    if STATE["encoder_loaded"]:
        return
    if STATE["load_error"]:
        raise RuntimeError(f"Load failed: {STATE['load_error']}")

    try:
        from sentence_transformers import SentenceTransformer
        from src.llm_client import get_llm_client, get_llm_model
        from src.orchestrator import SYSTEM_PROMPT

        device = auto_device()
        STATE["device"] = device
        print(f"[load] device={device}")

        print("[load] loading BGE-M3 encoder...")
        t0 = time.perf_counter()
        STATE["encoder"] = SentenceTransformer("BAAI/bge-m3", device=device)
        print(f"[load]   encoder loaded in {time.perf_counter()-t0:.1f}s")

        STATE["llm_client"] = get_llm_client()
        STATE["llm_model"] = get_llm_model()
        STATE["system_prompt"] = SYSTEM_PROMPT
        STATE["session_manager"] = get_session_manager(device=device)
        STATE["encoder_loaded"] = True
        print(f"[load] READY model={STATE['llm_model']}")

    except Exception as e:
        STATE["load_error"] = str(e)
        print(f"[load] ERROR: {e}")
        raise


# ---------- Models ----------

class HealthResponse(BaseModel):
    status: str
    encoder_loaded: bool
    device: Optional[str]
    model: Optional[str]
    n_sessions: int
    max_file_mb: int
    max_session_mb: int
    error: Optional[str]


class CreateSessionResponse(BaseModel):
    session_id: str


class FileEntryDTO(BaseModel):
    filename: str
    bytes: int
    chunks: int
    uploaded_at: float


class SessionSummary(BaseModel):
    session_id: str
    files: list[FileEntryDTO]
    total_bytes: int
    total_chunks: int
    created_at: float
    last_accessed_at: float


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    bytes: int
    num_chunks: int
    encoding_seconds: float
    session_total_chunks: int
    session_total_bytes: int


class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 512
    top_k: Optional[int] = 10


class SourceItem(BaseModel):
    rank: int
    title: str         # filename + page
    filename: str
    page: Optional[int]
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    timings_ms: dict
    top1_score: float


# ---------- Endpoints ----------

@app.get("/api/health", response_model=HealthResponse)
def health():
    sm = STATE.get("session_manager")
    return HealthResponse(
        status="ok" if STATE["encoder_loaded"] else ("loading" if not STATE["load_error"] else "error"),
        encoder_loaded=STATE["encoder_loaded"],
        device=STATE["device"],
        model=STATE["llm_model"],
        n_sessions=len(sm.list_sessions()) if sm else 0,
        max_file_mb=MAX_FILE_BYTES // (1024 * 1024),
        max_session_mb=MAX_SESSION_BYTES // (1024 * 1024),
        error=STATE["load_error"],
    )


@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session():
    if not STATE["encoder_loaded"]:
        try:
            load_globals()
        except Exception as e:
            raise HTTPException(503, f"Server not ready: {e}")
    sm = STATE["session_manager"]
    session = sm.create_session()
    return CreateSessionResponse(session_id=session.session_id)


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
def get_session_summary(session_id: str):
    if not STATE["encoder_loaded"]:
        raise HTTPException(503, "Server not ready")
    sm = STATE["session_manager"]
    session = sm.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    summary = session.to_summary()
    return SessionSummary(**summary)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not STATE["encoder_loaded"]:
        raise HTTPException(503, "Server not ready")
    sm = STATE["session_manager"]
    deleted = sm.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, "Session not found")
    return {"deleted": True, "session_id": session_id}


@app.post("/api/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    if not STATE["encoder_loaded"]:
        try:
            load_globals()
        except Exception as e:
            raise HTTPException(503, f"Server not ready: {e}")

    sm = STATE["session_manager"]
    session = sm.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")

    # Read bytes (fastapi streams; for now load into memory)
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "File is empty")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            413,
            f"File too large: {len(file_bytes)/1e6:.1f} MB > {MAX_FILE_BYTES/1e6:.0f} MB max"
        )
    if session.total_bytes + len(file_bytes) > MAX_SESSION_BYTES:
        raise HTTPException(
            413,
            f"Session size limit: {(session.total_bytes+len(file_bytes))/1e6:.1f} MB > "
            f"{MAX_SESSION_BYTES/1e6:.0f} MB"
        )

    # Parse to chunks
    try:
        t_parse = time.perf_counter()
        chunks = load_file(file_bytes, file.filename)
        parse_seconds = time.perf_counter() - t_parse
        print(f"[upload] {file.filename}: {len(chunks)} chunks parsed in {parse_seconds:.1f}s")
    except ValueError as e:
        raise HTTPException(400, f"Cannot parse file: {e}")
    except Exception as e:
        raise HTTPException(500, f"Parse error: {e}")

    if not chunks:
        raise HTTPException(400, "No text could be extracted from file")

    # Encode chunks
    t_enc = time.perf_counter()
    texts = [c.text for c in chunks]
    embeddings = STATE["encoder"].encode(
        texts,
        batch_size=256,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).float()
    enc_seconds = time.perf_counter() - t_enc
    print(f"[upload] encoded {len(chunks)} chunks in {enc_seconds:.1f}s ({len(chunks)/max(enc_seconds,0.001):.0f}/s)")

    # Add to session
    try:
        session = sm.add_file_to_session(
            session_id=session_id,
            filename=file.filename,
            file_bytes_count=len(file_bytes),
            chunks=chunks,
            embeddings=embeddings,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        bytes=len(file_bytes),
        num_chunks=len(chunks),
        encoding_seconds=round(enc_seconds, 2),
        session_total_chunks=session.total_chunks,
        session_total_bytes=session.total_bytes,
    )


@app.post("/api/sessions/{session_id}/query", response_model=QueryResponse)
def query_session(session_id: str, req: QueryRequest):
    if not STATE["encoder_loaded"]:
        raise HTTPException(503, "Server not ready")
    if not req.query.strip():
        raise HTTPException(400, "Query is empty")

    sm = STATE["session_manager"]
    session = sm.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    if session.total_chunks == 0:
        raise HTTPException(400, "Session has no documents. Upload a file first.")

    encoder = STATE["encoder"]
    brain = session.brain
    llm_client = STATE["llm_client"]
    llm_model = STATE["llm_model"]
    system_prompt = STATE["system_prompt"]
    top_k = min(req.top_k or 10, session.total_chunks)

    timings = {}

    # 1. Encode query
    t = time.perf_counter()
    q_emb = encoder.encode(
        [req.query], convert_to_tensor=True, normalize_embeddings=True
    ).float()
    timings["embed_ms"] = (time.perf_counter() - t) * 1000

    # 2. Brain recall
    t = time.perf_counter()
    retrieved = brain.recall(q_emb, top_k=top_k)
    timings["retrieve_ms"] = (time.perf_counter() - t) * 1000

    # 3. Build context
    chunks_text = []
    for r in retrieved:
        doc = r["doc"]
        title = doc.get("title", "untitled")
        text = doc.get("text", "")[:1200]
        chunks_text.append(
            f"[Source {r['rank']}: {title}] (relevance={r['score']:.4f})\n{text}"
        )
    context = "\n\n".join(chunks_text)

    # 4. LLM call
    t = time.perf_counter()
    try:
        completion = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {req.query}"},
            ],
            max_tokens=req.max_tokens or 512,
            temperature=0.3,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")
    timings["generate_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = sum(timings.values())

    # Build sources with filename + page
    sources = []
    for r in retrieved:
        doc = r["doc"]
        text = doc.get("text", "")
        snippet = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
        sources.append(SourceItem(
            rank=r["rank"],
            title=doc.get("title", "untitled"),
            filename=doc.get("filename", ""),
            page=doc.get("page"),
            score=r["score"],
            snippet=snippet,
        ))

    return QueryResponse(
        answer=answer,
        sources=sources,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
        top1_score=retrieved[0]["score"] if retrieved else 0.0,
    )


# ---------- Static UI ----------
STATIC_DIR = ROOT / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Brain RAG. UI not built yet."}
    return FileResponse(str(index_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
