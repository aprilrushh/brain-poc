"""
Session Manager: per-user in-memory Brain index store.

Each session has:
- session_id (uuid)
- N files (each with chunks)
- BrainMemory (single, accumulating chunks from all files)
- created_at, last_accessed_at
- total_bytes (uploaded raw bytes), total_chunks

Limits (Phase 1):
- MAX_FILE_BYTES = 100MB
- MAX_SESSION_BYTES = 500MB
- MAX_SESSIONS = 100 (LRU eviction when exceeded)
- SESSION_TTL_HOURS = 6 (idle session auto-eviction)
"""
from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .brain import BrainMemory
from .document_loader import Chunk


# Limits
MAX_FILE_BYTES = 100 * 1024 * 1024       # 100 MB / file (Phase 1, Gemini parity)
MAX_SESSION_BYTES = 500 * 1024 * 1024    # 500 MB / session (Phase 1)
MAX_SESSIONS = 100                        # LRU eviction beyond this
SESSION_TTL_SECONDS = 6 * 3600            # 6 hours idle


@dataclass
class FileEntry:
    filename: str
    bytes_count: int
    num_chunks: int
    uploaded_at: float


@dataclass
class Session:
    session_id: str
    brain: BrainMemory
    files: List[FileEntry] = field(default_factory=list)
    docs: List[dict] = field(default_factory=list)  # parallel to brain.docs
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes_count for f in self.files)

    @property
    def total_chunks(self) -> int:
        return sum(f.num_chunks for f in self.files)

    def touch(self):
        self.last_accessed_at = time.time()

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "files": [
                {
                    "filename": f.filename,
                    "bytes": f.bytes_count,
                    "chunks": f.num_chunks,
                    "uploaded_at": f.uploaded_at,
                }
                for f in self.files
            ],
            "total_bytes": self.total_bytes,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
        }


class SessionManager:
    """Thread-safe in-memory store of user sessions."""

    def __init__(self, device: str = "cpu", beta: float = 50.0):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.device = device
        self.beta = beta

    def create_session(self) -> Session:
        """Create a new empty session and return it."""
        with self._lock:
            self._evict_expired()
            self._evict_lru_if_needed()

            session_id = uuid.uuid4().hex
            empty_keys = torch.empty(0, 1024, device=self.device)
            brain = BrainMemory(
                keys=empty_keys,
                docs=[],
                beta=self.beta,
                device=self.device,
            )
            session = Session(session_id=session_id, brain=brain)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return session if exists, touching last_accessed_at."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.touch()
            return s

    def add_file_to_session(
        self,
        session_id: str,
        filename: str,
        file_bytes_count: int,
        chunks: List[Chunk],
        embeddings: torch.Tensor,
    ) -> Session:
        """
        Append chunks + embeddings to session's brain.

        Caller is responsible for: (1) chunking, (2) encoding, (3) size validation.
        This method just stitches into the session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            # Validate session size
            new_total = session.total_bytes + file_bytes_count
            if new_total > MAX_SESSION_BYTES:
                raise ValueError(
                    f"Session size limit exceeded: {new_total/1e6:.1f} MB > "
                    f"{MAX_SESSION_BYTES/1e6:.0f} MB"
                )

            # Append docs (parallel to brain.docs)
            new_docs = [
                {
                    "text": c.text,
                    "filename": c.filename,
                    "page": c.page,
                    "chunk_idx": c.chunk_idx,
                    "char_start": c.char_start,
                    # Synthetic title: filename + page (for orchestrator citation format)
                    "title": f"{c.filename}" + (f" p.{c.page}" if c.page else ""),
                    "url": "",
                }
                for c in chunks
            ]

            # Append embeddings to brain
            if session.brain.keys.shape[0] == 0:
                session.brain.keys = embeddings.to(self.device)
                session.brain.docs = new_docs
            else:
                session.brain.keys = torch.cat(
                    [session.brain.keys, embeddings.to(self.device)], dim=0
                )
                session.brain.docs = session.brain.docs + new_docs

            session.files.append(FileEntry(
                filename=filename,
                bytes_count=file_bytes_count,
                num_chunks=len(chunks),
                uploaded_at=time.time(),
            ))
            session.touch()
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                # Free GPU memory by reassigning keys
                self._sessions[session_id].brain.keys = torch.empty(0)
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[dict]:
        with self._lock:
            return [s.to_summary() for s in self._sessions.values()]

    def stats(self) -> dict:
        with self._lock:
            total_chunks = sum(s.total_chunks for s in self._sessions.values())
            total_bytes = sum(s.total_bytes for s in self._sessions.values())
            return {
                "n_sessions": len(self._sessions),
                "total_chunks_across_sessions": total_chunks,
                "total_bytes_uploaded": total_bytes,
                "max_sessions": MAX_SESSIONS,
                "max_file_bytes": MAX_FILE_BYTES,
                "max_session_bytes": MAX_SESSION_BYTES,
            }

    # -------- Internal eviction --------

    def _evict_expired(self):
        """Remove sessions idle longer than TTL. Caller holds lock."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_accessed_at > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self._sessions[sid].brain.keys = torch.empty(0)
            del self._sessions[sid]
            print(f"[session] evicted expired session {sid[:8]}")

    def _evict_lru_if_needed(self):
        """If at MAX_SESSIONS, evict least recently accessed. Caller holds lock."""
        while len(self._sessions) >= MAX_SESSIONS:
            lru_sid = min(
                self._sessions.keys(),
                key=lambda sid: self._sessions[sid].last_accessed_at,
            )
            self._sessions[lru_sid].brain.keys = torch.empty(0)
            del self._sessions[lru_sid]
            print(f"[session] evicted LRU session {lru_sid[:8]}")


# Global singleton (one per server)
_global_manager: Optional[SessionManager] = None


def get_session_manager(device: str = "cpu") -> SessionManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = SessionManager(device=device)
    return _global_manager
