"""
Second Brain layer — DB schema (helpers in Step 1B).

Brain = layer 1 (Phase 1 product, 5 tables in src/db.py).
Second Brain = layer 2 ("또 다른 나" entity, 13 tables here).

Both layers share the same SQLite file (DB_PATH from src.db).
src.db.init_db() executes both schemas (idempotent).

Source of truth:
  - ledger v0.5: https://www.notion.so/356c78cb12ce8196b8faec48c9da4119
  - ledger v0.6: https://www.notion.so/356c78cb12ce81878275d9c1493ef63c

13 entity:
  - clusters / chat_extractions / mechanisms / cluster_cross_links
  - outline_sections / section_history / chat_section_memberships / synthesis_docs
  - papers / external_search_logs / discoveries / calibration_logs
  - chat_brain_events  <- 13th, paradigm visible signature, Phase 4 backbone

Helpers: deferred to Step 1B (separate paste).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.db import get_conn


# ============================================================
# Schema — 13 tables, 25 indexes, 8 CHECK, 25 FK
# Forward-reference OK in SQLite (FK constraint checked on INSERT, not CREATE)
# Order chosen for readability: parent tables before child tables.
# ============================================================

SB_SCHEMA_SQL = """
-- 1. clusters (G2 output, project_id 별)
CREATE TABLE IF NOT EXISTS clusters (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id              INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name                    TEXT NOT NULL,
    user_renamed_at         TIMESTAMP,
    sticky_origin_chat_id   INTEGER REFERENCES chats(id) ON DELETE SET NULL,
    member_count            INTEGER DEFAULT 0,
    last_split_at           TIMESTAMP,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_clusters_project ON clusters(project_id);
CREATE INDEX IF NOT EXISTS idx_clusters_origin ON clusters(sticky_origin_chat_id);

-- 2. chat_extractions (G1 output, 1 chat = 1 row)
CREATE TABLE IF NOT EXISTS chat_extractions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id                 INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    schema_version          TEXT NOT NULL,
    open_questions_json     TEXT,
    methods_json            TEXT,
    frameworks_json         TEXT,
    vocabulary_json         TEXT,
    raw_llm_response        TEXT,
    extracted_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_extractions_chat ON chat_extractions(chat_id);

-- 3. mechanisms (G2-A unit, FK to clusters via SET NULL)
CREATE TABLE IF NOT EXISTS mechanisms (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_id           INTEGER NOT NULL REFERENCES chat_extractions(id) ON DELETE CASCADE,
    statement               TEXT NOT NULL,
    primary_cluster_id      INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    secondary_cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    primary_weight          REAL DEFAULT 1.0,
    position_in_chat        INTEGER,
    embedding_path          TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mechanisms_extraction ON mechanisms(extraction_id);
CREATE INDEX IF NOT EXISTS idx_mechanisms_primary_cluster ON mechanisms(primary_cluster_id);

-- 4. cluster_cross_links (G2-C, canonical UNIQUE smaller_id < larger_id)
CREATE TABLE IF NOT EXISTS cluster_cross_links (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_a_id            INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    cluster_b_id            INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    link_type               TEXT NOT NULL CHECK(link_type IN ('analogy','method_transfer','contrast','prerequisite','divergence')),
    strength                REAL DEFAULT 0.5,
    evidence_chat_ids_json  TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK(cluster_a_id < cluster_b_id)
);
CREATE INDEX IF NOT EXISTS idx_cross_links_a ON cluster_cross_links(cluster_a_id);
CREATE INDEX IF NOT EXISTS idx_cross_links_b ON cluster_cross_links(cluster_b_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cross_links_unique ON cluster_cross_links(cluster_a_id, cluster_b_id, link_type);

-- 5. outline_sections (G3, hierarchical, max depth 3)
CREATE TABLE IF NOT EXISTS outline_sections (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id              INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    parent_id               INTEGER REFERENCES outline_sections(id) ON DELETE CASCADE,
    path                    TEXT NOT NULL,
    title                   TEXT NOT NULL,
    depth                   INTEGER NOT NULL CHECK(depth BETWEEN 1 AND 3),
    order_in_parent         INTEGER NOT NULL,
    last_edited_by          TEXT NOT NULL CHECK(last_edited_by IN ('brain','user')) DEFAULT 'brain',
    last_edited_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_outline_project ON outline_sections(project_id);
CREATE INDEX IF NOT EXISTS idx_outline_parent ON outline_sections(parent_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_outline_path ON outline_sections(project_id, path);

-- 6. section_history (G3-C time slider source)
CREATE TABLE IF NOT EXISTS section_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id      INTEGER NOT NULL REFERENCES outline_sections(id) ON DELETE CASCADE,
    snapshot_json   TEXT NOT NULL,
    edited_by       TEXT NOT NULL CHECK(edited_by IN ('brain','user')),
    edit_summary    TEXT,
    edited_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_history_section ON section_history(section_id);
CREATE INDEX IF NOT EXISTS idx_history_edited ON section_history(edited_at DESC);

-- 7. chat_section_memberships (G2-A mechanism-level, M:N)
CREATE TABLE IF NOT EXISTS chat_section_memberships (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id    INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    section_id      INTEGER NOT NULL REFERENCES outline_sections(id) ON DELETE CASCADE,
    weight          REAL DEFAULT 1.0,
    mapped_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_membership_mechanism ON chat_section_memberships(mechanism_id);
CREATE INDEX IF NOT EXISTS idx_membership_section ON chat_section_memberships(section_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_membership_unique ON chat_section_memberships(mechanism_id, section_id);

-- 8. synthesis_docs (G3 Layer 3, 5 layer hallucination 검증)
CREATE TABLE IF NOT EXISTS synthesis_docs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id                  INTEGER NOT NULL REFERENCES outline_sections(id) ON DELETE CASCADE,
    version                     INTEGER NOT NULL DEFAULT 1,
    content_md                  TEXT NOT NULL,
    citations_json              TEXT,
    hallucination_layer_json    TEXT,
    generated_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_synthesis_section ON synthesis_docs(section_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_synthesis_unique ON synthesis_docs(section_id, version);

-- 9. papers (검색된 paper 영구 archive, 중복 X via doi UNIQUE partial)
CREATE TABLE IF NOT EXISTS papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doi             TEXT,
    source          TEXT NOT NULL CHECK(source IN ('openalex','semscholar','biorxiv','pubmed')),
    title           TEXT NOT NULL,
    authors_json    TEXT,
    year            INTEGER,
    abstract_text   TEXT,
    url             TEXT,
    fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);

-- 10. external_search_logs (A query + 결과)
CREATE TABLE IF NOT EXISTS external_search_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id         INTEGER REFERENCES chats(id) ON DELETE SET NULL,
    mechanism_id    INTEGER REFERENCES mechanisms(id) ON DELETE SET NULL,
    source          TEXT NOT NULL CHECK(source IN ('openalex','semscholar','biorxiv','pubmed')),
    query_text      TEXT NOT NULL,
    negative_search INTEGER DEFAULT 0,
    n_results       INTEGER DEFAULT 0,
    results_json    TEXT,
    latency_ms      INTEGER,
    ran_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_search_chat ON external_search_logs(chat_id);
CREATE INDEX IF NOT EXISTS idx_search_mechanism ON external_search_logs(mechanism_id);
CREATE INDEX IF NOT EXISTS idx_search_ran ON external_search_logs(ran_at DESC);

-- 11. discoveries (B output, 4 type, calibrated confidence, evidence chain)
CREATE TABLE IF NOT EXISTS discoveries (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id                 INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    mechanism_id            INTEGER REFERENCES mechanisms(id) ON DELETE SET NULL,
    paper_id                INTEGER REFERENCES papers(id) ON DELETE SET NULL,
    type                    TEXT NOT NULL CHECK(type IN ('strong_extension','negative','cross_project','contrarian')),
    evidence_chain_json     TEXT NOT NULL,
    l6_critique_json        TEXT,
    confidence_raw          REAL,
    confidence_calibrated   REAL,
    user_action             TEXT NOT NULL CHECK(user_action IN ('promoted','saved','dismissed','pending')) DEFAULT 'pending',
    acted_at                TIMESTAMP,
    generated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_discoveries_chat ON discoveries(chat_id);
CREATE INDEX IF NOT EXISTS idx_discoveries_action ON discoveries(user_action);
CREATE INDEX IF NOT EXISTS idx_discoveries_generated ON discoveries(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_discoveries_type ON discoveries(type);

-- 12. calibration_logs (B-B, 학자 promote/dismiss 누적, ML calibration source)
CREATE TABLE IF NOT EXISTS calibration_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_id            INTEGER NOT NULL REFERENCES discoveries(id) ON DELETE CASCADE,
    user_id                 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    predicted_probability   REAL NOT NULL,
    actual_outcome          INTEGER CHECK(actual_outcome IN (0, 1)),
    context_json            TEXT,
    logged_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_calibration_user ON calibration_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_calibration_logged ON calibration_logs(logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_calibration_outcome ON calibration_logs(actual_outcome);

-- 13. chat_brain_events (paradigm visible signature, Phase 4 backbone)
-- ref_table + ref_id = polymorphic soft FK (DB 차원 FK 안 검, application layer 검증)
CREATE TABLE IF NOT EXISTS chat_brain_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id             INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL CHECK(event_type IN (
        'extraction_completed',
        'mechanism_clustered',
        'cluster_split',
        'cross_link_formed',
        'outline_section_updated',
        'synthesis_regenerated',
        'discovery_generated',
        'discovery_acted',
        'cross_project_revival'
    )),
    ref_table           TEXT NOT NULL CHECK(ref_table IN (
        'mechanisms',
        'clusters',
        'cluster_cross_links',
        'outline_sections',
        'synthesis_docs',
        'discoveries',
        'external_search_logs'
    )),
    ref_id              INTEGER NOT NULL,
    summary_text        TEXT NOT NULL,
    icon                TEXT,
    visible_to_user     INTEGER DEFAULT 1,
    seen_at             TIMESTAMP,
    user_action         TEXT,
    occurred_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_events_chat ON chat_brain_events(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_events_occurred ON chat_brain_events(chat_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_chat_events_seen ON chat_brain_events(chat_id, seen_at);
"""


def init_sb_db() -> None:
    """Idempotent — creates 13 Second Brain tables if not exists.
    Called by src.db.init_db() after the core 5 tables.
    """
    with get_conn() as conn:
        conn.executescript(SB_SCHEMA_SQL)


# ============================================================
# JSON helpers
# ============================================================

def _dumps(obj: Any) -> Optional[str]:
    return json.dumps(obj) if obj is not None else None


def _loads(s: Optional[str]) -> Any:
    return json.loads(s) if s else None


# ============================================================
# 1. clusters (G2 output, project 별)
# ============================================================

def create_cluster(project_id: int, name: str,
                   sticky_origin_chat_id: Optional[int] = None) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO clusters (project_id, name, sticky_origin_chat_id) VALUES (?, ?, ?)",
            (project_id, name, sticky_origin_chat_id),
        )
        cid = cur.lastrowid
    return get_cluster(cid)


def get_cluster(cluster_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        return dict(row) if row else None


def list_clusters(project_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM clusters WHERE project_id = ? ORDER BY id", (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_cluster_name(cluster_id: int, name: str, by_user: bool = False) -> bool:
    with get_conn() as conn:
        if by_user:
            cur = conn.execute(
                "UPDATE clusters SET name = ?, user_renamed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (name, cluster_id),
            )
        else:
            cur = conn.execute("UPDATE clusters SET name = ? WHERE id = ?", (name, cluster_id))
        return cur.rowcount > 0


def update_cluster_member_count(cluster_id: int, member_count: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE clusters SET member_count = ? WHERE id = ?", (member_count, cluster_id),
        )
        return cur.rowcount > 0


# ============================================================
# 2. chat_extractions (G1 output, 1 chat = 1 row UNIQUE)
# ============================================================

def create_chat_extraction(chat_id: int, schema_version: str,
                           open_questions: Optional[list] = None,
                           methods: Optional[list] = None,
                           frameworks: Optional[list] = None,
                           vocabulary: Optional[list] = None,
                           raw_llm_response: Optional[str] = None) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_extractions "
            "(chat_id, schema_version, open_questions_json, methods_json, "
            " frameworks_json, vocabulary_json, raw_llm_response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, schema_version, _dumps(open_questions), _dumps(methods),
             _dumps(frameworks), _dumps(vocabulary), raw_llm_response),
        )
        eid = cur.lastrowid
    return get_chat_extraction(eid)


def get_chat_extraction(extraction_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chat_extractions WHERE id = ?", (extraction_id,)).fetchone()
        return dict(row) if row else None


def get_chat_extraction_by_chat(chat_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chat_extractions WHERE chat_id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None


# ============================================================
# 3. mechanisms (G2-A unit)
# ============================================================

def create_mechanism(extraction_id: int, statement: str,
                     primary_cluster_id: Optional[int] = None,
                     secondary_cluster_id: Optional[int] = None,
                     primary_weight: float = 1.0,
                     position_in_chat: Optional[int] = None,
                     embedding_path: Optional[str] = None) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO mechanisms "
            "(extraction_id, statement, primary_cluster_id, secondary_cluster_id, "
            " primary_weight, position_in_chat, embedding_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (extraction_id, statement, primary_cluster_id, secondary_cluster_id,
             primary_weight, position_in_chat, embedding_path),
        )
        mid = cur.lastrowid
    return get_mechanism(mid)


def get_mechanism(mechanism_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM mechanisms WHERE id = ?", (mechanism_id,)).fetchone()
        return dict(row) if row else None


def list_mechanisms_by_extraction(extraction_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM mechanisms WHERE extraction_id = ? ORDER BY position_in_chat, id",
            (extraction_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_mechanisms_by_cluster(cluster_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM mechanisms WHERE primary_cluster_id = ? OR secondary_cluster_id = ? "
            "ORDER BY id",
            (cluster_id, cluster_id),
        ).fetchall()
        return [dict(r) for r in rows]


def update_mechanism_clusters(mechanism_id: int,
                              primary_cluster_id: Optional[int] = None,
                              secondary_cluster_id: Optional[int] = None,
                              primary_weight: Optional[float] = None) -> bool:
    sets, vals = [], []
    if primary_cluster_id is not None:
        sets.append("primary_cluster_id = ?"); vals.append(primary_cluster_id)
    if secondary_cluster_id is not None:
        sets.append("secondary_cluster_id = ?"); vals.append(secondary_cluster_id)
    if primary_weight is not None:
        sets.append("primary_weight = ?"); vals.append(primary_weight)
    if not sets:
        return False
    vals.append(mechanism_id)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE mechanisms SET {', '.join(sets)} WHERE id = ?", vals)
        return cur.rowcount > 0


# ============================================================
# 4. cluster_cross_links (G2-C, canonical a < b)
# ============================================================

def create_cross_link(cluster_a_id: int, cluster_b_id: int, link_type: str,
                      strength: float = 0.5,
                      evidence_chat_ids: Optional[list] = None) -> dict:
    if cluster_a_id == cluster_b_id:
        raise ValueError("self-link forbidden")
    a, b = sorted((cluster_a_id, cluster_b_id))
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO cluster_cross_links "
            "(cluster_a_id, cluster_b_id, link_type, strength, evidence_chat_ids_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (a, b, link_type, strength, _dumps(evidence_chat_ids)),
        )
        lid = cur.lastrowid
        row = conn.execute("SELECT * FROM cluster_cross_links WHERE id = ?", (lid,)).fetchone()
        return dict(row)


def list_cross_links_by_cluster(cluster_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cluster_cross_links WHERE cluster_a_id = ? OR cluster_b_id = ? "
            "ORDER BY id",
            (cluster_id, cluster_id),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_cross_link(link_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM cluster_cross_links WHERE id = ?", (link_id,))
        return cur.rowcount > 0


# ============================================================
# 5. outline_sections (G3, hierarchical)
# ============================================================

def create_outline_section(project_id: int, title: str, depth: int,
                           order_in_parent: int,
                           parent_id: Optional[int] = None,
                           path: Optional[str] = None,
                           last_edited_by: str = 'brain') -> dict:
    if last_edited_by not in ('brain', 'user'):
        raise ValueError(f"last_edited_by must be 'brain' or 'user', got {last_edited_by!r}")
    if path is None:
        path = str(order_in_parent)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO outline_sections "
            "(project_id, parent_id, path, title, depth, order_in_parent, last_edited_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, parent_id, path, title, depth, order_in_parent, last_edited_by),
        )
        sid = cur.lastrowid
    return get_outline_section(sid)


def get_outline_section(section_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM outline_sections WHERE id = ?", (section_id,)).fetchone()
        return dict(row) if row else None


def list_outline_sections(project_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM outline_sections WHERE project_id = ? ORDER BY path",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_outline_children(parent_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM outline_sections WHERE parent_id = ? ORDER BY order_in_parent",
            (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_outline_section(section_id: int, title: Optional[str] = None,
                           last_edited_by: Optional[str] = None) -> bool:
    sets, vals = [], []
    if title is not None:
        sets.append("title = ?"); vals.append(title)
    if last_edited_by is not None:
        if last_edited_by not in ('brain', 'user'):
            raise ValueError(f"last_edited_by must be 'brain' or 'user', got {last_edited_by!r}")
        sets.append("last_edited_by = ?"); vals.append(last_edited_by)
    if not sets:
        return False
    sets.append("last_edited_at = CURRENT_TIMESTAMP")
    vals.append(section_id)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE outline_sections SET {', '.join(sets)} WHERE id = ?", vals)
        return cur.rowcount > 0


def delete_outline_section(section_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM outline_sections WHERE id = ?", (section_id,))
        return cur.rowcount > 0


# ============================================================
# 6. section_history (G3-C time slider, immutable append)
# ============================================================

def create_section_history(section_id: int, snapshot: dict, edited_by: str,
                           edit_summary: Optional[str] = None) -> dict:
    if edited_by not in ('brain', 'user'):
        raise ValueError(f"edited_by must be 'brain' or 'user', got {edited_by!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO section_history (section_id, snapshot_json, edited_by, edit_summary) "
            "VALUES (?, ?, ?, ?)",
            (section_id, _dumps(snapshot), edited_by, edit_summary),
        )
        row = conn.execute("SELECT * FROM section_history WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_section_history(section_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM section_history WHERE section_id = ? ORDER BY edited_at",
            (section_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# 7. chat_section_memberships (G2-A mechanism-level, M:N)
# ============================================================

def create_membership(mechanism_id: int, section_id: int, weight: float = 1.0) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_section_memberships (mechanism_id, section_id, weight) "
            "VALUES (?, ?, ?)",
            (mechanism_id, section_id, weight),
        )
        row = conn.execute("SELECT * FROM chat_section_memberships WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_memberships_by_section(section_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_section_memberships WHERE section_id = ? ORDER BY weight DESC",
            (section_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_memberships_by_mechanism(mechanism_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_section_memberships WHERE mechanism_id = ? ORDER BY weight DESC",
            (mechanism_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_membership(membership_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM chat_section_memberships WHERE id = ?", (membership_id,))
        return cur.rowcount > 0


# ============================================================
# 8. synthesis_docs (G3 Layer 3, auto version+1)
# ============================================================

def create_synthesis(section_id: int, content_md: str,
                     citations: Optional[list] = None,
                     hallucination_layer: Optional[dict] = None) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM synthesis_docs WHERE section_id = ?",
            (section_id,),
        )
        next_version = cur.fetchone()[0] + 1
        cur = conn.execute(
            "INSERT INTO synthesis_docs "
            "(section_id, version, content_md, citations_json, hallucination_layer_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (section_id, next_version, content_md,
             _dumps(citations), _dumps(hallucination_layer)),
        )
        row = conn.execute("SELECT * FROM synthesis_docs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_latest_synthesis(section_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM synthesis_docs WHERE section_id = ? ORDER BY version DESC LIMIT 1",
            (section_id,),
        ).fetchone()
        return dict(row) if row else None


def list_synthesis_versions(section_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM synthesis_docs WHERE section_id = ? ORDER BY version",
            (section_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# 9. papers (영구 archive, doi UNIQUE upsert)
# ============================================================

def upsert_paper(source: str, title: str,
                 doi: Optional[str] = None,
                 authors: Optional[list] = None,
                 year: Optional[int] = None,
                 abstract_text: Optional[str] = None,
                 url: Optional[str] = None) -> dict:
    if doi:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
            if row:
                return dict(row)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO papers (doi, source, title, authors_json, year, abstract_text, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doi, source, title, _dumps(authors), year, abstract_text, url),
        )
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_paper(paper_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None


# ============================================================
# 10. external_search_logs (A query + 결과)
# ============================================================

def create_search_log(source: str, query_text: str,
                      chat_id: Optional[int] = None,
                      mechanism_id: Optional[int] = None,
                      negative_search: bool = False,
                      n_results: int = 0,
                      results: Optional[list] = None,
                      latency_ms: Optional[int] = None) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO external_search_logs "
            "(chat_id, mechanism_id, source, query_text, negative_search, "
            " n_results, results_json, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, mechanism_id, source, query_text,
             1 if negative_search else 0, n_results, _dumps(results), latency_ms),
        )
        row = conn.execute("SELECT * FROM external_search_logs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_search_logs_by_chat(chat_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM external_search_logs WHERE chat_id = ? ORDER BY ran_at DESC",
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# 11. discoveries (B output, 4 type)
# ============================================================

def create_discovery(chat_id: int, type: str, evidence_chain: list,
                     mechanism_id: Optional[int] = None,
                     paper_id: Optional[int] = None,
                     l6_critique: Optional[dict] = None,
                     confidence_raw: Optional[float] = None,
                     confidence_calibrated: Optional[float] = None) -> dict:
    if type not in ('strong_extension', 'negative', 'cross_project', 'contrarian'):
        raise ValueError(f"discovery type invalid: {type!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO discoveries "
            "(chat_id, mechanism_id, paper_id, type, evidence_chain_json, "
            " l6_critique_json, confidence_raw, confidence_calibrated, user_action) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (chat_id, mechanism_id, paper_id, type,
             _dumps(evidence_chain), _dumps(l6_critique),
             confidence_raw, confidence_calibrated),
        )
        did = cur.lastrowid
    return get_discovery(did)


def get_discovery(discovery_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM discoveries WHERE id = ?", (discovery_id,)).fetchone()
        return dict(row) if row else None


def list_discoveries_by_chat(chat_id: int, only_pending: bool = False) -> list[dict]:
    with get_conn() as conn:
        if only_pending:
            sql = ("SELECT * FROM discoveries WHERE chat_id = ? AND user_action = 'pending' "
                   "ORDER BY generated_at DESC")
        else:
            sql = "SELECT * FROM discoveries WHERE chat_id = ? ORDER BY generated_at DESC"
        rows = conn.execute(sql, (chat_id,)).fetchall()
        return [dict(r) for r in rows]


def list_pending_discoveries_for_user(user_id: int) -> list[dict]:
    """Daily Discoveries panel — user 의 모든 chat 에서 pending."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT d.* FROM discoveries d "
            "JOIN chats c ON d.chat_id = c.id "
            "JOIN projects p ON c.project_id = p.id "
            "WHERE p.user_id = ? AND d.user_action = 'pending' "
            "ORDER BY d.generated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_discovery_action(discovery_id: int, user_action: str) -> bool:
    if user_action not in ('promoted', 'saved', 'dismissed', 'pending'):
        raise ValueError(f"user_action invalid: {user_action!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE discoveries SET user_action = ?, acted_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user_action, discovery_id),
        )
        return cur.rowcount > 0


# ============================================================
# 12. calibration_logs (B-B, 학자 promote/dismiss 누적)
# ============================================================

def create_calibration_log(discovery_id: int, user_id: int,
                           predicted_probability: float,
                           actual_outcome: Optional[int] = None,
                           context: Optional[dict] = None) -> dict:
    if actual_outcome is not None and actual_outcome not in (0, 1):
        raise ValueError(f"actual_outcome must be 0 or 1, got {actual_outcome!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calibration_logs "
            "(discovery_id, user_id, predicted_probability, actual_outcome, context_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (discovery_id, user_id, predicted_probability,
             actual_outcome, _dumps(context)),
        )
        row = conn.execute("SELECT * FROM calibration_logs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_calibration_logs_by_user(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM calibration_logs WHERE user_id = ? ORDER BY logged_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# 13. chat_brain_events (paradigm visible signature, Phase 4 backbone)
# ============================================================

VALID_EVENT_TYPES = {
    'extraction_completed', 'mechanism_clustered', 'cluster_split',
    'cross_link_formed', 'outline_section_updated', 'synthesis_regenerated',
    'discovery_generated', 'discovery_acted', 'cross_project_revival',
}
VALID_REF_TABLES = {
    'mechanisms', 'clusters', 'cluster_cross_links', 'outline_sections',
    'synthesis_docs', 'discoveries', 'external_search_logs',
}


def create_brain_event(chat_id: int, event_type: str,
                       ref_table: str, ref_id: int,
                       summary_text: str,
                       icon: Optional[str] = None,
                       visible_to_user: bool = True) -> dict:
    """paradigm visible signature — chat 끝 timeline append.
    summary_text = LLM 미리 작성 (render 시 호출 zero).
    """
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"event_type invalid: {event_type!r}")
    if ref_table not in VALID_REF_TABLES:
        raise ValueError(f"ref_table invalid: {ref_table!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_brain_events "
            "(chat_id, event_type, ref_table, ref_id, summary_text, icon, visible_to_user) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, event_type, ref_table, ref_id, summary_text,
             icon, 1 if visible_to_user else 0),
        )
        row = conn.execute("SELECT * FROM chat_brain_events WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def list_brain_events(chat_id: int, only_visible: bool = True,
                      only_unread: bool = False) -> list[dict]:
    """chat 끝 timeline render — 시간 순."""
    sql = "SELECT * FROM chat_brain_events WHERE chat_id = ?"
    vals = [chat_id]
    if only_visible:
        sql += " AND visible_to_user = 1"
    if only_unread:
        sql += " AND seen_at IS NULL"
    sql += " ORDER BY occurred_at"
    with get_conn() as conn:
        rows = conn.execute(sql, vals).fetchall()
        return [dict(r) for r in rows]


def mark_events_seen(chat_id: int) -> int:
    """학자 chat 진입 시 모든 unread event seen_at update. Returns count updated."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE chat_brain_events SET seen_at = CURRENT_TIMESTAMP "
            "WHERE chat_id = ? AND seen_at IS NULL",
            (chat_id,),
        )
        return cur.rowcount


def update_event_user_action(event_id: int, user_action: str) -> bool:
    """B output discovery_acted 의 click 즉시 반영."""
    if user_action not in ('promoted', 'saved', 'dismissed'):
        raise ValueError(f"user_action invalid: {user_action!r}")
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE chat_brain_events SET user_action = ? WHERE id = ?",
            (user_action, event_id),
        )
        return cur.rowcount > 0


def count_unread_events(chat_id: int) -> int:
    """sidebar badge — 단순 count."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM chat_brain_events "
            "WHERE chat_id = ? AND visible_to_user = 1 AND seen_at IS NULL",
            (chat_id,),
        ).fetchone()
        return row[0]


def list_recent_brain_events_by_user(user_id: int, limit: int = 5) -> list[dict]:
    """Cross-chat recent visible Brain events — sidebar Second Brain section render.

    Ownership chain: chat_brain_events → chats → projects.user_id
    """
    sql = """
        SELECT cbe.id, cbe.chat_id, cbe.event_type, cbe.summary_text, cbe.icon,
               cbe.occurred_at, cbe.seen_at,
               c.title AS chat_title, c.project_id,
               p.name AS project_name
        FROM chat_brain_events cbe
        JOIN chats    c ON c.id = cbe.chat_id
        JOIN projects p ON p.id = c.project_id
        WHERE p.user_id = ? AND cbe.visible_to_user = 1
        ORDER BY cbe.occurred_at DESC
        LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(sql, [user_id, limit]).fetchall()
        return [dict(r) for r in rows]
