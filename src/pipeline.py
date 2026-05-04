"""
Pipeline orchestration — APScheduler 5min poll + idle chat detection + G1 trigger.

Detects chats with `chats.updated_at` older than IDLE_THRESHOLD (30 min, UTC)
that don't yet have a chat_extraction row, and triggers run_g1_extraction(chat_id)
for each (capped at POLL_BATCH_SIZE per cycle).

Trigger lifecycle:
  - start_pipeline(): registers scheduler job, called from app/server.py
                      @app.on_event("startup")
  - stop_pipeline():  shuts down scheduler, called from
                      @app.on_event("shutdown")
  - poll_idle_chats(): the actual polling function, runs every POLL_INTERVAL_MINUTES

Source of truth: ledger v0.7
  https://www.notion.so/356c78cb12ce810fb355e2162be4e0b5
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from src.db import get_conn
from src.g1_extractor import run_g1_extraction

logger = logging.getLogger(__name__)


# ============================================================
# Configuration (ledger v0.7 결정 + dev 단계 sharper)
# ============================================================

POLL_INTERVAL_MINUTES = 5      # APScheduler interval
IDLE_THRESHOLD_MINUTES = 30    # chats.updated_at 기준 idle 임계
POLL_BATCH_SIZE = 10           # one poll cycle 처리 max chat 수 (Together rate limit 보호)

JOB_ID = 'g1_idle_poll'

# Module-level singleton scheduler
_scheduler: Optional[BackgroundScheduler] = None


# ============================================================
# Idle chat detection (SQL helper)
# ============================================================

def find_idle_chats_pending_extraction(cutoff_iso: str,
                                       limit: int = POLL_BATCH_SIZE) -> list[dict]:
    """Find chats with updated_at < cutoff AND no extraction yet.

    SQL strategy:
      LEFT JOIN chat_extractions on chat_id, filter where extraction is NULL.
      ORDER BY updated_at ASC (oldest first — fairness across users).

    Args:
      cutoff_iso: ISO 8601 datetime string (UTC). Chats with updated_at older.
      limit: max chats to return per call (default POLL_BATCH_SIZE).

    Returns:
      list of dicts: {id, project_id, title, updated_at}
    """
    sql = """
        SELECT c.id, c.project_id, c.title, c.updated_at
        FROM chats c
        LEFT JOIN chat_extractions e ON c.id = e.chat_id
        WHERE e.id IS NULL
          AND c.updated_at < ?
        ORDER BY c.updated_at ASC
        LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (cutoff_iso, limit)).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# Poll function — APScheduler triggers this every POLL_INTERVAL_MINUTES
# ============================================================

def poll_idle_chats() -> dict:
    """Main polling cycle. Find idle chats, trigger G1 for each.

    Returns dict for testability:
      {
        'cutoff':     iso str,
        'candidates': int,         # chats matched by SQL filter
        'completed':  list[int],   # chat_ids successfully extracted this cycle
        'skipped':    list[tuple], # (chat_id, status) for skipped statuses
        'failed':     list[tuple], # (chat_id, error_str) on unexpected exception
      }

    Retry policy: deferred to Phase 2.
    For now, fail-fast — chat re-evaluated on next poll cycle. If
    run_g1_extraction was partially successful (extraction created but event
    failed), idempotency check (UNIQUE constraint on chat_extractions.chat_id)
    skips it on retry.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=IDLE_THRESHOLD_MINUTES)
    cutoff_iso = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    chats = find_idle_chats_pending_extraction(cutoff_iso)

    result = {
        'cutoff':     cutoff_iso,
        'candidates': len(chats),
        'completed':  [],
        'skipped':    [],
        'failed':     [],
    }

    if not chats:
        logger.debug(f'[pipeline] no idle chats pending extraction (cutoff={cutoff_iso})')
        return result

    logger.info(f'[pipeline] {len(chats)} idle chat(s) pending extraction (cutoff={cutoff_iso})')

    for chat in chats:
        chat_id = chat['id']
        try:
            g1_result = run_g1_extraction(chat_id)
            if g1_result.status == 'completed':
                result['completed'].append(chat_id)
                cost_str = f'${g1_result.cost_usd:.6f}' if g1_result.cost_usd else 'n/a'
                logger.info(
                    f'[pipeline] chat {chat_id} extracted: '
                    f'{len(g1_result.mechanism_ids)} mech, '
                    f'cost={cost_str}, latency={g1_result.latency_ms}ms'
                )
            else:
                result['skipped'].append((chat_id, g1_result.status))
                logger.debug(f'[pipeline] chat {chat_id} skipped: {g1_result.status}')
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            result['failed'].append((chat_id, err))
            logger.error(f'[pipeline] chat {chat_id} unexpected error: {err}', exc_info=True)

    return result


# ============================================================
# Scheduler lifecycle (called from app/server.py)
# ============================================================

def start_pipeline() -> None:
    """Register the G1 polling job with APScheduler.

    Called from FastAPI @app.on_event("startup") in app/server.py.
    Idempotent — calling twice doesn't double-register.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.warning('[pipeline] start_pipeline called but scheduler already running')
        return

    _scheduler = BackgroundScheduler(timezone='UTC')
    _scheduler.add_job(
        poll_idle_chats,
        'interval',
        minutes=POLL_INTERVAL_MINUTES,
        id=JOB_ID,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=1),
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info(
        f'[pipeline] started — poll every {POLL_INTERVAL_MINUTES}min, '
        f'idle threshold {IDLE_THRESHOLD_MINUTES}min, batch size {POLL_BATCH_SIZE}'
    )


def stop_pipeline() -> None:
    """Shutdown the scheduler gracefully.

    Called from FastAPI @app.on_event("shutdown") in app/server.py.
    """
    global _scheduler
    if _scheduler is None:
        logger.debug('[pipeline] stop_pipeline called but scheduler never started')
        return
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info('[pipeline] stopped')
    _scheduler = None


def is_running() -> bool:
    """Returns True if scheduler is alive (helper for health endpoint)."""
    return _scheduler is not None and _scheduler.running
