"""
G1 — chat extraction (5 entity).

Extracts mechanisms / open_questions / methods / frameworks / vocabulary
from a chat thread. JSON output, pydantic validated, hallucination L1+L2 applied.

Triggered by src.pipeline (APScheduler 5min poll), or directly via run_g1_extraction(chat_id).

Source of truth: ledger v0.7
  https://www.notion.so/356c78cb12ce810fb355e2162be4e0b5
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from src.db import get_chat, list_messages
from src.db_second_brain import (
    get_chat_extraction_by_chat,
    create_chat_extraction,
    create_mechanism,
    create_brain_event,
)
from src.llm_client import get_llm_client

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

G1_SCHEMA_VERSION = 'v1'
G1_MAX_TOKENS = 2048
G1_TEMPERATURE = 0.3  # entity 추출 = 정확성 우선 (orchestrator default 0.7 대비 lower)
G1_MIN_MESSAGES_THRESHOLD = 2  # null chat skip — message 1개 이하면 G1 skip

LLM_CALL_LOG_PATH = Path('logs/llm_calls.jsonl')

# Cost (Together AI Qwen3-235B-A22B-Instruct-2507-tput, web verified 2026-05-04)
# tokencost github: $0.20 input / $6.00 output per 1M tokens
# Source conflict noted in ledger v0.7 — verify against Together console billing.
COST_INPUT_PER_1M = 0.20
COST_OUTPUT_PER_1M = 6.00


# ============================================================
# pydantic schema — 5 BaseModel + G1Output container
# ============================================================

class Mechanism(BaseModel):
    statement: str = Field(..., min_length=10, max_length=500)
    position_in_chat: int = Field(..., ge=1)


class OpenQuestion(BaseModel):
    text: str = Field(..., min_length=5, max_length=300)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Method(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    source_paragraph: Optional[str] = Field(default=None, max_length=500)


class Framework(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    definition: Optional[str] = Field(default=None, max_length=500)


class VocabularyTerm(BaseModel):
    term: str = Field(..., min_length=1, max_length=100)
    context: Optional[str] = Field(default=None, max_length=300)


class G1Output(BaseModel):
    mechanisms: list[Mechanism] = Field(default_factory=list, max_length=3)
    open_questions: list[OpenQuestion] = Field(default_factory=list, max_length=3)
    methods: list[Method] = Field(default_factory=list, max_length=5)
    frameworks: list[Framework] = Field(default_factory=list, max_length=5)
    vocabulary: list[VocabularyTerm] = Field(default_factory=list, max_length=10)


# ============================================================
# G1 system prompt — 5 entity 추출 + L1 hallucination 방지 + self-criticism
# ============================================================

G1_SYSTEM_PROMPT = """당신은 학자 chat 에서 사고 unit 을 추출하는 분석가입니다.

[INPUT]
chat 전체 thread (user + assistant turns).

[TASK]
다음 5 entity 를 추출하여 **JSON 만** 출력하세요:

1. mechanisms (max 3): 인과 쪽 설명. 학자의 chat 본문에서 직접 quote 또는 충실 paraphrase.
   각 mechanism: { "statement": "<설명, 10-500자>", "position_in_chat": <1, 2, 3...> }

2. open_questions (max 3): 학자가 unresolved 채로 둔 의문.
   각 question: { "text": "<의문>", "confidence": <0.0-1.0> }

3. methods (max 5): 구체 실험 기법 또는 분석 방법.
   각 method: { "name": "<기법명>", "source_paragraph": "<chat 본문에서 quote, optional>" }

4. frameworks (max 5): 이론 틀 또는 개념 모델.
   각 framework: { "name": "<frame 이름>", "definition": "<정의, optional>" }

5. vocabulary (max 10): 특수 용어 또는 도메인 술어.
   각 term: { "term": "<용어>", "context": "<등장 맥락, optional>" }

[CONSTRAINTS — 매우 중요]
- chat 안 source 가 명백히 있는 entity 만 추출 (hallucination 방지 Layer 1).
- mechanism statement 는 학자 chat 의 원문에서 quote 또는 의미 보존 paraphrase. 새 사실 추가 금지.
- 단순 의문만 등장하면 mechanism 으로 작성하지 말 것 (open_question 으로만).
- 빈 list 도 valid — 추출할 entity 없으면 빈 array 반환.
- JSON 외 어떤 텍스트도 출력 금지 (주석, 설명, codefence 모두 금지).

[SELF-CRITICISM — 출력 전 자체 검증]
- 모든 mechanism statement 가 chat 본문에 source 있는가?
- mechanism 으로 작성한 것이 사실은 단순 의문 또는 가정이 아닌가?
- frameworks/methods 가 chat 안 명백히 등장한 것인가, 아니면 LLM 이 추론한 것인가?

[OUTPUT FORMAT]
{
  "mechanisms": [...],
  "open_questions": [...],
  "methods": [...],
  "frameworks": [...],
  "vocabulary": [...]
}

JSON 만 출력. 다른 텍스트 모두 금지.
"""


# ============================================================
# Helpers
# ============================================================

def _format_transcript(messages: list[dict]) -> str:
    """Format chat messages into a transcript for G1 input.
    Includes both user and assistant turns (full thinking context).
    Strips dual mode separator (D5.0 OptionB) for cleaner input.
    """
    lines = []
    separator = '— — —'
    for m in messages:
        role = m['role'].upper()
        content = m['content'] or ''
        if separator in content:
            content = content.split(separator)[0].strip()
        lines.append(f'[{role}]\n{content}\n')
    return '\n'.join(lines)


def _strip_codefence(text: str) -> str:
    """Remove leading/trailing markdown codefence if present (LLM hiccup)."""
    cleaned = text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines)
    return cleaned


def _calc_cost(input_tokens: int, output_tokens: int) -> float:
    """Together AI Qwen3-235B pricing estimate. Verify against billing."""
    return (input_tokens / 1_000_000) * COST_INPUT_PER_1M + \
           (output_tokens / 1_000_000) * COST_OUTPUT_PER_1M


def _log_llm_call(chat_id: int, input_tokens: int, output_tokens: int,
                  cost_usd: float, latency_ms: int, status: str,
                  error: Optional[str] = None) -> None:
    """Append-only cost log. Phase 2 entity migration: file -> DB import."""
    LLM_CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'stage': 'g1',
        'chat_id': chat_id,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cost_usd': round(cost_usd, 6),
        'latency_ms': latency_ms,
        'status': status,
    }
    if error:
        entry['error'] = error[:500]  # cap long tracebacks
    with LLM_CALL_LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _validate_l2_substring(mechanisms: list[Mechanism], chat_text: str) -> list[Mechanism]:
    """L2 post-validation: weak substring check.

    완전 substring 검증은 paraphrase 차단 — 대신 statement 의 핵심 noun (3+ 글자) 중
    최소 1개가 chat 안에 있는지 검증. 없으면 hallucination 의심 -> drop.

    Phase 2 deferred: BGE-M3 embedding similarity 기반 (현재 simple).
    """
    chat_lower = chat_text.lower()
    validated = []
    dropped = []
    for m in mechanisms:
        words = [w for w in m.statement.lower().split() if len(w) >= 3]
        if not words:
            continue
        if any(w in chat_lower for w in words):
            validated.append(m)
        else:
            dropped.append(m.statement[:60])
    if dropped:
        logger.warning(f'L2 dropped {len(dropped)} mechanism(s): {dropped}')
    return validated


# ============================================================
# Main — run_g1_extraction
# ============================================================

class G1Result(BaseModel):
    """Return type of run_g1_extraction()."""
    chat_id: int
    status: str  # completed | skipped_already_done | skipped_too_short | failed
    extraction_id: Optional[int] = None
    mechanism_ids: list[int] = Field(default_factory=list)
    event_id: Optional[int] = None
    error: Optional[str] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None


def run_g1_extraction(chat_id: int) -> G1Result:
    """Trigger G1 for a chat.

    Idempotent: if extraction already exists for this chat (UNIQUE constraint on
    chat_extractions.chat_id), returns 'skipped_already_done'. Concurrent calls
    safe via DB constraint.

    Steps:
      1. Idempotency check
      2. Load messages, validate threshold
      3. Format transcript (strip dual mode separator)
      4. LLM call (Together AI Qwen3-235B, temp=0.3)
      5. Parse JSON (strip codefence), pydantic validate
      6. L2 post-validation (mechanism substring check)
      7. Transaction-effect: create extraction + mechanisms + chat_brain_event
      8. Cost log append
    """
    # 1. Idempotency
    existing = get_chat_extraction_by_chat(chat_id)
    if existing:
        return G1Result(chat_id=chat_id, status='skipped_already_done',
                        extraction_id=existing['id'])

    # 2. Threshold
    chat = get_chat(chat_id)
    if not chat:
        return G1Result(chat_id=chat_id, status='failed',
                        error=f'chat {chat_id} not found')

    messages = list_messages(chat_id)
    if len(messages) < G1_MIN_MESSAGES_THRESHOLD:
        return G1Result(chat_id=chat_id, status='skipped_too_short',
                        error=f'only {len(messages)} messages '
                              f'(threshold {G1_MIN_MESSAGES_THRESHOLD})')

    # 3. Transcript
    transcript = _format_transcript(messages)
    chat_text_for_l2 = ' '.join((m['content'] or '') for m in messages)

    # 4. LLM call
    adapter = get_llm_client()
    t0 = time.perf_counter()
    try:
        result = adapter.chat_complete(
            messages=[
                {'role': 'system', 'content': G1_SYSTEM_PROMPT},
                {'role': 'user', 'content': transcript},
            ],
            max_tokens=G1_MAX_TOKENS,
            temperature=G1_TEMPERATURE,
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _log_llm_call(chat_id, 0, 0, 0.0, latency_ms,
                      'failed_llm_call', error=str(e))
        return G1Result(chat_id=chat_id, status='failed',
                        error=f'LLM call: {type(e).__name__}: {e}',
                        latency_ms=latency_ms)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    raw_text = result['text'] or ''
    input_tokens = result['input_tokens']
    output_tokens = result['output_tokens']
    cost_usd = _calc_cost(input_tokens, output_tokens)

    # 5. Parse JSON (strip codefence), pydantic validate
    cleaned = _strip_codefence(raw_text)
    try:
        parsed = G1Output.model_validate_json(cleaned)
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        _log_llm_call(chat_id, input_tokens, output_tokens, cost_usd, latency_ms,
                      'failed_parse', error=str(e))
        return G1Result(chat_id=chat_id, status='failed',
                        error=f'JSON parse / validation: {type(e).__name__}: {e}',
                        cost_usd=cost_usd, latency_ms=latency_ms)

    # 6. L2 post-validation (mechanism only)
    parsed.mechanisms = _validate_l2_substring(parsed.mechanisms, chat_text_for_l2)

    # 7. Transaction-effect: extraction + mechanisms + event
    try:
        extraction = create_chat_extraction(
            chat_id=chat_id,
            schema_version=G1_SCHEMA_VERSION,
            open_questions=[q.model_dump() for q in parsed.open_questions],
            methods=[m.model_dump() for m in parsed.methods],
            frameworks=[f.model_dump() for f in parsed.frameworks],
            vocabulary=[v.model_dump() for v in parsed.vocabulary],
            raw_llm_response=raw_text,
        )
        mechanism_ids: list[int] = []
        for m in parsed.mechanisms:
            mech = create_mechanism(
                extraction_id=extraction['id'],
                statement=m.statement,
                position_in_chat=m.position_in_chat,
            )
            mechanism_ids.append(mech['id'])

        # paradigm visible signature — first event
        # NOTE: ref_table CHECK list (ledger v0.6) does not include 'chat_extractions',
        # so if no mechanisms, skip event (no valid ref_id available).
        event_id = None
        if mechanism_ids:
            n_mech = len(parsed.mechanisms)
            n_q = len(parsed.open_questions)
            event = create_brain_event(
                chat_id=chat_id,
                event_type='extraction_completed',
                ref_table='mechanisms',
                ref_id=mechanism_ids[0],
                summary_text=f'✨ mechanism {n_mech}개 + open question {n_q}개 추출됨',
                icon='✨',
                visible_to_user=True,
            )
            event_id = event['id']

    except Exception as e:
        _log_llm_call(chat_id, input_tokens, output_tokens, cost_usd, latency_ms,
                      'failed_db_insert', error=str(e))
        return G1Result(chat_id=chat_id, status='failed',
                        error=f'DB insert: {type(e).__name__}: {e}',
                        cost_usd=cost_usd, latency_ms=latency_ms)

    # 8. Cost log
    _log_llm_call(chat_id, input_tokens, output_tokens, cost_usd, latency_ms,
                  'completed')

    return G1Result(
        chat_id=chat_id,
        status='completed',
        extraction_id=extraction['id'],
        mechanism_ids=mechanism_ids,
        event_id=event_id,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
