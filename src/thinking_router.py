"""
Hybrid thinking router — default ON + auto OFF for simple queries.

Signals:
1. Query keyword (Korean + English)
2. Brain retrieval pattern (sharp / flat / weak)
3. Query length
"""

from __future__ import annotations
from typing import Optional, Tuple


SIMPLE_KO = ["누구", "누가", "뭐야", "뭐예요", "어디", "언제", "몇", "얼마"]
SIMPLE_EN = ["who is", "what is", "where", "when", "how many", "how much"]

COMPLEX_KO = [
    "왜", "어떻게", "이유", "비교", "차이", "분석", "정리", "요약",
    "vs", "대비", "관계", "원인", "결과", "설명", "추론", "예측",
    "계산", "풀어", "증명", "논리",
]
COMPLEX_EN = [
    "why", "how does", "compare", "difference", "analy", "explain",
    "reason", "vs", "summarize", "predict", "infer",
    "calculate", "solve", "prove", "derive",
]

CREATIVE = ["써줘", "만들어", "작성", "지어", "write", "create", "compose"]


def classify_query_complexity(query: str) -> str:
    """'simple' | 'complex' | 'creative' | 'unknown'"""
    q = query.lower().strip()

    if any(kw in q for kw in CREATIVE):
        return "creative"
    if any(kw in q for kw in COMPLEX_KO + COMPLEX_EN):
        return "complex"
    if any(kw in q for kw in SIMPLE_KO + SIMPLE_EN):
        return "simple"

    if len(q) < 20:
        return "simple"
    if len(q) > 80:
        return "complex"
    return "unknown"


def decide_thinking(
    query: str,
    retrieval_pattern: str,
    user_override: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Decide whether to enable LLM thinking mode.

    Args:
        query: user question
        retrieval_pattern: from BrainMemory.retrieval_pattern() — sharp/flat/weak
        user_override: "force_on" | "force_off" | None

    Returns:
        (thinking_enabled, reason)
    """
    if user_override == "force_on":
        return True, "user_override_on"
    if user_override == "force_off":
        return False, "user_override_off"

    q_class = classify_query_complexity(query)

    if q_class == "creative":
        return True, "creative_query"
    if q_class == "complex":
        return True, f"complex_query+{retrieval_pattern}"
    if q_class == "simple" and retrieval_pattern == "sharp":
        return False, "simple+sharp"
    if retrieval_pattern == "weak":
        return False, "weak_retrieval"

    if retrieval_pattern == "flat":
        return True, f"flat_retrieval+{q_class}"
    if retrieval_pattern == "sharp":
        return False, f"sharp_retrieval+{q_class}"

    return True, "default_on"


if __name__ == "__main__":
    samples = [
        ("셜록이 모리아티 처음 만난 곳?", "sharp"),
        ("X 와 Y 의 차이점은?", "flat"),
        ("왜 그렇게 됐나?", "flat"),
        ("hi", "weak"),
        ("Write a poem about Sherlock", "flat"),
    ]
    for q, p in samples:
        on, reason = decide_thinking(q, p)
        flag = "ON " if on else "OFF"
        print(f"[{flag}] '{q}' (pattern={p}) -> {reason}")
