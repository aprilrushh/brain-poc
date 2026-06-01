"""
Phase 1 Evaluation Runner.

Loads Brain (wiki_index) + 100 queries → measures 8 metrics → saves JSON + markdown report.

Usage:
    python3 eval/run_eval.py
    python3 eval/run_eval.py --limit 20         # quick test
    python3 eval/run_eval.py --skip-llm         # retrieval-only metrics
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before any module that reads os.environ
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

import torch
from sentence_transformers import SentenceTransformer

from src.brain import BrainMemory, auto_device, benchmark_recall
from src.orchestrator import RAGOrchestrator


REFUSAL_PATTERNS = [
    r"don't have information",
    r"do not have information",
    r"not in the provided sources",
    r"cannot answer",
    r"can't answer",
    r"no information",
    r"unable to answer",
]


def _pctl(sorted_list, p):
    if not sorted_list:
        return 0
    k = int(round(p * (len(sorted_list) - 1)))
    return sorted_list[min(k, len(sorted_list) - 1)]


def _median(vals):
    if not vals:
        return 0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def matches_keywords(text: str, keywords: list) -> tuple[bool, list]:
    """Case-insensitive substring match. Returns (any_match, matched_list)."""
    if not text or not keywords:
        return False, []
    lower = text.lower()
    matched = [k for k in keywords if k.lower() in lower]
    return (len(matched) > 0), matched


def is_refusal(text: str) -> bool:
    """Check if answer is a 'don't know' style refusal."""
    if not text:
        return False
    lower = text.lower()
    return any(re.search(p, lower) for p in REFUSAL_PATTERNS)


def article_match(retrieved: list, expected: list) -> tuple[bool, int]:
    """Check if any expected article title matches any retrieved doc title.
    Returns (any_match, best_rank). best_rank=999 if no match."""
    if not expected:
        return False, 999
    for r in retrieved:
        title = r["doc"].get("title", "").lower()
        for exp in expected:
            if exp.lower() in title or title in exp.lower():
                return True, r["rank"]
    return False, 999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only N queries (for quick test)")
    parser.add_argument("--skip-llm", action="store_true", help="Retrieval metrics only, skip LLM calls")
    parser.add_argument("--queries", default="eval/queries_v1.json")
    parser.add_argument("--index", default="data/wiki_index")
    args = parser.parse_args()

    # Setup
    device = auto_device()
    print(f"Device: {device}")

    queries_path = ROOT / args.queries
    index_dir = ROOT / args.index
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    queries = json.loads(queries_path.read_text())
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} queries from {queries_path.name}")

    # Load Brain
    print(f"Loading Brain from {index_dir}/...")
    t0 = time.perf_counter()
    brain = BrainMemory.load(str(index_dir), device=device)
    print(f"  loaded {len(brain):,} docs in {time.perf_counter() - t0:.1f}s")

    # Load encoder
    print("Loading encoder BAAI/bge-m3...")
    encoder = SentenceTransformer("BAAI/bge-m3", device=device)

    # Setup orchestrator (skip if --skip-llm)
    orchestrator = None
    if not args.skip_llm:
        orchestrator = RAGOrchestrator(brain=brain, encoder=encoder, top_k=10)
        print(f"LLM model: {orchestrator.model}")

    # Run evaluation
    print(f"\n=== Running evaluation on {len(queries)} queries ===\n")
    results = []
    t_start = time.perf_counter()

    for i, q in enumerate(queries, 1):
        result = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "language": q["language"],
            "answerable": q["answerable"],
            "query": q["query"],
        }

        # 1. Encode query
        q_emb = encoder.encode(
            [q["query"]], convert_to_tensor=True, normalize_embeddings=True
        ).float()

        # 2. Brain recall + latency benchmark (50 iter)
        recall_ms = benchmark_recall(brain, q_emb, top_k=5, n_iter=50)
        retrieved = brain.recall(q_emb, top_k=5)
        result["recall_ms"] = round(recall_ms, 3)
        result["top1_score"] = retrieved[0]["score"]
        result["top1_title"] = retrieved[0]["doc"].get("title", "?")
        result["top5_titles"] = [r["doc"].get("title", "?") for r in retrieved]

        # 3. Retrieval metrics (article match)
        any_match, best_rank = article_match(retrieved, q.get("expected_source_articles", []))
        result["expected_article_in_top5"] = any_match
        result["expected_article_rank"] = best_rank if any_match else None
        result["mrr"] = (1.0 / best_rank) if any_match else 0.0

        # 4. LLM answer (skip if --skip-llm)
        if orchestrator is not None:
            try:
                t0 = time.perf_counter()
                ttft_ms = None
                brain_done_ms = None
                brain_chunks = []
                answer = ""
                completion_tokens = 0
                for kind, val in orchestrator.query_stream(q["query"], max_tokens=512):
                    if kind == "brain_chunk":
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        brain_chunks.append(val)
                    elif kind == "brain_done":
                        brain_done_ms = (time.perf_counter() - t0) * 1000
                        completion_tokens = (val or {}).get("output_tokens") or 0
                    elif kind == "meta":
                        answer = (val or {}).get("answer", "") or "".join(brain_chunks).strip()
                total_ms = (time.perf_counter() - t0) * 1000
                result["answer"] = answer
                result["ttft_ms"] = round(ttft_ms, 1) if ttft_ms is not None else None
                result["brain_done_ms"] = round(brain_done_ms, 1) if brain_done_ms is not None else None
                result["total_ms"] = round(total_ms, 1)
                result["gen_ms"] = round(total_ms, 1)
                result["completion_tokens"] = completion_tokens

                # Keyword match
                keyword_match, matched = matches_keywords(answer, q.get("expected_answer_keywords", []))
                result["keyword_match"] = keyword_match
                result["matched_keywords"] = matched
                result["is_refusal"] = is_refusal(answer)

                # Hallucination + false-refusal logic
                if q["answerable"]:
                    # Should answer correctly
                    result["correct"] = keyword_match and not is_refusal(answer)
                    result["false_refusal"] = is_refusal(answer)  # refused when shouldn't
                    result["hallucination"] = False  # n/a for answerable
                else:
                    # Should refuse (or note false premise)
                    result["correct"] = is_refusal(answer) or keyword_match
                    # hallucination = answered confidently without refusal AND no recognized signal keyword
                    result["hallucination"] = (not is_refusal(answer)) and (not keyword_match)
                    result["false_refusal"] = False

            except Exception as e:
                result["error"] = str(e)
                result["answer"] = ""
                result["correct"] = False

        # Print progress
        status = "✓" if result.get("correct") else "✗"
        ar = "ANS" if q["answerable"] else "REF"
        ans_preview = (result.get("answer") or "")[:60].replace("\n", " ")
        print(f"  [{i:3}/{len(queries)}] {status} {q['id']:18} ({q['category'][:12]:12} {q['difficulty']:6} {q['language']:2} {ar}) "
              f"recall={result['recall_ms']:5.2f}ms top1={result['top1_score']:.3f} | {ans_preview}")

        results.append(result)

    elapsed_total = time.perf_counter() - t_start

    # Aggregate metrics
    print(f"\n=== Aggregating ({elapsed_total/60:.1f} min) ===\n")
    agg = compute_aggregates(results)
    print_summary(agg, len(brain))

    # Save JSON
    timestamp = int(time.time())
    out_json = log_dir / f"eval_phase1_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "device": device,
            "n_docs": len(brain),
            "n_queries": len(results),
            "elapsed_sec": round(elapsed_total, 1),
            "skip_llm": args.skip_llm,
            "aggregates": agg,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {out_json}")

    # Save markdown report
    out_md = log_dir / f"eval_phase1_{timestamp}.md"
    with open(out_md, "w") as f:
        f.write(format_markdown_report(agg, len(brain), results, elapsed_total))
    print(f"Saved Markdown: {out_md}")


def compute_aggregates(results: list) -> dict:
    """Compute per-category and overall metrics."""
    by_cat = defaultdict(list)
    overall = []
    for r in results:
        by_cat[r["category"]].append(r)
        overall.append(r)

    def stats(rs):
        n = len(rs)
        if n == 0:
            return {}
        recall_list = [r["recall_ms"] for r in rs]
        top1_list = [r["top1_score"] for r in rs]
        n_retrieval_match = sum(1 for r in rs if r["expected_article_in_top5"])
        mrr_list = [r["mrr"] for r in rs]
        n_correct = sum(1 for r in rs if r.get("correct"))
        n_hallucination = sum(1 for r in rs if r.get("hallucination"))
        n_false_refusal = sum(1 for r in rs if r.get("false_refusal"))
        gen_list = [r.get("gen_ms", 0) for r in rs if r.get("gen_ms")]
        ttft_list = [r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None]
        bdone_list = [r["brain_done_ms"] for r in rs if r.get("brain_done_ms") is not None]
        SPIKE_MS = 3000.0
        ttft_sorted = sorted(ttft_list)
        n_spike = sum(1 for v in ttft_list if v > SPIKE_MS)
        comp_tokens = [r.get("completion_tokens", 0) for r in rs if r.get("completion_tokens")]
        return {
            "n": n,
            "avg_recall_ms": round(sum(recall_list) / n, 3),
            "avg_top1_score": round(sum(top1_list) / n, 4),
            "retrieval_recall_at5": round(n_retrieval_match / n * 100, 1),
            "mrr": round(sum(mrr_list) / n, 3),
            "answer_accuracy_pct": round(n_correct / n * 100, 1),
            "hallucination_pct": round(n_hallucination / n * 100, 1),
            "false_refusal_pct": round(n_false_refusal / n * 100, 1),
            "avg_gen_ms": round(sum(gen_list) / max(len(gen_list), 1), 1) if gen_list else 0,
            "avg_ttft_ms": round(sum(ttft_list) / max(len(ttft_list), 1), 1) if ttft_list else 0,
            "avg_brain_done_ms": round(sum(bdone_list) / max(len(bdone_list), 1), 1) if bdone_list else 0,
            "med_ttft_ms": round(_median(ttft_list), 1) if ttft_list else 0,
            "p95_ttft_ms": round(_pctl(ttft_sorted, 0.95), 1) if ttft_list else 0,
            "max_ttft_ms": round(ttft_sorted[-1], 1) if ttft_list else 0,
            "med_brain_done_ms": round(_median(bdone_list), 1) if bdone_list else 0,
            "spike_count": n_spike,
            "spike_rate": round(100.0 * n_spike / max(len(ttft_list), 1), 1) if ttft_list else 0,
            "avg_completion_tokens": round(sum(comp_tokens) / max(len(comp_tokens), 1), 0) if comp_tokens else 0,
        }

    return {
        "overall": stats(overall),
        "by_category": {cat: stats(rs) for cat, rs in by_cat.items()},
    }


def print_summary(agg: dict, n_docs: int):
    o = agg["overall"]
    print(f"--- Overall (n_docs={n_docs:,}) ---")
    print(f"  avg_recall_ms:        {o['avg_recall_ms']:.3f} ms (PDF claim: 6.96ms)")
    print(f"  avg_top1_score:       {o['avg_top1_score']:.4f} (PDF claim: 0.9809)")
    print(f"  retrieval_recall@5:   {o['retrieval_recall_at5']}%")
    print(f"  MRR:                  {o['mrr']:.3f}")
    print(f"  answer_accuracy:      {o['answer_accuracy_pct']}%")
    print(f"  hallucination_rate:   {o['hallucination_pct']}%")
    print(f"  false_refusal_rate:   {o['false_refusal_pct']}%")
    print(f"  avg_ttft_ms:          {o.get('avg_ttft_ms', 0)} ms  (첫 토큰, Zero)")
    print(f"  med_ttft_ms:          {o.get('med_ttft_ms', 0)} ms  (중앙값=진짜 대표값)")
    print(f"  p95/max_ttft_ms:      {o.get('p95_ttft_ms', 0)} / {o.get('max_ttft_ms', 0)} ms  (꼬리=spike)")
    print(f"  spike(>3s):           {o.get('spike_count', 0)} ({o.get('spike_rate', 0)}%)  ← Together serverless")
    print(f"  avg_brain_done_ms:    {o.get('avg_brain_done_ms', 0)} ms  (Zero 완성)")
    print(f"  avg_gen_ms (total):   {o['avg_gen_ms']} ms  (Zero+General 완성)")
    print(f"  avg_completion_tokens: {o['avg_completion_tokens']}")
    print()
    print(f"--- By Category ---")
    for cat, s in agg["by_category"].items():
        print(f"  {cat:25} n={s['n']:3} | acc={s['answer_accuracy_pct']:5.1f}% | "
              f"recall@5={s['retrieval_recall_at5']:5.1f}% | "
              f"top1={s['avg_top1_score']:.3f} | hallucination={s['hallucination_pct']:5.1f}% | "
              f"false_refusal={s['false_refusal_pct']:5.1f}%")


def format_markdown_report(agg: dict, n_docs: int, results: list, elapsed: float) -> str:
    o = agg["overall"]
    md = []
    md.append(f"# Phase 1 Evaluation Report\n")
    md.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- Brain n_docs: **{n_docs:,}**")
    md.append(f"- Queries: {len(results)}")
    md.append(f"- Elapsed: {elapsed/60:.1f} min\n")
    md.append(f"## Overall Metrics\n")
    md.append(f"| Metric | Value | PDF Claim |")
    md.append(f"|---|---|---|")
    md.append(f"| Avg recall latency | **{o['avg_recall_ms']:.3f} ms** | 6.96 ms |")
    md.append(f"| Avg top1 score (의미집중도) | **{o['avg_top1_score']:.4f}** | 0.9809 |")
    md.append(f"| Retrieval Recall@5 | **{o['retrieval_recall_at5']}%** | — |")
    md.append(f"| MRR | **{o['mrr']:.3f}** | — |")
    md.append(f"| Answer accuracy | **{o['answer_accuracy_pct']}%** | — |")
    md.append(f"| Hallucination rate | **{o['hallucination_pct']}%** | < 10% target |")
    md.append(f"| False-refusal rate | **{o['false_refusal_pct']}%** | < 10% target |")
    md.append(f"| Avg TTFT (Zero 첫 토큰) | {o.get('avg_ttft_ms', 0)} ms | < 1000 |")
    md.append(f"| Median TTFT | {o.get('med_ttft_ms', 0)} ms | < 500 |")
    md.append(f"| p95 / max TTFT | {o.get('p95_ttft_ms', 0)} / {o.get('max_ttft_ms', 0)} ms | — |")
    md.append(f"| Spike rate (>3s) | {o.get('spike_count', 0)} ({o.get('spike_rate', 0)}%) | < 5% |")
    md.append(f"| Avg Zero 완성 (brain_done) | {o.get('avg_brain_done_ms', 0)} ms | — |")
    md.append(f"| Avg total (Zero+General) | {o['avg_gen_ms']} ms | — |")
    md.append(f"\n## By Category\n")
    md.append(f"| Category | n | Accuracy | Recall@5 | Top1 | Hallucination | False-Refusal |")
    md.append(f"|---|---|---|---|---|---|---|")
    for cat, s in agg["by_category"].items():
        md.append(f"| {cat} | {s['n']} | {s['answer_accuracy_pct']}% | {s['retrieval_recall_at5']}% | "
                  f"{s['avg_top1_score']:.3f} | {s['hallucination_pct']}% | {s['false_refusal_pct']}% |")
    return "\n".join(md)


if __name__ == "__main__":
    main()
