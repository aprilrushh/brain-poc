"""
Benchmark Wikipedia Brain - recall latency + 의미집중도 (top1 score).

PDF 주장 검증:
  - GH200 영어 위키 508만: 6.96 ms recall
  - top1 의미집중도: 0.9809
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sentence_transformers import SentenceTransformer

from src.brain import BrainMemory, auto_device, benchmark_recall


BENCHMARK_QUERIES = [
    "Albert Einstein",
    "Theory of general relativity",
    "Capital of France",
    "Who is Marie Curie?",
    "How does photosynthesis work?",
    "Mount Everest height",
    "World War II",
    "Quantum entanglement",
    "Python programming language",
    "Sherlock Holmes Conan Doyle",
    "셜록 홈즈 누구",
    "양자역학",
]


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    index_dir = data_dir / "wiki_index"
    log_dir = Path(os.environ.get("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device()
    print(f"Device: {device}")

    print(f"\nLoading Brain from {index_dir}/...")
    t0 = time.perf_counter()
    brain = BrainMemory.load(str(index_dir), device=device)
    print(f"  loaded {len(brain):,} docs in {time.perf_counter() - t0:.1f}s")

    print(f"\nLoading encoder...")
    encoder = SentenceTransformer("BAAI/bge-m3", device=device)

    print(f"\n=== Recall benchmark (50 iter per query) ===")
    results = []
    for q in BENCHMARK_QUERIES:
        q_emb = encoder.encode([q], convert_to_tensor=True, normalize_embeddings=True).float()
        avg_ms = benchmark_recall(brain, q_emb, top_k=5, n_iter=50)
        recall = brain.recall(q_emb, top_k=5)
        top1_score = recall[0]["score"]
        top1_title = recall[0]["doc"].get("title", "?")
        gap = top1_score - recall[1]["score"] if len(recall) > 1 else 0.0
        print(f"  '{q[:40]:40}' | {avg_ms:6.3f} ms | top1={top1_score:.4f} (gap={gap:.4f}) | {top1_title[:30]}")
        results.append({
            "query": q,
            "avg_recall_ms": round(avg_ms, 3),
            "top1_score": top1_score,
            "top1_title": top1_title,
            "gap_top1_top2": gap,
        })

    avg_recall = sum(r["avg_recall_ms"] for r in results) / len(results)
    avg_top1 = sum(r["top1_score"] for r in results) / len(results)
    sharp_count = sum(1 for r in results if r["top1_score"] > 0.5)

    print(f"\n=== Summary ===")
    print(f"  n_docs: {len(brain):,}")
    print(f"  avg recall: {avg_recall:.3f} ms")
    print(f"  avg top1 score (의미집중도): {avg_top1:.4f}")
    print(f"  sharp queries (top1 > 0.5): {sharp_count}/{len(results)}")
    print(f"\nPDF 주장 비교:")
    print(f"  - GH200 6.96ms vs ours: {avg_recall:.3f}ms ({'일치' if avg_recall < 20 else '차이'})")
    print(f"  - 의미집중도 0.9809 vs ours: {avg_top1:.4f} ({'근접' if avg_top1 > 0.7 else '차이'})")

    log_path = log_dir / f"wiki_bench_{int(time.time())}.json"
    with open(log_path, "w") as f:
        json.dump({
            "device": device,
            "n_docs": len(brain),
            "avg_recall_ms": avg_recall,
            "avg_top1_score": avg_top1,
            "sharp_count": sharp_count,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved log to {log_path}")


if __name__ == "__main__":
    main()
