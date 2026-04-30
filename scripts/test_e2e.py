"""
End-to-end test: load Brain, run sample queries, print answers + sources.

Requires:
  - data/index/{keys.pt, docs.json, meta.json}  (from load_corpus.py)
  - .env with TOGETHER_API_KEY (or LLM_API_KEY)
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import torch
from sentence_transformers import SentenceTransformer

from src.brain import BrainMemory, auto_device, benchmark_recall
from src.orchestrator import RAGOrchestrator


SAMPLE_QUESTIONS = [
    "Where did Holmes and Watson first meet?",
    "What is 221B Baker Street?",
    "Who is Professor Moriarty?",
    "셜록 홈즈는 누구의 친구입니까?",
    "Compare the personalities of Holmes and Watson.",
    "Why did Holmes use cocaine?",
    "What happened at the Reichenbach Falls?",
    "Write a short summary of A Study in Scarlet in 3 sentences.",
]


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    index_dir = data_dir / "index"
    log_dir = Path(os.environ.get("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    if not (index_dir / "keys.pt").exists():
        print(f"[ERROR] No index in {index_dir}/. Run scripts/load_corpus.py first.")
        sys.exit(1)

    if not (os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY")):
        print("[ERROR] LLM_API_KEY or TOGETHER_API_KEY not set in .env")
        sys.exit(1)

    device = auto_device()
    print(f"Device: {device}")

    # Load Brain
    print(f"\nLoading Brain from {index_dir}/ ...")
    t0 = time.perf_counter()
    brain = BrainMemory.load(str(index_dir), device=device)
    print(f"  loaded {len(brain)} docs in {time.perf_counter() - t0:.2f}s")

    # Load encoder
    print(f"\nLoading encoder BAAI/bge-m3 ...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # Recall benchmark
    print(f"\nBenchmarking recall (10 iter) ...")
    sample_q = encoder.encode(
        ["Sherlock Holmes"], convert_to_tensor=True, normalize_embeddings=True
    ).float()
    avg_ms = benchmark_recall(brain, sample_q, top_k=5, n_iter=10)
    print(f"  avg recall: {avg_ms:.3f} ms")

    # Orchestrator
    orch = RAGOrchestrator(brain=brain, encoder=encoder, top_k=5)
    print(f"\nLLM model: {orch.model}")
    print(f"LLM mode:  {os.environ.get('LLM_MODE', 'api')}")
    print(f"LLM base:  {os.environ.get('LLM_API_BASE', 'https://api.together.xyz/v1')}")

    results = []
    print("\n" + "=" * 70)
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[{i}/{len(SAMPLE_QUESTIONS)}] Q: {q}")
        try:
            r = orch.query(q, max_tokens=2048)
            print(f"    thinking: {r['thinking_enabled']} ({r['thinking_reason']})")
            print(f"    pattern:  {r['retrieval_pattern']}")
            print(f"    timings:  embed={r['timings']['embed_ms']:.1f} ms / "
                  f"retrieve={r['timings']['retrieve_ms']:.2f} ms / "
                  f"generate={r['timings']['generate_ms']:.0f} ms")
            print(f"    top sources:")
            for src in r["retrieved"][:3]:
                print(f"      - {src['title']} p{src['page']} (score={src['score']:.4f})")
            print(f"    A: {r['answer'][:400]}{'...' if len(r['answer']) > 400 else ''}")
            results.append(r)
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e}")
            results.append({"question": q, "error": str(e)})

    # Save log
    log_path = log_dir / f"e2e_{int(time.time())}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": device,
            "n_docs": len(brain),
            "recall_avg_ms": avg_ms,
            "model": orch.model,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved log to {log_path}")
    success = sum(1 for r in results if "error" not in r)
    print(f"Success: {success}/{len(SAMPLE_QUESTIONS)}")


if __name__ == "__main__":
    main()
