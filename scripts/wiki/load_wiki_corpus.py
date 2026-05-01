"""
Encode Wikipedia articles into Brain index using BGE-M3.

Strategy: 1 chunk per article = title + first 1024 chars of text.
Output: data/wiki_index/{keys.pt, docs.json, meta.json}
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
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.brain import BrainMemory, auto_device


DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
MAX_CHUNK_CHARS = 1024
BATCH_SIZE = 256
ENCODER = "BAAI/bge-m3"


def article_to_chunk(row: dict) -> dict:
    """1 chunk per article: title + first 1024 chars."""
    title = row.get("title", "")
    text = row.get("text", "") or ""
    chunk_text = (title + ". " + text)[:MAX_CHUNK_CHARS].strip()
    return {
        "id": row.get("id"),
        "title": title,
        "url": row.get("url", ""),
        "text": chunk_text,
    }


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    cache_dir = data_dir / "hf_cache"
    index_dir = data_dir / "wiki_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device()
    print(f"Device: {device}")
    print(f"Loading dataset (cached at {cache_dir})...")
    t0 = time.perf_counter()
    ds = load_dataset(DATASET, CONFIG, cache_dir=str(cache_dir), split="train")
    print(f"  loaded {len(ds):,} articles in {time.perf_counter() - t0:.0f}s")

    print(f"\nLoading encoder {ENCODER}...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER, device=device)
    print(f"  loaded in {time.perf_counter() - t0:.0f}s")

    print(f"\nBuilding chunks (1 per article, max {MAX_CHUNK_CHARS} chars)...")
    docs = []
    texts = []
    for row in tqdm(ds, total=len(ds), desc="chunking"):
        chunk = article_to_chunk(row)
        if not chunk["text"]:
            continue
        docs.append(chunk)
        texts.append(chunk["text"])
    print(f"  total chunks: {len(docs):,}")

    print(f"\nEncoding {len(texts):,} chunks (batch={BATCH_SIZE}) on {device}...")
    t0 = time.perf_counter()
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).float()
    enc_elapsed = time.perf_counter() - t0
    print(f"  encoded in {enc_elapsed/60:.1f} min, shape={tuple(embeddings.shape)}")
    print(f"  rate: {len(texts)/enc_elapsed:.0f} chunks/sec")

    print(f"\nSaving Brain index...")
    brain = BrainMemory(keys=embeddings, docs=docs, beta=50.0, device=device)
    brain.save(str(index_dir))
    keys_size_gb = (index_dir / "keys.pt").stat().st_size / 1e9
    docs_size_mb = (index_dir / "docs.json").stat().st_size / 1e6
    print(f"  keys.pt: {keys_size_gb:.2f} GB")
    print(f"  docs.json: {docs_size_mb:.0f} MB")
    print(f"  total chunks: {len(brain):,} x {embeddings.shape[-1]}")
    print(f"\nSaved index to {index_dir}/")


if __name__ == "__main__":
    main()
