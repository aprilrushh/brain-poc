"""
v2: Fine-grained chunking — sliding window 800 chars + 200 overlap, max 10 chunks/article.

Goal: improve top1_score (의미집중도) from 0.27 (v1, article-level) to 0.6+ (article-body level).

Output: data/wiki_index_v2/{keys.pt, docs.json, meta.json}
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
CHUNK_CHARS = 800
OVERLAP_CHARS = 200
MAX_CHUNKS_PER_ARTICLE = 10
BATCH_SIZE = 256
ENCODER = "BAAI/bge-m3"


def article_to_chunks(row: dict) -> list:
    """Sliding window chunks for one article. First chunk includes title for context."""
    title = row.get("title", "")
    text = row.get("text", "") or ""
    if not text.strip():
        return []

    chunks = []
    stride = CHUNK_CHARS - OVERLAP_CHARS  # 600 chars per stride
    pos = 0
    chunk_idx = 0
    article_id = row.get("id", "")
    url = row.get("url", "")

    while pos < len(text) and chunk_idx < MAX_CHUNKS_PER_ARTICLE:
        chunk_text = text[pos : pos + CHUNK_CHARS]
        # Prepend title for first chunk so context is preserved
        if chunk_idx == 0:
            chunk_text = title + ". " + chunk_text
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append({
                "id": f"{article_id}_{chunk_idx}",
                "article_id": article_id,
                "chunk_idx": chunk_idx,
                "title": title,
                "url": url,
                "text": chunk_text[:CHUNK_CHARS],  # cap final length
            })
        pos += stride
        chunk_idx += 1

    return chunks


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    cache_dir = data_dir / "hf_cache"
    index_dir = data_dir / "wiki_index_v2"
    index_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device()
    print(f"Device: {device}")
    print(f"Strategy: sliding window {CHUNK_CHARS}c + {OVERLAP_CHARS}c overlap, max {MAX_CHUNKS_PER_ARTICLE}/article")

    print(f"Loading dataset (cached at {cache_dir})...")
    t0 = time.perf_counter()
    ds = load_dataset(DATASET, CONFIG, cache_dir=str(cache_dir), split="train")
    print(f"  loaded {len(ds):,} articles in {time.perf_counter() - t0:.0f}s")

    print(f"\nBuilding fine-grained chunks...")
    t0 = time.perf_counter()
    docs = []
    texts = []
    chunk_count_per_article = []
    for row in tqdm(ds, total=len(ds), desc="chunking"):
        chunks = article_to_chunks(row)
        chunk_count_per_article.append(len(chunks))
        for c in chunks:
            docs.append(c)
            texts.append(c["text"])
    chunk_elapsed = time.perf_counter() - t0
    avg_chunks = sum(chunk_count_per_article) / max(len(chunk_count_per_article), 1)
    print(f"  total chunks: {len(docs):,} (avg {avg_chunks:.1f}/article) in {chunk_elapsed/60:.1f} min")
    print(f"  expected fp16 HBM: {len(docs) * 1024 * 2 / 1e9:.1f} GB")

    print(f"\nLoading encoder {ENCODER}...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER, device=device)
    print(f"  loaded in {time.perf_counter() - t0:.0f}s")

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
