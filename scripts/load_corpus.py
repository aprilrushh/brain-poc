"""
Load corpus: read raw .txt files, chunk, embed with BGE-M3, save Brain index.

Output: data/index/{keys.pt, docs.json, meta.json}
"""

from __future__ import annotations
import os
import re
import sys
import time
from pathlib import Path

# Add repo root to path so 'src' can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.brain import BrainMemory, auto_device


CHUNK_CHARS = 800
CHUNK_OVERLAP = 200
ENCODER_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


# Gutenberg header/footer markers
GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg license header/footer."""
    m_start = GUTENBERG_START.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = GUTENBERG_END.search(text)
    if m_end:
        text = text[:m_end.start()]
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Naive char-based chunker with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if len(chunk) >= 100:
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def page_estimate(chunk_idx: int, chunks_per_page: int = 3) -> int:
    """Crude page number estimate based on chunk index."""
    return chunk_idx // chunks_per_page + 1


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    raw_dir = data_dir / "corpus" / "raw"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files in {raw_dir}/")
        print(f"        Run scripts/download_sherlock.py first.")
        sys.exit(1)

    print(f"Found {len(txt_files)} files in {raw_dir}/")

    # Build doc list
    docs = []
    for path in txt_files:
        title = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_gutenberg(raw)
        chunks = chunk_text(cleaned)
        print(f"  {title}: {len(cleaned):,} chars -> {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            docs.append({
                "title": title,
                "page": page_estimate(i),
                "chunk_idx": i,
                "text": c,
            })

    print(f"\nTotal chunks: {len(docs)}")
    if not docs:
        print("[ERROR] No chunks produced.")
        sys.exit(1)

    # Encode
    device = auto_device()
    print(f"\nLoading encoder {ENCODER_NAME} on {device} ...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts)} chunks (batch_size={BATCH_SIZE}) ...")
    t0 = time.perf_counter()
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).float()
    print(f"  encoded in {time.perf_counter() - t0:.1f}s, shape={tuple(embeddings.shape)}")

    # Save Brain
    brain = BrainMemory(keys=embeddings, docs=docs, beta=50.0, device=device)
    brain.save(str(index_dir))
    print(f"\nSaved index to {index_dir}/")
    print(f"  - keys.pt   ({len(brain)} x {embeddings.shape[-1]})")
    print(f"  - docs.json ({len(brain)} entries)")
    print(f"  - meta.json")


if __name__ == "__main__":
    main()
