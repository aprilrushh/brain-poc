"""
Download HuggingFace wikimedia/wikipedia (English) dataset.
Saves to: data/hf_cache/ (parquet shards)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    cache_dir = data_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET} / {CONFIG}")
    print(f"Cache: {cache_dir}")
    print("(첫 실행은 20GB 다운로드 - 30-60분 소요)")

    t0 = time.perf_counter()
    ds = load_dataset(
        DATASET,
        CONFIG,
        cache_dir=str(cache_dir),
        num_proc=os.cpu_count() or 4,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nLoaded in {elapsed:.0f}s")
    print(f"Splits: {list(ds.keys())}")
    train = ds["train"]
    print(f"Train rows: {len(train):,}")
    print(f"Columns: {train.column_names}")
    print(f"\nFirst row sample:")
    row = train[0]
    print(f"  id: {row.get('id')}")
    print(f"  title: {row.get('title')}")
    print(f"  url: {row.get('url')}")
    print(f"  text (first 200 chars): {row.get('text', '')[:200]}...")
    print(f"\nDataset ready at {cache_dir}")


if __name__ == "__main__":
    main()
