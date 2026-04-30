"""
Download 4 Sherlock Holmes books from Project Gutenberg.

Saves to: data/corpus/raw/<title>.txt
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import requests


BOOKS = [
    {"id": 244,  "title": "A_Study_in_Scarlet"},
    {"id": 2097, "title": "The_Sign_of_the_Four"},
    {"id": 1661, "title": "The_Adventures_of_Sherlock_Holmes"},
    {"id": 2852, "title": "The_Hound_of_the_Baskervilles"},
]

URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36 BrainProject/0.1",
}


def download_one(book: dict, out_dir: Path) -> bool:
    out_path = out_dir / f"{book['title']}.txt"
    if out_path.exists() and out_path.stat().st_size > 50_000:
        print(f"  [SKIP] {book['title']} already exists ({out_path.stat().st_size:,} bytes)")
        return True

    last_err = None
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(id=book["id"])
        try:
            print(f"  GET  {url}")
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 50_000:
                out_path.write_bytes(r.content)
                print(f"  [OK]   {book['title']}: {len(r.content):,} bytes")
                return True
            else:
                last_err = f"HTTP {r.status_code} or too small ({len(r.content)} bytes)"
        except Exception as e:
            last_err = str(e)
    print(f"  [FAIL] {book['title']}: {last_err}")
    return False


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    out_dir = data_dir / "corpus" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(BOOKS)} books to {out_dir}/")
    print("=" * 60)

    ok = 0
    for book in BOOKS:
        if download_one(book, out_dir):
            ok += 1

    print("=" * 60)
    print(f"Result: {ok}/{len(BOOKS)} downloaded")
    sys.exit(0 if ok == len(BOOKS) else 1)


if __name__ == "__main__":
    main()
