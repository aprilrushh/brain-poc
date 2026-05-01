"""
Document loader: PDF + TXT -> chunks for Brain encoding.

Strategy:
- PDF: pypdf로 page별 텍스트 추출 (page 번호 보존)
- TXT: utf-8 그대로
- Chunking: sliding window 800c + 200c overlap
- Per-file metadata: filename, page (PDF only), chunk_idx
"""
from __future__ import annotations
import io
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union


CHUNK_CHARS = 800
OVERLAP_CHARS = 200
MAX_CHUNKS_PER_FILE = 5000  # safety cap (covers ~500MB PDF at avg density)


@dataclass
class Chunk:
    text: str
    filename: str
    page: Optional[int]      # PDF only, None for TXT
    chunk_idx: int           # within file
    char_start: int          # within file or page

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pdf(file_bytes: bytes, filename: str) -> List[tuple]:
    """Extract per-page text from PDF bytes. Returns [(page_num, text), ...]."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf required. Install: pip install pypdf")

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
        except Exception as e:
            print(f"[pdf-extract] {filename} page {i}: {e}")
            continue
    return pages


def extract_txt(file_bytes: bytes) -> str:
    """Decode TXT bytes. Try utf-8, fallback latin-1."""
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode('utf-8', errors='replace')


def sliding_chunks(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[tuple]:
    """Sliding window chunks. Returns [(char_start, chunk_text), ...]."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_chars:
        return [(0, text)]

    stride = chunk_chars - overlap
    chunks = []
    pos = 0
    while pos < len(text):
        chunk = text[pos:pos + chunk_chars].strip()
        if chunk:
            chunks.append((pos, chunk))
        pos += stride
    return chunks


def load_file(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Main entry. Detects file type, extracts text, chunks.
    Raises ValueError for unsupported types.
    """
    name_lower = filename.lower()
    chunks: List[Chunk] = []

    if name_lower.endswith('.pdf'):
        pages = extract_pdf(file_bytes, filename)
        for page_num, page_text in pages:
            for char_start, chunk_text in sliding_chunks(page_text):
                chunks.append(Chunk(
                    text=chunk_text,
                    filename=filename,
                    page=page_num,
                    chunk_idx=len(chunks),
                    char_start=char_start,
                ))
                if len(chunks) >= MAX_CHUNKS_PER_FILE:
                    print(f"[load_file] {filename}: hit MAX_CHUNKS_PER_FILE={MAX_CHUNKS_PER_FILE}, truncating")
                    return chunks

    elif name_lower.endswith('.txt') or name_lower.endswith('.md'):
        text = extract_txt(file_bytes)
        for char_start, chunk_text in sliding_chunks(text):
            chunks.append(Chunk(
                text=chunk_text,
                filename=filename,
                page=None,
                chunk_idx=len(chunks),
                char_start=char_start,
            ))
            if len(chunks) >= MAX_CHUNKS_PER_FILE:
                print(f"[load_file] {filename}: hit MAX_CHUNKS_PER_FILE={MAX_CHUNKS_PER_FILE}, truncating")
                return chunks
    else:
        raise ValueError(f"Unsupported file type: {filename} (only .pdf, .txt, .md)")

    return chunks


def estimate_encoding_seconds(num_chunks: int, rate: float = 247.0) -> float:
    """Rough ETA based on H100 BGE-M3 throughput (247 chunks/sec proven)."""
    return num_chunks / rate


# Quick test entry
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.document_loader <file.pdf|file.txt>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Not found: {p}")
        sys.exit(1)

    file_bytes = p.read_bytes()
    chunks = load_file(file_bytes, p.name)
    print(f"File: {p.name} ({len(file_bytes):,} bytes)")
    print(f"Chunks: {len(chunks)}")
    print(f"Estimated encoding time on H100: {estimate_encoding_seconds(len(chunks)):.1f}s")
    if chunks:
        print(f"\n--- First chunk ---")
        c = chunks[0]
        print(f"page={c.page} idx={c.chunk_idx} char_start={c.char_start}")
        print(f"text[:300]: {c.text[:300]}")
        print(f"\n--- Last chunk ---")
        c = chunks[-1]
        print(f"page={c.page} idx={c.chunk_idx} char_start={c.char_start}")
        print(f"text[:300]: {c.text[:300]}")
