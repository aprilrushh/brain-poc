#!/usr/bin/env python3
"""
D-2.2: Compare 4 IO patterns to localize Phase B's 350 MB/s bottleneck.

Workload: 80 chunks x 128 MiB = 10 GiB total.
Mimics 70B model writing all KV cache to disk during one offload event.

fio T2 ceiling (this server): 7,604 MB/s
Phase B v2e reported:           350 MB/s (4.6% of ceiling)

Goal: identify which combination of (fsync, file-per-layer, single-thread)
caused the Phase B ceiling, and what Phase D can target.
"""
import os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

NUM_CHUNKS = 80
CHUNK_MB = 128
WORKDIR = Path.home() / "xhbm" / "fio_bench" / "d22"
WORKDIR.mkdir(parents=True, exist_ok=True)

CHUNK_BYTES = CHUNK_MB * 1024 * 1024
TOTAL_BYTES = NUM_CHUNKS * CHUNK_BYTES
TOTAL_GB = TOTAL_BYTES / (1024**3)

print(f"Workload: {NUM_CHUNKS} chunks x {CHUNK_MB} MiB = {TOTAL_GB:.2f} GiB")
print(f"fio T2 ceiling: 7,604 MB/s")
print(f"Phase B v2e:     350 MB/s\n")

# Pre-allocate buffer (mimics CPU-side KV after GPU copy)
buf = bytearray(CHUNK_BYTES)


def cleanup():
    for p in WORKDIR.glob("*.bin"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def report(name, t, expected_relation=""):
    elapsed = t
    mb_s = (TOTAL_BYTES / (1024**2)) / elapsed
    pct = 100 * mb_s / 7604
    print(f"  {name:50s}  {elapsed:6.2f}s  {mb_s:7.0f} MB/s  ({pct:4.1f}% of fio ceiling)  {expected_relation}")


# ============================================================
# Pattern A: Phase B v1/v2e replica (sync write + flush + fsync per layer)
# ============================================================
def pattern_A():
    cleanup()
    t0 = time.perf_counter()
    for i in range(NUM_CHUNKS):
        path = WORKDIR / f"a_layer_{i:03d}.bin"
        with open(path, "wb") as f:
            f.write(buf)
            f.flush()
            os.fsync(f.fileno())
    return time.perf_counter() - t0


# ============================================================
# Pattern B: Drop fsync (everything else same)
# ============================================================
def pattern_B():
    cleanup()
    t0 = time.perf_counter()
    for i in range(NUM_CHUNKS):
        path = WORKDIR / f"b_layer_{i:03d}.bin"
        with open(path, "wb") as f:
            f.write(buf)
            f.flush()
    return time.perf_counter() - t0


# ============================================================
# Pattern C: Single big file, many buffered 8MB writes (no fsync)
# ============================================================
def pattern_C():
    cleanup()
    path = WORKDIR / "c_all.bin"
    SUB = 8 * 1024 * 1024  # 8 MB sub-write
    sub_buf = bytearray(SUB)
    t0 = time.perf_counter()
    with open(path, "wb", buffering=4 * 1024 * 1024) as f:
        for _ in range(TOTAL_BYTES // SUB):
            f.write(sub_buf)
    return time.perf_counter() - t0


# ============================================================
# Pattern D: Threaded parallel (4 workers, separate files, no fsync)
# ============================================================
def write_one(i):
    path = WORKDIR / f"d_layer_{i:03d}.bin"
    with open(path, "wb") as f:
        f.write(buf)
        f.flush()


def pattern_D():
    cleanup()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(write_one, range(NUM_CHUNKS)))
    return time.perf_counter() - t0


# ============================================================
# Run
# ============================================================
print("Pattern                                                Time    Throughput")
print("-" * 90)
report("A: Phase B replica (sync + flush + fsync per layer)", pattern_A(),
       "<- match Phase B 350 MB/s expected")
report("B: drop fsync (sync + flush per layer)              ", pattern_B(),
       "<- isolates fsync overhead")
report("C: single file, 8 MB buffered writes                ", pattern_C(),
       "<- isolates per-file overhead")
report("D: 4-thread parallel, separate files (no fsync)     ", pattern_D(),
       "<- adds concurrency")

cleanup()
print("\nAll temp files cleaned up.")
