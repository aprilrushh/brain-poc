#!/usr/bin/env python3
"""
phase_d_mitigation_warmup_threads.py — D-7.3.2 multi-thread warmup sweep.

Single-thread warmup (D-7.3.1 T3): 0.66 GiB/s = 8.6% of fio ceiling 7.43 GiB/s.
Sweep n_threads ∈ {4, 8, 16}, measure throughput.

Strategy: 30 safetensors files distributed across N worker threads, each
reading sequentially. Threads share global counter for total bytes.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots"
RESULTS_DIR = Path.home() / "xhbm" / "results"
THREAD_SWEEP = [4, 8, 16]


def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    log("drop_caches ...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    time.sleep(1)


def buff_cache_GB():
    out = subprocess.check_output(["free", "-b"]).decode()
    return int(out.split("\n")[1].split()[5]) / (1024**3)


def read_one_file(path):
    """Sequential read one file, return bytes."""
    total = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024 * 1024)  # 64 MB
            if not chunk:
                break
            total += len(chunk)
    return total


def warmup_parallel(n_threads):
    snapshot_dir = list(MODEL_DIR.glob("*"))[0]
    sf_files = sorted(snapshot_dir.glob("model-*-of-*.safetensors"))
    real_paths = [f.resolve() for f in sf_files]
    log(f"  parallel warmup: {len(real_paths)} files, {n_threads} threads")
    t0 = time.perf_counter()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for nb in ex.map(read_one_file, real_paths):
            total_bytes += nb
    elapsed = time.perf_counter() - t0
    throughput = (total_bytes / (1024**3)) / elapsed
    return {
        "n_threads": n_threads,
        "elapsed_sec": elapsed,
        "total_GiB": total_bytes / (1024**3),
        "throughput_GiBps": throughput,
        "ceiling_pct": throughput / 7.43 * 100,
    }


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mitigation_warmup_threads_{run_id}.json"
    results = {"run_id": run_id, "single_thread_baseline_sec": 200.1, "trials": []}

    for n in THREAD_SWEEP:
        drop_caches()
        bc = buff_cache_GB()
        log(f"=== n_threads={n}: buff/cache before = {bc:.1f} GB ===")
        r = warmup_parallel(n)
        bc_after = buff_cache_GB()
        r["buff_cache_after_GB"] = bc_after
        log(f"=== n={n}: {r['elapsed_sec']:.1f}s @ {r['throughput_GiBps']:.2f} GiB/s "
            f"({r['ceiling_pct']:.1f}% of fio ceiling); buff/cache: {bc:.1f}→{bc_after:.1f} GB ===")
        results["trials"].append(r)

    out_path.write_text(json.dumps(results, indent=2))
    log(f"DONE — wrote {out_path}")
    log("=== Summary ===")
    log(f"  Single thread (D-7.3.1 T3): 200.1s @ 0.66 GiB/s  (8.6% ceiling)")
    for r in results["trials"]:
        log(f"  {r['n_threads']:>2} threads: {r['elapsed_sec']:.1f}s @ {r['throughput_GiBps']:.2f} GiB/s  ({r['ceiling_pct']:.1f}% ceiling)")


if __name__ == "__main__":
    main()
