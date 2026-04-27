#!/usr/bin/env python3
"""
phase_d_mitigation_cold_load.py — D-7.3.1 cold-load mitigation measurement.

Three scenarios:
  T1 (cold): drop_caches → load           (server boot / restart)
  T2 (warm process): T1 + free + reload   (model swap in same process)
  T3 (warmed cache): drop_caches + sequential read all weights → load
                                          (server boot with explicit warmup)

Each scenario measures full model load time. Outputs JSON.
"""
import gc, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots"
RESULTS_DIR = Path.home() / "xhbm" / "results"


def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    log("drop_caches: sync + echo 3 > drop_caches ...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    time.sleep(1)


def buff_cache_GB():
    out = subprocess.check_output(["free", "-b"]).decode()
    line = out.split("\n")[1].split()
    return int(line[5]) / (1024**3)


def warmup_read_all_weights():
    """Sequential read all model weights to fill page cache."""
    snapshot_dir = list(MODEL_DIR.glob("*"))[0]
    sf_files = sorted(snapshot_dir.glob("model-*-of-*.safetensors"))
    log(f"warmup_read_all_weights: {len(sf_files)} files, sequential read ...")
    t0 = time.perf_counter()
    total_bytes = 0
    for f in sf_files:
        # Resolve symlink
        real = f.resolve()
        with open(real, "rb") as fh:
            while True:
                chunk = fh.read(64 * 1024 * 1024)  # 64 MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
    elapsed = time.perf_counter() - t0
    throughput_GBps = (total_bytes / (1024**3)) / elapsed
    log(f"warmup done: {total_bytes/(1024**3):.1f} GiB in {elapsed:.1f}s = {throughput_GBps:.2f} GiB/s")
    return {"warmup_seconds": elapsed, "warmup_GB": total_bytes / (1024**3),
            "warmup_throughput_GBps": throughput_GBps}


def measure_load(label):
    log(f"=== {label}: from_pretrained start ===")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    bc_before = buff_cache_GB()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        REPO, quantization_config=bnb_cfg,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.eval()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bc_after = buff_cache_GB()
    gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
    log(f"=== {label}: load OK: {elapsed:.1f}s gpu={gpu_alloc:.1f}GB buff/cache={bc_before:.1f}→{bc_after:.1f}GB ===")
    return {
        "label": label,
        "load_seconds": elapsed,
        "gpu_alloc_GB": gpu_alloc,
        "buff_cache_before_GB": bc_before,
        "buff_cache_after_GB": bc_after,
    }, model


def cleanup_model(model):
    log("cleanup: unload model ...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mitigation_cold_load_{run_id}.json"
    results = {"run_id": run_id, "scenarios": []}

    # === T1: Cold baseline ===
    drop_caches()
    log(f"buff/cache after drop: {buff_cache_GB():.1f}GB")
    r1, model = measure_load("T1_cold_baseline")
    results["scenarios"].append(r1)
    cleanup_model(model)

    # === T2: Warm process (same Python process, page cache from T1) ===
    log(f"buff/cache after T1 cleanup: {buff_cache_GB():.1f}GB")
    r2, model = measure_load("T2_warm_process")
    results["scenarios"].append(r2)
    cleanup_model(model)

    # === T3: Explicit warmup ===
    drop_caches()
    log(f"buff/cache after drop (T3): {buff_cache_GB():.1f}GB")
    warmup = warmup_read_all_weights()
    log(f"buff/cache after warmup: {buff_cache_GB():.1f}GB")
    r3, model = measure_load("T3_explicit_warmup")
    r3.update(warmup)
    results["scenarios"].append(r3)
    cleanup_model(model)

    out_path.write_text(json.dumps(results, indent=2))
    log(f"DONE — wrote {out_path}")

    # Summary
    log("=== Summary ===")
    for r in results["scenarios"]:
        log(f"  {r['label']}: {r['load_seconds']:.1f}s")


if __name__ == "__main__":
    main()
