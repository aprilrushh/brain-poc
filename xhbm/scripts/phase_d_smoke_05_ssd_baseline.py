#!/usr/bin/env python3
"""
phase_d_smoke_05_ssd_baseline.py — Phase D Smoke #5

Purpose: First end-to-end measurement of 70B NF4 + KV cache SSD offload.
         Compares directly to smoke_03b (RAM offload, K+V) results.

Critical comparisons:
  - HBM peak: should match smoke_03b (offload destination doesn't change peak)
  - fwd time: smoke_03b was +0.4% over baseline. SSD slower; how much?
  - bit-exact: should match smoke_03a/b at same ctx (offload is mathematically identical)

Pre-cleanup of SSD work dir on each ctx (avoid disk fill across ctx).

North Star §3 targets:
  - HBM ≥ 10%, throughput ≤ 20% degradation, bit-exact, n=3 (later)
"""
import gc, hashlib, json, shutil, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path.home() / "xhbm" / "scripts"))
from xhbm_ssd_cache import SSDOffloadCache

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
CTX_LIST = [4096, 16384, 32768, 65536, 98304, 131072]
BATCH = 1
SSD_DIR_BASE = Path.home() / "xhbm" / "fio_bench" / "smoke_05_ssd"
RESULTS_DIR = Path.home() / "xhbm" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = 4

def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)

def gpu_snapshot(tag):
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "alloc_MiB": torch.cuda.memory_allocated() / (1024**2),
        "reserved_MiB": torch.cuda.memory_reserved() / (1024**2),
        "peak_MiB": torch.cuda.max_memory_allocated() / (1024**2),
    }

def hash_tensor(t):
    arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]

def disk_used_mib(path: Path):
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except FileNotFoundError:
                pass
    return total / (1024**2)

def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"smoke_05_ssd_{run_id}.json"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    meta = {
        "script": "phase_d_smoke_05_ssd_baseline.py",
        "repo": REPO,
        "use_cache": True,
        "cache_backend": "SSDOffloadCache",
        "num_workers": NUM_WORKERS,
        "ssd_dir_base": str(SSD_DIR_BASE),
        "quantization": {
            "backend": "bitsandbytes",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
        },
        "run_id": run_id,
        "ctx_list": CTX_LIST,
        "batch": BATCH,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    log(f"meta: torch={torch.__version__} device={meta['cuda_device']} batch={BATCH} backend=SSD")
    log(f"ctx_list: {CTX_LIST}  workers={NUM_WORKERS}  ssd_dir_base={SSD_DIR_BASE}")

    torch.cuda.reset_peak_memory_stats()
    meta["pre_load"] = gpu_snapshot("pre_load")

    log(f"loading {REPO} with bnb NF4 ...")
    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            REPO,
            quantization_config=bnb_cfg,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        model.eval()
    except Exception as e:
        log(f"LOAD FAILED: {type(e).__name__}: {e}")
        meta["load"] = {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:500],
            "traceback": traceback.format_exc()[-2000:],
        }
        meta["runs"] = []
        out_path.write_text(json.dumps(meta, indent=2))
        sys.exit(1)

    load_sec = time.perf_counter() - t0
    snap = gpu_snapshot("post_load")
    meta["load"] = {
        "ok": True,
        "seconds": load_sec,
        **snap,
        "num_params": sum(p.numel() for p in model.parameters()),
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
        "num_kv_heads": getattr(model.config, "num_key_value_heads", None),
    }
    log(f"load OK: {load_sec:.1f}s peak={snap['peak_MiB']:.0f}MiB params={meta['load']['num_params']/1e9:.1f}B")

    vocab_size = model.config.vocab_size
    runs = []
    consecutive_fail = 0

    for ctx in CTX_LIST:
        log(f"=== ctx={ctx} batch={BATCH} (SSDOffloadCache, workers={NUM_WORKERS}) ===")
        run = {"ctx": ctx, "batch": BATCH}
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        run["pre_forward"] = gpu_snapshot(f"ctx{ctx}_pre")

        # Per-ctx SSD directory (cleaned each iteration)
        ssd_dir = SSD_DIR_BASE / f"ctx_{ctx}"
        if ssd_dir.exists():
            shutil.rmtree(ssd_dir)
        ssd_dir.mkdir(parents=True, exist_ok=True)

        g = torch.Generator(device="cpu").manual_seed(42 + ctx)
        input_ids = torch.randint(0, vocab_size, (BATCH, ctx), generator=g).to("cuda:0")

        cache = SSDOffloadCache(ssd_dir=ssd_dir, config=model.config, num_workers=NUM_WORKERS)

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
            torch.cuda.synchronize()
            fwd_sec = time.perf_counter() - t0
            last_logits = out.logits[:, -1, :]

            pool_stats = cache.get_pool_stats()
            disk_used = disk_used_mib(ssd_dir)

            run["forward"] = {
                "ok": True,
                "seconds": fwd_sec,
                "logit_last_hash_fp32": hash_tensor(last_logits),
                "argmax_last": int(last_logits.argmax(dim=-1)[0].item()),
                "ssd_disk_used_MiB": disk_used,
                "pool_stats": pool_stats,
            }
            run["post_forward"] = gpu_snapshot(f"ctx{ctx}_post")
            log(f"ctx={ctx} OK: fwd={fwd_sec:.2f}s "
                f"peak={run['post_forward']['peak_MiB']:.0f}MiB "
                f"ssd_disk={disk_used:.0f}MiB "
                f"write_throughput={pool_stats['write_throughput_MBps']:.0f}MB/s "
                f"argmax={run['forward']['argmax_last']} "
                f"hash={run['forward']['logit_last_hash_fp32']}")
            consecutive_fail = 0
            del out, last_logits
        except torch.cuda.OutOfMemoryError as e:
            run["forward"] = {"ok": False, "error_type": "CUDA_OOM", "error_msg": str(e)[:400]}
            log(f"ctx={ctx} OOM")
            consecutive_fail += 1
        except Exception as e:
            run["forward"] = {
                "ok": False,
                "error_type": type(e).__name__,
                "error_msg": str(e)[:400],
                "traceback": traceback.format_exc()[-1500:],
            }
            log(f"ctx={ctx} ERROR: {type(e).__name__}: {e}")
            consecutive_fail += 1

        # Cleanup before next ctx
        try:
            cache.shutdown()
        except Exception:
            pass
        del input_ids, cache
        if ssd_dir.exists():
            shutil.rmtree(ssd_dir)
        torch.cuda.empty_cache()
        gc.collect()
        runs.append(run)

        if consecutive_fail >= 2:
            log("2 consecutive failures — stopping ctx sweep early")
            break

    meta["runs"] = runs
    out_path.write_text(json.dumps(meta, indent=2))
    log(f"DONE — wrote {out_path}")

if __name__ == "__main__":
    main()
