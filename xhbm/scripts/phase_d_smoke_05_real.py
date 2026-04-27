#!/usr/bin/env python3
"""
phase_d_smoke_05_real.py — Phase D Smoke #5 (real prompts variant)

Forked from phase_d_smoke_05_ssd_baseline.py (2026-04-25 09:27 version).
Sole change: random tokens → first-N-tokens of Sherlock Holmes (4 books, PG).

Validates that smoke_05 random-token results generalize to real text input.
SSD substrate (xhbm_ssd_cache.py) UNCHANGED for direct comparison.
"""
import gc, hashlib, json, shutil, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path.home() / "xhbm" / "scripts"))
from xhbm_ssd_cache import SSDOffloadCache

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
CTX_LIST = [4096, 16384, 32768, 65536, 98304, 131072]
BATCH = 1
SSD_DIR_BASE = Path.home() / "xhbm" / "fio_bench" / "smoke_05_real"
RESULTS_DIR = Path.home() / "xhbm" / "results"
SHERLOCK_PATH = Path.home() / "xhbm" / "data" / "sherlock" / "sherlock_merged.txt"
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
    out_path = RESULTS_DIR / f"smoke_05_real_{run_id}.json"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- D-5.0 NEW: tokenize Sherlock Holmes once ---
    log(f"loading tokenizer + sherlock prompt corpus ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    if not SHERLOCK_PATH.exists():
        log(f"FATAL: {SHERLOCK_PATH} not found. Run data prep step first.")
        sys.exit(1)
    with open(SHERLOCK_PATH, encoding="utf-8") as f:
        sherlock_text = f.read()
    all_tokens = tokenizer.encode(sherlock_text, add_special_tokens=False)
    max_ctx_needed = max(CTX_LIST)
    if len(all_tokens) < max_ctx_needed:
        log(f"FATAL: sherlock has {len(all_tokens)} tokens, need ≥{max_ctx_needed}")
        sys.exit(1)
    log(f"sherlock tokens: {len(all_tokens):,} (need ≥{max_ctx_needed:,})  ✓")

    meta = {
        "script": "phase_d_smoke_05_real.py",
        "repo": REPO,
        "use_cache": True,
        "cache_backend": "SSDOffloadCache",
        "num_workers": NUM_WORKERS,
        "ssd_dir_base": str(SSD_DIR_BASE),
        "prompt_source": {
            "type": "real_text",
            "name": "Sherlock Holmes (4 books, Project Gutenberg)",
            "books": [
                "A Study in Scarlet (#244)",
                "The Sign of the Four (#2097)",
                "The Adventures of Sherlock Holmes (#1661)",
                "The Memoirs of Sherlock Holmes (#834)",
            ],
            "total_tokens_available": len(all_tokens),
            "slice_method": "first-N tokens, deterministic",
        },
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
    log(f"meta: torch={torch.__version__} device={meta['cuda_device']} batch={BATCH} backend=SSD prompts=REAL")
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

    runs = []
    consecutive_fail = 0

    for ctx in CTX_LIST:
        log(f"=== ctx={ctx} batch={BATCH} (SSDOffloadCache, real prompts) ===")
        run = {"ctx": ctx, "batch": BATCH}
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        run["pre_forward"] = gpu_snapshot(f"ctx{ctx}_pre")

        ssd_dir = SSD_DIR_BASE / f"ctx_{ctx}"
        if ssd_dir.exists():
            shutil.rmtree(ssd_dir)
        ssd_dir.mkdir(parents=True, exist_ok=True)

        # --- D-5.0 NEW: real text slice instead of randint ---
        slice_tokens = all_tokens[:ctx]
        input_ids = torch.tensor(slice_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
        run["input_first_token_id"] = int(slice_tokens[0])
        run["input_last_token_id"] = int(slice_tokens[-1])

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

            argmax_id = int(last_logits.argmax(dim=-1)[0].item())
            argmax_text = tokenizer.decode([argmax_id])

            run["forward"] = {
                "ok": True,
                "seconds": fwd_sec,
                "logit_last_hash_fp32": hash_tensor(last_logits),
                "argmax_last": argmax_id,
                "argmax_text": argmax_text,
                "ssd_disk_used_MiB": disk_used,
                "pool_stats": pool_stats,
            }
            run["post_forward"] = gpu_snapshot(f"ctx{ctx}_post")
            log(f"ctx={ctx} OK: fwd={fwd_sec:.2f}s "
                f"peak={run['post_forward']['peak_MiB']:.0f}MiB "
                f"ssd_disk={disk_used:.0f}MiB "
                f"write={pool_stats['write_throughput_MBps']:.0f}MB/s "
                f"argmax={argmax_id}({argmax_text!r}) "
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
