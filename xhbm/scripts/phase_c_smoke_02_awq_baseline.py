#!/usr/bin/env python3
"""
phase_c_smoke_02_awq_baseline.py — Phase C Smoke #2

Purpose: OOM boundary for Llama 3.1 70B AWQ INT4 baseline prefill,
         H100 PCIe 80GB, batch=1, use_cache=False.
         Defines the activation-limited ceiling that XHBM V-offload
         cannot exceed. smoke_03 (KV cache + V offload) compares to this.

North Star: honest measurement (OOM = OOM), correctness hash for later bit-exact.
"""
import gc, hashlib, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

REPO = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
CTX_LIST = [4096, 8192, 16384, 32768, 65536, 131072]
BATCH = 1
RESULTS_DIR = Path.home() / "xhbm" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"smoke_02_{run_id}.json"

    meta = {
        "script": "phase_c_smoke_02_awq_baseline.py",
        "repo": REPO,
        "run_id": run_id,
        "ctx_list": CTX_LIST,
        "batch": BATCH,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    log(f"meta: torch={torch.__version__} device={meta['cuda_device']} batch={BATCH}")
    log(f"ctx_list: {CTX_LIST}")

    torch.cuda.reset_peak_memory_stats()
    meta["pre_load"] = gpu_snapshot("pre_load")

    # ---- LOAD ----
    log(f"loading {REPO} ...")
    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            REPO,
            torch_dtype="auto",
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
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
        log(f"wrote {out_path}")
        sys.exit(1)

    load_sec = time.perf_counter() - t0
    snap = gpu_snapshot("post_load")
    meta["load"] = {
        "ok": True,
        "seconds": load_sec,
        **snap,
        "param_dtype": str(next(model.parameters()).dtype),
        "num_params": sum(p.numel() for p in model.parameters()),
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
    }
    log(f"load OK: {load_sec:.1f}s peak={snap['peak_MiB']:.0f}MiB "
        f"alloc={snap['alloc_MiB']:.0f}MiB "
        f"params={meta['load']['num_params']/1e9:.1f}B "
        f"dtype={meta['load']['param_dtype']}")

    # ---- CTX SWEEP ----
    vocab_size = model.config.vocab_size
    runs = []
    consecutive_fail = 0

    for ctx in CTX_LIST:
        log(f"=== ctx={ctx} batch={BATCH} (use_cache=False) ===")
        run = {"ctx": ctx, "batch": BATCH}
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        run["pre_forward"] = gpu_snapshot(f"ctx{ctx}_pre")

        g = torch.Generator(device="cpu").manual_seed(42 + ctx)
        input_ids = torch.randint(0, vocab_size, (BATCH, ctx), generator=g).to("cuda:0")

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=input_ids, use_cache=False)
            torch.cuda.synchronize()
            fwd_sec = time.perf_counter() - t0
            last_logits = out.logits[:, -1, :]
            run["forward"] = {
                "ok": True,
                "seconds": fwd_sec,
                "logit_last_hash_fp32": hash_tensor(last_logits),
                "argmax_last": int(last_logits.argmax(dim=-1)[0].item()),
            }
            run["post_forward"] = gpu_snapshot(f"ctx{ctx}_post")
            log(f"ctx={ctx} OK: fwd={fwd_sec:.2f}s "
                f"peak={run['post_forward']['peak_MiB']:.0f}MiB "
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

        del input_ids
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
