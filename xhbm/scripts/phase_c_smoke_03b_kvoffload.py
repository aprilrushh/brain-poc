#!/usr/bin/env python3
"""
phase_c_smoke_03b_kvoffload.py — Phase C Smoke #3b

Purpose: Measure HBM savings + fwd time cost of K+V cache CPU RAM offload
         using transformers official DynamicCache(offloading=True).
         Compare against smoke_03a baseline at same ctx values.
         Critical test: ctx >= 64K where smoke_03a OOM'd.

North Star §3 metrics:
  - HBM savings ≥ 30% target
  - throughput degradation ≤ 20% target
  - bit-exact correctness vs smoke_03a (offload is mathematically identical)
"""
import gc, hashlib, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# Targeted ctx list:
#   4K, 16K, 32K  : direct comparison vs smoke_03a (must be bit-exact)
#   64K, 96K, 128K: smoke_03a OOM region — does offload save the day?
CTX_LIST = [4096, 16384, 32768, 65536, 98304, 131072]
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

def cache_size_breakdown(cache):
    """Return (gpu_MiB, cpu_MiB, layer_count) of K+V across all layers."""
    gpu_bytes = 0
    cpu_bytes = 0
    n = 0
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is None:
                    continue
                size = t.element_size() * t.numel()
                if t.device.type == "cuda":
                    gpu_bytes += size
                else:
                    cpu_bytes += size
            n += 1
    return gpu_bytes / (1024**2), cpu_bytes / (1024**2), n

def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"smoke_03b_{run_id}.json"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    meta = {
        "script": "phase_c_smoke_03b_kvoffload.py",
        "repo": REPO,
        "use_cache": True,
        "cache_offloading": True,
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
    log(f"meta: torch={torch.__version__} device={meta['cuda_device']} batch={BATCH}")
    log(f"ctx_list: {CTX_LIST}  cache_offloading=True")

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
    log(f"load OK: {load_sec:.1f}s peak={snap['peak_MiB']:.0f}MiB "
        f"params={meta['load']['num_params']/1e9:.1f}B")

    vocab_size = model.config.vocab_size
    runs = []
    consecutive_fail = 0

    for ctx in CTX_LIST:
        log(f"=== ctx={ctx} batch={BATCH} (DynamicCache offloading=True) ===")
        run = {"ctx": ctx, "batch": BATCH}
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        run["pre_forward"] = gpu_snapshot(f"ctx{ctx}_pre")

        g = torch.Generator(device="cpu").manual_seed(42 + ctx)
        input_ids = torch.randint(0, vocab_size, (BATCH, ctx), generator=g).to("cuda:0")

        # Fresh offloading cache per ctx
        cache = DynamicCache(offloading=True)

        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
            torch.cuda.synchronize()
            fwd_sec = time.perf_counter() - t0
            last_logits = out.logits[:, -1, :]

            kv_cache = out.past_key_values
            kv_gpu, kv_cpu, kv_layers = cache_size_breakdown(kv_cache)

            run["forward"] = {
                "ok": True,
                "seconds": fwd_sec,
                "logit_last_hash_fp32": hash_tensor(last_logits),
                "argmax_last": int(last_logits.argmax(dim=-1)[0].item()),
                "kv_cache_gpu_MiB": kv_gpu,
                "kv_cache_cpu_MiB": kv_cpu,
                "kv_cache_layers_observed": kv_layers,
            }
            run["post_forward"] = gpu_snapshot(f"ctx{ctx}_post")
            log(f"ctx={ctx} OK: fwd={fwd_sec:.2f}s "
                f"peak={run['post_forward']['peak_MiB']:.0f}MiB "
                f"kv_gpu={kv_gpu:.0f} kv_cpu={kv_cpu:.0f}MiB "
                f"argmax={run['forward']['argmax_last']} "
                f"hash={run['forward']['logit_last_hash_fp32']}")
            consecutive_fail = 0
            del out, last_logits, kv_cache
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

        del input_ids, cache
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
