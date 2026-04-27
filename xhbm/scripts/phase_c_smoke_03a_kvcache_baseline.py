#!/usr/bin/env python3
"""
phase_c_smoke_03a_kvcache_baseline.py — Phase C Smoke #3a

Purpose: Establish OOM boundary with use_cache=True (KV cache active) for
         70B NF4 baseline on H100 PCIe 80GB. This defines the ctx region
         where V offload becomes meaningful (KV cache must exist for V
         offload to have anything to offload).

Compares against smoke_02_bnb (use_cache=False) to isolate KV cache
HBM cost: peak_03a - peak_02 = KV cache size for that ctx.

North Star: honest measurement (OOM = OOM), correctness hash for bit-exact.
"""
import gc, hashlib, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# smoke_02_bnb peak @ 128K = 71,891 MiB. KV cache @ 128K (80 layers × 2 ×
# 8 KV heads × 128 dim × 128K × 2 bytes bf16) ≈ 4 GB. expected OOM around
# 96K-128K with KV active. start sweep narrower than smoke_02.
CTX_LIST = [4096, 8192, 16384, 32768, 65536, 98304, 131072]
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
    out_path = RESULTS_DIR / f"smoke_03a_{run_id}.json"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    meta = {
        "script": "phase_c_smoke_03a_kvcache_baseline.py",
        "repo": REPO,
        "use_cache": True,
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
    log(f"meta: torch={torch.__version__} device={meta['cuda_device']} batch={BATCH} use_cache=True")
    log(f"ctx_list: {CTX_LIST}")

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
        "num_kv_heads": getattr(model.config, "num_key_value_heads", None),
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
    }
    log(f"load OK: {load_sec:.1f}s peak={snap['peak_MiB']:.0f}MiB "
        f"params={meta['load']['num_params']/1e9:.1f}B "
        f"kv_heads={meta['load']['num_kv_heads']} head_dim={meta['load']['head_dim']}")

    vocab_size = model.config.vocab_size
    runs = []
    consecutive_fail = 0

    for ctx in CTX_LIST:
        log(f"=== ctx={ctx} batch={BATCH} (use_cache=True) ===")
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
                out = model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            fwd_sec = time.perf_counter() - t0
            last_logits = out.logits[:, -1, :]

            # KV cache size verification
            kv_cache = out.past_key_values
            kv_total_bytes = 0
            kv_layer_count = 0
            try:
                # transformers 5.x DynamicCache structure
                if hasattr(kv_cache, "layers"):
                    for layer in kv_cache.layers:
                        if hasattr(layer, "keys") and layer.keys is not None:
                            kv_total_bytes += layer.keys.element_size() * layer.keys.numel()
                        if hasattr(layer, "values") and layer.values is not None:
                            kv_total_bytes += layer.values.element_size() * layer.values.numel()
                        kv_layer_count += 1
            except Exception as e:
                log(f"  kv inspect skipped: {e}")

            run["forward"] = {
                "ok": True,
                "seconds": fwd_sec,
                "logit_last_hash_fp32": hash_tensor(last_logits),
                "argmax_last": int(last_logits.argmax(dim=-1)[0].item()),
                "kv_cache_total_MiB": kv_total_bytes / (1024**2),
                "kv_cache_layers_observed": kv_layer_count,
            }
            run["post_forward"] = gpu_snapshot(f"ctx{ctx}_post")
            log(f"ctx={ctx} OK: fwd={fwd_sec:.2f}s "
                f"peak={run['post_forward']['peak_MiB']:.0f}MiB "
                f"kv_cache={run['forward']['kv_cache_total_MiB']:.0f}MiB "
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
