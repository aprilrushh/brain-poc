#!/usr/bin/env python3
"""phase_d_batch_driver.py — D-7.5 autonomous 6-scenario batch."""
import gc, hashlib, json, shutil, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path.home() / "xhbm" / "scripts"))
from xhbm_tiered_cache import TieredKVCache
from multi_conv_simulator_v2 import ConvSchedule, MultiConvSimulator

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
RESULTS_DIR = Path.home() / "xhbm" / "results"
SHERLOCK_PATH = Path.home() / "xhbm" / "data" / "sherlock" / "sherlock_merged.txt"
SSD_DIR_BASE = Path.home() / "xhbm" / "fio_bench" / "batch_v2"
PROGRESS_PATH = RESULTS_DIR / "batch_PROGRESS.md"

SCENARIOS = [
    ("S1_n1_8x32k_15min_RAM", 8, 32768, 900.0, 64),
    ("S2_n2_8x32k_15min_RAM", 8, 32768, 900.0, 64),
    ("S3_n3_8x32k_15min_RAM", 8, 32768, 900.0, 64),
]


def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def progress_append(line):
    cur = PROGRESS_PATH.read_text() if PROGRESS_PATH.exists() else ""
    PROGRESS_PATH.write_text(cur + line + "\n")


def hash_tensor(t):
    return hashlib.sha256(t.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def gpu_snap():
    torch.cuda.synchronize()
    return {"alloc_MiB": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "peak_MiB": round(torch.cuda.max_memory_allocated() / (1024**2), 1)}


def hard_cleanup(convs):
    for c in convs:
        try:
            c.cache.shutdown()
        except Exception:
            pass
    convs.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def warmup_files(n_threads=8):
    snap = list((Path.home() / ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots").glob("*"))[0]
    files = [f.resolve() for f in sorted(snap.glob("model-*-of-*.safetensors"))]
    def _r(p):
        with open(p, "rb") as fh:
            while fh.read(64 * 1024 * 1024):
                pass
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(_r, files))
    return time.perf_counter() - t0


def run_scenario(name, n_convs, prefill_ctx, sim_dur, decode_n,
                 model, tokenizer, all_tokens, run_id):
    log(f"=== {name}: n_convs={n_convs} ctx={prefill_ctx} sim={sim_dur}s ===")
    progress_append(f"### {name} — STARTED at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    progress_append(f"  config: n={n_convs} ctx={prefill_ctx} sim={sim_dur}s decode={decode_n}")

    result = {"name": name, "n_convs": n_convs, "prefill_ctx": prefill_ctx,
              "sim_duration_sec": sim_dur, "decode_n": decode_n,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "RUNNING"}

    needed = n_convs * (prefill_ctx + 2048)
    if len(all_tokens) < needed:
        result["status"] = "SKIPPED_INSUFFICIENT_TOKENS"
        progress_append(f"  ❌ SKIPPED — need {needed} tokens, have {len(all_tokens)}")
        return result

    ssd_dir = SSD_DIR_BASE / name
    if ssd_dir.exists():
        shutil.rmtree(ssd_dir)
    ssd_dir.mkdir(parents=True, exist_ok=True)

    convs = []
    prefills = []
    try:
        log(f"  prefill {n_convs} convs")
        for i in range(n_convs):
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats()
            start = i * prefill_ctx
            slice_t = all_tokens[start:start + prefill_ctx]
            iids = torch.tensor(slice_t, dtype=torch.long).unsqueeze(0).to("cuda:0")
            cache = TieredKVCache(ssd_dir=ssd_dir, config=model.config,
                                  num_workers=4, conv_id=f"conv_{i}")
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=iids, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            pf_sec = time.perf_counter() - t0
            ll = out.logits[:, -1, :]
            ph = hash_tensor(ll)
            aid = int(ll.argmax(dim=-1)[0].item())
            atxt = tokenizer.decode([aid])
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            tdem = time.perf_counter()
            # D-7.6.3: prefill -> RAM tier (5min idle automatic SSD demote)
            cache.demote_to_ram()
            dem_sec = time.perf_counter() - tdem
            log(f"    conv_{i}: pf={pf_sec:.1f}s peak={peak:.0f}MiB dem={dem_sec:.1f}s argmax={aid}({atxt!r}) hash={ph}")
            prefills.append({"conv_id": f"conv_{i}", "prefill_sec": pf_sec, "peak_MiB": peak,
                            "demote_sec": dem_sec, "argmax_id": aid, "argmax_text": atxt, "logit_hash": ph})
            pool_start = start + prefill_ctx
            pool_end = min(pool_start + 2048, len(all_tokens))
            mpool = all_tokens[pool_start:pool_end]
            is_freq = (i < n_convs - 1) if n_convs > 1 else True
            convs.append(ConvSchedule(conv_id=f"conv_{i}", cache=cache,
                        prefill_token_count=prefill_ctx, is_frequent=is_freq,
                        message_token_pool=mpool))
            del out, ll, iids
            torch.cuda.empty_cache(); gc.collect()

        result["prefill"] = prefills
        result["post_prefill_gpu"] = gpu_snap()

        log(f"  simulation {sim_dur}s")
        torch.cuda.reset_peak_memory_stats()
        sim = MultiConvSimulator(model=model, conversations=convs, duration_sec=sim_dur,
                                 decode_n_tokens=decode_n, sample_interval_sec=1.0, seed=42)
        t_s = time.perf_counter()
        sim.run()
        result["sim_actual_sec"] = time.perf_counter() - t_s
        result["simulation"] = sim.summary()
        result["post_sim_gpu"] = gpu_snap()
        result["status"] = "DONE"
        s = result["simulation"]
        log(f"  done: fwd={s['forward_count']} prom_p50/p95/p99={s['promote_latency_p50_sec']}/{s['promote_latency_p95_sec']}/{s['promote_latency_p99_sec']}s")
        progress_append(f"  ✅ DONE — fwd={s['forward_count']} promote p50/p95/p99={s['promote_latency_p50_sec']}/{s['promote_latency_p95_sec']}/{s['promote_latency_p99_sec']}s GPU peak={result['post_sim_gpu']['peak_MiB']}MiB")

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)[:500]
        result["traceback"] = traceback.format_exc()[-2000:]
        log(f"  FAILED: {type(e).__name__}: {e}")
        progress_append(f"  ❌ FAILED — {type(e).__name__}: {str(e)[:200]}")
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        hard_cleanup(convs)
        if ssd_dir.exists():
            try:
                shutil.rmtree(ssd_dir)
            except Exception:
                pass

    return result


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SSD_DIR_BASE.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        f"# XHBM Batch Test Progress\n\nStarted: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\nRun ID: {run_id}\n\n## Scenarios planned\n\n"
    )
    for s in SCENARIOS:
        progress_append(f"- {s[0]}: n={s[1]} ctx={s[2]} sim={s[3]}s decode_n={s[4]}")
    progress_append("")
    progress_append("## Cold-start phase")

    log("drop_caches ...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    time.sleep(2)

    log("multi-thread warmup (8 threads) ...")
    t_w = warmup_files(8)
    log(f"warmup: {t_w:.1f}s")
    progress_append(f"- warmup (8-thread): {t_w:.1f}s")

    log("loading tokenizer + sherlock ...")
    tok = AutoTokenizer.from_pretrained(REPO)
    with open(SHERLOCK_PATH, encoding="utf-8") as f:
        all_tokens = tok.encode(f.read(), add_special_tokens=False)
    log(f"sherlock tokens: {len(all_tokens):,}")

    log("loading model ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(REPO, quantization_config=bnb,
                                                 device_map={"": "cuda:0"},
                                                 attn_implementation="sdpa", dtype=torch.bfloat16)
    model.eval()
    load_s = time.perf_counter() - t0
    log(f"model load: {load_s:.1f}s")
    progress_append(f"- model load: {load_s:.1f}s")
    progress_append(f"- cold-start total: {(t_w + load_s):.1f}s")
    progress_append("\n## Scenarios\n")

    overall = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
               "warmup_sec": t_w, "model_load_sec": load_s,
               "sherlock_token_count": len(all_tokens), "scenarios": []}

    for s in SCENARIOS:
        sr = run_scenario(*s, model=model, tokenizer=tok, all_tokens=all_tokens, run_id=run_id)
        overall["scenarios"].append(sr)
        per_path = RESULTS_DIR / f"batch_{run_id}_{s[0]}.json"
        per_path.write_text(json.dumps(sr, indent=2, default=str))
        agg_path = RESULTS_DIR / f"batch_{run_id}_OVERALL.json"
        agg_path.write_text(json.dumps(overall, indent=2, default=str))

    overall["finished_at"] = datetime.now(timezone.utc).isoformat()
    progress_append(f"\n## ALL DONE at {overall['finished_at']}\n")
    progress_append("## Summary\n")
    progress_append("| Scenario | Status | fwd | promote_p50 | p95 | p99 | GPU peak |")
    progress_append("|---|---|---|---|---|---|---|")
    for s in overall["scenarios"]:
        sim = s.get("simulation", {})
        gpu = s.get("post_sim_gpu", {}).get("peak_MiB", "-")
        progress_append(f"| {s['name']} | {s['status']} | {sim.get('forward_count', '-')} | {sim.get('promote_latency_p50_sec', '-')} | {sim.get('promote_latency_p95_sec', '-')} | {sim.get('promote_latency_p99_sec', '-')} | {gpu} |")

    final = RESULTS_DIR / f"batch_{run_id}_OVERALL.json"
    final.write_text(json.dumps(overall, indent=2, default=str))
    log(f"DONE — {final}")


if __name__ == "__main__":
    main()
