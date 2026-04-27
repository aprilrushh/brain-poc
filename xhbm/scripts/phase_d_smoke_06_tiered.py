#!/usr/bin/env python3
"""
phase_d_smoke_06_tiered.py — Phase D D-5.2.3 (sanity scale).

Multi-conv tiered KV cache: 2-conv × 1-min sanity smoke.
After pass, scale to 8-conv × 5-min in D-5.2.4.
"""
import gc, hashlib, json, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path.home() / "xhbm" / "scripts"))
from xhbm_tiered_cache import TieredKVCache
from multi_conv_simulator_v2 import ConvSchedule, MultiConvSimulator

REPO = "meta-llama/Meta-Llama-3.1-70B-Instruct"
N_CONVS = 2
PREFILL_CTX = 32768
SIM_DURATION_SEC = 90.0
SSD_DIR_BASE = Path.home() / "xhbm" / "fio_bench" / "smoke_06_tiered"
RESULTS_DIR = Path.home() / "xhbm" / "results"
SHERLOCK_PATH = Path.home() / "xhbm" / "data" / "sherlock" / "sherlock_merged.txt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def gpu_snapshot(tag):
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "alloc_MiB": torch.cuda.memory_allocated() / (1024**2),
        "peak_MiB": torch.cuda.max_memory_allocated() / (1024**2),
    }


def hash_tensor(t):
    arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"smoke_06_tiered_{run_id}.json"

    if SSD_DIR_BASE.exists():
        shutil.rmtree(SSD_DIR_BASE)
    SSD_DIR_BASE.mkdir(parents=True, exist_ok=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    log("loading tokenizer + sherlock corpus ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    with open(SHERLOCK_PATH, encoding="utf-8") as f:
        all_tokens = tokenizer.encode(f.read(), add_special_tokens=False)
    log(f"sherlock tokens: {len(all_tokens):,}")
    needed = N_CONVS * (PREFILL_CTX + 2048)
    if len(all_tokens) < needed:
        log(f"FATAL: sherlock has {len(all_tokens)}, need {needed} for {N_CONVS} convs")
        sys.exit(1)

    meta = {
        "script": "phase_d_smoke_06_tiered.py",
        "stage": "D-7.6.2 RAM tier sanity",
        "n_convs": N_CONVS,
        "prefill_ctx": PREFILL_CTX,
        "sim_duration_sec": SIM_DURATION_SEC,
        "run_id": run_id,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }

    torch.cuda.reset_peak_memory_stats()
    meta["pre_load"] = gpu_snapshot("pre_load")

    log(f"loading {REPO} with bnb NF4 ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        REPO, quantization_config=bnb_cfg,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.eval()
    load_sec = time.perf_counter() - t0
    snap = gpu_snapshot("post_load")
    meta["load"] = {"ok": True, "seconds": load_sec, **snap}
    log(f"load OK: {load_sec:.1f}s peak={snap['peak_MiB']:.0f}MiB")

    log(f"=== Prefill phase: {N_CONVS} convs ===")
    prefill_results = []
    convs = []
    for i in range(N_CONVS):
        log(f"  prefilling conv_{i} (start_offset={i * PREFILL_CTX}) ...")
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats()

        start = i * PREFILL_CTX
        slice_tokens = all_tokens[start:start + PREFILL_CTX]
        input_ids = torch.tensor(slice_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")

        cache = TieredKVCache(
            ssd_dir=SSD_DIR_BASE, config=model.config,
            num_workers=4, conv_id=f"conv_{i}",
        )

        t_pre = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        prefill_sec = time.perf_counter() - t_pre
        last_logits = out.logits[:, -1, :]
        prefill_hash = hash_tensor(last_logits)
        argmax_id = int(last_logits.argmax(dim=-1)[0].item())
        argmax_text = tokenizer.decode([argmax_id])
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

        # D-7.6: prefill 직후 RAM tier 로 (5min idle 시 자동 SSD 강등)
        t_dem = time.perf_counter()
        cache.demote_to_ram()
        demote_sec = time.perf_counter() - t_dem

        log(f"  conv_{i}: prefill={prefill_sec:.1f}s peak={peak_mib:.0f}MiB "
            f"demote={demote_sec:.2f}s argmax={argmax_id}({argmax_text!r}) hash={prefill_hash}")

        prefill_results.append({
            "conv_id": f"conv_{i}",
            "prefill_seconds": prefill_sec,
            "peak_MiB": peak_mib,
            "demote_seconds": demote_sec,
            "argmax_last": argmax_id,
            "argmax_text": argmax_text,
            "logit_hash": prefill_hash,
            "tier_state_after_demote": cache.get_tier_state(),
            "tier_stats_after_demote": cache.get_tier_stats(),
        })

        is_frequent = (i < N_CONVS - 1) if N_CONVS > 1 else True
        # D-7.4: reserve real-text token pool for new messages of this conv.
        # Located after this conv's prefill chunk, takes 2K tokens (enough for
        # ~15-30 messages of 64-128 tokens each).
        pool_start = start + PREFILL_CTX
        pool_end = min(pool_start + 2048, len(all_tokens))
        message_pool = all_tokens[pool_start:pool_end]

        convs.append(ConvSchedule(
            conv_id=f"conv_{i}", cache=cache,
            prefill_token_count=PREFILL_CTX, is_frequent=is_frequent,
            message_token_pool=message_pool,
        ))

        del out, last_logits, input_ids
        torch.cuda.empty_cache(); gc.collect()

    meta["prefill"] = prefill_results
    snap = gpu_snapshot("post_prefill_all_in_ssd")
    log(f"post-prefill (all SSD): GPU alloc={snap['alloc_MiB']:.0f}MiB")
    meta["post_prefill"] = snap

    log(f"=== Simulation phase: {SIM_DURATION_SEC}s ===")
    torch.cuda.reset_peak_memory_stats()
    sim = MultiConvSimulator(model=model, conversations=convs,
                             duration_sec=SIM_DURATION_SEC, seed=42)
    t_sim = time.perf_counter()
    events = sim.run()
    sim_sec = time.perf_counter() - t_sim

    summary = sim.summary()
    snap = gpu_snapshot("post_sim")
    log(f"sim done: {sim_sec:.1f}s events={summary['total_events']} "
        f"fwd={summary['forward_count']} promote={summary['promote_count']} "
        f"demote={summary['demote_count']}")
    log(f"  promote latency P50={summary['promote_latency_p50_sec']}s "
        f"P95={summary['promote_latency_p95_sec']}s")
    log(f"  GPU peak during sim: {snap['peak_MiB']:.0f}MiB")

    meta["simulation"] = summary
    meta["post_sim"] = snap
    meta["events"] = [
        {"t": round(e.timestamp, 3), "type": e.event_type, "conv": e.conv_id,
         "dur": round(e.duration_sec, 3), **e.extra}
        for e in events
    ]

    for c in convs:
        try:
            c.cache.shutdown()
        except Exception:
            pass

    out_path.write_text(json.dumps(meta, indent=2, default=str))
    log(f"DONE — wrote {out_path}")


if __name__ == "__main__":
    main()
