#!/usr/bin/env python3
"""
Phase C smoke 01: Llama 3.1 70B fp8 baseline — prefill-only forward sweep (n=1).

목적:
  - 70B fp8 단일 H100 로드 자체 검증
  - weight가 HBM 어디에 얼마나 있는지 측정
  - ctx 계단식 증가로 OOM 경계 탐색 (use_cache=False, batch=1)
  - 결과 JSON을 ~/xhbm/results/로 저장

운영 원칙 준수:
  - 정직한 측정: OOM이면 OOM으로 기록. baseline batch 내리지 않음.
  - n=1 smoke (sweep 아님, sweep은 통과 후 별도)
  - last-token logit hash + argmax 기록 (후속 XHBM 실험 correctness 기준선)
"""
import argparse, gc, hashlib, json, os, sys, time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"


def gb(x):
    return x / (1024 ** 3)


def hbm_snapshot(tag=""):
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "alloc_GB": round(gb(torch.cuda.memory_allocated()), 3),
        "reserv_GB": round(gb(torch.cuda.memory_reserved()), 3),
        "peak_alloc_GB": round(gb(torch.cuda.max_memory_allocated()), 3),
        "peak_reserv_GB": round(gb(torch.cuda.max_memory_reserved()), 3),
    }


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-list", default="512,2048,4096,8192,16384,32768",
                    help="comma separated ctx lengths")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctx_list = [int(x) for x in args.ctx_list.split(",")]
    out_path = args.out or os.path.expanduser(
        f"~/xhbm/results/smoke_01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    result = {
        "script": "phase_c_smoke_01_baseline",
        "repo": REPO,
        "start_iso": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_GB": round(gb(torch.cuda.get_device_properties(0).total_memory), 2),
        "ctx_list": ctx_list,
        "batch": args.batch,
        "use_cache": False,
        "snapshots": [],
        "forwards": [],
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result["snapshots"].append(hbm_snapshot("before_load"))

    log(f"Loading tokenizer: {REPO}")
    tok = AutoTokenizer.from_pretrained(REPO)
    result["vocab_size"] = tok.vocab_size

    log("Loading model (fp8, device_map=cuda:0, attn_impl=sdpa) — 예상 1~2분")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.eval()
    load_sec = time.time() - t0
    result["load_sec"] = round(load_sec, 2)
    result["snapshots"].append(hbm_snapshot("after_load"))
    result["sdpa_flash"] = torch.backends.cuda.flash_sdp_enabled()
    result["sdpa_mem"] = torch.backends.cuda.mem_efficient_sdp_enabled()

    log(f"Loaded in {load_sec:.1f}s, weight HBM={result['snapshots'][-1]['alloc_GB']} GB, "
        f"headroom={result['gpu_total_GB'] - result['snapshots'][-1]['alloc_GB']:.2f} GB")

    # forward sweep — prefill only, use_cache=False
    for ctx in ctx_list:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pre = hbm_snapshot(f"pre_fwd_ctx{ctx}")
        log(f"forward: ctx={ctx} batch={args.batch}")
        fwd = {"ctx": ctx, "batch": args.batch, "pre": pre}
        try:
            input_ids = torch.randint(0, min(tok.vocab_size, 128000),
                                      (args.batch, ctx), device="cuda:0")
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out = model(input_ids, use_cache=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
            post = hbm_snapshot(f"post_fwd_ctx{ctx}")

            last_logits = out.logits[:, -1, :].detach().to("cpu", torch.float32)
            lh = hashlib.sha256(last_logits.numpy().tobytes()).hexdigest()[:16]
            amax = last_logits.argmax(dim=-1).tolist()

            fwd.update({
                "ok": True,
                "forward_sec": round(dt, 3),
                "post": post,
                "logits_shape": list(out.logits.shape),
                "last_logits_hash": lh,
                "argmax_last": amax,
            })
            log(f"  OK {dt:.2f}s, peak={post['peak_alloc_GB']} GB, argmax_last={amax}")
            del out, input_ids, last_logits
        except torch.cuda.OutOfMemoryError as e:
            fwd.update({"ok": False, "OOM": True, "error": str(e)[:400]})
            log(f"  OOM: {str(e)[:200]}")
            torch.cuda.empty_cache()
        except Exception as e:
            fwd.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
            log(f"  ERR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

        result["forwards"].append(fwd)

    result["end_iso"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved: {out_path}")

    # Summary to stdout
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    w_gb = result["snapshots"][-1]["alloc_GB"]
    print(f"  load               = {result['load_sec']}s")
    print(f"  weight HBM         = {w_gb} GB")
    print(f"  GPU total          = {result['gpu_total_GB']} GB")
    print(f"  headroom post-load = {result['gpu_total_GB'] - w_gb:.2f} GB")
    print(f"  sdpa_flash         = {result['sdpa_flash']}")
    print()
    print(f"  {'ctx':>8} {'batch':>5} {'fwd_sec':>8} {'peak_GB':>8}  {'status':<8} {'argmax'}")
    for f in result["forwards"]:
        status = "OOM" if f.get("OOM") else ("OK" if f.get("ok") else "ERR")
        fs = f.get("forward_sec", "-")
        pk = (f.get("post") or {}).get("peak_alloc_GB", "-")
        am = f.get("argmax_last", "-")
        print(f"  {f['ctx']:>8} {f['batch']:>5} {str(fs):>8} {str(pk):>8}  {status:<8} {am}")
    print(f"\n  JSON: {out_path}")


if __name__ == "__main__":
    main()
