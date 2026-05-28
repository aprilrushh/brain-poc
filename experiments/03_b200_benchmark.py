"""
Brain Scale Benchmark — GPU/CUDA edition (B200 verified 2026-05-28)
01/02 (맥북 mps/cpu) 와 달리 cuda auto-detect + cuda.synchronize() 정확 타이밍
+ dim=1024 (BGE-M3 실제 차원) + 결과를 NFS filesystem 에 영구 저장.
사용: python3 experiments/03_b200_benchmark.py
"""
import time, json, torch, numpy as np
from pathlib import Path
import faiss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
DIM = 1024   # BGE-M3 실제 임베딩 차원
BETA = 50.0  # 운영값 (의미집중도 0.9809 검증값)
print(f"Device: {DEVICE} | {DEVICE_NAME}")


def synth(n, dim=DIM, seed=42):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    v = torch.randn(n, dim, device=DEVICE, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def bench(n, nq=50):
    keys = synth(n)
    q = synth(nq, seed=999)
    # warmup
    _ = torch.topk(torch.softmax(BETA * (q[:1] @ keys.T), -1), 3); sync()
    # Brain batched (전체 query 한 번에 = O(1) 병렬)
    t0 = time.perf_counter()
    _ = torch.topk(torch.softmax(BETA * (q @ keys.T), -1), 3); sync()
    brain_batched = (time.perf_counter() - t0) / nq * 1000
    # Brain sequential (query 하나씩)
    t0 = time.perf_counter()
    for i in range(nq):
        _ = torch.topk(torch.softmax(BETA * (q[i:i+1] @ keys.T), -1), 3)
    sync()
    brain_seq = (time.perf_counter() - t0) / nq * 1000
    # Faiss (CPU index — 비교 baseline. GPU faiss 는 cu 버전 충돌 회피)
    idx = faiss.IndexFlatIP(DIM)
    idx.add(keys.cpu().numpy().astype(np.float32))
    qn = q.cpu().numpy().astype(np.float32)
    t0 = time.perf_counter()
    for i in range(nq):
        idx.search(qn[i:i+1], 3)
    faiss_ms = (time.perf_counter() - t0) / nq * 1000
    mem_gb = keys.element_size() * keys.nelement() / 1e9
    return dict(n=n, brain_batched_ms=brain_batched, brain_seq_ms=brain_seq,
                faiss_cpu_ms=faiss_ms, mem_gb=mem_gb)


def main():
    sizes = [1000, 10000, 100000, 1000000, 6400000]
    results = []
    for n in sizes:
        print(f"\n--- {n:,} memories ---")
        r = bench(n)
        results.append(r)
        print(f"  Brain batched: {r['brain_batched_ms']:.4f} ms/query")
        print(f"  Brain seq:     {r['brain_seq_ms']:.4f} ms/query")
        print(f"  Faiss (CPU):   {r['faiss_cpu_ms']:.4f} ms/query")
        print(f"  index memory:  {r['mem_gb']:.2f} GB")

    out = {
        "note": "Faiss=CPU index (GPU-Brain vs CPU-Faiss 비교, 외부 claim 부적합). 합성 랜덤 벡터 — 속도 측정용, 정확도/의미집중도와 무관.",
        "device": DEVICE_NAME, "dim": DIM, "beta": BETA, "results": results,
    }
    outdir = Path(__file__).resolve().parent.parent / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "gpu_scale_benchmark.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nSaved: {outpath}")
    print("===== BENCH DONE =====")


if __name__ == "__main__":
    main()
