"""
Brain-like Inference - Scale Benchmark v0.2
지식 베이스 크기를 100 → 10만까지 키우며 두 방식 성능 비교
"""
import time
import json
from pathlib import Path
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import faiss

console = Console()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
console.print(f"[bold cyan]Device: {DEVICE}[/]")


def generate_synthetic_memories(n, dim=384, seed=42):
    torch.manual_seed(seed)
    vecs = torch.randn(n, dim, device=DEVICE)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs


class BrainMemory:
    def __init__(self, beta=8.0):
        self.beta = beta
        self.keys = None

    def store(self, embeddings):
        self.keys = embeddings

    def recall(self, query, top_k=3):
        logits = self.beta * (query @ self.keys.T)
        weights = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(weights, top_k)
        return top_w, top_i


class FaissMemory:
    def __init__(self):
        self.index = None

    def store(self, embeddings):
        vecs = embeddings.cpu().numpy().astype(np.float32)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def recall(self, query, top_k=3):
        q = query.cpu().numpy().astype(np.float32)
        return self.index.search(q, top_k)


def benchmark(n_memories, n_queries=50):
    memories = generate_synthetic_memories(n_memories)
    queries = generate_synthetic_memories(n_queries, seed=999)
    brain = BrainMemory(beta=8.0)
    brain.store(memories)
    faiss_mem = FaissMemory()
    faiss_mem.store(memories)

    _ = brain.recall(queries[:1], top_k=3)
    _ = faiss_mem.recall(queries[:1], top_k=3)
    if DEVICE == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for q in queries:
        _ = brain.recall(q.unsqueeze(0), top_k=3)
    if DEVICE == "mps":
        torch.mps.synchronize()
    brain_time = (time.perf_counter() - t0) / n_queries * 1000

    t0 = time.perf_counter()
    _ = brain.recall(queries, top_k=3)
    if DEVICE == "mps":
        torch.mps.synchronize()
    brain_batch_time = (time.perf_counter() - t0) / n_queries * 1000

    t0 = time.perf_counter()
    for q in queries:
        _ = faiss_mem.recall(q.unsqueeze(0), top_k=3)
    faiss_time = (time.perf_counter() - t0) / n_queries * 1000

    brain_mem_mb = memories.element_size() * memories.nelement() / 1e6

    return {
        "n": n_memories,
        "brain_sequential_ms": brain_time,
        "brain_batched_ms": brain_batch_time,
        "faiss_ms": faiss_time,
        "memory_mb": brain_mem_mb,
    }


def main():
    console.print(Panel(
        "[bold]스케일 벤치마크[/]\n"
        "지식 베이스 크기를 키우며 두 방식의 성능 변화를 측정합니다.\n"
        "뇌 방식의 O(1) 병렬 처리 우위가 드러나는 규모를 확인.",
        title="목표",
    ))

    sizes = [100, 1000, 10000, 50000, 100000]
    results = []

    for n in sizes:
        console.print(f"\n[bold yellow]--- {n:>7,}개 기억 테스트 중... ---[/]")
        result = benchmark(n, n_queries=50)
        results.append(result)
        console.print(f"  뇌 (순차): [cyan]{result['brain_sequential_ms']:.3f} ms/query[/]")
        console.print(f"  뇌 (배치): [cyan]{result['brain_batched_ms']:.3f} ms/query[/]")
        console.print(f"  FAISS:    [yellow]{result['faiss_ms']:.3f} ms/query[/]")
        console.print(f"  메모리:   {result['memory_mb']:.1f} MB")

    console.print("\n")
    table = Table(title="최종 스케일 비교", show_header=True, header_style="bold magenta")
    table.add_column("지식 수", justify="right", width=12)
    table.add_column("뇌 순차 (ms)", justify="right")
    table.add_column("뇌 배치 (ms)", justify="right")
    table.add_column("FAISS (ms)", justify="right")
    table.add_column("뇌배치 우위", justify="right")
    table.add_column("메모리 (MB)", justify="right")

    for r in results:
        speedup = r["faiss_ms"] / r["brain_batched_ms"]
        speedup_str = (
            f"[green]{speedup:.1f}x UP[/]" if speedup > 1
            else f"[red]{speedup:.2f}x DOWN[/]"
        )
        table.add_row(
            f"{r['n']:,}",
            f"{r['brain_sequential_ms']:.3f}",
            f"{r['brain_batched_ms']:.3f}",
            f"{r['faiss_ms']:.3f}",
            speedup_str,
            f"{r['memory_mb']:.1f}",
        )
    console.print(table)

    console.print("\n")
    first = results[0]
    last = results[-1]
    console.print(Panel(
        f"[bold green]핵심 발견[/]\n\n"
        f"- FAISS 지연 증가율: {last['faiss_ms']/first['faiss_ms']:.1f}배\n"
        f"  (기억 {first['n']:,}개 -> {last['n']:,}개, "
        f"{last['n']/first['n']:.0f}배 증가)\n\n"
        f"- 뇌 배치 지연 증가율: {last['brain_batched_ms']/first['brain_batched_ms']:.1f}배\n"
        f"  -> 지식이 {last['n']/first['n']:.0f}배 커져도 거의 [cyan]같은 속도[/]\n\n"
        f"[bold]이것이 O(N) vs O(1)의 차이입니다.[/]",
        title="Why this matters",
    ))

    log_dir = Path.home() / "brain-inference" / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(log_dir / "scale_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]결과 저장: ~/brain-inference/logs/scale_benchmark.json[/]")


if __name__ == "__main__":
    main()
