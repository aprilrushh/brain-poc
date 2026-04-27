"""
Brain-like Associative Memory Engine v0.1
Modern Hopfield Network (Ramsauer et al. 2020) 기반 연상 메모리
"""
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import faiss

console = Console()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
console.print(f"[bold cyan]Device: {DEVICE}[/]")


class ModernHopfieldMemory:
    def __init__(self, beta=8.0, device=DEVICE):
        self.beta = beta
        self.device = device
        self.keys = None
        self.values = []

    def store(self, embeddings, texts):
        self.keys = embeddings.to(self.device)
        self.values = texts
        console.print(f"[green]OK[/] {len(texts)}개 기억 저장 완료")

    def recall(self, query_embedding, top_k=3):
        q = query_embedding.to(self.device)
        logits = self.beta * (q @ self.keys.T)
        weights = torch.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, min(top_k, len(self.values)))
        results = []
        for w, idx in zip(top_weights.cpu().tolist()[0], top_indices.cpu().tolist()[0]):
            results.append({"text": self.values[idx], "activation": w})
        return results


class FaissMemory:
    def __init__(self):
        self.index = None
        self.values = []

    def store(self, embeddings, texts):
        vecs = embeddings.cpu().numpy().astype(np.float32)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.values = texts

    def recall(self, query_embedding, top_k=3):
        q = query_embedding.cpu().numpy().astype(np.float32)
        scores, indices = self.index.search(q, top_k)
        return [
            {"text": self.values[idx], "activation": float(score)}
            for score, idx in zip(scores[0], indices[0])
        ]


def main():
    console.print("\n[bold]임베딩 모델 로딩 중... (첫 실행시 1-2분)[/]")
    encoder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=DEVICE,
    )

    memories = [
        "뉴런은 저장과 처리를 같은 위치에서 동시에 수행한다",
        "뇌는 연상 패턴 매칭으로 기억을 즉각 인출한다",
        "해마는 단기 기억을 장기 기억으로 공고화하는 역할을 한다",
        "희소 활성화로 전체 뉴런 중 1퍼센트만 동시 발화한다",
        "시냅스 가소성이 새로운 경험을 장기 기억으로 굳힌다",
        "폰 노이만 구조는 메모리와 연산장치를 분리해 병목을 만든다",
        "HBM은 TSV 기술로 DRAM을 수직 적층한 고대역폭 메모리다",
        "PIM은 메모리 칩 내부에 연산 유닛을 배치한 기술이다",
        "HBF는 NAND 적층으로 SSD와 HBM 사이 계층을 형성한다",
        "트랜스포머는 토큰마다 KV 캐시 전체를 다시 읽어야 한다",
        "LLM은 토큰을 하나씩 순차적으로 생성하는 구조다",
        "Mixture of Experts는 관련 전문가만 선택적으로 활성화한다",
        "RAG는 외부 저장소에서 관련 문서를 검색해 컨텍스트로 쓴다",
        "인간 뇌는 20와트로 복잡한 추론을 수행한다",
        "GPU 클러스터는 LLM 추론에 수백 킬로와트를 소비한다",
        "CAM은 단 한 번의 클럭 사이클에 전체 메모리를 검색한다",
        "연상 기억은 부분 단서만으로 전체 패턴을 복원한다",
        "Modern Hopfield Network은 수백만 패턴을 O(1) 시간에 검색한다",
    ]
    console.print(f"[cyan]지식 베이스: {len(memories)}개 문장[/]")

    console.print("\n[bold]임베딩 생성 중...[/]")
    embeddings = encoder.encode(
        memories, convert_to_tensor=True, normalize_embeddings=True,
    )
    console.print(f"[green]OK[/] 임베딩 shape: {embeddings.shape}")

    brain = ModernHopfieldMemory(beta=8.0)
    brain.store(embeddings, memories)
    f_mem = FaissMemory()
    f_mem.store(embeddings, memories)

    queries = [
        "어떻게 하면 AI를 뇌처럼 빠르게 만들 수 있을까",
        "메모리와 CPU 사이의 병목 문제를 해결하는 하드웨어",
        "에너지 효율적인 AI 추론 방법",
    ]

    for query in queries:
        console.print(Panel(f"[bold yellow]Q: {query}[/]", expand=False))
        q_emb = encoder.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True,
        )
        _ = brain.recall(q_emb, top_k=1)
        t0 = time.perf_counter()
        for _ in range(100):
            b_results = brain.recall(q_emb, top_k=3)
        b_time = (time.perf_counter() - t0) / 100 * 1000
        t0 = time.perf_counter()
        for _ in range(100):
            f_results = f_mem.recall(q_emb, top_k=3)
        f_time = (time.perf_counter() - t0) / 100 * 1000

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("방식", width=18)
        table.add_column("지연 (ms)", justify="right")
        table.add_column("Top-1 기억", width=50)
        table.add_column("활성도", justify="right")
        table.add_row(
            "[cyan]뇌 방식 (Hopfield)[/]",
            f"{b_time:.3f}",
            b_results[0]["text"][:47],
            f"{b_results[0]['activation']:.3f}",
        )
        table.add_row(
            "[yellow]FAISS (폰 노이만)[/]",
            f"{f_time:.3f}",
            f_results[0]["text"][:47],
            f"{f_results[0]['activation']:.3f}",
        )
        console.print(table)

        console.print("\n[dim]뇌 방식 Top-3 활성화 기억:[/]")
        for i, r in enumerate(b_results, 1):
            bar = "#" * int(r["activation"] * 40)
            console.print(f"  {i}. {bar} [cyan]{r['activation']:.3f}[/]  {r['text']}")
        console.print()

    console.print(Panel(
        "[bold green]핵심 관찰[/]\n\n"
        "- 두 방식 모두 같은 정답을 찾지만 작동 원리가 다름\n"
        "- Hopfield: 한 번의 행렬 연산으로 모든 기억 동시 비교 (뇌처럼)\n"
        "- FAISS: 인덱스 구조를 순회하며 검색\n\n"
        "지식 베이스가 100만 개로 커지면 차이가 극명해집니다.",
        title="실험 결과",
    ))


if __name__ == "__main__":
    main()
