"""
Brain Memory — Modern Hopfield Network based associative retrieval.

Algorithm:
    weights = softmax(beta * Q * K^T)
    top_k = topk(weights, k=5)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


def auto_device() -> str:
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BrainMemory:
    """
    Modern Hopfield associative memory.

    Stores normalized embeddings (keys) and corresponding docs.
    On query: softmax(beta * Q * K^T) -> top-k docs.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        docs: List[Dict[str, Any]],
        beta: float = 50.0,
        device: Optional[str] = None,
    ):
        self.device = device or auto_device()
        self.keys = keys.to(self.device).float()
        self.docs = docs
        self.beta = beta

        if self.keys.shape[0] != len(self.docs):
            raise ValueError(
                f"keys ({self.keys.shape[0]}) and docs ({len(self.docs)}) "
                f"must have the same length"
            )

    def __len__(self) -> int:
        return len(self.docs)

    def recall(
        self,
        q_emb: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k docs.

        Args:
            q_emb: (1, D) or (D,) tensor, expected to be normalized.
            top_k: how many docs to return.

        Returns:
            List of {doc, score, rank} dicts, sorted by descending score.
        """
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        q_emb = q_emb.to(self.device).float()

        # Modern Hopfield: softmax(beta * Q * K^T)
        logits = self.beta * (q_emb @ self.keys.T)
        weights = torch.softmax(logits, dim=-1)
        tw, ti = torch.topk(weights, min(top_k, len(self.docs)), dim=-1)

        scores = tw[0].cpu().tolist()
        indices = ti[0].cpu().tolist()

        return [
            {
                "rank": rank + 1,
                "score": float(score),
                "doc": self.docs[idx],
            }
            for rank, (score, idx) in enumerate(zip(scores, indices))
        ]

    def retrieval_pattern(
        self,
        q_emb: torch.Tensor,
        top_k: int = 5,
    ) -> str:
        """
        Classify retrieval distribution: 'sharp' | 'flat' | 'weak'.
        Used by thinking_router.
        """
        results = self.recall(q_emb, top_k=top_k)
        if not results:
            return "weak"
        scores = [r["score"] for r in results]
        top1 = scores[0]
        if top1 < 0.10:
            return "weak"
        if len(scores) < 2:
            return "sharp"
        gap = top1 - scores[1]
        if gap > 0.15 and top1 > 0.30:
            return "sharp"
        return "flat"

    def save(self, dir_path: str) -> None:
        """Save keys.pt + docs.json to dir_path."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.keys.half().cpu(), path / "keys.pt")
        with open(path / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=None)
        meta = {"beta": self.beta, "n_docs": len(self.docs),
                "dim": self.keys.shape[-1]}
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str, device: Optional[str] = None) -> "BrainMemory":
        """Load keys.pt + docs.json from dir_path."""
        path = Path(dir_path)
        keys = torch.load(path / "keys.pt", weights_only=True)
        with open(path / "docs.json", encoding="utf-8") as f:
            docs = json.load(f)
        beta = 50.0
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                beta = json.load(f).get("beta", 50.0)
        return cls(keys=keys, docs=docs, beta=beta, device=device)


def benchmark_recall(
    brain: BrainMemory,
    q_emb: torch.Tensor,
    top_k: int = 5,
    n_iter: int = 10,
) -> float:
    """Average recall latency in milliseconds."""
    _ = brain.recall(q_emb, top_k=top_k)
    if brain.device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = brain.recall(q_emb, top_k=top_k)
    if brain.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000
    return elapsed_ms
