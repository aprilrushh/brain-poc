"""
RAG Orchestrator: query -> Brain -> LLM -> answer + sources.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer

from .brain import BrainMemory
from .llm_client import get_llm_client, get_llm_model, is_openrouter
from .thinking_router import decide_thinking


SYSTEM_PROMPT = """You are a research assistant. Your priority is to give the user a useful answer based on the SOURCES provided.

Primary directive:
Answer the user's question as directly and helpfully as possible, using the information in the SOURCES. If the sources contain the answer (directly OR through multi-hop reasoning that combines facts from multiple sources), give it clearly and concisely.

Multi-hop reasoning:
If the question requires combining facts from multiple sources to deduce the answer (e.g., "the wife of the man who proposed X" requires finding both the man and his wife), DO synthesize across sources. The answer does not need to appear in a single sentence — combine related facts from different sources. Only refuse if the necessary facts are genuinely absent.

When information is missing:
If the sources genuinely do not contain enough information even after multi-hop reasoning, say "I don't have information about this in the provided sources" and stop. Do not invent facts. But before refusing, check whether facts can be combined.

Style:
- Be concise. One-sentence answers for one-sentence questions.
- Cite sources inline as [Source N: <title>] when stating specific facts.
- If sources conflict, mention the conflict briefly.

Notes (apply only when relevant, do not over-apply):
- For clearly fictional entities (Wakanda, Hogwarts), you may note the fictional nature briefly if the question seems to assume reality.
- For obviously future events, you may note that the event has not occurred.
- For private information requests (passwords, personal contact details), decline politely.

Default to giving the answer. Only refuse when the sources truly lack the information."""


class RAGOrchestrator:
    def __init__(
        self,
        brain: BrainMemory,
        encoder: SentenceTransformer,
        top_k: int = 5,
        max_chunk_chars: int = 1200,
    ):
        self.brain = brain
        self.encoder = encoder
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        self.client = get_llm_client()
        self.model = get_llm_model()

    def encode_query(self, query: str) -> torch.Tensor:
        emb = self.encoder.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).float()
        return emb

    def build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        chunks = []
        for r in retrieved:
            doc = r["doc"]
            title = doc.get("title", "untitled")
            page = doc.get("page", "?")
            text = doc.get("text", "")[: self.max_chunk_chars]
            score = r["score"]
            chunks.append(
                f"[Source {r['rank']}: {title}, p{page}] (relevance={score:.4f})\n{text}"
            )
        return "\n\n".join(chunks)

    def query(
        self,
        question: str,
        user_thinking_override: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run full pipeline: encode -> retrieve -> generate.
        """
        timings = {}

        t0 = time.perf_counter()
        q_emb = self.encode_query(question)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retrieved = self.brain.recall(q_emb, top_k=self.top_k)
        retrieval_pattern = self.brain.retrieval_pattern(q_emb, top_k=self.top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        thinking_enabled, thinking_reason = decide_thinking(
            question, retrieval_pattern, user_thinking_override
        )

        context = self.build_context(retrieved)
        user_msg = f"Sources:\n\n{context}\n\nQuestion: {question}"
        if not thinking_enabled:
            user_msg = "/no_think " + user_msg

        t0 = time.perf_counter()
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        if is_openrouter():
            kwargs["extra_body"] = {"reasoning": {"enabled": thinking_enabled}}

        response = self.client.chat.completions.create(**kwargs)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        timings["total_ms"] = (
            timings["embed_ms"] + timings["retrieve_ms"] + timings["generate_ms"]
        )

        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details"):
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        return {
            "question": question,
            "answer": answer.strip(),
            "retrieved": [
                {
                    "rank": r["rank"],
                    "score": r["score"],
                    "title": r["doc"].get("title"),
                    "page": r["doc"].get("page"),
                    "url": r["doc"].get("url", ""),
                    "text": (r["doc"].get("text", "") or "")[:300],
                }
                for r in retrieved
            ],
            "thinking_enabled": thinking_enabled,
            "thinking_reason": thinking_reason,
            "retrieval_pattern": retrieval_pattern,
            "timings": timings,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "reasoning_tokens": reasoning_tokens,
            },
        }
