"""
RAG Orchestrator: query -> Brain -> LLM -> answer + sources.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional

import concurrent.futures
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


GENERAL_KNOWLEDGE_PROMPT = """You are a helpful assistant answering from your general knowledge.

The user is using a document-grounded research tool, but they may also benefit from broader context. Provide a concise answer to their question using your general knowledge.

Style:
- Be concise (2-4 sentences).
- Do NOT cite sources or pretend you have access to documents.
- Start with the direct answer.
- If the question asks about something fictional or impossible (Wakanda GDP, future events), briefly note that.
- If you genuinely don't know, say so. Don't invent facts.

This answer will be displayed alongside a separate document-grounded answer; the user will see both."""


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
        # adapter is LLMAdapter (OpenAIAdapter or AnthropicAdapter)
        self.adapter = get_llm_client()
        self.model = self.adapter.model

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
        extra_context: Optional[str] = None,
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
        # D4.5d-bonus: chat-scoped attached files (Claude-style 📎) injected as separate section
        attached_section = ""
        if extra_context:
            attached_section = f"\n\n=== Attached files (chat-scoped) ===\n{extra_context}\n=== End attached ===\n"
        user_msg = f"Sources:\n\n{context}{attached_section}\n\nQuestion: {question}"
        if not thinking_enabled:
            user_msg = "/no_think " + user_msg

        t0 = time.perf_counter()
        # Build provider-aware kwargs (shared for both calls)
        base_kwargs = {"max_tokens": max_tokens, "temperature": 0.7}
        if is_openrouter():
            base_kwargs["extra_body"] = {
                "reasoning": {"enabled": thinking_enabled}
            }

        # Two parallel calls:
        #  1. Document-grounded (uses SYSTEM_PROMPT + sources)
        #  2. General knowledge (uses GENERAL_KNOWLEDGE_PROMPT, no sources)
        def _call_doc():
            return self.adapter.chat_complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                **base_kwargs,
            )

        def _call_general():
            return self.adapter.chat_complete(
                messages=[
                    {"role": "system", "content": GENERAL_KNOWLEDGE_PROMPT},
                    {"role": "user", "content": question},
                ],
                **base_kwargs,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_doc = ex.submit(_call_doc)
            fut_general = ex.submit(_call_general)
            try:
                result = fut_doc.result(timeout=60)
            except Exception as e:
                result = {"text": f"[Document answer error: {e}]",
                          "input_tokens": 0, "output_tokens": 0,
                          "stop_reason": "error", "model": self.model}
            try:
                result_general = fut_general.result(timeout=60)
            except Exception as e:
                result_general = {"text": f"[General answer error: {e}]",
                                  "input_tokens": 0, "output_tokens": 0,
                                  "stop_reason": "error", "model": self.model}

        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        answer = result["text"]
        answer_general = result_general["text"]
        timings["total_ms"] = (
            timings["embed_ms"] + timings["retrieve_ms"] + timings["generate_ms"]
        )

        reasoning_tokens = 0

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
            "answer_general": answer_general.strip(),
            "usage": {
                "prompt_tokens": result["input_tokens"] + result_general["input_tokens"],
                "completion_tokens": result["output_tokens"] + result_general["output_tokens"],
                "reasoning_tokens": reasoning_tokens,
                "doc_prompt_tokens": result["input_tokens"],
                "doc_completion_tokens": result["output_tokens"],
                "general_prompt_tokens": result_general["input_tokens"],
                "general_completion_tokens": result_general["output_tokens"],
            },
            "stop_reason": result["stop_reason"],
            "model_used": result["model"],
        }
