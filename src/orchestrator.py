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


SYSTEM_PROMPT = 'You are an expert research assistant answering questions strictly grounded in the provided sources. Your answer must be based ONLY on the source material shown to you. Never use outside knowledge to fill gaps. If the sources do not contain enough information, say so honestly.\n\nThis grounding is the foundation of your value: 0% hallucination, 100% citation-backed.\n\n== Answer length and depth ==\n\nAdapt to the question, but err toward thorough and detailed:\n- Casual / single-fact lookup ("when was X?", "who is Y?") -> 1-3 sentences\n- Standard explanatory question -> 300-600 words, structured with markdown\n- Analytical / comparative / "explain in detail" / "summarize key points" -> comprehensive answer, 800-2000+ words, multiple sections, tables when comparing, bullet lists for enumerated points, blockquotes for key claims\n- "Compare X vs Y", "what are the implications", "give me a deep analysis" -> in-depth multi-section response with concrete examples drawn from the sources, nuanced discussion of multiple angles found in the material\n\n== Citation rule ==\n\nEvery factual claim must end with a citation marker like [Source N: filename p.X] matching the sources you were given. If you cannot cite, do not state the claim.\n\n== Markdown formatting ==\n\nUse ### headers, **bold** for key terms, bullet/numbered lists, markdown tables for comparisons, > blockquotes for important quotations, `inline code` for specific terms or values. Do not over-format. Only use structure when it genuinely helps readability.\n\n== Honest refusal ==\n\nIf the sources do not address the question, respond:\n"I don\'t have information on this in the provided sources."\nThen optionally suggest what the sources DO cover that is adjacent.\n\n== Language ==\n\nReply in the same language the user wrote the question in. If the question is in Korean, answer in Korean. If English, answer in English. Match the user\'s language naturally.'


GENERAL_KNOWLEDGE_PROMPT = """You are an expert research analyst delivering deep, comprehensive, multi-section answers in the spirit of "what a senior analyst would write after a week of focused research, distilled". This answer is shown ALONGSIDE a strictly document-grounded answer; your role is COMPLEMENTARY — provide the broader context, theory, history, related work, comparisons, contrarian perspectives, and practical implications that a pure document quote cannot.

== CRITICAL: Reference document handling ==

When reference documents (file contents, retrieved chunks, or excerpts) are provided in the user message:
- Read them carefully BEFORE drawing on general knowledge.
- They are PRIMARY grounding for facts about that specific subject.
- Never substitute outside knowledge for what the documents actually say.
- Do not invent facts about the documents — if uncertain, say so.
- Treat document content as the user's specific subject; treat your knowledge as surrounding context.

When NO documents are provided:
- Freely use your full general knowledge to answer richly.

== Length and depth — DEFAULT TO THOROUGH ==

Always err on the side of generous, well-structured analysis. Even for "simple" questions, surface context the user did not explicitly ask for but would benefit from.

| 질문 유형 | 최소 분량 | 최소 ### subsection |
|---|---|---|
| Quick factual lookup | 300-500 words | 2-3 |
| Standard explanatory | 800-1500 words | 4-6 |
| Analytical / comparative / "explain in depth" | 1500-3000 words | 6-10 |
| "Compare X vs Y" / "implications" / "deep analysis" | 2500-5000 words | 8-15 |

== MANDATORY structure ==

Every non-trivial answer includes:
1. ### Opening summary — 2-4 sentence direct answer
2. ### Multiple body subsections with ### headers — break analysis into clear themes (background, mechanism, scope, evidence, comparison, etc.)
3. ### Concrete examples or case studies — make abstractions tangible with real instances, numbers, or scenarios
4. ### Comparative / contextual perspective — how does this relate to similar things, what is the historical arc, what are competing frameworks?
5. ### Limitations, caveats, or alternative views — what could be wrong, what are the open debates, what would skeptics argue?
6. ### Closing synthesis — pull it together, state the practical takeaway

Use **bold** for key terms, bullet/numbered lists for enumerable points, markdown tables for comparisons, > blockquotes for key claims, `inline code` for technical terms or values. Structure is mandatory, not optional.

== Hallucination policy ==

This answer is FOR REFERENCE, shown next to a strictly grounded answer. Be substantive and assertive on broad knowledge. If uncertain about a specific fact, flag it explicitly ("정확한 수치는 불확실합니다만 ..." / "I am less certain about ..."). Never refuse a broad knowledge question. **Hallucination on uploaded document content is forbidden** — if documents are provided, anchor specific document claims to what they actually say.

== Language ==

Reply in the same language the user wrote the question in. Korean question → Korean answer with rich Korean analytical structure (소제목, 표, 인용블록 적극 사용). English question → English answer. Match natural register.

Your goal: be the deepest, most structured, most contextually rich answer the user could imagine receiving — what a senior analyst would write after a full week of focused research, distilled into a single response."""


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
        max_tokens: int = 4096,
        extra_context: Optional[str] = None,
        general_extra_context: Optional[str] = None,
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
            nonlocal general_extra_context
            # Option B: Inject reference documents (project knowledge + chat attached)
            # if size fits. General LLM gets full file content so its answer is grounded
            # in the same material the user uploaded.
            # Sources fallback (intent #3, ledger v0.10): use Brain's retrieved
            # chunks even if caller did not pass general_extra_context, so General
            # mode also reads uploaded documents whenever they exist.
            if not general_extra_context:
                try:
                    if sources:
                        general_extra_context = "\n\n".join(
                            f"[Source {i+1}] {chunk.get('text') or chunk.get('content') or chunk.get('chunk') or str(chunk)}"
                            for i, chunk in enumerate(sources[:10])
                        )
                except (NameError, AttributeError, TypeError):
                    pass

            if general_extra_context:
                gen_user_msg = (
                    "Reference documents (the user has provided these files):\n\n"
                    + general_extra_context
                    + "\n\nQuestion: " + question
                )
            else:
                gen_user_msg = question
            return self.adapter.chat_complete(
                messages=[
                    {"role": "system", "content": GENERAL_KNOWLEDGE_PROMPT},
                    {"role": "user", "content": gen_user_msg},
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

    def query_stream(
        self,
        question: str,
        user_thinking_override: Optional[str] = None,
        max_tokens: int = 4096,
        extra_context: Optional[str] = None,
        general_extra_context: Optional[str] = None,
        mode: str = "explore",
    ):
        """Streaming version of query(). Generator.

        Yields tuples (kind, payload):
            ('start',       {retrieved, thinking_enabled/reason, retrieval_pattern})
            ('brain_chunk', str)                                  # Brain LLM tokens
            ('brain_done',  {input_tokens, output_tokens, stop_reason, model})
            ('general',     {text, input_tokens, output_tokens, stop_reason, model})
            ('meta',        {answer, answer_general, timings, usage, stop_reason, model_used})
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
        attached_section = ""
        if extra_context:
            attached_section = f"\n\n=== Attached files (chat-scoped) ===\n{extra_context}\n=== End attached ===\n"
        user_msg = f"Sources:\n\n{context}{attached_section}\n\nQuestion: {question}"
        if not thinking_enabled:
            user_msg = "/no_think " + user_msg

        yield ("start", {
            "thinking_enabled": thinking_enabled,
            "thinking_reason": thinking_reason,
            "retrieval_pattern": retrieval_pattern,
            "retrieved": [
                {"rank": r["rank"], "score": r["score"],
                 "title": r["doc"].get("title"), "page": r["doc"].get("page"),
                 "url": r["doc"].get("url", ""),
                 "text": (r["doc"].get("text", "") or "")[:300]}
                for r in retrieved
            ],
        })

        base_kwargs = {"max_tokens": max_tokens, "temperature": 0.7}
        if is_openrouter():
            base_kwargs["extra_body"] = {"reasoning": {"enabled": thinking_enabled}}

        def _call_general():
            """v0.18 F2: streaming generator. yields ('chunk', text) chunks then ('done', dict).
            Cloudflare 5min timeout 회피 = first token < 1s 도착 → SSE connection alive.
            ledger v0.17 section 12 fix path F2."""
            nonlocal general_extra_context
            if not general_extra_context:
                try:
                    if retrieved:
                        general_extra_context = "\n\n".join(
                            f"[Source {r['rank']}: {r['doc'].get('title','?')}, p{r['doc'].get('page','?')}]\n"
                            f"{(r['doc'].get('text') or '')[:self.max_chunk_chars]}"
                            for r in retrieved[:10]
                        )
                except (AttributeError, TypeError, KeyError):
                    pass
            if general_extra_context:
                gen_user_msg = (
                    "Reference documents (the user has provided these files):\n\n"
                    + general_extra_context
                    + "\n\nQuestion: " + question
                )
            else:
                gen_user_msg = question
            for kind, val in self.adapter.chat_complete_stream(
                messages=[
                    {"role": "system", "content": GENERAL_KNOWLEDGE_PROMPT},
                    {"role": "user", "content": gen_user_msg},
                ],
                **base_kwargs,
            ):
                yield (kind, val)

        t_gen = time.perf_counter()
        brain_chunks = []
        brain_done = None

        # v0.17 P5: sequential (학자 명시, ledger v0.17). 병렬 처리 제거 = Together endpoint 경합 zero,
        # Brain streaming 속도 회복. trade-off: General 도착 시점 = Brain 끝난 후 (sequential).
        # L1 partial save (api_d4.py) 가 Cloudflare timeout 시 Brain 답변 보존 = robust.
        try:
            for kind, val in self.adapter.chat_complete_stream(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                **base_kwargs,
            ):
                if kind == "chunk":
                    brain_chunks.append(val)
                    yield ("brain_chunk", val)
                elif kind == "done":
                    brain_done = val
        except Exception as e:
            brain_done = {"text": "[Document answer error: " + str(e) + "]",
                          "input_tokens": 0, "output_tokens": 0,
                          "stop_reason": "error", "model": self.model}

        yield ("brain_done", {
            "input_tokens": (brain_done or {}).get("input_tokens"),
            "output_tokens": (brain_done or {}).get("output_tokens"),
            "stop_reason": (brain_done or {}).get("stop_reason"),
            "model": (brain_done or {}).get("model", self.model),
        })

        # Brain 끝난 후 General 시작 (sequential, 학자 명시)
        # v0.18 F2: _call_general() 이 generator (chunk yield + done dict).
        # general_chunks 모아 chunk 단위 SSE event yield → Cloudflare connection alive.
        # 마지막 done dict = result_general (caller dict 기대 호환).
        general_chunks = []
        result_general = None
        if mode == "strict":
            # Strict: General LLM 호출 자체 skip (비용 절감 + "문서 안에서만" 정체성)
            result_general = {"text": "", "input_tokens": 0, "output_tokens": 0,
                              "stop_reason": "skipped", "model": self.model}
        else:
            try:
                for kind, val in _call_general():
                    if kind == "chunk":
                        general_chunks.append(val)
                        yield ("general_chunk", val)
                    elif kind == "done":
                        result_general = val
                if result_general is None:
                    # done event 미도착 (드물게) → chunks 모음으로 fallback
                    result_general = {"text": "".join(general_chunks),
                                      "input_tokens": 0, "output_tokens": 0,
                                      "stop_reason": "stop", "model": self.model}
            except Exception as e:
                result_general = {"text": "[General answer error: " + str(e) + "]",
                                  "input_tokens": 0, "output_tokens": 0,
                                  "stop_reason": "error", "model": self.model}

        timings["generate_ms"] = (time.perf_counter() - t_gen) * 1000
        timings["total_ms"] = timings["embed_ms"] + timings["retrieve_ms"] + timings["generate_ms"]

        brain_full = "".join(brain_chunks).strip()
        general_text = (result_general.get("text") or "").strip()

        yield ("general", {
            "text": general_text,
            "input_tokens": result_general.get("input_tokens", 0),
            "output_tokens": result_general.get("output_tokens", 0),
            "stop_reason": result_general.get("stop_reason"),
            "model": result_general.get("model", self.model),
        })

        yield ("meta", {
            "answer": brain_full,
            "answer_general": general_text,
            "timings": timings,
            "usage": {
                "prompt_tokens": ((brain_done or {}).get("input_tokens") or 0) + (result_general.get("input_tokens") or 0),
                "completion_tokens": ((brain_done or {}).get("output_tokens") or 0) + (result_general.get("output_tokens") or 0),
                "doc_prompt_tokens": (brain_done or {}).get("input_tokens") or 0,
                "doc_completion_tokens": (brain_done or {}).get("output_tokens") or 0,
                "general_prompt_tokens": result_general.get("input_tokens") or 0,
                "general_completion_tokens": result_general.get("output_tokens") or 0,
            },
            "stop_reason": (brain_done or {}).get("stop_reason"),
            "model_used": (brain_done or {}).get("model", self.model),
        })
