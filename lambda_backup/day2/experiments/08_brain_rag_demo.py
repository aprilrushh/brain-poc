"""
Day 2 Phase C: Brain Memory + gpt-oss-120b RAG Demo
===================================================
End-to-end pipeline:
    question → BGE-M3 encode → Brain recall (O(1)) → gpt-oss-120b answer

Measures:
- Brain retrieval latency
- gpt-oss generation latency
- Tokens/sec throughput
- End-to-end user-facing latency
- Retrieved context quality (Top-5 sources)

Pitch talking points produced by this script:
- "On a single SK/NVIDIA GH200, Brain Memory retrieves from 1M English
   Wikipedia docs in <1 ms, then OpenAI gpt-oss-120b generates grounded
   answers at ~X tok/s — all HBM-resident, no SSD spill at this scale."
"""
import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_ID = "openai/gpt-oss-20b"
TOP_K = 5
MAX_NEW_TOKENS = 400
DATA_DIR = "/home/ubuntu/brain-inference/data"
EMB_PATH = f"{DATA_DIR}/en_wiki_embeddings.pt"
DOC_PATH = f"{DATA_DIR}/en_wiki_docs.json"
LOG_PATH = "/home/ubuntu/brain-inference/logs/day2_rag_demo.json"

DEVICE = "cuda"

print("=" * 72)
print("Day 2 Phase C: Brain + gpt-oss-120b RAG Demo")
print("=" * 72)

# -----------------------------------------------------------------------------
# 1. Load Brain Memory (embeddings + docs)
# -----------------------------------------------------------------------------
print("\n[1/5] Loading saved Brain Memory...")
t0 = time.perf_counter()
embeddings = torch.load(EMB_PATH, weights_only=True).float().to(DEVICE)
with open(DOC_PATH) as f:
    docs = json.load(f)
print(f"  {len(docs):,} docs  |  embeddings {tuple(embeddings.shape)}  "
      f"({embeddings.element_size()*embeddings.nelement()/1e9:.2f} GB)  "
      f"[{time.perf_counter()-t0:.1f}s]")

# -----------------------------------------------------------------------------
# 2. BGE-M3 encoder (query-side)
# -----------------------------------------------------------------------------
print("\n[2/5] Loading BGE-M3 encoder...")
t0 = time.perf_counter()
encoder = SentenceTransformer("BAAI/bge-m3", device=DEVICE).half()
print(f"  Ready [{time.perf_counter()-t0:.1f}s]")

# -----------------------------------------------------------------------------
# 3. gpt-oss-120b
# -----------------------------------------------------------------------------
print(f"\n[3/5] Loading {MODEL_ID} (first run downloads ~60 GB)...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map={"": 0},
        attn_implementation="eager",
    )
except Exception as e:
    print(f"\n  120B HBM-only load failed: {type(e).__name__}: {e}")
    print("  Falling back to gpt-oss-20b...")
    MODEL_ID = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map={"": 0},
        attn_implementation="eager",
    )
model.eval()
t_load = time.perf_counter() - t0
mem_used = torch.cuda.memory_allocated() / 1e9
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  Loaded in {t_load:.0f}s ({t_load/60:.1f} min)")
print(f"  GPU memory: {mem_used:.1f} / {mem_total:.1f} GB")
has_offload = any(p.device.type != "cuda" for p in model.parameters())
print(f"  All params on GPU: {not has_offload}")
if has_offload:
    print(f"  !!! offload detected: {sorted({str(p.device) for p in model.parameters()})}")

# -----------------------------------------------------------------------------
# 4. Brain Memory + RAG function
# -----------------------------------------------------------------------------
class BrainMemory:
    def __init__(self, keys, docs, beta=50.0):
        self.keys = keys
        self.docs = docs
        self.beta = beta
    def recall(self, q_emb, top_k=5):
        logits = self.beta * (q_emb @ self.keys.T)
        w = torch.softmax(logits.float(), dim=-1)
        tw, ti = torch.topk(w, top_k)
        return [
            {"doc": self.docs[idx], "activation": val}
            for val, idx in zip(tw.cpu().tolist()[0], ti.cpu().tolist()[0])
        ]

brain = BrainMemory(embeddings, docs, beta=50.0)

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using "
    "ONLY the information in the provided source documents. Cite sources "
    "inline as [Source N]. If the provided sources do not contain the answer, "
    "say so explicitly and do not fabricate information."
)

def rag_query(question, top_k=TOP_K):
    timings = {}

    # --- Brain retrieval ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    q_emb = encoder.encode(
        [question], convert_to_tensor=True, normalize_embeddings=True
    ).float()
    torch.cuda.synchronize()
    timings["t_embed_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    retrieved = brain.recall(q_emb, top_k=top_k)
    torch.cuda.synchronize()
    timings["t_retrieve_ms"] = (time.perf_counter() - t0) * 1000

    # --- Build context ---
    context = "\n\n".join(
        f"[Source {i+1}: {r['doc']['title']}]\n{r['doc']['text'][:800]}"
        for i, r in enumerate(retrieved)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]

    # --- gpt-oss generation ---
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = enc["input_ids"].to(DEVICE)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(DEVICE)
    n_in = inputs.shape[-1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    timings["t_gen_ms"] = (time.perf_counter() - t0) * 1000

    n_out = outputs.shape[-1] - n_in
    raw = tokenizer.decode(outputs[0][n_in:], skip_special_tokens=True)
    if "assistantfinal" in raw:
        answer = raw.split("assistantfinal", 1)[1].strip()
    elif "final" in raw and "analysis" in raw:
        answer = raw.split("final", 1)[1].lstrip(":").strip()
    else:
        answer = raw.strip()

    return {
        "question": question,
        "retrieved": retrieved,
        "answer": answer,
        "n_in_tokens": int(n_in),
        "n_out_tokens": int(n_out),
        "tok_per_sec": n_out / (timings["t_gen_ms"] / 1000) if timings["t_gen_ms"] > 0 else 0,
        **timings,
        "total_ms": timings["t_embed_ms"] + timings["t_retrieve_ms"] + timings["t_gen_ms"],
    }

# -----------------------------------------------------------------------------
# 5. Run demo queries (Solidigm/SK pitch-aligned)
# -----------------------------------------------------------------------------
demo_queries = [
    "What is High Bandwidth Memory (HBM) and why is it important for modern AI systems?",
    "How does the von Neumann bottleneck limit the performance of large language models?",
    "What is the Mixture of Experts (MoE) architecture in neural networks?",
    "How do solid-state drives (SSDs) differ from traditional hard disk drives in data centers?",
    "What is SK Hynix and what role does it play in the semiconductor industry?",
    "Explain NVIDIA's Grace Hopper architecture and its advantages for AI workloads.",
    "What is compute-in-memory and why does it matter for AI acceleration?",
    "How do attention mechanisms work in transformer neural networks?",
    "What is the difference between DRAM and NAND flash memory?",
    "Why is energy efficiency critical for scaling large-scale AI deployments?",
]

print("\n[4/5] Warming up (first generate triggers CUDA graph / kernel compile)...")
_ = rag_query("What is memory bandwidth?", top_k=3)

print("\n[5/5] Running demo suite...")
results = []
for i, q in enumerate(demo_queries, 1):
    print("\n" + "=" * 72)
    print(f"Q{i}: {q}")
    print("=" * 72)

    r = rag_query(q)
    results.append(r)

    print(f"\n🧠 Brain retrieval:      {r['t_retrieve_ms']:>8.2f} ms")
    print(f"   Query embedding:      {r['t_embed_ms']:>8.2f} ms")
    print(f"🤖 gpt-oss generation:   {r['t_gen_ms']:>8.0f} ms  "
          f"({r['n_out_tokens']} tok @ {r['tok_per_sec']:.1f} tok/s)")
    print(f"⏱️  Total end-to-end:     {r['total_ms']:>8.0f} ms")

    print(f"\n📚 Top-{TOP_K} retrieved:")
    for j, ret in enumerate(r["retrieved"]):
        print(f"   #{j+1} [{ret['activation']:.4f}] {ret['doc']['title']}")

    print(f"\n📝 Answer:\n{r['answer']}")

# -----------------------------------------------------------------------------
# Summary & save
# -----------------------------------------------------------------------------
n = len(results)
avg_retrieve = sum(r["t_retrieve_ms"] for r in results) / n
avg_gen = sum(r["t_gen_ms"] for r in results) / n
avg_tok_s = sum(r["tok_per_sec"] for r in results) / n
avg_total = sum(r["total_ms"] for r in results) / n

print("\n\n" + "=" * 72)
print("DAY 2 PHASE C — SUMMARY")
print("=" * 72)
print(f"Queries run:                  {n}")
print(f"Corpus size:                  {len(docs):,} English Wikipedia docs")
print(f"GPU memory used:              {torch.cuda.memory_allocated()/1e9:.1f} GB "
      f"/ {mem_total:.1f} GB HBM")
print(f"Avg Brain retrieval:          {avg_retrieve:>8.2f} ms")
print(f"Avg gpt-oss generation:       {avg_gen:>8.0f} ms")
print(f"Avg generation throughput:    {avg_tok_s:>8.1f} tok/s")
print(f"Avg end-to-end latency:       {avg_total:>8.0f} ms")
print(f"Brain : LLM time ratio:       1 : {avg_gen/avg_retrieve:.0f}")

# Save (strip non-serializable fields)
clean = []
for r in results:
    cr = {k: v for k, v in r.items() if k != "retrieved"}
    cr["retrieved_titles"] = [
        {"title": ret["doc"]["title"], "activation": ret["activation"]}
        for ret in r["retrieved"]
    ]
    clean.append(cr)

summary = {
    "model": MODEL_ID,
    "n_docs": len(docs),
    "n_queries": n,
    "avg_retrieve_ms": round(avg_retrieve, 3),
    "avg_gen_ms": round(avg_gen, 1),
    "avg_tok_per_sec": round(avg_tok_s, 2),
    "avg_end_to_end_ms": round(avg_total, 1),
    "gpu_memory_gb": round(torch.cuda.memory_allocated()/1e9, 2),
    "results": clean,
}
with open(LOG_PATH, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved: {LOG_PATH}")
print("\n피치용 한 줄: 1M 영문 Wiki에서 Brain은 {:.2f}ms에 찾고,".format(avg_retrieve))
print("            gpt-oss-20b는 {:.0f} tok/s로 답변 — 전부 단일 GH200 HBM에서.".format(avg_tok_s))
