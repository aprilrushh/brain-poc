"""
Day 2 Phase B v5: Same as v4 but with robust key-article check (OR variants).
Critical articles (HBM, SK Hynix, Von Neumann) already confirmed present in v4.
"""
import os, re, json, time, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MIN_TEXT_LEN = 400
TEXT_CLIP = 1500
BATCH_SIZE = 512
OUT_DIR = "/home/ubuntu/brain-inference/data"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("Day 2 Phase B v5: FULL corpus + OR-variant key check")
print("=" * 72)

assert torch.cuda.is_available()
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n[1/4] Loading cached full English Wikipedia...")
t0 = time.perf_counter()
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(f"  {len(ds):,} docs  ({time.perf_counter()-t0:.0f}s)")

print("\n[2/4] Filtering full corpus...")
DISAMBIG = ("may refer to", "commonly refers to", "usually refers to")
docs, skipped = [], 0
t0 = time.perf_counter()
for item in ds:
    title = (item.get("title") or "").strip()
    text = (item.get("text") or "").strip()
    if len(text) < MIN_TEXT_LEN: skipped += 1; continue
    head = text[:120].lower()
    if any(m in head for m in DISAMBIG): skipped += 1; continue
    if re.match(r"^\d{1,4}s?$", title): skipped += 1; continue
    if title.startswith(("List of","Index of","Outline of")): skipped += 1; continue
    docs.append({"id": len(docs), "title": title,
                 "text": text[:TEXT_CLIP],
                 "combined": f"{title}. {text[:TEXT_CLIP]}"})
    if len(docs) % 1_000_000 == 0 and len(docs) > 0:
        print(f"  {len(docs):>9,} kept / {skipped:>9,} skipped ({time.perf_counter()-t0:.0f}s)")
print(f"\n  Final: {len(docs):,} kept, {skipped:,} skipped in {time.perf_counter()-t0:.0f}s")

# Robust OR-variant key article check
title_set = {d["title"].lower() for d in docs}
check_groups = [
    ("HBM",      ["High Bandwidth Memory"]),
    ("SK Hynix", ["SK Hynix", "SK hynix"]),
    ("Von Neumann", ["Von Neumann architecture"]),
    ("Samsung",  ["Samsung Electronics"]),
    ("SSD",      ["Solid-state drive", "Solid state drive"]),
    ("Nvidia",   ["Nvidia", "Nvidia Corporation", "NVIDIA"]),
    ("DRAM",     ["DRAM", "Dynamic random-access memory"]),
    ("Transformer", ["Transformer (deep learning architecture)",
                     "Transformer (machine learning model)",
                     "Attention (machine learning)"]),
    ("MoE",      ["Mixture of experts", "Mixture-of-experts"]),
    ("LLM",      ["Large language model"]),
]
present = 0
print("\n  Key concept coverage (OR-variant):")
for concept, variants in check_groups:
    found = next((v for v in variants if v.lower() in title_set), None)
    if found:
        present += 1
        print(f"    [OK]   {concept:12s} via '{found}'")
    else:
        print(f"    [miss] {concept:12s} — tried: {variants}")
print(f"\n  Coverage: {present}/{len(check_groups)}")

if present < 5:
    print("\n  Too low. Abort.")
    import sys; sys.exit(1)
print("  Proceeding to embedding.")

print(f"\n[3/4] BGE-M3 FP16 encoding (batch={BATCH_SIZE}) — ETA ~40 min...")
encoder = SentenceTransformer("BAAI/bge-m3", device=DEVICE).half()
t0 = time.perf_counter()
embeddings = encoder.encode(
    [d["combined"] for d in docs],
    batch_size=BATCH_SIZE, convert_to_tensor=True,
    normalize_embeddings=True, show_progress_bar=True,
)
embeddings_fp16 = embeddings.half().cpu()
embeddings = embeddings.float()
t_embed = time.perf_counter() - t0
print(f"\n  Shape: {tuple(embeddings.shape)}")
print(f"  FP16 disk: {embeddings_fp16.element_size()*embeddings_fp16.nelement()/1e9:.2f} GB")
print(f"  FP32 GPU:  {embeddings.element_size()*embeddings.nelement()/1e9:.2f} GB")
print(f"  Encoding:  {t_embed:.0f}s ({t_embed/60:.1f} min)")

print("\n[4/4] Save + sanity queries...")
torch.save(embeddings_fp16, f"{OUT_DIR}/en_wiki_embeddings.pt")
with open(f"{OUT_DIR}/en_wiki_docs.json", "w") as f:
    json.dump(docs, f, ensure_ascii=False)

def recall(q_emb, beta=50.0, k=5):
    w = torch.softmax(beta * (q_emb @ embeddings.T), dim=-1).float()
    tw, ti = torch.topk(w, k)
    return [(docs[i]["title"], v) for v, i in
            zip(tw.cpu().tolist()[0], ti.cpu().tolist()[0])]

sanity_qs = [
    "What is High Bandwidth Memory and why is it important for AI?",
    "How do transformer models use attention mechanisms?",
    "What is SK Hynix and what semiconductors do they make?",
    "Explain the von Neumann architecture and its limitations.",
    "What is the Mixture of Experts architecture?",
    "How do solid-state drives differ from hard disk drives?",
]
lats = []
for q in sanity_qs:
    qe = encoder.encode([q], convert_to_tensor=True, normalize_embeddings=True).float()
    _ = recall(qe); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): r = recall(qe)
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/10*1000; lats.append(dt)
    print(f"\n  [{dt:.2f} ms] {q}")
    for i, (t, a) in enumerate(r):
        print(f"    #{i+1} ({a:.4f})  {t}")

print(f"\n  Avg retrieval: {sum(lats)/len(lats):.2f} ms over {len(docs):,} docs")
print(f"  Coverage: {present}/{len(check_groups)}")
print("=" * 72)
