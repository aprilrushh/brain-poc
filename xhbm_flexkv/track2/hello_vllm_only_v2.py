"""
Track 2 sanity #0 v2: 일반 AWQ (Marlin 우회).
"""
import time
from vllm import LLM, SamplingParams

MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

t0 = time.time()
llm = LLM(
    model=MODEL,
    quantization="awq",  # ← 명시. auto-detect 가 awq_marlin 으로 가지 않게
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enable_prefix_caching=False,
    dtype="float16",
)
load_t = time.time() - t0
print(f"\n=== Load: {load_t:.1f}s ===\n")

prompts = [
    "The capital of France is",
    "Quantization in language models is",
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_t = time.time() - t0

print(f"\n=== Gen: {gen_t:.2f}s ===\n")
for o in outputs:
    print(f"PROMPT: {o.prompt!r}")
    print(f"GEN: {o.outputs[0].text!r}")
    print("-" * 60)
