"""
Track 2 sanity #0: vLLM + AWQ-INT4 only (FlexKV OFF).
Just verify the model loads and generates correctly.
"""
import time
from vllm import LLM, SamplingParams

MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

t0 = time.time()
llm = LLM(
    model=MODEL,
    quantization="awq_marlin",  # H100 supports marlin kernel
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enable_prefix_caching=False,
    dtype="float16",
)
load_t = time.time() - t0
print(f"\n=== Load complete: {load_t:.1f}s ===\n")

prompts = [
    "The capital of France is",
    "Quantization in language models is",
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_t = time.time() - t0

print(f"\n=== Generate complete: {gen_t:.2f}s ===\n")
for o in outputs:
    print(f"PROMPT: {o.prompt!r}")
    print(f"GEN: {o.outputs[0].text!r}")
    print("-" * 60)
