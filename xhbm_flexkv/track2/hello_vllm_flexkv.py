"""
Track 2 baseline #1: vLLM + AWQ-INT4 + FlexKV CPU offload (32GB).

baseline #0 와의 유일한 차이:
  - enable_prefix_caching=True
  - kv_transfer_config={"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}
  - FLEXKV_CPU_CACHE_GB=32 (env)

prefix_caching_flexkv.py 의 패턴: 같은 prefix 를 가진 prompt 들을 두 번 generate.
첫 generate = warmup (FlexKV demote 발생). 두 번째 = prefix cache hit 기대.
"""
import os
import time

# FlexKV 환경변수 — config 파일 안 쓰고 env 로
os.environ["FLEXKV_CPU_CACHE_GB"] = "32"
os.environ["FLEXKV_SSD_CACHE_GB"] = "0"  # CPU only 먼저
os.environ.pop("FLEXKV_CONFIG_PATH", None)

from vllm import LLM, SamplingParams

MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

# 긴 system prompt (= shared prefix). prefix caching 효과 보려면 길어야 함.
PREFIX = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: "
)

prompts_short = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
generating_prompts = [PREFIX + p for p in prompts_short]

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

print("=== Track 2 baseline #1: vLLM + AWQ + FlexKV CPU offload ===\n")

t0 = time.time()
llm = LLM(
    model=MODEL,
    quantization="awq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enable_prefix_caching=True,
    kv_transfer_config={
        "kv_connector": "FlexKVConnectorV1",
        "kv_role": "kv_both",
    },
    dtype="float16",
)
load_t = time.time() - t0
print(f"\n=== Load: {load_t:.1f}s ===\n")

# Warmup: 같은 prefix 한 번 generate (FlexKV 가 KV demote 시작)
print("=== Warmup (1 prompt, prefix 채우기) ===")
t0 = time.time()
_ = llm.generate(generating_prompts[0], sampling_params)
warmup_t = time.time() - t0
print(f"warmup gen: {warmup_t:.2f}s\n")

# offload 끝날 때까지 대기 (prefix_caching_flexkv.py 와 동일)
time.sleep(2)

# 첫 본 측정: 4 prompts
print("=== Run 1 (4 prompts, prefix 1번 캐시됨) ===")
t0 = time.time()
outputs = llm.generate(generating_prompts, sampling_params)
run1_t = time.time() - t0
print(f"run 1 gen: {run1_t:.2f}s\n")
for o in outputs:
    print(f"  PROMPT(short): {o.prompt[len(PREFIX):]!r}")
    print(f"  GEN: {o.outputs[0].text[:80]!r}")
print()

# prefix cache reset → FlexKV 만 사용
print("=== reset_prefix_cache (FlexKV 만 활용) ===")
llm.reset_prefix_cache()
time.sleep(2)

# 두 번째 본 측정: 같은 4 prompts (FlexKV 에서 promote 기대)
print("=== Run 2 (4 prompts, FlexKV 가 promote) ===")
t0 = time.time()
outputs = llm.generate(generating_prompts, sampling_params)
run2_t = time.time() - t0
print(f"run 2 gen: {run2_t:.2f}s\n")
for o in outputs:
    print(f"  PROMPT(short): {o.prompt[len(PREFIX):]!r}")
    print(f"  GEN: {o.outputs[0].text[:80]!r}")

print("\n=== Summary ===")
print(f"  Load + init: {load_t:.1f}s")
print(f"  Warmup gen (1 prompt):    {warmup_t:.2f}s")
print(f"  Run 1 gen (4 prompts, prefix-cache): {run1_t:.2f}s")
print(f"  Run 2 gen (4 prompts, FlexKV promote): {run2_t:.2f}s")
