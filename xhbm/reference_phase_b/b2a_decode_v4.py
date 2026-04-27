#!/usr/bin/env python3
"""
B2a Decode v4 — V offload to CPU RAM with int8 per-token quantization.
Based on v3b (pre-allocated pinned buffers), but V is quantized fp16 -> int8 on spill,
dequantized int8 -> fp16 on reload.
Transfer volume halved; argmax accuracy measured against baseline.
"""
import argparse, os, time, json, subprocess, gc, hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_PATH = "/home/ubuntu/models/Llama-3.1-8B-Instruct"

def nvsmi_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    return int(r.stdout.strip().split("\n")[0])

def cpu_ram_used_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1

def quantize_per_token(v_fp):
    """
    Symmetric per-token int8 quantization.
    V shape: [B, H, S, D]. Quantize along dim=-1 (D) for each (B, H, S) triple.
    Returns (int8_tensor, scale_fp16). scale shape: [B, H, S, 1].
    """
    absmax = v_fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 127.0
    q = torch.round(v_fp / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)

def dequantize_per_token(q_int8, scale_fp16):
    return q_int8.to(torch.float16) * scale_fp16

class XHBMCacheV4(DynamicCache):
    """
    V-only offload to pinned CPU RAM, with int8 per-token quantization.
    Pre-allocated pinned buffers at max_seq_len.
    """
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._cpu_bytes_peak = 0
        self._d2h_sec = 0.0
        self._h2d_sec = 0.0
        self._quant_sec = 0.0
        self._dequant_sec = 0.0
        self._alloc_sec = 0.0
        self._n_allocs = 0
        self._cpu_q = {}      # layer_idx -> pinned int8 CPU tensor [B, H, max_seq, D]
        self._cpu_scale = {}  # layer_idx -> pinned fp16 CPU tensor [B, H, max_seq, 1]
        self._cur_seq = {}

    def _ensure_buffer(self, layer_idx, v_gpu):
        if layer_idx in self._cpu_q:
            return
        t0 = time.time()
        B, H, _, D = v_gpu.shape
        self._cpu_q[layer_idx] = torch.empty(
            (B, H, self.max_seq_len, D), dtype=torch.int8,
            device="cpu", pin_memory=True,
        )
        self._cpu_scale[layer_idx] = torch.empty(
            (B, H, self.max_seq_len, 1), dtype=torch.float16,
            device="cpu", pin_memory=True,
        )
        self._alloc_sec += time.time() - t0
        self._n_allocs += 1
        total = sum(t.element_size() * t.numel()
                    for t in list(self._cpu_q.values()) + list(self._cpu_scale.values()))
        if total > self._cpu_bytes_peak:
            self._cpu_bytes_peak = total

    def _spill(self, layer_idx, v_gpu):
        self._ensure_buffer(layer_idx, v_gpu)
        v = v_gpu.detach().contiguous()
        seq = v.shape[2]
        # Quantize on GPU
        t_q = time.time()
        q, scale = quantize_per_token(v)
        torch.cuda.synchronize()
        self._quant_sec += time.time() - t_q
        # D2H copy (halved volume: int8 instead of fp16)
        t_d = time.time()
        self._cpu_q[layer_idx][:, :, :seq, :].copy_(q, non_blocking=False)
        self._cpu_scale[layer_idx][:, :, :seq, :].copy_(scale, non_blocking=False)
        self._d2h_sec += time.time() - t_d
        self._cur_seq[layer_idx] = seq

    def _reload(self, layer_idx, device):
        seq = self._cur_seq[layer_idx]
        # H2D copy
        t_h = time.time()
        q_slice = self._cpu_q[layer_idx][:, :, :seq, :].contiguous().to(device, non_blocking=False)
        scale_slice = self._cpu_scale[layer_idx][:, :, :seq, :].contiguous().to(device, non_blocking=False)
        self._h2d_sec += time.time() - t_h
        # Dequantize on GPU
        t_dq = time.time()
        v_fp = dequantize_per_token(q_slice, scale_slice)
        torch.cuda.synchronize()
        self._dequant_sec += time.time() - t_dq
        return v_fp

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        target_device = key_states.device
        if layer_idx in self._cur_seq and layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if hasattr(layer, 'values') and layer.values.numel() == 0:
                layer.values = self._reload(layer_idx, target_device)

        k_out, v_out = super().update(key_states, value_states, layer_idx, *args, **kwargs)

        layer = self.layers[layer_idx]
        v_live = layer.values
        self._spill(layer_idx, v_live)
        layer.values = torch.empty(0, device=v_live.device, dtype=v_live.dtype)
        return k_out, v_out

def logits_hash(tensor):
    t = tensor.detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]

def compare_logits(baseline_logits_cpu, xhbm_logits_cpu):
    """Return (max_abs_diff, l2_distance, argmax_match)."""
    b = baseline_logits_cpu.to(torch.float32)
    x = xhbm_logits_cpu.to(torch.float32)
    diff = (b - x).abs()
    max_abs = float(diff.max())
    l2 = float(((b - x) ** 2).sum().sqrt())
    argmax_match = bool(b.argmax(dim=-1).eq(x.argmax(dim=-1)).all())
    return max_abs, l2, argmax_match

def run(mode, ctx_prefill, batch, n_decode, repeat_idx=0, seed=0, compare_to_baseline=False):
    torch.manual_seed(seed)
    print(f"\n======== mode={mode}  prefill={ctx_prefill}  batch={batch}  decode={n_decode}  r={repeat_idx} ========", flush=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); gc.collect()

    cpu_ram_pre = cpu_ram_used_mib()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa",
    )
    model.eval()
    load_sec = time.time() - t0

    g = torch.Generator(device="cuda:0").manual_seed(seed + ctx_prefill + batch)
    input_ids = torch.randint(0, tok.vocab_size, (batch, ctx_prefill), device="cuda:0", generator=g)

    torch.cuda.reset_peak_memory_stats()
    hbm_pre = nvsmi_mib()
    max_seq = ctx_prefill + n_decode
    cache = XHBMCacheV4(max_seq_len=max_seq) if mode == "xhbm" else DynamicCache()

    from torch.nn.attention import sdpa_kernel, SDPBackend
    t_pf = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    prefill_sec = time.time() - t_pf
    prefill_peak = torch.cuda.max_memory_allocated() / (1024**2)
    prefill_logits = out.logits[:, -1, :].detach().cpu()
    prefill_logits_h = logits_hash(out.logits[:, -1, :])

    # IMPORTANT: baseline greedy derivation vs xhbm greedy derivation
    # We want to measure how well xhbm's sequence matches baseline's sequence.
    # Strategy: at each step, feed next_tok derived from CURRENT logits.
    # For argmax comparison, we also record all step logits for later analysis.

    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.reset_peak_memory_stats()
    decode_lat = []
    step_logits_list = []   # logits at each decode step
    step_argmax_list = []   # argmax at each decode step
    t_dec = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        for step in range(n_decode):
            t_s = time.time()
            out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            decode_lat.append(time.time() - t_s)
            step_logits_list.append(out.logits[:, -1, :].detach().cpu())
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_argmax_list.append(next_tok.detach().cpu())
    total_decode_sec = time.time() - t_dec

    decode_peak = torch.cuda.max_memory_allocated() / (1024**2)
    post_alloc = torch.cuda.memory_allocated() / (1024**2)
    hbm_post = nvsmi_mib()
    torch.cuda.empty_cache()
    post_empty = torch.cuda.memory_allocated() / (1024**2)
    hbm_post_empty = nvsmi_mib()
    cpu_ram_post = cpu_ram_used_mib()

    lat_sorted = sorted(decode_lat)
    n = len(lat_sorted)

    # Save step argmaxes and prefill logits for later cross-mode comparison
    argmax_seq = torch.cat(step_argmax_list, dim=1).tolist()  # [batch, n_decode]

    result = {
        "mode": mode, "ctx_prefill": ctx_prefill, "batch": batch, "n_decode": n_decode,
        "repeat_idx": repeat_idx,
        "load_sec": round(load_sec, 2),
        "prefill_sec": round(prefill_sec, 3),
        "prefill_peak_MiB": round(prefill_peak, 1),
        "prefill_logits_sha256_16": prefill_logits_h,
        "decode_total_sec": round(total_decode_sec, 3),
        "decode_median_lat_ms": round(lat_sorted[n//2] * 1000, 2),
        "decode_p95_lat_ms": round(lat_sorted[min(int(n*0.95), n-1)] * 1000, 2),
        "decode_mean_lat_ms": round(sum(decode_lat)/n * 1000, 2),
        "decode_min_lat_ms": round(min(decode_lat) * 1000, 2),
        "decode_max_lat_ms": round(max(decode_lat) * 1000, 2),
        "decode_peak_MiB": round(decode_peak, 1),
        "decode_argmax_sequence": argmax_seq,
        "post_alloc_MiB": round(post_alloc, 1),
        "post_empty_alloc_MiB": round(post_empty, 1),
        "nvsmi_pre_MiB": hbm_pre,
        "nvsmi_post_MiB": hbm_post,
        "nvsmi_post_empty_MiB": hbm_post_empty,
        "cpu_ram_pre_MiB": cpu_ram_pre,
        "cpu_ram_post_MiB": cpu_ram_post,
    }
    if mode == "xhbm":
        result["cpu_v_peak_MiB"] = round(cache._cpu_bytes_peak / (1024**2), 1)
        result["d2h_sec"] = round(cache._d2h_sec, 3)
        result["h2d_sec"] = round(cache._h2d_sec, 3)
        result["quant_sec"] = round(cache._quant_sec, 3)
        result["dequant_sec"] = round(cache._dequant_sec, 3)
        result["alloc_sec"] = round(cache._alloc_sec, 3)
        result["n_allocs"] = cache._n_allocs
    print("RESULT_JSON=" + json.dumps(result))
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "xhbm"])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--decode", type=int, required=True)
    ap.add_argument("--repeat-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.mode, args.ctx, args.batch, args.decode, args.repeat_idx, args.seed)
