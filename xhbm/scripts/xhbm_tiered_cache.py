"""
xhbm_tiered_cache.py — Phase D D-5.1: 3-tier KV cache (GPU + RAM + SSD).

Design:
  - Tier 1 (GPU): active forward only. transformers native offloading=True
    handles GPU<->RAM automatically per forward.
  - Tier 2 (RAM): default idle state (where cache lives between forwards).
  - Tier 3 (SSD): cold conversations (idle 5+ min). We add demote/promote
    methods on top of transformers native offloading.

Reuses SSDKVPool from xhbm_ssd_cache.py (thread pool, pinned buffer, stats).

D-5.0 PCIe bottleneck finding (forward 73% in GPU<->CPU copy at 128K) is the
core justification: hot path SSD round-trip hits PCIe limit, but cold tiering
moves rare cold->hot promotions, leaving PCIe free for active forwards.
"""
import time
from pathlib import Path
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from xhbm_ssd_cache import SSDKVPool


class TieredKVCache(DynamicCache):
    """
    3-tier KV cache for multi-conv serving.

    Default behavior is identical to DynamicCache(offloading=True): GPU<->RAM
    per forward, cache lives in RAM between forwards.

    Adds:
      - demote_to_ssd(): RAM -> SSD. Used when conv idle 5+ min.
      - promote_from_ssd(): SSD -> RAM. Used when cold conv receives new request.
      - touch(): mark as active (resets idle timer).
      - idle_seconds(), get_tier_state(), get_tier_stats(): observability.
    """

    def __init__(self, ssd_dir: Path, config, num_workers: int = 4, conv_id: str = "conv_0"):
        super().__init__(config=config, offloading=True, offload_only_non_sliding=False)
        self.ssd_dir = Path(ssd_dir)
        self.conv_id = conv_id
        self.num_workers = num_workers
        self._ssd_pool: Optional[SSDKVPool] = None
        # tier states: "gpu" (active forward), "ram" (in CPU pinned RAM, fast promote),
        # "ssd" (on disk, slow promote)
        self._ssd_state = "gpu"  # initial state during prefill
        self._ram_buffers = None  # list of (k_cpu_tensor, v_cpu_tensor) per layer when ram-resident
        self._ram_meta = None  # list of (k_shape, k_dtype, v_shape, v_dtype) per layer
        self.last_active_ts = time.monotonic()
        # D-7.6 stats for RAM tier
        self.ram_demote_count = 0
        self.ram_demote_time_sec = 0.0
        self.ram_promote_count = 0
        self.ram_promote_time_sec = 0.0
        self.ram_demote_bytes = 0
        self.ram_promote_bytes = 0

        self.demote_count = 0
        self.promote_count = 0
        self.demote_time_sec = 0.0
        self.promote_time_sec = 0.0
        self.demote_bytes = 0
        self.promote_bytes = 0

    def _ensure_ssd_pool(self):
        if self._ssd_pool is None:
            ssd_path = self.ssd_dir / self.conv_id
            ssd_path.mkdir(parents=True, exist_ok=True)
            self._ssd_pool = SSDKVPool(ssd_path, num_workers=self.num_workers)

    def touch(self):
        self.last_active_ts = time.monotonic()

    def idle_seconds(self):
        return time.monotonic() - self.last_active_ts

    def get_tier_state(self):
        return self._ssd_state

    def demote_to_ram(self):
        """GPU -> CPU RAM. For 30s-5min idle window. Faster promote than SSD."""
        if self._ssd_state == "ram_resident":
            return
        if self._ssd_state == "ssd":
            return  # already on SSD
        t0 = time.perf_counter()
        bytes_moved = 0
        ram_buffers = []
        ram_meta = []
        for layer in self.layers:
            if not getattr(layer, "is_initialized", False):
                ram_buffers.append(None)
                ram_meta.append(None)
                continue
            k = layer.keys
            v = layer.values
            if k.numel() == 0:
                ram_buffers.append(None)
                ram_meta.append(None)
                continue
            k_meta = (tuple(k.shape), k.dtype)
            v_meta = (tuple(v.shape), v.dtype)
            # Move to CPU pinned (faster GPU<->CPU later)
            # D-7.6.4: pin_memory=False to avoid OOM at 8+ conv (was 80GB pinned)
            # Trade-off: promote latency 0.2s -> ~1-3s estimated
            k_cpu = torch.empty(k.shape, dtype=k.dtype, pin_memory=False)
            v_cpu = torch.empty(v.shape, dtype=v.dtype, pin_memory=False)
            k_cpu.copy_(k.detach(), non_blocking=False)
            v_cpu.copy_(v.detach(), non_blocking=False)
            ram_buffers.append((k_cpu, v_cpu))
            ram_meta.append((k_meta, v_meta))
            bytes_moved += k.element_size() * k.numel()
            bytes_moved += v.element_size() * v.numel()
            # Replace GPU tensor with empty placeholder on GPU
            layer.keys = torch.tensor([], dtype=k.dtype, device="cuda:0")
            layer.values = torch.tensor([], dtype=v.dtype, device="cuda:0")
        torch.cuda.synchronize()
        self._ram_buffers = ram_buffers
        self._ram_meta = ram_meta
        self._ssd_state = "ram_resident"
        self.ram_demote_count += 1
        self.ram_demote_time_sec += time.perf_counter() - t0
        self.ram_demote_bytes += bytes_moved

    def promote_from_ram(self):
        """CPU RAM -> GPU. Fast promote (no SSD I/O)."""
        if self._ssd_state == "gpu":
            return
        if self._ssd_state != "ram_resident":
            return
        t0 = time.perf_counter()
        bytes_moved = 0
        for i, layer in enumerate(self.layers):
            if self._ram_buffers[i] is None:
                continue
            k_cpu, v_cpu = self._ram_buffers[i]
            layer.keys = k_cpu.to("cuda:0", non_blocking=True)
            layer.values = v_cpu.to("cuda:0", non_blocking=True)
            bytes_moved += k_cpu.element_size() * k_cpu.numel()
            bytes_moved += v_cpu.element_size() * v_cpu.numel()
        torch.cuda.synchronize()
        self._ram_buffers = None
        self._ram_meta = None
        self._ssd_state = "gpu"
        self.ram_promote_count += 1
        self.ram_promote_time_sec += time.perf_counter() - t0
        self.ram_promote_bytes += bytes_moved
        self.touch()

    def demote_to_ssd(self):
        if self._ssd_state == "ssd":
            return
        # If currently in RAM tier, take from RAM buffers; otherwise from GPU
        if self._ssd_state == "ram_resident":
            # Restore from RAM to GPU briefly, then SSD path takes over
            # Simpler: write RAM buffers directly to SSD without GPU round-trip
            self._demote_from_ram_to_ssd()
            return
        self._ensure_ssd_pool()
        t0 = time.perf_counter()
        bytes_demoted = 0

        futures = []
        for i, layer in enumerate(self.layers):
            if not getattr(layer, "is_initialized", False):
                continue
            keys = layer.keys
            values = layer.values
            if keys.numel() == 0:
                continue

            layer._tier_k_meta = (tuple(keys.shape), keys.dtype)
            layer._tier_v_meta = (tuple(values.shape), values.dtype)

            if keys.device.type != "cpu":
                keys = keys.detach().cpu()
                values = values.detach().cpu()
            else:
                keys = keys.detach()
                values = values.detach()

            k_path = self._ssd_pool.ssd_dir / f"layer_{i:03d}_K.bin"
            v_path = self._ssd_pool.ssd_dir / f"layer_{i:03d}_V.bin"
            layer._tier_k_path = k_path
            layer._tier_v_path = v_path

            kf = self._ssd_pool.submit_write_torch(k_path, keys)
            vf = self._ssd_pool.submit_write_torch(v_path, values)
            futures.append((i, kf, vf))

            bytes_demoted += keys.element_size() * keys.numel()
            bytes_demoted += values.element_size() * values.numel()

            # Empty placeholder on GPU (device must match next forward's key_states)
            layer.keys = torch.tensor([], dtype=layer._tier_k_meta[1], device="cuda:0")
            layer.values = torch.tensor([], dtype=layer._tier_v_meta[1], device="cuda:0")

        for i, kf, vf in futures:
            try:
                k_nbytes, k_elapsed = kf.result()
                self._ssd_pool.record_write(k_nbytes, k_elapsed)
                v_nbytes, v_elapsed = vf.result()
                self._ssd_pool.record_write(v_nbytes, v_elapsed)
            except Exception as e:
                print(f"[TieredKVCache:{self.conv_id}] demote layer {i} failed: {e}")

        self._ssd_state = "ssd"
        self.demote_count += 1
        self.demote_time_sec += time.perf_counter() - t0
        self.demote_bytes += bytes_demoted

    def _demote_from_ram_to_ssd(self):
        """Direct RAM -> SSD path, bypassing GPU. Used when ram_resident times out."""
        self._ensure_ssd_pool()
        t0 = time.perf_counter()
        bytes_moved = 0
        futures = []
        for i, layer in enumerate(self.layers):
            if self._ram_buffers[i] is None:
                continue
            k_cpu, v_cpu = self._ram_buffers[i]
            k_meta, v_meta = self._ram_meta[i]
            layer._tier_k_meta = k_meta
            layer._tier_v_meta = v_meta
            k_path = self._ssd_pool.ssd_dir / f"layer_{i:03d}_K.bin"
            v_path = self._ssd_pool.ssd_dir / f"layer_{i:03d}_V.bin"
            layer._tier_k_path = k_path
            layer._tier_v_path = v_path
            kf = self._ssd_pool.submit_write_torch(k_path, k_cpu)
            vf = self._ssd_pool.submit_write_torch(v_path, v_cpu)
            futures.append((i, kf, vf))
            bytes_moved += k_cpu.element_size() * k_cpu.numel()
            bytes_moved += v_cpu.element_size() * v_cpu.numel()
        for i, kf, vf in futures:
            try:
                k_nb, k_el = kf.result()
                self._ssd_pool.record_write(k_nb, k_el)
                v_nb, v_el = vf.result()
                self._ssd_pool.record_write(v_nb, v_el)
            except Exception as e:
                print(f"[TieredKVCache:{self.conv_id}] ram->ssd layer {i} failed: {e}")
        self._ram_buffers = None
        self._ram_meta = None
        self._ssd_state = "ssd"
        self.demote_count += 1
        self.demote_time_sec += time.perf_counter() - t0
        self.demote_bytes += bytes_moved

    def promote_from_ssd(self):
        if self._ssd_state == "ram":
            return
        if self._ssd_pool is None:
            return
        t0 = time.perf_counter()
        bytes_promoted = 0

        futures = []
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, "_tier_k_path"):
                continue
            k_meta = layer._tier_k_meta
            v_meta = layer._tier_v_meta

            elem_k = torch.empty(0, dtype=k_meta[1]).element_size()
            k_nbytes = elem_k
            for d in k_meta[0]:
                k_nbytes *= d
            elem_v = torch.empty(0, dtype=v_meta[1]).element_size()
            v_nbytes = elem_v
            for d in v_meta[0]:
                v_nbytes *= d

            kf = self._ssd_pool.submit_read_torch(layer._tier_k_path, k_nbytes)
            vf = self._ssd_pool.submit_read_torch(layer._tier_v_path, v_nbytes)
            futures.append((i, kf, vf, k_meta, v_meta))

        for i, kf, vf, k_meta, v_meta in futures:
            try:
                k_bytes, k_elapsed = kf.result()
                v_bytes, v_elapsed = vf.result()
                self._ssd_pool.record_read(len(k_bytes), k_elapsed)
                self._ssd_pool.record_read(len(v_bytes), v_elapsed)

                k_buf = torch.frombuffer(bytearray(k_bytes), dtype=torch.uint8)
                k_tensor = k_buf.view(k_meta[1]).reshape(k_meta[0])
                v_buf = torch.frombuffer(bytearray(v_bytes), dtype=torch.uint8)
                v_tensor = v_buf.view(v_meta[1]).reshape(v_meta[0])

                # Move directly to GPU (not CPU) so forward sees consistent device.
                # Bypasses transformers native offloading internal state tracking.
                self.layers[i].keys = k_tensor.to("cuda:0", non_blocking=True)
                self.layers[i].values = v_tensor.to("cuda:0", non_blocking=True)

                bytes_promoted += len(k_bytes) + len(v_bytes)
            except Exception as e:
                print(f"[TieredKVCache:{self.conv_id}] promote layer {i} failed: {e}")

        self._ssd_state = "ram"
        self.promote_count += 1
        self.promote_time_sec += time.perf_counter() - t0
        self.promote_bytes += bytes_promoted
        self.touch()

    def get_tier_stats(self):
        s = {
            "conv_id": self.conv_id,
            "tier_state": self._ssd_state,
            "ssd_demote_count": self.demote_count,
            "ssd_promote_count": self.promote_count,
            "ssd_demote_time_sec": round(self.demote_time_sec, 3),
            "ssd_promote_time_sec": round(self.promote_time_sec, 3),
            "ssd_demote_MiB": round(self.demote_bytes / (1024**2), 1),
            "ssd_promote_MiB": round(self.promote_bytes / (1024**2), 1),
            "ram_demote_count": self.ram_demote_count,
            "ram_promote_count": self.ram_promote_count,
            "ram_demote_time_sec": round(self.ram_demote_time_sec, 3),
            "ram_promote_time_sec": round(self.ram_promote_time_sec, 3),
            "ram_demote_MiB": round(self.ram_demote_bytes / (1024**2), 1),
            "ram_promote_MiB": round(self.ram_promote_bytes / (1024**2), 1),
            "idle_seconds": round(self.idle_seconds(), 1),
        }
        if self._ssd_pool is not None:
            s["ssd_pool"] = self._ssd_pool.stats()
        return s

    def get_gpu_resident_kv_bytes(self):
        """Sum of K and V bytes residing on GPU across all initialized layers."""
        total = 0
        for layer in self.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            try:
                k = layer.keys
                if k.numel() > 0 and k.device.type == "cuda":
                    total += k.element_size() * k.numel()
                v = layer.values
                if v.numel() > 0 and v.device.type == "cuda":
                    total += v.element_size() * v.numel()
            except Exception:
                pass
        return total

    def shutdown(self):
        if self._ssd_pool is not None:
            self._ssd_pool.shutdown()
