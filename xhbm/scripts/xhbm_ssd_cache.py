"""
xhbm_ssd_cache.py — Phase D core: SSD-backed KV cache for transformers 5.6.

D-4.4 changes (D-4.2/3 found pinned RAM quota cliff at large ctx):
  - SHARED pinned buffer at pool level (all 80 layers share)
  - Layer offload is sequential, so no race on buffer
  - Reduces pinned RAM by 80x: from 80*256MB = 20GB to ~256MB total
"""
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional
import threading

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class SSDKVPool:
    def __init__(self, ssd_dir: Path, num_workers: int = 4):
        self.ssd_dir = Path(ssd_dir)
        self.ssd_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="ssdkv")
        self.bytes_written = 0
        self.bytes_read = 0
        self.write_time = 0.0
        self.read_time = 0.0
        self.write_count = 0
        self.read_count = 0
        self.gpu_to_cpu_time = 0.0
        self.gpu_to_cpu_bytes = 0
        self.gpu_to_cpu_count = 0
        self.submit_overhead_time = 0.0
        self.prefetch_wait_time = 0.0
        self.prefetch_wait_count = 0
        # D-4.4: shared pinned buffer (all layers reuse)
        # Need 2 K-buffers + 2 V-buffers so write thread can use one while
        # next layer fills the other (double-buffering)
        self._shared_pin_lock = threading.Lock()
        self._k_pinned_a = None
        self._v_pinned_a = None
        self._k_pinned_b = None
        self._v_pinned_b = None
        self._next_buf_idx = 0  # 0 = a, 1 = b

    def get_pinned_pair(self, k_shape, k_dtype, v_shape, v_dtype):
        """Return (k_pinned, v_pinned) double-buffer pair, alternating a/b."""
        with self._shared_pin_lock:
            idx = self._next_buf_idx
            self._next_buf_idx ^= 1
        if idx == 0:
            need_k = (self._k_pinned_a is None
                      or self._k_pinned_a.shape != k_shape
                      or self._k_pinned_a.dtype != k_dtype)
            if need_k:
                self._k_pinned_a = torch.empty(k_shape, dtype=k_dtype, pin_memory=True)
            need_v = (self._v_pinned_a is None
                      or self._v_pinned_a.shape != v_shape
                      or self._v_pinned_a.dtype != v_dtype)
            if need_v:
                self._v_pinned_a = torch.empty(v_shape, dtype=v_dtype, pin_memory=True)
            return self._k_pinned_a, self._v_pinned_a
        else:
            need_k = (self._k_pinned_b is None
                      or self._k_pinned_b.shape != k_shape
                      or self._k_pinned_b.dtype != k_dtype)
            if need_k:
                self._k_pinned_b = torch.empty(k_shape, dtype=k_dtype, pin_memory=True)
            need_v = (self._v_pinned_b is None
                      or self._v_pinned_b.shape != v_shape
                      or self._v_pinned_b.dtype != v_dtype)
            if need_v:
                self._v_pinned_b = torch.empty(v_shape, dtype=v_dtype, pin_memory=True)
            return self._k_pinned_b, self._v_pinned_b

    def submit_write_torch(self, path: Path, tensor: torch.Tensor) -> Future:
        assert tensor.device.type == "cpu", f"expected CPU tensor, got {tensor.device}"
        nbytes = tensor.element_size() * tensor.numel()
        storage = tensor.untyped_storage()
        def _write():
            t0 = time.perf_counter()
            with open(path, "wb") as f:
                buf = (torch.empty(nbytes, dtype=torch.uint8)
                       .set_(storage, 0, (nbytes,)))
                f.write(buf.numpy().tobytes())
                f.flush()
            elapsed = time.perf_counter() - t0
            return (nbytes, elapsed)
        return self.executor.submit(_write)

    def submit_read_torch(self, path: Path, expected_bytes: int) -> Future:
        def _read():
            t0 = time.perf_counter()
            with open(path, "rb") as f:
                data = f.read(expected_bytes)
            elapsed = time.perf_counter() - t0
            return (data, elapsed)
        return self.executor.submit(_read)

    def record_write(self, nbytes, elapsed):
        self.bytes_written += nbytes
        self.write_time += elapsed
        self.write_count += 1

    def record_read(self, nbytes, elapsed):
        self.bytes_read += nbytes
        self.read_time += elapsed
        self.read_count += 1

    def record_gpu_to_cpu(self, nbytes, elapsed):
        self.gpu_to_cpu_time += elapsed
        self.gpu_to_cpu_bytes += nbytes
        self.gpu_to_cpu_count += 1

    def record_submit_overhead(self, elapsed):
        self.submit_overhead_time += elapsed

    def record_prefetch_wait(self, elapsed):
        self.prefetch_wait_time += elapsed
        self.prefetch_wait_count += 1

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def stats(self):
        gpu_to_cpu_GBps = (
            (self.gpu_to_cpu_bytes / max(self.gpu_to_cpu_time, 1e-9)) / (1024**3)
            if self.gpu_to_cpu_time > 0 else 0
        )
        return {
            "bytes_written_MiB": self.bytes_written / (1024**2),
            "bytes_read_MiB": self.bytes_read / (1024**2),
            "write_count": self.write_count,
            "read_count": self.read_count,
            "write_throughput_MBps": (self.bytes_written / max(self.write_time, 1e-9)) / (1024**2)
                if self.write_time > 0 else 0,
            "read_throughput_MBps": (self.bytes_read / max(self.read_time, 1e-9)) / (1024**2)
                if self.read_time > 0 else 0,
            "gpu_to_cpu_time_sec": self.gpu_to_cpu_time,
            "gpu_to_cpu_throughput_GBps": gpu_to_cpu_GBps,
            "gpu_to_cpu_count": self.gpu_to_cpu_count,
            "submit_overhead_time_sec": self.submit_overhead_time,
            "prefetch_wait_time_sec": self.prefetch_wait_time,
            "prefetch_wait_count": self.prefetch_wait_count,
        }


class SSDOffloadLayer(DynamicLayer):
    _pool: Optional[SSDKVPool] = None
    _layer_counter = 0

    def __init__(self):
        super().__init__()
        self._ssd_layer_idx = SSDOffloadLayer._layer_counter
        SSDOffloadLayer._layer_counter += 1
        self._k_path = None
        self._v_path = None
        self._k_meta = None
        self._v_meta = None
        self._write_futures = []

    def _ensure_paths(self):
        if self._k_path is None:
            pool = SSDOffloadLayer._pool
            self._k_path = pool.ssd_dir / f"layer_{self._ssd_layer_idx:03d}_K.bin"
            self._v_path = pool.ssd_dir / f"layer_{self._ssd_layer_idx:03d}_V.bin"

    def offload(self):
        if not self.is_initialized:
            return
        if self.keys.device.type != "cuda":
            return
        if self.keys.numel() == 0:
            return
        self._ensure_paths()
        pool = SSDOffloadLayer._pool
        self._k_meta = (tuple(self.keys.shape), self.keys.dtype, self.keys.device)
        self._v_meta = (tuple(self.values.shape), self.values.dtype, self.values.device)

        # D-4.4: shared pinned buffer (double-buffered to allow write thread
        # to drain while next layer fills the other buffer)
        k_pinned, v_pinned = pool.get_pinned_pair(
            self.keys.shape, self.keys.dtype,
            self.values.shape, self.values.dtype,
        )

        t_d2h = time.perf_counter()
        k_pinned.copy_(self.keys.detach(), non_blocking=True)
        v_pinned.copy_(self.values.detach(), non_blocking=True)
        torch.cuda.synchronize()
        d2h_elapsed = time.perf_counter() - t_d2h
        d2h_bytes = (k_pinned.element_size() * k_pinned.numel()
                     + v_pinned.element_size() * v_pinned.numel())
        pool.record_gpu_to_cpu(d2h_bytes, d2h_elapsed)

        t_submit = time.perf_counter()
        kf = pool.submit_write_torch(self._k_path, k_pinned)
        vf = pool.submit_write_torch(self._v_path, v_pinned)
        pool.record_submit_overhead(time.perf_counter() - t_submit)
        self._write_futures = [kf, vf]

        self.keys = torch.tensor([], dtype=self._k_meta[1], device=self._k_meta[2])
        self.values = torch.tensor([], dtype=self._v_meta[1], device=self._v_meta[2])

    def prefetch(self):
        if self._k_meta is None:
            return
        if self.keys.numel() > 0:
            return
        pool = SSDOffloadLayer._pool

        t_wait = time.perf_counter()
        for f in self._write_futures:
            try:
                nbytes, elapsed = f.result()
                pool.record_write(nbytes, elapsed)
            except Exception:
                pass
        pool.record_prefetch_wait(time.perf_counter() - t_wait)
        self._write_futures = []

        k_shape, k_dtype, k_device = self._k_meta
        v_shape, v_dtype, v_device = self._v_meta
        elem_size_k = torch.empty(0, dtype=k_dtype).element_size()
        elem_size_v = torch.empty(0, dtype=v_dtype).element_size()
        k_nbytes = elem_size_k
        for d in k_shape:
            k_nbytes *= d
        v_nbytes = elem_size_v
        for d in v_shape:
            v_nbytes *= d

        kf = pool.submit_read_torch(self._k_path, k_nbytes)
        vf = pool.submit_read_torch(self._v_path, v_nbytes)
        k_bytes, k_elapsed = kf.result()
        v_bytes, v_elapsed = vf.result()
        pool.record_read(len(k_bytes), k_elapsed)
        pool.record_read(len(v_bytes), v_elapsed)

        k_buf = torch.frombuffer(bytearray(k_bytes), dtype=torch.uint8)
        k_tensor = k_buf.view(k_dtype).reshape(k_shape)
        v_buf = torch.frombuffer(bytearray(v_bytes), dtype=torch.uint8)
        v_tensor = v_buf.view(v_dtype).reshape(v_shape)

        self.keys = k_tensor.to(k_device, non_blocking=True)
        self.values = v_tensor.to(v_device, non_blocking=True)


class SSDOffloadCache(DynamicCache):
    def __init__(self, ssd_dir: Path, config=None, num_workers: int = 4):
        SSDOffloadLayer._pool = SSDKVPool(Path(ssd_dir), num_workers=num_workers)
        SSDOffloadLayer._layer_counter = 0
        super().__init__(config=config, offloading=True, offload_only_non_sliding=False)
        new_layers = []
        for i, old_layer in enumerate(self.layers):
            new_layer = SSDOffloadLayer()
            if hasattr(old_layer, "is_sliding"):
                new_layer.is_sliding = old_layer.is_sliding
            new_layers.append(new_layer)
        self.layers = new_layers

    def get_pool_stats(self):
        return SSDOffloadLayer._pool.stats()

    def shutdown(self):
        SSDOffloadLayer._pool.shutdown()
