"""
multi_conv_simulator_v2.py — D-7.5 production-grade multi-conv simulator.

v2 over v1:
  - TimeSeriesSampler thread (1 Hz: active count + GPU/SSD KV bytes)
  - Promote/TTFT P50/P95/P99 percentiles
  - Total PCIe time aggregation
  - Real-text message pool (with random fallback)
  - Graceful shutdown buffer (last 30s no new messages)
  - Decode loop with P50/P95 step latency
"""
import random, threading, time
from dataclasses import dataclass, field
from typing import Optional
import torch
from xhbm_tiered_cache import TieredKVCache

IDLE_THRESHOLD_SEC = 30.0       # GPU -> RAM
RAM_TO_SSD_THRESHOLD_SEC = 300.0  # RAM -> SSD (5 min)
GRACEFUL_BUFFER_SEC = 30.0


@dataclass
class ConvSchedule:
    conv_id: str
    cache: TieredKVCache
    prefill_token_count: int
    is_frequent: bool
    next_message_at: float = 0.0
    total_messages: int = 0
    rng: Optional[random.Random] = None
    message_token_pool: Optional[list] = None
    pool_offset: int = 0

    def sample_idle_delay(self):
        if self.is_frequent:
            return self.rng.uniform(5.0, 60.0)
        else:
            return self.rng.uniform(60.0, 300.0)


@dataclass
class SimEvent:
    timestamp: float
    event_type: str
    conv_id: str
    duration_sec: float = 0.0
    extra: dict = field(default_factory=dict)


class TimeSeriesSampler(threading.Thread):
    def __init__(self, conversations, sample_interval_sec=1.0):
        super().__init__(daemon=True)
        self.conversations = conversations
        self.sample_interval = sample_interval_sec
        self.samples = []
        self._stop_event = threading.Event()
        self.start_time = None

    def run(self):
        if self.start_time is None:
            self.start_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            t_rel = now - self.start_time
            active = 0
            gpu_kv = 0
            ssd_kv = 0
            for c in self.conversations:
                state = c.cache.get_tier_state()
                if state == "ssd":
                    ssd_kv += c.cache.demote_bytes
                else:
                    try:
                        gpu_kv += c.cache.get_gpu_resident_kv_bytes()
                    except Exception:
                        pass
                if c.cache.idle_seconds() < IDLE_THRESHOLD_SEC:
                    active += 1
            self.samples.append({
                "t_sec": round(t_rel, 2),
                "active_count": active,
                "gpu_kv_MiB": round(gpu_kv / (1024**2), 1),
                "ssd_kv_MiB_approx": round(ssd_kv / (1024**2), 1),
            })
            self._stop_event.wait(self.sample_interval)

    def stop(self):
        self._stop_event.set()


class MultiConvSimulator:
    def __init__(self, model, conversations, duration_sec,
                 new_message_token_range=(32, 128),
                 decode_n_tokens=64, seed=42, sample_interval_sec=1.0):
        self.model = model
        self.conversations = conversations
        self.duration_sec = duration_sec
        self.new_message_token_range = new_message_token_range
        self.decode_n_tokens = decode_n_tokens
        self.events = []
        self.start_time = 0.0
        self.rng = random.Random(seed)
        for i, conv in enumerate(conversations):
            if conv.rng is None:
                conv.rng = random.Random(seed + i + 1)
        self.sampler = TimeSeriesSampler(conversations, sample_interval_sec)

    def _t(self):
        return time.monotonic() - self.start_time

    def _record(self, event_type, conv_id, duration=0.0, **extra):
        self.events.append(SimEvent(self._t(), event_type, conv_id, duration, extra))

    def _check_demotes(self):
        for conv in self.conversations:
            state = conv.cache.get_tier_state()
            idle = conv.cache.idle_seconds()
            # GPU -> RAM at 30s idle
            if state == "gpu" and idle > IDLE_THRESHOLD_SEC:
                t0 = time.perf_counter()
                conv.cache.demote_to_ram()
                self._record("demote_ram", conv.conv_id,
                             duration=time.perf_counter() - t0)
            # RAM -> SSD at 5min idle
            elif state == "ram_resident" and idle > RAM_TO_SSD_THRESHOLD_SEC:
                t0 = time.perf_counter()
                conv.cache.demote_to_ssd()
                self._record("demote_ssd", conv.conv_id,
                             duration=time.perf_counter() - t0)

    def _process_due_message(self, conv):
        state = conv.cache.get_tier_state()
        if state == "ssd":
            t0 = time.perf_counter()
            conv.cache.promote_from_ssd()
            self._record("promote_ssd", conv.conv_id,
                         duration=time.perf_counter() - t0,
                         promote_MiB=round(conv.cache.promote_bytes / (1024**2), 1))
        elif state == "ram_resident":
            t0 = time.perf_counter()
            conv.cache.promote_from_ram()
            self._record("promote_ram", conv.conv_id,
                         duration=time.perf_counter() - t0,
                         promote_MiB=round(conv.cache.ram_promote_bytes / (1024**2), 1))
        # else state == "gpu": no promote needed

        n_tok = self.rng.randint(*self.new_message_token_range)
        if (conv.message_token_pool is not None
                and conv.pool_offset + n_tok <= len(conv.message_token_pool)):
            new_tokens = conv.message_token_pool[conv.pool_offset:conv.pool_offset + n_tok]
            conv.pool_offset += n_tok
            tokens_source = "real"
        else:
            new_tokens = [self.rng.randint(0, self.model.config.vocab_size - 1)
                          for _ in range(n_tok)]
            tokens_source = "random"
        input_ids = torch.tensor([new_tokens], dtype=torch.long).cuda()

        kv_before = -1
        try:
            kv_before = conv.cache.layers[0].keys.shape[-2]
        except Exception:
            pass

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, past_key_values=conv.cache, use_cache=True)
        torch.cuda.synchronize()
        prefill_dur = time.perf_counter() - t0
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        first_tok = int(next_tok[0, 0].item())
        last_tok = first_tok
        del out

        decode_step_times = []
        for _ in range(self.decode_n_tokens):
            ts = time.perf_counter()
            with torch.inference_mode():
                out = self.model(input_ids=next_tok, past_key_values=conv.cache, use_cache=True)
            torch.cuda.synchronize()
            decode_step_times.append(time.perf_counter() - ts)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last_tok = int(next_tok[0, 0].item())
            del out

        del input_ids, next_tok
        kv_after = -1
        try:
            kv_after = conv.cache.layers[0].keys.shape[-2]
        except Exception:
            pass
        decode_total = sum(decode_step_times)

        self._record("forward", conv.conv_id,
                     duration=prefill_dur + decode_total,
                     new_tokens=n_tok, tokens_source=tokens_source,
                     decode_steps=len(decode_step_times),
                     ttft_sec=round(prefill_dur, 4),
                     decode_total_sec=round(decode_total, 4),
                     decode_p50_step_sec=round(sorted(decode_step_times)[len(decode_step_times)//2], 4) if decode_step_times else 0.0,
                     first_decode_token=first_tok, last_decode_token=last_tok,
                     kv_len_before=kv_before, kv_len_after=kv_after,
                     total_messages_after=conv.total_messages + 1)

        conv.cache.touch()
        conv.total_messages += 1
        conv.next_message_at = time.monotonic() + conv.sample_idle_delay()

    def run(self):
        self.start_time = time.monotonic()
        self.sampler.start_time = self.start_time
        self.sampler.start()
        for conv in self.conversations:
            conv.next_message_at = self.start_time + self.rng.uniform(0, 5)
        end = self.start_time + self.duration_sec
        cutoff = end - GRACEFUL_BUFFER_SEC
        try:
            while time.monotonic() < end:
                now = time.monotonic()
                self._check_demotes()
                if now >= cutoff:
                    time.sleep(min(end - now, 1.0))
                    continue
                due = [c for c in self.conversations if c.next_message_at <= now]
                if not due:
                    nd = min(c.next_message_at for c in self.conversations)
                    time.sleep(max(0.05, min(nd - now, 1.0)))
                    continue
                conv = min(due, key=lambda c: c.next_message_at)
                self._process_due_message(conv)
        finally:
            self.sampler.stop()
            self.sampler.join(timeout=2.0)
        return self.events

    def _pct(self, vals, p):
        if not vals:
            return 0.0
        sv = sorted(vals)
        return sv[max(0, min(len(sv) - 1, int(len(sv) * p / 100)))]

    def summary(self):
        fwds = [e for e in self.events if e.event_type == "forward"]
        promotes_ssd = [e for e in self.events if e.event_type == "promote_ssd"]
        promotes_ram = [e for e in self.events if e.event_type == "promote_ram"]
        demotes_ram = [e for e in self.events if e.event_type == "demote_ram"]
        demotes_ssd_event = [e for e in self.events if e.event_type == "demote_ssd"]
        # Unified for legacy compatibility
        promotes = promotes_ssd + promotes_ram
        demotes = demotes_ram + demotes_ssd_event
        p_durs = [e.duration_sec for e in promotes]
        f_durs = [e.duration_sec for e in fwds]
        ttfts = [e.extra.get("ttft_sec", 0) for e in fwds]
        d_steps = [e.extra.get("decode_p50_step_sec", 0) for e in fwds]

        total_pcie = 0.0
        total_promote_b = 0
        total_demote_b = 0
        for c in self.conversations:
            stats = c.cache.get_tier_stats()
            if stats.get("ssd_pool"):
                total_pcie += stats["ssd_pool"].get("gpu_to_cpu_time_sec", 0)
            total_promote_b += c.cache.promote_bytes
            total_demote_b += c.cache.demote_bytes

        return {
            "duration_sec": self.duration_sec,
            "total_events": len(self.events),
            "forward_count": len(fwds),
            "demote_count": len(demotes),
            "promote_count": len(promotes),
            "promote_ssd_count": len(promotes_ssd),
            "promote_ram_count": len(promotes_ram),
            "demote_ram_count": len(demotes_ram),
            "demote_ssd_count": len(demotes_ssd_event),
            "promote_ssd_p50_sec": round(self._pct([e.duration_sec for e in promotes_ssd], 50), 3),
            "promote_ssd_p95_sec": round(self._pct([e.duration_sec for e in promotes_ssd], 95), 3),
            "promote_ram_p50_sec": round(self._pct([e.duration_sec for e in promotes_ram], 50), 4),
            "promote_ram_p95_sec": round(self._pct([e.duration_sec for e in promotes_ram], 95), 4),
            "promote_latency_p50_sec": round(self._pct(p_durs, 50), 3),
            "promote_latency_p95_sec": round(self._pct(p_durs, 95), 3),
            "promote_latency_p99_sec": round(self._pct(p_durs, 99), 3),
            "ttft_p50_sec": round(self._pct(ttfts, 50), 3),
            "ttft_p95_sec": round(self._pct(ttfts, 95), 3),
            "ttft_p99_sec": round(self._pct(ttfts, 99), 3),
            "decode_p50_step_sec": round(self._pct(d_steps, 50), 4),
            "decode_p95_step_sec": round(self._pct(d_steps, 95), 4),
            "forward_p50_sec": round(self._pct(f_durs, 50), 3),
            "forward_p95_sec": round(self._pct(f_durs, 95), 3),
            "aggregate_msg_per_sec": (
                round(len(fwds) / self.duration_sec, 4) if self.duration_sec > 0 else 0.0
            ),
            "total_pcie_sec": round(total_pcie, 2),
            "total_promote_GiB": round(total_promote_b / (1024**3), 2),
            "total_demote_GiB": round(total_demote_b / (1024**3), 2),
            "time_series_samples": len(self.sampler.samples),
            "time_series": self.sampler.samples,
            "per_conv": [
                {
                    "conv_id": c.conv_id, "is_frequent": c.is_frequent,
                    "prefill_tokens": c.prefill_token_count,
                    "total_messages": c.total_messages,
                    "tier_stats": c.cache.get_tier_stats(),
                } for c in self.conversations
            ],
        }
