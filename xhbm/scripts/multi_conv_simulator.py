"""
multi_conv_simulator.py — Phase D D-5.2.2.

Event-loop multi-conversation simulator for D-5 cold KV tiering measurement.

Models N simultaneous conversations sharing one model on one GPU.
Each conv has its own TieredKVCache. Conv behavior:
  - Initial state: prefilled to some ctx (set up by caller before run())
  - Schedule: random delay between messages, drawn from {frequent, rare}
  - Demote: if cache idle > IDLE_THRESHOLD_SEC -> RAM->SSD
  - Promote: when next message due and cache in SSD -> SSD->RAM
  - Forward: model(new_tokens, past_key_values=cache)

Distribution: power-law approximation. Default 7 frequent + 1 rare.
"""
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from xhbm_tiered_cache import TieredKVCache


IDLE_THRESHOLD_SEC = 30.0


@dataclass
class ConvSchedule:
    conv_id: str
    cache: TieredKVCache
    prefill_token_count: int
    is_frequent: bool
    next_message_at: float = 0.0
    total_messages: int = 0
    rng: Optional[random.Random] = None
    # D-7.4: real-text message pool. List of token IDs to use for new messages.
    # Each new message consumes new_tok_count tokens from the head of this list.
    # If None or exhausted, falls back to random tokens.
    message_token_pool: Optional[list] = None
    pool_offset: int = 0

    def sample_idle_delay(self) -> float:
        if self.is_frequent:
            return self.rng.uniform(5.0, 60.0)
        else:
            return self.rng.uniform(60.0, 300.0)


@dataclass
class SimEvent:
    timestamp: float
    event_type: str  # 'demote' | 'promote' | 'forward'
    conv_id: str
    duration_sec: float = 0.0
    extra: dict = field(default_factory=dict)


class MultiConvSimulator:
    def __init__(
        self,
        model,
        conversations: list,
        duration_sec: float,
        new_message_token_range: tuple = (32, 128),
        decode_n_tokens: int = 64,
        seed: int = 42,
    ):
        self.model = model
        self.conversations = conversations
        self.duration_sec = duration_sec
        self.new_message_token_range = new_message_token_range
        self.decode_n_tokens = decode_n_tokens
        self.events: list = []
        self.start_time: float = 0.0
        self.rng = random.Random(seed)
        # Per-conv RNG for reproducible idle delays
        for i, conv in enumerate(conversations):
            if conv.rng is None:
                conv.rng = random.Random(seed + i + 1)

    def _t(self) -> float:
        return time.monotonic() - self.start_time

    def _record(self, event_type, conv_id, duration=0.0, **extra):
        self.events.append(SimEvent(self._t(), event_type, conv_id, duration, extra))

    def _check_demotes(self):
        for conv in self.conversations:
            if (conv.cache.get_tier_state() == "ram"
                    and conv.cache.idle_seconds() > IDLE_THRESHOLD_SEC):
                t0 = time.perf_counter()
                conv.cache.demote_to_ssd()
                dur = time.perf_counter() - t0
                self._record("demote", conv.conv_id, duration=dur)

    def _process_due_message(self, conv):
        # Promote if cold
        if conv.cache.get_tier_state() == "ssd":
            t0 = time.perf_counter()
            conv.cache.promote_from_ssd()
            promote_dur = time.perf_counter() - t0
            self._record("promote", conv.conv_id, duration=promote_dur,
                         promote_MiB=round(conv.cache.promote_bytes / (1024**2), 1))

        # Forward (prefill new message tokens)
        new_tok_count = self.rng.randint(*self.new_message_token_range)
        # D-7.4: use real text from pool if available, else random fallback
        if (conv.message_token_pool is not None
                and conv.pool_offset + new_tok_count <= len(conv.message_token_pool)):
            new_tokens = conv.message_token_pool[conv.pool_offset:conv.pool_offset + new_tok_count]
            conv.pool_offset += new_tok_count
            tokens_source = "real"
        else:
            new_tokens = [self.rng.randint(0, self.model.config.vocab_size - 1)
                          for _ in range(new_tok_count)]
            tokens_source = "random"
        input_ids = torch.tensor([new_tokens], dtype=torch.long).cuda()

        # Track KV size before/after for instrumentation
        kv_len_before = 0
        if len(conv.cache.layers) > 0 and getattr(conv.cache.layers[0], "is_initialized", False):
            try:
                kv_len_before = conv.cache.layers[0].keys.shape[-2]
            except Exception:
                kv_len_before = -1

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=input_ids,
                             past_key_values=conv.cache,
                             use_cache=True)
        torch.cuda.synchronize()
        prefill_msg_dur = time.perf_counter() - t0
        next_token_logits = out.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        del out

        # Decode loop
        decode_steps_done = 0
        ttft_sec = prefill_msg_dur  # First-token time = prefill time for new message
        decode_step_times = []
        first_decode_token_id = int(next_token[0, 0].item())
        last_decode_token_id = first_decode_token_id

        decode_target = self.decode_n_tokens
        for step in range(decode_target):
            t_step = time.perf_counter()
            with torch.inference_mode():
                out = self.model(input_ids=next_token,
                                 past_key_values=conv.cache,
                                 use_cache=True)
            torch.cuda.synchronize()
            step_dur = time.perf_counter() - t_step
            decode_step_times.append(step_dur)
            next_token_logits = out.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            last_decode_token_id = int(next_token[0, 0].item())
            decode_steps_done += 1
            del out

        del input_ids, next_token

        kv_len_after = -1
        if len(conv.cache.layers) > 0:
            try:
                kv_len_after = conv.cache.layers[0].keys.shape[-2]
            except Exception:
                kv_len_after = -1

        decode_total_sec = sum(decode_step_times)
        decode_p50 = sorted(decode_step_times)[len(decode_step_times)//2] if decode_step_times else 0.0

        self._record("forward", conv.conv_id, duration=prefill_msg_dur + decode_total_sec,
                     new_tokens=new_tok_count,
                     tokens_source=tokens_source,
                     decode_steps=decode_steps_done,
                     ttft_sec=round(ttft_sec, 4),
                     decode_total_sec=round(decode_total_sec, 4),
                     decode_p50_step_sec=round(decode_p50, 4),
                     first_decode_token=first_decode_token_id,
                     last_decode_token=last_decode_token_id,
                     kv_len_before=kv_len_before,
                     kv_len_after=kv_len_after,
                     total_messages_after=conv.total_messages + 1)

        conv.cache.touch()
        conv.total_messages += 1
        conv.next_message_at = time.monotonic() + conv.sample_idle_delay()

    def run(self):
        self.start_time = time.monotonic()
        # Stagger initial messages so they don't all fire at t=0
        for conv in self.conversations:
            conv.next_message_at = self.start_time + self.rng.uniform(0, 5)

        end_time = self.start_time + self.duration_sec
        # D-7.4: graceful shutdown buffer. Don't start new messages in the last
        # GRACEFUL_BUFFER_SEC seconds, since prefill+decode takes ~30s.
        graceful_cutoff = end_time - 30.0
        while time.monotonic() < end_time:
            now = time.monotonic()
            self._check_demotes()

            # Refuse new messages past graceful_cutoff to avoid sim overrun
            if now >= graceful_cutoff:
                time.sleep(min(end_time - now, 1.0))
                continue

            due = [c for c in self.conversations if c.next_message_at <= now]
            if not due:
                next_due = min(c.next_message_at for c in self.conversations)
                sleep_sec = max(0.05, min(next_due - now, 1.0))
                time.sleep(sleep_sec)
                continue

            conv = min(due, key=lambda c: c.next_message_at)
            self._process_due_message(conv)

        return self.events

    def summary(self):
        forwards = [e for e in self.events if e.event_type == "forward"]
        promotes = [e for e in self.events if e.event_type == "promote"]
        demotes = [e for e in self.events if e.event_type == "demote"]

        promote_durs = sorted([e.duration_sec for e in promotes])
        forward_durs = sorted([e.duration_sec for e in forwards])

        def pctl(lst, p):
            if not lst:
                return 0.0
            i = max(0, min(len(lst) - 1, int(len(lst) * p / 100)))
            return lst[i]

        return {
            "duration_sec": self.duration_sec,
            "total_events": len(self.events),
            "forward_count": len(forwards),
            "demote_count": len(demotes),
            "promote_count": len(promotes),
            "promote_latency_p50_sec": round(pctl(promote_durs, 50), 3),
            "promote_latency_p95_sec": round(pctl(promote_durs, 95), 3),
            "promote_latency_p99_sec": round(pctl(promote_durs, 99), 3),
            "forward_p50_sec": round(pctl(forward_durs, 50), 3),
            "forward_p95_sec": round(pctl(forward_durs, 95), 3),
            "aggregate_msg_per_sec": round(len(forwards) / self.duration_sec, 3) if self.duration_sec > 0 else 0.0,
            "per_conv": [
                {
                    "conv_id": c.conv_id,
                    "is_frequent": c.is_frequent,
                    "prefill_tokens": c.prefill_token_count,
                    "total_messages": c.total_messages,
                    "tier_stats": c.cache.get_tier_stats(),
                }
                for c in self.conversations
            ],
        }
