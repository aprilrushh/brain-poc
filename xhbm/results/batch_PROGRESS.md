# XHBM Batch Test Progress

Started: 2026-04-26T08:51:23+00:00
Run ID: 20260426_085123

## Scenarios planned

- S1_n1_8x32k_15min_RAM: n=8 ctx=32768 sim=900.0s decode_n=64
- S2_n2_8x32k_15min_RAM: n=8 ctx=32768 sim=900.0s decode_n=64
- S3_n3_8x32k_15min_RAM: n=8 ctx=32768 sim=900.0s decode_n=64

## Cold-start phase
- warmup (8-thread): 29.1s
- model load: 31.8s
- cold-start total: 60.8s

## Scenarios

### S1_n1_8x32k_15min_RAM — STARTED at 2026-04-26T08:52:43+00:00
  config: n=8 ctx=32768 sim=900.0s decode=64
  ✅ DONE — fwd=32 promote p50/p95/p99=1.425/1.852/9.84s GPU peak=49772.4MiB
### S2_n2_8x32k_15min_RAM — STARTED at 2026-04-26T09:10:30+00:00
  config: n=8 ctx=32768 sim=900.0s decode=64
  ✅ DONE — fwd=33 promote p50/p95/p99=1.437/1.832/8.319s GPU peak=49772.4MiB
### S3_n3_8x32k_15min_RAM — STARTED at 2026-04-26T09:28:19+00:00
  config: n=8 ctx=32768 sim=900.0s decode=64
  ✅ DONE — fwd=33 promote p50/p95/p99=1.432/1.818/8.266s GPU peak=49772.4MiB

## ALL DONE at 2026-04-26T09:46:07.048318+00:00

## Summary

| Scenario | Status | fwd | promote_p50 | p95 | p99 | GPU peak |
|---|---|---|---|---|---|---|
| S1_n1_8x32k_15min_RAM | DONE | 32 | 1.425 | 1.852 | 9.84 | 49772.4 |
| S2_n2_8x32k_15min_RAM | DONE | 33 | 1.437 | 1.832 | 8.319 | 49772.4 |
| S3_n3_8x32k_15min_RAM | DONE | 33 | 1.432 | 1.818 | 8.266 | 49772.4 |
