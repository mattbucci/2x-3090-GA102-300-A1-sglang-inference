# Custom allreduce captures again on v0.5.22 — +10 % decode on qwen38 (2026-10-31)

**History:** the custom (CUDA-IPC) allreduce was disabled on 2026-04-12 (v0.5.11) because it broke cuda-graph capture on
3090 TP=2 (`Capture cuda graph failed: invalid argument`, `custom_all_reduce.cuh:508`), re-confirmed broken on v0.5.12
([receipt](allreduce-accel-null-2026-06-15.md)); `launch.sh` kept `--disable-custom-all-reduce` with an
`ENABLE_CUSTOM_AR=1` toggle "to re-test at each rebase". The qwen38 agentic profile charges the NCCL allreduce 12 % of
the decode step (129 × 13.6 µs at 14.6K context).

**How it was found:** by accident. A re-test boot under `ENABLE_CUSTOM_AR=1` was declared failed by a driver's premature
liveness check, its `pkill` missed the still-starting python, and the server finished booting; the next two drivers
adopted it through `/health` and benched it as a "memory-clock lock" arm and as the real custom-AR arm — both exactly
86.4 / 80.6 tok/s, with a KV pool 2.3K tokens smaller than every other boot (the IPC buffers). The clean A/B/A below
reproduces it.

**Instrument:** `scripts/bench/bench_long_context.py` (M=1, 100 output tokens, server-verified depths), `qwen38` as
shipped (INT8 head, conv state in the model dtype, fp8 KV, TP=2 / 262144, 300 W profile), each arm a fresh boot from a
drained GPU, port checked empty before each boot, per-arm `max_total_num_tokens` logged.

| arm | KV pool | 1K | 32K | 261,916 |
|---|---:|---:|---:|---:|
| custom AR off (baseline, today's tripwire level) | 682,923 | 78.0 | 73.2 | 54.4 |
| **custom AR on** | 680,638 | **86.3 (+10.6 %)** | **80.5 (+10.0 %)** | **58.3 (+7.2 %)** |
| custom AR off again | 682,923 | 77.9 | 73.1 | 54.4 |

The gain is a fixed ≈ 1.2 ms per step at every depth (TPOT 12.8 → 11.6 ms at 1K, 18.4 → 17.1 ms at 262K) — the
allreduce latency, as the profile predicted. Quality gate (fleet harness MMLU + HumanEval, sampled needle 131K / 250K
on qwen38 and qwen36) and the fleet tripwire under the new default follow below.

## Quality gate (fleet harness, MMLU 60 + HumanEval 30 at `--mc-budget 2560`; sampled needle 131K / 250K × 0.1 / 0.5 / 0.9)

| preset | custom AR on | today's other arms (off) |
|---|---|---|
| qwen38 | **52/57 · 29/30 · needle 6/6** | 52–55/57 · 29/30 · 6/6 |
| qwen36 | **53/57 · 29/30** (needle below) | 53–54/57 · 29/30 · 6/6 |

Parity → `launch.sh` turns the custom allreduce ON by default (`DISABLE_CUSTOM_AR=1` opts out); the fleet tripwire is
re-saved under it and the serving image rebuilt (below).

