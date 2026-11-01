# W4A8 prefill on qwen38 (patch 071) — served A/B, 2026-11-01

Same tree, same session, arms back to back (`SGLANG_AWQ_W4A8_PREFILL_MIN_TOKENS=0` then `2048`), natural-text fresh prefill (`bench_long_context.py --prompt-source natural`, cache flushed per point), TP=2, 300 W, shipped `qwen38` preset otherwise (fp8_e4m3 KV, chunked prefill 8192).

| context (actual) | TTFT, Marlin W4A16 | TTFT, INT8 prefill | Δ | decode tok/s (both arms) |
|---|---:|---:|---:|---:|
| 32,768 | 13.38 s | **8.60 s** | **−36 %** | 80.7 / 80.5 |
| 131,072 | 71.35 s | **50.98 s** | **−29 %** | 68.9 / 68.9 |
| 261,916 | 185.6 s | **144.9 s** | **−22 %** | 58.7 / 58.7 |

- Matches the kernel-level sizing (INT8 path 2.0–2.3× the Marlin GEMM chain, which was 67 % of a cold 32K fill and 46 % at 131K; attention and NCCL take the rest at depth).
- Decode is untouched by construction: calls with fewer rows than the threshold stay on Marlin W4A16 (the INT8 path would double the weight bytes a decode step streams).
- The INT8 weight copy is made per call from the Marlin-permuted INT4 tile through the recovered tile map (`test_marlin_tile_perm.py`) and lives only for the GEMM; load-time cost is the per-channel scale (exact from the group scales / zeros, no dequant pass).
- Quality gate on the INT8 arm (needle 131K + 250K, MMLU / HumanEval) — see below.

Receipts: `arm-0.json`, `arm-2048.json` (this directory); unit test `benchmarks/profiling/w4a8-prefill-unit-2026-11-01.json`.

## Quality gate on the INT8 arm (same server)

MMLU **30/30** (deterministic 30-subject subset; 57-sample reference 52/57), HumanEval **29/30** (reference 29/30), natural needle **6/6** at 131K + 250K (`qwen38-w4a8-2026-11-01.json`, `qwen38-w4a8-needle-2026-11-01.json`). Shipped as the `qwen38` preset default.
