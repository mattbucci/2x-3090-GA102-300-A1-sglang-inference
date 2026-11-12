# Flint one-pass prefill route (patch 080)

Extends of ≥ 257 rows on Flint's full-attention layers run flashinfer's paged prefill kernel over the
fp8_e5m2 pool instead of the Triton extend kernel. `SGLANG_EXTEND_FLASHINFER_MIN_ROWS=257` (0 = off).

| Flint, cold, served | Triton extend | flashinfer route |
|---|:--:|:--:|
| TTFT 32K / 131K / 250K | 2.94 / 23.2 / 67.0 s | 2.69 / 18.3 / 50.0 s |
| TPOT 32K / 131K / 250K | 6.0 / 7.1 / 8.7 ms | 6.0 / 7.1 / 8.7 ms |
| MMLU-171, two passes | 144 + 146 | 144 + 148 |
| HumanEval-164, two passes | 141 + 131 | 138 + 134 |
| Haystack needle 196K–254K × 5 depths, two passes | 12 + 13 of 15 | 13 + 12 of 15 |

Arms interleaved in one session per gate. A first 57 / 30-sample gate read 48 / 21 against 49 / 25 from
another session; the spread within one arm above is larger than that gap.

- `ttft-off.json`, `ttft-fi.json` — served TTFT arms; `ttft-fi-int64idx.json` — the first arm with an
  int64 slot table (78.3 s at 250K): the same flashinfer kernel runs ×1.76 slower on int64 indices
  (`scripts/bench/bench_flashinfer_plan_idx_dtype.py`).
- `kernel-microbench-*.json` — kernel-only timing (`scripts/bench/bench_prefill_attn_kernel.py`).
- Quality: `benchmarks/quality/flint-fipre-g2-{0,257}-{a,b}-2026-11-12.json`,
  `benchmarks/quality/flint-fipre-g3-{0,257}-{a,b}-needle-2026-11-12.json`.
