# Decode roofline: theoretical maximum, bottlenecks, where each preset stands

The single-user decode ceiling on this box is the **memory-bandwidth roofline**: the bytes one
decode step must read on each GPU (the active weights plus the KV cache the attention layers scan),
divided by the GA102's memory bandwidth (936 GB/s on paper; a 4 GiB read measures 889 GB/s on both cards at the 300 W cap, `gpu-bandwidth-300w.json` — the floors below use the paper figure, so they are ~5 % optimistic). Compute and kernel latency only ever add to it. A preset "at the
roofline" has no serving-side decode lever left; a preset far below it is bound by something else,
and the kernel trace says what.

Instrument: `/start_profile` kernel trace of 60 steps on rank 0 (TP=2, 300 W cap), grouped by
`benchmarks/profiling/roofline-2026-11-13/group_trace.py`; bytes from the checkpoint's safetensors
headers. Step times are GPU-side (client tok/s is a few percent lower).

## Standing

| Preset | Depth | Measured step | Bandwidth floor | Standing | Bound by |
|---|---|---|---|---|---|
| `flint` (40B-A2.5B MoE, 14 full + 35 SWA layers) | 34K | 6.30 ms (159 tok/s) | ~1.1 ms | ~17 % | kernel count, small-GEMM latency |
| `flint` | 241K | 8.07 ms (124 tok/s) | ~2.7 ms | ~33 % | same; KV read on the 128-key tile (patch 083) |
| `qwen38` (dense 27B) | short | 11.55 ms (87 tok/s) | weights at ~925 GB/s | at roofline | memory bandwidth |
| `qwen38` | 244K | 16.87 ms (59 tok/s) | KV read at ~900 GB/s | at roofline | memory bandwidth |
| `qwen36` (35B-A3B MoE) | short | 4.48 ms (223 tok/s) | — | 14 % idle gap | ~1,000 kernels/step |
| `qwen36` | 244K | 6.78 ms (148 tok/s) | attention at roofline | attention 45 % of step | KV read |
| `gemma4-31b` (dense 31B, 10 global + 50 SWA layers) | 37K | 18.56 ms (50 tok/s) | ~12.7 ms | ~68 % | weights + bf16 tied head, at bandwidth |
| `gemma4-31b` | 223K | 23.9 ms (42 tok/s) | ~16.9 ms | ~71 % | global-layer KV read at ~74 % of bandwidth on the 64-key tile (patch 083; 28.0 ms / ~60 % before) |

`qwen38` / `qwen36` rows: [`decode-profile-2026-10-31.md`](../benchmarks/profiling/decode-profile-2026-10-31.md).
`gemma4-31b` ([`gemma4-31b-groups.json`](../benchmarks/profiling/roofline-2026-11-13/gemma4-31b-groups.json)): 1,246 kernels per step with 0.14 ms idle, so launch count is not the bound. Per rank and step: INT4 linears 9.92 ms for ~7.65 GB of weights (87 % of the 889 GB/s ceiling); the tied bf16 `lm_head` is one 3.17 ms gemv over 2.82 GB (at the ceiling — 17 % of the step at 37K, and patch 084 now serves an INT8 copy of it: 39.8 → 44.3 tok/s at 262K, the rows above predate it); attention is 3.55 ms at 37K and 12.97 ms at 223K, of which the 10 global layers (512-wide heads, e5m2 KV, 2,048 B per token per rank) take 1.13 ms each for 0.46 GB — ~400 GB/s, under half the ceiling — while the 50 sliding-window layers cost 12 µs each. Not yet broken down this way: prefill (compute-bound: a tensor-rate roofline, not this one).

## Flint, plain decode at 241K

Per token each rank reads ~0.77 GB of weights (dense projections + shared expert ~0.37 GB, 6 of 256
experts per layer ~0.24 GB, INT8 `lm_head` ~0.15 GB) and 1.73 GB of fp8 KV over the 14
full-attention layers.

| Cost | Measured | Bandwidth floor | Note |
|---|---|---|---|
| Attention over KV | 3.63 ms | 1.85 ms | depth-dependent part (2.47 ms) at ~75 % of the roofline; 1.16 ms is fixed overhead of 233 small kernels |
| Dense INT4 GEMM | 1.80 ms | ~0.4–0.55 ms | 197 calls on 2048-wide matrices, ~9 µs each: latency-bound |
| MoE expert INT4 GEMM | 0.98 ms | ~0.26 ms | |
| Glue: elementwise 0.89, MoE routing 0.63, norms 0.34, all-reduce 0.30 | 2.16 ms | 0 | ~1,170 of the step's 1,702 kernels, no model bytes |
| Idle gaps | 0.24 ms | 0 | |

Flint is not memory-bound: ~3.3 ms of the step (glue, attention fixed overhead, idle) carries no
roofline bytes, and the GEMMs run at 20–27 % of bandwidth because the matrices are small.

## Past the roofline: speculative decode

One verify pass reads the weights once for several tokens, so accepted tokens per cycle multiply the
ceiling. Flint DFlash2 with patch 082: cycle 11.6 ms at 34K, 18.1 ms at 241K, against plain steps of
6.3 / 8.8 ms → break-even at 1.8 / 2.1 accepted tokens. Head-to-head on a natural-text summary (two interleaved passes, `flint-plain-vs-dflash2.json`): **166 → 234 / 252 tok/s at 34K, 115 → 195 / 176 tok/s at 241K** — at depth the spec arm runs at about half of the plain-decode bandwidth ceiling, the plain arm at 30 %. In the cycle at 241K attention is 9.5 ms
(8 verify rows scan the same KV the 1-row decode kernel scans in ~2.5 ms), MoE GEMM 2.3 ms (expert
fan-out across rows), glue 3.4 ms. `qwen38` DSpark is the shipped example (1.4–1.5×). Code generation over a source-code context holds the same ratio (169 → 277 / 289 tok/s at 27K, 127 → 203 / 183 at 183K), and the `flint-spec` tripwire arm reads 298 / 238 / 162 tok/s at 1K / 32K / 262K against 115 plain at 262K. Quality gate vs plain decode is at parity (tool use, cached turns, deep needle; the one reasoning-probe cell that differed is a near-tie branch plain decode also takes on a cache hit — ledger).

## Levers this ranks

1. Flint verify attention at depth: 8 rows cost ~3× the 1-row KV scan for the same bytes.
2. Flint glue kernel count (MoE routing, elementwise): 2.2 ms plain, 3.4 ms per spec cycle.
3. Flint attention fixed overhead: 1.16 ms across 49 layers at any depth.
4. Dense presets: model-side only (spec drafts, fewer KV bytes).
