# bench_long_context depth bug — every labeled-deep decode number was shallow (2026-07-14)

**Found chasing R9700's roofline lever:** a full-attention model (coder-30b) benched identical
TPOT at 131K and 250K — physically impossible — and server-side ground truth (`Prefill batch,
#new-token` sums) showed a "2K+131K+250K" sweep prefilled only **121,922 tokens total**.

**Root cause:** `bench_long_context.py` delegates to `sglang.bench_serving --dataset-name random`
without `--random-range-ratio`, and the upstream default is **0.0** — `compute_random_lens`
then draws the prompt length **uniform in [1, requested]** (`benchmark/datasets/common.py:60`).
Every deep point ever produced by this harness measured a coin-flip depth (expectation ≈ half
the label), with ShareGPT decode–re-encode drift on top.

**Fix (this commit):** pin `--random-range-ratio 1`, record `actual_input_tokens` from
bench_serving output, and WARN + flag `depth_shortfall` on any point that under-fills >5%.
Verified on coder-30b-eval: server-side total ≈ 250K for the sweep, and the depth curve is
physical again — **199.2 @2K / 102.0 @131K / 71.6 @250K (TPOT 5.0/9.8/14.0 ms)** vs the broken
instrument's flat 9.3ms at both deep points. (TTFT at deep points reflects radix-cached prefix
prefill — decode TPOT is unaffected; KV holds the full depth at decode.)

**Blast radius:** all `benchmarks/<slug>/results.json` deep points + the README Performance
table + `all_models_{decode,context}.png` (fleet re-sweep in flight). A/B verdicts measured on
the broken instrument (qknorm null, kv-splits flat, MoE-tune nulls) stand — both arms shared the
same actual depths — but their depth labels were wrong. Needle / tool-use / reasoning probes are
unaffected (separate instruments, usage-verified true token counts).

Receipts: `coder30b-realdepth-2026-07-14.json` (fixed instrument), `/tmp` sweep logs archived in
the session; upstream-PR candidate: bench_serving's range_ratio=0 default is a depth-benchmark
footgun.
