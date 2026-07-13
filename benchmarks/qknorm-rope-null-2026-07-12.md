# `--enable-fused-qk-norm-rope` on sm_86 — NULL (2026-07-12)

Port-triage follow-up to R9700's North/Laguna v0.5.15 optimization campaign (their Triton
RMSNorm + fused q/k-norm patches measured 4–8× *kernel* speedups on ROCm). A/B on our stack:
`coder-30b-eval` (Qwen3MoeForCausalLM — the arch the flag is wired for), TP=2 @262144,
M=1 fresh prefill, `bench_long_context.py --contexts 2048 131072`, flag verified live in
`server_args` (`enable_fused_qk_norm_rope=True`).

| arm | 2K decode | 2K TPOT | 131K decode | 131K TPOT |
|---|---:|---:|---:|---:|
| baseline | 201.6 tok/s | 4.96 ms | 107.4 tok/s | 9.31 ms |
| `--enable-fused-qk-norm-rope` | 202.0 tok/s | 4.95 ms | 107.5 tok/s | 9.30 ms |

**Verdict: null (±0.2%, within noise).** Mechanism: R9700's win is vs ROCm's unfused
norm/rope kernels; on CUDA the per-step q/k-norm + rope cost is microseconds against a
5–9 ms M=1 decode step, so fusing it buys nothing measurable. Keep the flag off (default).
Receipts: `qknorm-rope-null-{baseline,flagged}-2026-07-12.json`.
