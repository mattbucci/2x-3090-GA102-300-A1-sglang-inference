# Inkling-2-Mini on Ampere (sm_86)

Base: upstream SGLang PR sgl-project/sglang#51926 (Inkling-2-Mini, SM90/SM100/SM120 only), head `9c64454bba`,
in a separate tree `/data/sglang-main-inkling` + env `sglang-main` (production stays on v0.5.23). Apply with
`git -C /data/sglang-main-inkling apply patches/inkling2-main/001-inkling2-ampere-sm86.patch`.

What 001 changes for SM8x (all marked `[3090 inkling2-ampere]`):
- attention: no FA4 (grouped-head decode + extend kernels: each K/V tile serves the KV head's 3 query heads; decode tables built on device so decode CUDA graphs capture) → `inkling2_attn_ampere.py` writes K/V through the Triton backend (window sub-pool aware) and runs
  `inkling2_rel_bias_ampere.py` (split-KV paged attention with the compact relative bias, SWA window, GQA; returns o + LSE
  for the existing gate/sink epilogue).
- CuTe-DSL rms_norm / qk_l2_clip / attn_res → torch fallbacks (`kernels/ops/inkling2_ampere.py`).
- experts: Marlin NVFP4 / MXFP8 path enabled on SM8x; cluster GEMV disabled (thread-block clusters are sm_90).
- e4m3 activation fake-quant: Triton cannot cast to fp8e4nv below sm_89 → exact fp32 round-to-e4m3 emulation.
- PDL (`griddepcontrol`) guarded to sm_90 in the Inkling 2 CUDA/Triton kernels; no NVLink multicast → group all-reduce.
- vision attention: per-image SDPA on CUDA below SM90.

Tests (`tests/`): rel-bias attention vs fp32 reference (3e-3), Marlin GEMM vs dequant (3.4e-3), e4m3 rounding bit-exact.

## Results (2x3090, TP=2, 256K context, decode CUDA graphs, natural prompts, single user)

| ctx | decode tok/s | TTFT |
|---|---|---|
| 1K | 65.5 | 0.4 s |
| 32K | 62.2 | 11 s |
| 131K | 54.1 | 56 s |
| 250K | 46.9 | 153 s |

Receipts: `benchmarks/inkling2/long-context-*.json` (eager 8 tok/s → graphs 28.9 @250K → grouped decode 46.9; prefill 216 → 153 s @250K).
