# gemma4: MoE-tune attempt BLOCKED + FlashInfer idea REFUTED (2026-07-14)

Closing data points for two open levers, both on `gemma4` (26B A4B MoE).

**1. Triton MoE config tuning — BLOCKED at the harness.** The M=1 pass benchmarked all 640
kernel configs and every one failed (`assert best_config is not None`): the same shape-hostility
that pushes gemma4 off Marlin (per-rank expert N=352, group-32 int4) defeats the tuning kernel's
benchmark path too. With the nemotron A/B already NULL (Ampere heuristic defaults at-optimum,
`nemotron-moe-tune-null-2026-07-14.md`), the config-tuning lever is closed fleet-wide. Fresh
baseline banked: **83.3 tok/s @2K / 39.5 @131K / 39.4 @250K** (matches the README table).
Harness note: `common_utils.get_model_config` gained a `Gemma4ForConditionalGeneration` branch
(topk field is `top_k_experts`) alongside the NemotronH + AWQ block-shape fixes.

**2. Per-layer-type FlashInfer for Gemma sliding layers — REFUTED, model-level.** Probe boot
with `OVERRIDE_ARGS="--attention-backend flashinfer"` (knob added at the launch.sh tail for
argparse last-wins):
`AssertionError: Gemma4 only supports trtllm_mha, triton, or intel_xpu attention backend, got prefill=flashinfer, decode=flashinfer`
— upstream gates Gemma4 off FlashInfer entirely, so the "sliding layers are FlashInfer-legal"
hypothesis never reaches the head_dim=256 question. Triton stays the Gemma path on sm_86;
the surviving fix directions for gemma4 decode remain the Marlin-friendly requant and/or a
narrow-shape AWQ-GEMM kernel (see the fallback-kernel root cause in the README Tooling triage).
