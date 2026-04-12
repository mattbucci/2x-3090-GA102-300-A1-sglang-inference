# Rules for AI Agents — 2x RTX 3090

## Inference Engine

All inference MUST use SGLang. No vLLM, no llama.cpp unless explicitly for comparison benchmarks.

## Server Launch

```bash
source scripts/common.sh
activate_conda
setup_nvidia_env
./scripts/launch.sh <model>
```

Always source `common.sh` + `activate_conda` + `setup_nvidia_env` before launching.

## VRAM Budget

48GB total (2x 24GB). Model weights + KV cache must fit. AWQ INT4 uses ~0.55 bytes/param:

| Model | Weight VRAM | Max practical context |
|-------|-------------|----------------------|
| Devstral-24B AWQ | ~13 GB | 32K |
| Coder-30B MoE AWQ | ~17 GB | 16K |
| Gemma 4 26B MoE AWQ | ~14 GB | 4K |
| Gemma4-31B AWQ | ~17 GB | 4K |
| Qwen3.5-27B AWQ | ~15 GB | 32K |
| Coder-Next-80B AWQ | ~44 GB | DOES NOT FIT |

Do NOT attempt 80B+ models — they exceed 48GB with any useful KV cache.

## Quantization Backend

Use `awq_marlin` (Marlin kernels) for all AWQ INT4 models. Marlin supports sm_80+ (Ampere and newer). SGLang auto-promotes AWQ to `awq_marlin` at load time, but we set it explicitly for clarity.

Marlin kernels provide optimized memory access patterns (async loads, better L2 cache utilization, fused dequant+matmul). Standard AWQ model checkpoints work — Marlin repacking happens at runtime.

Requirements for Marlin compatibility:
- `quant_method: "awq"`, `bits: 4`, `group_size: 128` (standard config)
- Output dim divisible by 64, input dim divisible by 128 (fallback to generic AWQ for non-conforming layers)

## Key Differences from AMD (RDNA4) Setup

- **AWQ Marlin kernels** — optimized CUDA kernels (AMD uses Triton AWQ GEMM)
- **CUDA graphs work** — no `--disable-cuda-graph`
- **FlashInfer attention** — default backend, no need to force `--attention-backend triton`
- **Custom all-reduce works** — no `--disable-custom-all-reduce`
- **Overlap schedule works** — no `--disable-overlap-schedule`
- **NCCL native** — no RCCL, no custom builds
- **FP8 KV cache** — works via type casting on Ampere (sm_86)
- Patches may still be needed for performance tuning or model-specific fixes

## Quantization Rules

These apply identically to the AMD setup:

- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts
- **DeltaNet/SSM layers cannot be INT4** — recurrent state error accumulation destroys quality
- Pipeline: GPTQ calibration (llmcompressor) -> CT format -> AWQ conversion
- Use expert-balanced sampling for MoE models

## Benchmarking

- Always measure **TPOT** (time per output token) with `sglang.bench_serving`
- Never use wall-clock time (mixes prefill and decode)
- Concurrency sweep: 1/2/4/8/16/32
- Context sweep: powers of 2 from 128 to model max
- Save to `benchmarks/{model}/results.json`
- Regenerate charts after new results
- Run regression test before committing changes
- Regression threshold: >10% deviation triggers alert
