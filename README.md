# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill). For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## Next up (autonomous queue, 2026-04-24)

User reconfirmed autonomous multi-hour calibration mode. In flight / on deck:

1. **Qwen3-VL-30B MoE AWQ self-calibration — IN FLIGHT.** Community vLLM AWQ produces garbage under `awq_marlin` (weight-name mismatch for `Qwen3VLMoeForConditionalGeneration`). Plan: download BF16 base (~60 GB, running now), adapt the Qwen3.6-27B `thinking_vision` recipe for 30B-A3B (preserve vision tower + DeltaNet `in_proj_a/b`), calibrate on CPU (~10-13h), run CT→AWQ conversion (BF16 fallback for any `(1,H)` shared_expert_gate), validate 4/4.
2. **Qwen3.5-28B REAP thinking recalibration (re-open).** Previously cancelled 2026-04-19 before we had the 27B/35B recipe template. Now know the correct DeltaNet ignore pattern and have R9700's upgraded `thinking_vision_video_audio` dataset — worth a v3 attempt once the 30B slot frees.
3. **Long-context frontier for 35B.** Decode drops 33 → 2.6 tok/s from short → 250K on flashinfer. R9700-style tunables (CHUNKED/DECODE_STEPS/MAMBA_CACHE) and triton attention both tested, neither helps. Deeper kernel-side investigation needed to match R9700's flat ~20 tok/s @131K.

## Known Issues (open)

- **Gemma 4 26B MoE + Gemma 4 21B REAP vision — blocked on SGLang `clippable_linear`** (same root cause). SGLang logs `Ignore import error when loading sglang.srt.models.{gemma4_audio,gemma4_mm,gemma4_vision}: No module named 'sglang.srt.layers.clippable_linear'`. All three Gemma 4 multimodal loaders fail to import; 26B MoE falls back to `Gemma4ForCausalLM` (no expert routing → `<pad>` output), 21B dense loads as text-only and discards image tokens ("i cannot see the image"). Not a calibration defect or launch-flag fix — needs the `clippable_linear` layer added to SGLang upstream or the `gemma4_mm` loader decoupled. Skipped in autonomous queue.
- **Qwen3.6-35B-A3B long-context decode curve** — 33 tok/s short / 5.8 @160K / 2.6 @250K on flashinfer. Steeper than R9700's flat 20 tok/s @131K on ROCm-triton. A/B'd: matching R9700's tunables doesn't help; `--attention-backend triton` is *worse* on Ampere (3.4 tok/s @131K). R9700's flat curve is triton-kernel-specific. Working but open frontier.
- **Qwen3-VL-30B MoE AWQ** — community vLLM checkpoint broken. Self-calibration in flight (see Next up #1).
- **Qwen3.5-27B DeltaNet stuck at 32K context** — DeltaNet layers replicated across GPUs (19 GB/GPU), leaving 2.2 GB for KV cache. REAM/REAP MoE variants unlock longer context; `launch.sh qwen3-ream` is the recommended path for the DeltaNet architecture class.
- **Qwen3.5-28B REAP `<think>` tags broken** — recal re-opened in queue (#2). Meanwhile use `qwen36` or `qwen3-ream` for thinking-at-long-context.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- **Piecewise CUDA graph `quant_type=None`** — would unblock decode speedups on REAP/REAM/Qwen3.6 (all currently run with graphs disabled for safety).

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.10, apply patches, create conda env

./scripts/launch.sh qwen3-ream              # fastest 256K — reference model
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B AWQ-native thinking+vision (262K, 4/4)
./scripts/launch.sh devstral-long           # Devstral-24B at 217K single-user ceiling
./scripts/launch.sh devstral                # Devstral-24B default (131K, better short-ctx + multi-user)
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput

python scripts/eval/validate_capabilities.py --port 23334                 # thinking + vision + basic probe
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

- 2x NVIDIA RTX 3090 (24 GB each, 48 GB total) with NVLink bridge
- NVIDIA driver 595+ / CUDA 13.x
- Miniforge3 or Conda
- ~150 GB disk for models

## Model Support

Single-user tok/s measured at the max-context value in the table. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **74** | 13.5 ms | `qwen3-ream` | **Hits 256K target** |
| **Qwen3.6-35B-A3B AWQ-native** | DeltaNet+MoE (256 exp, VL) | **262K** | 2.6 | 385 ms | `qwen36` | thinking+vision 4/4; 33 @ short / 5.8 @160K / 2.6 @250K |
| **Qwen3.6-27B CT thinking+vision** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` + MODEL env | Self-calibrated v3, 4/4 |
| **Qwen3-VL-32B CT thinking+vision** | Dense (VL) | **150K** | **40** | 25 ms | `qwen3-vl-32b` + MODEL env | Self-calibrated, 4/4 |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Qwen3-VL-32B Dense AWQ (community) | Dense (VL) | 8K | 24 | 45 ms | `qwen3-vl-32b` | Working; self-cal above is preferred |
| Gemma 4 31B Dense | Dense | 16K | 28 | 35 ms | `gemma4-31b` | basic+thinking PASS, vision hallucinates (plumbing works) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | 13.5 | 74 ms | `qwen35` | Working; superseded by Qwen3.6-27B |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | Thinking broken — recal queued |
| Gemma 4 26B MoE | MoE (103 exp) | — | — | — | `gemma4` | Blocked: `clippable_linear` missing |
| Gemma 4 21B REAP AWQ | MoE | — | — | — | — | Blocked: `clippable_linear` missing |

### VRAM context limits (KV dtype varies, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-native | 9.87 GB | ~8 KB hybrid | 262K |
| Coder-30B AWQ | 8.0 GB | 36 KB | 262K |
| Devstral-24B AWQ (long preset) | 7.0 GB | 80 KB | **217K** (true 3090 ceiling for 24B dense @ TP=2) |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 131K |
| Qwen3.5-27B AWQ | 19.0 GB | 24 KB | 32K (weights replicated for DeltaNet TP) |

## Benchmarks

Per-model long-context sweep JSON in `benchmarks/<model>/`. Reference: Qwen3-30B REAM AWQ at `benchmarks/qwen3-30b-ream/long-context-262k.json`. Qwen3.6-35B-A3B AWQ-native detailed curve + tuning experiments in `benchmarks/qwen3.6-35b-a3b/awq-native-thinking-vision.json`.

### Quality (REAP vs REAM vs original)

![Quality Comparison](benchmarks/quality/quality_comparison.png)

| Model | MMLU | HumanEval | Needle (65K) |
|-------|:----:|:---------:|:------------:|
| Coder-30B (128 exp) | 73% | 100% | 100% |
| REAP-28B DeltaNet (205 exp) | 70% | 80% | 100% |
| REAM-30B (96 exp) | 63% | 80% | 100% |

Methodology: `scripts/eval/eval_and_chart.py` — MMLU (200 samples), HumanEval pass@1 (30 samples), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 × 50), needle-in-a-haystack (1K→65K). Temperature=0, full context as reasoning budget.

**Still TODO:** [RULER](https://github.com/NVIDIA/RULER) (4K→256K synthetic), [LongBench Pro](https://arxiv.org/html/2601.02872v1), [LiveCodeBench](https://livecodebench.github.io/).

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
cd components/sglang && git checkout v0.5.10
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e ".[srt]"
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.10 + 14 local patches |
| PyTorch | 2.9.1 + cu128 |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 |
| transformers | 5.5.3 |

## Patches

14 patches on top of SGLang v0.5.10 — full details in [`patches/README.md`](patches/README.md).

## Quantization

Self-calibrated models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                       # MoE CT→native AWQ
```

For thinking+vision-preserving calibration: `scripts/quantize/calibration_datasets.py` builds `thinking_text` / `thinking_vision` / `code_vision` / `code_thinking` recipes (drawing from AM-Thinking-v1-Distilled, NuminaMath-CoT, LLaVA-Instruct-150K, UltraChat, the-stack). See [rules-for-agents.md](rules-for-agents.md) and [REAM.md](scripts/quantize/REAM.md).

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
RAM:    96 GB (92 usable)
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24 GB GDDR6X each)
NVLink: NV4 — 4 lanes × 14 GB/s = 56 GB/s bidirectional
Driver: 595.58.03   CUDA 13.2   Python 3.12
```

## Repo layout

```
patches/                  # SGLang v0.5.10 patches — see patches/README.md for full narratives
benchmarks/               # Per-model benchmark JSON + charts
  quality/                #   MMLU / HumanEval / LAB-Bench / Needle
  <model>/                #   throughput + long-context sweeps
scripts/
  launch.sh               # unified launcher (launch.sh <preset>)
  common.sh               # shared conda + NVIDIA env setup
  setup.sh                # full setup (conda, SGLang install, patch apply)
  bench/                  # throughput benchmarks
  eval/                   # quality evals + chat template validator
  quantize/               # GPTQ → CT → AWQ pipeline + calibration recipes
  test/                   # kernel microbenchmarks + profiling
components/sglang/        # SGLang v0.5.10 + patches (cloned by setup.sh)
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).
