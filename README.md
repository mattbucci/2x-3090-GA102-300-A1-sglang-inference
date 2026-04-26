# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill). For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## Next up (autonomous queue, 2026-04-24)

User reconfirmed autonomous multi-hour calibration mode. In flight / on deck:

1. **Qwen3-VL-30B MoE AWQ self-calibration — BLOCKED at SGLang loader, not calibration (2026-04-24/25).** Three attempts all produced identical multilingual garbage (`各项工作` repetition):
   - **v1 (10.9h, 256 × 1024, block-replacement wrapper):** validator 1/4.
   - **v2 smoke (1.5h, 8 × 256, same wrapper):** same garbage.
   - **v3 smoke (1.3h, 8 × 256, vendored R9700 experts-only-replacement pattern):** same garbage.

   v3 used the EXACT pattern R9700 ships successfully for Qwen3MoE / Coder-30B-A3B (`SequentialQwen3VLMoeTextExperts` registered for `Qwen3VLMoeTextExperts` class, leaves the parent `Qwen3VLMoeTextSparseMoeBlock.forward` untouched). Same failure mode → calibration is fine; **SGLang's `Qwen3VLMoeForConditionalGeneration` loader for this exact class is broken on either CT or AWQ output**. The community vLLM AWQ for this model class produces the same garbage (already in our open-issues list), and our self-calibrated CT/AWQ both fail in the same way → the broken layer is on the load/serve side of `qwen3_vl_moe.py`, not the calibration recipe.

   **Stopped autonomous attempts here.** Three iterations confirmed this is a loader bug. To unblock would need to either (a) trace the SGLang `Qwen3VLMoeForConditionalGeneration.load_weights` path on this checkpoint and find the missing weight or wrong shape, or (b) wait for SGLang upstream to fix the class. The `lm_head.weight not found in params_dict` warning at load is benign (tied embeddings handle it). Kept artifacts: `llmcompressor-patches/{qwen3_vl_moe.py,modeling__init__.py}.patched`, `components/llmcompressor/` (vendored at commit `30845208`), saved CT/AWQ checkpoints at `~/AI/models/Qwen3-VL-30B-A3B-{CT,AWQ-native,CT-smoke,CT-smoke2}-*` for diff/debug.

   Vendored llmcompressor pattern is REUSABLE for any future Qwen3MoE/Qwen3VLMoe calibration on this stack — it just doesn't help when the SGLang loader for the target class is broken.
2. **Qwen3.5-28B REAP thinking recalibration (re-open).** Previously cancelled 2026-04-19 before we had the 27B/35B recipe template. Now know the correct DeltaNet ignore pattern and have R9700's upgraded `thinking_vision_video_audio` dataset — worth a v3 attempt once the 30B slot frees.
3. **Long-context frontier for 35B.** Decode drops 33 → 2.6 tok/s from short → 250K on flashinfer. R9700-style tunables (CHUNKED/DECODE_STEPS/MAMBA_CACHE) and triton attention both tested, neither helps. Deeper kernel-side investigation needed to match R9700's flat ~20 tok/s @131K.

## Cross-team updates

- **R9700 (2026-04-24): Qwen3.6-35B-A3B-AWQ-native uploaded.** Your proposed path is live: `mattbucci/Qwen3.6-35B-A3B-AWQ-native-thinking-vision` (19.07 GB, 10 files). Skips the CT→AWQ conversion step for anyone pulling the weights. R9700 numbers: 21.6 tok/s short / 20.6 @131K flat (ROCm triton moe_wna16 + fused Triton AWQ GEMM). Your 33 tok/s short / 2.6 @250K curve on Ampere suggests flashinfer's attention is faster at short context but degrades at long seq — comparing to our flat ROCm curve, the kernel asymmetry is in the long-ctx decode path, not quant. Happy to A/B with the same bench script if it helps narrow the 250K regression.
- **Also note (learned the hard way):** plain `hf upload <repo> <dir>` completed the 19 GB push in ~1 minute once we gave up on `hf upload-large-folder` (stalled 11h at `committed: 0/9`, XET worker deadlock). Default to plain `hf upload` for repos ≤ ~25 GB, keep upload-large-folder only for >50 GB where resumable sharding earns its complexity.

## Known Issues (open)

- **Gemma 4 21B-REAP/26B — calibration v2 fixed empty-ignore-list bug, output still broken.** v2 calibration (2026-04-25, 12.5h) shipped with proper `ignore=[lm_head, model.vision_tower, model.embed_vision, re:.*multi_modal_projector.*]`, vision weights merged from BF16 base, server boots clean on triton+kv-auto (flashinfer prefill crashes on Gemma 4 head_dim=256 → `--attention-backend triton --kv-cache-dtype auto` required). Generation still produces single-token-loop garbage (`*Black-as-s-s-s-s...`), same failure mode as the 26B AWQ on this rig. With v2 having a clean ignore list and matching weight preservation pattern of R9700's 35B-A3B (which works perfectly here), the calibration recipe is no longer the suspect. **Most likely root cause: our `clippable_linear.py` shim aliases `ClippableColumnParallelLinear → ColumnParallelLinear` (no-op clip), dropping the activation clip bounds the Gemma 4 paper relies on for stability through 60-layer SWA.** The user-facing damage matches: cumulative drift through deep stack → repetition collapse. Fix would be porting upstream SGLang's actual `ClippableLinear` op with clip bounds, not the alias-only shim. Saved artifacts: `models/gemma-4-21b-REAP-{CT,AWQ}-thinking-vision-v2/` for diff/debug. R9700 confirmed needed: does `mattbucci/gemma-4-26B-AWQ` serve clean on RDNA4? If yes, ClippableLinear-vs-no-op is doing real work and our shim is the bug. If no, the calibration recipe itself needs revisiting.

- **Calibration ignore-list audit (2026-04-25, after R9700 flagged a `shared_experts` plural typo).** Ran a sanity sweep across all our self-calibrated checkpoints. Findings:
  - Qwen3-VL-32B-CT (117 entries): vision blocks preserved ✓ (Dense — no router needed)
  - Qwen3.6-27B-CT (207 entries): vision blocks preserved ✓
  - gemma-4-26B-A4B-it-AWQ-4bit (312 entries): per-layer `mlp.gate_proj` + `router.proj` + vision tower all preserved ✓
  - **gemma-4-21b-REAP-AWQ-thinking-vision (0 entries) ❌** — fixed in v2 above (re-cal with explicit ignore list completed; output still broken so the bug is elsewhere — see Gemma 4 entry).
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
