# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Current Focus

**Target: single-user 256K context performance.** Multi-user throughput is secondary. Aligned with the RDNA4 sister project; both teams share 256K progress bidirectionally.

Reference model for the target: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill). Qwen3.6-35B-A3B-GPTQ text-only hits the same target at 14 tok/s.

**Cross-team parity on Qwen3.6 (2026-04-18):** RDNA4 sister repo also has Qwen3.6-35B-A3B-GPTQ loaded via the `qwen36-moe` preset — 13.3 tok/s @ 262K, thinking validator passes on first probe (396 tok, `finish=stop`).  Same architecture class (`Qwen3_5MoeForConditionalGeneration`) as Qwen3.5-35B; patch 009 covers it.  Their `flatten_qwen36_config.py` now has `--arch` flag: default `Qwen3_5MoeForConditionalGeneration` for RDNA4 patch 009 registration, `--arch Qwen3_5MoeForCausalLM` for your upstream-registered class.  Conclusion: Qwen3.6 works cleanly on both stacks without recalibration — thinking survives community GPTQ where it failed on Qwen3.5 AWQ.

**Cross-team — RDNA4 multimodal calibration upgrade (2026-04-19):** R9700 web-searched current best multimodal calibration sets and replaced VATEX with `lmms-lab/LLaVA-Video-178K` (178K caps + 960K open-ended QA + 196K MC, FPS=1 untrimmed video, already in chat-template format) and added `google/covost2` for instruction-style audio.  Updated recipes in `scripts/quantize/calibration_datasets.py` on RDNA4 main: `thinking_vision_video` and `thinking_vision_video_audio`.  Recommend pulling these before kicking off any fresh multimodal calibration on this side — saves a recal cycle vs caption-only data.  R9700 also shipped Qwen3.5-27B-AWQ-thinking v2 to `mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated` (in-place upgrade, basic test went FAIL→PASS) and started Qwen3.6-35B AWQ self-calibration with `thinking_vision` recipe (5-7h CPU).

## Active work (short list)

1. **Qwen3.5-28B REAP thinking recalibration — CANCELLED (2026-04-19).** v1 died at layer 13/41 when the harness was interrupted; v2 (detached via setsid) was killed at layer 1/41 by decision. Rationale: R9700 shipped a thinking-preserving Qwen3.5-27B-AWQ v2 (`mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`, basic FAIL→PASS), and Qwen3.6-35B-A3B community GPTQ passes the thinking validator on both 3090 and R9700 — the regression this was fixing is effectively superseded by switching the long-context preset to Qwen3.6 or Qwen3-30B REAM. Future multimodal recal will use the upgraded `thinking_vision_video_audio` recipe (LLaVA-Video-178K + CoVoST2) that R9700 shipped after our v2 started.
2. **Gemma 4 26B MoE quality debug.** Server boots (patches 015 + 016 + `SGLANG_FORCE_CT_MOE_TRITON=1`) but generation emits `<pad>` tokens. Next: profile layer 0 to localize, then re-calibrate with the upgraded `thinking_vision_video_audio` recipe.
3. **Qwen3-VL-32B Dense self-calibration — SHIPPED (2026-04-21).** Calibrated with `thinking_vision` recipe, 256 samples @ 1024 tokens, 13.5h CPU on 32B Dense. Vision tower preserved in BF16 via `ignore=[re:.*visual\..*, re:.*vision_tower.*]`. Launches via `MODEL=/path/to/Qwen3-VL-32B-CT-thinking-vision QUANT=compressed-tensors EXTRA_ARGS="--enable-multimodal --reasoning-parser qwen3" ./scripts/launch.sh qwen3-vl-32b`. Validator 4/4 PASS: basic `(reasoning)paris`, thinking 108 tok `reasoning_seen+answer_ok`, vision `saw=['red','circle','round']` on solid-red-circle probe (video skipped — imageio missing in sglang env). Patch 018 (R9700 backport) unblocked the `vision_config` dict→SimpleNamespace wrap needed for CT-saved configs. Qwen3-VL-30B **MoE** version still pending.
4. **Gemma 4 21B REAP AWQ vision — KNOWN BROKEN (2026-04-21).** Our self-calibrated `gemma-4-21b-REAP-AWQ-thinking-vision` serves cleanly via `QUANT=awq_marlin EXTRA_ARGS="--attention-backend torch_native --disable-cuda-graph --disable-piecewise-cuda-graph --enable-multimodal" ./scripts/launch.sh gemma4` but validator is 3/4: basic PASS `(reasoning)paris`, thinking PASS 604 tok, **vision FAIL** — response `"i cannot see the image you are referring to"`. R9700 hit the identical failure on their 21B REAP and diagnosed it as template/processor plumbing in the SGLang Gemma 4 multimodal path, not a weight or calibration issue. Parked pending SGLang-side fix (not recalibration).
5. **Qwen3.6-27B — SHIPPED v3 (2026-04-23).** Dense 27B hybrid (DeltaNet + gated attention), 64 layers, 262K native, vision+video, `Qwen3_5ForConditionalGeneration`. Validator 4/4 PASS after diagnosing SGLang's `Qwen3_5GatedDeltaNet` loader layout:
    - **v1 (9.5h):** Qwen3-VL script, all linear_attn INT4 → `!!!!!`.  
    - **v2 (7.6h):** `re:.*linear_attn\..*` excluded → still `!!!!!`.
    - **v3 (7h):** ignore only `in_proj_a$` + `in_proj_b$` → 4/4 PASS.  
   
    Root cause: SGLang's loader merges `in_proj_qkv` + `in_proj_z` → `in_proj_qkvz` (passes through `quant_config` → expects INT4) and `in_proj_b` + `in_proj_a` → `in_proj_ba` (hardcoded `quant_config=None` → expects BF16). So the layout it wants is: in_proj_qkv/z/out_proj INT4, in_proj_a/b BF16, conv1d BF16 (it's `nn.Conv1d` not Linear — GPTQ doesn't touch it). Bench sweep:
    ```
    ctx=  1024: TPOT=32.9ms  TTFT=58ms     30.4 tok/s
    ctx=  8192: TPOT=33.2ms  TTFT=3841ms   30.1 tok/s
    ctx= 32768: TPOT=33.6ms  TTFT=7611ms   29.7 tok/s
    ctx=131072: TPOT=47.4ms  TTFT=34100ms  21.1 tok/s
    ```
    DeltaNet Triton kernels cap short-ctx throughput at ~30 tok/s (vs Qwen3-VL-32B Dense's 69 tok/s on FlashInfer). 18.6 GB CT output.
    
    **Cross-validated against R9700's shipped `mattbucci/Qwen3.6-27B-AWQ-thinking-vision` (2026-04-23):** downloaded their 18 GB HF model and ran the same validator + bench.  Throughput identical within noise (30.1/30.8/29.8/21.0 tok/s across 1K/8K/32K/131K vs our 30.4/30.1/29.7/21.1).  Quality trade: R9700's model gives a cleaner vision caption (`"a red circle on a white background"` vs ours `"the flag of japan."` — though both triggered the validator's red/circle keyword match) and ours terminates thinking more concisely (259 tok vs R9700's hitting the 4096-token reasoning ceiling on the same probe).  Different calibration recipes (R9700 used `thinking_vision_video`, we used `thinking_vision`) likely account for the different trade-offs.
4. **Piecewise CUDA graph `quant_type=None` fix.** Would unblock decode speedups on REAP/REAM/Qwen3.6 (all currently run with graphs disabled for safety).

## Known Issues (open)

- **Gemma 4 26B MoE — boots but emits `<pad>`** via `SGLANG_FORCE_CT_MOE_TRITON=1 EXTRA_ARGS="--attention-backend torch_native --disable-cuda-graph" ./scripts/launch.sh gemma4`. Suspect weight-layout mismatch in the CUDA Triton MoE path or calibration that dropped the gelu activation profile. See `patches/README.md` → Gemma 4 notes.
- **Gemma 4 31B Dense** — ~~Blocked~~ **UNBLOCKED 2026-04-23** via `QUANT=compressed-tensors EXTRA_ARGS="--attention-backend torch_native --disable-cuda-graph --disable-piecewise-cuda-graph" ./scripts/launch.sh gemma4-31b`. 11.2 GB/GPU weights, 48K max tokens at 16K context. Validator 3/4: basic + thinking PASS, vision **receives image tokens and generates** (response `"the image shows a single cuneiform character"`) — it's just hallucinated content, not the "i cannot see the image" plumbing failure the 21B REAP has. Short-ctx bench: 28 tok/s TPOT 35 ms @ 1K. Note model config is `compressed-tensors`, not AWQ, despite the directory name.
- **Qwen3-VL-30B MoE AWQ** — Community vLLM checkpoint produces garbage under `awq_marlin` (weight-name mapping mismatch for `Qwen3VLMoeForConditionalGeneration`). Workaround `SGLANG_FORCE_MOE_WNA16=1` needs a compressed-tensors checkpoint. Fix: self-calibrate CT with multimodal data.
- **Qwen3.5-27B DeltaNet stuck at 32K context** — DeltaNet layers replicated across GPUs (19 GB/GPU), leaving only 2.2 GB for KV cache. REAM/REAP MoE variants unlock longer context; `launch.sh qwen3-ream` is the recommended path for this architecture class.
  - **Cross-team (M4 SGLang/MLX, 2026-04-18 evening — REVISION):** the earlier M4 cross-team note claiming DeltaNet quality is broken everywhere was WRONG about M4. M4 patch 013 root-caused the M4-specific brokenness to a cache-wiring bug in the SGLang↔MLX bridge, not a DeltaNet architectural issue: when Qwen3.5/3.6 load via `mlx_vlm.load` (vision_config in their config.json), `_acquire_cache` couldn't find `make_cache` on the outer wrapper and built uniform `ContiguousKVCache` for every layer, giving DeltaNet's hybrid layers the wrong cache type. Both Qwen3.5-27B-4bit and Qwen3.5-9B-MLX-8bit now produce correct factual answers on M4 with patch 013. **Lesson:** before assuming DeltaNet itself is broken on a backend, verify the cache plumbing: each architecture-specific cache type (ArraysCache, KVCache, RotatingKVCache) must reach the layer it was built for. The M4 MMLU 16.7-33.3% numbers in the previous note are stale and should not be cited as evidence of universal DeltaNet brokenness.
- **Qwen3.5-28B REAP `<think>` tags broken** — deprioritized; prefer `launch.sh qwen36` or `launch.sh qwen3-ream` for thinking-at-long-context workloads. Recal was cancelled in favor of these alternatives.
- **60B+ models** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) don't fit in 48 GB.
- **Qwen3.6-35B-A3B vision path** — `Qwen3VLMoeVisionModel` receives `vision_config` as a dict on this checkpoint. Text-only works after `flatten_qwen36_config.py`; vision needs a loader fix or cleaner self-calibrated config.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.10, apply patches, create conda env

./scripts/launch.sh qwen3-ream              # fastest 256K — reference model
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B text-only (262K, vision TBD)
./scripts/launch.sh devstral-long           # Devstral-24B at 217K single-user ceiling
./scripts/launch.sh devstral                # Devstral-24B default (131K, better short-ctx + multi-user)
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput

python scripts/eval/validate_chat_template.py --model /path/to/model      # pre-launch capability check
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
| **Qwen3.6-35B-A3B GPTQ** | DeltaNet+MoE (256 exp) | **262K** | **14** | 72 ms | `qwen36` | Text-only; vision TBD |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling |
| Devstral-24B AWQ | Dense | 131K | 55.8 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | Thinking broken (recal running) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | 13.5 | 74 ms | `qwen35` | Working |
| **Qwen3.6-27B CT thinking+vision (ours)** | DeltaNet+attn (vision) | **131K** | **21** | 47.4 ms | `qwen35` + MODEL env | **Self-calibrated v3, validator 4/4** |
| Qwen3-VL-32B Dense AWQ | Dense (vision) | 8K | 24 | 45 ms | `qwen3-vl-32b` | Working (community) |
| **Qwen3-VL-32B CT thinking+vision (ours)** | Dense (vision) | **150K** | **40** | 24.7 ms | `qwen3-vl-32b` + MODEL env | **Self-calibrated, validator 4/4** |
| Gemma 4 26B MoE | MoE (103 exp) | 4K | — | — | `gemma4` | Boots via patches 015/016, `<pad>` output |
| Gemma 4 31B Dense | Dense | 16K | 28 | 35 ms | `gemma4-31b` | Working w/ torch_native (basic+thinking PASS, vision hallucinates) |

### VRAM context limits (FP8 KV, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 262K |
| Coder-30B AWQ | 8.0 GB | 36 KB | 262K |
| Devstral-24B AWQ (long preset) | 7.0 GB | 80 KB | **217K** (true 3090 ceiling for 24B dense @ TP=2) |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 131K |
| Qwen3.5-27B AWQ | 19.0 GB | 24 KB | 32K (weights replicated for DeltaNet TP) |

## Benchmarks (single-user long-context)

### Qwen3-30B REAM AWQ — 262K

| Context | TPOT | tok/s |
|:-------:|:----:|:-----:|
| 1K  |  5.4 ms | 185 |
| 16K |  5.6 ms | 179 |
| 64K |  7.0 ms | 143 |
| 131K |  9.4 ms | 107 |
| **250K** | **13.5 ms** | **74** |

Source: `benchmarks/qwen3-30b-ream/long-context-262k.json`. Peak batch: 1,832 tok/s @16 concurrent at 16K context (multi-user preset).

### Qwen3.6-35B-A3B GPTQ-Int4 — 262K (text-only)

| Context | TPOT | tok/s |
|:-------:|:----:|:-----:|
| 1K  | 54 ms | 18.4 |
| 16K | 59 ms | 17.1 |
| 64K | 62 ms | 16.2 |
| 131K | 66 ms | 15.1 |
| **250K** | **72 ms** | **14.0** |

Launch via `./scripts/launch.sh qwen36`. Requires one-shot config flatten: `python scripts/quantize/flatten_qwen36_config.py $MODELS_DIR/Qwen3.6-35B-A3B-GPTQ-Int4` (promotes `text_config.*` to top level, switches to `Qwen3_5MoeForCausalLM`). Source: `benchmarks/qwen3.6-35b-a3b/long-context-*.json`.

### Devstral-24B AWQ — 217K single-user ceiling

| Context | TPOT | tok/s |
|:-------:|:----:|:-----:|
| 1K  | 11.3 ms | 88.7 |
| 16K | 12.1 ms | 82.6 |
| 64K | 14.6 ms | 68.5 |
| **131K** | **17.9 ms** | 55.8 |
| **200K** | **17.9 ms** | 55.9 |

Launch via `./scripts/launch.sh devstral-long` (MEM=0.97, chunked=2048, disables CUDA graph / overlap / radix). TPOT plateaus at 17.9 ms past 131K — memory-bandwidth bound. Can't reach a full 262K prompt — Devstral has no sliding window and full-attention KV at 80 KB/token exceeds per-GPU budget.

### Quality (REAP vs REAM vs original)

![Quality Comparison](benchmarks/quality/quality_comparison.png)

| Model | MMLU | HumanEval | Needle (65K) |
|-------|:----:|:---------:|:------------:|
| Coder-30B (128 exp) | 73% | 100% | 100% |
| REAP-28B DeltaNet (205 exp) | 70% | 80% | 100% |
| REAM-30B (96 exp) | 63% | 80% | 100% |

Methodology: `scripts/eval/eval_and_chart.py` — MMLU (200 samples), HumanEval pass@1 (30 samples), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 × 50), needle-in-a-haystack (1K→65K). Full context as reasoning budget, temperature=0.

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
| SGLang | v0.5.10 + 12 local patches |
| PyTorch | 2.9.1 + cu128 |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 |
| transformers | 5.5.3 |

## Patches

12 patches on top of SGLang v0.5.10 — details in [`patches/README.md`](patches/README.md).

| # | Patch | LOC | What |
|:-:|-------|:---:|------|
| 001 | upstream-sync | ~3,000 | Gemma 4, Qwen3.5 / 3-Next, Triton attention |
| 002 | nvidia-model-fixes | 923 | Marlin shape fallback, DeltaNet TP, Gemma4 config |
| 003 | deltanet-triton-dtype-fix | 51 | DeltaNet conv_state bf16/fp16 cast |
| 004 | gemma4-causal-lm-fix | 19 | CausalLM multimodal detection bypass |
| 005 | ampere-fp8-triton-fallback | 59 | FP8 KV on sm_86 (PyTorch fallback) |
| 006 | awq-bf16-activation-support | 15 | BF16 activations for Gemma 4 |
| 007 | ampere-deltanet-kernel-tuning | 48 | BV=64 kernel tuning (1.57× DeltaNet) |
| 008 | awq-moe-wna16-fallback | 64 | `SGLANG_FORCE_MOE_WNA16=1` (saves ~7 GB peak) |
| 009 | qwen35-moe-causalLM | — | Qwen3.5 MoE text-only wrapper |
| 011 | triton-attention-fp32 | — | FP32 online-softmax (R9700 backport) |
| 012 | sliding-window-decode-fix | — | SWA `window_kv_offsets` (R9700 backport) |
| 014 | gemma4-reasoning-parser | 40 | `--reasoning-parser gemma4` (R9700 backport) |
| 015 | ct-wna16-dequant-layout-fix | — | CT dequant `[out, in]` layout (unblocks Gemma 4 MLP) |
| 016 | ct-moe-gelu-triton-route | 47 | `SGLANG_FORCE_CT_MOE_TRITON=1` + gelu in CT MoE |

## Quantization

Self-calibrated AWQ models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_llmcompressor.py
python scripts/quantize/convert_qwen35_ct_to_awq.py
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
patches/                  # SGLang v0.5.10 patches — see patches/README.md
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
