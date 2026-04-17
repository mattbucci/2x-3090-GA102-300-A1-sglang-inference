# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Known Issues

- **Gemma 4 (26B MoE, 31B Dense)** — Blocked by FlashInfer `BatchPrefillWithPagedKVCache` on sm_86. Gemma 4's full-attention layers use `global_head_dim=512` which FlashInfer doesn't support on Ampere (only 64/128/256). See [FlashInfer details](#flashinfer-head_dim-support).
- **Qwen3-VL-30B MoE AWQ** — Two issues. (1) `--quantization awq` (base AWQConfig) has [no FusedMoE handler](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/layers/quantization/awq.py#L187-L191) — MoE expert layers fall back to unquantized fp16 weights (384 MB/layer × 48 layers → 23 GB/GPU OOM). Fix: use `--quantization awq_marlin`. (2) Even with `awq_marlin`, the [community vLLM checkpoint](https://huggingface.co/cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ) produces garbage output — likely a weight-name mapping mismatch between vLLM and SGLang for `Qwen3VLMoeForConditionalGeneration`. Workaround: use `SGLANG_FORCE_MOE_WNA16=1` to skip Marlin MoE repack (saves ~7 GB peak VRAM), but needs a compressed-tensors checkpoint since AWQ packing ≠ WNA16 packing. Self-calibrating a CT checkpoint via llmcompressor would fix both issues.
- **Qwen3.5-27B DeltaNet context limited to 32K** — DeltaNet layers replicated across GPUs (19 GB/GPU), leaving only 2.2 GB for KV cache. REAM/REAP MoE variants would reduce weights and unlock longer context.
- **AWQ Marlin MoE peak VRAM** — `awq_marlin_moe_repack` doubles weight memory during repacking (old + new tensors coexist). For 128-expert MoE models this adds ~7 GB peak per GPU. Patch adds `SGLANG_FORCE_MOE_WNA16=1` env var to bypass Marlin repack and use [MoeWNA16 Triton kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/moe_wna16.py) instead (only works with compressed-tensors format, not native AWQ).
- **Triton attention BF16 precision** — RDNA4 team found that SGLang's Triton attention kernels do BF16 accumulation in online softmax, causing catastrophic precision loss on deep models (60+ layers). [Patch 011](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/patches/011-rdna4-triton-attention-fp32.patch) fixes it with FP32 casts in QK dot product + value accumulation. We use FlashInfer (which does FP32 internally) for most models, but this affects any model forced to Triton attention (e.g. Gemma 4 if head_dim=512 workaround is found). The Qwen3-VL-32B Dense (64 layers) may also be affected.
- **Qwen3.5-28B MoE REAP** — **Working** (patch 009). Required extensive integration: CausalLM wrapper with lm_head/logits_processor/mrope, FusedMoE TP overflow guards, GPTQ calibration with in-memory expert fusion (BF16 source has per-expert weights but HF class expects fused FusedMoE format — calibrating without fusion silently produces garbage), config flattening (text_config fields to top level), FlashInfer architectures None guard. 33 tok/s decode, constant across context lengths.
- **CUDA graphs** — Only bs=1 works. `--cuda-graph-max-bs 1 --disable-custom-all-reduce`.
- **60B+ models** — Coder-Next-REAM (35GB), GLM-4.5-Air-REAP (43GB) don't fit in 48GB.

## Next to Try

- **Qwen3.5-28B MoE REAP** — ✅ Working at 33 tok/s. Enable piecewise CUDA graphs for speed. Extend context beyond 4K.
- **Qwen3-VL-30B MoE** — AutoRound calibration running in background (`/tmp/autoround-qwen3vl-30b.log`). Will produce native AWQ format.
- **Qwen3-VL-32B Dense AWQ** — Working at 24 tok/s, 8K context. Needs full benchmark suite.
- **Coder-30B REAP → AWQ/Marlin** — Currently `auto-round` format. AWQ/Marlin would use faster kernels.
- **REAM Qwen3-Coder-30B** — Prune 128→96 experts for Marlin-optimized coding at 128K.

### Findings from RDNA4 R9700 system

The sister [2x R9700 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference) found:

- **ROOT CAUSE: Triton attention BF16 precision bug** — Online softmax accumulates `e_max`/`e_sum`/`re_scale` in BF16 and `tl.dot()` lacks `out_dtype=tl.float32`. Causes 15% mean error vs FP32 after 128 KV tokens. Compounds catastrophically over 60 layers (Gemma 31B). `--attention-backend torch_native` produces perfect output. [Patch 011](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/patches/011-rdna4-triton-attention-fp32.patch) adds FP32 casts throughout decode + extend kernels.
- **Gemma 31B Dense quality FIXED** — Near-perfect output with FP32 triton attention + Intel AutoRound GPTQ + FP8 KV cache. Currently ~2 tok/s via GPTQ torch fallback; need AWQ conversion or native GPTQ HIP kernels for speed.
- **AutoRound > GPTQ > AWQ for INT4 quality** — Intel AutoRound uses SignSGD (200 iterations) to jointly optimize rounding + clipping, directly minimizing `||WX - W_qX||`. Can export to both GPTQ and AWQ formats. [RedHatAI reports 99.4%+ quality](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) with uniform GPTQ INT4 on CUDA.
- **BF16 precision affects all new architectures** — Both RDNA4 and [Blackwell SM12.x](https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009) hit attention precision issues that Ampere/Hopper tolerate. Fix: FP32 accumulation in online softmax.
- **Qwen3.5-27B at 26 tok/s** on RDNA4 (vs our 13.5 tok/s). Same replication strategy.
- **Community AWQ fails for DeltaNet** — both teams confirmed. GPTQ calibration + CT→AWQ conversion required.
- **DeltaNet layers must stay BF16** — INT4 destroys recurrent state quality. Architectural limit.
- **Coder-Next 80B fits on R9700 (32GB/GPU)** — but not on 3090 (24GB/GPU).

## Quick Start

```bash
# 1. Setup: clone SGLang v0.5.10, apply patches, create conda env
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B AWQ — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ — best throughput
./scripts/launch.sh coder-reap          # Coder-REAP-25B — fastest single-user
./scripts/launch.sh qwen35              # Qwen3.5-27B DeltaNet AWQ

# 3. Test quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334
```

## Prerequisites

- 2x NVIDIA RTX 3090 (24GB GDDR6X each, 48GB total) with NVLink bridge
- NVIDIA drivers (595+) + CUDA 13.x
- Miniforge3/Conda
- ~150GB disk for models

## Model Support (SGLang)

### Agent / coding workloads (single-user, max context)

| Model | Type | Max context | 1-user tok/s | TPOT | Launch | Status |
|-------|------|:----------:|:------------:|:----:|:------:|:------:|
| **Qwen3-30B REAM AWQ** | **MoE (96 experts)** | **262K** | **197** | **5ms** | `launch.sh qwen3-ream` | **Working** |
| Coder-REAP-25B W4A16 | MoE (103 experts) | 131K | 134 | 7ms | `launch.sh coder-reap` | Working |
| **Devstral-24B AWQ Marlin** | **Dense** | **131K** | **87** | **12ms** | `launch.sh devstral` | **Working** |
| **Coder-30B AWQ Marlin** | **MoE (128 experts)** | **16K** | **193** | **5ms** | `launch.sh coder-30b` | **Working** |
| **Qwen3.5-28B MoE REAP** | **DeltaNet+MoE (205 exp)** | **262K** | **33** | **31ms** | `launch.sh qwen35-moe` | **Working** |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | 13.5 | 74ms | `launch.sh qwen35` | Working |
| Qwen3-VL-32B Dense AWQ | Dense (vision+text) | 8K | 24 | 45ms | `launch.sh qwen3-vl-32b` | Working |
| Gemma 4 26B REAP | MoE (103 experts) | — | — | — | — | Blocked (FlashInfer) |

### VRAM context length limits (FP8 KV cache, TP=2, 48GB total)

KV cache is the dominant VRAM constraint. REAM/REAP MoE models have smaller weights, leaving more room for context.

| Model | Wt/GPU | KV/token | Free VRAM | **Max context** |
|-------|:------:|:--------:|:---------:|:---------------:|
| **Qwen3-30B REAM AWQ** | **6.2 GB** | **36 KB** | **16.3 GB** | **262K** |
| Devstral-24B AWQ | 7.0 GB | 80 KB | 15.5 GB | **131K** |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 16.0 GB | **131K** |
| Coder-30B AWQ | 8.0 GB | 36 KB | 14.5 GB | **262K** |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 15.2 GB | **262K** |
| **Qwen3.5-27B AWQ** | **19.0 GB** | 24 KB | **2.2 GB** | **32K** |

Future: [TurboQuant](https://github.com/sgl-project/sglang/issues/21618) (3-bit KV, ICLR 2026) would give ~3x more context, but SGLang integration is WIP/unmerged.

### Batch throughput (multi-user)

| Model | Peak total tok/s | Best conc | Context |
|-------|:----------------:|:--------:|:-------:|
| **Qwen3-30B REAM AWQ** | **1,832** | **@16** | **16K** |
| Devstral-24B AWQ | 1,647 | @32 | 32K |
| Coder-30B AWQ | 1,201 | @32 | 16K |

**Weights:** Community AWQ works for standard architectures (Devstral, Coder-30B) but fails for:
- **Qwen3.5** — community AWQ produces garbage on DeltaNet layers; we calibrate with GPTQ, keep DeltaNet/SSM in BF16
- **Gemma 4** — standard GPTQ only calibrates 1/128 experts; needs forced-routing calibration
- **Devstral** — community AWQ works but needs custom chat template (BOS token fix)

Self-calibrated models use the pipeline in `scripts/quantize/` (GPTQ calibration → CT→AWQ conversion).

## Performance (2x RTX 3090, TP=2, SGLang v0.5.10 + patches)

**Methodology:** `bench_all_unified.py` uses `sglang.bench_serving` for proper TPOT/TTFT measurement.

### Devstral-24B AWQ (up to 131K context)

24B dense transformer. ~7 GB/GPU. FP8 KV cache enables long context.

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 63.4 |
| 1K | 62.4 |
| 4K | 51.9 |
| 8K | 44.0 |
| 16K | 32.8 |
| **32K** | **21.1** |

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 64 |
| 4 | 241 |
| 8 | 476 |
| 16 | 955 |
| **32** | **1,647** |

### Coder-REAP-25B W4A16 (131K context, 103 experts)

25B / 3B active MoE. REAP-pruned (128→103 experts). ~6.5 GB/GPU. Uses `auto-round` quantization.

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 134 |
| 1K | 126 |
| 4K | 98 |
| 8K | 63 |
| 16K | 57 |
| **32K** | **46** |

### Coder-30B MoE AWQ (16K context, 128 experts)

30B / 3B active MoE. ~8 GB/GPU. Best throughput scaling.

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 42.9 |
| 1K | 41.0 |
| 4K | 37.7 |
| 8K | 33.8 |
| **16K** | **27.4** |

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 42 |
| 4 | 146 |
| 8 | 308 |
| 16 | 607 |
| **32** | **1,201** |

### Qwen3.5-27B AWQ DeltaNet (32K context)

27B DeltaNet hybrid. ~19 GB/GPU (replicated). Decode speed is constant regardless of context length (no KV attention scaling).

| Context Length | tok/s | TTFT | TPOT |
|:--------------:|:-----:|:----:|:----:|
| 128 | 13.3 | 175ms | 75ms |
| 512 | 13.5 | 511ms | 74ms |
| 1K | 13.5 | 986ms | 74ms |
| 4K | 13.6 | 3.9s | 74ms |
| 16K | 12.9 | 5.0s | 78ms |

Previously 7 tok/s — patches 005-007 (FP8 KV, BF16 AWQ, DeltaNet kernel tuning) improved to 13.5 tok/s.

### Qwen3.5-28B MoE REAP (4K context, 205 experts)

28B / 3B active DeltaNet+MoE hybrid. REAP-pruned (256→205 experts). ~8 GB/GPU. Constant decode speed (DeltaNet advantage: no KV attention scaling).

| Context Length | tok/s | TTFT | TPOT |
|:--------------:|:-----:|:----:|:----:|
| 128 | 33 | 85ms | 31ms |
| 512 | 33 | 82ms | 30ms |
| 1K | 33 | 80ms | 31ms |
| **4K** | **33** | **147ms** | **31ms** |

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 33 |
| 4 | 65 |
| 8 | 60 |

First working INT4 quantization of Qwen3.5 DeltaNet+MoE. Required: GPTQ calibration with in-memory expert fusion (BF16 source has per-expert weights, HF model class expects fused FusedMoE format), DeltaNet/gate/vision layers excluded from INT4, custom CausalLM wrapper with logits processor + mrope handling (patch 009).

### Qwen3-30B-Instruct REAM AWQ Marlin (262K context, 96 experts)

30B / 3B active MoE. REAM-merged (128→96 experts). ~6.2 GB/GPU. **Fastest model on the rig.**

| Context Length | tok/s | TTFT |
|:--------------:|:-----:|:----:|
| 128 | **179** | 0.3s |
| 512 | 168 | 0.6s |
| 1K | 176 | 0.6s |
| 4K | 140 | 0.9s |
| 8K | 116 | 1.1s |
| **16K** | **71** | **1.8s** |

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 191 |
| 4 | 600 |
| 8 | 1,074 |
| **16** | **1,832** |
| 32 | 1,301 |

197 tok/s single-user decode (5ms TPOT). 1,832 tok/s peak batch throughput at 16 concurrent — fastest on the rig, beating Devstral-24B (1,647). Self-calibrated GPTQ with all-expert routing + CT→AWQ conversion for Marlin kernels. Required patching llmcompressor for fused `Qwen3MoeExperts` (transformers 5.x) by creating per-expert `nn.Linear` wrappers during calibration.

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

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | v0.5.10 + 12 patches | editable install from source |
| PyTorch | 2.9.1+cu128 | CUDA toolkit 12.8 |
| CUDA | 13.2 | driver 595.58 |
| NCCL | 2.27.5 | P2P over NVLink |
| FlashInfer | 0.6.7.post3 | JIT cubins for sm_86 |
| transformers | 5.5.3 | Gemma4/Qwen3.5 support |

## Patches

8 patches on top of SGLang v0.5.10. Apply in order:

1. **001-upstream-sync** (3,000 LOC) — Upstream cherry-picks: Gemma 4, Qwen3.5/3-Next, Triton attention, pool_configurator
2. **002-nvidia-model-fixes** (923 LOC) — Marlin shape fallback, DeltaNet TP replication, Gemma4 config fixes
3. **003-deltanet-triton-dtype-fix** (51 LOC) — DeltaNet conv_state bf16/fp16 cast fix
4. **004-gemma4-causal-lm-fix** (19 LOC) — CausalLM multimodal detection bypass
5. **005-ampere-fp8-triton-fallback** (59 LOC) — FP8 KV cache on sm_86 (PyTorch fallback for `fp8e4nv`)
6. **006-awq-bf16-activation-support** (15 LOC) — BF16 activations with AWQ dequant
7. **007-ampere-deltanet-kernel-tuning** (48 LOC) — DeltaNet BV=64 tuning for sm_86 (1.57x kernel speedup)
8. **008-awq-moe-wna16-fallback** (64 LOC) — `SGLANG_FORCE_MOE_WNA16=1` env var to bypass Marlin MoE repack (saves ~7 GB peak VRAM for 128-expert models)

## Key Findings

1. **AWQ Marlin is the fast path on Ampere** — compressed-tensors auto-promotes to Marlin on sm_80+. FP32 accumulation avoids FP16 overflow.
2. **DeltaNet replication mandatory for TP=2** — FP16 rounding accumulation through recurrent state destroys quality. Full model per GPU.
3. **FP8 KV cache works on Ampere via fallback** — patch 005 routes `fp8e4nv` to PyTorch. FlashInfer handles FP8 KV for head_dim ≤ 256.
4. **DeltaNet decode is framework-overhead-bound** — 312 kernel launches/token at ~316us each. Raw compute 44ms, actual 143ms (pre-patches). Kernel fusion or CUDA graphs would close the gap.
5. **REAM/REAP MoE unlocks longer context** — smaller weights = more VRAM for KV cache. Critical for 48GB.

## MoE Quantization Lessons

Standard GPTQ/AWQ **fails** for MoE models (MoEQuant, ICML 2025):

1. **Inter-expert imbalance**: Router unevenly distributes calibration data — rare experts get zero/garbage calibration.
2. **DeltaNet/SSM sensitivity**: Recurrent state `S(t) = g*S(t-1) + delta` accumulates INT4 noise. DeltaNet layers MUST stay BF16.

**Solutions**: Expert-balanced sampling (MoEQuant EBSS, GPTQModel FailSafe), skip recurrent layers. See `scripts/quantize/`.

## Quantization

Self-calibrated AWQ models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_llmcompressor.py
python scripts/quantize/convert_qwen35_ct_to_awq.py
```

See [rules-for-agents.md](rules-for-agents.md) for full pipeline and [REAM.md](scripts/quantize/REAM.md) for MoE expert pruning.

## Qwen3.5-27B Technical Details

Hybrid DeltaNet (linear attention) + full attention. TP=2 requires replicating all layers to avoid FP16 precision errors accumulating through DeltaNet recurrence.

**Root cause:** TP RowParallelLinear splits matmul: `W_0@x_0 + W_1@x_1` differs from `W@x` by ~1 ULP in FP16. DeltaNet's `S(t) = g*S(t-1) + delta` compounds this across 48 layers x N tokens.

**Fix:** Replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`.

VRAM per GPU: ~19 GB model (replicated) + 1.27 GB DeltaNet state + 0.92 GB KV cache (FP8) = ~21 GB. Only 32K context fits.

### Triton kernel tuning (patch 007)

DeltaNet decode kernel defaults (`BV=32`, `num_warps=1`) under-utilize RTX 3090. Our sweep found BV=64 gives 1.57x:

| Config | BV | ms/layer | Speedup |
|--------|:--:|:--------:|:-------:|
| baseline | 32 | 0.018 | 1.00x |
| **BV64-w1** | **64** | **0.011** | **1.57x** |

### Pipeline bottleneck analysis

| Operation | ms/model | % |
|-----------|:--------:|:-:|
| MLP forward | 19.9 | 45% |
| Recurrent update | 8.3 | 19% |
| QKV projection | 7.9 | 18% |
| Output projection | 2.9 | 7% |
| RMSNorm + gating | 2.1 | 5% |
| Conv1d | 1.3 | 3% |
| **Theoretical** | **44.1** | **22.7 tok/s** |
| **Actual** | **74** | **13.5 tok/s** |

MoE kernel configs generated for RTX 3090 (Triton 3.5.1): `E=128,N=768` (Coder-30B, Qwen3-VL-30B), `E=128,N=704` (Gemma 4), `E=103,N=768` (Coder-REAP). Auto-loaded by `device_name=NVIDIA_GeForce_RTX_3090`.

## Gemma 4 Notes

Blocked on SGLang. RDNA4 team working on mixed precision + softmax patch for accuracy. Cross-team findings:

1. **FP16 overflow at layer 2** (hidden_size=5376). Fix: `--dtype bfloat16` + patch 006.
2. **CT→AWQ conversion quality poor** — cosine similarity 0.845 on q_proj. Models generate garbage.
3. **Missing chat template** — embed jinja into `tokenizer_config.json`.
4. **`num_experts` is None for Dense** — fix: `getattr(config, "num_experts", 0) or 0`.

## FlashInfer head_dim support

| head_dim | FlashInfer (sm_86) | Models |
|:--------:|:------------------:|--------|
| 64-256 | Supported | Qwen, Devstral |
| **512** | **Not supported** | **Gemma 4** (blocked) |

Possible fixes: SDPA fallback, TRTLLM FMHA path, [FFPA kernels](https://github.com/DefTruth/ffpa-attn-mma), or llama.cpp (80-110 tok/s for Gemma 4).

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
RAM:    96 GB (92 GB usable, ~4 GB reserved by iGPU)
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24GB GDDR6X each)
GPU interconnect: NVLink (NV4, 4 lanes x 14 GB/s = 56 GB/s bidirectional)
Driver: 595.58.03
CUDA:   13.2 (PyTorch uses cu128 toolkit)
Python: 3.12
```

## Structure

```
patches/                           # SGLang v0.5.10 patches (7 total)
benchmarks/                        # Benchmark results (per-model directories)
scripts/
  launch.sh                       #   Unified model launcher (launch.sh <model>)
  common.sh                       #   Shared NVIDIA environment setup
  setup.sh                        #   Full setup (conda, SGLang install)
  bench/                          #   Benchmark scripts
  eval/                           #   Quality evaluation + warmup
  quantize/                       #   Quantization pipeline (GPTQ -> CT -> AWQ)
  test/                           #   Kernel microbenchmarks + profiling
components/sglang/                 # SGLang v0.5.10 + patches (cloned by setup.sh)
```
