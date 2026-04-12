# SGLang Inference: 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 12.8.

## Known Issues

- _No known issues yet — initial setup, benchmarks pending._

## Quick Start

```bash
# 1. Setup: create conda env, install SGLang
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B AWQ — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ — best throughput
./scripts/launch.sh gemma4              # Gemma 4 26B MoE AWQ
./scripts/launch.sh qwen35              # Qwen3.5-27B DeltaNet AWQ

# 3. Test quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334
```

## Prerequisites

- 2x NVIDIA RTX 3090 (24GB GDDR6X each, 48GB total)
- NVIDIA drivers (550+) + CUDA toolkit 12.x
- Miniforge3/Conda
- ~80GB disk for models

## Model Support (SGLang)

All models run on SGLang with AWQ Marlin kernels (fused dequant+matmul, optimized for sm_80+).

### Agent / coding workloads (single-user, max context)

Primary use case: agent and coding workflows with maximum context at fast decode speeds.

| Model | Type | Max context | 1-user tok/s | TPOT | Launch | Status |
|-------|------|:----------:|:------------:|:----:|:------:|:------:|
| Devstral-24B AWQ | Dense | 32K | — | — | `launch.sh devstral` | Not yet tested |
| Coder-30B AWQ | MoE (128 experts) | 16K | — | — | `launch.sh coder-30b` | Not yet tested |
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | — | — | `launch.sh gemma4` | Not yet tested |
| Gemma 4 31B AWQ | Dense | 4K | — | — | `launch.sh gemma4-31b` | Not yet tested |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | — | — | `launch.sh qwen35` | Not yet tested |

All numbers measured with `sglang.bench_serving` (TPOT = Time Per Output Token, decode only).

### Batch throughput (multi-user)

| Model | Peak total tok/s | Max conc | Context | Status |
|-------|:----------------:|:--------:|:-------:|:------:|
| — | — | — | — | Not yet tested |

### Models that don't fit (48GB limit)

| Model | Params | Why |
|-------|--------|-----|
| Qwen3-Coder-Next-80B | 80B MoE | ~44GB weights alone, no room for KV cache |
| GLM-4.5-Air-82B | 82B MoE | Same — exceeds VRAM |

**Quantization:** AWQ INT4 with Marlin kernel backend (`--quantization awq_marlin`). SGLang auto-promotes standard AWQ checkpoints to Marlin at load time. Standard community AWQ models (group_size=128, 4-bit) work directly — Marlin repacking happens at runtime.

**DeltaNet hybrid models (Qwen3.5):** DeltaNet/attention layers kept in BF16 — INT4 quantization destroys quality due to recurrent state error accumulation.

**MoE quantization:** Standard GPTQ under-calibrates rare experts (inter-expert imbalance). Use expert-balanced calibration. See `rules-for-agents.md`.

**FP8 KV cache:** Enabled by default (`--kv-cache-dtype fp8_e4m3`). Works on Ampere via type casting. Saves ~50% KV memory vs FP16.

## Performance (2x RTX 3090, TP=2, SGLang)

_Not yet benchmarked. Results will be populated after running benchmarks._

**Methodology:** All numbers use `sglang.bench_serving` which measures TPOT (decode latency per token) and TTFT (prefill latency) separately. See [benchmarks/README.md](benchmarks/README.md) for full methodology. Regression tests: `./scripts/bench/bench_regression.sh <model>`.

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
# Clone SGLang
git clone --branch v0.5.10 --depth 1 https://github.com/sgl-project/sglang.git components/sglang

# Create conda env, install dependencies
conda create -n sglang python=3.12
conda activate sglang
pip install -e components/sglang/python[srt]
pip install transformers>=5.0 gguf
```

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | v0.5.10 | stock, no patches |
| PyTorch | 2.8.0+cu128 | CUDA 12.8 |
| CUDA | 12.8 | driver 595+ |
| NCCL | system | P2P over PCIe |

## Architecture

- **Quantization**: AWQ Marlin (fused dequant+matmul, async memory access, L2 cache optimized)
- **Attention**: FlashInfer (default SGLang backend for NVIDIA)
- **Multi-GPU**: NCCL TP=2 over PCIe, P2P enabled
- **CUDA graphs**: Enabled (decode acceleration)
- **KV cache**: FP8 E4M3 (saves VRAM for larger context windows)
- **Overlap schedule**: Enabled (prefill/decode overlap)

## MoE Quantization Notes

Standard GPTQ/AWQ **fails** for MoE models (MoEQuant, ICML 2025). Two critical issues:

1. **Inter-expert imbalance**: Router unevenly distributes calibration data — rare experts get
   zero/garbage calibration.
2. **DeltaNet/SSM sensitivity**: Recurrent state `S(t) = g*S(t-1) + delta` accumulates INT4
   noise across tokens. DeltaNet layers MUST stay BF16.

**Solutions**: Expert-balanced sampling (MoEQuant EBSS, GPTQModel FailSafe), skip recurrent layers.
See [rules-for-agents.md](rules-for-agents.md) for full quantization rules.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
CPU:    AMD (system CPU)
RAM:    64 GB DDR5
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24GB GDDR6X each)
Driver: 595.58.03
CUDA:   12.8
Python: 3.12
```

## Structure

```
benchmarks/                        # Benchmark results (per-model directories)
  {model}/results.json            #   Structured data from bench_all_unified.py
  baselines.json                  #   Regression test baselines
scripts/
  launch.sh                       #   Unified model launcher (launch.sh <model>)
  common.sh                       #   Shared NVIDIA environment setup
  setup.sh                        #   Full setup (conda, SGLang install)
  bench/                          #   Benchmark scripts
  eval/                           #   Quality evaluation + warmup
components/sglang/                 # SGLang v0.5.10 (cloned by setup.sh)
```
