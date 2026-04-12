# SGLang Inference: 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 12.8.

## Known Issues

- **Gemma 4 26B/31B** — Multimodal processor registry missing in v0.5.10 base. Needs custom calibrated checkpoint.
- **Qwen3.5-27B** — Community AWQ quantizes DeltaNet layers causing Triton bf16/fp16 mismatch. Needs self-calibrated checkpoint (DeltaNet layers kept BF16). Calibration pipeline ready.
- **CUDA graphs** — Disabled. Graph capture OOMs with custom all-reduce on 24GB GPUs with TP=2.

## Quick Start

```bash
# 1. Setup: clone SGLang v0.5.10, apply patches, create conda env
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B AWQ — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE AWQ — best throughput

# 3. Test quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334
```

## Prerequisites

- 2x NVIDIA RTX 3090 (24GB GDDR6X each, 48GB total)
- NVIDIA drivers (550+) + CUDA toolkit 12.x
- Miniforge3/Conda
- ~150GB disk for models

## Model Support (SGLang)

### Agent / coding workloads (single-user, max context)

| Model | Type | Max context | 1-user tok/s | TPOT | Launch | Status |
|-------|------|:----------:|:------------:|:----:|:------:|:------:|
| Devstral-24B AWQ | Dense | 32K | 63 | 16ms | `launch.sh devstral` | Working |
| Coder-30B AWQ | MoE (128 experts) | 16K | 43 | 23ms | `launch.sh coder-30b` | Working |
| Coder-Next-REAM-60B AWQ | MoE (384 experts) | 4K | — | — | `launch.sh coder-next-ream` | Not yet tested |
| GLM-4.5-Air-REAP AWQ | MoE (96 experts) | 4K | — | — | `launch.sh glm45-air` | Not yet tested |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | — | — | `launch.sh qwen35` | Needs calibration |
| Gemma 4 26B AWQ | MoE (128 experts) | 4K | — | — | `launch.sh gemma4` | Blocked |
| Gemma 4 31B AWQ | Dense | 4K | — | — | `launch.sh gemma4-31b` | Blocked |

All numbers measured with `bench_all_unified.py` (tok/s = completion tokens / elapsed time, single user).

### Batch throughput (multi-user)

| Model | Peak total tok/s | Best conc | Context | Status |
|-------|:----------------:|:--------:|:-------:|:------:|
| Devstral-24B AWQ | 1,647 | @32 | 32K | Working |
| Coder-30B AWQ | 1,201 | @32 | 16K | Working |

### Models that don't fit (48GB limit)

| Model | Params | Weight size | Why |
|-------|--------|:-----------:|-----|
| Coder-Next-REAM-60B | 60B MoE (384 experts) | 35 GB | ~17.5GB/GPU, OOM on init overhead |
| GLM-4.5-Air-REAP-82B | 82B MoE (96 experts) | 43 GB | ~21.5GB/GPU, no room for KV cache |
| Qwen3-Coder-Next-80B | 80B MoE (512 experts) | ~44 GB | Exceeds 48GB total |

## Performance (2x RTX 3090, TP=2, SGLang v0.5.10 + patches)

**Methodology:** All numbers use `bench_all_unified.py` which runs single-user context sweeps and concurrent throughput sweeps. See [benchmarks/README.md](benchmarks/README.md) for full methodology.

### Devstral-24B AWQ (32K context)

24B dense transformer. ~14 GB/GPU. Best single-user speed.

![Devstral context scaling](benchmarks/devstral-24b-awq/context_vs_toks.png)

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 63.4 |
| 1K | 62.4 |
| 4K | 51.9 |
| 8K | 44.0 |
| 16K | 32.8 |
| **32K** | **21.1** |

![Devstral concurrency](benchmarks/devstral-24b-awq/concurrency_vs_toks.png)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 64 |
| 4 | 241 |
| 8 | 476 |
| 16 | 955 |
| **32** | **1,647** |

### Coder-30B MoE AWQ (16K context, 128 experts)

30B total / 3B active MoE. ~16 GB/GPU. Best throughput scaling.

![Coder-30B context scaling](benchmarks/coder-30b-awq/context_vs_toks.png)

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 42.9 |
| 1K | 41.0 |
| 4K | 37.7 |
| 8K | 33.8 |
| **16K** | **27.4** |

![Coder-30B concurrency](benchmarks/coder-30b-awq/concurrency_vs_toks.png)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 42 |
| 4 | 146 |
| 8 | 308 |
| 16 | 607 |
| **32** | **1,201** |

## Patches

2 patches on top of SGLang v0.5.10. Apply in order:

### 001-upstream-sync (3,000 LOC)
Cherry-picks from upstream main for model support. No NVIDIA-specific changes.
- Gemma 4 model + fused ops + config transformer
- Qwen3.5/Qwen3-Next model updates
- Triton attention backend + prefill improvements
- pool_configurator.py (MemoryPoolConfig refactor)

### 002-nvidia-model-fixes (923 LOC)
NVIDIA-specific fixes and model compatibility.
- MemoryPoolConfig: runtime import (not TYPE_CHECKING only)
- Marlin shape fallback: torch dequant for layers where dim not divisible by 64
- sharded_weight_loader: override_tp_rank for replicated DeltaNet layers
- pool_configurator: is_dflash guard
- Gemma4: text_config unwrap, top_k_experts config lookup
- Qwen3.5: mamba cache params, DeltaNet TP replication
- Devstral/LLaVA: chat template BOS fix, text-only VLM warmup

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
cd components/sglang && git checkout v0.5.10
git apply ../../patches/001-upstream-sync.patch
git apply ../../patches/002-nvidia-model-fixes.patch
cd python && pip install -e ".[srt]"
```

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | v0.5.10 + 2 patches | editable install from source |
| PyTorch | 2.9.1+cu128 | CUDA 12.8 |
| CUDA | 12.8 | driver 595+ |
| NCCL | 2.27.5 | P2P over PCIe |
| transformers | 5.5.3 | Gemma4 support |

## Quantization

Self-calibrated AWQ models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_llmcompressor.py
python scripts/quantize/convert_qwen35_ct_to_awq.py
```

See [rules-for-agents.md](rules-for-agents.md) for full quantization pipeline and rules.

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
RAM:    92 GB
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24GB GDDR6X each)
Driver: 595.58.03
CUDA:   12.8
Python: 3.12
```

## Structure

```
patches/                           # SGLang v0.5.10 patches
  001-upstream-sync.patch         #   Upstream cherry-picks (Gemma4, Qwen3.5, etc.)
  002-nvidia-model-fixes.patch    #   NVIDIA-specific fixes
benchmarks/                        # Benchmark results (per-model directories)
  {model}/results.json            #   Structured data from bench_all_unified.py
  baselines.json                  #   Regression test baselines
scripts/
  launch.sh                       #   Unified model launcher (launch.sh <model>)
  common.sh                       #   Shared NVIDIA environment setup
  setup.sh                        #   Full setup (conda, SGLang install)
  bench/                          #   Benchmark scripts
  eval/                           #   Quality evaluation + warmup
  quantize/                       #   Quantization pipeline (GPTQ → CT → AWQ)
components/sglang/                 # SGLang v0.5.10 + patches (cloned by setup.sh)
```
