# 2x RTX 3090 SGLang Inference

SGLang inference server for **2x NVIDIA RTX 3090** (GA102-300-A1, 24GB each, 48GB total) with tensor parallelism and AWQ_Marlin kernels.

## Quick Start

```bash
# 1. Setup
./scripts/setup.sh

# 2. Launch a model
./scripts/launch.sh devstral

# 3. Evaluate quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Devstral-24B AWQ" --port 23334 \
    --output benchmarks/devstral-24b-awq/results.json
```

## Prerequisites

- 2x NVIDIA RTX 3090 (24GB GDDR6X each)
- NVIDIA drivers (550+) + CUDA toolkit
- Conda (miniforge3 recommended)
- ~50GB disk for models

## Model Support

| Model | Params | Quant | Context | VRAM Est. | Status |
|-------|--------|-------|---------|-----------|--------|
| Devstral-24B | 24B dense | AWQ Marlin | 32K | ~13 GB | Ready |
| Qwen3-Coder-30B | 30B MoE | AWQ Marlin | 16K | ~17 GB | Ready |
| Gemma 4 26B | 26B MoE | AWQ Marlin | 4K | ~14 GB | Ready |
| Gemma 4 31B | 31B dense | AWQ Marlin | 4K | ~17 GB | Ready |
| Qwen3.5-27B | 27B hybrid | AWQ Marlin | 32K | ~15 GB | Ready |

### Models That Don't Fit (48GB limit)

| Model | Params | Why |
|-------|--------|-----|
| Qwen3-Coder-Next-80B | 80B MoE | ~44GB weights alone, no room for KV cache |
| GLM-4.5-Air-82B | 82B MoE | Same — exceeds VRAM |

## NVIDIA vs AMD Differences

This is the NVIDIA counterpart to the [AMD RDNA4 inference repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Key differences:

| | NVIDIA (3090) | AMD (R9700) |
|---|---|---|
| SGLang | AWQ_Marlin kernels, minimal patches | 5 patches (~5K LOC) |
| Attention | FlashInfer (default) | Triton (forced) |
| CUDA graphs | Work natively | Disabled |
| All-reduce | NCCL native | Custom disabled |
| VRAM | 48GB (2x24) | 64GB (2x32) |
| Setup | `pip install` | Build Triton, sgl_kernel, HIP GEMV |

## Launch Options

```bash
# Basic
./scripts/launch.sh devstral

# Custom context and port
./scripts/launch.sh devstral --context-length 16384 --port 8000

# Custom model path
MODEL=/path/to/weights ./scripts/launch.sh coder-30b

# All options
./scripts/launch.sh <model> \
    --context-length 32768 \
    --port 23334 \
    --mem-fraction 0.85 \
    --max-running 32 \
    --decode-steps 4 \
    --chunked-prefill 4096 \
    --watchdog 600
```

## Performance Results

_Not yet benchmarked. Run benchmarks and update._

## Directory Structure

```
scripts/
  launch.sh              # Unified model launcher
  common.sh              # Shared NVIDIA environment
  setup.sh               # Full setup
  bench/                 # Benchmark scripts
  eval/                  # Quality evaluation
benchmarks/              # Performance results
```

## Test System

```
GPU: 2x NVIDIA RTX 3090 (GA102-300-A1, 24GB GDDR6X each)
VRAM: 48 GB total
SGLang: v0.5.10 (stock)
```
