# Rules for AI Agents — 2x RTX 3090

## Inference Engine

All inference MUST use SGLang. No vLLM, no llama.cpp unless explicitly for comparison benchmarks.

## Hardware
- 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere, sm_86)
- 24 GB GDDR6X each, 48 GB total
- NCCL over PCIe for TP=2

## Server Launch

```bash
source scripts/common.sh
activate_conda
setup_nvidia_env
./scripts/launch.sh <model>
```

Always source `common.sh` + `activate_conda` + `setup_nvidia_env` before launching.

### Launch flags
- `--disable-cuda-graph --disable-custom-all-reduce` — required, CUDA graph capture OOMs with TP=2 on 24GB GPUs
- `--quantization compressed-tensors` — for cyankiwi community checkpoints
- `--quantization awq` — for our self-calibrated AWQ checkpoints (auto-promotes to `awq_marlin`)

## VRAM Budget

48GB total (2x 24GB). Model weights + KV cache must fit.

| Model | Weight VRAM | Max practical context |
|-------|-------------|----------------------|
| Devstral-24B AWQ | ~14 GB | 32K |
| Coder-30B MoE AWQ | ~16 GB | 16K |
| Qwen3.5-27B AWQ | ~15 GB | 32K |
| Coder-Next-REAM-60B AWQ | ~33 GB | 4K (very tight) |
| GLM-4.5-Air-REAP-82B AWQ | ~34 GB | 4K (very tight) |
| Coder-Next-80B AWQ | ~44 GB | DOES NOT FIT |

## Quantization Pipeline

All models use **AWQ 4-bit** format. The pipeline:

```
BF16 model → GPTQ calibration (llmcompressor) → compressed-tensors → CT→AWQ conversion → native AWQ
```

### CRITICAL: Use a clean conda env for calibration

**llmcompressor MUST run in a separate conda env from sglang.** The two have conflicting
dependencies (transformers, compressed-tensors, torch versions). Mixing them breaks both.

```bash
# Create clean quant env (one-time)
conda create -n quant python=3.12 -y
conda activate quant
pip install llmcompressor transformers compressed-tensors accelerate datasets
```

- Calibration runs on **CPU only** (`CUDA_VISIBLE_DEVICES=""`): the BF16 model is too large for 48GB VRAM
- CPU calibration takes ~6 hours for a 27B model (memory-mapped from safetensors)
- The `sglang` conda env is for inference only — never install llmcompressor into it

### Step 1: GPTQ calibration (quant env, CPU)
```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_llmcompressor.py
```
- Output: compressed-tensors format (`.weight_packed` + `.weight_scale` per layer)
- 256 samples × 512 tokens for dense models
- 512+ samples for MoE models (expert balance)

### Step 2: CT → AWQ conversion (quant env)
```bash
python scripts/quantize/convert_qwen35_ct_to_awq.py
```
- Repacks weights from CT sequential to AWQ interleaved format
- Transposes weight layout for Marlin kernel compatibility
- Clamp scales to [-65504, 65504] before FP16 cast (prevents inf overflow)

### Step 3: Inference (sglang env)
```bash
conda activate sglang
MODEL=~/AI/models/Qwen3.5-27B-AWQ-4bit-calibrated ./scripts/launch.sh qwen35
```

### DeltaNet/Mamba/SSM layers — DO NOT quantize to INT4
Models with recurrent state accumulate quantization error: `S(t) = gating * S(t-1) + delta`.
- **Qwen3.5-27B**: Exclude `in_proj_a`, `in_proj_b` (dim 48, DeltaNet gates) from GPTQ
- **Coder-Next**: DeltaNet + attention layers kept BF16
- Community AWQ checkpoints that quantize these layers produce garbage output or
  Triton kernel dtype mismatches (bf16 vs fp16 in conv_state branches)

### MoE calibration — CRITICAL
Standard GPTQ fails for MoE due to expert routing imbalance:
- Use **at least 512 calibration samples** with sequence length ≥1024
- Verify all experts receive calibration data — check scales for inf/nan/zero
- For fused expert Parameters: monkey-patch to per-expert nn.Linear before calibration
- Consider GPTQModel with MoE.Routing FailSafe mode

### Chat templates — ALWAYS verify
SGLang reads chat templates from the tokenizer, NOT from standalone jinja files.
Many HuggingFace models ship `chat_template.jinja` as a separate file that SGLang ignores.

**After downloading or calibrating any model, verify:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
assert tok.chat_template is not None, "Missing chat template!"
```

If `chat_template` is None:
1. Check for `chat_template.jinja` in the model directory
2. Embed its contents into `tokenizer_config.json` as the `chat_template` field
3. Or pass `--chat-template path/to/template.jinja` to SGLang launch

Without a chat template, SGLang falls back to a generic format that produces
wrong outputs (no system prompt handling, wrong special tokens, etc.).

### AWQ checkpoint format
- Marlin requires: output dim divisible by 64, input dim divisible by 128
- Layers that don't meet alignment fall back to torch dequant (our 002 patch)
- Expert naming: `experts.{id}.{proj}.{suffix}` (SGLang format)
- quant_method: "awq", version: "gemm", zero_point: true, group_size: 128

## Benchmarking

- Always measure **TPOT** with `sglang.bench_serving` — never wall-clock time
- Concurrency sweep: 1/2/4/8/16/32
- Context sweep: powers of 2 from 128 to model max
- Save to `benchmarks/{model}/results.json`
- Regenerate charts: `python scripts/bench/generate_charts.py`
- Run regression test before committing: `./scripts/bench/bench_regression.sh <model>`
- Regression threshold: >10% deviation triggers alert
