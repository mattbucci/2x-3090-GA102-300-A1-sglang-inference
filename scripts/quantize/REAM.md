# REAM: Expert Merging for MoE Models

REAM (Router-weighted Expert Activation Merging) compresses MoE models by merging groups of experts instead of dropping them. Reduces expert count by 25% with minimal quality loss.

- Paper: [arxiv.org/abs/2604.04356](https://arxiv.org/abs/2604.04356)
- Code: [github.com/SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream)

## Why REAM?

With 48GB VRAM (2x RTX 3090), the full Qwen3-Coder-30B (128 experts, ~16GB weights) limits context to 16K. REAM'd to 96 experts (~12GB weights) enables 128K+ context.

REAM merges experts rather than pruning (REAP), preserving more of the original model's capability. Quality retention is >=94% on benchmarks.

## Requirements

- **RAM**: ~60GB for Qwen3-30B BF16 model (we have 92GB)
- **GPU**: Not required — runs on CPU. GPU makes saliency computation faster but is optional.
- **Disk**: ~120GB (source BF16 + output BF16)
- **Time**: Several hours on CPU

## Setup

```bash
# Clone REAM (includes pre-tokenized calibration data for qwen3)
git clone https://github.com/SamsungSAILMontreal/ream.git
cd ream

# Create a clean env (REAM needs vllm which conflicts with sglang)
conda create -n ream python=3.12 -y
conda activate ream
pip install -r requirements.txt
```

## Step 1: REAM merge (CPU)

Merge 128 experts → 96 experts (~25% reduction):

```bash
# For coding workloads, use code-heavy calibration mix (0.0 c4, 0.3 math, 0.7 code)
CUDA_VISIBLE_DEVICES="" python merge.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --merge_size 96 \
    --saliency reap \
    --merging logits+weights \
    --grouping ream \
    --dataset "c4+math+code" \
    --mix_ratio "0.0,0.3,0.7" \
    --save_path ~/AI/models/Qwen3-Coder-30B-REAM-BF16

# For general workloads, use balanced mix
CUDA_VISIBLE_DEVICES="" python merge.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --merge_size 96 \
    --saliency reap \
    --merging logits+weights \
    --grouping ream \
    --dataset "c4+math+code" \
    --mix_ratio "0.34,0.33,0.33" \
    --save_path ~/AI/models/Qwen3-30B-REAM-BF16
```

Output: BF16 safetensors model with 96 experts per MoE layer.

## Step 2: Quantize to AWQ (our pipeline)

```bash
# Switch to quant env
conda activate quant

# Drop caches first
echo 3 | sudo tee /proc/sys/vm/drop_caches

# GPTQ calibration
CUDA_VISIBLE_DEVICES="" MODELS_DIR=~/AI/models \
    python scripts/quantize/quantize_moe_llmcompressor.py \
    --model ~/AI/models/Qwen3-Coder-30B-REAM-BF16 \
    --output ~/AI/models/Qwen3-Coder-30B-REAM-CT

# CT → AWQ conversion
python scripts/quantize/convert_moe_ct_to_awq.py \
    --src ~/AI/models/Qwen3-Coder-30B-REAM-CT \
    --dst ~/AI/models/Qwen3-Coder-30B-REAM-AWQ
```

## Step 3: Run inference

```bash
conda activate sglang
MODEL=~/AI/models/Qwen3-Coder-30B-REAM-AWQ ./scripts/launch.sh coder-reap
```

## Pre-made REAM models

| Model | Source | Experts | Params |
|-------|--------|:-------:|:------:|
| [Qwen3-30B-A3B-Instruct-2507-REAM](https://huggingface.co/SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM) | SamsungSAIL | 96 | 23B |
| [Qwen3-Coder-Next-REAM](https://huggingface.co/bknyaz/Qwen3-Coder-Next-REAM) | bknyaz | 384 | 60B |

No REAM'd Qwen3-Coder-30B exists yet — we'd need to create it ourselves.

## Notes

- REAM calibration data is pre-tokenized in the repo (`data/*.pt` files)
- The `--mix_ratio` controls c4:math:code ratio — for coding use `0.0,0.3,0.7`
- `CUDA_VISIBLE_DEVICES=""` forces CPU-only mode (slower but no GPU needed)
- For models with MTP layers (Qwen3), additional safetensor renaming may be needed
- REAM vs REAP: REAM merges experts (preserves function), REAP prunes (drops experts)
