# MoE Expert Compression: REAM & REAP

Two methods to shrink MoE models by reducing expert count. Both run on CPU with ~60GB RAM.

| Method | What it does | Quality | Code |
|--------|-------------|---------|------|
| **REAM** | Merges expert groups | >=94% retained | [SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream) |
| **REAP** | Prunes low-impact experts | Better on generative tasks | [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) |

## Why compress?

With 48GB VRAM (2x RTX 3090), the full Qwen3-Coder-30B (128 experts, ~16GB AWQ weights) limits context to 16K. Compressed to 96-103 experts (~12GB AWQ weights) enables 128K+ context.

## Candidate models

### Pure MoE — fits 48GB after compression + AWQ INT4, easiest to run

| Model | Params | Experts | After | AWQ est. | Method | Status |
|-------|--------|:-------:|:-----:|:--------:|--------|--------|
| Qwen3-Coder-30B | 30B | 128 | 96 (23B) | ~12 GB | REAM/REAP | Need to compress |
| Qwen3-30B-Instruct-2507 | 30B | 128 | 96 (23B) | ~12 GB | REAM | [Pre-made (SamsungSAIL)](https://huggingface.co/SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM) |
| Qwen3-30B-A3B-Base | 30B | 128 | 96 (23B) | ~12 GB | REAM/REAP | Need to compress |
| Gemma 4 26B MoE | 26B | 128 | 103 (21B) | ~10 GB | REAP | [Pre-made (0xSero)](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) |
| Qwen3-Coder-REAP-25B | 25B | 103 | — | ~13 GB | REAP | [Pre-made (cerebras)](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B), running at 134 tok/s |

### DeltaNet hybrid MoE — fits but needs BF16 DeltaNet layers in calibration

| Model | Params | Experts | After | AWQ est. | Notes |
|-------|--------|:-------:|:-----:|:--------:|-------|
| Qwen3.5-27B | 27B | dense+DeltaNet | — | ~15 GB | Calibrating now, exclude DeltaNet from INT4 |
| Qwen3.5-35B-A3B | 35B | 256 | 192 (27B) | ~14 GB | DeltaNet hybrid, needs same treatment |

### Too large even after compression

| Model | Params | After | Why |
|-------|--------|:-----:|-----|
| Qwen3-Coder-Next (80B) | 80B | 60B | 384 experts, ~33GB AWQ |
| Qwen3-Next-80B-A3B | 80B | 60B | Same |
| GLM-4.5-Air (106B) | 106B | 82B | Still exceeds 48GB |
| Qwen3-235B | 235B | 180B | Way too large |

## REAP (Cerebras) — wider model support

REAP supports more model families out of the box. Adding a new model requires only a dict entry in `src/reap/model_util.py`.

**Supported models**: Qwen3, GLM, Llama-4, Mixtral, DeepSeek, Ernie

### Setup

```bash
git clone https://github.com/CerebrasResearch/reap.git
cd reap
bash scripts/build.sh   # or use docker
```

### Prune Qwen3-Coder-30B (128 → 96 experts)

```bash
bash experiments/pruning-cli.sh 0 \
    Qwen/Qwen3-Coder-30B-A3B-Instruct \
    reap 42 0.25 \
    "theblackcat102/evol-codealpaca-v1:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096" \
    true true false false false
```

### Adding Gemma 4 to REAP

Add to `src/reap/model_util.py` `MODEL_ATTRS`:

```python
"Gemma4ForCausalLM": {
    "moe_block": "moe",            # Gemma4 MoE submodule name
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "experts": "experts",
    "fused": False,
    "router": "router",            # Gemma4 router attribute
    "num_experts": "num_experts",
    "num_experts_per_tok": "top_k_experts",  # Gemma4 uses top_k_experts
},
```

Note: exact attribute names need verification from the HF Gemma4 model code. The pre-made [0xSero/gemma-4-21b-a4b-it-REAP](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) was made with REAP, confirming it works on Gemma 4.

## REAM (SamsungSAIL) — better quality merging

REAM merges experts instead of dropping them. Currently supports **Qwen3** and **GLM** only.

### Setup

```bash
git clone https://github.com/SamsungSAILMontreal/ream.git
cd ream
conda create -n ream python=3.12 -y
conda activate ream
pip install -r requirements.txt
```

### REAM Qwen3-Coder-30B (128 → 96 experts)

```bash
# Code-heavy calibration mix for coding models
CUDA_VISIBLE_DEVICES="" python merge.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --merge_size 96 \
    --saliency reap \
    --merging logits+weights \
    --grouping ream \
    --dataset "c4+math+code" \
    --mix_ratio "0.0,0.3,0.7" \
    --save_path ~/AI/models/Qwen3-Coder-30B-REAM-BF16
```

## After compression: quantize to AWQ

```bash
conda activate quant
echo 3 | sudo tee /proc/sys/vm/drop_caches

# GPTQ calibration (CPU, ~4-6 hours)
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_moe_llmcompressor.py \
    --model ~/AI/models/Qwen3-Coder-30B-REAM-BF16 \
    --output ~/AI/models/Qwen3-Coder-30B-REAM-CT

# CT → AWQ conversion
python scripts/quantize/convert_moe_ct_to_awq.py \
    --src ~/AI/models/Qwen3-Coder-30B-REAM-CT \
    --dst ~/AI/models/Qwen3-Coder-30B-REAM-AWQ
```

## Requirements

- **RAM**: ~60GB (BF16 model loaded on CPU)
- **GPU**: Optional — speeds up saliency computation but not required
- **Disk**: ~120GB (source BF16 + output BF16)
- **Time**: Several hours on CPU for merging, then ~4-6 hours for AWQ calibration

## Pre-made compressed models

| Model | Method | Experts | Params | Source |
|-------|--------|:-------:|:------:|--------|
| [Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) | REAP | 103 | 25B | Cerebras |
| [Qwen3-30B-Instruct-2507-REAM](https://huggingface.co/SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM) | REAM | 96 | 23B | SamsungSAIL |
| [Qwen3-Coder-Next-REAM](https://huggingface.co/bknyaz/Qwen3-Coder-Next-REAM) | REAM | 384 | 60B | bknyaz |
| [gemma-4-21b-a4b-it-REAP](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) | REAP | 103 | 21B | 0xSero |

Compressed BF16 models can be pushed to HuggingFace and shared across systems (e.g., NVIDIA 3090 + AMD R9700 rigs).

## Adding new models to REAM

REAM uses REAP for its saliency scoring step. REAP already supports more model families
(Gemma 4, DeepSeek, Llama-4, etc.) via its `MODEL_ATTRS` config. Since REAM's `merger.py`
needs the same MoE layer attribute names, adding a model to REAM is straightforward if REAP
already supports it:

1. Check REAP's `src/reap/model_util.py` for the model's `MODEL_ATTRS` entry
2. Add the same tokenizer/model detection to REAM's `merge.py` (the `tokenizer_name` check)
3. Port the MoE layer access patterns to REAM's `merger.py`

REAP's wider model support serves as a reference implementation for each architecture's
expert layer naming, router structure, and config field names.
