# MoE Expert Compression: REAM & REAP

Two methods to shrink MoE models by reducing expert count. Both run on CPU with ~60GB RAM.

| Method | What it does | Quality | Code |
|--------|-------------|---------|------|
| **REAM** | Merges expert groups | >=94% retained | [SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream) |
| **REAP** | Prunes low-impact experts | Better on generative tasks | [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) |

## Why compress?

With 48GB VRAM (2x RTX 3090), compression is essential for fitting MoE models at long context:
- Full Coder-30B (128 experts, ~16GB AWQ) limits context to 16K
- REAP'd Coder-30B (103 experts, ~13GB) enables 131K context at 134 tok/s
- For DeltaNet hybrids, expert savings matter more since DeltaNet layers stay BF16

## Candidate models

### Pure MoE — easiest to run, best performance

| Model | Params | Experts | After | AWQ est. | Method | Status |
|-------|--------|:-------:|:-----:|:--------:|--------|--------|
| Qwen3-Coder-30B | 30B | 128 | 96 (23B) | ~12 GB | REAM/REAP | Need to REAM ourselves |
| Gemma 4 26B MoE | 26B | 128 | 103 (21B) | ~10 GB | REAP | Calibrating (GPTQ) |
| Qwen3-Coder-REAP-25B | 25B | 103 | — | ~13 GB | REAP | Running at 134 tok/s |
| Qwen3-30B-Instruct-2507 | 30B | 128 | 96 (23B) | ~12 GB | REAM | [Pre-made (SamsungSAIL)](https://huggingface.co/SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM) |

### DeltaNet hybrid MoE — needs BF16 DeltaNet layers

| Model | Params | Experts | After | AWQ est. | Notes |
|-------|--------|:-------:|:-----:|:--------:|-------|
| Qwen3.5-27B | 27B | dense | — | ~15 GB | Working at 7 tok/s (needs profiling) |
| Qwen3.5-35B-A3B | 35B | 256 | 192 (27B) | ~14 GB | REAM target, DeltaNet hybrid |

#### Qwen3.5-35B-A3B pre-made REAP variants

| Model | Experts | Pruned | Source |
|-------|:-------:|:------:|--------|
| [0xSero/Qwen-3.5-28B-A3B-REAP](https://huggingface.co/0xSero/Qwen-3.5-28B-A3B-REAP) | 205/256 | 20% | BF16 |
| [atbender/Qwen3.5-REAP-20B-A3B](https://huggingface.co/atbender/Qwen3.5-REAP-20B-A3B) | 128/256 | 50% | BF16 + W4A16 AutoRound |

#### Gemma 4 pre-made REAP variants

| Model | Experts | Pruned | Source |
|-------|:-------:|:------:|--------|
| [0xSero/gemma-4-21b-a4b-it-REAP](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) | 103/128 | 20% | BF16, calibrating now |

### Too large even after compression (48GB limit)

| Model | Params | After | Why |
|-------|--------|:-----:|-----|
| Qwen3-Coder-Next (80B) | 80B | 60B | 384 experts, ~33GB AWQ |
| GLM-4.5-Air (106B) | 106B | 82B | Still exceeds 48GB |
| Qwen3-235B | 235B | 180B | Way too large |

## REAM Setup (SamsungSAIL)

Merges experts instead of dropping them. Currently supports **Qwen3** and **GLM**.

```bash
git clone https://github.com/SamsungSAILMontreal/ream.git
cd ream
conda create -n ream python=3.12 -y
conda activate ream
pip install -r requirements.txt
```

### REAM Qwen3-Coder-30B (128 → 96 experts)

```bash
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

### REAM Qwen3.5-35B-A3B (256 → 192 experts)

```bash
CUDA_VISIBLE_DEVICES="" python merge.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --merge_size 192 \
    --saliency reap \
    --merging logits+weights \
    --grouping ream \
    --dataset "c4+math+code" \
    --mix_ratio "0.0,0.3,0.7" \
    --save_path ~/AI/models/Qwen3.5-35B-A3B-REAM-BF16
```

**Note:** Qwen3.5 MoE may need model support added to REAM. REAP already supports it —
use REAP's `MODEL_ATTRS` as reference for porting.

## REAP Setup (Cerebras)

Wider model support: Qwen3, GLM, Llama-4, Mixtral, DeepSeek, Ernie, Gemma 4.

```bash
git clone https://github.com/CerebrasResearch/reap.git
cd reap
bash scripts/build.sh
```

### Adding Gemma 4 to REAP (if not already supported)

Add to `src/reap/model_util.py` `MODEL_ATTRS`:
```python
"Gemma4ForCausalLM": {
    "moe_block": "moe",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "experts": "experts",
    "fused": False,
    "router": "router",
    "num_experts": "num_experts",
    "num_experts_per_tok": "top_k_experts",
},
```

## After compression: quantize

### Pure MoE (no DeltaNet)
```bash
conda activate quant
echo 3 | sudo tee /proc/sys/vm/drop_caches
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_ream_qwen3.py
```

### DeltaNet hybrid MoE
```bash
# Uses DeltaNet-aware pipeline (excludes in_proj_a, in_proj_b, MoE gates)
./scripts/quantize/quantize_qwen35_moe_ream.sh
```

SGLang loads compressed-tensors directly: `--quantization compressed-tensors`.

## Requirements

- **RAM**: ~60-70GB (BF16 model on CPU + GPTQ Hessians)
- **GPU**: Optional — speeds up saliency computation but not required
- **Disk**: ~140GB (source BF16 + REAM BF16 + quantized output)
- **Time**: Several hours for REAM/REAP merging, then ~4-12 hours for GPTQ calibration

Compressed BF16 models can be pushed to HuggingFace and shared across systems.
