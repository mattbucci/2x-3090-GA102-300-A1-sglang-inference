# Quantization Pipeline

All models use **INT4 quantized** weights on SGLang. Pipeline:

```
BF16 model → GPTQ calibration (llmcompressor, quant env) → compressed-tensors format
```

SGLang loads compressed-tensors directly with `--quantization compressed-tensors`.
For optimized Marlin kernels, an optional CT→AWQ conversion step can be added, but
has known quality issues for some models (see Gemma 4 notes in README).

## Environment Setup

**CRITICAL: Use a separate conda env.** llmcompressor conflicts with sglang deps.

```bash
conda create -n quant python=3.12 -y
conda activate quant
pip install llmcompressor transformers compressed-tensors accelerate datasets

# Or install dev versions for latest model support:
pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
```

Always drop filesystem caches before calibration:
```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## GPTQModifier API Reference

Source: [github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)

### Basic usage (preset scheme)

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

recipe = GPTQModifier(
    targets="Linear",          # What to quantize
    scheme="W4A16",            # Preset: 4-bit weights, 16-bit activations (group_size=128)
    ignore=["lm_head"],        # Skip these modules
    offload_hessians=True,     # Keep hessians on CPU to save VRAM
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
    processor=tokenizer,
)

model.save_pretrained(output_dir, save_compressed=True)
```

### Custom group_size (config_groups)

When `group_size=128` doesn't divide evenly (e.g., Gemma 4 moe_intermediate_size=704):

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

recipe = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True,
                strategy="group", group_size=32,  # 704 % 32 = 0
            ),
        ),
    },
    offload_hessians=True,
)
```

### Ignore patterns

Supports exact names and regex:

```python
ignore = [
    "lm_head",              # Exact: output head
    "re:.*in_proj_a$",      # Regex: DeltaNet alpha gates (dim 48)
    "re:.*in_proj_b$",      # Regex: DeltaNet beta gates (dim 48)
    "re:.*mlp.gate$",       # Regex: MoE router gates (sensitive)
    "re:visual.*",          # Regex: vision encoder
]
```

## Model-Specific Recipes

### Dense model (Devstral-24B)

Standard — all Linear layers quantized.

```python
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"],
                      offload_hessians=True)
```

### Pure MoE (Qwen3-Coder-30B, Gemma 4 26B)

Same as dense, but for Gemma 4 use group_size=32 because moe_intermediate_size=704
is not divisible by 128.

```python
# Qwen3 MoE (group_size=128 works)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"],
                      offload_hessians=True)

# Gemma 4 MoE (needs group_size=32)
recipe = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                                    strategy="group", group_size=32),
        ),
    },
    offload_hessians=True,
)
```

### DeltaNet hybrid (Qwen3.5-27B)

Exclude DeltaNet gate layers (dim=48, tiny, keep in BF16):

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*in_proj_b$",   # DeltaNet beta gate (dim 48)
        "re:.*in_proj_a$",   # DeltaNet alpha gate (dim 48)
    ],
    offload_hessians=True,
)
```

## Running Calibration

All calibration runs on **CPU only** (`CUDA_VISIBLE_DEVICES=""`).

```bash
conda activate quant
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Dense / MoE
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_reap.py

# DeltaNet hybrid
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_llmcompressor.py
```

**Expected time**: ~4-6 hours for 27B model on CPU (92GB RAM).

**Expected RAM**: ~50-60GB for 27B BF16 model + hessians.

## Output Format

`save_pretrained(save_compressed=True)` produces compressed-tensors format:

- `weight_packed`: int32 with packed 4-bit values
- `weight_scale`: per-group scales
- `weight_zero_point`: per-group zero points (if asymmetric)
- `weight_g_idx`: column→group mapping (if using actorder=GROUP)

SGLang loads this directly with `--quantization compressed-tensors`.

## Optional: CT → AWQ Conversion

For SGLang's AWQ Marlin kernels (faster decode), convert compressed-tensors to native AWQ:

```bash
python scripts/quantize/convert_qwen35_ct_to_awq.py
```

**WARNING**: CT→AWQ conversion has known quality issues for Gemma 4 (cosine similarity
drops on large output dimensions). Use compressed-tensors directly when possible.

## Scripts

| Script | Model | group_size | Notes |
|--------|-------|:----------:|-------|
| `quantize_qwen35_llmcompressor.py/sh` | Qwen3.5-27B | 128 | Skips DeltaNet layers |
| `quantize_devstral_llmcompressor.py/sh` | Devstral-24B | 128 | Dense, skips vision |
| `quantize_gemma4_reap.py` | Gemma 4 21B REAP | 32 | MoE, 704 not div by 128 |
| `quantize_ream_qwen3.py` | REAM'd Qwen3 | 128 | Pure MoE |
| `quantize_moe_llmcompressor.py` | Generic MoE | 128 | Template |
| `convert_qwen35_ct_to_awq.py` | Qwen3.5 CT→AWQ | — | Optional Marlin conversion |
| `convert_devstral_ct_to_awq.py` | Devstral CT→AWQ | — | Optional Marlin conversion |
| `convert_moe_ct_to_awq.py` | Generic MoE CT→AWQ | — | Template |

## MoE Expert Compression (REAM/REAP)

See [REAM.md](REAM.md) for reducing expert count to fit in 48GB VRAM.
