#!/usr/bin/env python3
"""Qwen3-VL-30B-A3B MoE GPTQ W4A16 with thinking + vision aware calibration.

Based on quantize_qwen3vl_32b_thinking_vision.py but for the 128-expert MoE
variant Qwen3VLMoeForConditionalGeneration (Qwen3-VL-30B-A3B-Instruct).

Key differences from the 32B Dense script:
  - MoE: `mlp.gate` (router) MUST stay BF16 (otherwise expert routing collapses
    to ~15 distinct values and the model emits gibberish).
  - No DeltaNet — Qwen3-VL-30B predates Qwen3.5/3.6, so no `in_proj_a/b` to
    preserve.
  - Experts ship fused (`experts.gate_up_proj [128, 2048, 1536]` +
    `experts.down_proj [128, 768, 2048]`) — llmcompressor may need to unfuse
    before oneshot(). If it fails with a tuple/shape error, apply R9700's
    llmcompressor patch 001 (`qwen3-moe tuple router logits` / unfuse path).

Community vLLM AWQ at `cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit` / similar
is broken on SGLang's `awq_marlin` loader (weight-name mismatch for
`Qwen3VLMoeForConditionalGeneration`). Self-calibrating a CT checkpoint +
running it through `convert_moe_ct_to_awq.py` produces a working AWQ that
awq_marlin can load, same approach as Qwen3.6-35B.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" \
        BASE_MODEL=/home/letsrtfm/AI/models/Qwen3-VL-30B-A3B-Instruct-BF16 \
        OUTPUT_DIR=/home/letsrtfm/AI/models/Qwen3-VL-30B-A3B-CT-thinking-vision \
        python scripts/quantize/quantize_qwen3vl_30b_moe_thinking_vision.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
)
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

BASE_MODEL = os.environ.get("BASE_MODEL", "/home/letsrtfm/AI/models/Qwen3-VL-30B-A3B-Instruct-BF16")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/Qwen3-VL-30B-A3B-CT-thinking-vision")

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build thinking + vision calibration dataset ---
print("\n[1/4] Building thinking + vision calibration dataset...")
rows = build_calibration_dataset(
    recipe="thinking_vision_video",
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Render through chat template ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(f"{BASE_MODEL} missing chat_template")

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")
dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load model on CPU ---
print("\n[3/4] Loading model on CPU...")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")

# --- 4. GPTQ calibration ---
print("\n[4/4] Running GPTQ calibration...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        # Vision tower / multimodal projector — preserve BF16
        r"re:.*vision_tower.*",
        r"re:.*visual\..*",
        r"re:.*multi_modal_projector.*",
        r"re:.*embed_vision.*",
        # MoE router — (128, H) gate matrix must stay BF16 so expert routing
        # isn't clamped to ~15 distinct values. Matches Qwen3.6-35B learnings.
        r"re:.*mlp\.gate$",
    ],
    offload_hessians=True,
)

t0 = time.time()
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
)
elapsed = time.time() - t0
print(f"\nGPTQ complete in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"  Saved preprocessor (image) config to {OUTPUT_DIR}")

print("Done.")
