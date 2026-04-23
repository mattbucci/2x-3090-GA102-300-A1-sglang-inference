#!/usr/bin/env python3
"""Qwen3.6-27B GPTQ W4A16 with thinking + vision aware calibration.

Dedicated script for Qwen3.6-27B (and any future Qwen3.5/3.6 hybrid model).
Differs from quantize_qwen3vl_32b_thinking_vision.py in ONE place: the
ignore regex also excludes `in_proj_a` and `in_proj_b` — the tiny DeltaNet
gating scalars.

Why (corrected 2026-04-23 after two failed calibrations): SGLang's
`Qwen3_5GatedDeltaNet` loader merges the checkpoint layout into two fused
params: `in_proj_qkvz` (from `in_proj_qkv` + `in_proj_z`, uses the outer
`quant_config`) and `in_proj_ba` (from `in_proj_b` + `in_proj_a`, hardcoded
`quant_config=None` per qwen3_5.py:186 — "output dim 48 is not divisible
by TP=2 * 16 alignment"). So the loader wants:
  - in_proj_qkv / in_proj_z / out_proj → INT4 (Marlin)
  - in_proj_b / in_proj_a → BF16 always
  - conv1d → BF16 (nn.Conv1d, not touched by GPTQModifier targets="Linear")

v1 (ignore=[]): quantized everything incl. in_proj_a/b → loader wanted BF16.
v2 (ignore=linear_attn.*): kept everything BF16 → loader wanted INT4 qkvz.
v3 (this script): ignore only `in_proj_a$` + `in_proj_b$`, matching R9700's
working Qwen3.5-27B recipe and SGLang's exact weight-layout expectations.

Everything else matches the Qwen3VL script: `thinking_vision` recipe,
`drop_images=True`, AutoProcessor pre-flight, no try/except around save.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" \
        BASE_MODEL=/home/letsrtfm/AI/models/Qwen3.6-27B-BF16 \
        OUTPUT_DIR=/home/letsrtfm/AI/models/Qwen3.6-27B-CT-thinking-vision \
        python scripts/quantize/quantize_qwen36_27b_thinking_vision.py
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

BASE_MODEL = os.environ.get("BASE_MODEL", "/home/letsrtfm/AI/models/Qwen3.6-27B-BF16")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/Qwen3.6-27B-CT-thinking-vision")

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
    recipe="thinking_vision",
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
        # DeltaNet gating scalars — SGLang's Qwen3_5GatedDeltaNet loader
        # hardcodes quant_config=None for in_proj_ba (= in_proj_b + in_proj_a
        # fused), since output dim 48 isn't divisible by TP alignment.
        # Everything else in linear_attn (in_proj_qkv, in_proj_z, out_proj)
        # must be INT4 because the loader passes the outer quant_config
        # through to in_proj_qkvz.  Matches R9700's working Qwen3.5-27B
        # recipe (quantize_qwen35_moe_ream.py).
        r"re:.*in_proj_a$",
        r"re:.*in_proj_b$",
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
