#!/usr/bin/env python3
"""Devstral-Small-2-24B-Instruct-2512 GPTQ W4A16 with code + vision + TOOL calibration.

Why this exists: the community AWQ (androiddrew/...-2512-AWQ-4bit) degenerates on
the structured tool context ([AVAILABLE_TOOLS] JSON -> bracket-salad / prompt echo),
while basic chat + vision survive. Root cause (proven via raw /v1/completions on
v0.5.12): the `[TOOL_CALLS]`/`[AVAILABLE_TOOLS]` rare-token pathway is under-
calibrated — the original calibration had no tool-format data. This rebuild uses
the `code_vision_tools` recipe (30% Hermes function-calling rendered through the
chat template WITH tools=, 30% python-instruct code, 20% LLaVA vision, 10%
ultrachat, 10% NuminaMath) so those tokens see real activations.

Base is the official FP8-native checkpoint dequantized to BF16 (Mistral's
recommended path — they no longer publish BF16). Language model -> INT4 W4A16;
the Pixtral vision_tower + multi_modal_projector + lm_head stay BF16 (ignored),
matching the fp8 base's modules_to_not_convert and the working gemma-4 layout.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_devstral2_code_vision_tools.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU calibration — keep both 3090s free

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
)
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BASE_MODEL = os.environ.get(
    "BASE_MODEL", "/data/models/Devstral-Small-2-24B-Instruct-2512-BF16-dequant"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"{MODELS_DIR}/Devstral-Small-2-24B-2512-AWQ-CT-code-vision-tools"
)
RECIPE = os.environ.get("RECIPE", "code_vision_tools")
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))  # tool prompts run long

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Base:   {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"Recipe: {RECIPE}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build code + vision + tool calibration dataset ---
print("\n[1/5] Building calibration dataset...")
rows = build_calibration_dataset(
    recipe=RECIPE,
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Tokenizer + chat template (Devstral-2 ships a Mistral tool template) ---
print("\n[2/5] Loading tokenizer + rendering chat template (tools threaded in)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    template_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "devstral_chat_template.jinja"))
    with open(template_path) as f:
        tokenizer.chat_template = f.read()
    print(f"  Loaded custom Devstral chat_template from {template_path}")

text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=False,   # Devstral has no thinking mode
    drop_images=True,        # vision encoder isn't quantized; LM sees placeholder slots
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")

# Verify the tool pathway actually made it into the calibration text — this is the
# whole point of the rebuild; bail loudly before a 12h run on the wrong data.
joined = "\n".join(r["text"] for r in text_dataset)
n_avail = joined.count("[AVAILABLE_TOOLS]")
n_calls = joined.count("[TOOL_CALLS]")
print(f"  tool-format coverage: [AVAILABLE_TOOLS]={n_avail}  [TOOL_CALLS]={n_calls}")
if n_avail < 10 or n_calls < 5:
    raise RuntimeError(
        f"Tool-format pathway under-represented in calibration text "
        f"([AVAILABLE_TOOLS]={n_avail}, [TOOL_CALLS]={n_calls}). "
        f"Hermes mix or tools= threading is broken — fix before calibrating."
    )

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load full multimodal model on CPU (keeps Pixtral vision tower in place) ---
print("\n[3/5] Loading model on CPU...")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")

# --- 4. GPTQ calibration (LM only; vision + projector + lm_head stay BF16) ---
print("\n[4/5] Running GPTQ calibration...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        r"re:.*vision_tower.*",
        r"re:.*multi_modal_projector.*",
        r"re:.*patch_merger.*",
        r"re:.*vision_encoder.*",
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

# --- 5. Save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUTPUT_DIR)
try:
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    proc.save_pretrained(OUTPUT_DIR)
    print("  Saved preprocessor (image) config")
except Exception as e:
    print(f"  WARN: could not save preprocessor ({e!r})")

print("Done.")
print("Next:")
print(f"  1. CT->AWQ: python scripts/quantize/convert_devstral_ct_to_awq.py (verify it keeps vision FP16)")
print(f"  2. Scale audit: python scripts/eval/check_awq_scales.py <awq-dir>")
print(f"  3. Validate: MODEL=<awq-dir> scripts/launch.sh devstral  (tool_call must PASS)")
