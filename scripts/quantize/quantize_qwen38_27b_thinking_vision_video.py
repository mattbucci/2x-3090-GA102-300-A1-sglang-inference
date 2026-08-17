#!/usr/bin/env python3
r"""Qwen3.8-27B GPTQ W4A16 with thinking + vision + VIDEO aware calibration.

Qwen3.8-27B (released 2026-08-14) declares `Qwen3_5ForConditionalGeneration` /
`model_type: qwen3_5` — it is a Qwen3.5-FAMILY hybrid, so the whole existing
stack applies unchanged (patches 002/031 DeltaNet AWQ weight loaders, 003/007
GDN kernels, 056 conv-state dtype, 018 vision config dict-wrap, 035 config
fallbacks). Shape is a near-twin of Qwen3.6-27B dense, which is why this
script is a direct descendant of quantize_qwen36_27b_thinking_vision.py:

    64 layers = 48 linear_attention (Gated DeltaNet) + 16 full_attention
    (full_attention_interval=4), hidden 5120, FFN 17408, head_dim 256,
    24 Q heads / 4 KV heads, partial_rotary_factor 0.25, vocab 248320,
    native 262144 ctx, untied embeddings, vision tower depth 27 (hidden 1152,
    out 5120, patch 16), plus in-checkpoint MTP layers.

FOUR deltas from the Qwen3.6-27B donor:

1. RECIPE defaults to `thinking_vision_video` (not balanced_thinking_vision).
   Qwen3.8 does hour-scale VIDEO natively, and the project rule is explicit:
   never calibrate a video-capable model without video samples or the
   temporal-attention weights drift. thinking_vision_video is ~45/55
   thinking/non-thinking, so it also stays clear of the old 70%-thinking
   `thinking_vision` mix that M4 traced to the </think> repetition loop.

2. `re:.*mtp.*` added to the ignore list. The checkpoint ships MTP layers
   (text_config.mtp_num_hidden_layers). sglang's MAIN model loader skips every
   weight whose name contains "mtp" (qwen3_5.py: `if "mtp" in name: continue`,
   4 sites) — they are only consumed by Qwen3_5ForCausalLMMTP when speculative
   decoding is enabled. Leaving them BF16 costs disk but zero VRAM on the
   normal serving path, keeps them loadable, and avoids shipping quantized
   weights that nothing in our validation ever exercises.

3. lm_head stays BF16 (as always) but note this model UNTIES embeddings
   (tie_word_embeddings=false) with a 248,320 vocab — embed_tokens + lm_head
   are ~2.5 GB each in BF16. Expect the INT4 ship around ~19 GB, not the
   ~15 GB a 27B "should" be. Still ~9.5 GB/card at TP=2, leaving room for the
   16-full-attention-layer KV pool (~64 KB/token → ~8 GB/card at 256K).

4. DeltaNet ignore is UNCHANGED (in_proj_a$ / in_proj_b$ only) — the v3
   pattern. Restating why, because it has cost this project two failed
   calibrations: sglang's Qwen3_5GatedDeltaNet fuses the checkpoint layout into
   `in_proj_qkvz` (outer quant_config → INT4) and `in_proj_ba` (hardcoded
   quant_config=None → BF16). v1 (ignore nothing) quantized in_proj_a/b that
   the loader wants BF16; v2 (ignore all of linear_attn) kept qkvz BF16 that
   the loader wants INT4. Both produced `!!!!!` garbage. Probe the ship with
   "What is the capital of France?" before believing any benchmark.

IGNORE-LIST RECEIPT (verified against the real checkpoint index, 2026-08-17,
BEFORE burning calibration hours — this is the check whose absence cost the
project 16h on Gemma 4 26B v3). Simulated llmcompressor module-granularity
matching over all 985 modules in model.safetensors.index.json:

    lm_head                     ->   1 module
    re:.*visual\..*             -> 167 modules   (model.visual.blocks.*)
    re:.*mtp.*                  ->  15 modules   (mtp.fc / mtp.layers.* / mtp.norm*)
    re:.*in_proj_a$             ->  48 modules   (one per DeltaNet layer)
    re:.*in_proj_b$             ->  48 modules   (one per DeltaNet layer)
    re:.*vision_tower.*         ->   0   inert here (kept: family-wide net)
    re:.*multi_modal_projector.*->   0   inert here (kept: family-wide net)
    re:.*embed_vision.*         ->   0   inert here (kept: family-wide net)

Remaining quantizable inside linear_attn: in_proj_qkv, in_proj_z, out_proj
(exactly the three the loader feeds the outer quant_config) — 48 each.
conv1d and norm are not nn.Linear, so targets="Linear" never touches them.
Weights live under `model.language_model.*` / `model.visual.*` / `mtp.*`.

Usage (calibration device — NOT the eval/serving box; see Rule 1):
    conda activate quant
    CUDA_VISIBLE_DEVICES="" \
        BASE_MODEL=$MODELS_DIR/Qwen3.8-27B-BF16 \
        OUTPUT_DIR=$MODELS_DIR/Qwen3.8-27B-CT-thinking-vision-video \
        python scripts/quantize/quantize_qwen38_27b_thinking_vision_video.py

Then CT -> native AWQ via scripts/quantize/convert_qwen35_ct_to_awq.py (same
family layout), and ALWAYS run scripts/eval/check_awq_scales.py on the result
before shipping.
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

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BASE_MODEL = os.environ.get("BASE_MODEL", f"{MODELS_DIR}/Qwen3.8-27B-BF16")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"{MODELS_DIR}/Qwen3.8-27B-CT-thinking-vision-video"
)

NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "1024"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build thinking + vision + video calibration dataset ---
print("\n[1/4] Building thinking + vision + video calibration dataset...")
rows = build_calibration_dataset(
    recipe=os.environ.get("RECIPE", "thinking_vision_video"),
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
        # Vision tower / multimodal projector — preserve BF16. This checkpoint
        # names the tower `visual.*` (sglang's loader keys off "visual"), but
        # keep the family-wide set: regex form is mandatory so DESCENDANT
        # Linears are excluded too, not just the container module.
        r"re:.*vision_tower.*",
        r"re:.*visual\..*",
        r"re:.*multi_modal_projector.*",
        r"re:.*embed_vision.*",
        # MTP layers — main loader skips them entirely; only the speculative
        # MTP model consumes them. Keep BF16 (see docstring note 2).
        r"re:.*mtp.*",
        # DeltaNet gating scalars — sglang's Qwen3_5GatedDeltaNet loader
        # hardcodes quant_config=None for in_proj_ba (= in_proj_b + in_proj_a
        # fused), since output dim isn't divisible by TP alignment. Everything
        # else in linear_attn (in_proj_qkv, in_proj_z, out_proj) MUST be INT4
        # because the loader passes the outer quant_config to in_proj_qkvz.
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

# The base ships BOTH preprocessor_config.json (image) and
# video_preprocessor_config.json (video). AutoProcessor.save_pretrained does
# not always re-emit the video one, and a checkpoint missing it loses the
# video modality at serve time while image still works — the same silent
# modality-drop class M4 hit on community checkpoints. Verify and backfill.
import shutil

for cfg in ("preprocessor_config.json", "video_preprocessor_config.json"):
    src_cfg = os.path.join(BASE_MODEL, cfg)
    dst_cfg = os.path.join(OUTPUT_DIR, cfg)
    if os.path.exists(dst_cfg):
        print(f"  {cfg}: present")
    elif os.path.exists(src_cfg):
        shutil.copy2(src_cfg, dst_cfg)
        print(f"  {cfg}: MISSING after save -> copied from base")
    else:
        print(f"  {cfg}: absent in base too (nothing to copy)")

print("Done.")
