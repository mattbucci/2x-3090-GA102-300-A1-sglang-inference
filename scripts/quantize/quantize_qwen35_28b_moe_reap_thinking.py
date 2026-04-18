#!/usr/bin/env python3
"""Qwen3.5-28B-A3B-REAP GPTQ W4A16 with thinking-aware calibration.

The existing `Qwen3.5-28B-A3B-REAP-AWQ` was calibrated on UltraChat
(no thinking traces). Result: `<think>` tags are broken — model emits
free-form "The user is asking..." instead of `<think>...</think>`,
causing runaway generation and blocking `--reasoning-parser qwen3`.

This script uses the R9700 team's `calibration_datasets` mixed recipe
(AM-Thinking-v1-Distilled + NuminaMath-CoT + UltraChat) so the model
sees real `<think>...</think>` patterns during calibration and keeps
the capability post-quant.

DeltaNet layers, MoE router gate, and shared expert gate are still
excluded from INT4 (same constraint as the baseline pipeline).

Runs on CPU (54 GB BF16 model + Hessians won't fit on 2x24GB 3090).
Expected ~3-6h on this rig. Safe to leave running overnight.

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_28b_moe_reap_thinking.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make calibration_datasets importable from scripts/quantize/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
    verify_thinking_preserved,
)


MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-BF16")
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-CT-thinking"),
)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))  # thinking traces are long


print(f"Model:       {MODEL_PATH}")
print(f"Output:      {OUTPUT_DIR}")
print(f"Calibration: recipe=thinking_text  {NUM_SAMPLES} samples x {MAX_SEQ_LEN} tokens")


# ---- 1. Load BF16 model on CPU ----

print("\n[1/5] Loading BF16 model on CPU...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded: {type(model).__name__}")


# ---- 2. Fuse per-expert weights into FusedMoE format ----
# Same fix as the baseline pipeline: BF16 source has experts.{id}.{gate,up,down}_proj
# but the HF model class expects fused experts.{gate_up,down}_proj. If we skip this
# fusion, calibration silently runs on random-initialized fused weights and the
# resulting model outputs garbage.

print("\n[2/5] Fusing per-expert weights...")
shard_files = sorted(glob.glob(os.path.join(MODEL_PATH, "model*.safetensors")))
ckpt: dict[str, torch.Tensor] = {}
for sf in shard_files:
    with safe_open(sf, framework="pt") as f:
        for k in f.keys():
            if "mlp.experts." in k and re.search(r"experts\.\d+\.", k):
                ckpt[k] = f.get_tensor(k)

num_experts = model.config.text_config.num_experts
print(f"  num_experts={num_experts}")

lang_model = (
    model.model.language_model if hasattr(model.model, "language_model") else model.model
)
num_layers = len(lang_model.layers)

fused_count = 0
for layer_idx in range(num_layers):
    layer = lang_model.layers[layer_idx]
    if not hasattr(layer.mlp, "experts"):
        continue
    experts_mod = layer.mlp.experts
    gate_list, up_list, down_list = [], [], []
    for eid in range(num_experts):
        for prefix in (
            f"model.language_model.layers.{layer_idx}.mlp.experts.{eid}",
            f"model.layers.{layer_idx}.mlp.experts.{eid}",
        ):
            gw = ckpt.get(f"{prefix}.gate_proj.weight")
            uw = ckpt.get(f"{prefix}.up_proj.weight")
            dw = ckpt.get(f"{prefix}.down_proj.weight")
            if gw is not None:
                break
        if gw is None:
            print(f"  WARNING: layer {layer_idx} expert {eid} missing in checkpoint")
            continue
        gate_list.append(gw)
        up_list.append(uw)
        down_list.append(dw)

    if len(gate_list) != num_experts:
        print(f"  WARNING: layer {layer_idx} has {len(gate_list)}/{num_experts} experts")
        continue

    gate_stacked = torch.stack(gate_list)
    up_stacked = torch.stack(up_list)
    gate_up = torch.cat([gate_stacked, up_stacked], dim=1)
    down_stacked = torch.stack(down_list)
    experts_mod.gate_up_proj.data.copy_(gate_up)
    experts_mod.down_proj.data.copy_(down_stacked)
    fused_count += 1

print(f"  Fused {fused_count} layers")
del ckpt  # free RAM


# ---- 3. Build thinking-aware calibration set ----

print("\n[3/5] Building calibration dataset (recipe=thinking_text)...")
rows = build_calibration_dataset(
    recipe="thinking_text",
    num_samples=NUM_SAMPLES,
    seed=42,
)

text_ds = rows_to_text(rows, tokenizer, enable_thinking=True)
verify_thinking_preserved(text_ds, min_fraction=0.10)

tok_ds = tokenize_text_dataset(text_ds, tokenizer, max_length=MAX_SEQ_LEN)
print(f"Tokenized dataset: {len(tok_ds)} samples")


# ---- 4. Verification forward pass (sanity check) ----

print("\n[4/5] Verification forward pass...")
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    top5 = torch.topk(logits[0, -1], 5)
    for tid in top5.indices.tolist():
        print(f"  {tokenizer.decode([tid])!r}: {logits[0, -1, tid]:.2f}")


# ---- 5. GPTQ calibration ----

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",              # vision encoder, text-only GPTQ
        "re:.*in_proj_b$",            # DeltaNet beta gate (tiny, INT4-sensitive)
        "re:.*in_proj_a$",            # DeltaNet alpha gate
        "re:.*mlp\\.gate$",           # MoE router (full precision for routing)
        "re:.*shared_expert_gate$",   # shared expert gate ([1, hidden])
    ],
    offload_hessians=True,
)

print(f"\n[5/5] Running GPTQ ({NUM_SAMPLES} samples x {MAX_SEQ_LEN} tokens)...")
t0 = time.time()
oneshot(
    model=model,
    dataset=tok_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
    processor=tokenizer,
)
print(f"\nGPTQ done in {(time.time() - t0) / 3600:.2f}h")


# ---- Save ----

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")


# ---- Fix config: text-only CausalLM arch + rope params ----

import json

config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    cfg = json.load(f)

src_cfg_path = os.path.join(MODEL_PATH, "config.json")
with open(src_cfg_path) as f:
    src_cfg = json.load(f)

rope_params = src_cfg.get("text_config", {}).get("rope_parameters", {})
rope_params.pop("mrope_section", None)
rope_params.pop("mrope_interleaved", None)
if rope_params:
    cfg["rope_parameters"] = rope_params
    cfg["rope_scaling"] = rope_params
    print(f"Added rope_parameters (no mrope): {rope_params}")

cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
cfg["model_type"] = "qwen3_5_moe_text"
cfg.setdefault("norm_topk_prob", True)

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Fixed config: arch={cfg['architectures']}")

print("\nNext: convert CT→AWQ with scripts/quantize/convert_qwen35_moe_reap_ct_to_awq.py")
print("      then validate: scripts/eval/validate_chat_template.py --model", OUTPUT_DIR)
