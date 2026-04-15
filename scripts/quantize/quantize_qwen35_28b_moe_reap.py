#!/usr/bin/env python3
"""Quantize Qwen3.5-28B-A3B-REAP to W4A16 GPTQ. DeltaNet hybrid MoE.

The BF16 source is a VL model with per-expert weights (experts.0.gate_proj.weight)
but the HF model class expects fused experts (experts.gate_up_proj). We load the
model, fuse experts in-memory, then run GPTQ calibration.
"""
import os, re, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-BF16")
OUTPUT_DIR = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-CT")

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded: {type(model).__name__}")


# ---- Fuse per-expert weights into FusedMoE format ----
# The checkpoint has experts.{id}.{gate,up,down}_proj.weight but the
# model class created fused params experts.gate_up_proj / experts.down_proj.
# We need to populate these from the per-expert checkpoint weights.
print("\nFusing per-expert weights...")
import glob
shard_files = sorted(glob.glob(os.path.join(MODEL_PATH, "model*.safetensors")))
ckpt = {}
for sf in shard_files:
    with safe_open(sf, framework="pt") as f:
        for k in f.keys():
            if "mlp.experts." in k and re.search(r"experts\.\d+\.", k):
                ckpt[k] = f.get_tensor(k)

# Get num_experts from config
num_experts = model.config.text_config.num_experts
print(f"  num_experts={num_experts}")

# Find the language model's layers
lang_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
num_layers = len(lang_model.layers)

fused_count = 0
for layer_idx in range(num_layers):
    layer = lang_model.layers[layer_idx]
    if not hasattr(layer.mlp, "experts"):
        continue
    experts_mod = layer.mlp.experts

    # Collect per-expert weights from checkpoint
    gate_list, up_list, down_list = [], [], []
    for eid in range(num_experts):
        # Keys use model.language_model prefix in checkpoint
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts.{eid}"
        gw = ckpt.get(f"{prefix}.gate_proj.weight")
        uw = ckpt.get(f"{prefix}.up_proj.weight")
        dw = ckpt.get(f"{prefix}.down_proj.weight")
        if gw is None:
            # Try without language_model prefix
            prefix = f"model.layers.{layer_idx}.mlp.experts.{eid}"
            gw = ckpt.get(f"{prefix}.gate_proj.weight")
            uw = ckpt.get(f"{prefix}.up_proj.weight")
            dw = ckpt.get(f"{prefix}.down_proj.weight")
        if gw is None:
            print(f"  WARNING: layer {layer_idx} expert {eid} not found in checkpoint")
            continue
        gate_list.append(gw)
        up_list.append(uw)
        down_list.append(dw)

    if len(gate_list) != num_experts:
        print(f"  WARNING: layer {layer_idx} has {len(gate_list)}/{num_experts} experts")
        continue

    # Fuse gate+up: [E, intermediate, hidden] cat along dim=1
    gate_stacked = torch.stack(gate_list)  # [E, intermediate, hidden]
    up_stacked = torch.stack(up_list)      # [E, intermediate, hidden]
    gate_up = torch.cat([gate_stacked, up_stacked], dim=1)  # [E, 2*intermediate, hidden]
    down_stacked = torch.stack(down_list)  # [E, hidden, intermediate]

    # Assign to model params
    experts_mod.gate_up_proj.data.copy_(gate_up)
    experts_mod.down_proj.data.copy_(down_stacked)
    fused_count += 1

print(f"  Fused {fused_count} layers")
del ckpt  # Free memory

# Verify the model produces sensible output
print("\nVerification forward pass...")
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    top5 = torch.topk(logits[0, -1], 5)
    for tid in top5.indices.tolist():
        print(f"  {tokenizer.decode([tid])}: {logits[0, -1, tid]:.2f}")


# ---- Prepare calibration data ----
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]").shuffle(seed=42)
ds = ds.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)})
ds = ds.map(lambda x: tokenizer(x["text"], padding=False, max_length=512, truncation=True, add_special_tokens=False), remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*in_proj_b$",            # DeltaNet beta gate (tiny, precision-sensitive)
        "re:.*in_proj_a$",            # DeltaNet alpha gate (tiny, precision-sensitive)
        "re:.*mlp\\.gate$",           # MoE router gates (full precision for routing)
        "re:.*shared_expert_gate$",   # Shared expert gate ([1, hidden], too small for INT4)
    ],
    offload_hessians=True,
)

print(f"\nRunning GPTQ...")
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=512, num_calibration_samples=256, processor=tokenizer)
print(f"\nDone in {(time.time()-t0)/3600:.1f}h")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

# Fix config: ensure rope_parameters (without mrope) and correct architectures
import json
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    cfg = json.load(f)

# Copy rope_parameters from source text_config, strip VL-only mrope
src_config_path = os.path.join(MODEL_PATH, "config.json")
with open(src_config_path) as f:
    src_cfg = json.load(f)
rope_params = src_cfg.get("text_config", {}).get("rope_parameters", {})
rope_params.pop("mrope_section", None)
rope_params.pop("mrope_interleaved", None)
if rope_params:
    cfg["rope_parameters"] = rope_params
    cfg["rope_scaling"] = rope_params
    print(f"Added rope_parameters (no mrope): {rope_params}")

# Set text-only CausalLM architecture
cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
cfg["model_type"] = "qwen3_5_moe_text"
cfg.setdefault("norm_topk_prob", True)

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Fixed config: arch={cfg['architectures']}")
