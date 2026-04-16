#!/usr/bin/env python3
"""Quantize Qwen3-30B-Instruct REAM (96 experts) to W4A16 GPTQ.

Standard Qwen3 MoE — no DeltaNet, no expert fusion needed.
Architecture: Qwen3MoeForCausalLM (same as Coder-30B).

Usage:
    python scripts/quantize/quantize_qwen3_30b_ream.py
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = os.path.expanduser("~/AI/models/Qwen3-30B-Instruct-2507-REAM-BF16")
OUTPUT_DIR = os.path.expanduser("~/AI/models/Qwen3-30B-Instruct-2507-REAM-CT")

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded: {type(model).__name__}")

# Verify model loaded correctly
import torch
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
    top = torch.topk(out.logits[0, -1], 3)
    for tid in top.indices.tolist():
        print(f"  {tokenizer.decode([tid])}: {out.logits[0, -1, tid]:.2f}")

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]").shuffle(seed=42)
ds = ds.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)})
ds = ds.map(lambda x: tokenizer(x["text"], padding=False, max_length=512, truncation=True, add_special_tokens=False), remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*mlp\\.gate$",           # MoE router gates (full precision)
        "re:.*shared_expert_gate$",   # Shared expert gate (too small for INT4)
    ],
    offload_hessians=True,
)

print(f"\nRunning GPTQ ({len(ds)} samples)...")
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=512, num_calibration_samples=256, processor=tokenizer)
print(f"\nDone in {(time.time()-t0)/3600:.1f}h")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
