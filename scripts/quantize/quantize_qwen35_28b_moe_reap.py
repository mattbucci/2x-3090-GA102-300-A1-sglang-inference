#!/usr/bin/env python3
"""Quantize Qwen3.5-28B-A3B-REAP to W4A16 GPTQ. DeltaNet hybrid MoE."""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-BF16")
OUTPUT_DIR = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-CT")

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded: {type(model).__name__}")

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

# Fix config: ensure rope_parameters and correct architectures for text-only CausalLM
import json
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    cfg = json.load(f)

# Copy rope_parameters from source text_config if missing
src_config_path = os.path.join(MODEL_PATH, "config.json")
with open(src_config_path) as f:
    src_cfg = json.load(f)
rope_params = src_cfg.get("text_config", {}).get("rope_parameters")
if rope_params and "rope_parameters" not in cfg:
    cfg["rope_parameters"] = rope_params
    cfg["rope_scaling"] = rope_params
    print(f"Added rope_parameters from source")

# Set text-only CausalLM architecture
cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
cfg["model_type"] = "qwen3_5_moe_text"
cfg.setdefault("norm_topk_prob", True)

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Fixed config: arch={cfg['architectures']}, rope_params={'rope_parameters' in cfg}")
