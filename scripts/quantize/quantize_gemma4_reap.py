#!/usr/bin/env python3
"""Quantize Gemma 4 21B REAP to W4A16 using llm-compressor GPTQ.

Pure MoE (103 experts), no DeltaNet. All Linear layers quantized except lm_head.
Runs on CPU (~40GB BF16 model, needs ~50GB RAM).
"""
import os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODEL_PATH = os.path.expanduser("~/AI/models/gemma-4-21b-REAP-BF16-textonly")
OUTPUT_DIR = os.path.expanduser("~/AI/models/gemma-4-21b-REAP-AWQ-CT")

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

print(f"Model:  {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

print("\nLoading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

print(f"\nLoading calibration data...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH,
                     truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True,
                strategy="group", group_size=32,  # Gemma 4 moe_intermediate_size=704 not div by 128
            ),
        ),
    },
    ignore=["lm_head"],
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration...")
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        processor=tokenizer)
elapsed = time.time() - t0
print(f"\nGPTQ completed in {elapsed / 3600:.1f} hours")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Saved to {OUTPUT_DIR}")
