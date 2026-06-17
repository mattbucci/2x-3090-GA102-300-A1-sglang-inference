#!/usr/bin/env python3
"""CohereLabs North-Mini-Code-1.0 (Cohere2MoeForCausalLM) GPTQ W4A16.

30B/3B-A MoE — 49 layers (1 dense prefix + 48 MoE), 128 experts top-8, sigmoid
router, sliding+full attention 3:1, no shared experts. text-only code/agentic.

Architecture: Cohere2MoeForCausalLM (native in transformers ≥5.10). No on-disk
patches needed (unlike NemotronH). hidden_size=2048, MoE expert intermediate=768
(clean 768/128=6 groups → standard AWQ group_size=128, no override needed).

License: Apache 2.0 (upstream); ships under same to mattbucci/North-Mini-Code-1.0-AWQ.

Calibration: code_thinking recipe (40% the-stack + 25% AM-Thinking + 20%
NuminaMath + 15% ultrachat) at 1024 samples × 2048 tokens to match the
high-quality Nemotron Phase 2 budget — Cohere2MoE is structurally simpler than
NemotronH (no Mamba2, no multimodal towers) so wall-clock should be much faster
(~1-2 d expected vs Nemotron's 9 d).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_north_mini_code.py
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
    verify_thinking_preserved,
)
from expert_utilization import ExpertUtilizationTracker
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Inline recipe — same shape as RECIPE_CODE_THINKING but swaps `thestack_code`
# for `code_instruct`. Two reasons:
#   (1) bigcode/the-stack-smol's config layout collapsed from per-language
#       ("data/python", "data/javascript", …) to a single "default" config
#       that concatenates ALL 88 languages alphabetically — the first ~30K
#       rows are asm/bat/cpp, no Python until much deeper. The recipe wanted
#       Python code, not assembly. Filtering would work but adds fragility.
#   (2) `code_instruct` (iamtarun/python_code_instructions_18k_alpaca) is
#       already wired up in calibration_datasets.py with a comment marking
#       it as "Non-gated replacement for bigcode/the-stack-smol." It's
#       instruction-formatted, which better matches North-Mini-Code's actual
#       agentic-coding use case than raw code dumps would.
RECIPE = {
    "code_instruct":  0.40,  # Python instruction -> code (agentic match)
    "am_thinking":    0.25,  # reasoning traces with <think> tags
    "numina_math":    0.20,  # math + CoT (boosts GPTQ accuracy ~10% per recipe doc)
    "ultrachat":      0.15,  # general multi-turn dialogue
}

BASE_MODEL = os.environ.get(
    "BASE_MODEL", os.path.expanduser("~/AI/models/North-Mini-Code-1.0-BF16")
)
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"{MODELS_DIR}/North-Mini-Code-1.0-W4A16-CT"
)

# Match Nemotron-class budget: 1024 × 2048 = 2.1 M calibration tokens.
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1024"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:  {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")

# --- 1. Build code + thinking calibration set ---
print("\n[1/4] Building code + thinking calibration dataset...")
rows = build_calibration_dataset(
    recipe=RECIPE,
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Render through chat template ---
print("\n[2/4] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.chat_template is None:
    raise RuntimeError(f"{BASE_MODEL} missing chat_template")

# North-Mini-Code is a text-only code/agent model: no images, no thinking-toggle
# kwarg (the model defaults to its own thinking discipline via system prompt).
text_dataset = rows_to_text(
    rows,
    tokenizer,
    enable_thinking=True,
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")
# Cohere2 may not use <think>...</think> tagging the same way as Qwen — keep the
# verification step but tolerate a low fraction (don't hard-fail).
try:
    verify_thinking_preserved(text_dataset, min_fraction=0.05)
except Exception as e:
    print(f"  (thinking-preservation soft fail, continuing: {e})")

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load model on CPU ---
print("\n[3/4] Loading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=False,  # Cohere2MoE is native in transformers ≥5.10
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")

# --- 4. GPTQ calibration ---
print("\n[4/4] Running GPTQ calibration...")
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",  # group_size=128 default; 768/128=6 — works out of the box.
    ignore=[
        "lm_head",
        # MoE router gate stays full precision. Cohere2MoE uses `mlp.gate` like
        # Qwen3MoE, so the same regex captures it.
        r"re:.*mlp\.gate$",
        # `first_k_dense_replace=1`: layer 0 is dense (no MoE). Its MLP gates/
        # up/down stay INT4 like the rest — no exclusion needed.
        # `num_shared_experts=0` — no shared_expert.* modules to ignore.
    ],
    offload_hessians=True,
)

top_k = getattr(model.config, "num_experts_per_tok", 8)
tracker = ExpertUtilizationTracker(model, top_k=top_k)

# moe_calibrate_all_experts=True — same rationale as Coder-30B: rare-routed
# experts get zero/few samples otherwise, leading to degenerate AWQ scales.
# Per feedback_moe_quant_best_practices.md.
t0 = time.time()
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
    moe_calibrate_all_experts=True,
)
elapsed = time.time() - t0
print(f"\nGPTQ complete in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("\n" + tracker.summary())
tracker.dump_json(os.path.join(OUTPUT_DIR, "expert_utilization.json"))
if tracker.has_blocking_issues():
    print("*** WARNING: at least one expert saw ZERO routing decisions during calibration. ***")
    print("*** AWQ scales for these experts will be degenerate. Inspect expert_utilization.json ***")

print(f"\nSaving CT output to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
