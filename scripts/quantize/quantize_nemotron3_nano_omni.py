#!/usr/bin/env python3
"""Nemotron-3-Nano-Omni-30B-A3B-Reasoning GPTQ W4A16 calibration.

NVIDIA's NemotronH_Nano_Omni_Reasoning_V3: a Mamba2-Transformer hybrid MoE
(31B total / 3B active) with CRADIO v4-H vision encoder + Parakeet audio
encoder + reasoning mode default ON. Modalities: text + image + video + audio.
SGLang day-0 supported (nemotron_h.py / nano_nemotron_vl.py / nemotron_h_mtp.py
already in v0.5.12 tree); the gap is an AWQ ship — none exists for the
Omni-Reasoning variant yet. We're first.

INT4 eligibility (derived from llm_config.hybrid_override_pattern in the
pre-flight task #18):
    Mamba2 layers (BF16):  [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25,
                            28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]
    Attention (BF16):      [5, 12, 19, 26, 33, 42]  (stelterlab rule: keep BF16)
    MLP/MoE (INT4):        [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27,
                            29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]
Vision tower (RADIO), audio tower (Parakeet), MoE routers/gates, embeddings,
and lm_head all stay BF16 (same cardinal rules as Gemma 4 / Qwen 3.5 hybrids).

Calibration recipe: omni_thinking_tools — covers every live modality + the
XML tool-call pathway. Calibration time estimate: 12-20h on CPU (2x larger
than devstral's 3h baseline due to richer modalities + 128 experts + 23 MoE
layers vs Devstral's 1).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/quantize_nemotron3_nano_omni.py \\
        > /tmp/nemotron3-calib.log 2>&1 &
    # (or detach via setsid pattern in CLAUDE.md)

Override via env:  BASE_MODEL, OUTPUT_DIR, RECIPE, NUM_SAMPLES, MAX_SEQ_LEN.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU calibration — keep both 3090s free

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _patch_nemotron_cache_typo() -> None:
    """Patch a typo in NVIDIA's published modeling.py: it calls
    `create_causal_mask(input_embeds=...)` (singular), but the transformers 5
    signature is `inputs_embeds=` (plural). The kwarg is rejected as unknown
    and every forward call dies before the GPTQ Hessian phase.

    Must run *before* any AutoModel.from_pretrained(...trust_remote_code=True)
    so the patch lands on disk before transformers imports the dynamic module.
    Idempotent — re-runs do nothing once the typo is fixed.
    """
    import glob
    root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    for path in glob.glob(f"{root}/Nemotron*/*/modeling*.py"):
        with open(path) as fh:
            src = fh.read()
        if "input_embeds=" in src and "inputs_embeds=" not in src.replace(
            "input_embeds=", ""
        ):
            patched = src.replace("input_embeds=", "inputs_embeds=")
            with open(path, "w") as fh:
                fh.write(patched)
            print(f"  patched {os.path.basename(path)} (input_embeds → inputs_embeds)")


_patch_nemotron_cache_typo()

from calibration_datasets import (
    build_calibration_dataset,
    rows_to_text,
    tokenize_text_dataset,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
BASE_MODEL = os.environ.get(
    "BASE_MODEL", "/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{MODELS_DIR}/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-W4A16-CT",
)
RECIPE = os.environ.get("RECIPE", "omni_thinking_tools")
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1024"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQ_LEN", "2048"))

# Per-layer-index ignore — Nemotron-H uses a single `mixer` module name for
# BOTH Mamba2 AND attention layers (distinguished only by layer type), so
# generic `re:.*mixer.*` would over-quantize. Use the explicit index list.
MAMBA_LAYERS = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]
ATTN_LAYERS = [5, 12, 19, 26, 33, 42]
KEEP_BF16_LAYERS = sorted(MAMBA_LAYERS + ATTN_LAYERS)
KEEP_BF16_LAYER_RE = "|".join(str(i) for i in KEEP_BF16_LAYERS)  # "0|2|4|5|7|...|50"

IGNORE_PATTERNS = [
    "lm_head",
    # All Mamba2 + Attention layers stay BF16
    rf"re:^.*\.layers\.({KEEP_BF16_LAYER_RE})\..*$",
    # MoE routers / gates
    "re:.*router.*",
    r"re:.*\.gate$",
    "re:.*_gate$",
    # Vision encoder (CRADIO v4-H) + image projector
    "re:.*vision_tower.*",
    "re:.*radio.*",
    "re:.*image_embed.*",
    "re:.*image_projector.*",
    "re:.*multi_modal_projector.*",
    # Audio encoder (Parakeet) + audio projector
    "re:.*audio_tower.*",
    "re:.*sound_tower.*",
    "re:.*parakeet.*",
    "re:.*audio_embed.*",
    "re:.*sound_embed.*",
    "re:.*audio_projector.*",
    "re:.*sound_projector.*",
    # Embeddings
    "re:.*embed_tokens.*",
]

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Base:   {BASE_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"Recipe: {RECIPE}")
print(f"RAM:    {ram_gb:.1f} GB")
print(f"Calibration: {NUM_CALIBRATION_SAMPLES} samples x {MAX_SEQUENCE_LENGTH} tokens")
print(f"KEEP-BF16 layers ({len(KEEP_BF16_LAYERS)}): {KEEP_BF16_LAYERS}")
print(f"INT4-eligible MLP/MoE layers (52 - {len(KEEP_BF16_LAYERS)} = {52 - len(KEEP_BF16_LAYERS)})")

# --- 1. Build calibration dataset ---
print("\n[1/5] Building calibration dataset...")
rows = build_calibration_dataset(
    recipe=RECIPE,
    num_samples=NUM_CALIBRATION_SAMPLES,
    seed=42,
)

# --- 2. Tokenizer + chat template ---
print("\n[2/5] Loading tokenizer + rendering chat template...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
# The Omni BF16 ships a 14277-byte chat_template.jinja with thinking + tool-call
# XML format. Use it as-is.
if tokenizer.chat_template is None:
    template_path = os.path.join(BASE_MODEL, "chat_template.jinja")
    with open(template_path) as f:
        tokenizer.chat_template = f.read()
    print(f"  Loaded chat_template.jinja from {template_path}")

text_dataset = rows_to_text(
    rows,
    tokenizer,
    # Reasoning is default ON in this model; calibration data must exercise
    # the </think> termination pathway (otherwise thinking degenerates).
    enable_thinking=True,
    # Vision / audio encoders aren't quantized; LM sees placeholder tokens
    # only. Same approach as Gemma4 / Devstral calibrations.
    drop_images=True,
    max_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Rendered {len(text_dataset)} calibration rows")

# Verify the tool-call XML pathway made it into calibration text — the model's
# tool format is <tool_call><function=NAME>... (qwen3_coder shape), so we
# spot-check for those tokens. Hermes formatter renders into [TOOL_CALLS]-
# style Mistral format by default — we may need to re-format. For now we
# accept either: count both shapes.
joined = "\n".join(r["text"] for r in text_dataset)
n_tool_xml = joined.count("<tool_call>")
n_tool_mistral = joined.count("[TOOL_CALLS]")
n_think = joined.count("<think>")
print(f"  tool-format coverage: <tool_call>={n_tool_xml}  [TOOL_CALLS]={n_tool_mistral}")
print(f"  thinking coverage:    <think>={n_think}")
if (n_tool_xml + n_tool_mistral) < 10:
    raise RuntimeError(
        f"Tool-format pathway under-represented "
        f"(<tool_call>={n_tool_xml}, [TOOL_CALLS]={n_tool_mistral}). "
        f"Hermes mix didn't render — fix before calibrating."
    )
if n_think < 10:
    raise RuntimeError(
        f"Thinking pathway under-represented (<think>={n_think}). "
        f"am_thinking + numina mix didn't render — fix before calibrating."
    )

dataset = tokenize_text_dataset(text_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
print(f"Tokenized {len(dataset)} samples at max_seq_len={MAX_SEQUENCE_LENGTH}")

# --- 3. Load model on CPU (full Omni wrapper; encoders stay attached) ---
# Note: NemotronH_Nano_Omni_Reasoning_V3 is a custom architecture exposed via
# trust_remote_code. We use AutoModelForCausalLM since the wrapper extends a
# causal LM interface (per auto_map in config.json). If the smoke step (#21)
# reveals this loads only the LLM and not the full Omni, switch to AutoModel
# (generic) and add a `target_module="llm"`-style hint.
print("\n[3/5] Loading model on CPU (trust_remote_code=True, may be slow)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # Force eager attention on CPU calibration — transformers 5 otherwise picks
    # flash_attention_2 from the config and raises ImportError because flash-attn
    # is a CUDA-only kernel and the quant env is CPU-only (CUDA_VISIBLE_DEVICES="").
    attn_implementation="eager",
)
print(f"Model loaded in {time.time() - t0:.0f}s ({type(model).__name__})")
print(f"  Parameter count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# --- 4. GPTQ W4A16 calibration ---
print("\n[4/5] Running GPTQ calibration...")
print(f"  Ignore patterns ({len(IGNORE_PATTERNS)}):")
for p in IGNORE_PATTERNS:
    print(f"    {p}")

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=IGNORE_PATTERNS,
    # MoE: ensure every expert sees calibration mass; without this, rare
    # experts get garbage scales (proven across our REAM / REAP / 128-expert
    # MoE ships). Cost: ~3x calibration time vs default routing-only.
    # llmcompressor >=0.11 moved `moe_calibrate_all_experts` from GPTQModifier
    # onto the `oneshot()` call below, where it defaults to True — so all-experts
    # coverage is preserved without an explicit knob.
    offload_hessians=True,
    # Nemotron-3-Nano-Omni expert MLPs have down_proj columns=1856 (not divisible
    # by group_size=128). Without this flag llmcompressor raises ValueError before
    # GPTQ starts. SGLang ≥v0.5.11's AWQ-Marlin loader handles non-divisible
    # dimensions via a torch-dequant fallback (rules-for-agents.md → AWQ checkpoint
    # format), so the served model still runs — just dequants those Linears in
    # software instead of the Marlin INT4 fast path. Better than excluding ~33%
    # of expert weight (the down_projs) from INT4 entirely.
    bypass_divisibility_checks=True,
)

t0 = time.time()
# Target the inner NemotronHForCausalLM, NOT the multimodal wrapper:
#  - The wrapper's `forward(pixel_values, image_flags=None, ...)` unconditionally
#    runs `image_flags.squeeze(-1)`, which AttributeErrors on every text-only
#    calibration row (~7/8 of the mix). Both `independent` and `sequential`
#    pipelines AST-trace the entry point and trip on it.
#  - We only need INT4 on the language-model MLP/MoE Linears anyway: vision
#    (CRADIO), audio (Parakeet), routers, embeddings, lm_head are all already
#    in IGNORE_PATTERNS and stay BF16.
#  - The wrapper module remains attached on `model` so the saved checkpoint
#    still ships the encoders + Omni assembly; oneshot just writes the
#    quantized weights inside `language_model.*`.
# This matches the script's pre-flight note about a `target_module="llm"` hint.
oneshot(
    model=model.language_model,
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

# Save processor (image + audio configs) so downstream serving has both
# encoders' preprocessing pipelines.
try:
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    proc.save_pretrained(OUTPUT_DIR)
    print("  Saved processor (image + audio config)")
except Exception as e:
    print(f"  WARN: could not save AutoProcessor ({e!r})")
    # Fallback: copy the relevant config files directly. The Omni model
    # ships configuration_*.py + image_processing.py + audio_model.py
    # alongside config.json — these need to land in OUTPUT_DIR for the
    # trust_remote_code path to work at serve time.
    import shutil
    for fname in (
        "configuration.py", "configuration_nemotron_h.py", "configuration_radio.py",
        "image_processing.py", "audio_model.py", "evs.py",
        "chat_template.jinja", "generation_config.json",
        "preprocessor_config.json",  # if it exists
    ):
        src = os.path.join(BASE_MODEL, fname)
        if os.path.isfile(src):
            shutil.copy2(src, OUTPUT_DIR)
            print(f"  Copied {fname}")

print("\nDone.")
print("Next:")
print(f"  1. CT->AWQ:    python scripts/quantize/convert_moe_ct_to_awq.py {OUTPUT_DIR} \\")
print(f"                   /data/models/hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ \\")
print(f"                   --group-size 128")
print(f"  2. Scale audit: python scripts/eval/check_awq_scales.py <awq-dir>  (non-zero exit = do NOT ship)")
print(f"  3. Validate:    scripts/launch.sh nemotron3-omni  + 6-modality probe (#25)")
