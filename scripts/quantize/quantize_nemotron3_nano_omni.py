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


def _patch_nemotron_typo_on_disk() -> None:
    """Patch a typo in NVIDIA's published modeling.py: it calls
    `create_causal_mask(input_embeds=...)` (singular), but the transformers 5
    signature is `inputs_embeds=` (plural). The kwarg is rejected as unknown
    and every forward call dies before the GPTQ Hessian phase.

    Transformers 5 may create a fresh hash dir under
    ~/.cache/huggingface/modules/transformers_modules/Nemotron*/ on every
    from_pretrained call, so the file patch alone is racy. We patch:

      1. The BASE_MODEL source dir (the file transformers *copies from* when
         populating a new cache hash dir) — this fixes future hashes.
      2. Every existing modeling*.py under the cache (any depth) — fixes
         already-resolved hashes that an in-flight loader may still read.

    The runtime monkey-patch in `_patch_create_causal_mask` is the real safety
    net; this on-disk pass is best-effort.
    Idempotent — the replace is a no-op once the typo is gone.
    """
    import glob
    roots = [
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules"),
        os.environ.get("BASE_MODEL", ""),
    ]
    candidates = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        # any depth — cache top-level + hash subdirs + BASE_MODEL root
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.startswith("modeling") and fn.endswith(".py"):
                    candidates.append(os.path.join(dirpath, fn))
    # Substitutions to apply:
    #   1. input_embeds=  →  inputs_embeds=  (NVIDIA typo, see docstring)
    #   2. `if use_cache and past_key_values is None:` →
    #      `if past_key_values is None:`
    #      Rationale: llmcompressor's autowrap traces NemotronHModel.forward.
    #      During the trace, past_key_values is a torch.fx proxy wrapping
    #      None. The downstream `_preprocess_mask_arguments` has the right
    #      None-guard (`if past_key_values is not None:`), but the proxy
    #      itself is not literally None — the guard passes, then
    #      `past_key_values.get_seq_length()` triggers __getattr__ on a
    #      proxy with `_metadata=None`, which fails with
    #      `AttributeError: 'NoneType' object has no attribute
    #      'get_seq_length'`. Forcing a real NemotronHHybridDynamicCache to
    #      always be constructed when None means the proxy wraps a real
    #      Cache object — metadata exists, get_seq_length resolves, mask
    #      computed normally. use_cache=False at runtime still applies
    #      (the cache is dropped at the return, line 1229).
    subs = [
        ("input_embeds=", "inputs_embeds="),
        (
            "if use_cache and past_key_values is None:",
            "if past_key_values is None:  # patched: always create cache for autowrap trace",
        ),
    ]
    for path in candidates:
        try:
            with open(path) as fh:
                src = fh.read()
        except OSError:
            continue
        new = src
        for old, repl in subs:
            new = new.replace(old, repl)
        if new != src:
            with open(path, "w") as fh:
                fh.write(new)
            applied = [old for old, repl in subs if old in src]
            print(f"  patched {path} ({', '.join(applied)})")


def _patch_create_causal_mask() -> None:
    """Wrap transformers.masking_utils.create_causal_mask so the typo'd
    `input_embeds=` kwarg is silently remapped to `inputs_embeds=`. This is
    the real fix — it works regardless of which cache hash dir NVIDIA's
    autowrapped modeling.py ends up importing from, because the import resolves
    against the in-process function object.

    Must run before any module that imports `create_causal_mask` (i.e. before
    from_pretrained triggers NVIDIA's modeling_nemotron_h.py to load).
    Idempotent — re-wrapping is a no-op after the first call.
    """
    try:
        import transformers.masking_utils as _mu
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: could not import transformers.masking_utils: {e}")
        return
    if getattr(_mu.create_causal_mask, "_nemotron_typo_wrapped", False):
        return
    import inspect
    _orig = _mu.create_causal_mask
    _accepted = set(inspect.signature(_orig).parameters.keys())

    def _wrapped(*args, **kwargs):
        # NVIDIA's modeling_nemotron_h.py was written against transformers
        # ~4.5x. Three differences against transformers 5 must be papered over
        # before the live signature will accept the call:
        #   - `input_embeds=` (typo of `inputs_embeds=`) — remap
        #   - `cache_position=` (docstring says "Deprecated and unused" in
        #     transformers 5, was removed from the signature) — drop
        #   - `past_key_values=None` — declared `Cache | None` in the type
        #     hint, but the body calls `.get_seq_length()` without a None
        #     guard. Substitute an empty DynamicCache so the seq length is
        #     reported as 0 (semantically identical to "no cache").
        # Generalize the unknown-kwarg case: drop anything else not in the
        # live signature, logging once so we notice if a future transformers
        # version drops something we actually need.
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        for k in [k for k in list(kwargs) if k not in _accepted]:
            if k not in _wrapped._logged_drops:
                print(f"  [create_causal_mask wrapper] dropping unknown kwarg '{k}'")
                _wrapped._logged_drops.add(k)
            kwargs.pop(k)
        if "past_key_values" in kwargs and kwargs["past_key_values"] is None:
            try:
                from transformers.cache_utils import DynamicCache
                kwargs["past_key_values"] = DynamicCache()
            except Exception:  # noqa: BLE001
                pass
        return _orig(*args, **kwargs)

    _wrapped._nemotron_typo_wrapped = True  # type: ignore[attr-defined]
    _wrapped._logged_drops = set()  # type: ignore[attr-defined]
    _mu.create_causal_mask = _wrapped
    print(f"  wrapped transformers.masking_utils.create_causal_mask "
          f"(accepted: {sorted(_accepted)})")


def _patch_accumulate_hessian() -> None:
    """llmcompressor 0.11.0's accumulate_hessian only handles 2D/3D Linear
    inputs (line 16: `if len(inp.shape) == 3: inp = inp.reshape(...)`). The
    NemotronH MoE block's shared_experts.up_proj sees a 4D input during the
    sequential pipeline trace — likely because autowrap inflates batched
    inputs across the routing dimension. `.t()` on 4D raises
    `t() expects a tensor with <= 2 dimensions`.

    Semantically the right thing is to flatten any N-dim Linear input to
    `[prod(leading_dims), in_features]` before the transpose — the GPTQ
    Hessian is `inp @ inp.t()` and is invariant to the leading dim shape.
    Patch the `== 3` check to `> 2`.

    Idempotent: re-runs are no-ops because the wrapper replaces the function
    object once and the marker attribute is checked.
    """
    try:
        import llmcompressor.modifiers.gptq.gptq_quantize as _gq
        import transformers as _tx
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: could not import llmcompressor.modifiers.gptq.gptq_quantize: {e}")
        return
    if getattr(_gq.accumulate_hessian, "_nemotron_ndim_patched", False):
        return
    import math
    import torch
    _orig = _gq.accumulate_hessian
    _GPTQ_PRECISION = _gq.GPTQ_PRECISION

    def _wrapped(inp, module, H, num_samples):
        inp = inp.to(device=H.device)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        num_added = inp.shape[0]
        if isinstance(module, (torch.nn.Linear, _tx.Conv1D)):
            # FIX: original was `if len(inp.shape) == 3:` — broken for 4D MoE inputs
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(module, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        num_samples += num_added
        inp = inp.to(dtype=_GPTQ_PRECISION)
        inp = math.sqrt(2) * inp
        H += inp.matmul(inp.t())
        return H, num_samples

    _wrapped._nemotron_ndim_patched = True  # type: ignore[attr-defined]
    _gq.accumulate_hessian = _wrapped
    print("  patched llmcompressor.modifiers.gptq.gptq_quantize.accumulate_hessian "
          "(handle any N-dim Linear input, not just 2D/3D)")


_patch_nemotron_typo_on_disk()
_patch_create_causal_mask()
_patch_accumulate_hessian()

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
