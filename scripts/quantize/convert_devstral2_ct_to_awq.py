#!/usr/bin/env python3
"""Convert Devstral-Small-2-24B (Mistral3) compressed-tensors -> native AWQ.

Unlike the older convert_devstral_ct_to_awq.py (which assumed a text-only
Ministral3ForCausalLM CT and re-merged vision from a separate original), this
checkpoint was calibrated as the FULL Mistral3ForConditionalGeneration via
AutoModelForImageTextToText, so:
  - keys are already in VLM format (model.language_model.model.layers.*,
    model.vision_tower.*, model.multi_modal_projector.*, language_model.lm_head)
    -> NO key remapping.
  - vision_tower + multi_modal_projector + lm_head are already present in-place
    as BF16 (they were in the GPTQ ignore list) -> pass through to FP16, NO
    copy-from-original.

Only the language-model Linear weights (*.weight_packed) are converted to AWQ
qweight/scales/qzeros (symmetric, zero_point=8). Same packing as the gemma-4-31B
and qwen converters. modules_to_not_convert matches the fp8 base + community AWQ.

Usage:
    CT_INPUT=/data/models/Devstral-Small-2-24B-2512-AWQ-CT-code-vision-tools \
    AWQ_OUTPUT=/data/models/Devstral-Small-2-24B-2512-AWQ \
    python scripts/quantize/convert_devstral2_ct_to_awq.py
"""
import gc
import glob
import json
import os
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
SRC_DIR = os.environ.get("CT_INPUT", f"{MODELS_DIR}/Devstral-Small-2-24B-2512-AWQ-CT-code-vision-tools")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/Devstral-Small-2-24B-2512-AWQ")

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    raise SystemExit(1)

# group_size from config
with open(os.path.join(SRC_DIR, "config.json")) as f:
    _cfg = json.load(f)
GROUP_SIZE = 128
for group_cfg in _cfg.get("quantization_config", {}).get("config_groups", {}).values():
    gs = group_cfg.get("weights", {}).get("group_size")
    if gs:
        GROUP_SIZE = gs
        break

print(f"Source:     {SRC_DIR}")
print(f"Output:     {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def strip_model_prefix(k: str) -> str:
    """The full Mistral3ForConditionalGeneration save_pretrained prefixes every
    key with an extra leading `model.` (model.language_model.model.layers.*,
    model.vision_tower.*, model.multi_modal_projector.*). SGLang's Mistral3
    loader — and the working community AWQ — expect the HF-standard layout
    WITHOUT that leading prefix (language_model.model.*, vision_tower.*,
    multi_modal_projector.*; lm_head stays language_model.lm_head). Strip it.
    Getting this wrong loads weights into the wrong slots -> <unk> garbage."""
    return k[len("model."):] if k.startswith("model.") else k


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    assert values.shape[-1] % PACK_FACTOR == 0
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


# Copy non-weight files (config, tokenizer, chat template, system prompts, etc.)
for fname in (
    glob.glob(f"{SRC_DIR}/*.json")
    + glob.glob(f"{SRC_DIR}/*.txt")
    + glob.glob(f"{SRC_DIR}/*.model")
    + glob.glob(f"{SRC_DIR}/*.jinja")
):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")

shard_files = sorted(glob.glob(f"{SRC_DIR}/model*.safetensors"))
if not shard_files:
    print(f"No model*.safetensors in {SRC_DIR}")
    raise SystemExit(1)

print(f"\nBuilding cross-shard index for {len(shard_files)} shards...")
key_to_shard = {}
for sp in shard_files:
    with safe_open(sp, framework="pt") as sf:
        for k in sf.keys():
            key_to_shard[k] = sp

weight_map = {}
total_converted = total_kept = 0

for shard_path in shard_files:
    shard_name = os.path.basename(shard_path)
    print(f"\n=== {shard_name} ===")
    f = safe_open(shard_path, framework="pt")
    keys = list(f.keys())
    converted = OrderedDict()
    processed = set()

    for key in keys:
        if key in processed:
            continue
        if key.endswith(".weight_packed"):
            base = key[: -len(".weight_packed")]
            packed = f.get_tensor(key)
            scale_key = f"{base}.weight_scale"
            if scale_key not in key_to_shard:
                print(f"  SKIP {base}: scale missing")
                continue
            if scale_key in keys:
                scale = f.get_tensor(scale_key)
            else:
                with safe_open(key_to_shard[scale_key], framework="pt") as sf2:
                    scale = sf2.get_tensor(scale_key)

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR
            unpacked = unpack_int32_to_4bit(packed)
            qweight = pack_4bit_to_int32_awq(unpacked.T.contiguous())
            scales = scale.T.contiguous().clamp(-65504, 65504).to(torch.float16)

            num_groups = in_features // GROUP_SIZE
            num_out_packed = out_features // PACK_FACTOR
            qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
            zp = torch.tensor([8], dtype=torch.int32)
            for i in range(PACK_FACTOR):
                qzeros |= zp << (AWQ_PACK_ORDER[i] * W_BIT)

            ob = strip_model_prefix(base)
            converted[f"{ob}.qweight"] = qweight
            converted[f"{ob}.scales"] = scales
            converted[f"{ob}.qzeros"] = qzeros
            processed.update({key, scale_key, f"{base}.weight_shape"})
            total_converted += 1
        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue
        else:
            t = f.get_tensor(key)
            converted[strip_model_prefix(key)] = t.to(torch.float16) if t.dtype == torch.bfloat16 else t
            total_kept += 1

    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)
    for k in converted:
        weight_map[k] = shard_name
    print(f"  Saved {len(converted)} tensors")
    del converted
    gc.collect()

# index
index = {
    "metadata": {"total_size": sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        for fn in os.listdir(OUTPUT_DIR) if fn.endswith(".safetensors"))},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# config -> AWQ
cfg_path = os.path.join(OUTPUT_DIR, "config.json")
with open(cfg_path) as f:
    config = json.load(f)
config["quantization_config"] = {
    "bits": W_BIT,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": ["lm_head", "vision_tower", "multi_modal_projector"],
}
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)

# Embed chat template into tokenizer_config if present and missing
jinja = os.path.join(OUTPUT_DIR, "chat_template.jinja")
tc = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
if os.path.exists(jinja) and os.path.exists(tc):
    with open(jinja) as jf:
        tmpl = jf.read()
    with open(tc) as tf:
        tcfg = json.load(tf)
    if "chat_template" not in tcfg:
        tcfg["chat_template"] = tmpl
        with open(tc, "w") as tf:
            json.dump(tcfg, tf, indent=2, ensure_ascii=False)
        print("  Embedded chat_template into tokenizer_config.json")

gb = index["metadata"]["total_size"] / 1024**3
print(f"\nDone. quantized={total_converted} kept_fp16={total_kept} size={gb:.1f}GB")
print(f"Output: {OUTPUT_DIR}")
