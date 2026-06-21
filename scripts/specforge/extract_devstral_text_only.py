#!/usr/bin/env python3
"""Extract a text-only Devstral CausalLM from the VLM-wrapped AWQ checkpoint.

SpecForge EAGLE3 expects a plain CausalLM target. Devstral ships as
`Mistral3ForConditionalGeneration` (vision + language), whose `.language_model`
is a *base* model (no lm_head -> no `outputs.logits`), which breaks
`generate_eagle3_data`. This re-keys the AWQ tensors into a standalone
`Ministral3ForCausalLM` (text decoder only) WITHOUT re-quantizing — pure key
rename + drop vision tensors. Then SpecForge loads it via vanilla
AutoModelForCausalLM (returns logits, layers at .model.layers, embedding at
model.embed_tokens) — no VLM-extract patch, no --embedding-key override needed.

  language_model.model.*  -> model.*
  language_model.lm_head.* -> lm_head.*
  (drop) model.vision_tower.* / model.multi_modal_projector.* / *vision* / *patch_merger*

Usage: python extract_devstral_text_only.py <src_awq_dir> <dst_dir>
"""
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DROP = ("vision_tower", "multi_modal_projector", "patch_merger", "vision_")


def remap(key):
    if any(d in key for d in DROP):
        return None
    if key.startswith("language_model.model."):
        return "model." + key[len("language_model.model."):]
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head."):]
    if key.startswith("language_model."):
        return key[len("language_model."):]
    # already-flat or non-language top-level (drop vision-side leftovers)
    return None if any(d in key for d in DROP) else key


def main():
    src, dst = sys.argv[1], sys.argv[2]
    Path(dst).mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(src).glob("*.safetensors"))
    print(f"src={src}  {len(shards)} shards -> dst={dst}")
    kept = dropped = 0
    for shard in shards:
        out = {}
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                nk = remap(k)
                if nk is None:
                    dropped += 1
                    continue
                out[nk] = f.get_tensor(k)
                kept += 1
        save_file(out, os.path.join(dst, shard.name), metadata={"format": "pt"})
        print(f"  {shard.name}: wrote {len(out)} tensors")
    print(f"kept={kept} dropped={dropped}")

    # config.json: promote text_config to top-level Ministral3ForCausalLM
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg.get("text_config", cfg)
    tc = dict(tc)
    tc["architectures"] = ["Ministral3ForCausalLM"]
    if "quantization_config" in cfg:
        tc["quantization_config"] = cfg["quantization_config"]
    tc.setdefault("torch_dtype", cfg.get("torch_dtype", "bfloat16"))
    for tok_key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if tok_key in cfg and tok_key not in tc:
            tc[tok_key] = cfg[tok_key]
    json.dump(tc, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"config.json: model_type={tc.get('model_type')} arch={tc['architectures']}")

    # rebuild a weight index over the remapped keys
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        new_wm = {nk: wm[k] for k in wm if (nk := remap(k))}
        total = sum(os.path.getsize(os.path.join(dst, s.name)) for s in shards)
        json.dump({"metadata": {"total_size": total}, "weight_map": new_wm},
                  open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
        print(f"index: {len(new_wm)} entries")

    # copy tokenizer / processor / generation config (text-relevant only)
    for fn in os.listdir(src):
        if any(fn.startswith(p) for p in ("tokenizer", "special_tokens", "generation_config", "chat_template")) \
           or fn in ("vocab.json", "merges.txt"):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    print("tokenizer/generation files copied. DONE.")


if __name__ == "__main__":
    main()
