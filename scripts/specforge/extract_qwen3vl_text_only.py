#!/usr/bin/env python3
"""Extract a text-only Qwen3 CausalLM from the Qwen3-VL-32B AWQ checkpoint.

Same pattern as extract_devstral_text_only.py: SpecForge EAGLE3 needs a plain
CausalLM target with logits; the VL wrapper (`Qwen3VLForConditionalGeneration`)
nests the decoder at `.model.language_model` (base model, no lm_head there —
lm_head is top-level). Pure key rename + drop vision tensors, NO requant:

  model.language_model.*  -> model.*
  lm_head.*               -> lm_head.*   (already top-level)
  (drop) model.visual.*

Config: text_config promoted to top level, architectures=["Qwen3ForCausalLM"],
model_type="qwen3"; quantization_config carried over (AWQ int4 loads via
AutoModelForCausalLM+autoawq/gptq path or HF quantizer as with devstral).

Usage: python extract_qwen3vl_text_only.py <src_awq_dir> <dst_dir>
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def remap(key):
    if key.startswith("model.visual."):
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key  # lm_head.weight etc.


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    print(f"src={src}  {len(shards)} shards -> dst={dst}")
    weight_map, total = {}, 0
    kept = dropped = 0
    for i, shard in enumerate(shards, 1):
        out = {}
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                nk = remap(k)
                if nk is None:
                    dropped += 1
                    continue
                out[nk] = f.get_tensor(k)
                kept += 1
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        if out:
            save_file(out, str(dst / name), metadata={"format": "pt"})
            for k, t in out.items():
                weight_map[k] = name
                total += t.numel() * t.element_size()
        print(f"  {shard.name}: kept {len(out)}")
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(dst / "model.safetensors.index.json", "w"), indent=2)

    cfg = json.load(open(src / "config.json"))
    tc = cfg["text_config"]
    new = dict(tc)
    new["architectures"] = ["Qwen3ForCausalLM"]
    new["model_type"] = "qwen3"
    new["torch_dtype"] = cfg.get("torch_dtype", tc.get("torch_dtype", "float16"))
    if "quantization_config" in cfg:
        new["quantization_config"] = cfg["quantization_config"]
    json.dump(new, open(dst / "config.json", "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    print(f"done: kept {kept}, dropped {dropped} (vision), config+tokenizer written")


if __name__ == "__main__":
    main()
