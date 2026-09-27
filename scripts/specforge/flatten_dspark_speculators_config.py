#!/usr/bin/env python3
"""Flatten RedHatAI's vLLM speculators-format DSpark config to the SGLang draft
layout, per R9700 6b59279's recipe + what parse_dspark_draft_config actually
reads: lift transformer_layer_config to top level, aux_hidden_state_layer_ids ->
target_layer_ids, drop mrope_section (draft is text-only), drop the speculators_*
/ auto_map wrapper. Weights are symlinked, not copied.
"""
import json, os, pathlib, sys

SRC = pathlib.Path("/data/models/Qwen3.8-27B-speculator.dspark")
DST = pathlib.Path("/data/models/Qwen3.8-27B-speculator.dspark-sgl")
DST.mkdir(exist_ok=True)

d = json.load(open(SRC / "config.json"))
tlc = dict(d.get("transformer_layer_config", {}))

# rope: keep theta/type, drop mrope_section (a target VL field; draft is text-only)
rope = dict(tlc.get("rope_parameters", {}))
rope.pop("mrope_section", None)

out = {
    "architectures": ["DSparkDraftModel"],   # in SGLang EntryClass
    "model_type": tlc.get("model_type", "qwen3"),
    # --- lifted transformer_layer_config (the draft backbone) ---
    "hidden_size": tlc["hidden_size"],
    "intermediate_size": tlc["intermediate_size"],
    "num_hidden_layers": tlc["num_hidden_layers"],
    "num_attention_heads": tlc["num_attention_heads"],
    "num_key_value_heads": tlc["num_key_value_heads"],
    "head_dim": tlc["head_dim"],
    "hidden_act": tlc.get("hidden_act", "silu"),
    "rms_norm_eps": tlc.get("rms_norm_eps", 1e-6),
    "max_position_embeddings": tlc.get("max_position_embeddings", 262144),
    "vocab_size": tlc["vocab_size"],
    "attention_bias": tlc.get("attention_bias", False),
    "tie_word_embeddings": d.get("tie_word_embeddings", False),
    "rope_theta": rope.get("rope_theta", 1e7),
    "rope_scaling": None,
    "sliding_window": tlc.get("sliding_window"),
    "use_sliding_window": tlc.get("use_sliding_window", False),
    "max_window_layers": tlc.get("max_window_layers"),
    "layer_types": tlc.get("layer_types"),
    "torch_dtype": d.get("dtype", "bfloat16"),
    # --- DSpark / DFlash draft fields ---
    "target_layer_ids": d["aux_hidden_state_layer_ids"],
    "target_hidden_size": tlc["hidden_size"],
    "draft_vocab_size": d.get("draft_vocab_size", tlc["vocab_size"]),
    "block_size": d["block_size"],
    "markov_rank": d["markov_rank"],
    "markov_head_type": d["markov_head_type"],
    "mask_token_id": d["mask_token_id"],
    "enable_confidence_head": d.get("enable_confidence_head", True),
    "confidence_head_with_markov": d.get("confidence_head_with_markov", True),
    "sample_from_anchor": d.get("sample_from_anchor", True),
    "sliding_window_non_causal": d.get("sliding_window_non_causal", False),
}
json.dump(out, open(DST / "config.json", "w"), indent=1)

# symlink weights + tokenizer-adjacent files (draft shares the target tokenizer,
# but ship what's here for standalone load)
for f in ["model.safetensors"]:
    link = DST / f
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(SRC / f)

print("wrote", DST / "config.json")
print(json.dumps(out, indent=1))
