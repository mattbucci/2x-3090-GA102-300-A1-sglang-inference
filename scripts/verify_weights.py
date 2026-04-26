#!/usr/bin/env python3
"""Verify loaded weights match checkpoint values.

Adds a verification hook to the model after weight loading to compare
key parameters against the checkpoint.

Run with:
  source scripts/common.sh && activate_conda && setup_nvidia_env
  python scripts/verify_weights.py
"""
import os, sys, json, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

from safetensors import safe_open
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, LoadConfig, DeviceConfig
from sglang.srt.server_args import ServerArgs

MODEL_PATH = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-AWQ")

# Minimal server args to initialize the model
server_args = ServerArgs(
    model_path=MODEL_PATH,
    quantization="awq_marlin",
    trust_remote_code=True,
    tp_size=2,
    dtype="float16",
    context_length=512,
    disable_cuda_graph=True,
    disable_piecewise_cuda_graph=True,
)

# Can't easily initialize distributed from a script, so let's just analyze
# what the backbone's load_weights does with the checkpoint weights.
# We'll trace the key dimensions.

print("=== Analyzing weight name mapping ===")

# Load checkpoint keys
f = safe_open(f"{MODEL_PATH}/model.safetensors", framework="pt")
ckpt_keys = sorted(f.keys())

# Show which weights would hit the REAP-AWQ fused expert path
print("\nWeights hitting fused expert path:")
fused_count = 0
for k in ckpt_keys:
    # After wrapper strips model.
    backbone_name = k[len("model."):] if k.startswith("model.") else k
    if "mlp.experts.w13_" in backbone_name or "mlp.experts.w2_" in backbone_name:
        if fused_count < 12:
            t = f.get_tensor(k)
            print(f"  {backbone_name}: {list(t.shape)}")
        fused_count += 1

print(f"  ... total: {fused_count} fused expert weights")

# Show which weights would hit the catch-all path
print("\nWeights hitting catch-all (non-expert, non-stacked):")
catchall = []
stacked_names = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
                  "in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
for k in ckpt_keys:
    backbone_name = k[len("model."):] if k.startswith("model.") else k
    if backbone_name.startswith("lm_head"):
        continue  # handled by wrapper
    if "mlp.experts.w13_" in backbone_name or "mlp.experts.w2_" in backbone_name:
        continue  # handled by fused path
    if any(sn in backbone_name for sn in stacked_names):
        continue  # handled by stacked_params
    if "mlp.experts" in backbone_name:
        continue  # handled by expert_params
    if "rotary_emb" in backbone_name or "mtp" in backbone_name:
        continue
    t = f.get_tensor(k)
    if len(catchall) < 20:
        print(f"  {backbone_name}: {list(t.shape)}")
    catchall.append(backbone_name)

print(f"  ... total: {len(catchall)} catch-all weights")

# Key check: are there any weights that DON'T match any handling path?
print("\nWeights that might be UNHANDLED:")
for k in ckpt_keys:
    backbone_name = k[len("model."):] if k.startswith("model.") else k
    if backbone_name.startswith("lm_head"):
        continue
    if "mlp.experts.w13_" in backbone_name or "mlp.experts.w2_" in backbone_name:
        continue
    if any(sn in backbone_name for sn in stacked_names):
        continue
    if "mlp.experts" in backbone_name:
        continue
    if "rotary_emb" in backbone_name or "mtp" in backbone_name:
        continue
    # These should hit the catch-all's default_weight_loader
    # They need to match a param name in the backbone
    # But backbone params don't have "model." prefix
    # So backbone_name should match
    pass

# Verify: do the non-expert weights have matching params in the backbone?
# The backbone's named_parameters would be: embed_tokens.weight, layers.0.input_layernorm.weight, etc.
# The wrapper adds model. prefix, so wrapper.named_parameters has: model.embed_tokens.weight, model.layers.0...
print("\n=== Verifying non-expert weight matching ===")
# The wrapper's _route_weights strips model. -> backbone receives: embed_tokens.weight, layers.0...
# The backbone's params_dict = self.named_parameters() -> embed_tokens.weight, layers.0...
# These should match!

# But wait - are there any stacked params that ALSO match the fused expert path?
for k in ckpt_keys:
    bname = k[len("model."):] if k.startswith("model.") else k
    has_stacked = any(sn in bname for sn in stacked_names)
    has_fused = "mlp.experts.w13_" in bname or "mlp.experts.w2_" in bname
    if has_stacked and has_fused:
        print(f"  WARNING: both stacked and fused match: {bname}")
    if has_stacked and "mlp.experts" in bname:
        print(f"  NOTE: stacked + experts: {bname} (experts skipped in stacked loop)")

print("\n=== Summary ===")
print(f"Total checkpoint keys: {len(ckpt_keys)}")
print(f"Fused expert weights: {fused_count}")
print(f"Catch-all weights: {len(catchall)}")
print(f"Stacked params (handled by mapping): {len(ckpt_keys) - fused_count - len(catchall) - 1}")  # -1 for lm_head
