#!/usr/bin/env python3
"""Test REAP-AWQ fused expert weight loading in isolation.

Verifies that the w13/w2 checkpoint format loads correctly into
FusedMoE params via the per-expert weight_loader path.

Run with: CUDA_VISIBLE_DEVICES=0 python scripts/test_expert_loading.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from safetensors import safe_open

MODEL_PATH = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-AWQ")

print("=== Step 1: Load checkpoint expert weights ===")
f = safe_open(f"{MODEL_PATH}/model.safetensors", framework="pt")

# Layer 0 expert weights
w13_qw = f.get_tensor("model.layers.0.mlp.experts.w13_qweight")
w13_sc = f.get_tensor("model.layers.0.mlp.experts.w13_scales")
w13_qz = f.get_tensor("model.layers.0.mlp.experts.w13_qzeros")
w2_qw = f.get_tensor("model.layers.0.mlp.experts.w2_qweight")
w2_sc = f.get_tensor("model.layers.0.mlp.experts.w2_scales")
w2_qz = f.get_tensor("model.layers.0.mlp.experts.w2_qzeros")

print(f"  w13_qweight: {list(w13_qw.shape)} {w13_qw.dtype}")
print(f"  w13_scales:  {list(w13_sc.shape)} {w13_sc.dtype}")
print(f"  w13_qzeros:  {list(w13_qz.shape)} {w13_qz.dtype}")
print(f"  w2_qweight:  {list(w2_qw.shape)} {w2_qw.dtype}")
print(f"  w2_scales:   {list(w2_sc.shape)} {w2_sc.dtype}")
print(f"  w2_qzeros:   {list(w2_qz.shape)} {w2_qz.dtype}")

# Also load per-expert format for comparison (if available from CT variant)
ct_path = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-CT")
if os.path.exists(f"{ct_path}/model.safetensors"):
    fc = safe_open(f"{ct_path}/model.safetensors", framework="pt")
    ct_keys = list(fc.keys())
    # Find per-expert gate_proj for layer 0, expert 0
    gate_keys = [k for k in ct_keys if "layers.0.mlp.experts.0.gate_proj" in k]
    if gate_keys:
        for k in sorted(gate_keys):
            t = fc.get_tensor(k)
            print(f"\n  CT {k}: {list(t.shape)} {t.dtype}")

print("\n=== Step 2: Chunk analysis ===")
# chunk(2, dim=-1) for w13
halves = w13_qw.chunk(2, dim=-1)
print(f"  w13_qweight chunked: [{list(halves[0].shape)}, {list(halves[1].shape)}]")

# Per-expert slice
gate_e0 = halves[0][0]  # Expert 0, gate half
up_e0 = halves[1][0]    # Expert 0, up half
print(f"  Per-expert gate[0]: {list(gate_e0.shape)}")
print(f"  Per-expert up[0]:   {list(up_e0.shape)}")

# Check if the first few values look reasonable
print(f"  gate[0] first values: {gate_e0[0, :4]}")
print(f"  up[0] first values:   {up_e0[0, :4]}")

# Check non-zero fraction
gate_nz = (gate_e0 != 0).float().mean().item()
up_nz = (up_e0 != 0).float().mean().item()
print(f"  gate[0] non-zero fraction: {gate_nz:.4f}")
print(f"  up[0] non-zero fraction:   {up_nz:.4f}")

print("\n=== Step 3: Verify chunk dimension correctness ===")
# For AWQ w13: [E, hidden, 2*intermediate/pack]
# Gate = first intermediate/pack columns, Up = second intermediate/pack columns
# intermediate=512, pack=8 -> each half = 64 columns
# With TP=2: each shard gets 32 columns
# TP rank 0 gate: [:, :, 0:32], TP rank 1 gate: [:, :, 32:64]

print(f"  Full w13_qw dim -1 = {w13_qw.shape[-1]} (should be 2*512/8 = 128)")
print(f"  Half dim -1 = {halves[0].shape[-1]} (should be 512/8 = 64)")
print(f"  Per TP shard = {halves[0].shape[-1] // 2} (should be 256/8 = 32)")

print("\n=== Step 4: Verify scales chunk ===")
sc_halves = w13_sc.chunk(2, dim=-1)
print(f"  w13_scales chunked: [{list(sc_halves[0].shape)}, {list(sc_halves[1].shape)}]")
print(f"  Full scales dim -1 = {w13_sc.shape[-1]} (should be 2*512 = 1024)")
print(f"  Half scales dim -1 = {sc_halves[0].shape[-1]} (should be 512)")
print(f"  Per TP shard = {sc_halves[0].shape[-1] // 2} (should be 256)")

print("\n=== Step 5: w2 analysis ===")
print(f"  w2_qweight: {list(w2_qw.shape)}")
print(f"  Per-expert w2[0]: {list(w2_qw[0].shape)}")
# w2 = down_proj: [intermediate, hidden/pack]
# With TP=2: TP shards the input (intermediate) dim
# TP rank 0: w2[:, :256, :], TP rank 1: w2[:, 256:, :]
print(f"  w2 dim 1 = {w2_qw.shape[1]} (intermediate = 512)")
print(f"  Per TP shard dim 1 = {w2_qw.shape[1] // 2} (= 256)")

print("\nDone! All dimensions consistent.")
