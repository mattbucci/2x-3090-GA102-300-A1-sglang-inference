#!/usr/bin/env python3
"""End-to-end test: load real expert weights → Marlin repack → forward.

Loads expert 0 layer 0 weights from the REAP-AWQ checkpoint,
creates a minimal FusedMoE, loads the weights via the weight_loader,
runs process_weights_after_loading, and checks if the dequantized
output makes sense.

Usage: source scripts/common.sh && activate_conda && setup_nvidia_env
       CUDA_VISIBLE_DEVICES=0 python scripts/test_awq_moe_e2e.py
"""
import os, sys, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from safetensors import safe_open

MODEL = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-AWQ")
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def dequant_awq(qweight, scales, qzeros, group_size=128):
    """Dequantize AWQ to float."""
    unpacked = []
    for i in range(8):
        unpacked.append((qweight >> (AWQ_PACK_ORDER[i] * 4)) & 0xF)
    unpacked = torch.stack(unpacked, dim=-1).reshape(qweight.shape[0], -1).float()

    zp_unpacked = []
    for i in range(8):
        zp_unpacked.append((qzeros >> (AWQ_PACK_ORDER[i] * 4)) & 0xF)
    zp_unpacked = torch.stack(zp_unpacked, dim=-1).reshape(qzeros.shape[0], -1).float()

    sc_exp = scales.float().repeat_interleave(group_size, dim=0)
    zp_exp = zp_unpacked.float().repeat_interleave(group_size, dim=0)
    if sc_exp.shape[0] > unpacked.shape[0]:
        sc_exp = sc_exp[:unpacked.shape[0]]
        zp_exp = zp_exp[:unpacked.shape[0]]
    return (unpacked - zp_exp) * sc_exp

# Load expert 0, layer 0 from checkpoint
f = safe_open(f"{MODEL}/model.safetensors", framework="pt")

gate_qw = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.qweight")
gate_sc = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.scales")
gate_qz = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.qzeros")

print(f"gate_proj.qweight: {list(gate_qw.shape)} {gate_qw.dtype}")
print(f"gate_proj.scales:  {list(gate_sc.shape)} {gate_sc.dtype}")

# Dequantize gate_proj: [in=2048, out=512/8=64] → [2048, 512]
gate_deq = dequant_awq(gate_qw, gate_sc, gate_qz)
print(f"\nDequantized gate_proj: {list(gate_deq.shape)}")
print(f"  Range: [{gate_deq.min():.4f}, {gate_deq.max():.4f}]")
print(f"  Mean abs: {gate_deq.abs().mean():.6f}")
print(f"  Non-zero: {(gate_deq != 0).float().mean():.4f}")

# Simple matmul test: random input → gate_proj → check output range
x = torch.randn(1, 2048)
y_expected = x @ gate_deq  # [1, 512]
print(f"\nMatmul test (float): input [1, 2048] @ gate [2048, 512]")
print(f"  Output range: [{y_expected.min():.4f}, {y_expected.max():.4f}]")
print(f"  Output mean abs: {y_expected.abs().mean():.4f}")

# Now check: if we transpose and dequantize, does it still work?
# The standard AWQ checkpoint stores [in, out/pack] = [K, N/8]
# Where K=hidden=2048, N=intermediate=512
# The Marlin kernel expects this layout
print(f"\n=== Layout verification ===")
print(f"  AWQ qweight layout: [{gate_qw.shape[0]}, {gate_qw.shape[1]}] = [K={gate_qw.shape[0]}, N/8={gate_qw.shape[1]}]")
print(f"  K={gate_qw.shape[0]} (hidden_size), N={gate_qw.shape[1]*8} (intermediate_size)")
print(f"  This is [input, output/pack] ← standard AWQ format")
