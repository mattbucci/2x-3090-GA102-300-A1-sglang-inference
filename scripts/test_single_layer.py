#!/usr/bin/env python3
"""Minimal test: load one expert, dequantize, matmul, compare.

Tests whether our AWQ checkpoint's expert weights produce correct
matmul results when dequantized vs when used through the Marlin kernel.

This bypasses SGLang entirely — just raw weight validation.
"""
import os, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from safetensors import safe_open

MODEL = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-AWQ")
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def dequant_awq_expert(qweight, scales, qzeros, group_size=128):
    """Dequantize one expert's AWQ weight to float. Shape: [K, N]"""
    unpacked = []
    for i in range(8):
        unpacked.append((qweight >> (AWQ_ORDER[i] * 4)) & 0xF)
    unpacked = torch.stack(unpacked, dim=-1).reshape(qweight.shape[0], -1).float()

    zp = []
    for i in range(8):
        zp.append((qzeros >> (AWQ_ORDER[i] * 4)) & 0xF)
    zp = torch.stack(zp, dim=-1).reshape(qzeros.shape[0], -1).float()

    sc_exp = scales.float().repeat_interleave(group_size, dim=0)
    zp_exp = zp.float().repeat_interleave(group_size, dim=0)
    if sc_exp.shape[0] > unpacked.shape[0]:
        sc_exp = sc_exp[:unpacked.shape[0]]
        zp_exp = zp_exp[:unpacked.shape[0]]
    return (unpacked - zp_exp) * sc_exp

# Load expert 0 gate + up + down from checkpoint
f = safe_open(f"{MODEL}/model.safetensors", framework="pt")

gate_deq = dequant_awq_expert(
    f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.qweight"),
    f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.scales"),
    f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.qzeros"),
)  # [2048, 512]

up_deq = dequant_awq_expert(
    f.get_tensor("model.layers.0.mlp.experts.0.up_proj.qweight"),
    f.get_tensor("model.layers.0.mlp.experts.0.up_proj.scales"),
    f.get_tensor("model.layers.0.mlp.experts.0.up_proj.qzeros"),
)  # [2048, 512]

down_deq = dequant_awq_expert(
    f.get_tensor("model.layers.0.mlp.experts.0.down_proj.qweight"),
    f.get_tensor("model.layers.0.mlp.experts.0.down_proj.scales"),
    f.get_tensor("model.layers.0.mlp.experts.0.down_proj.qzeros"),
)  # [512, 2048]

print(f"gate: {list(gate_deq.shape)}, range=[{gate_deq.min():.4f}, {gate_deq.max():.4f}]")
print(f"up:   {list(up_deq.shape)}, range=[{up_deq.min():.4f}, {up_deq.max():.4f}]")
print(f"down: {list(down_deq.shape)}, range=[{down_deq.min():.4f}, {down_deq.max():.4f}]")

# Forward pass: x → gate*up → SiLU → down
x = torch.randn(1, 2048)
gate_out = x @ gate_deq    # [1, 512]
up_out = x @ up_deq        # [1, 512]
activated = torch.nn.functional.silu(gate_out) * up_out  # [1, 512]
# down: [512, 2048], but down_deq is [K=512, N=2048]
result = activated @ down_deq  # [1, 512] @ [512, 2048] = [1, 2048]

print(f"\nExpert 0 forward (float):")
print(f"  Input: {list(x.shape)}, Output: {list(result.shape)}")
print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")
print(f"  Output mean abs: {result.abs().mean():.4f}")

# Also check embed_tokens
embed = f.get_tensor("model.embed_tokens.weight")  # [vocab, hidden]
print(f"\nEmbed tokens: {list(embed.shape)} {embed.dtype}")
print(f"  Range: [{embed.min():.4f}, {embed.max():.4f}]")
print(f"  Mean abs: {embed.abs().mean():.6f}")

# Check lm_head
lm = f.get_tensor("lm_head.weight")
print(f"lm_head: {list(lm.shape)} {lm.dtype}")
print(f"  Range: [{lm.min():.4f}, {lm.max():.4f}]")

# Quick end-to-end: embed → expert → lm_head logits
token_id = 2  # some token
emb = embed[token_id].unsqueeze(0)  # [1, 2048]
gate_o = emb @ gate_deq
up_o = emb @ up_deq
act = torch.nn.functional.silu(gate_o) * up_o
result = act @ down_deq  # [1, 2048]
logits = result @ lm.T.float()  # [1, vocab]
top5 = torch.topk(logits[0], 5)
print(f"\nEnd-to-end (token {token_id} → expert 0 → logits):")
print(f"  Top 5 logits: {top5.values.tolist()}")
print(f"  Top 5 token IDs: {top5.indices.tolist()}")
print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
print("  (If logits are all similar or NaN, weights are broken)")
