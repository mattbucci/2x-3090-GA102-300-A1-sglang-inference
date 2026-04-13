#!/usr/bin/env python3
"""Microbenchmark for DeltaNet decode step on sm_86.

Isolates the GDN (Gated DeltaNet) kernel to measure per-layer decode time
without SGLang serving overhead. Helps identify if the bottleneck is:
- The Triton GDN kernel itself
- The conv1d kernel
- The state update
- The MLP/attention interleave

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test/bench_deltanet_decode.py
    ncu --target-processes all python scripts/test/bench_deltanet_decode.py  # kernel profiling
"""
import torch
import time
import sys

# Qwen3.5-27B dimensions
HIDDEN = 5120
NUM_HEADS_V = 48    # linear_num_value_heads
NUM_HEADS_K = 16    # linear_num_key_heads
HEAD_DIM_V = 128    # linear_value_head_dim
HEAD_DIM_K = 128    # linear_key_head_dim
CONV_KERNEL = 4     # linear_conv_kernel_dim
NUM_LAYERS = 48     # total layers (36 DeltaNet + 12 attention)
DELTANET_LAYERS = 36

DEVICE = "cuda"
DTYPE = torch.float16
BATCH = 1

print(f"DeltaNet Decode Microbenchmark")
print(f"Hidden: {HIDDEN}, Heads V: {NUM_HEADS_V}, Head dim: {HEAD_DIM_V}")
print(f"DeltaNet layers: {DELTANET_LAYERS}")
print(f"Device: {torch.cuda.get_device_name()}")
print()

# Simulate one DeltaNet layer's decode step
# Key operations:
# 1. Linear projections (QKV, gate, output)
# 2. Conv1d state update
# 3. Delta rule recurrence: S(t) = g*S(t-1) + delta
# 4. Output projection

# Allocate weights (simulating AWQ dequantized)
qkv_weight = torch.randn(HIDDEN, (NUM_HEADS_V + NUM_HEADS_K * 2) * HEAD_DIM_V, device=DEVICE, dtype=DTYPE)
gate_weight = torch.randn(HIDDEN, NUM_HEADS_V, device=DEVICE, dtype=DTYPE)
out_weight = torch.randn(NUM_HEADS_V * HEAD_DIM_V, HIDDEN, device=DEVICE, dtype=DTYPE)

# Conv1d state: [batch, hidden, conv_kernel-1]
conv_state = torch.randn(BATCH, NUM_HEADS_V * HEAD_DIM_V, CONV_KERNEL - 1, device=DEVICE, dtype=DTYPE)

# Recurrent state: S in [batch, num_heads, head_dim_k, head_dim_v]
recurrent_state = torch.randn(BATCH, NUM_HEADS_V, HEAD_DIM_K, HEAD_DIM_V, device=DEVICE, dtype=DTYPE)

# Input token
x = torch.randn(BATCH, HIDDEN, device=DEVICE, dtype=DTYPE)

# Warmup
for _ in range(10):
    # QKV projection
    qkv = x @ qkv_weight
    # Gate
    g = torch.sigmoid(x @ gate_weight)
    # Conv1d update (simplified)
    conv_input = qkv[:, :NUM_HEADS_V * HEAD_DIM_V].unsqueeze(-1)
    conv_state = torch.cat([conv_state[:, :, 1:], conv_input], dim=-1)
    conv_out = conv_state.sum(dim=-1)
    # Delta rule (simplified): S = g * S + k^T @ v
    k = conv_out.view(BATCH, NUM_HEADS_V, HEAD_DIM_V)[:, :, :HEAD_DIM_K]
    v = conv_out.view(BATCH, NUM_HEADS_V, HEAD_DIM_V)
    delta = torch.einsum("bnh,bnv->bnhv", k, v)
    recurrent_state = g.unsqueeze(-1).unsqueeze(-1) * recurrent_state + delta
    # Output
    out = torch.einsum("bnhv,bnh->bnv", recurrent_state, k)
    out_flat = out.reshape(BATCH, -1)
    y = out_flat @ out_weight

torch.cuda.synchronize()

# Benchmark individual operations
def bench(name, fn, iters=100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"  {name:<30s}: {ms:>8.3f} ms")
    return ms

print("=== Per-operation timing (1 layer, 1 token) ===")

t_qkv = bench("QKV projection", lambda: x @ qkv_weight)
t_gate = bench("Gate projection", lambda: torch.sigmoid(x @ gate_weight))
t_conv = bench("Conv1d state update", lambda: torch.cat([conv_state[:, :, 1:], qkv[:, :NUM_HEADS_V*HEAD_DIM_V].unsqueeze(-1)], dim=-1).sum(dim=-1))

k = conv_out.view(BATCH, NUM_HEADS_V, HEAD_DIM_V)[:, :, :HEAD_DIM_K]
v = conv_out.view(BATCH, NUM_HEADS_V, HEAD_DIM_V)
t_delta = bench("Delta rule (einsum k^T@v)", lambda: torch.einsum("bnh,bnv->bnhv", k, v))
t_state = bench("State update (g*S + delta)", lambda: g.unsqueeze(-1).unsqueeze(-1) * recurrent_state + delta)
t_out_ein = bench("Output (einsum S@k)", lambda: torch.einsum("bnhv,bnh->bnv", recurrent_state, k))
t_out_proj = bench("Output projection", lambda: out_flat @ out_weight)

total_per_layer = t_qkv + t_gate + t_conv + t_delta + t_state + t_out_ein + t_out_proj
total_model = total_per_layer * DELTANET_LAYERS

print()
print(f"  {'Total per layer':<30s}: {total_per_layer:>8.3f} ms")
print(f"  {'Total model ({DELTANET_LAYERS} DeltaNet layers)':<30s}: {total_model:>8.3f} ms")
print(f"  {'Theoretical tok/s':<30s}: {1000/total_model:>8.1f}")
print()
print("Note: Real SGLang kernel has additional overhead from Triton JIT,")
print("conv_state management, and mamba2 metadata. This is a lower bound.")
