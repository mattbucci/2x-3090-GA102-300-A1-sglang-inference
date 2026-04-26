#!/usr/bin/env python3
"""Profile the full DeltaNet decode pipeline on RTX 3090.

Measures per-operation timing for each kernel in the DeltaNet decode path
to identify the dominant bottleneck. Uses CUDA events for accurate GPU timing.

The decode path per DeltaNet layer has 6-7 kernel launches:
  1a. in_proj_qkvz GEMM (Marlin/cuBLAS)
  1b. in_proj_ba GEMM (Marlin/cuBLAS)
  2.  fused_qkvzba_split_reshape_cat (Triton)
  3.  causal_conv1d_update (Triton)
  4.  packed_decode recurrent (Triton)
  5.  RMSNorm + gating (Triton)
  6.  out_proj GEMM (Marlin/cuBLAS)

This script simulates each operation in isolation to measure timing,
then estimates total pipeline time vs actual SGLang overhead.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test/profile_deltanet_pipeline.py
    CUDA_VISIBLE_DEVICES=0 python scripts/test/profile_deltanet_pipeline.py --tp 1
"""
import argparse
import torch
import time

# Qwen3.5-27B dimensions (full model, before TP sharding)
HIDDEN_DIM = 5120        # model hidden size
NUM_HEADS_K = 16         # linear_num_key_heads
NUM_HEADS_V = 48         # linear_num_value_heads
HEAD_DIM_K = 128         # linear_key_head_dim
HEAD_DIM_V = 128         # linear_value_head_dim
CONV_KERNEL = 4          # linear_conv_kernel_dim
DELTANET_LAYERS = 36     # DeltaNet layers
ATTN_LAYERS = 12         # Full-attention layers
TOTAL_LAYERS = 48        # Total layers
INTER_SIZE = 14336       # MLP intermediate size

DEVICE = "cuda"
DTYPE = torch.float16


def bench_op(name, fn, warmup=20, iters=200):
    """Benchmark a single GPU operation using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--replicate-deltanet", action="store_true", default=True,
                        help="Replicate DeltaNet layers (no TP sharding)")
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    B = args.batch
    TP = args.tp

    # With DeltaNet replication, DeltaNet layers use full heads (no TP sharding)
    # Only attention layers and final MLP use TP
    if args.replicate_deltanet:
        dn_hv = NUM_HEADS_V       # 48 (replicated)
        dn_hk = NUM_HEADS_K       # 16 (replicated)
    else:
        dn_hv = NUM_HEADS_V // TP
        dn_hk = NUM_HEADS_K // TP

    K = HEAD_DIM_K
    V = HEAD_DIM_V

    # Derived dimensions
    qkvz_out = dn_hk * K * 2 + dn_hv * V + dn_hv * V  # Q + K + V + Z
    ba_out = dn_hv + dn_hv                               # B + A
    mixed_qkv_dim = dn_hk * K * 2 + dn_hv * V           # Q + K + V (no Z)
    conv_dim = mixed_qkv_dim
    out_proj_in = dn_hv * V
    out_proj_out = HIDDEN_DIM

    gpu_name = torch.cuda.get_device_name()
    print(f"{'='*70}")
    print(f"DeltaNet Full-Pipeline Profiling — Qwen3.5-27B")
    print(f"GPU: {gpu_name}  TP={TP}  Batch={B}")
    print(f"DeltaNet: HV={dn_hv} HK={dn_hk} K={K} V={V}")
    print(f"Replicate DeltaNet: {args.replicate_deltanet}")
    print(f"{'='*70}")
    print()

    # ========================================================================
    # 1a. in_proj_qkvz: [B, hidden] -> [B, qkvz_out]
    # ========================================================================
    # Simulated as AWQ Marlin GEMM (we use fp16 matmul as proxy since
    # Marlin requires special weight format)
    x = torch.randn(B, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    w_qkvz = torch.randn(qkvz_out, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    t_qkvz = bench_op("in_proj_qkvz", lambda: torch.mm(x, w_qkvz.T), iters=args.iters)

    # ========================================================================
    # 1b. in_proj_ba: [B, hidden] -> [B, ba_out]
    # ========================================================================
    w_ba = torch.randn(ba_out, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    t_ba = bench_op("in_proj_ba", lambda: torch.mm(x, w_ba.T), iters=args.iters)

    # ========================================================================
    # 2. fused_qkvzba_split: data rearrangement (cheap)
    # ========================================================================
    qkvz_out_tensor = torch.randn(B, qkvz_out, device=DEVICE, dtype=DTYPE)
    def split_reshape():
        q = qkvz_out_tensor[:, :dn_hk*K]
        k = qkvz_out_tensor[:, dn_hk*K:dn_hk*K*2]
        v = qkvz_out_tensor[:, dn_hk*K*2:dn_hk*K*2+dn_hv*V]
        z = qkvz_out_tensor[:, dn_hk*K*2+dn_hv*V:]
        return torch.cat([q, k, v], dim=-1)
    t_split = bench_op("qkvzba_split", split_reshape, iters=args.iters)

    # ========================================================================
    # 3. causal_conv1d_update: [B, conv_dim] conv with width=4
    # ========================================================================
    conv_input = torch.randn(B, conv_dim, device=DEVICE, dtype=DTYPE)
    conv_weight = torch.randn(conv_dim, CONV_KERNEL, device=DEVICE, dtype=DTYPE)
    conv_state = torch.randn(B, conv_dim, CONV_KERNEL - 1, device=DEVICE, dtype=DTYPE)
    def conv1d_update():
        # Simulated conv1d: full window multiply-accumulate (width=4)
        full_window = torch.cat([conv_state, conv_input.unsqueeze(-1)], dim=-1)
        out = (full_window * conv_weight.unsqueeze(0)).sum(dim=-1)
        return torch.nn.functional.silu(out)
    t_conv = bench_op("causal_conv1d", conv1d_update, iters=args.iters)

    # ========================================================================
    # 4. packed_decode recurrent: delta rule update on [HV, V, K] state
    # ========================================================================
    q = torch.randn(B, dn_hk, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, dn_hk, K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, dn_hv, V, device=DEVICE, dtype=DTYPE)
    g = torch.randn(B, dn_hv, device=DEVICE, dtype=torch.float32) * 0.1
    beta = torch.randn(B, dn_hv, device=DEVICE, dtype=torch.float32)
    state = torch.randn(B, dn_hv, V, K, device=DEVICE, dtype=DTYPE)

    def recurrent_update():
        # Gate state (convert to fp16 for einsum compatibility)
        g_exp = torch.exp(g.half()).unsqueeze(-1).unsqueeze(-1)
        gated = state * g_exp
        # Expand k for GQA: [B, HK, K] -> [B, HV, K]
        k_expand = k.repeat_interleave(dn_hv // dn_hk, dim=1)
        # Delta rule
        delta_v = v - torch.einsum("bnvk,bnk->bnv", gated, k_expand)
        beta_sig = torch.sigmoid(beta.half()).unsqueeze(-1)
        delta_v = delta_v * beta_sig
        new_state = gated + torch.einsum("bnv,bnk->bnvk", delta_v, k_expand)
        # Output
        q_expand = q.repeat_interleave(dn_hv // dn_hk, dim=1)
        out = torch.einsum("bnvk,bnk->bnv", new_state, q_expand)
        return out
    t_recurrent = bench_op("recurrent_update", recurrent_update, iters=args.iters)

    # ========================================================================
    # 5. RMSNorm + gating: [B, HV*V]
    # ========================================================================
    norm_input = torch.randn(B, dn_hv * V, device=DEVICE, dtype=DTYPE)
    z_gate = torch.randn(B, dn_hv * V, device=DEVICE, dtype=DTYPE)
    norm_weight = torch.randn(dn_hv * V, device=DEVICE, dtype=DTYPE)
    def rmsnorm_gated():
        rms = torch.sqrt(torch.mean(norm_input ** 2, dim=-1, keepdim=True) + 1e-6)
        normed = norm_input / rms * norm_weight
        return normed * z_gate
    t_norm = bench_op("rmsnorm_gated", rmsnorm_gated, iters=args.iters)

    # ========================================================================
    # 6. out_proj: [B, HV*V] -> [B, hidden]
    # ========================================================================
    w_out = torch.randn(out_proj_out, out_proj_in, device=DEVICE, dtype=DTYPE)
    out_proj_input = torch.randn(B, out_proj_in, device=DEVICE, dtype=DTYPE)
    t_outproj = bench_op("out_proj", lambda: torch.mm(out_proj_input, w_out.T), iters=args.iters)

    # ========================================================================
    # 7. All-reduce (TP communication) — only for attention layers, not DeltaNet
    # ========================================================================
    t_allreduce = 0.0
    if TP > 1 and not args.replicate_deltanet:
        # Simulated as a copy (real would use NCCL)
        ar_data = torch.randn(B, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
        t_allreduce = bench_op("allreduce_sim", lambda: ar_data.clone(), iters=args.iters)

    # ========================================================================
    # 8. MLP per layer: gate_up + silu + down
    # ========================================================================
    # For replicated DeltaNet, MLP is also replicated
    mlp_in = HIDDEN_DIM
    mlp_inter = INTER_SIZE if args.replicate_deltanet else INTER_SIZE // TP
    w_gate = torch.randn(mlp_inter, mlp_in, device=DEVICE, dtype=DTYPE)
    w_up = torch.randn(mlp_inter, mlp_in, device=DEVICE, dtype=DTYPE)
    w_down = torch.randn(mlp_in, mlp_inter, device=DEVICE, dtype=DTYPE)
    mlp_input = torch.randn(B, mlp_in, device=DEVICE, dtype=DTYPE)
    def mlp_forward():
        gate = torch.mm(mlp_input, w_gate.T)
        up = torch.mm(mlp_input, w_up.T)
        act = torch.nn.functional.silu(gate) * up
        return torch.mm(act, w_down.T)
    t_mlp = bench_op("mlp_forward", mlp_forward, iters=args.iters)

    # ========================================================================
    # 9. Residual add + possible norm
    # ========================================================================
    residual = torch.randn(B, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    layer_out = torch.randn(B, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    t_residual = bench_op("residual_add", lambda: residual + layer_out, iters=args.iters)

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'Operation':<30s} {'ms/call':>10s} {'per model':>10s} {'% total':>8s}")
    print("-" * 62)

    total_per_layer = t_qkvz + t_ba + t_split + t_conv + t_recurrent + t_norm + t_outproj + t_allreduce + t_mlp + t_residual

    ops = [
        ("in_proj_qkvz GEMM", t_qkvz, DELTANET_LAYERS),
        ("in_proj_ba GEMM", t_ba, DELTANET_LAYERS),
        ("qkvzba_split", t_split, DELTANET_LAYERS),
        ("causal_conv1d", t_conv, DELTANET_LAYERS),
        ("recurrent_update", t_recurrent, DELTANET_LAYERS),
        ("rmsnorm_gated", t_norm, DELTANET_LAYERS),
        ("out_proj GEMM", t_outproj, DELTANET_LAYERS),
        ("allreduce (if TP)", t_allreduce, DELTANET_LAYERS),
        ("MLP forward", t_mlp, DELTANET_LAYERS),
        ("residual_add", t_residual, TOTAL_LAYERS),
    ]

    total_model = sum(ms * layers for _, ms, layers in ops)

    for name, ms, layers in ops:
        model_ms = ms * layers
        pct = model_ms / total_model * 100 if total_model > 0 else 0
        print(f"  {name:<28s} {ms:>9.3f}  {model_ms:>9.3f}  {pct:>6.1f}%")

    print("-" * 62)
    print(f"  {'Total per DeltaNet layer':<28s} {total_per_layer:>9.3f}")
    print(f"  {'Total model (raw ops)':<28s} {'':>9s}  {total_model:>9.3f}")
    print(f"  {'Theoretical tok/s':<28s} {'':>9s}  {1000/total_model:>9.1f}")
    print()

    # Compare to actual SGLang
    sglang_toks = 7.0  # measured
    sglang_ms = 1000.0 / sglang_toks
    print(f"  Actual SGLang:      {sglang_ms:.1f} ms/token ({sglang_toks} tok/s)")
    print(f"  Theoretical:        {total_model:.1f} ms/token ({1000/total_model:.1f} tok/s)")
    print(f"  Overhead ratio:     {sglang_ms/total_model:.1f}x")
    print()

    # Kernel launch overhead estimate
    kernels_per_dn_layer = 7  # projection, split, conv, recurrent, norm, outproj, mlp (~3)
    kernels_per_attn_layer = 5  # projection, attention, norm, outproj, mlp
    total_kernels = DELTANET_LAYERS * kernels_per_dn_layer + ATTN_LAYERS * kernels_per_attn_layer
    overhead_ms = sglang_ms - total_model
    per_kernel_overhead = overhead_ms / total_kernels if total_kernels > 0 else 0
    print(f"  Estimated kernel launches:  {total_kernels}")
    print(f"  Overhead per launch:        {per_kernel_overhead:.3f} ms ({per_kernel_overhead*1000:.0f} us)")
    print(f"  Total overhead:             {overhead_ms:.1f} ms")

    # State memory analysis
    state_per_layer_bytes = dn_hv * V * K * 2  # fp16
    total_state_bytes = state_per_layer_bytes * DELTANET_LAYERS
    print(f"\n  DeltaNet state memory:      {total_state_bytes / 1024 / 1024:.1f} MB ({DELTANET_LAYERS} layers)")
    state_rw_bytes = total_state_bytes * 2  # read + write per token
    bw_gb = 936  # RTX 3090
    state_time_ms = state_rw_bytes / (bw_gb * 1e9) * 1000
    print(f"  State R/W bandwidth time:   {state_time_ms:.3f} ms (at {bw_gb} GB/s)")


if __name__ == "__main__":
    main()
