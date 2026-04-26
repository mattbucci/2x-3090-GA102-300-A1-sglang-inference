#!/usr/bin/env python3
"""Sweep DeltaNet Triton kernel parameters for sm_86 (RTX 3090).

Tests different BV (block size along V dimension), num_warps, and num_stages
combinations for the fused_recurrent DeltaNet decode kernel to find optimal
parameters for our hardware.

Current upstream defaults:
  BV = min(next_power_of_2(V), 32) = 32   (capped!)
  num_warps = 1
  num_stages = 3

RTX 3090 (sm_86) specs:
  82 SMs, 48 warps/SM max, 65536 registers/SM, 100KB shared memory/SM
  936 GB/s memory bandwidth

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test/bench_deltanet_triton_sweep.py
    CUDA_VISIBLE_DEVICES=0 python scripts/test/bench_deltanet_triton_sweep.py --model qwen35-moe
"""
import argparse
import json
import sys
import time
from typing import Optional

import torch
import triton
import triton.language as tl

# ============================================================================
# Triton kernel: parameterized DeltaNet decode (single-step recurrence)
# Mirrors fused_recurrent_gated_delta_rule_packed_decode_kernel from FLA
# ============================================================================

@triton.jit
def deltanet_decode_bench_kernel(
    q_ptr, k_ptr, v_ptr,
    g_ptr,           # gating values (scalar per head)
    beta_ptr,        # beta gating
    state_ptr,       # [num_slots, HV, V, K] recurrent state
    out_ptr,         # [B, HV, V] output
    state_indices,   # [B] slot indices
    scale,
    B: tl.constexpr,
    HV: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    """Single-step DeltaNet decode: load state, gate, delta update, output, store state."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Load state index
    state_idx = tl.load(state_indices + i_n).to(tl.int64)

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(out_ptr.dtype.element_ty)
        p_o = out_ptr + (i_n * HV + i_hv) * V + o_v
        tl.store(p_o, zero, mask=mask_v)
        return

    # Load recurrent state tile [BV, BK]
    p_h = state_ptr + state_idx * HV * V * K + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    # Load q, k, v vectors
    p_q = q_ptr + i_n * H * K + i_h * K + o_k
    p_k = k_ptr + i_n * H * K + i_h * K + o_k
    p_v = v_ptr + i_n * HV * V + i_hv * V + o_v
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

    # Optional L2 norm
    if USE_L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # Load gating
    b_g = tl.load(g_ptr + i_n * HV + i_hv).to(tl.float32)
    b_beta = tl.load(beta_ptr + i_n * HV + i_hv).to(tl.float32)
    b_beta = 1.0 / (1.0 + tl.exp(-b_beta))

    # Gate state: h *= exp(g)
    b_h *= tl.exp(b_g)

    # Delta rule: v -= sum(h * k, dim=1)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta

    # State update: h += v[:, None] * k[None, :]
    b_h += b_v[:, None] * b_k[None, :]

    # Output: o = sum(h * q, dim=1)
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # Store output
    p_o = out_ptr + (i_n * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # Store updated state
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


# ============================================================================
# Benchmark harness
# ============================================================================

MODEL_CONFIGS = {
    "qwen35": {
        "name": "Qwen3.5-27B DeltaNet",
        "H": 16,           # num_key_heads
        "HV": 48,          # num_value_heads
        "K": 128,           # head_dim_k
        "V": 128,           # head_dim_v
        "deltanet_layers": 36,
        "total_layers": 48,
    },
    "qwen35-moe": {
        "name": "Qwen3.5-28B MoE DeltaNet",
        "H": 16,
        "HV": 48,
        "K": 128,
        "V": 128,
        "deltanet_layers": 36,
        "total_layers": 48,
    },
}

# Configs to sweep: (BV, num_warps, num_stages)
SWEEP_CONFIGS = [
    # Baseline (upstream default)
    (32, 1, 3, "baseline"),
    # Vary num_warps with BV=32
    (32, 2, 3, "BV32-w2"),
    (32, 4, 3, "BV32-w4"),
    (32, 1, 2, "BV32-s2"),
    (32, 1, 1, "BV32-s1"),
    (32, 2, 2, "BV32-w2-s2"),
    (32, 2, 1, "BV32-w2-s1"),
    (32, 4, 2, "BV32-w4-s2"),
    # BV=64 (doubles work per block, halves grid)
    (64, 1, 3, "BV64-w1"),
    (64, 2, 3, "BV64-w2"),
    (64, 2, 2, "BV64-w2-s2"),
    (64, 4, 3, "BV64-w4"),
    (64, 4, 2, "BV64-w4-s2"),
    (64, 4, 1, "BV64-w4-s1"),
    # BV=128 (one V-tile per block — fewest grid blocks)
    (128, 4, 3, "BV128-w4"),
    (128, 4, 2, "BV128-w4-s2"),
    (128, 4, 1, "BV128-w4-s1"),
    (128, 8, 2, "BV128-w8-s2"),
    (128, 8, 1, "BV128-w8-s1"),
]


def bench_config(
    B: int,
    HV: int,
    H: int,
    K: int,
    V: int,
    BV: int,
    num_warps: int,
    num_stages: int,
    num_layers: int,
    warmup: int = 20,
    iters: int = 100,
    device: str = "cuda",
) -> Optional[float]:
    """Benchmark one DeltaNet decode kernel config. Returns ms per layer or None on failure."""
    BK = triton.next_power_of_2(K)
    NV = triton.cdiv(V, BV)
    scale = K ** -0.5

    # Allocate tensors
    num_slots = B + 4  # small buffer
    q = torch.randn(B, H * K, device=device, dtype=torch.float16)
    k = torch.randn(B, H * K, device=device, dtype=torch.float16)
    v = torch.randn(B, HV * V, device=device, dtype=torch.float16)
    g = torch.randn(B, HV, device=device, dtype=torch.float32) * 0.1
    beta = torch.randn(B, HV, device=device, dtype=torch.float32)
    state = torch.randn(num_slots, HV, V, K, device=device, dtype=torch.float16)
    out = torch.empty(B, HV * V, device=device, dtype=torch.float16)
    indices = torch.arange(B, device=device, dtype=torch.int32)

    grid = (NV, B * HV)

    # Warmup (also triggers JIT)
    try:
        for _ in range(warmup):
            deltanet_decode_bench_kernel[grid](
                q, k, v, g, beta, state, out, indices, scale,
                B=B, HV=HV, H=H, K=K, V=V, BK=BK, BV=BV,
                USE_L2NORM=True,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        deltanet_decode_bench_kernel[grid](
            q, k, v, g, beta, state, out, indices, scale,
            B=B, HV=HV, H=H, K=K, V=V, BK=BK, BV=BV,
            USE_L2NORM=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    end.record()
    torch.cuda.synchronize()

    ms_per_call = start.elapsed_time(end) / iters
    return ms_per_call, None


def bench_correctness(
    B: int, HV: int, H: int, K: int, V: int,
    BV: int, num_warps: int, num_stages: int,
    device: str = "cuda",
) -> bool:
    """Verify that the kernel produces correct output vs baseline (BV=32, w=1, s=3)."""
    BK = triton.next_power_of_2(K)
    scale = K ** -0.5

    num_slots = B + 4
    q = torch.randn(B, H * K, device=device, dtype=torch.float16)
    k = torch.randn(B, H * K, device=device, dtype=torch.float16)
    v = torch.randn(B, HV * V, device=device, dtype=torch.float16)
    g = torch.randn(B, HV, device=device, dtype=torch.float32) * 0.1
    beta = torch.randn(B, HV, device=device, dtype=torch.float32)
    indices = torch.arange(B, device=device, dtype=torch.int32)

    # Run baseline
    state_ref = torch.randn(num_slots, HV, V, K, device=device, dtype=torch.float16)
    out_ref = torch.empty(B, HV * V, device=device, dtype=torch.float16)
    state_test = state_ref.clone()
    out_test = torch.empty_like(out_ref)

    NV_ref = triton.cdiv(V, 32)
    deltanet_decode_bench_kernel[(NV_ref, B * HV)](
        q, k, v, g, beta, state_ref, out_ref, indices, scale,
        B=B, HV=HV, H=H, K=K, V=V, BK=BK, BV=32,
        USE_L2NORM=True, num_warps=1, num_stages=3,
    )
    torch.cuda.synchronize()

    # Run test config
    NV_test = triton.cdiv(V, BV)
    deltanet_decode_bench_kernel[(NV_test, B * HV)](
        q, k, v, g, beta, state_test, out_test, indices, scale,
        B=B, HV=HV, H=H, K=K, V=V, BK=BK, BV=BV,
        USE_L2NORM=True, num_warps=num_warps, num_stages=num_stages,
    )
    torch.cuda.synchronize()

    # Compare
    out_close = torch.allclose(out_ref, out_test, atol=1e-2, rtol=1e-2)
    state_close = torch.allclose(state_ref.float(), state_test.float(), atol=1e-2, rtol=1e-2)
    return out_close and state_close


def main():
    parser = argparse.ArgumentParser(description="DeltaNet Triton kernel sweep for sm_86")
    parser.add_argument("--model", default="qwen35", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch", type=int, default=1, help="Batch size (decode tokens)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism (halves HV)")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--json", type=str, help="Save results as JSON")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    HV = cfg["HV"] // args.tp  # TP sharding
    H = cfg["H"] // args.tp
    K = cfg["K"]
    V = cfg["V"]
    B = args.batch
    deltanet_layers = cfg["deltanet_layers"]

    device = "cuda"
    gpu_name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()

    print(f"{'='*70}")
    print(f"DeltaNet Triton Kernel Parameter Sweep — {cfg['name']}")
    print(f"GPU: {gpu_name} (sm_{cc[0]}{cc[1]})")
    print(f"Batch={B}  HV={HV} (TP={args.tp})  H={H}  K={K}  V={V}")
    print(f"DeltaNet layers: {deltanet_layers}  |  Iters: {args.iters}")
    print(f"{'='*70}")
    print()

    # Header
    print(f"{'Config':<20s} {'BV':>4s} {'Wrps':>4s} {'Stg':>4s} {'Grid':>8s} "
          f"{'ms/lyr':>8s} {'ms/model':>9s} {'tok/s':>8s} {'Speedup':>8s} {'Correct':>8s}")
    print("-" * 100)

    baseline_ms = None
    results = []

    for BV, num_warps, num_stages, label in SWEEP_CONFIGS:
        NV = triton.cdiv(V, BV)
        grid_size = NV * B * HV

        # Correctness check
        correct = "skip"
        if not args.skip_correctness:
            try:
                ok = bench_correctness(B, HV, H, K, V, BV, num_warps, num_stages, device)
                correct = "PASS" if ok else "FAIL"
            except Exception as e:
                correct = "ERR"

        # Benchmark
        ms, err = bench_config(B, HV, H, K, V, BV, num_warps, num_stages,
                               deltanet_layers, iters=args.iters, device=device)

        if err:
            print(f"{label:<20s} {BV:>4d} {num_warps:>4d} {num_stages:>4d} {grid_size:>8d} "
                  f"{'FAILED':>8s}  -- {err[:40]}")
            continue

        model_ms = ms * deltanet_layers
        tok_s = 1000.0 / model_ms if model_ms > 0 else 0

        if baseline_ms is None:
            baseline_ms = ms

        speedup = baseline_ms / ms if ms > 0 else 0

        marker = " <-- BEST" if speedup > 1.0 and label != "baseline" else ""
        if label == "baseline":
            marker = " [default]"

        print(f"{label:<20s} {BV:>4d} {num_warps:>4d} {num_stages:>4d} {grid_size:>8d} "
              f"{ms:>8.3f} {model_ms:>9.3f} {tok_s:>8.1f} {speedup:>7.2f}x {correct:>8s}{marker}")

        results.append({
            "label": label, "BV": BV, "num_warps": num_warps, "num_stages": num_stages,
            "grid_size": grid_size, "ms_per_layer": round(ms, 4),
            "ms_per_model": round(model_ms, 4), "tok_s": round(tok_s, 1),
            "speedup": round(speedup, 3), "correct": correct,
        })

    print()

    # Summary
    if results:
        valid = [r for r in results if r["correct"] in ("PASS", "skip")]
        if valid:
            best = min(valid, key=lambda r: r["ms_per_layer"])
            baseline = next((r for r in results if r["label"] == "baseline"), None)
            print(f"Best config: {best['label']}  BV={best['BV']} warps={best['num_warps']} stages={best['num_stages']}")
            print(f"  {best['ms_per_layer']:.3f} ms/layer  →  {best['tok_s']:.1f} tok/s  ({best['speedup']:.2f}x vs baseline)")
            if baseline:
                print(f"Baseline:   {baseline['ms_per_layer']:.3f} ms/layer  →  {baseline['tok_s']:.1f} tok/s")

            # Register/occupancy analysis
            print()
            print("--- Occupancy Analysis (sm_86, 82 SMs) ---")
            for r in valid[:5]:
                regs_per_thread = (r["BV"] * 128 * 4) // (r["num_warps"] * 32)  # fp32 bytes
                regs_per_thread_count = (r["BV"] * 128) // (r["num_warps"] * 32)  # register count
                blocks_per_sm = r["grid_size"] / 82
                warps_per_sm = blocks_per_sm * r["num_warps"]
                occupancy_pct = min(warps_per_sm / 48 * 100, 100)
                print(f"  {r['label']:<18s}: grid={r['grid_size']:>4d}  blk/SM={blocks_per_sm:.1f}  "
                      f"wrp/SM={warps_per_sm:.1f}  occ={occupancy_pct:.0f}%  "
                      f"~regs/thrd={regs_per_thread_count}")

    if args.json and results:
        with open(args.json, "w") as f:
            json.dump({
                "gpu": gpu_name,
                "compute_capability": f"sm_{cc[0]}{cc[1]}",
                "model": cfg["name"],
                "batch": B,
                "tp": args.tp,
                "HV": HV, "H": H, "K": K, "V": V,
                "deltanet_layers": deltanet_layers,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
