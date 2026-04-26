#!/usr/bin/env python3
"""Sweep MoE Triton kernel configs for RTX 3090 (sm_86).

Generates optimized fused_moe kernel configurations for each batch size,
tuned for our specific model dimensions. Outputs JSON configs compatible
with SGLang's fused_moe_triton_config.py loader.

Models:
  Qwen3-Coder-30B / Qwen3-VL-30B: E=128, N=768, K=2048
  Gemma 4 26B MoE:                 E=128, N=704, K=2816
  Coder-REAP-25B:                  E=103, N=768, K=2048

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test/sweep_moe_configs.py
    CUDA_VISIBLE_DEVICES=0 python scripts/test/sweep_moe_configs.py --model gemma4
"""
import argparse
import json
import os
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
DTYPE = torch.float16

# Model MoE dimensions
MOE_MODELS = {
    "qwen3-coder": {"E": 128, "N": 768, "K": 2048, "topk": 8, "name": "Qwen3-Coder-30B"},
    "qwen3-vl":    {"E": 128, "N": 768, "K": 2048, "topk": 8, "name": "Qwen3-VL-30B"},
    "gemma4":      {"E": 128, "N": 704, "K": 2816, "topk": 8, "name": "Gemma 4 26B MoE"},
    "coder-reap":  {"E": 103, "N": 768, "K": 2048, "topk": 8, "name": "Coder-REAP-25B"},
}

# Batch sizes to tune for (matching SGLang's config format)
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Configs to sweep
SWEEP_SPACE = []
for bm in [16, 32, 64, 128]:
    for bn in [32, 64, 128, 256]:
        for bk in [32, 64, 128]:
            for gm in [1, 4, 8, 16, 32]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4]:
                        SWEEP_SPACE.append({
                            "BLOCK_SIZE_M": bm,
                            "BLOCK_SIZE_N": bn,
                            "BLOCK_SIZE_K": bk,
                            "GROUP_SIZE_M": gm,
                            "num_warps": nw,
                            "num_stages": ns,
                        })

# Reduced sweep for faster iteration
FAST_SWEEP = [
    # Small M configs (decode, M=1-8) — need small BLOCK_SIZE_M
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2},
    # Medium M configs (small batch prefill, M=8-32)
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2},
    # Large M configs (M=64-256)
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
    # SGLang defaults for comparison
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
]


@triton.jit
def moe_gemm_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N: tl.constexpr, K: tl.constexpr,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Simplified MoE GEMM kernel for benchmarking tile configs."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty))


def bench_moe_config(M, N, K, config, warmup=10, iters=50):
    """Benchmark a single MoE GEMM config."""
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    bk = config["BLOCK_SIZE_K"]
    gm = config["GROUP_SIZE_M"]
    nw = config.get("num_warps", 4)
    ns = config.get("num_stages", 3)

    # Validate divisibility
    if N % bn != 0 or K % bk != 0:
        return None
    # Pad M up to BLOCK_SIZE_M (MoE kernel masks unused rows)
    padded_M = max(M, bm)

    A = torch.randn(padded_M, K, device=DEVICE, dtype=DTYPE)
    B = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    C = torch.empty(padded_M, N, device=DEVICE, dtype=DTYPE)

    grid = lambda meta: (triton.cdiv(padded_M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    try:
        for _ in range(warmup):
            moe_gemm_kernel[grid](
                A, B, C, padded_M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm, num_warps=nw, num_stages=ns,
            )
        torch.cuda.synchronize()
    except Exception:
        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        moe_gemm_kernel[grid](
            A, B, C, padded_M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=gm, num_warps=nw, num_stages=ns,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sweep_for_model(model_key, fast=True):
    """Run the full sweep for a model and return optimal configs per batch size."""
    cfg = MOE_MODELS[model_key]
    E, N, K, topk = cfg["E"], cfg["N"], cfg["K"], cfg["topk"]

    gpu_name = torch.cuda.get_device_name()
    print(f"\n{'='*70}")
    print(f"MoE Kernel Sweep — {cfg['name']}")
    print(f"GPU: {gpu_name}  E={E}  N={N}  K={K}  top_k={topk}")
    print(f"{'='*70}")

    configs = FAST_SWEEP if fast else SWEEP_SPACE
    optimal = {}

    for M in BATCH_SIZES:
        # Effective M after expert routing: M * topk / E (tokens per expert)
        # For decode (M=1): each expert sees ~topk/E * 1 ≈ 0.06 tokens
        # SGLang accumulates across experts, so effective M for kernel = M
        effective_M = max(M, 1)

        best_ms = float("inf")
        best_cfg = None

        for config in configs:
            ms = bench_moe_config(effective_M, N, K, config)
            if ms is not None and ms < best_ms:
                best_ms = ms
                best_cfg = config.copy()

        if best_cfg:
            optimal[M] = best_cfg
            bm = best_cfg["BLOCK_SIZE_M"]
            bn = best_cfg["BLOCK_SIZE_N"]
            bk = best_cfg["BLOCK_SIZE_K"]
            gm = best_cfg["GROUP_SIZE_M"]
            nw = best_cfg.get("num_warps", 4)
            ns = best_cfg.get("num_stages", 3)
            print(f"  M={M:>4d}: BM={bm:>3d} BN={bn:>3d} BK={bk:>3d} GM={gm:>3d} "
                  f"W={nw} S={ns}  → {best_ms:.4f} ms")

    return optimal


def save_config(model_key, optimal_configs, output_dir):
    """Save optimal configs as SGLang-compatible JSON."""
    cfg = MOE_MODELS[model_key]
    E, N = cfg["E"], cfg["N"]

    device_name = torch.cuda.get_device_name().replace(" ", "_")
    triton_ver = triton.__version__.replace(".", "_")

    # Create version-specific directory
    version_dir = os.path.join(output_dir, f"triton_{triton_ver}")
    os.makedirs(version_dir, exist_ok=True)

    filename = f"E={E},N={N},device_name={device_name}.json"
    filepath = os.path.join(version_dir, filename)

    # Convert to SGLang format: keys are batch size strings
    json_configs = {str(k): v for k, v in optimal_configs.items()}

    with open(filepath, "w") as f:
        json.dump(json_configs, f, indent=2)

    print(f"\n  Saved: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(MOE_MODELS.keys()) + ["all"])
    parser.add_argument("--full-sweep", action="store_true", help="Run full sweep (slow)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for configs (default: SGLang configs dir)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../components/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs"
        )

    models = list(MOE_MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models:
        optimal = sweep_for_model(model_key, fast=not args.full_sweep)
        if optimal:
            save_config(model_key, optimal, args.output_dir)

    print(f"\nDone! Configs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
