#!/usr/bin/env python3
"""Per-GPU tensor-core GEMM throughput by dtype — the receipt for the GA102
tensor-rate unlock (Zagitta/gpu-unlock) and any clock/power-cap change.

Measures an N^2 GEMM (default 8192, 1.1 TFLOP per call) as fp16 and bf16 with
fp32 accumulate (cuBLAS default — the half-rate path on GeForce GA102), fp16
with reduced-precision reduction, tf32, fp32, int8 (torch._int_mm), plus a
device-memory read (fp16 sum, GB/s) as the bandwidth control that an unlock
must NOT move. Each figure is the median of 5 x ITERS calls after warm-up.
Prints one JSON line per GPU with the power limit and the SM clock seen
during the run — compare rows only at the same power cap.

Usage (serving env, GPUs idle):
    source scripts/common.sh; activate_conda; setup_nvidia_env
    python scripts/bench/tensor_rate.py [N]        # receipts: benchmarks/gpu-unlock-2026-10-22/*.json
"""
import json
import subprocess
import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
ITERS = 20


def bench(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / ITERS)
    ts.sort()
    return ts[2]


def smi(idx, q):
    return subprocess.run(["nvidia-smi", "-i", str(idx), f"--query-gpu={q}",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


for dev in range(torch.cuda.device_count()):
    torch.cuda.set_device(dev)
    out = {"gpu": dev, "name": torch.cuda.get_device_name(dev), "n": N,
           "power_limit_w": smi(dev, "power.limit"),
           "clocks_max_sm_mhz": smi(dev, "clocks.max.sm")}
    flop = 2.0 * N * N * N
    a16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out["fp16_fp32acc_tflops"] = round(flop / bench(lambda: a16 @ b16) / 1e12, 1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    out["fp16_reducedacc_tflops"] = round(flop / bench(lambda: a16 @ b16) / 1e12, 1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    abf = a16.to(torch.bfloat16)
    bbf = b16.to(torch.bfloat16)
    out["bf16_tflops"] = round(flop / bench(lambda: abf @ bbf) / 1e12, 1)
    out["sm_clock_during_mhz"] = smi(dev, "clocks.sm")  # sampled right after the tensor GEMMs
    a32 = a16.float()
    b32 = b16.float()
    torch.backends.cuda.matmul.allow_tf32 = True
    out["tf32_tflops"] = round(flop / bench(lambda: a32 @ b32) / 1e12, 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    out["fp32_tflops"] = round(flop / bench(lambda: a32 @ b32) / 1e12, 1)
    a8 = torch.randint(-127, 127, (N, N), device="cuda", dtype=torch.int8)
    b8 = torch.randint(-127, 127, (N, N), device="cuda", dtype=torch.int8)
    out["int8_tops"] = round(flop / bench(lambda: torch._int_mm(a8, b8)) / 1e12, 1)
    # Bandwidth control: a 2 GB fp16 reduction is DRAM-bound (an integer sum
    # promotes to int64 and turns compute-bound — do not use uint8 here).
    big = torch.empty(1024 ** 3, device="cuda", dtype=torch.float16)
    out["read_bw_gbs"] = round(big.numel() * 2 / bench(lambda: big.sum()) / 1e9, 0)
    del a16, b16, abf, bbf, a32, b32, a8, b8, big
    torch.cuda.empty_cache()
    print(json.dumps(out), flush=True)
