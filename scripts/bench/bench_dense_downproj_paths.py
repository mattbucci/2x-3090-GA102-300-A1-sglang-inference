#!/usr/bin/env python3
"""Sprint B2' — size the prize of fixing gemma4-26B's dense down_proj fallback.

The 26B dense mlp.down_proj (per-rank K=1056, N=5376, group_size=32) fails
marlin's K % 128 == 0 thread-tile check (1056 % 128 = 32; full-width 2112 %
128 = 64 — replication would NOT help), so SGLang serves it on the
"unoptimized AWQ kernels" path. This bench measures, at M=1 decode:

  A) the production dequant path:  awq_dequantize (sgl_kernel) + torch.mm
  B) triton fused W4A16 gemm (awq_gemm_triton), if available in-tree
  C) marlin at the K-PADDED shape (1152 = 9*128, zeros padded, g=32) —
     the upper bound a pad-to-128 fix could reach

Prize = (A - best(B,C)) * 30 layers * decode TPOT share. Receipt: JSON to
benchmarks/sprint-2026-06-kv-decode/B2-downproj-microbench.json
"""
import json, os, sys, time
import torch

torch.manual_seed(0)
DEV = "cuda:0"
K_RANK, N, G = 1056, 5376, 32
K_PAD = 1152  # 9 * 128
M = 1
ITERS, WARMUP = 2000, 200
out = {"shape": {"K_rank": K_RANK, "N": N, "group": G, "K_pad": K_PAD, "M": M}}


def mk_awq(k, n, g):
    qweight = torch.randint(0, 2**31 - 1, (k, n // 8), dtype=torch.int32, device=DEV)
    qzeros = torch.randint(0, 2**31 - 1, (k // g, n // 8), dtype=torch.int32, device=DEV)
    scales = (torch.rand(k // g, n, dtype=torch.float16, device=DEV) + 0.5) * 0.002
    return qweight, qzeros, scales


def bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us per call


x = torch.randn(M, K_RANK, dtype=torch.float16, device=DEV)
qw, qz, sc = mk_awq(K_RANK, N, G)

# --- A) production dequant + mm (the fallback's M>=threshold path) ---
from sgl_kernel import awq_dequantize

def path_dequant():
    w = awq_dequantize(qw, sc, qz)          # [K, N] fp16
    return torch.mm(x, w)

out["A_dequant_mm_us"] = bench(path_dequant)

# is there a GEMV-style fused path in sgl_kernel? (the actual kernel.apply
# may route M<=N_THRESH here instead of dequant+mm)
try:
    from sgl_kernel import awq_gemm
    def path_awq_gemm():
        return awq_gemm(x, qw, qz, sc, 8)
    out["A2_sgl_awq_gemm_us"] = bench(path_awq_gemm)
except Exception as ex:
    out["A2_sgl_awq_gemm_us"] = f"unavailable: {type(ex).__name__} {ex}"

# --- B) in-tree triton fused W4A16 ---
try:
    from sglang.srt.layers.quantization.awq.awq_triton import awq_gemm_triton
    def path_triton():
        return awq_gemm_triton(x, qw, sc, qz, split_k_iters=8)
    out["B_triton_gemm_us"] = bench(path_triton)
except Exception as ex:
    out["B_triton_gemm_us"] = f"unavailable: {type(ex).__name__} {ex}"

# --- C) marlin at padded K ---
try:
    from sglang.jit_kernel.awq_marlin_repack import awq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_make_workspace,
        marlin_permute_scales,
    )
    from sgl_kernel import awq_marlin_gemm
    from sglang.srt.layers.quantization.utils import scalar_types

    qw_p, qz_p, sc_p = mk_awq(K_PAD, N, G)
    mqw = awq_marlin_repack(qw_p, K_PAD, N, 4)
    msc = marlin_permute_scales(sc_p, K_PAD, N, G)
    mzp = awq_to_marlin_zero_points(qz_p, K_PAD // G, N, 4)
    ws = marlin_make_workspace(DEV)
    xp = torch.zeros(M, K_PAD, dtype=torch.float16, device=DEV)
    xp[:, :K_RANK] = x
    g_idx = torch.empty(0, dtype=torch.int32, device=DEV)
    g_sort = torch.empty(0, dtype=torch.int32, device=DEV)

    def path_marlin():
        return awq_marlin_gemm(
            xp, None, mqw, None, msc, None, mzp, g_idx, g_sort, ws,
            scalar_types.uint4, M, N, K_PAD, True, False, True, False,
        )

    # smoke once (API drift would throw here, not in the timed loop)
    path_marlin()
    out["C_marlin_padded_us"] = bench(path_marlin)
except Exception as ex:
    out["C_marlin_padded_us"] = f"unavailable: {type(ex).__name__}: {ex}"

# --- reference: pure fp16 mm at same shape (floor) ---
w_ref = torch.randn(K_RANK, N, dtype=torch.float16, device=DEV)
out["ref_fp16_mm_us"] = bench(lambda: torch.mm(x, w_ref))

# prize math
a = out["A_dequant_mm_us"]
best = min(v for v in [out.get("B_triton_gemm_us"), out.get("C_marlin_padded_us"),
                       out.get("A2_sgl_awq_gemm_us")] if isinstance(v, float))
out["per_layer_saving_us"] = a - best
out["per_token_saving_ms_30layers"] = (a - best) * 30 / 1000.0
out["note"] = "TPOT @256K is ~31-32 ms (A2 receipts); share = saving/TPOT"
print(json.dumps(out, indent=1))
p = os.path.join(os.path.dirname(__file__), "../../benchmarks/sprint-2026-06-kv-decode/B2-downproj-microbench.json")
json.dump(out, open(os.path.abspath(p), "w"), indent=1)
