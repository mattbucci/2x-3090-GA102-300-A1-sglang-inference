#!/usr/bin/env python
"""Prefill GEMM paths at M = 8192 (one chunk) on the dense qwen38 / qwen36-dense shapes: what the W4A8 prefill idea
is worth before any serving code moves. Times (CUDA events, 20 iters, GPU 0):
  * Marlin W4A16 (uint4b8, group 128 — the shipped AWQ path at prefill), via the same pack/repack the INT8-head patch uses;
  * CUTLASS INT8 W8A8 (`sgl_kernel.int8_scaled_mm`) including the per-token activation quant (`per_token_quant_int8`);
  * the transient INT4 -> INT8 weight materialisation modelled as its bandwidth cost (read packed INT4, write INT8 [N, K]);
  * FP16 torch.mm with fp32 accumulate as the reference.
Receipt: benchmarks/profiling/prefill-gemm-paths-<date>.json"""
import json, time, torch
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace, marlin_permute_scales)
from sglang.srt.layers.quantization.lm_head_int8_marlin import _pack_rows_int32_cuda
from sglang.srt.layers.quantization.utils import scalar_types  # noqa

def bench(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / iters / 1000

def marlin_w4(w, group=128):
    n, k = w.shape; dev = w.device
    g = w.float().reshape(n, k // group, group); sg = g.abs().amax(dim=2, keepdim=True).clamp_(min=1e-8) / 7.0
    q = torch.round(g / sg).clamp_(-8, 7).reshape(n, k); q_u = (q + 8).to(torch.uint8).t().contiguous()
    packed = _pack_rows_int32_cuda(q_u, 4)
    mq = gptq_marlin_repack(packed, perm=torch.empty(0, dtype=torch.int, device=dev), size_k=k, size_n=n, num_bits=4)
    ms = marlin_permute_scales(sg.reshape(n, k // group).t().to(w.dtype).contiguous(), size_k=k, size_n=n, group_size=group)
    e = marlin_make_empty_g_idx(dev); ws = marlin_make_workspace(dev)
    return dict(weight=mq, weight_scale=ms, weight_zp=e, g_idx=e, g_idx_sort_indices=e, workspace=ws, wtype=scalar_types.uint4b8,
                output_size_per_partition=n, input_size_per_partition=k, is_k_full=True)

rows = []
for (M, K, N) in ((8192, 5120, 17408), (8192, 17408, 5120), (8192, 5120, 5120), (8192, 5120, 8192)):
    flop = 2 * M * K * N
    x = torch.randn(M, K, device="cuda", dtype=torch.float16); w = (torch.randn(N, K, device="cuda") * 0.02).half()
    fp16 = bench(lambda: x @ w.t())
    mk = marlin_w4(w)
    marlin = bench(lambda: apply_gptq_marlin_linear(input=x, bias=None, **mk))
    w8 = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8); w8_cm = w8.t(); ws = torch.rand(N, device="cuda")  # [K, N] column-major view of the row-major [N, K] weight
    xq, xs = per_token_quant_int8(x)
    int8_only = bench(lambda: int8_scaled_mm(xq, w8_cm, xs, ws, torch.float16, None))
    def _int8_full():
        xq_, xs_ = per_token_quant_int8(x)
        return int8_scaled_mm(xq_, w8_cm, xs_, ws, torch.float16, None)
    int8_full = bench(_int8_full)
    # transient INT4 -> INT8 weight: modelled as a bandwidth-bound pass (read N*K/2 bytes, write N*K bytes)
    src = torch.empty(N * K // 2, device="cuda", dtype=torch.uint8); dst = torch.empty(N, K, device="cuda", dtype=torch.int8)
    def _deq():
        dst.view(-1)[: N * K // 2].copy_(src); dst.view(-1)[N * K // 2:].copy_(src)
    deq = bench(_deq)
    r = {"M": M, "K": K, "N": N, "fp16_ms": round(fp16 * 1e3, 2), "fp16_tflops": round(flop / fp16 / 1e12, 1),
         "marlin_w4a16_ms": round(marlin * 1e3, 2), "marlin_tflops": round(flop / marlin / 1e12, 1),
         "int8_gemm_ms": round(int8_only * 1e3, 2), "int8_tops": round(flop / int8_only / 1e12, 1),
         "int8_gemm_plus_actquant_ms": round(int8_full * 1e3, 2), "int4_to_int8_weight_pass_ms": round(deq * 1e3, 2),
         "w4a8_estimate_ms": round((int8_full + deq) * 1e3, 2), "speedup_vs_marlin": round(marlin / (int8_full + deq), 2)}
    rows.append(r); print(r, flush=True)
p = f"benchmarks/profiling/prefill-gemm-paths-{time.strftime('%Y-%m-%d')}.json"
json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0), "rows": rows}, open(p, "w"), indent=1); print("wrote", p)
