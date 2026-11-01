#!/usr/bin/env python
"""W4A8 prefill path (3090 patch 071) end to end on synthetic AWQ layers at the qwen38 shapes: (1) the Triton
Marlin-layout -> INT8 dequantiser must reproduce round((q - z) * s / sc) exactly (ties aside), (2) the INT8 GEMM output
must track the fp32 reference within the per-token INT8 activation error, (3) timing vs Marlin W4A16 at M = 8192.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_awq_w4a8_prefill.py --out benchmarks/profiling/w4a8-prefill-unit-2026-11-01.json
"""
import argparse, json, os, statistics, sys, time
os.environ.setdefault("SGLANG_AWQ_W4A8_PREFILL_MIN_TOKENS", "1")
import torch
from sglang.srt.layers.quantization.marlin_utils import (apply_awq_marlin_linear, awq_to_marlin_zero_points, marlin_make_empty_g_idx,
                                                          marlin_make_workspace, marlin_permute_scales, scalar_types)
from sglang.srt.layers.quantization.awq_w4a8_prefill import _awq_marlin_repack, _awq_pack, apply_w4a8, dequant_to_int8, prepare_layer


def bench(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shapes", default="5120x5120,5120x17408,17408x5120,5120x1024")
    ap.add_argument("--m", type=int, default=8192); ap.add_argument("--group", type=int, default=128); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device("cuda"); g = a.group; results = []; fails = 0
    e = marlin_make_empty_g_idx(dev); ws = marlin_make_workspace(dev)
    for shp in a.shapes.split(","):
        K, N = (int(v) for v in shp.split("x")); torch.manual_seed(K * 31 + N)
        q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
        z = torch.randint(0, 16, (K // g, N), device=dev, dtype=torch.int32)
        s = (torch.rand(K // g, N, device=dev) * 0.02 + 0.005).half()
        mq = _awq_marlin_repack(_awq_pack(q).contiguous(), K, N, 4)
        ms = marlin_permute_scales(s, K, N, g); mz = awq_to_marlin_zero_points(_awq_pack(z), K // g, N, 4)
        layer = torch.nn.Module(); layer.qweight = torch.nn.Parameter(mq, requires_grad=False)
        prepare_layer(layer, s, z, g)
        # (1) dequantiser exactness
        w_ref = (q.float() - z.repeat_interleave(g, dim=0).float()) * s.repeat_interleave(g, dim=0).float()  # [K, N]
        w8_ref = torch.round(w_ref / layer.w4a8_chan_scale[None, :]).clamp_(-127, 127)
        w8 = dequant_to_int8(layer, K, N).t().float()  # [K, N]
        d = (w8 - w8_ref).abs(); exact = (d == 0).float().mean().item(); dmax = d.max().item()
        # (2) GEMM numerics at M rows
        x = (torch.randn(a.m, K, device=dev) * 0.5).half()
        y_ref = x.float() @ w_ref
        y_marlin = apply_awq_marlin_linear(input=x, weight=mq, weight_scale=ms, weight_zp=mz, g_idx=e, g_idx_sort_indices=e, workspace=ws,
                                           quant_type=scalar_types.uint4, output_size_per_partition=N, input_size_per_partition=K)
        y_w4a8 = apply_w4a8(layer, x, None, K, N)
        def rel(y): return ((y.float() - y_ref).norm() / y_ref.norm()).item()
        def cos(y): return torch.nn.functional.cosine_similarity(y.float().flatten(), y_ref.flatten(), dim=0).item()
        r_m, r_8 = rel(y_marlin), rel(y_w4a8); c_8 = cos(y_w4a8)
        # (3) timing
        t_marlin = bench(lambda: apply_awq_marlin_linear(input=x, weight=mq, weight_scale=ms, weight_zp=mz, g_idx=e, g_idx_sort_indices=e, workspace=ws,
                                                         quant_type=scalar_types.uint4, output_size_per_partition=N, input_size_per_partition=K))
        t_w4a8 = bench(lambda: apply_w4a8(layer, x, None, K, N)); t_deq = bench(lambda: dequant_to_int8(layer, K, N))
        ok = dmax <= 1 and exact >= 0.999 and torch.isfinite(y_w4a8).all().item() and r_8 <= 0.015 and c_8 >= 0.9999
        fails += 0 if ok else 1
        r = {"K": K, "N": N, "M": a.m, "group": g, "status": "ok" if ok else "FAIL", "int8_exact_frac": round(exact, 6), "int8_max_diff": dmax,
             "rel_err_marlin": r_m, "rel_err_w4a8": r_8, "cos_w4a8": c_8, "marlin_us": round(t_marlin), "w4a8_us": round(t_w4a8), "dequant_us": round(t_deq),
             "speedup": round(t_marlin / t_w4a8, 2)}
        results.append(r)
        print(f"K={K:>5} N={N:>5} M={a.m}  int8 exact {exact*100:6.2f}% maxdiff {dmax:.0f} | rel err marlin {r_m:.2e} w4a8 {r_8:.2e} cos {c_8:.6f} | "
              f"marlin {t_marlin:7.0f} us  w4a8 {t_w4a8:7.0f} us (dequant {t_deq:5.0f})  x{t_marlin/t_w4a8:.2f}  {'ok' if ok else 'FAIL'}", flush=True)
    if a.out:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0), "results": results}, open(a.out, "w"), indent=1); print("wrote", a.out)
    print("RESULT:", "FAIL" if fails else "PASS"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
