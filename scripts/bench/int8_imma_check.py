#!/usr/bin/env python
"""Is the INT8 tensor path (IMMA) open on these GA102s after the gpu-unlock? The 2026-10-22 receipt's INT8 figure
(torch._int_mm, 41 -> 43 TOPS stock -> unlocked) sits 5x under the 284 TOPS spec while FP16 doubled, so either
cuBLASLt's _int_mm picks a poor kernel or INT8 is rate-gated by something the unlock does not open. This times three
paths at serving-like shapes and prints TOPS next to the FP16 (fp32-accumulate) rate: torch.mm fp16, torch._int_mm,
and sgl_kernel.int8_scaled_mm (the CUTLASS W8A8 path SGLang serves with). W4A8 prefill only makes sense if INT8
clearly beats unlocked FP16. Receipt: benchmarks/gpu-unlock-2026-10-22/int8-imma-check-<date>.json"""
import json, subprocess, sys, time, torch
from sgl_kernel import int8_scaled_mm

def bench(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / iters / 1000  # s

rows = []
for (M, K, N) in ((8192, 8192, 8192), (8192, 5120, 5120), (8192, 5120, 17408)):
    flop = 2 * M * K * N
    a = torch.randn(M, K, device="cuda", dtype=torch.float16); b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    fp16 = flop / bench(lambda: a @ b) / 1e12
    a8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8); b8 = torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
    try: intmm = flop / bench(lambda: torch._int_mm(a8, b8)) / 1e12
    except Exception as e: intmm = f"err {str(e)[:60]}"
    sa = torch.rand(M, 1, device="cuda", dtype=torch.float32); sb = torch.rand(N, device="cuda", dtype=torch.float32)
    b8_cm = b8.t().contiguous().t()   # the CUTLASS path wants a column-major [K, N] (an [N, K] row-major view transposed)
    try: sglmm = flop / bench(lambda: int8_scaled_mm(a8, b8_cm, sa, sb, torch.float16, None)) / 1e12
    except Exception as e: sglmm = f"err {str(e)[:80]}"
    r = {"M": M, "K": K, "N": N, "fp16_fp32acc_tflops": round(fp16, 1), "torch_int_mm_tops": intmm if isinstance(intmm, str) else round(intmm, 1),
         "sgl_int8_scaled_mm_tops": sglmm if isinstance(sglmm, str) else round(sglmm, 1)}
    rows.append(r); print(r, flush=True)
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,clocks.sm,clocks.max.sm", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
out = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0), "smi": smi, "rows": rows,
       "note": "GPU 0 only; fp16 = torch.mm with fp32 accumulate (reduced-precision reduction off)."}
p = f"benchmarks/gpu-unlock-2026-10-22/int8-imma-check-{time.strftime('%Y-%m-%d')}.json"
json.dump(out, open(p, "w"), indent=1); print("wrote", p)
