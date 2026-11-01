"""Isolated micro-benchmark (receipt: benchmarks/profiling/ba-gemv-microbench-2026-10-31.md): the GDN in_proj_ba projection at decode (M=1), fp16 W[N, K] with N = 2*num_v_heads/tp.
Compares F.linear (cuBLAS, what SGLang runs), torch.mv, an elementwise-reduce, and a Triton gemv (one program per
BLOCK_N rows, fp32 accumulate). CUDA-graph replay timing so launch overhead is excluded like in serving."""
import sys, torch, triton, triton.language as tl

@triton.jit
def _gemv_kernel(x_ptr, w_ptr, y_ptr, N, K, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rk, mask=rk < K, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=rn < N)

def triton_gemv(x, w, BLOCK_N=16, BLOCK_K=1024):
    N, K = w.shape; y = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
    _gemv_kernel[(triton.cdiv(N, BLOCK_N),)](x, w, y, N, K, w.stride(0), BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return y

def time_graph(fn, iters=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): fn()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (iters * 20)  # us per call

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5120
for N in [int(a) for a in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "64", "96"])]:
    w = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.02
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    ref = torch.nn.functional.linear(x, w)
    cands = {
        "F.linear (cuBLAS)": lambda: torch.nn.functional.linear(x, w),
        "torch.mv": lambda: torch.mv(w, x[0]),
        "(w*x).sum": lambda: (w * x).sum(-1),
        "triton gemv BN16 BK1024": lambda: triton_gemv(x, w, 16, 1024),
        "triton gemv BN8 BK2048": lambda: triton_gemv(x, w, 8, 2048),
        "triton gemv BN32 BK512": lambda: triton_gemv(x, w, 32, 512),
    }
    print(f"\nN={N} K={K} fp16 ({N*K*2/1e3:.0f} KB weights)")
    for name, fn in cands.items():
        out = fn().reshape(-1).float(); err = (out - ref.reshape(-1).float()).abs().max().item()
        print(f"  {name:26} {time_graph(fn):7.2f} us   max|err| {err:.2e}")
