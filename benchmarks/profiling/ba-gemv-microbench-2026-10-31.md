# GDN in_proj_ba gemv — isolated micro-benchmark, 2026-10-31

The decode profiles charge the 48 cuBLAS `gemvx` calls of the GDN in_proj_ba projection ~33 µs each (1.6 ms of kernel time per step on qwen38). In isolation the same shape is a 4.5 µs pair (`gemvx` 3.0 + `splitKreduce` 1.9 µs — the identical kernels, verified under torch.profiler); the traced 33 µs is the kernel stretched by running on the alt stream under the Marlin qkvz GEMM, i.e. memory-bandwidth sharing, not a bad cuBLAS heuristic. A Triton one-program-per-16-rows GEMV is 2× slower in isolation. Verdict: no kernel to write; the lever estimate in the 2026-09-10 profile (−2 %) was already the exposed share and is the ceiling. CUDA-graph replay timing, 2× RTX 3090 at 300 W, driver 595, one card idle.

```

N=32 K=5120 fp16 (328 KB weights)
  F.linear (cuBLAS)             4.22 us   max|err| 0.00e+00
  torch.mv                      4.22 us   max|err| 0.00e+00
  (w*x).sum                     8.32 us   max|err| 7.32e-04
  triton gemv BN16 BK1024      14.27 us   max|err| 0.00e+00
  triton gemv BN8 BK2048        9.75 us   max|err| 0.00e+00
  triton gemv BN32 BK512       17.08 us   max|err| 0.00e+00

N=48 K=5120 fp16 (492 KB weights)
  F.linear (cuBLAS)             4.47 us   max|err| 0.00e+00
  torch.mv                      4.12 us   max|err| 0.00e+00
  (w*x).sum                     7.46 us   max|err| 9.77e-04
  triton gemv BN16 BK1024      12.56 us   max|err| 1.95e-03
  triton gemv BN8 BK2048        8.59 us   max|err| 0.00e+00
  triton gemv BN32 BK512       14.89 us   max|err| 0.00e+00

N=64 K=5120 fp16 (655 KB weights)
  F.linear (cuBLAS)             4.02 us   max|err| 0.00e+00
  torch.mv                      4.01 us   max|err| 0.00e+00
  (w*x).sum                     7.43 us   max|err| 1.95e-03
  triton gemv BN16 BK1024      12.34 us   max|err| 0.00e+00
  triton gemv BN8 BK2048        8.46 us   max|err| 0.00e+00
  triton gemv BN32 BK512       14.76 us   max|err| 0.00e+00

N=32 K=2048 fp16 (131 KB weights)
  F.linear (cuBLAS)             2.94 us   max|err| 0.00e+00
  torch.mv                      2.94 us   max|err| 0.00e+00
  (w*x).sum                     5.05 us   max|err| 4.88e-04
  triton gemv BN16 BK1024       6.41 us   max|err| 0.00e+00
  triton gemv BN8 BK2048        4.33 us   max|err| 0.00e+00
  triton gemv BN32 BK512        7.51 us   max|err| 0.00e+00

N=48 K=2048 fp16 (197 KB weights)
  F.linear (cuBLAS)             3.50 us   max|err| 0.00e+00
  torch.mv                      3.51 us   max|err| 0.00e+00
  (w*x).sum                     5.97 us   max|err| 9.77e-04
  triton gemv BN16 BK1024       6.41 us   max|err| 0.00e+00
  triton gemv BN8 BK2048        4.37 us   max|err| 0.00e+00
  triton gemv BN32 BK512        7.55 us   max|err| 0.00e+00

isolated torch.profiler, F.linear x[1,5120] fp16 · W[48,5120]:
   2.99 us x 50  internal::gemvx::kernel<int, int, __half, float, float, float, ...>
   1.89 us x 50  cublasLt::splitKreduce_kernel<32, 16, int, float, __half, ...>
```

Generator: `scripts/bench/bench_ba_gemv.py <K> <N,N,...>`.
