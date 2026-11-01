# qwen38 fresh-prefill profile — where a cold 32K / 131K prompt's time goes (2026-11-01)

`scripts/bench/profile_prefill.sh` (torch profiler over the whole fresh prefill, cache flushed, natural text sized against the served tokenizer; attributed per 8192-token EXTEND step by `trace_step_anatomy.py`, exposed-time sweep). Shipped `qwen38` (TP=2, 300 W, custom allreduce on for decode, INT8 head). Rank 0.

| prompt | steps | GPU span / step | Marlin INT4 GEMM | attention (flashinfer prefill) | NCCL allreduce | GDN chunk kernels | norm + act + misc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 35,087 tokens (13.75 s wall) | 5 × 8192 | 2.74 s | **1.83 s (67 %)** | 0.45 s (16 %) | 0.27 s (10 %) | 0.09 s (3 %) | 0.10 s |
| 142,336 tokens (78.4 s wall) | 18 × 8192 | 4.35 s | **2.02 s (46 %)** | 1.75 s (40 %) | 0.36 s (8 %) | 0.10 s (2 %) | 0.12 s |

**Reading.** The Marlin W4A16 GEMM chain is a flat ~1.8–2.0 s per 8192-token chunk (1,792 Marlin launches per step at `256, 4, 16, 4` tiles — ~60 TFLOPS effective out of the unlocked FP16 rate at 300 W); attention grows with the prefix (0.45 → 1.75 s per chunk) and overtakes the GEMM around 180K. NCCL's share is the per-layer fp16 allreduce of the 8192 × 5120 chunk (84 MB × 129 per step, ~2.1 ms each ≈ 40 GB/s, i.e. the PCIe limit): above the custom allreduce's 8–16 MB buffer, so it stays on NCCL, and no allreduce implementation moves it — fewer bytes would. GDN prefill is negligible.

**What W4A8 prefill is worth here:** replacing the Marlin chain with INT8 IMMA GEMMs at the measured 185–196 TOPS ([INT8 check](../gpu-unlock-2026-10-22/int8-imma-check-2026-11-01.md)) takes the 1.8–2.0 s to ~0.6–0.7 s per chunk: **−40 % on a cold 32K fill, −30 % at 131K, ~−20 % at 262K** (attention-dominated there). The per-chunk INT4 → INT8 weight materialisation (read 45 MB packed, write 90 MB) is bandwidth-trivial next to a 16 ms GEMM. The GEMM-path micro-benchmark (`bench_prefill_gemm_paths.py`) prices the same step kernel-by-kernel.

**Build path (no new CUDA for v1):** per-token INT8 activations (`per_token_quant_int8`) × a transient per-output-channel INT8 copy of each Marlin weight, through `sgl_kernel.int8_scaled_mm`, for M ≥ a prefill threshold only (decode stays W4A16). The one missing piece is a dequantizer that reads the *Marlin-permuted* INT4 layout: `gptq_marlin_repack` is a fixed, value-independent permutation of nibble positions, so its inverse gather map can be recovered once per weight shape by repacking position-encoding patterns, after which a Triton gather + `(q − z)·s` + per-channel INT8 rounding writes the INT8 [N, K] copy per layer per chunk. Quality gate: INT8 activations shape the KV cache for the whole context, so needle at 131K / 250K and MMLU / HumanEval on `qwen38`, plus the natural-instrument TTFT.
