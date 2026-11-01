# Prefill GEMM paths at M = 8192 on the unlocked GA102 (300 W) — 2026-11-01

`scripts/bench/bench_prefill_gemm_paths.py`, GPU 0, CUDA-event timing. Marlin W4A16 = the shipped AWQ path (uint4b8, group 128, same pack / repack as the INT8-head patch); INT8 = CUTLASS `int8_scaled_mm` with per-token activation quant (`per_token_quant_int8`) and the transient INT4 → INT8 weight pass modelled as its bandwidth cost (read packed INT4, write INT8 [N, K]).

| M × K × N (qwen38 shape) | FP16 ref | Marlin W4A16 | INT8 GEMM | + act quant | + weight pass = W4A8 | speed-up vs Marlin |
|---|---:|---:|---:|---:|---:|---:|
| 8192 × 5120 × 17408 | 13.06 ms (111.8 TF) | **17.35 ms** (84.2 TF) | 7.9 ms (184.9 TOPS) | 7.71 ms | **8.04 ms** | **2.16×** |
| 8192 × 17408 × 5120 | 13.09 ms (111.6 TF) | **18.68 ms** (78.2 TF) | 7.46 ms (195.7 TOPS) | 7.58 ms | **7.89 ms** | **2.37×** |
| 8192 × 5120 × 5120 | 4.05 ms (106.1 TF) | **5.04 ms** (85.3 TF) | 2.29 ms (187.2 TOPS) | 2.35 ms | **2.45 ms** | **2.06×** |
| 8192 × 5120 × 8192 | 6.61 ms (104.0 TF) | **8.23 ms** (83.5 TF) | 3.61 ms (190.2 TOPS) | 3.65 ms | **3.81 ms** | **2.16×** |

**Reading:** Marlin at M = 8192 runs 78–85 TFLOPS (its INT4 dequant sits on the FP16 tensor-core path, ~25 % under plain FP16); the INT8 path lands at 185–196 TOPS and the two extra passes cost < 5 %. Net **2.1–2.4× per GEMM**, i.e. the 1.8–2.0 s Marlin chain per prefill chunk ([profile](prefill-profile-2026-11-01.md)) becomes ~0.85 s. Quality (per-token INT8 activations on Qwen's outlier channels) is the open question the served gate answers.
