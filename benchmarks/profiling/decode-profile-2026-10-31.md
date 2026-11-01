# Decode step profiles after the 2026-10-31 levers (custom allreduce, INT8 lm_head, GDN conv dtype)

Torch-profiler windows of 64 single-user decode steps (3 warm-up steps skipped) after a 1-token warm of the same natural-text prompt, attributed by `scripts/bench/trace_step_anatomy.py` (EXPOSED time: a sweep line over the GPU timeline — overlapped categories are reported as `a+b`). Driver `scripts/bench/profile_decode_depth.sh` (port of R9700's), one boot per preset, as shipped (TP=2, 300 W, gpu-unlock, graphs). Rank 0 shown; rank 1 within 2 %. Traces under `/data/tmp/profile-*`, per-rank tables beside this file.

## qwen38 · short window (actual prompt_tokens=135)

`CPU wall 1.75 ms | GPU span 11.55 ms | kernel-sum 12.93 ms`

| ms/step | share | category |
|---:|---:|---|
| 7.914 | 68.5% | marlin(int4) |
| 1.401 | 12.1% | gemm/gemv(fp16)+marlin(int4) |
| 0.559 | 4.8% | gdn |
| 0.486 | 4.2% | norm |
| 0.433 | 3.7% | misc |
| 0.278 | 2.4% | gemm/gemv(fp16) |
| 0.236 | 2.0% | attention |
| 0.121 | 1.1% | act |
| 0.054 | 0.5% | idle-gap |
| 0.040 | 0.3% | nccl |

## qwen38 · deep window (actual prompt_tokens=244115)

`CPU wall 1.77 ms | GPU span 16.87 ms | kernel-sum 18.26 ms`

| ms/step | share | category |
|---:|---:|---|
| 7.992 | 47.4% | marlin(int4) |
| 5.047 | 29.9% | attention |
| 1.413 | 8.4% | gemm/gemv(fp16)+marlin(int4) |
| 0.745 | 4.4% | misc |
| 0.593 | 3.5% | gdn |
| 0.521 | 3.1% | norm |
| 0.298 | 1.8% | gemm/gemv(fp16) |
| 0.132 | 0.8% | act |
| 0.047 | 0.3% | idle-gap |
| 0.047 | 0.3% | nccl |

## qwen36 · short window (actual prompt_tokens=135)

`CPU wall 1.95 ms | GPU span 4.48 ms | kernel-sum 4.70 ms`

| ms/step | share | category |
|---:|---:|---|
| 1.644 | 36.7% | marlin(int4) |
| 0.627 | 14.0% | idle-gap |
| 0.435 | 9.7% | gemm/gemv(fp16)+marlin(int4) |
| 0.433 | 9.7% | misc |
| 0.327 | 7.3% | gdn |
| 0.269 | 6.0% | marlin(int4)+misc |
| 0.255 | 5.7% | norm |
| 0.143 | 3.2% | act |
| 0.129 | 2.9% | attention |
| 0.079 | 1.8% | act+misc |
| 0.046 | 1.0% | gemm/gemv(fp16) |
| 0.044 | 1.0% | gemm/gemv(fp16)+memcpy/memset |

## qwen36 · deep window (actual prompt_tokens=244115)

`CPU wall 1.80 ms | GPU span 6.78 ms | kernel-sum 7.53 ms`

| ms/step | share | category |
|---:|---:|---|
| 3.033 | 44.8% | attention |
| 1.629 | 24.0% | marlin(int4) |
| 0.464 | 6.8% | misc |
| 0.426 | 6.3% | gemm/gemv(fp16)+marlin(int4) |
| 0.311 | 4.6% | gdn |
| 0.254 | 3.8% | marlin(int4)+misc |
| 0.243 | 3.6% | norm |
| 0.136 | 2.0% | act |
| 0.076 | 1.1% | act+misc |
| 0.065 | 1.0% | idle-gap |
| 0.043 | 0.6% | gemm/gemv(fp16) |
| 0.042 | 0.6% | gemm/gemv(fp16)+memcpy/memset |

## Reading

- **qwen38 (27B dense GDN hybrid), short:** 11.55 ms GPU span = 86 tok/s. INT4 weight streaming (Marlin) is 68.5 % exposed; NCCL is gone (0.3 %, the custom allreduce kernel sits in `misc` at 0.35 ms); the INT8 `lm_head` is one 0.76 ms Marlin call. What is left outside weight streaming: the GDN recurrent step 0.56 ms, norms 0.49 ms, the custom allreduce 0.35 ms, attention 0.24 ms (16 full-attention layers at 135 tokens).
- **qwen38, deep (244K):** 16.87 ms = 59 tok/s. The extra 5.3 ms is attention: 16 flashinfer decode kernels at 0.31 ms each, which is the fp8 KV read for the layer (244K × 2 KV heads × 256 × 2 × 1 B ≈ 250 MB per rank → 0.28 ms at 900 GB/s): **attention runs at the KV-read roofline**; the only way down is fewer KV bytes (fp8 is already on; KV at lower precision or a sliding window are model changes, not serving knobs).
- **qwen36 (35B-A3B MoE GDN hybrid), short:** 4.48 ms = 286 tok/s, kernel-sum 4.70. Dense Marlin 1.23 + MoE Marlin 0.81 + custom allreduce 0.34 + `lm_head` 0.32 + GDN 0.27. **14 % of the step (0.63 ms) is idle gap** — ~1,000 kernels per step (80 router/fill/align/sum kernels per MoE layer) and the alt-stream handoffs; inter-kernel bubbles inside the graph replay. That is the one non-bandwidth item of size on the A3B presets: a kernel-count reduction in the MoE glue (fills, `align_single_token`, `moe_sum_reduce`, router) — upstream-tracked territory, parked.
- **qwen36, deep (244K):** 6.78 ms = 148 tok/s; attention 3.03 ms (10 full-attention layers × 0.29 ms, the same KV-read roofline), idle gap shrinks to 1 % (the attention kernels fill the bubbles).
- **`in_proj_ba` fp16 gemv is NOT a lever** — the traced 33 µs per call (48 per step on qwen38, 1.6 ms kernel-sum) is the kernel stretched under the concurrent Marlin GEMM; in isolation the identical cuBLAS pair takes 4.5 µs and a Triton GEMV is 2× slower ([micro-benchmark](ba-gemv-microbench-2026-10-31.md)). Its true cost is the 0.28 ms/step it is exposed alone (2.4 %), which a different kernel cannot recover because the shared window is Marlin-bound. Lesson for every future read of these tables: a kernel's traced duration under a concurrent stream is not its cost — verify in isolation before building a replacement.

## What this leaves on the table (serving side)

| preset | step | weight streaming | attention (KV read) | everything else | of which addressable |
|---|---:|---:|---:|---:|---|
| qwen38 short | 11.55 ms | 7.9 ms (68 %) | 0.24 ms | 3.4 ms | norms 0.49 (fusion candidates are per-layer RMSNorm+residual, already fused), custom AR 0.35 (at the IPC floor), GDN 0.56 |
| qwen38 deep | 16.87 ms | 8.0 ms (47 %) | 5.05 ms (30 %, roofline) | 3.8 ms | same |
| qwen36 short | 4.48 ms | 2.0 ms (45 %) | 0.13 ms | 2.3 ms | **idle gaps 0.63 ms** (kernel count), custom AR 0.34, MoE glue |
| qwen36 deep | 6.78 ms | 2.0 ms (29 %) | 3.03 ms (45 %, roofline) | 1.7 ms | custom AR, MoE glue |

Weight streaming at 7.9 ms for ~7.3 GB of INT4 per rank per step on qwen38 is ≈ 925 GB/s — the GA102 at its 936 GB/s spec. The dense presets are at the memory roofline for both weights and KV; decode gains from here are model-side (fewer bytes: smaller/compressed KV, speculative decoding — DSpark is the shipped example at 1.4–1.5×) rather than kernel-side. The A3B MoE presets keep one serving-side lever: the per-step kernel count.
