# INT8 tensor path (IMMA) on the unlocked GA102s — 2026-11-01

GPU 0, 300 W cap, gpu-unlock on, CUDA-event timing over 20 iterations. The 2026-10-22 receipt's INT8 number (torch `_int_mm`, 41 → 43 TOPS stock → unlocked) sat 5× under spec while FP16 doubled; this separates the kernel choice from the hardware.

| M × K × N | FP16 (fp32 acc) TFLOPS | torch `_int_mm` TOPS | `sgl_kernel.int8_scaled_mm` (CUTLASS) TOPS | INT8 / FP16 |
|---|---:|---:|---:|---:|
| 8192 × 8192 × 8192 | 93.8 | 51.7 | **163.2** | 1.74× |
| 8192 × 5120 × 5120 | 82.2 | 52.3 | **196.1** | 2.39× |
| 8192 × 5120 × 17408 | 111.1 | 47.3 | **185.3** | 1.67× |

**Verdict:** INT8 IMMA is open and runs ~2× unlocked FP16 at the 300 W cap; `_int_mm` was a cuBLASLt heuristic miss, not a gate. W4A8 prefill is a GO. First version without new CUDA: per-token INT8 activations (`per_token_quant_int8`, already in the tree) × a per-output-channel INT8 copy of the Marlin weight made transiently per prefill chunk (AWQ dequant → INT8 round; an INT4 grid re-rounds onto INT8 nearly losslessly), through `int8_scaled_mm`, only when M is large (prefill); decode stays on Marlin W4A16. Gate: prefill TTFT on the natural instrument at 32K / 131K / 262K, MMLU 60 + HumanEval 30 + needle at 131K / 250K on `qwen38`; the activation quant is the quality risk (Qwen outlier channels, SmoothQuant regime) and it also shapes the KV cache for the whole context.

Generator: `scripts/bench/int8_imma_check.py`; raw `int8-imma-check-2026-11-01.json`.
