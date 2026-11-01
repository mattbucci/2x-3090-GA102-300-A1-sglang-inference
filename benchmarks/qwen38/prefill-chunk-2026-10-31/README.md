# qwen38 prefill chunk-size sweep — 2026-10-31 (NULL; ship stays 8192)

`--chunked-prefill-size` arms on the shipped `qwen38` config (TP=2, 262144, fp8 KV, MEM 0.85, 300 W, gpu-unlock), natural-text prompts with a cache flush before every point (`bench_long_context.py --prompt-source natural`), so TTFT is a true fresh prefill. Override via `OVERRIDE_ARGS` (the preset hardcodes 8192; `--max-prefill-tokens` raised to match above 16384). One boot per arm, same KV pool on every arm (680,638 tokens).

| chunk | TTFT @32K | @131K | @262K | decode @32K / 131K / 262K |
|---:|---:|---:|---:|---|
| 4096 | 13.18 s | 72.08 s | 189.2 s | 80.6 / 68.9 / 58.7 |
| **8192 (ship)** | **13.39 s** | **71.30 s** | **185.5 s** | 80.8 / 69.1 / 58.7 |
| 16384 | 14.09 s | 71.92 s | 186.8 s | 80.6 / 69.0 / 58.4 |
| 32768 | OOM on the first 32K prefill (activations of a 32K chunk do not fit the ~3 GB left beside the pool on either card) | | | |

Spread across the three serving arms is ≤ 5 % at 32K and ≤ 2 % at depth, inside run-to-run noise; the chunk size is not a lever. Prefill throughput on this rig is **2.45K tok/s at 32K → 1.84K at 131K → 1.41K at 262K**: bound by the fp16 tensor rate at the 300 W cap (the 10-24 power sweep moved 128K TTFT −17 % at 330 W, and the GA102 unlock's 2× tensor rate only appears at 400 W) plus the quadratic attention share at depth. The prefill levers that remain are the tensor rate itself — W4A8 Marlin (INT8 tensor cores run 2× fp16 on Ampere; activation-quant quality gate) or the power cap the thermal profile forbids — not scheduling.

Files: `chunk-{4096,8192,16384}.json`.
