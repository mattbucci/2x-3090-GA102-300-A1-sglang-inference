# Fleet chart refresh 2026-10-31

Every chart preset re-swept (10-point depth sweep, `bench_long_context.py`, 300 W cap, gpu-unlock on, CPU boost off) as shipped after the day's levers: custom allreduce ON, INT8 `lm_head` (patch 069), GDN conv state in the model dtype, Flint on fp8_e5m2 KV. One boot per preset, same instrument as the 2026-10-25 refresh, whose sweeps stay beside each `results.json` as `results-300w.json` (they pair with the same-boot `results-400w.json` for the power-cap chart).

| slug | 1K 10-25 → 10-31 | deep 10-25 → 10-31 (ctx) |
|---|---|---|
| flint | 166.1 → **185.2** (+11 %) | 93.2 → **114.8** (+23 %) @ 261,916 |
| qwen38 | 73.1 → **86.3** (+18 %) | 52.1 → **58.6** (+12 %) @ 261,916 |
| qwen3.6-35b-a3b | 222.7 → **288.2** (+29 %) | 129.0 → **148.6** (+15 %) @ 261,916 |
| qwen3.6-ream | 223.7 → **285.7** (+28 %) | 157.2 → **186.6** (+19 %) @ 261,916 |
| qwen3.6-27b | 73.2 → **86.4** (+18 %) | 52.0 → **58.6** (+13 %) @ 261,916 |
| qwen3.5-28b-moe | 221.7 → **285.7** (+29 %) | 156.5 → **185.9** (+19 %) @ 261,916 |
| qwen3-30b-ream | 206.6 → **248.8** (+20 %) | 73.4 → **78.4** (+7 %) @ 261,916 |
| coder-30b-awq | 210.1 → **252.5** (+20 %) | 74.0 → **78.9** (+7 %) @ 261,916 |
| coder-reap-25b | 206.6 → **250.0** (+21 %) | 73.6 → **78.4** (+7 %) @ 261,916 |
| devstral-24b-awq | 92.5 → **103.1** (+11 %) | 40.6 → **42.6** (+5 %) @ 261,915 |
| gemma4-26b-awq | 84.2 → **90.4** (+7 %) | 61.2 → **64.6** (+6 %) @ 261,916 |
| gemma4-31b | 57.8 → **61.7** (+7 %) | 32.3 → **33.4** (+3 %) @ 261,916 |
| gemma4-12b | 108.3 → **117.9** (+9 %) | 61.8 → **65.0** (+5 %) @ 261,916 |
| gemma4-21b-reap | 84.9 → **90.2** (+6 %) | 61.6 → **64.5** (+5 %) @ 261,916 |
| nemotron3-omni | 101.3 → **100.5** (-1 %) | 93.7 → **95.2** (+2 %) @ 261,916 |
| qwen3-vl-32b | 66.1 → **74.0** (+12 %) | 38.0 → **40.5** (+7 %) @ 130,844 |
| qwen35-27b-awq | 73.1 → **86.3** (+18 %) | 69.2 → **80.9** (+17 %) @ 32,540 |

The DSpark spec arm was NOT re-swept: the random-prompt sweep is not a spec-arm instrument (acceptance on random tokens, deep request served from the warmup prefix); the spec chart keeps its 2026-09-27 natural-text receipts.
