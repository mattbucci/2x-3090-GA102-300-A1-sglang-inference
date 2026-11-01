# Fleet quality gate on the shipped image — custom allreduce + INT8 lm_head + GDN conv dtype (2026-10-31)

Fleet harness (`eval_and_chart.py --mmlu-samples 60 --humaneval-samples 30 --mc-budget 2560`) + sampled needle (`needle_band_probe.py`, 131K / 250K × depths 0.1 / 0.5 / 0.9, cache flushed, model-default sampling; Flint on the natural haystack) on every preset the levers touch, as shipped on `sglang-cuda-3090:local` = edfac69d2106's tree. qwen38 / qwen36 ran in the afternoon's custom-allreduce gate; the other seven in the evening.

| preset | MMLU | HumanEval | needle (found / cells) | same-day or prior reference |
|---|---:|---:|---:|---|
| `qwen38` | 52/57 | 29/30 | 6/6 | 53→52/57 · 29/30 (INT8-head A/B) |
| `qwen36` | 53/57 | 29/30 | 6/6 | 53→54/57 · 29/30 |
| `qwen36-dense` | 54/57 | 30/30 | 6/6 | 54→55/57 · 29/30 |
| `qwen35-moe` | 46/57 | 30/30 | 6/6 | 51→47/57 (143=143/171) · 29/30 |
| `coder-30b-eval` | 49/57 | 30/30 | 6/6 | 46→48/57 · 30/30 |
| `devstral` | 43/57 | 29/30 | 6/6 | 25/30 · 24/25 (10-25, 30-sample) |
| `gemma4` | 46/57 | 29/30 | 6/6 | 26/30 · 24/25 (10-25) |
| `flint` | 47/57 | 25/30 | 3/6 (3 cells were window-overflow 400s, not misses — re-probed below) | 68 % MMLU-30 · 29/40 HE (KV A/B) |
| `nemotron3-omni` | 50/57 | 30/30 | 6/6 | 30/30 · 25/25 (10-25) |

**Verdict:** no preset moves outside its same-day arms where one exists (five presets), and the four with only a 30-sample 10-25 reference (devstral, gemma4, nemotron3-omni) or a different-harness reference (flint) sit within binomial noise of it; HumanEval is 29–30/30 on every preset but Flint (25/30, above its 29/40 KV-A/B arms). Needle 6/6 on eight presets; Flint's 131K cells 3/3 and its 250K cells were three 400s from the probe (the natural haystack tokenizes past the window at the default chars/token — the probe now rescales from the server's count and records `error`; re-probe below).

## Same-day control arms (allreduce off; devstral also with the INT8 head off = the 10-25 config)

| preset | arm | MMLU | HumanEval | read |
|---|---|---:|---:|---|
| `devstral` | shipped (AR on, INT8 head) | 43/57 | 29/30 | |
| `devstral` | AR off | 43/57 | 29/30 | identical |
| `devstral` | AR off + bf16 head (the 10-25 config) | 43/57 | 29/30 | identical — the 10-25 25/30 was the 30-sample set, not a lever |
| `gemma4` | shipped (AR on; head tied, INT8 inert) | 46/57 | 29/30 | |
| `gemma4` | AR off | 47/57 | 29/30 | within one item — cleared |
| `nemotron3-omni` | shipped (AR on; INT8 head pinned off) | 50/57 | 30/30 | |
| `nemotron3-omni` | AR off | 52/57 | 30/30 | within two items, needle 6/6 — cleared |
| `flint` | AR on, 250K natural re-probe at 244,606 actual, 2 sampled passes | needle 2/6 (pass 1: 0.5 + 0.9 found, 0.1 "I don't know"; pass 2: 0/3 — 0.1 "I don't know", 0.5 "Bazinga", 0.9 think-loop to the budget) | | the known text-dependent band (≥ 240K, both KV dtypes, R9700 FP8 too); AR-off re-run below decides whether the allreduce shifts the odds |
| `flint` | AR off, same re-probe | _running_ | | |

