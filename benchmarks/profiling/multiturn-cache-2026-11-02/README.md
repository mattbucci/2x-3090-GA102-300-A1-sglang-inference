# Prefill CUDA graph — fleet gate on the flashinfer presets (2026-11-02)

Follow-up to the `qwen38` ship ([2026-11-01 receipt](../multiturn-cache-2026-11-01/README.md)). Per preset, the shipped
config (`off`) against `--cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 8 16 32 64 128
--cuda-graph-max-bs-prefill 1 --cuda-graph-max-seq-len-prefill 128` (`on`), same boot order, bare tree
(`/data/sglang-rebase-v0522`, patches 070–073). Instrument: `scripts/bench/measure_extend_cost.py --multi-turn`
(prime a prompt + 200-token reply, then an agentic-shaped follow-up: the reply replayed as history plus a 1- or
64-token user message; TTFT of that turn, 2 runs, median). Gate on the `on` arm: needle 131K (natural haystack, 3 depths,
sampled), MMLU-57 + HumanEval-30 (`eval_and_chart.py`), `validate_capabilities.py`.

| Preset | arch / backend | 1-token turn off → on, ms (32K / 131K / 197K) | 64-token turn off → on, ms | Gate on the graph arm | Decision |
|---|---|---|---|---|---|
| `qwen36` | DeltaNet+MoE, flashinfer | 77 / 89 / 83 → **32 / 37 / 37** | 75 / 83 / 87 → 29 / 43 / 51 | needle 3/3 (one sampled runaway cell at 0.9, 3/3 on re-probe ×3), MMLU 54/57, HE 30/30, caps 5/5 | **ON** |
| `qwen36-dense` | DeltaNet dense, flashinfer | 84 / 91 / 90 → **(427†) / 44 / 48** | 82 / 97 / 102 → 43 / 87 / 102 | needle 3/3, MMLU 55/57, HE 30/30, caps 5/5 | **ON** |
| `qwen35-moe` | DeltaNet+MoE, flashinfer | 78 / 88 / 83 → **32 / 34 / 38** | 77 / 84 / 89 → 29 / 45 / 53 | needle 3/3, MMLU 46/57, HE 30/30, caps 5/5 | **ON** |
| `coder-30b-eval` | MoE (compressed-tensors Marlin-MoE), flashinfer | (164†) / 84 / 108 → **41 / 63 / 84** | 59 / 67 / 84 → 56 / 58 / 77 | needle 3/3, MMLU 50/57, HE 30/30, caps 2/2 | **ON** (~25 %: the expert GEMMs sit outside the breakable graph) |
| `devstral` | dense GQA (Mistral), flashinfer | 50 / 78 / 106 → 51 / 73 / 111 | 38 / 62 / 80 → 35 / 62 / 84 | needle 3/3, MMLU 43/57, HE 29/30, caps 3/3 | **NULL** — the extend is attention-bound, not launch-bound; stays off. First arm crashed at capture → patch 073 |
| `nemotron3-omni` | Mamba2 hybrid, flashinfer | 148 / 501 / 158 → identical | identical | — | **n/a** — server disables the capture ("some layers do not apply Standard GQA"). The 142–608 ms turn itself is the hybrid-cache arch-list gap (patch 074, A/B pending) |

† first request of the arm after boot (warm-up), not a steady-state number.

**Reading.** On the DeltaNet presets the cache-hit extend was launch-bound (≈180 ms of CPU wall for 70–85 ms of
kernels at any extend size), so graph replay takes the agentic 1-token turn to 32–48 ms flat in depth, 2.3× the eager
path and the same level `qwen38` reached yesterday. The CT Marlin-MoE coder gains less because its expert GEMMs run
outside the breakable graph; the dense Mistral model gains nothing because its extend is already GPU-bound. The
capture costs ≈ 0.1 GB per preset and fit beside every KV pool at the presets' `MEM` settings (coder-30b-eval had
1.9 GB free at capture).

**Patch 073 (found here).** `PrefillCudaGraphRunner` read `quant_config` off the top-level model object; Devstral's
`Mistral3ForConditionalGeneration` is a `__getattr__` proxy over Llava with no such attribute, and
`unified_apply_gptq_marlin_gemm` dereferences `.quant_type` on it inside the first capture
(`AttributeError: 'NoneType' object has no attribute 'quant_type'`). The runner now resolves the config through
`language_model` / `inner` / the first quantized module. The fixed arm measured the null above.

**Files.** `<preset>-pg-{off,on}.json` (this directory; devstral's `on` arm from the patched tree, nemotron's two
arms identical), quality JSONs `benchmarks/quality/<preset>-pg-2026-11-02.json` and `<preset>-pg-needle-2026-11-02.json`;
nemotron turn-cost sweeps `nemotron3-omni-turn-{depths,131k}.json`. Launch-side changes: `scripts/launch.sh`
`CUDA_GRAPH=` lines of the four flipped presets.

## Prefix-aware split-KV route cap (patch 070 v4, 2026-11-02)

Same instrument, suffixes 64 / 96 / 128 new tokens, `off` = shipped cap 64 everywhere, `on` = `SGLANG_EXTEND_SPLITKV_MAX_TOKENS_DEEP=128` once the cached prefix is ≥ 32,768 tokens (`*-deepcap-{off,on}.json`).

| Preset | turn | 32K off → on | 131K off → on | 197K off → on |
|---|--:|---|---|---|
| `flint` | 64 | 93 → 84 | 94 → 92 | 106 → 103 |
| `flint` | 96 | 77 → 76 | 95 → 93 | 107 → 104 |
| `flint` | 128 | 76 → 73 | **136 → 105** | **194 → 121** |
| `gemma4-31b` | 64 | 90 → 86 | 152 → 151 | 318 → 316 (first-row effect, both arms) |
| `gemma4-31b` | 96 | 83 → 84 | 153 → 153 | 207 → 207 |
| `gemma4-31b` | 128 | 111 → 111 | **273 → 190** | **382 → 254** |

Reading: at 128 new tokens the serial extend is 1.5–1.9× the chunked route at depth on both presets; at 96 the two already cost the same, so the win is confined to the 97–128 band and to prefixes ≥ 32K (below that the cap stays 64 on purpose). Greedy 131K needle 3/3 on both on-arms with outputs identical to the off-arm runs (gemma 96 / 120 / 121 tokens). Shipped as the default on `flint`, `gemma4-31b`, `gemma4`.
