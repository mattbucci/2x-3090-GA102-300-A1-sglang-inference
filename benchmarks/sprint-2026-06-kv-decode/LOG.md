# Experimentation sprint — KV economics + Gemma decode (2026-06-10, bake-off paused)

Lab notebook. One entry per experiment run: hypothesis → method → result → decision. All runs TP=2 on the live v0.5.12 tree (24 patches, `b212aa2`), single variable changed at a time, receipts as JSON next to this file. Instruments identical to the fleet eval (`run_v0512_fleet_eval.sh` mechanics: port 23334, MAX_RUNNING=6, `bench_long_context.py`, `probe_256k_tooluse.py`, `validate_capabilities.py`) so results compare directly to existing baselines in `benchmarks/<slug>/results.json`.

Patterns mined from the patch set that motivate the tracks:
- 043/047 (hybrid-SWA KV routing): the 12B "serves 256K" but **KV caps at 102K full / 81K swa**; 26B KV-walled at ~118K. SGLang sizes the SWA sub-pool at `--swa-full-tokens-ratio` **0.8** of the full pool by default — but a sliding layer can only attend `window=1024` tokens, and 40-of-48 (12B) / 25-of-30 (26B) layers are sliding. The SWA pool is the dominant KV consumer yet almost all of it is physically unreachable at MAX_RUNNING=1.
- 011/017/021 + the cuda-graph memory ("flat TPOT-vs-ctx = missing-graph tell"): Gemma 4 26B is our only family still decode-slow (~36-53 tok/s vs 126-196 elsewhere) because head_dim=512 forces triton attention → cuda-graph OFF.
- 007 (BV=64 sweep): default kernel configs under-serve sm_86 at M=1 — same logic applies to the MoE runner choice (marlin vs wna16) for the 26B.

## Track A — SWA sub-pool right-sizing (goal: true 256K KV on the Gemma hybrids)

**H-A1:** On `gemma4-12b` (window 1024, 40/48 sliding, int4 AWQ), `--swa-full-tokens-ratio` is the binding constraint on KV capacity. Lowering it from 0.8 toward the physical floor (~window + prefill chunk + slack ≈ 16K tokens ⇒ ratio ≈ 0.0625 at 262144 full) lifts the full pool to ≥ 262144 with **no** quality or decode regression — sliding layers cannot attend beyond 1024 tokens, so the freed slots were dead weight.

**Method:**
1. *Ladder* (boot-only, ~5 min/run): ratios {0.8 control, 0.5, 0.25, 0.1, 0.0625, 0.03}; scrape full/swa pool tokens + mem headroom from server log. Output: capacity-vs-ratio curve, `A1/ladder-*.json`.
2. *Probe* (3 runs: control, candidate, candidate-1-step): `validate_capabilities.py` (all modalities) + `probe_256k_tooluse.py` @ {16K, 65K, 131K, 196K, 256K} + `bench_long_context.py` decode curve. A too-small swa pool should fail loudly here (256K prefill thrash or assert), not silently.

**Decision rule:** ship-ratio = smallest ratio with boot OK ∧ caps all-PASS ∧ tool-use ≥ control at every length ∧ decode within 5% of control at every ctx — then ×2 safety. Wire into preset, note in README VRAM table. If no ratio reaches 262144 full tokens, record the actual ceiling + what became binding (weights? full pool? mem-fraction) and stop Track A1.

**H-A2:** same for `gemma4` (26B MoE AWQ, 25/30 sliding); KV wall ~118K → expect full 256K at ratio ≈ 0.0625–0.1.

## Track B — Gemma 4 26B decode (goal: ≥1.3× at 131K+, 5/5 caps held)

**H-B1:** torch_native attention + cuda-graph ON beats triton + graph OFF at M=1 long-ctx decode — launch-overhead domination (the R9700 dispatch-bound profile: ~48% GPU util) outweighs torch_native's slower kernels once graphs remove the fixed ~27 ms/step overhead.
Method: A/B `_ENV_CUDA_GRAPH` + `EXTRA_ARGS="--attention-backend torch_native"` vs preset; decode @ {1K, 32K, 131K, max}; caps battery both arms. Also record *why* triton+graph fails (capture error) — if it's shallow, that's a patch candidate.

**H-B2:** MoE runner choice (awq_marlin vs moe_wna16) is measurably different at M=1 E=128 gelu; the preset may be on the slower one.
Method: A/B QUANT routing, decode @ {1K, 131K}, same caps battery.

## Track C — fleet TPOT-vs-ctx audit (no GPU; runs in wait gaps)

Mine every `benchmarks/*/results.json` for the flat-TPOT tell (missing graphs / launch-bound) and for decode cliffs; tabulate slope 1K→max per preset. Anomalies become Track-B-style experiments.

---

## Run log

(entries appended chronologically by the sweep runner + analysis)

### 2026-06-10 01:55 — Track C: fleet TPOT-vs-ctx audit (existing results.json, fleet eval 2026-06-07)

| preset | max KV tok | ctx measured | TPOT lo→hi ms | ratio | tok/s lo→hi | cliff |
|---|---:|---|---|---:|---|---|
| devstral | 172,058 | 1K–128K | 10.8→17.4 | 1.61 | 92→58 | 64K→128K −19% |
| **gemma4-12b** | 102,094 | 1K–64K | 23.4→25.1 | **1.07** | 43→40 | – |
| **gemma4 (26B)** | 117,840 | 1K–64K | 29.3→29.6 | **1.01** | 34→34 | – |
| **gemma4-31b** | 24,248 | 1K–16K | 29.9→29.8 | **0.99** | 33→34 | – |
| qwen3-ream | 578,433 | 1K–256K | 5.5→9.5 | 1.75 | 183→105 | **64K→128K −26%** |
| qwen35-moe | 2,108,823 | 1K–256K | 5.9→7.3 | 1.23 | 170→138 | – |
| qwen36-dense | 657,010 | 1K–256K | 14.6→18.3 | 1.26 | 69→55 | – |
| qwen36 | 875,947 | 1K–256K | 5.8→7.9 | 1.38 | 174→126 | – |
| qwen36-ream | 2,166,819 | 1K–256K | 5.8→7.2 | 1.24 | 172→139 | – |

**Findings.** (1) The flat-TPOT missing-graph tell fires on **all three Gemma presets** (1.07 / 1.01 / 0.99 — a decode step that doesn't get more expensive from 1K to 64K KV is launch-bound, not compute-bound). Track B therefore targets the whole Gemma family, expected headroom 2–4× like the qwen36-family graph win. (2) `qwen3-ream` has a real −26% decode cliff at 64K→128K (5.5→9.5 ms TPOT) — healthy at 105 tok/s @256K but the step is anomalous → queued as **B3** (suspect: attention kernel regime change, e.g. split-KV path switch). (3) Coverage gaps corroborate Track A: the Gemma rows stop exactly where their KV walls sit (102K/118K/24K vs 578K–2.2M for the Qwen fleet); `coder-30b-awq/results.json` is an old format with no usable rows — re-bench when convenient. (4) Qwen3.5-27B legacy row (Apr 13, 13 tok/s, 75 ms flat TPOT) is the known replicated-DeltaNet case — superseded by qwen36-dense, not pursuing.
