# DSpark speculative decoding on qwen38 — CUDA (flashinfer verify) depth A/B

**Result: DSpark is a 1.4–1.5× single-user decode win on our CUDA stack across short AND deep context — it does NOT reproduce R9700's 0.24×-at-49K Triton cliff.** The at-depth cost R9700 measured is Triton-verify-specific; our flashinfer target-verify handles the deep KV, so DSpark is a genuine serving lever at the bake-off's 34–37K median context and beyond.

Ran during a deliberate ~40-min bake-off pause (opencode-dcp lane at 197/300; stopped the queue tree + docker server, tested bare-metal on :23399, reverted, relaunched with `--skip-existing` — the routine reboot-recovery path; the 198th instance re-rolls).

## Setup

- Target: `qwen38` preset checkpoint (`hf-mattbucci/Qwen3.8-27B-AWQ`, awq_marlin), TP=2, 260 W cap, CTX 65536, mem 0.85, `--max-mamba-cache-size 8`, cuda-graph decode bs=1.
- Draft: `RedHatAI/Qwen3.8-27B-speculator.dspark` (DSpark = DFlash + Markov + confidence head; 5 layers, ~2B, bf16, sliding-window 2048), config flattened from the vLLM `speculators` layout to the SGLang draft layout.
- SGLang **v0.5.20 + patch 099** (the `pp_proxy_tensors=` kwarg port into `DSparkWorkerV2.forward_batch_generation`; without it every DSPARK boot dies `TypeError` on the first forward — R9700 `6b59279` named this, verified against our tree, reconstructed here).
- `--speculative-algorithm DSPARK --speculative-num-draft-tokens 9` (gamma=8, = block_size 8 + 1) `--speculative-draft-model-quantization unquant --speculative-attention-mode decode --disable-overlap-schedule --mamba-radix-cache-strategy extra_buffer --dtype bfloat16`. Boot: `Initialized DSpark draft runner … gamma=8, verify_num_draft_tokens=9, markov_head=VanillaMarkov`, verify backend **flashinfer**.
- Protocol (R9700's): greedy (temp 0), `flush_cache` before each request, 2 runs/arm, decode tok/s = (completion_tokens−1)/(t_last−t_first) from streaming; accept-len from server `Decode batch` lines.

## Results (decode tok/s, single-user)

| depth | output | baseline (no-spec) | DSpark | speedup |
|---|---|---:|---:|---:|
| 60-tok | code, 400 tok | 77.8 | 119.5 | **1.54×** |
| 34K (bake-off median) | dense prose, 500 tok | 70.1 | 103.0 | **1.47×** |
| 49K | dense prose, 500 tok | 67.0 | 92.8 | **1.39×** |
| 49K | short factual continuation, 95 tok | 68.0 | 189.5 | 2.79× |

Accept-len 1.8–4.3 (higher on predictable continuations, ~2.5 on dense technical prose). The headline is the conservative dense-prose regime (1.4–1.5×); highly-predictable output peaks near 2.8×. Greedy spec-decode is lossless by construction — verify emits the target's exact tokens (R9700 confirmed byte-identical).

**Contrast with R9700 (`6b59279`, their FP8/RDNA4/Triton):** they measured 4.3× short / **0.24× at 49K** and attributed the depth cost to the target `TARGET_VERIFY` forward over deep KV on the Triton unsplit-extend kernel (their split-KV patch 065 gates on a tree `custom_mask` the linear γ+1 verify never sets). On our flashinfer verify that regression is absent — DSpark stays ≥1.39× at 49K. This is exactly the CUDA-vs-Triton divergence they flagged as worth one datapoint.

## Pool cost (the ship-gate caveat)

Draft weights + draft KV shrink the target pool. At CTX 65536 / mem 0.85: baseline `max_total_num_tokens=324317` → DSpark `154909` (≈halved), still ≫ 49K. R9700 saw 530K→192–207K at higher ctx and 244K unservable. **Before shipping at the production 262144 window, confirm the pool still holds ≥262144 with the draft** (baseline pool at 262144 is ~697K per the preset note; halving → ~350K > 262144, so likely fine — but verify, and 244K prompts may not serve at mem 0.85).

## Disposition / ship path

DSpark is the highest-value qwen38/qwen36-family serving lever measured to date — validated as a win at our real serving depth. To ship (boundary work, gated on needle + HumanEval + tool-probe though greedy spec is lossless):
1. Formalize **patch 099** via the 3-gate (`patches/099-dspark-pp-proxy-tensors-kwarg.patch`), rebuild the OCI image.
2. Add a `SPEC_DECODE=dspark` opt-in to the `qwen38` (and try `qwen36`) preset in `launch.sh` pointing at the flattened draft, `--speculative-num-draft-tokens 9`, `--mamba-radix-cache-strategy extra_buffer`.
3. Confirm 262144 pool holds with the draft; re-measure decode at the true KV cap + fresh prefill; `bench_regression.sh` the preset.
4. Ship the flattened draft to `mattbucci/Qwen3.8-27B-DSpark-sgl` (config-flatten recipe below).

Reproduction (scratch, this run): draft flatten `scripts/specforge/flatten_dspark_speculators_config.py`, depth bench `scripts/specforge/dspark_depth_bench.py`; per-run JSONs in `dspark-cuda-2026-09-27/`.
