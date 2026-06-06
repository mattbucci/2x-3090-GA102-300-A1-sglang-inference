# int4-AWQ tool-call reliability & resolve rate vs context length (2026-06-05)

Tests the hypothesis: *AWQ models lose accuracy at long context and garble tool calls.*
Data: opencode `--format json` rollout logs from the v0.5.12 SWE-bench Lite bake-off.
Per **step_finish** we read the true prompt size (`tokens.input`); opencode marks a
malformed tool emission as `part.tool == "invalid"` ("Model tried to call unavailable
tool '<garbled text>'"). Tool: `evals/swebench/context_reliability_curve.py`.

## Result 1 — tool-call GARBLING does NOT replicate on our int4 stack
Invalid-tool-call rate by context bucket stays **< 0.5% everywhere**, and does not rise
monotonically with context:

| model (type) | 0-16K | 16-32K | 32-64K | 64-128K |
|---|---|---|---|---|
| qwen36 (MoE-think) | 0.05% | 0.13% | 0.47% | 0.21% |
| qwen36-ream (MoE-think) | 0.08% | 0.10% | 0.20% | 0.12% |
| coder-30b-ream | 2.47%¹ | 0.07% | 0.08% | 0.20% |
| coder-reap-25b | 0.18% | 0.00% | 0.00% | 0.00% |
| **devstral (dense, no-think)** | 0.00% | 0.00% | 0.00% | **0.00% (0 / 262K calls)** |

¹ a short-context spike (56 invalids in a few instances), not a long-context pattern.

**devstral made ~262K tool calls with zero garbles, including 86.8K calls at 64-128K.**
So int4-AWQ tool-call *format* is robust to context on our models. The garbling R9700
observed (71 invalid tool calls on xarray@82 steps) is **real but appears specific to
their dense Qwen3.5 / DeltaNet** — it is not a generic int4-at-long-context property.

## Result 2 — resolve rate DOES fall with context, but it is UNIVERSAL (hardness), not quant/thinking
Resolve rate by max context reached:

| model (type) | 16-32K | 32-64K | 64-128K | drop |
|---|---|---|---|---|
| qwen36 (MoE-think) | 76% | 59% | 36% | −53% |
| qwen36-ream (MoE-think) | 74% | 62% | 29% | −61% |
| coder-30b-ream | 56% | 35% | 11% | −80% |
| coder-reap-25b | 51% | 40% | 12% | −76% |
| devstral (dense) | 19% | 18% | 12% | (empty-limited) |

Every architecture — thinking and non-thinking coder alike — collapses at 64-128K, the
**coders steeper than the thinking models**. This is the *hardness confound*: deep-context
instances are intrinsically harder (bigger repos, more navigation), and all models do
worse on them. It is **not** a thinking-specific spiral nor a quant-specific tool decay.
If anything the MoE-thinking models are the **most context-robust** in the fleet.

## Honest conclusions
1. **Garbling-at-long-context is NOT happening on our int4 MoE/coder/dense models** (<0.5%,
   devstral 0/262K). The user's hypothesis is true for R9700's *dense Qwen3.5*, not ours.
2. **Resolve-vs-context decline is real but universal** → dominated by instance hardness,
   not by int4 tool-format decay or thinking-overthinking.
3. The earlier byte-size correlation ("empties 3.3× longer") was real but conflated *long*
   with *hard*; the token-level + cross-architecture controls dissolve the quant story.

## Caveats / what this does NOT prove
- **No data above ~128K.** SWE-bench Lite agentic contexts top out ~128K (the 128K+ bucket
  has n=2-4 per cell). The true **256K regime is unmeasured** — AWQ decay could still appear
  above 128K; this cannot validate *or* refute the 256K-agentic claim.
- **Hardness confound not fully removed.** Cleanly isolating quant×length needs int4-vs-FP8/bf16
  on the *same* instances at the *same* context (R9700's FP8 4/6 vs int4 0/6 is that control
  for dense). Our MoE int4 is already fleet-leading, lowering the urgency.

## Next test to actually defend 256K
A synthetic long-context agentic probe: inject benign filler to push true context to
128K / 192K / 256K and measure tool-call validity + a needle-style action correctness —
the only way to measure the 256K regime SWE-bench Lite never reaches.
