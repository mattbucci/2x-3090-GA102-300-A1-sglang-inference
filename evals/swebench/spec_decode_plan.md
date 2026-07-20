# Spec-decode wiring plan for SWE-bench bake-off (CLOSED — not viable at 256K on 24GB)

**Final status (2026-05-31):** CLOSED. Per user constraint, only 256K context is acceptable for this rig. Spec-decode + 256K + AWQ-int4 30B-class target simultaneously **does not fit on 24GB cards** (R9700's 32GB cards fit it; ours don't). The actual VRAM accounting:

- coder-30b AWQ-Marlin weights @ TP=2: ~15 GB total (~7.5 GB/card)
- EAGLE3 draft (BF16) + cuda graphs: ~3-5 GB headroom per card
- 256K KV at qwen3-coder's 36 KB/tok: ~9 GB
- Per-card budget: 7.5 + 4 + 4.5 (KV/2) = ~16 GB → fits at 256K **no-spec**
- Adding spec: 7.5 + 4 + 4.5 + 0.4 (draft) + ~5 (draft cuda graphs) = ~21 GB → OOM region on 24 GB MEM=0.85 (~20.4 GB/card target)

R9700 (32 GB) has ~8 GB more headroom; that's exactly what spec-decode needs at full context. Our 24 GB hardware doesn't have it. Closing the spec-decode lane for SWE-bench-class workloads.

**What remains preserved:** the `SPEC_DECODE=1` env opt-in wiring in `launch.sh` (commit 2ad2e53) stays in place for future **short-prompt** uses (chat bots, synthetic decode benchmarks where prompts are <16K). Just not for SWE-bench.

**One remaining viable path: Nemotron-3-Nano-Omni** specifically — A3B MoE with smaller active params + Mamba2-hybrid KV makes the spec-decode budget different. R9700 already serves the FP8 variant at full 256K (no spec yet). Tasks #27 (train EAGLE3) and #28 (NGRAM) for that specific model remain open, gated on the AWQ ship landing.

---

# Original plan (historical)


**Status:** PROPOSED — wait for the current v0.5.12 bake-off (started 2026-05-30
01:36, qwen36 cycle in flight) to finish before executing any of this. We are at
the **planning** stage; no servers should be restarted, no presets edited mid-run.

**Goal:** cut full 8-preset sweep ETA from ~5-7 days to ~2-3 days by acceleating
the decode portion of each opencode/claw-code/little-coder rollout via
speculative decoding. Coverage is ~half the queue; the other half runs at
baseline (no published drafts exist).

---

## Why this is bounded by decode-only acceleration

A SWE-bench instance is a multi-turn loop: opencode/claw/little-coder loads
repo files into the prompt (prefill, K tokens), the model writes a tool call /
patch / reasoning chunk (decode, smaller N tokens), the harness runs the
tool/test, and the conversation continues. EAGLE3 + DFlash accelerate **only the
decode phase** — the draft proposes N tokens, the target verifies them in one
forward pass. Prefill is untouched.

So the per-instance wall time savings ≠ the headline decode tok/s number. From
our 2026-05-29 measurements (`benchmarks/quality/specdec-v0512-2026-05-29.json`):

| Preset | Baseline | + spec | Decode speedup | Realistic instance speedup if decode is 60-70% of wall time |
|---|:---:|:---:|:---:|:---:|
| qwen36 + DFlash | 30.6 tok/s | 126.3 tok/s | 4.1× | ~2.0-2.5× |
| coder-30b + EAGLE3 | 185.5 tok/s | 306.0 tok/s | 1.65× | ~1.3-1.4× |

These are projections; the actual wall-time gain must be measured (see
"Bench script" below).

---

## ⚠ Reality check: prompt-size distribution makes spec-decode largely useless for SWE-bench at the documented caps

**Measured 2026-05-31 against the finished qwen36-opencode-v2 cycle (300 instances).** Peak prompt size per instance (max `tokens.input` across all opencode `step_finish` events) shows agentic SWE-bench Lite traffic is HEAVILY long-context:

| Quantile | Peak prompt tokens |
|---|---|
| median | **41K** |
| p75 | 63K |
| p90 | 82K |
| p95 | 103K |
| p99 | 133K |
| max | 230K |

| Cap | % instances exceeding |
|---|---:|
| EAGLE3 (16K) | **97.3%** |
| DFlash (32K) | **65.3%** |
| 64K | 22.3% |
| 96K | ~5% |
| 128K | <2% |

**Implication.** Running spec-decode at the documented caps would lose almost the entire workload. The original spec-decode plan implicitly assumed prompts fit in the cap; that assumption is wrong for SWE-bench agentic traffic. Two paths forward, both depend on task **#17 (dynamic spec→no-spec fallback)** which is now **mandatory not optional**:

1. **Dynamic per-instance fallback.** Inspect each instance's prompt token count and serve with spec at <16K, no-spec at >16K. Cost: per-instance server restart OR a two-server pattern OR a single SGLang process that can toggle. Highest-leverage path.
2. **Wider spec config.** Drop MEM=0.70 → 0.65 and see if EAGLE3 fits at 32K or 64K. May not fit on 24 GB cards; R9700's full ladder (topk=16/draft=32) OOMs at 16K here, so 64K spec is very unlikely.

**Task #17 sizing data (2026-07-19, lane 3090-E — [receipt](../../benchmarks/quality/coder30b-eagle3-poolcap-2026-07-19.json)).** The memory question is SOLVED and the threshold question is ANSWERED, and they point opposite ways:

- **Memory: path 2 works far beyond what the table above assumed.** `--max-total-tokens $((CTX+2048))` at MEM=0.80 boots coder-30b + EAGLE3 + cuda graphs at CTX = 32K / 64K / 96K / **full 262144** (10.9 / 11.9 / 12.7 / 16.3 GiB per card). The "draft needs ~15 GB headroom" row above was the *profiler's* over-provision (it allocates a 418,956-token pool at CTX=98K/MEM=0.80 — 4.3× context; the explicit cap reclaims ~7.2 GiB/card). A single 256K spec server is memory-feasible — no per-instance restart or two-server pattern needed for VRAM reasons.
- **But the fallback threshold is a DEPTH crossover, not a VRAM band.** Server-log-verified spec/no-spec ratios on novel-code generation: 1.53-1.76× at ~2.5K, **crossover ~8-10K**, then 0.86× @13.8K, 0.85× @39.6K, 0.76× @61.2K, 0.62× @91.9K, 0.49× @252.5K (accept_len 3.8 → 1.58). The fallback rule for a hypothetical toggle: **spec below ~8K prompt tokens, no-spec above** — which per the depth table covers only ~2.7% of SWE-bench traffic. EAGLE3 is a short-interactive lever, not a SWE-bench one; NGRAM remains the depth spec.
- `--speculative-draft-window-size` (the deep-KV counter-lever) is **blocked on v0.5.15**: FlashInfer's `FlashInferMultiStepDraftBackend` passes a shared `kv_indptr_buf` into each per-step backend, a path that asserts `num_wrappers==1`, while a draft sliding window forces `num_wrappers=2` (`WrapperDispatch.SLIDING_WINDOW`, `flashinfer_backend.py` ~L414). The flag validates at arg-parse and dies at boot — upstream-issue-worthy. Until that lands, the crossover above is the final word on this rig.

Receipt: [`benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json`](../../benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json). Task #12 done.

## VRAM / context cost on 24 GB cards

Both spec configs we validated cap the served context lower than the baseline:

| Preset | Baseline ctx | Spec config ctx | Why |
|---|:---:|:---:|---|
| coder-30b + EAGLE3 (steps=4 / topk=4 / draft=8) | 256K @ MEM=0.85 | 16K @ MEM=0.70 | draft cuda graphs + KV need ~15 GB headroom |
| qwen36 + DFlash (default ladder) | 256K @ MEM=0.85 | 32K @ MEM=0.70 + `SGLANG_ENABLE_SPEC_V2=1` + `--mamba-scheduler-strategy extra_buffer` | Mamba scheduler buffer + DFlash KV cap |

**Implication:** any SWE-bench Lite instance whose prompt exceeds the spec-cap
context will either get truncated (silent quality loss) or rejected (loud
failure). We need to measure the prompt-length distribution from the current
qwen36-opencode run before committing — early sample suggests opencode
prompts are mostly bounded under 32K but a tail exists.

**Mitigation:** the runner can dynamically fall back from spec → no-spec for
instances whose prompt exceeds the cap. This is a follow-on optimization; the
first version of this plan accepts the cap and reports how many instances are
affected.

---

## Coverage of our 8-preset bake-off queue

| Preset (queue order) | Base | Draft | Status | Action |
|---|---|---|---|---|
| 1. qwen36 | Qwen3.6-35B-A3B | z-lab/Qwen3.6-35B-A3B-DFlash | ✅ validated (`benchmarks/quality/specdec-v0512-2026-05-29.json`) | Wire DFlash into preset |
| 2. coder-30b | Qwen3-Coder-30B-A3B | lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge | ✅ validated (same receipt) | Wire EAGLE3 into preset |
| 3. devstral | Devstral-Small-2-24B | — (no published draft) | ❌ no spec available | Baseline |
| 4. gemma4-31b | Gemma 4 31B Dense | — | ❌ no spec available | Baseline |
| 5. qwen3-ream | Qwen3-30B-Instruct (REAM-merged) | — (only Coder has EAGLE3) | ❌ no spec | Baseline |
| 6. coder-30b-eval | Qwen3-Coder-30B-A3B (CT format) | lmsys EAGLE3 (same base) | ⚠ untested — same base as coder-30b but CT-format target | Validate before wiring |
| 7. coder-reap-25b | Qwen3-Coder-30B-A3B-REAP (96 experts) | lmsys EAGLE3 (built for 128 exp) | ⚠ untested — REAP changes expert routing; draft may mismatch | Validate before wiring |
| 8. qwen36-ream | Qwen3.6-REAM-A3B (192 experts) | z-lab DFlash (built for 256 exp) | ⚠ untested — REAM changes expert layout | Validate before wiring |

**Net:** 2 confirmed (qwen36, coder-30b), 3 plausible (coder-30b-eval,
coder-reap-25b, qwen36-ream), 3 baseline (devstral, gemma4-31b, qwen3-ream).
If all 5 plausible/confirmed work, that's **5 of 8 presets** with spec-decode.

---

## Wiring approach

Keep the baseline preset working at 256K + MEM=0.85; add a per-preset env knob
to opt into spec. Pattern (already in use for QUANT override on coder-30b):

```bash
# In scripts/launch.sh, coder-30b preset block:
if [[ -n "${SPEC_DECODE:-}" ]]; then
    # Opt-in: EAGLE3 spec-decode (validates 2026-05-29: 1.65× decode, ctx capped at 16K)
    MEM=0.70
    CTX=16384
    EXTRA_ARGS+=" --speculative-algorithm EAGLE3 \
                  --speculative-draft-model-path $MODELS_DIR/drafts/eagle3-coder30b \
                  --speculative-draft-model-quantization unquant \
                  --speculative-num-steps 4 --speculative-eagle-topk 4 \
                  --speculative-num-draft-tokens 8 \
                  --speculative-attention-mode decode"
fi
```

Same pattern for qwen36 + DFlash (with the `SGLANG_ENABLE_SPEC_V2=1` env +
`--mamba-scheduler-strategy extra_buffer` + `--dtype bfloat16`).

The bake-off wrapper (`evals/swebench/run_model_cycle.sh`) then sets
`SPEC_DECODE=1` for the eligible presets. Baseline presets ignore the env.

---

## Bench script (acceptance gate before full re-sweep)

Write `evals/swebench/bench_swebench_instance_time.py`:

1. Pick 5 SWE-bench Lite instances spanning prompt sizes (e.g. 4K / 8K / 16K /
   24K / 32K-truncated).
2. For each instance, run opencode rollout twice: spec-on, spec-off.
3. Record per-instance: total wall time, decode tok/s observed, accept_len
   (from `/get_server_info`), prompt-token count, whether the patch resolved.
4. Output a table: `instance | prompt_tok | t_off (s) | t_on (s) | speedup |
   resolved_off | resolved_on`.
5. Gate: full re-sweep only if median speedup ≥ 1.5× and no resolved-count
   regression on the 5-instance set.

Run for both qwen36+DFlash and coder-30b+EAGLE3 separately (so we have two
data points per preset).

---

## Step ordering

1. **WAIT** for current bake-off to finish (~5-7 days from 2026-05-30 01:36).
2. **Measure** prompt-length distribution from the finished qwen36-opencode
   predictions (`predictions.jsonl`) — what fraction exceed 16K / 32K? This
   sizes the context-cap risk.
3. **Validate** the 3 untested cells (coder-30b-eval, coder-reap-25b,
   qwen36-ream) with a single coherence probe each — does the draft load,
   does the target server stay up, does the accept rate stay above ~0.3?
   Cells that fail this fall back to baseline; cells that pass move on.
4. **Bench script** (5-instance subset) on the confirmed cells. Gate: ≥1.5×
   median speedup, no resolved-count regression.
5. **Wire** the `SPEC_DECODE` opt-in env into the eligible launch.sh presets
   (commit + push).
6. **Re-run** the 8-preset bake-off with `SPEC_DECODE=1` set for the eligible
   cells; baseline cells run unchanged.
7. **Compare** the new resolved% numbers + per-instance wall times against
   the current bake-off's baseline. Publish receipts under
   `benchmarks/quality/swebench-spec-v0512-<date>.json`.

---

## Risk register

- **Context cap excludes instances** — measured in step 2, may require the
  dynamic-fallback runner extension.
- **Spec-on resolved% regresses** — accept rate translates to draft-accepted
  tokens; if the target's actual generation behavior shifts under spec (rare,
  but observable on some configs), some instances that resolved at baseline
  may fail at spec. Gated by step 4.
- **REAM/REAP base mismatch** — drafts trained on the unmodified base may
  produce stale logits when the target is REAM/REAP-merged. Gated by step 3.
- **Bake-off scheduling** — if the user redirects mid-stream (e.g. a new model
  to graft), pause this plan rather than racing it.
