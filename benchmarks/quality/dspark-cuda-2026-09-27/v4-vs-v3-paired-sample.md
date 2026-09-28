# qwen38 SWE-bench Lite opencode lane — DSpark spec (v4) vs no-spec (v3), instance-matched sample

**Date:** 2026-09-27 23:15 · **Decision gate:** "if spec walls MORE or errors, revert `qwen38` to no-spec at the lane boundary" · **Verdict: spec does NOT wall more and did not error — `SPEC_FOR=qwen38:1` stays on for the rest of the v4 cycle.**

Both arms: same model (`mattbucci/Qwen3.8-27B-AWQ`), same scaffold (opencode 1.14.25), same task prompt / sandbox contract (network none, stdin prompt, git re-init), same 1800 s wall, temp 1.0 (opencode default), same SGLang v0.5.20 stack (patch 099 added for v4), same true **262144** window. Only the served config differs:

| arm | run dir | served config (from `/get_server_info`, recorded in `meta.json`) |
|---|---|---|
| ref (no-spec) | `evals/swebench/runs/qwen38-opencode-v3` (300/300, closed 2026-09-24) | fp8_e4m3 KV, 652K-token pool, no draft |
| new (spec) | `evals/swebench/runs/qwen38-opencode-v4` (rolling; first 38) | `speculative_algorithm=DSPARK`, `num_draft_tokens=9`, draft `qwen38-dspark` (bf16), fp8_e4m3 KV, **307,659-token pool**, `context_length=262144` |

## Result (31 instance-matched pairs; wall = `rollout_seconds ≥ 1795` or rc 124)

```
                walls  empty  median s   mean s   sum h
ref (v3 no-spec)    3      3     991.6   1015.3    8.74
new (v4 DSpark)     2      3     738.9    854.1    7.36
per-instance speedup (ref/new, median): 1.16×
flips: wall→done 2 (django-11001, django-11283), done→wall 1 (django-11742), empty→patch 2, patch→empty 2 (django-11019, django-11742)
```

- **Walls:** 2 vs 3. `django-11564` walls in both arms (behavioural). `django-11742` v3 done 1524 s → v4 wall: the session was mid-final-summary at 1800 s with its edits on disk; the no-capture-at-wall convention discards them (capture-at-wall is a cycle-boundary item, not a spec effect).
- **Empties:** 3 vs 3 (walls carry no patch in either arm; the one non-wall empty, `django-11019`, is a model outcome — patch present in v3).
- **Rollout time:** median −25 % (992 → 739 s), mean −16 %, summed −1.4 h over 31 instances. `lane_overhead.py`: pre-session (image build + boot → first event) is unchanged at ~200 s in both arms, session median 735 s (v3, all 300) → 525 s (v4, n=31) — the gain is all in the model-bound phase, as expected from a decode-only lever.
- **Errors:** 0 server-side errors, 0 TRIPWIRE / BRIDGE CHECK / Traceback lines in the queue and wrapper logs, thinking + reasoning parser intact, Phase-0 request audit green on the cycle's first request.
- **Excluded from the pairing (not spec):** 7 consecutive v4 instances (`django-11905 … 12184`) whose per-instance rollout image failed to build during a ~3-minute npm-registry blip (21:23–21:26; first build failure in 535 instances). They have no scaffold session, are classed `infra_rollout_nonzero_rc` by `audit_predictions.py`, and are re-rolled automatically at lane close (Phase 4). `paired_lane_compare.py` drops infra rows from both arms before pairing.

## Live spec-arm decode telemetry (server log, 12,245 decode batches at 23:13)

| metric | median | mean |
|---|---|---|
| accept length (tokens per verify step, γ=8) | 3.02 | 3.25 |
| gen throughput, tok/s (single user, real agentic prompts 20K–200K) | 88.7 | 93.4 |

The no-spec v3 lane sat at ~48–71 tok/s over the same prompt-depth range (depth sweep: 71 short, 48.4 @ 261,916 actual), so the live lane speedup is in line with the isolated A/B (1.60× short → 1.21× @ 250K). GPUs 95–100 % utilised at the 260 W cap during decode.

## Caveats

- 31 pairs is a wall-rate *sample* (SE on a 2-vs-3 wall count is large); the full-300 v4 cell is the number that lands in the bake-off table. The sample exists to catch a regression early (walls, errors) — it found none.
- Resolved-rate is not compared here: v4 is not scored until the lane closes (scoring is CPU-only and sequenced by `run_model_cycle.sh`). Spec is rejection-sampled, so the output distribution at temp 1.0 is unchanged by construction; any resolved delta at lane close is sampling noise + the wall/timing effect above.
- Per-instance speedups are noisy (agentic trajectories diverge at temp 1.0): 10/31 instances were *slower* under spec (e.g. `astropy-14182` 541 → 1013 s, `django-11133` 543 → 1434 s) — trajectory length, not decode speed.

## Reproduce

```
python3 evals/swebench/paired_lane_compare.py evals/swebench/runs/qwen38-opencode-v3 evals/swebench/runs/qwen38-opencode-v4 \
    --json benchmarks/quality/dspark-cuda-2026-09-27/v4-vs-v3-paired-sample.json
python3 evals/swebench/lane_overhead.py --run evals/swebench/runs/qwen38-opencode-v4
```

Related receipts: [`dspark-cuda-depth-ab-2026-09-27.md`](../dspark-cuda-depth-ab-2026-09-27.md) (isolated A/B), [`dspark-fp8kv-deep.json`](dspark-fp8kv-deep.json) (250K point), [`docker-serving-smoke-qwen38-dspark-v4.json`](docker-serving-smoke-qwen38-dspark-v4.json) (served-config smoke).
