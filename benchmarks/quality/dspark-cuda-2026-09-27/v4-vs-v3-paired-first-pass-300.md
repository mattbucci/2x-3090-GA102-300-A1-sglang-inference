# qwen38 SWE-bench Lite opencode lane — DSpark spec (v4) vs no-spec (v3), full first pass (300/300)

**Date:** 2026-09-30 21:36 PDT (lane close) · follows the [31-pair sample](v4-vs-v3-paired-sample.md) (2026-09-27) and the 117-pair rolling read (2026-09-28). JSON: [`v4-vs-v3-paired-first-pass-300.json`](v4-vs-v3-paired-first-pass-300.json). Same arms, same contract as the sample; the only difference between arms is the served config (v4: `speculative_algorithm=DSPARK`, γ=8, bf16 draft, fp8_e4m3 KV, 307,659-token pool, `context_length=262144`).

**First pass** = before the Phase-4 infra re-roll: 8 v4 rows (`django-11905 … 12184`, `django-13315`) have no scaffold session because their per-instance rollout image failed to build (npm-registry blips; classed `infra_rollout_nonzero_rc`, rc −1) and are excluded from the pairing in both arms. The scored, re-rolled cell is the number that lands in the bake-off table.

## Result (292 instance-matched pairs; wall = rc 124)

```
                   walls  empty  median s   mean s   sum h
ref  (v3 no-spec)    52     53    1108.0   1147.4   93.07
new  (v4 DSpark)     33     33     801.1    959.8   77.85
per-instance speedup (ref/new, median): 1.18×
flips: wall→done 36, done→wall 17, empty→patch 37, patch→empty 17
```

- **Walls 33 vs 52 (11.3 % vs 17.8 %)** — a 37 % reduction in wall-rate on the same 292 tasks. 16 instances wall in both arms (behavioural); the 36/17 flip asymmetry is the spec effect plus run-to-run sampling noise at temp 1.0.
- **Empties 33 vs 53.** v4: 31 wall empties + 2 rc-0 `finish_reason=length` empties (`django-11019`, `sympy-13895`: the final thinking turn hit the 32,768 output budget; `audit_predictions` class `model_output_budget`); v3 had 1 such row (`sympy-19254`). Two v4 walls (`pytest-8365`, `sympy-13915`) still carry a captured diff.
- **Time:** median −28 % (1108 → 801 s), mean −16 %, summed −15.2 h over 292 instances. `lane_overhead.py`: pre-session (image build + boot → first event) 198 → 193 s (unchanged), session median 735 → 537 s (−27 %), post 0 s both, cleanup session present 235/235 vs 250/250 — the gain is entirely in the model-bound phase.
- **Wall classes (v4, `audit_wall_causes.py --walls-only`, from the session snapshots):** generating 15 · tool_running 14 · permission_hang 2 · step_boundary 1 · env_destroyed 1. The 15 generating + 14 tool_running (model still running a test suite inside the command's budget) are model walls; the 2 permission hangs are the opencode `task`-subagent `external_directory` prompt (cycle-boundary fix); `env_destroyed` = `sphinx-8474` removed its own testbed environment mid-session. v3 walls are blind (no snapshots in that cell).
- **Errors:** 0 server-side errors over the lane (no TRIPWIRE / BRIDGE CHECK / Traceback lines); Phase-0 request audit green; thinking + reasoning parser intact.
- **Leak gate:** `audit_leakage.py --require-isolation` → attempted 153/292 (52 %), **EXPOSED 0**, isolation proofs 292/300 on all four checks (the 8 infra rows have no transcript yet). Two classifier false positives were found and fixed at this close (`1679dd9`): a `gh pr view` whose binary is absent from the sandbox (`gh: command not found`) and a `curl -o` whose target was listed at 0 bytes after `exit: 6`.
- **Prompt sawtooth (`prompt_sawtooth.py`, 261 sessions matched in the server log):** 417 compaction events in 207 sessions (79 %); max prompt 227,652 tokens (inside the 262,144 window); never-compacted sessions 54 → 54/54 patched, median 424 s; all 31 matched walls compacted (median 3×). Compaction is a consequence of long sessions, not a loop — no session compacts without the prompt first growing past the threshold.

## Spec-arm decode telemetry (server log, 123,917 decode batches over the whole lane)

| metric | median | mean |
|---|---|---|
| accept length (tokens per verify step, γ=8) | 3.02 | 3.22 |
| gen throughput, tok/s (single user, real agentic prompts 20K–228K) | 88.0 | 91.9 |

Steady from the first hour (3.0 / 88) to the last; no drift at depth. The no-spec v3 lane sat at ~48–71 tok/s over the same prompt range.

## Verdict

Spec does not wall more, did not error, and shortens the model-bound phase by ~27 % on the full lane: `SPEC_FOR=qwen38:1` stays on for the rest of the v4 cycle (opencode-dcp → little-coder → little-coder-rtk → prime → dcode). Resolved-rate is compared only after Phase 7 scoring (rejection-sampled spec leaves the output distribution unchanged; any delta is sampling noise plus the wall effect above).
