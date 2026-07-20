# 3090-J: Second-request intrusion at 256K: radix-prefix survival, mixed-depth concurrency wall, and a measured ops rule for the production endpoint

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | 2-3 days elapsed: instrument ~0.5 day (no GPU until smoke); ~24-cell matrix + per-preset wall probe ~10-16h GPU in bake-off gaps; optional mitigation A/B ~2-4h |
| **GPU time** | ~10-16h serving occupancy on the eval box: 3 presets x [cold/warm/solo controls + 4 deep depths x 3 intruder sizes x 2 regimes, ~5-10 min/cell] + wall probe + one mitigation A/B cell. Cells are individually short — interleave with bake-off gaps, resume per-cell. |
| **Depends on** | None blocking: shipped launch.sh presets on the harness port 23334 + checkpoints already on box; serve_production.sh (89d0b62) is the motivation, NOT the test bed. Scheduling only: Rule 1/2 gaps. |
| **Provides to** | serve_production.sh + rules-for-agents.md: the first measured multi-request ops rule for the :30000 endpoint; R9700: radix-eviction-at-depth + pool-exhaustion receipts for the shared single-user-256K mission, plus the 059/069 v2-rep-cache bs==1 fallback priced under a real intruder (feeds their #39 CANDIDATE promotion); fleet: first multi-request-at-depth datapoint anywhere — all existing concurrency data is 256-token prompts |

## Objective

Measure what a second request does to a deep session on the 24GB cards, in both regimes the rig can actually run: as-shipped MAX_RUNNING=1 (head-of-line blocking + between-turn radix eviction) and OVERRIDE_ARGS max-running 2 (true overlap contention). Quantify the hidden cost-at-risk — the warm sub-second deep-TTFT receipts vs the never-measured cold 262K re-prefill an evicted prefix forces — find the (depth, intruder) wall per preset, name the pool-exhaustion behavior from server-log evidence, and commit a measured ops rule for the brand-new :30000 production endpoint.

## Hypothesis

On pool-roomy presets (coder-30b, ~900K-token pool) the deep prefix survives any intruder ≤32K and the only production hazard is head-of-line blocking (queued-regime intruder TTFT ≈ remaining deep decode — tens of seconds — while deep TPOT stays ~solo); on pool-tight presets (devstral, 202K pool) an intruder sized so D+I exceeds the pool forces radix eviction of the finished deep prefix, converting the next deep turn's warm TTFT (281.76ms receipt) into a full cold re-prefill — a ≥100x TTFT cliff invisible to every existing receipt. Corollary prediction to falsify cheaply: gemma4-31b's wall arrives at a SMALLER intruder than its 347K full-pool arithmetic predicts, because the 0.05-ratio SWA sub-pool binds first on intruder prefills (SWA-hybrid eviction structure, untested under concurrency anywhere).

## Background & receipts

- The repo's ONLY concurrency surface is short-context: bench_all_unified.py `bench_throughput()` = 256-in/256-out random at conc 1-32 (scripts/bench/bench_all_unified.py:99-119), context sweep capped at --context-max 32768 default (line 127) → benchmarks/all_models_concurrency.png. Nothing under benchmarks/ measures multi-request interaction at depth.
- Every quality/agentic/perf receipt is M=1: CLAUDE.md loop step 4 mandates it; the 256K presets hardcode MAX_RUNNING=1 (launch.sh:797 consumes it; devstral :108, coder-30b :192, gemma4-31b :346).
- Commit 89d0b62 (2026-07-20) exposed serve_production.sh on :30000, `--host 0.0.0.0`, validated on gemma4-31b — any client can now issue overlapping side-requests, and agent harnesses do exactly that (short title/summary calls against the same base URL): the intruder class this experiment prices.
- Warm receipts hide the cost-at-risk: benchmarks/baselines.json deep rows are all sub-second TTFT at 261,916 actual (coder-30b 378.61ms, devstral 281.76ms @201,686, gemma4 850.02ms) — radix-warm numbers (bench_serving's deterministic prompt seed). A COLD 262K prefill has never been measured on the fixed instrument; the cold-minus-warm delta IS what an eviction costs.
- Pool geometry decides the regime (README "Max context" ‡‡ note + serve logs): coder-30b ~900K-token pool (two 262K-capped requests co-fit — wall unreachable, pure queueing regime); gemma4-31b 347K full-attn pool via `--swa-full-tokens-ratio 0.05` (launch.sh:355) + a small SWA sub-pool with its own SWARadixCache eviction structure; devstral 202K pool, ~196K-capped (launch.sh:108) — the one preset where D+I arithmetic guarantees a wall.
- True overlap needs exactly one override: `OVERRIDE_ARGS='--max-running-requests 2'` lands after everything, argparse last-wins (launch.sh:854-860). Preset otherwise AS SHIPPED; must NOT edit presets.
- 059 interaction: patch 059's v2 rep-cache is bs==1-only, keyed on slot-0 request identity — any concurrency drops it to the v1 O(ctx)/step scorer (doc 09 risk list; README:74). README Quick Start recommends the `_ENV_GEMMA_TOPK` EXTRA on the production endpoint (README:169), so the production-recommended config has an unmeasured intruder path.
- Fleet lesson (CLAUDE.md invariants + spec-decode forensics): client-side TPOT under-measures ~2x — every TPOT claim here must cross-check server-log gen-throughput.

## Method

1. Preflight: nvidia-smi + docker ps clean (Rule 1/2); leave :30000 untouched — all measurement on 23334. Per preset, record the serve log's pool lines (max_total_num_tokens; BOTH sub-pool sizes on gemma4-31b) into the receipt BEFORE choosing wall steps.
2. Author scripts/bench/concurrency_at_depth.py (port 23334): builds a deep chat session from a deterministic marker document with a byte-identical message prefix across turns (radix-hit by construction). Per cell it writes benchmarks/concurrency-at-depth/<preset>/d<D>-i<I>-<regime>.json carrying: cold prefill wall-clock (after `curl -s -X POST localhost:23334/flush_cache` — the cost-at-risk), turn-2 warm TTFT (streaming first-token timestamp), solo TPOT over a 512-token decode, per-turn usage.prompt_tokens vs target D (±5%), and the server-log excerpt for the cell window (retract/evict/pool lines, prefill cached-token counts).
3. Instrument gate per preset, before any intruder cell: turn-2 warm TTFT must land in the sub-second warm class of the baselines.json deep rows. A radix MISS here invalidates every survival boolean — fix the instrument, do not proceed.
4. Queued-regime arms (AS SHIPPED, MAX_RUNNING=1): at each D, while the deep session decodes 512 tokens, fire one intruder I ∈ {1024, 8192, 32768} (randomized content per cell, 64-token output): record (a) intruder TTFT = head-of-line block, (b) deep TPOT during vs solo (expect ~equal — the intruder waits), (c) post-intruder deep turn-3 TTFT — >5x warm = prefix evicted (binary), cross-checked against server-log cached-token counts.
5. Overlap-regime arms: relaunch `OVERRIDE_ARGS='--max-running-requests 2' ./scripts/launch.sh <preset> --port 23334`; repeat step 4 — here (b) is the contention number: deep TPOT inflation % during intruder prefill overlap and during 64-token co-decode, server-log cross-checked.
6. Wall probe (per preset, overlap regime, max D): step I up 32K → 64K → 96K → 131K until the pool cannot hold both or the per-request cap binds; NAME the observed behavior from log evidence (queue / retract `#retracted_reqs` / radix evict / abort / crash). coder-30b: expected unreachable within 2x262K caps — record "queueing-only regime" as the finding, do not force it.
7. Matrix: coder-30b (roomy-pool control), gemma4-31b (production bring-up preset, SWA-hybrid eviction), devstral (tightest fit). D ∈ {65536, 131072, 204800, ~250000 true}; devstral ladder {65536, 131072, 163840, ~190000} under its 196K cap.
8. One optional cell: gemma4-31b + `_ENV_GEMMA_TOPK="--decode-topk-pages 256 --decode-topk-page-size 64"` + one 8K intruder at D≈200K — verify the v2→v1 rep-cache fallback neither garbles nor crashes, and price the v1 O(ctx)/step cost (this is the README-recommended production config).
9. If eviction dominates on any preset: A/B ONE mitigation at a single (D,I) — candidates: `--schedule-policy fcfs` vs default lpm, or declared-context headroom (serve CTX below pool so a bounded intruder class always fits). One mechanism, receipts for both arms.
10. Receipt + ops rule: benchmarks/concurrency-at-depth-<date>.md (per-cell table: TPOT-inflation %, intruder TTFT, survival boolean, re-prefill seconds; per-preset wall + named failure mode; warm-vs-cold delta table); commit a one-line measured rule quoting the numbers into serve_production.sh's header comment + rules-for-agents.md (e.g. "endpoint is single-session-safe; side-requests ≤X tok survive at depth Y; a ≥Z-tok intruder costs a W-s re-prefill"); README Tooling note; R9700 README push.

## Baseline & instrument

Solo controls measured by the same instrument in the same serve session: cold 262K prefill wall-clock (post-/flush_cache), warm turn-2 TTFT, solo 512-token TPOT — anchored against baselines.json deep rows (coder-30b 69.6 tok/s / 378.61ms; devstral 42.7 / 281.76ms @201,686). gemma4-31b has no baselines.json row yet (3090-K arms it): anchor on the README perf-table 13 tok/s @255K and the 059 verdict's graphs-on 12.9. Instrument: concurrency_at_depth.py, with server-log gen-throughput cross-checking every TPOT claim.

## Success criteria

- Receipt covers ≥3 presets x ≥3 deep depths x 3 intruder sizes x both regimes; every cell carries TPOT-inflation % (server-log cross-checked), intruder TTFT, prefix-survival boolean, and measured cold re-prefill seconds.
- Pool-exhaustion failure mode NAMED per preset with server-log evidence at the measured (D,I) wall — or an explicit "wall unreachable within per-request caps" verdict (coder-30b expected).
- The warm-vs-cold deep-TTFT delta published per preset — the number the 378ms-class receipts have been hiding.
- One committed ops rule quoting measured numbers in serve_production.sh header + rules-for-agents.md; README Tooling note landed.
- R9700 handed the eviction/wall receipts + the 059-under-concurrency cell verdict.
- All servers stopped, no orphaned sessions; nothing from these M=2 cells written into baselines.json.

## Kill criteria

- Instrument-gate kill: warm turn-2 TTFT not reproducing the sub-second warm class solo (radix miss) and not fixed within half a day — park with notes; every downstream survival boolean depends on it.
- Time-box: instrument + first preset (gemma4-31b) not producing valid cells within 1.5 days → cut the matrix to gemma4-31b only, publish, queue the rest as a follow-up.
- If pool exhaustion CRASHES the server (vs retract/queue): stop stepping immediately — the crash IS the finding; capture the log, do not retry-loop a crashing config unattended; flag the ops rule "endpoint must front-gate intruders" pending a fix.
- MAX_RUNNING=2 boot failure on any preset (graph/pool sizing): record it, keep that preset's queued-regime arms — the as-shipped regime is the production truth anyway.
- External: bake-off resume or kernel-BUG reboot mid-matrix — per-cell receipts persist; resume at the next cell, never re-baseline a half-measured arm.

## Deliverables

- scripts/bench/concurrency_at_depth.py (deterministic prefix builder, cold/warm/solo controls, intruder + wall modes, server-log excerpt capture)
- benchmarks/concurrency-at-depth/<preset>/d<D>-i<I>-<regime>.json per cell + benchmarks/concurrency-at-depth-<date>.md rollup (matrix table, per-preset wall + named failure mode, warm-vs-cold delta table)
- Ops rule: one measured line in scripts/serve_production.sh header + rules-for-agents.md; README Tooling note
- Mitigation A/B receipts (both arms) under benchmarks/concurrency-at-depth/ if step 9 runs
- Cross-repo: R9700 README note — radix-eviction-at-depth + wall receipts + the 059 v2-fallback-under-intruder verdict (feeds their #39 promotion decision)

## Constraints

- Rule 1 / Rule 2 (CLAUDE.md): no concurrent calibration/rollout/scoring; this occupies the eval box — the bake-off queue holds priority, slot into gaps.
- Measurement ONLY on harness port 23334, presets AS SHIPPED via launch.sh; the ONLY serve-config change allowed is `OVERRIDE_ARGS='--max-running-requests 2'` (launch.sh:860, argparse last-wins) — must NOT edit presets, must NOT use or perturb the :30000 production endpoint during measurement.
- Must NOT write any M=2 cell into benchmarks/baselines.json — the tripwire is M=1-only; receipts live under benchmarks/concurrency-at-depth/.
- Every TPOT claim server-log cross-checked (client TPOT under-measures ~2x); flush-isolate cells via POST /flush_cache only when the server is idle; setsid-detach any >30min batch of cells.
- 260W cap + gpu-fan-curve.service stay active. Read-only courtesy to the R9700 tree (README push only).

## Risks

- Streaming-client TTFT jitter could blur the survival boolean — mitigated: the signal is >5x (hundreds of ms vs tens of seconds) and every boolean is cross-checked against server-log cached-token counts.
- The SWA sub-pool may bind on gemma4-31b intruder prefills long before full-pool arithmetic predicts — treat as a first-class outcome (it is the corollary), not an instrument error; step 1 records both pool sizes so wall steps are chosen against the real binding pool.
- Deterministic long prompts tokenize differently per family and can miss target depths — verify usage.prompt_tokens per turn against D (±5%) and record actuals (the depth-label lesson from the 594c059 forensics).
- The intruder could radix-hit its own earlier cells and under-price eviction pressure — randomize intruder content per cell + flush isolation between cells.
- 512-token deep decodes at gemma4-31b's ~13 tok/s make each cell ~40-60s of decode plus prefills — budget cells honestly; bound every request with an 1800s-class timeout (bench_long_context.py:46 pattern).
- /flush_cache misbehaves with requests in flight — strictly sequence: poll idle, flush, then start the cell.

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
