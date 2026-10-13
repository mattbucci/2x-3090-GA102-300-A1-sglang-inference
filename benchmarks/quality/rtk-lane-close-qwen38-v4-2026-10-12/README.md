# qwen38 little-coder + RTK v4 — lane close (2026-10-12, 300/300, rc 0 at 23:41 PDT)

DSpark spec on, 256K window, 32K output budget, network=none, docker-served. Scaffold = little-coder
1.19.0 = `@earendil-works/pi-coding-agent` 0.83.0 from `/opt/lc-rtk` **with** the rtk extension
(`-e`, rtk 0.46.0, `/opt/rtk-home`). Control = `qwen38-little-coder-v4` (same prefix without `-e`,
re-pinned 2026-10-09, closed 2026-10-11 — receipt one directory over). The lane rolled in two stretches:
rows 0–175 on 2026-10-07 → 10-09 (before capture-at-wall `b66d27f` and the scaffold-pin rows landed —
the running process kept the old code in memory), then paused for the control's re-roll and resumed at
row 177 on 2026-10-11 20:59 PDT under the current code. Scored verdict comes from Phase 7; everything
here is pre-score rollout shape.

Rows: **255 rc-0 (254 patches, 1 empty), 24 walls (rc 124), 21 rc −1 infra.** Walls split 14 (rows 0–175,
all empty — pre-capture, Phase 4 re-rolls them as `infra_wall_capture_defect`) + 10 (rows 176–299, 9
diff-bearing via capture-at-wall, 1 empty). Pins on the 104 rows the current code wrote:
`little-coder@1.19.0/@earendil-works/pi-coding-agent@0.83.0+rtk@0.46.0`. rc-0 `rollout_seconds` median
693.0 s, mean 787.9 s, sum 55.81 h. Wall rate is the same in both stretches (14/176 = 8.0 %, 10/124 = 8.1 %).

**Infra rows — a second upstream-dependency outage.** Row 3 (astropy-14995) is a leftover from the
2026-10-09 prime-agent pruning. Rows 236–255 (sympy-13471 … sympy-16106, twenty in a row) crashed in
`docker build` between 18:45:16 Z and 18:50:28 Z on 2026-10-12: the per-instance image's
`npm install -g opencode-ai@1.14.25 little-coder@1.1.0` step failed with `ETARGET No matching version
found for @aws-sdk/credential-provider-node@^3.972.86` — a registry-side transient (a transitive dep
published against a not-yet-visible version; row 257 built fine five minutes later). Each failed build
costs ~6 s and each crash row ~15 s, so a five-minute registry blip eats twenty rows. Phase 4 re-rolls
them (`infra_rollout_nonzero_rc`); the prime lane that followed built its first image without incident.

## Leak gate — PASS on exposure (`leak-little-coder-rtk-v4.json`, classifier `071c642`)

`audit_leakage.py --require-isolation`: n=300, transcripts 279, attempted 229 (82 % — web_UPSTREAM 217,
web_SEARCH 12, web_OTHER 2, git_LIST 49; all dead under network=none), **EXPOSED 0**, isolation proof
279/279 on all four lines (the 21 transcript-less rows are the infra rows; the gate re-runs after the
re-roll). Gold overlap on the isolated set: n=261, median 0.057, ≥80 % on 52.

Loopback channel (the bridge self-dispatch class found at the control's close):

| instance | loopback | calls | what |
|---|---|---|---|
| pytest-5103 | SELF | 9 | the known case — `/v1/chat/completions` self-calls asking the served model for the upstream PR (got a hallucinated "PR #11923") |
| sympy-16281 | SELF | 2 | new: asked the served model which PR fixed "Product pretty print could be improved" and for the exact `_print_Product` width formula ("otherwise say I DONT KNOW"); first call unauthenticated → 401, second with a guessed bearer → 401. **0 successful calls.** |
| matplotlib-22711 | BRIDGE | 1 | `socket.create_connection` probe of 127.0.0.1:23334 while enumerating listeners |
| django-15695, django-16229 | LOCAL | 1 each | the model's own local servers |

Both SELF instances are recall attempts through the self-oracle; the cycle-boundary fix (bridge off `ps`
argv + a self-dispatch policy) is unchanged from the control's receipt. Also 4 `dispatch` sub-agent calls
on 4 instances (sub-agent sessions are not persisted — same caveat as the control) and 20 `BrowserNavigate`
calls on 20 instances (all fail at launch; no chromium in the rollout image).

## A/B vs the pi-0.83 control (`ab-receipt-little-coder-rtk-v4-vs-pi083.md`, 250 same-ID instances)

| | lane (rtk) | control |
|---|---|---|
| patched (non-empty diff) | 225 | 229 |
| empty diff | 1 | 0 |
| timeout (rc 124) | 24 | 21 |
| median wall (s) | 781.4 | 772.0 |
| p90 wall (s) | 1920.5 | 1891.0 |

Agreement: both patched 214, control-only 15, lane-only 11, neither 10. Paired summary
(`paired-little-coder-rtk-v4-vs-little-coder-v4-pi083.json`): walls 24 vs 21, median 781 vs 772 s, sum
63.56 vs 63.7 h, per-instance speedup (median) **1.00×**; flips wall→done 11, done→wall 14. (The paired
"empty" column — 16 vs 3 — is the capture-at-wall confound: 14 of the lane's 16 are pre-capture walls.)
Walls shared with the control: 10 of 24 (django-11564 16910, seaborn-2848, sphinx-8474, xarray-3364,
sympy-12236 13146 17630 18698 19254); 3 control walls sit on this lane's infra rows (sympy-13773 13895
13915), 14 lane-only, 11 control-only.

**rtk engagement: 249/250 = 100 %** (≥1 executed rewrite; median 17 rewrites per session, p90 34).
Rewritten families: grep 2,530 · ls 1,237 · cat 563 · find 178 · git 165 · wc 17 · du 4 · pytest 4 ·
ps 3. Tokens rtk reports saved: median 2,588 per session, 3,167,950 total (9 ledger parse failures).
Against sessions whose prompts peak around 100K tokens and re-prefill 30–40 times, that is a rounding
error — which matches the shape: same walls, same median, same empties, 0 phantom calls. The plugin ran
everywhere and changed nothing measurable about the rollout; whether it changed resolve rate is the scored
cell's call.

Against opencode-v4 (`paired-little-coder-rtk-v4-vs-opencode-v4.json`, 271 paired): walls 22 vs 32, empty
14 vs 33, median 740.9 vs 800.4 s, speedup 1.03×; flips wall→done 21, done→wall 11, empty→patch 27,
patch→empty 8 — the same ~30 % fewer walls than opencode the control showed.

## Where the time goes (`lane_overhead.py --run`, rc-0 rows under the wall)

| lane | rollout med | pre (build+boot→first row) | session | post (last row→exit) | cleanup pass |
|---|---|---|---|---|---|
| little-coder-rtk-v4 | 673 s | 184 s | 482 s | 0 s | 257/279 two sessions (22 walls one) |
| little-coder-v4 (pi 0.83 control) | 664 s | 180 s | 502 s | 0 s | 240/240 |

Walls silent ≥100 s before the kill: 5/24; with a finished step: 1/24.

Context shape (`prompt_sawtooth.py --cleanup-drop`, `sawtooth-little-coder-rtk-v4.json`): 94/278 sessions
compacted (34 % — control 41 %, pi 0.68 69 %, opencode-v4 79 %), 179 compaction events, median peak prompt
107,229 (compacted) / 90,559 (never), max 238,612, 262 sessions ever above 32K. Compacted: 70 patched /
24 walled, median 1398 s; never-compacted: 183 patched / 1 empty / 0 walled, median 566 s. Every wall
compacted (median 3).

## Walls — 24 rc 124 (`walls-little-coder-rtk-v4.json`, `audit_wall_causes.py`)

| class | n | instances |
|---|---|---|
| step_boundary | 18 | django-11564 (in cleanup) 13265 13315 16910, matplotlib-23299 23563 25079, seaborn-2848, xarray-3364, pytest-7490 8906, sphinx-8474, sympy-11870 12236 13146 18698 19007 (in cleanup) 19254 |
| tool_stuck (open past its own ceiling) | 3 | django-11742 (1249 s / 30 s), django-13551 (443 / 120), requests-863 (1739 / 120 — `/tmp/mock_httpbin.py` server left running) |
| tool_running | 3 | django-11905 (`time.sleep(240)` poll), django-12589 (runtests under 600 s), sympy-17630 (polys suite under 900 s) |

Same shape as the control: long sessions that did not finish, not test suites hung at the kill. Two walls
landed during the cleanup pass (main session complete, cleanup in flight) — the capture-at-wall path took
their diffs.

## The one rc-0 empty — sympy-21614 (thinking overflow, benchmark recall)

Zero `edit`/`write` calls. The main session's last three responses were pure thinking cut at the 32K output
budget (`stopReason: length`), pi nudged "Your previous response was empty" after each, and the session
ended on the third. The truncated thinks reason about "what the upstream fix was and what the test patch
checks … This is a SWE-ben[ch]" — the in-session recall pattern R9700's audit measures, here costing the
instance its answer. 721 s, peak prompt 137K, no compaction. The cleanup pass then reported "your fix is
already in the committed state" over an untouched tree.

## Cycle-boundary items this cell adds (never mid-cycle)

1. **Retry `docker build` before writing a crash row** — three attempts over ~10 min with backoff; a
   registry blip should cost minutes, not twenty rows. The multi-stage `Dockerfile.rollout` (scaffold
   layer built once per cycle) removes the exposure entirely and is already queued.
2. Unchanged from the control's receipt: bridge off `ps` argv + self-dispatch policy; `loopback_SELF` into
   the recall audit; patch 065 → tree + image; 063 + 065 upstream; prime-agent 0.10.0 re-pin; DCP cleanup
   pass; opencode permission block.
