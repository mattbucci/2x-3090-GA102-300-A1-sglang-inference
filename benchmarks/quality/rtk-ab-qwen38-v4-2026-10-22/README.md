# qwen38 · little-coder vs little-coder + RTK — scored A/B (v4, 2026-10-22)

Both cells 300/300 under the isolated v4 harness (DSpark spec on, 256K window, 32K output budget,
`network=none`, docker-served SGLang v0.5.22). Scaffold = little-coder 1.19.0 = `@earendil-works/pi-coding-agent`
0.83.0 from `/opt/lc-rtk`; the treatment adds the rtk extension (`-e`, rtk 0.46.0, `/opt/rtk-home`), the control
is the same prefix without `-e`. Scorer `swebench==4.1.0`. Pre-score rollout receipt for the lane:
[`rtk-lane-close-qwen38-v4-2026-10-12`](../rtk-lane-close-qwen38-v4-2026-10-12/README.md).
Tools: `evals/swebench/paired_resolve.py` (`paired-resolve.json`), `paired_lane_compare.py` (`paired-rollout.json`).

| | control (`qwen38-little-coder-v4`) | rtk (`qwen38-little-coder-rtk-v4`) |
|---|---:|---:|
| resolved / 300 | **193 (64.3 %)** | **202 (67.3 %)** |
| completed · empty · scorer error | 295 · 5 · 0 | 296 · 4 · 0 |
| walls (rc 124) | 25 (4 resolved at the wall) | 15 (4 resolved at the wall) |
| median rollout, rc-0 on both lanes (n=270) | 662.6 s | 653.2 s |

**Paired (300 same instances): both 180 · control-only 13 · rtk-only 22 · neither 85 — McNemar exact p = 0.18.**
Per-repo deltas (rtk − control): django +3 (79→82 of 114), sympy +5 (44→49 of 77), requests +2, astropy / xarray /
pytest +1, matplotlib / flask / pylint 0, scikit-learn / sphinx −1, seaborn −2 (3→1 of 4). Every repo sits inside
its binomial noise band; the +9 headline is 13 vs 22 discordant pairs on a coin with p = 0.18.

**Verdict: no detectable resolve effect from RTK** — the same reading as the pre-score receipt (walls, medians,
empties unchanged on the rollout side) and as the DCP A/B one lane over
([`dcp-ab-qwen38-v4-2026-10-22`](../dcp-ab-qwen38-v4-2026-10-22/README.md), p = 0.36).

## Where the +9 comes from — the wall flips, not the sessions that finished on both

- 15 instances walled on the control and finished under rtk: **7 resolved** (2 of them the control had already
  resolved at its wall). 5 finished on the control and walled under rtk: control resolved 2, rtk 2 at its wall.
  Net +5 of the +9 lives in the wall flips; among the 270 instances that finished on both lanes the discordant
  split is 10 vs 14 (p ≈ 0.5).
- Wall rate 25 vs 15 looks like a signal but is the mixed-code confound the lane receipt documents: 161 of the rtk
  rows were drawn 2026-10-07→10-09 by the old in-memory code (no `scaffold_pin`), their 14 pre-capture walls were
  re-drawn at Phase 4 — a second attempt for exactly the instances most likely to wall — while the control's 300 rows
  are single draws under the re-pinned code. The honest wall comparison is the lane receipt's 24 vs 21 on first draws.

## rtk engagement — ran everywhere, saved almost nothing that matters

`benchmarks/quality/rtk-engagement/<iid>.json` (shim ledger per session): **299/300 sessions executed ≥1 rewrite**,
median 17 commands rewritten, median 2,329 tokens saved per session, 3.78 M total. Against sessions whose prompts
peak around 100 K tokens and re-prefill 30–40 times, that is < 1 % of the context stream.

| tokens-saved tercile | saved range | resolved |
|---|---|---:|
| low | 0 – 1,278 | 77 / 100 |
| mid | 1,287 – 5,110 | 62 / 100 |
| high | 5,190 – 208,552 | 63 / 100 |

Resolve rate *falls* as rtk saves more — the sessions where rtk had the most to compress are the long exploratory
ones that were going to be hard anyway. Difficulty drives both columns; nothing here is causal.

## Reading

Two plugins, two lanes, two nulls on 300 paired instances each: a 30B-class model at a true 256K window with its
full thinking budget is not context-starved on SWE-bench Lite, so tools that trim tool output (RTK) or prune turns
(DCP) have nothing to buy back. The failures are "incomplete fix" (CLAUDE.md), not "ran out of room". Keep both lanes
in the matrix as the harness A/B they are, but stop expecting a resolve delta from them on this model class.
