# qwen38 opencode v3 — lane close (2026-09-24), pre-score recall audit

Lane closed 13:37:56 PDT at 300/300 under full isolation (network none, refs stripped,
neutral mounts, `/testbed` re-init to one `eval@local` commit, prompt on stdin, 262144 window,
template-max thinking, 32768 output budget). Leak gate: 0 exposed, four proofs 300/300
(`audit_leakage.py --require-isolation`, after the classifier fix in `02db004`). Scoring happens
at the cycle's Phase 5 (after all six lanes roll); `resolved` is not attached yet — the
official `recall-audit.json` lands at Phase 7b. This receipt is the R9700 relay promised at the
lane close (`audit_benchmark_recall.py --run runs/qwen38-opencode-v3`, JSON beside this file).

## Timing vs the two exposure-confounded v2 cells (same preset, same scaffold)

| cell | walls (rc 124) | empty patches | median s / instance | lane hours |
|---|---|---|---|---|
| v2-out8k (8K output cap, network open) | 10 (3.3 %) | 51 | 654 | 61.9 |
| v2-netopen (32K, network open) | 24 (8.0 %) | 27 | 668 | 68.4 |
| **v3 (32K, isolated)** | **53 (17.7 %)** | 54 (53 walls + 1 `length`) | **1122** | **96.0** |

Walls by repo: sympy 24/77, django 14/114, matplotlib 7/23, sphinx 4/16, xarray 2/5,
sklearn 2/23; astropy / seaborn / flask / requests / pylint / pytest 0. The 53 walled sessions
are unsnapshotted in this cell (the live snapshot-before-kill shipped with `dc67a38`, from the
opencode-dcp lane on) — the recall table below is the 247 finished sessions only.

## Recall table (247 finished sessions)

```
sessions_with_recall: 223/247      sessions naming "SWE-bench" literally: 178/247
sessions_with_dataset_hunt: 6/247  long_thinks: 781, with recall 671 (86 %)
short_turns_with_recall: 1154/10722   reasoning_share_in_long_thinks: 0.507
recall hits | n   | wall | empty | median s | median reasoning chars
0           |  24 |    0 |     0 |      332 |    11892
1-9         |  61 |    0 |     0 |      518 |    38887
10-29       |  54 |    0 |     0 |      836 |   106984
30+         | 108 |    0 |     1 |     1341 |   184627
```

Same monotone shape as R9700's v3/v4 (`1f8ec11`, `f89705e`): duration tracks recall volume 4×
across the buckets. The median first mention sits at 11 % of a session's reasoning text — the
model recognises the setting in the first read of the task, before touching the repo.

## Which residual cue triggers the recognition

Cue census within ±1.5K chars of each session's first literal "SWE-bench" (178 sessions):

| cue | share |
|---|---|
| inferred hidden / grading tests ("the test patch is applied separately", "fail-to-pass") | 95 (53 %) |
| the prompt's own "Do not modify tests" line | 64 (36 %) |
| `/testbed` path or the `testbed` conda env | 18 (10 %) |
| issue-text format ("problem statement") | 16 (9 %) |
| single-commit history | 9 (5 %) |

The prompt never says "hidden tests" — the model derives the grading protocol from
"Do not modify tests" + "your final state will be captured as a `git diff`", i.e. from the task
framing, not from harness paths. That matches R9700's v4 result (neutral cues: recall volume
−41 %, wall rate unchanged): the layout cues (`/testbed`, single commit) account for ~15 % of
first mentions here; the rest is the task contract itself, which cannot be removed without
changing the benchmark. Not a harness defect; no change proposed. The wall-rate gap vs v2 is
the cost of the isolation (v2 answered its recall from future git history / upstream fetches in
56–62 % of instances and finished in half the time) — v3's numbers are the model's own.
