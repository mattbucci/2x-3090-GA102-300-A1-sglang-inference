# DCP A/B on the isolated qwen38 v4 cells — opencode (control) vs opencode-dcp

Both cells full-300, `swebench==4.1.0`, served from the same image (v0.5.20 → v0.5.22 at the
2026-10-20 resume, DSpark on, true 256K), network `none`, leak gate PASS on both. Scored
2026-10-22 (`paired-resolve.json`, `paired-rollout.json`; `evals/swebench/paired_lane_compare.py`).

| | control (opencode) | opencode-dcp | Δ |
|---|---|---|---|
| resolved / 300 | **196 = 65.3 %** | **203 = 67.7 %** | +7 |
| walls (rc 124) | 13 | 14 | +1 |
| empty patches | 6 | 10 | +4 |
| scorer errors | 0 | 1 (`psf__requests-863`) | +1 |
| median rollout s | 804 | 923 | +15 % |
| both resolved / only control / only dcp / neither | 178 / 18 / 25 / 79 | | |

**Verdict: no detectable quality effect from the plugin.** The +7 is 18 vs 25 discordant pairs,
McNemar exact two-sided p = 0.36 — inside the per-draw noise of one attempt per instance (the
control's own re-drawn rows moved 21/41 on identical inputs). Per-repo deltas are all within ±2
except sympy (+6 of 77), which is where the longest sessions live. Flips in the rollout read are
symmetric (wall→done 7, done→wall 8).

**Engagement** (`benchmarks/quality/dcp-engagement/`, read-only in-container snapshots of the plugin's
state file, which only exists once DCP has pruned at least once): 264/300 instances wrote a state
file, **110 with `totalPruneTokens > 0`** (median 36.7K pruned, p25–p75 29.5–48.0K, max 88K), 154 at
0 (the plugin loaded and tracked but never fired), 36 no file (short sessions). Resolve rate by
bucket — pruned 57 % (63/110), tracked-but-0 69 % (106/154), no file 94 % (34/36) — tracks session
length, i.e. instance difficulty, not the plugin; it is NOT a causal read.

**Cost:** the lane paid the opencode startup `npm install @opencode-ai/plugin` boot tax (~70 s, root
cause in `667e3da`, fixed on the image the 37 re-drawn rows ran on) plus the full 120 s cleanup
ceiling on most instances (root-caused with it) — the 923 vs 804 s median is that tax, not model
time; the rollout-seconds columns are therefore mixed-image and only the resolve columns are a clean
A/B.

**Error row.** `psf__requests-863` (dcp) rolled 2026-10-02, six days before `b66d27f` taught the
`/testbed` re-init to exclude the image's untracked `build/`; its 874 KB diff sweeps
`build/lib/requests/*` and the scorer's `patch` refuses it ("already exists! Assuming -R"). The
control's same instance resolved. A harness-defect row, counted as unresolved in the cell number;
`audit_predictions.py` has no class for a *present* patch made of image residue, so it was not
re-rolled at Phase 4 — add `infra_patch_image_residue` (diff paths under `build/` on a pre-exclude row)
before the next re-roll pass.
