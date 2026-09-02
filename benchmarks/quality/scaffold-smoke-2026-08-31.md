# Scaffold-roster smoke — 2026-08-31 (qwen38, 2 instances/lane, final rollout image)

| lane | rc | predictions | non-empty diffs | engagement proof / note |
|------|----|-------------|-----------------|--------------------------|
| opencode-dcp | 0 | 2 | **2/2** | patches only — the "dcp tool_use events" first written here did not reproduce from the on-disk logs (DCP prunes inside the messages.transform hook, registers a `compress` tool, emits nothing to stdout). Plugin load verified in a live lane-2 container 2026-09-02 (`loading plugin @tarquinen/opencode-dcp@3.1.15`, `compress` permission in ruleset); per-instance prune receipts via `evals/swebench/dcp_engagement_poller.sh` → `benchmarks/quality/dcp-engagement/` |
| little-coder-rtk | 0 | 2 | 1/2 | rtk-shim log: `rtk rewrite git status` → executed `rtk git status`; empty = full 201 s session, model declined to edit (12907 solvable per other lanes) |
| prime | 0 | 2 | 1/2 | real 504 B patch @459 s; empty = rc=124 at the smoke's 900 s cap (production runs 1800 s) |
| dcode | 0 | 2 | **2/2** | inner `--timeout` now derived from the outer (outer−100 s) |

Mechanism receipts (hard-won, in CLAUDE.md too): headless pi runs skip extension
auto-discovery → rtk loads via explicit `-e`; the pi session jsonl records the
PRE-mutation command (grep-for-rtk falsely reads unengaged — use an invocation
shim); rtk needs the @earendil-works pi → the lane runs its own little-coder
1.19.0 prefix while the control lane stays at 1.1.0; prime-agent needs Node
≥22.8 (image now node 22.23.2, which the earlier 0-diff prime rounds lacked);
per-instance rollout images are cached by tag — every Dockerfile edit needs the
swebench-rollout nuke (bit us mid-smoke exactly as the landmine note predicts;
the smoke driver now forces --rebuild-image).
