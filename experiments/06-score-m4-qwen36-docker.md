# 3090-B: Docker-score M4's exported qwen36 SWE-bench predictions on the 3090 harness

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | 2-6h typical (first-run env-image builds dominate; hard ceiling 21 x 1800s ≈ 10.5h test time), plus ~30min import/receipt/flow-back work |
| **GPU time** | none — Docker CPU scoring only; no GPU occupancy on any rig |
| **Depends on** | M4 export file at m4-sglang-inference commit a5a52d8 or later (already committed/pushed — verified in the local M4 checkout).; 3090 Docker daemon + swebench harness operational (installed by scripts/setup.sh line 142; exercised by every bake-off cell — no new setup expected).; No live bake-off cycle occupying the rollout/score lock (Rule 2 sequencing). |
| **Provides to** | M4 repo (dormant rig): official full-26 Docker pass-rate replacing the 5/13 local headline in README.md + exports/README.md, per the handoff its exports/README defines.; Fleet-audit queues: closes 3090 README queue item 2 and M4 README queue item 4 (same finding, both sides).; M4 recovery path context: per-instance PASS/FAIL map informs the M4 in-house-quant queue item (which instance classes fail on the current mlx-community checkpoint). |

## Objective

Close the fleet's only unscored canonical-eval cell: M4's qwen36 SWE-bench Lite predictions (26 instances, 21 non-empty patches, exported 2026-05-18, unscored ~9 weeks) get official Docker-harness pass/fail on the 3090 rig — the only box that can build the 8 old-Python instance envs M4 cannot. This converts M4's 5/13-local-subset headline into a full-26 official rate and settles whether the fleet's M4 recommendation (qwen36 + opencode) holds under real tests.


## Background & receipts

- Export exists and is git-tracked in M4 repo: /home/letsrtfm/AI/m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl — verified 26 JSONL lines, 21 non-empty model_patch, model_name_or_path=sglang/qwen36; empty instances: django__django-11019, pallets__flask-4045, psf__requests-2148, sphinx-doc__sphinx-10451, sympy__sympy-11870 (M4 commit a5a52d8 'qwen36 N=26 widening').
- STALE-DOC WARNING: M4 exports/README.md still says '15 of 16 unique instances' and its 'Score on the 3090' command passes '--output evals/swebench/leaderboards/qwen36-m4.json' — score_docker.py has NO --output flag (verified full argparse list: --predictions/--dataset/--split/--max-workers/--timeout/--rewrite-reports/--cache-level/--run-id/--filter-helpers); that command would exit 2. This spec's command is the corrected one; fixing the M4 README is a deliverable.
- score_docker.py (3090 repo, evals/swebench/score_docker.py) auto-derives run_id from the predictions file's PARENT DIR name and writes scores-docker/ + scores-docker-summary.json next to it — so the import must live in a dedicated dir (m4-imports/qwen36-m4/predictions.jsonl -> run_id=qwen36-m4), not loose in m4-imports/.
- 3090 .gitignore ignores evals/swebench/runs/, logs/, and root-pattern sglang__*.json (the harness's CWD report copy) but NOT evals/swebench/m4-imports/ — the summary + the archived scores-docker/qwen36-m4.sglang__qwen36.report.json are committable receipts.
- swebench harness is installed by 3090 scripts/setup.sh line 142 (pip install swebench, with a comment naming score_docker.py); scripts/common.sh provides activate_conda.
- Rule 2 exact wording (3090 CLAUDE.md line 95): 'no concurrent rollout + score. Both spin per-instance docker containers. Concurrent VFS pressure triggers the kernel BUG.' score_docker.py defaults --max-workers 1 for the same reason (docstring cites the 2026-05-10 vfs_getattr_nosec kernel BUG). rescore_qwen36_ream.sh shows the established serialization pattern: mkdir -p /tmp/loop-bakeoff-logs + flock -x /tmp/loop-bakeoff-logs/score.lock (the 2026-06-15 incident was that dir vanishing mid-run).
- Not already done: no commit in 3090 git history (--all --grep m4) scores this export; evals/swebench/m4-imports/ does not exist in the checkout; the fleet-audit queue item is open at 3090 HEAD 7d3170f README line 10 with matching numbers (26 instances, 21 non-empty, ~9 weeks).
- Baseline numbers to beat/compare (M4 README lines 30, 71): patch-engagement 21/26 = 80.8%; M4-local resolved 5/13 = 38.5% on the score_local.py-buildable subset (receipts: M4 evals/swebench/runs/qwen36-score-local-2026-05-18/ and qwen36-widening-N5-2026-05-18/); 8 instances were INSTALL FAIL on M4 (old-Python venvs) and are scorable only via Docker. M4 exports/README sets expectation: 50-80% real pass-rate band, >50% confirms the qwen36 recommendation.
- M4 side of the interface (m4-sglang-inference/evals/swebench/exports/README.md) documents the intended handoff diagram ending in 'leaderboard back to M4' — the flow-back deliverables below implement exactly that, replacing the nonexistent leaderboards/ mechanism with the real scores-docker-summary.json.


## Method

1. Preflight on the 3090 box: `sudo docker ps` shows no swebench/rollout containers and no run_model_cycle.sh / run_all_cycles.sh processes (Rule 2). `mkdir -p /tmp/loop-bakeoff-logs` (the flock dir — its mid-run deletion caused the 2026-06-15 silent-score failure).
2. Import: `cd ~/AI/2x-3090-GA102-300-A1-sglang-inference && mkdir -p evals/swebench/m4-imports/qwen36-m4`. Preferred source: `git -C ~/AI/m4-sglang-inference pull` then `cp ~/AI/m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl evals/swebench/m4-imports/qwen36-m4/predictions.jsonl` (file is committed at M4 a5a52d8); fallback if no local M4 checkout: scp per M4 exports/README.md. The dedicated subdir is REQUIRED: score_docker.py derives run_id from the predictions file's parent dir name.
3. Verify the import: `wc -l` = 26 and `python3 -c "import json,sys; ls=[json.loads(l) for l in open('evals/swebench/m4-imports/qwen36-m4/predictions.jsonl')]; print(len(ls), sum(1 for d in ls if d['model_patch'].strip()))"` prints `26 21`. Mismatch -> stop, re-pull.
4. Launch scoring detached, serialized on the bake-off score lock: `source scripts/common.sh && activate_conda && setsid flock -x /tmp/loop-bakeoff-logs/score.lock python -u evals/swebench/score_docker.py --predictions evals/swebench/m4-imports/qwen36-m4/predictions.jsonl --max-workers 1 --timeout 1800 > logs/qwen36-m4-score.log 2>&1 &` — do NOT use the `--output` flag shown in M4's exports/README (it does not exist; argparse exits 2). Defaults are correct: dataset SWE-bench/SWE-bench_Lite, split test, cache-level env.
5. On completion, read evals/swebench/m4-imports/qwen36-m4/scores-docker-summary.json: expect total_predictions=26, empty_patch=5, per_instance covering all 26 ids, resolve_rate_pct computed over 26. If missing, the harness rc-0-but-stale trap applies — check logs/qwen36-m4-score.log and the CWD report sglang__qwen36.qwen36-m4.json before rerunning (a `--rewrite-reports` pass re-aggregates without rerunning tests).
6. Cross-check per_instance against M4's local cell (m4-sglang-inference/evals/swebench/runs/qwen36-score-local-2026-05-18/ and qwen36-widening-N5-2026-05-18/): the 5 M4-resolved instances should re-resolve under Docker; any flip is a finding to record verbatim in the receipt commit, not an error to fix.
7. Commit receipts in the 3090 repo: predictions.jsonl copy + scores-docker-summary.json + the archived scores-docker/qwen36-m4.*.report.json (committable; only the CWD sglang__*.json copy is gitignored), plus the README queue-item tick with the headline rate.
8. Flow back to M4: in the M4 checkout, update exports/README.md (21/26 count fix, corrected command, Results section citing the 3090 receipt path), drop in exports/qwen36-docker-summary.json, tick M4 README fleet-audit item 4, and update the qwen36 headline row from '5/13 M4-scorable' to the official full-26 Docker rate. Commit+push to the M4 repo; if the 3090 box has no M4 checkout or push access, emit the edits as a diff for the orchestrator and note the handoff in the 3090 cross-team README section.


## Baseline & instrument

M4-local score_local.py cell: resolved 5/13 = 38.5% on the M4-buildable subset + patch-engagement 21/26 = 80.8% (receipts in m4-sglang-inference/evals/swebench/runs/qwen36-score-local-2026-05-18/ and qwen36-widening-N5-2026-05-18/); this task produces the first full-26 official-harness number via score_docker.py's scores-docker-summary.json.


## Success criteria

- scores-docker-summary.json exists under evals/swebench/m4-imports/qwen36-m4/ with total_predictions=26, empty_patch=5, and a per_instance map covering all 26 instance ids (the fleet's instrument for Docker cells).
- A resolved/26 rate is recorded — any value counts as success; the M4 exports/README expectation band (50-80% strong, >50% confirms qwen36) is interpretation context, not a gate.
- Receipts committed in the 3090 repo (summary + archived report + predictions copy) and both README fleet-audit queue items ticked with the rate.
- M4-side flow-back landed: exports/README.md corrected (21/26, no --output flag) with a Results section, exports/qwen36-docker-summary.json present, README headline updated — or the equivalent diff handed to the orchestrator with the handoff noted.


## Kill criteria

- Harness cannot build/pull env images for more than 1/3 of the 21 non-empty instances after one retry — stop, record the error_ids list from the report as a finding (these are exactly the old-Python envs M4 couldn't build; if the 3090 also can't, that's the null result).
- Kernel-BUG reboot mid-run: resume once (--cache-level env makes the rerun cheap; --rewrite-reports re-aggregates completed instances); a second crash in the same run -> stop, commit the partial per_instance map as the receipt and flag the remainder.
- Scoring exceeds 12h wall (theoretical max 21 x 1800s = 10.5h plus builds) — kill, keep partials, investigate the stuck instance via scores-docker/run_evaluation logs before retrying.


## Deliverables

- 3090 repo: evals/swebench/m4-imports/qwen36-m4/predictions.jsonl (imported copy) + scores-docker-summary.json + scores-docker/qwen36-m4.*.report.json — committed (dir verified not gitignored).
- 3090 repo README.md: tick the 'Score M4's exported qwen36 predictions' fleet-audit queue bullet (line 10 at HEAD 7d3170f) with the resolved/26 rate and receipt path; add a one-line cross-team heartbeat note.
- M4 repo (checkout on the 3090 box, or diff handed to orchestrator if absent/no push): evals/swebench/exports/README.md — replace stale '15 of 16' with 21/26, delete the nonexistent --output flag from the documented command, add a Results section with the Docker rate + pointer to the 3090 receipt; copy scores-docker-summary.json in as exports/qwen36-docker-summary.json.
- M4 repo README.md: tick fleet-audit item 4 ('Re-send the Docker-scoring handoff', line 12) and update the qwen36 headline from 'resolved 5/13 = 38.5% (M4-scorable)' to the official full-26 Docker rate.
- Log file logs/qwen36-m4-score.log on the 3090 box (gitignored, local receipt of the run).


## Constraints

- Rule 2 (3090 CLAUDE.md line 95): no concurrent rollout + score — confirm no bake-off cycle or docker rollout is live (sudo docker ps + no run_model_cycle.sh/run_all_cycles.sh processes) before starting; serialize via the established flock (/tmp/loop-bakeoff-logs/score.lock) so a queued bake-off cannot collide.
- Keep --max-workers 1 (script default) — the 2026-05-10 vfs_getattr_nosec kernel BUG fires under concurrent docker I/O; the box has a recurring ~9-17h docker-I/O crash class.
- Detach the scoring run via setsid with a log file (fleet rule: >30min jobs detached); validate behavior not exit status — the summary JSON, not rc, is the receipt.
- READ-only on GPUs: this is a CPU/Docker workload; do not stop or restart the SGLang server for it, and do not raise workers to compete with a live server.
- Negative results are findings: instances the harness marks error/unresolved are data for M4's characterization, not failures of this task.


## Risks

- Docker-I/O kernel BUG mid-run (recurring ~9-17h class on this box) — mitigated by max-workers 1, cache-level env resumability, and the resume-once kill rule.
- The 8 M4-INSTALL-FAIL instances pin very old Pythons; some SWE-bench env images may fail to build even on x86 — bounded by the 1/3-error kill criterion and recorded as findings.
- Score inflation/deflation from model-generated helper files at testbed root: --filter-helpers stays OFF (repo default since 2026-05-15, it net-regressed 40.3->39.3); if empty-vs-error counts look anomalous, audit_predictions.py exists for inspection — do not silently rescore with the filter on.
- M4-side commit may not be possible from the 3090 box (checkout/push unknown from here) — fallback path (diff to orchestrator) is specced so results still flow back.
- Result may land below the 50% band that 'confirms the qwen36 recommendation' — that is a valid finding; the deliverable is the number, not a target.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
