# Flint × omp: per-tool key order probe (2026-11-11)

**Question.** Does keeping omp's `edit` in its declared, intent-first order (`DECLARED_ORDER_TOOLS = ("edit",)`, 25c40ea) remove the filler-argument `edit` calls that the v5 cell showed, without bringing back the `bash` defect?

**Setup.** 24 instances (seed 1111, [`instance-ids.txt`](instance-ids.txt)), bare serving from the tree (`FLINT_STRICT_TOOLS=1`, plugin reinstalled from `plugins/flint_sglang`), omp 18.12.2. Rollout `docker_rollout.py --timeout 1800 --max-empty-streak 30`, run `evals/swebench/runs/flint-omp-probe-editorder`. The v4 and v5 rows are the same 24 instances, taken from the full-300 cells.

| Same 24 instances | v4 (`i` first everywhere) | v5 (`i` last everywhere) | probe (`edit` declared order, `bash` `i` last) |
|---|---|---|---|
| `edit` calls | 237 | 1,620 | 244 |
| `edit` filler main argument | 2 (0.8 %) | 651 (40.2 %) | 12 (4.9 %, 9 instances) |
| `edit` errors | 130 | 1,525 | 162 |
| `bash` filler main argument | 57 / 1,690 (3.4 %) | 2 / 1,194 (0.2 %) | 1 / 1,234 (0.1 %) |
| resolved (docker scorer, information only) | 16 | 12 (3 empty) | 14 (1 empty) |

Filler = main argument (`bash.command`, `edit.input`) not a string or one of `true/false/x/y/none/null/test/replace/yes/no/""` ([`tool_arg_degeneracy.py`](tool_arg_degeneracy.py); on the full-300 cells it reads v4 7.9 % / 2.4 %, v5 0.4 % / 46.4 % for bash / edit, close to the lane-close receipt's 7.9 / 1.9 / 0.3 / 45.7 %).

**Verdict: pass.** The `edit` retry loop is gone (call count back to v4's level), and `bash` keeps v5's gain. A 4.9 % residual of filler `edit` inputs remains (Flint writes `True` / `x` after a correct `i`), so the per-tool order fixes most of the defect, not all of it. Two of v5's three empty patches resolve again (django-15851, pytest-7168). 24 instances cannot separate 14 from 16 resolves. The serving image still carries the v5 plugin until the next boundary rebuild.
