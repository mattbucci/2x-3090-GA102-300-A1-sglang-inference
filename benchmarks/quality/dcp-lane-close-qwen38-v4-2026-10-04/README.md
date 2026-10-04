# qwen38 opencode-dcp v4 — lane close (2026-10-04, 300/300, rc 0 at 12:48 PDT)

DSpark spec on, 256K window, 32K output budget, network=none, docker-served. Control = `qwen38-opencode-v4`
(same server, same cycle). Scored verdict comes from Phase 7 (`scores-docker-summary.json`); everything
here is pre-score rollout shape.

## Leak gate — PASS on exposure, FAIL on transcripts (expected)

`audit_leakage.py --require-isolation`: n=300, transcripts 298, attempted 152 (51 %), **EXPOSED 0**, isolation
proof 298/300 on all four lines (`leak-dcp-v4.json`). The two transcript-less rows are the rc −1 infra rows
(`django-13220`, `django-13230`) that Phase 6 re-rolls; the gate re-runs after that.

The first pass read EXPOSED 3 — all three were local output misattributed to a silent fetch inside a
compound command under `network=none`, fixed in the classifier (attribution-only, 182/186 netopen regression
and both v3 arms unchanged):

| instance | compound command shape | what the output really was | rule |
|---|---|---|---|
| astropy-14995 | `curl -s -o X … && diff X …` | `diff: X: No such file or directory` — `-o` target never created | `LOCAL_ERR_LINE_RE` + missing-target counts as empty download |
| django-11999 | `pip show … ; python -c … ; curl -s -o /dev/null -w "%{http_code}" …` | `000` from curl's own `-w` | `HTTP_CODE_W_RE` + `000` ⇒ no connection |
| matplotlib-25311 | `curl -s … \| python -c "json.load(sys.stdin)"` | `JSONDecodeError … (char 0)` on an empty body, sole non-scaffold output | `PIPED_PY_RE` + `JSON_EMPTY_RE` |
| pytest-5103 | `git remote -v; git log --oneline --all \| head; curl -s … \| head` | `9c17c3d base` — the sandbox's own re-initialised history | `GIT_ONELINE_RE`, only when the command has no `git fetch/pull/clone` |

## Paired vs opencode-v4 (290 same-ID instances, infra rows excluded)

|  | walls (rc 124) | empty | median s | mean s | sum h |
|---|---|---|---|---|---|
| opencode-v4 (control) | 33 | 35 | 802.2 | 962.5 | 77.53 |
| opencode-dcp-v4 | 35 | 38 | 939.8 | 1049.1 | 84.51 |

Per-instance speedup (control/lane, median) **0.86×** — the DCP lane is slower. Flips: wall→done 19,
done→wall 21, empty→patch 17, patch→empty 20. rc-0 empties in the lane: 4 (`django-11019`, `django-13315`,
`pytest-11148`, `pytest-5103`).

## Where the extra time goes (`lane_overhead.py --run`)

| | pre (build+boot→first event) | session | post (last event→exit) | cleanup session present |
|---|---|---|---|---|
| control | 193 s | 537 s | 0 s | 250/250 |
| dcp | 265 s | 450 s | 120 s | 11/256 |

The DCP lane pays ~70 s more boot (offline npm resolution of the plugin) and the full 120 s cleanup timeout
on 245/256 instances — the cleanup `opencode run` has to boot the plugin again and never registers a session
before `timeout 120` fires. The model session itself is 87 s *shorter* in median. Context shape
(`prompt_sawtooth.py`): 66 % of dcp sessions ever dropped (median peak prompt 101.5K) vs 79 % (112.5K) in
the control — pruning keeps the context smaller and auto-compaction rarer.

## DCP engagement A/B (`dcp-lane-receipt-qwen38-v4.md`)

Engaged (pruned > 0 tokens) on 95/217 observed containers (44 %); median 36,273 tokens pruned when engaged,
4.03 M total. On the 95 engaged IDs: lane patched 67 / walled 28 (median 1515 s) vs control on the same IDs
patched 77 / walled 18 (median 1269 s). On the 195 never-engaged IDs: 185 patched / 7 walls / 747 s median.
Engagement selects the long instances, so this is not a causal read — but on exactly the instances where DCP
acts, the lane walls more and patches less than the control.

## Walls — 35 rc 124 (`walls-dcp-v4.json`, `audit_wall_causes.py`)

| class | n | instances |
|---|---|---|
| generating (model mid-think at the kill) | 15 | django-11564 12453 12470 13265 15252 16910, matplotlib-25311, seaborn-2848, xarray-3364, pylint-6506, sphinx-10451 8474, sympy-11870 13146 16503 |
| tool_running (test suite inside the command budget) | 15 | django-13448 14667 15738 16816, matplotlib-23913 25079 25442, xarray-4248, sympy-12236 13177 13915 20212 20590 23191 24909 |
| step_boundary | 4 | django-14999, sympy-13895, sympy-17022, sympy-18532 |
| permission_hang (subagent `external_directory` prompt) | 1 | pytest-7220 |

Two of the four step_boundary walls — **django-14999 and sympy-17022 — are finished-but-walled**: the session
had completed and verified its fix (the sqlite snapshot holds exactly one session, so the kill landed before
the cleanup pass registered), but the wall-hit path captures no diff (`=== DIFF ===` is printed by the inner
script *after* the cleanup pass; on `TimeoutExpired` the harness only snapshots sessions, then kills), and
on the DCP lane the cleanup pass's ~70 s boot + 120 s ceiling eats the margin. Both opencode arms produced
rc-0 patches on these two instances. **2/300 complete fixes lost to the harness in this cell.**

### New capture-defect class: `=== DIFF ===` marker collision (`sympy-20212`)

The row is rc 124 **with a 21,856-byte "patch"** that is not a diff. The model ran
`ps aux | grep -E "sympy|python"` in the sandbox; the inner script sits in `bash -lc` argv, so its
`echo === DIFF ===` landed in a tool output inside the opencode JSON stream. On a wall the real marker is
never printed, `_extract_diff_from_stdout` uses `rfind`, and the "patch" is the event-stream tail after the
echoed copy. Cycle-wide sweep: 4 rows, all walls — dcp-v3 `pytest-5103` (97 KB), opencode-v4 `pytest-8365`
(77 KB) + `sympy-13915` (204 KB), dcp-v4 `sympy-20212`. Such a patch cannot apply (scores unresolved, like
the empty it should have been), so no resolve-rate changes; `paired_lane_compare.py` / `audit_wall_causes.py`
now read a non-diff `model_patch` as empty (reporting-only), which moves the committed full-300 v4-vs-v3 read
from empty 33 → 35 on the v4 arm (walls unchanged; see the erratum in that receipt).

## Cycle-boundary fixes this cell adds to the list (never mid-cycle)

1. Wall-hit capture: `docker exec … git -C /testbed add -A && git diff --cached` alongside the session
   snapshot (recovers finished-but-walled fixes and partial edits; tag `model_timeout`, per the R9700 proposal).
2. Marker: anchor the extractor to a line-start `=== DIFF ===` (the `ps`-echoed copy is mid-line inside a JSON
   string) and require the capture to start with `diff --git` or be empty — or write the diff to a file in the
   container and `docker cp` it, which removes the marker entirely.
3. DCP lane: vendor the plugin tarball (kill the ~70 s offline-npm boot) and let the cleanup pass start from
   the running opencode instead of a fresh boot, or skip the cleanup `opencode run` on the DCP lane.
