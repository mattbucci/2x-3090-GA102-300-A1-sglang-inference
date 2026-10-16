# qwen38 prime v4 — lane close (2026-10-16, 300/300, rc 0 at 14:51 PDT)

DSpark spec on (draft 9, fp8 KV, 307,659-token pool), 256K window (`context_window` 262144 from
`/v1/models`), 32K output budget, thinking at the template default (max), network=none, docker-served
(image `sha256:b8ce541a…`, per-cycle bearer). Scaffold = **prime-agent 0.8.1** (vendored tgz, since the
2026-10-09 release pruning — `vendor_prime_agent.sh`), pin `prime-agent@0.8.1` on 297/300 rows. Lane
rolled 2026-10-12 23:41 → 10-16 14:51 PDT (3 d 15 h) with no pause. Primary tool is an in-process
`ipython` kernel (16,984 calls; 44 `websearch`, ~10 malformed `%%bash…</parameter` names). Scored verdict
comes from Phase 7; everything here is pre-score rollout shape. The dcode lane started the same second.

Rows: **267 rc-0 (235 patches, 32 empty), 30 walls (rc 124; 28 diff-bearing via capture-at-wall, 2 empty),
3 rc −1 infra.** 263 patches total. `rollout_seconds` median 912 s (rc-0 + walls), mean 1,054 s,
sum 84.9 h over the 290 instances paired with opencode-v4.

**Infra rows — the third upstream-dependency outage of the cycle (all 2026-10-13, 20:10–20:35 Z).**
Row 37 (django-12470): the rtk-lane installer step `curl … github.com:443 → Connection refused` (3.1 s).
Rows 52–53 (django-13265, -13315): `npm install -g opencode-ai@1.14.25 little-coder@1.1.0` →
`ECONNRESET` from `registry.npmjs.org` (exit 152 / exit 1). None of the three is prime's own install
(vendored, offline) — the per-instance image still rebuilds every scaffold's toolchain on every row, so a
25-minute host-network blip costs three rows of a lane that never touches those tools. Phase 4 re-rolls them
(`infra_rollout_nonzero_rc`). Running count of rows lost to upstream outages this cycle, quarantined
cells included: **85** (82 before this lane + 3).

## Leak gate — PASS on exposure (`leak-prime-v4.json`, classifier as of this commit)

`audit_leakage.py --require-isolation`: n=300, transcripts 297, **attempted 103 (35 %)** — web_UPSTREAM 78,
web_SEARCH 25, web_OTHER 57, git_LIST 24, git_READ 10; all dead under network=none — **EXPOSED 0**,
isolation proof 297/297 on all four lines (the 3 transcript-less rows are the infra rows; the gate re-runs
after the re-roll, which is when `ok` flips to true). Gold overlap on the isolated set: n=257, median
0.125, ≥80 % on 58.

**Classifier fix shipped with this close (required before Phase 4.5 runs, ~4 days out).** The pre-fix audit
read EXPOSED=4 on prime and would have exit-1'd the cycle unscored. All four were false positives of the
same shape — prime's `ipython` tool reports `status: ok` for *the kernel*, not for the fetch, and the
shell-shaped scaffold-line rules did not recognise python-side evidence:

| instance | what the transcript actually shows | rule added |
|---|---|---|
| django-12284 | `curl` of the upstream PR, model printed `len: 0` | `key: N` scaffold lines (`rc=`, `len=`, `status=`, `bytes=` …) |
| django-14730 | model printed `6 0` — curl rc 6 (resolve failure), 0 bytes | bare `rc [bytes]` line |
| django-15388 | `000CURL_FAIL` — curl's `%{http_code}` glued to a sentinel | HTTP 000 followed by a sentinel word |
| django-12184 | `uv` usage error, `urlopen` never executed | CLI usage/`--help` trailers; `SyntaxError` blocks (`File "<string>"` / `Cell In[n]`) |

The same pass closed a bigger blind spot: the inline-python fetch rule only fired on shell-shaped
`python -c '…requests.get…'` segments, so 102 **direct in-kernel** `requests.get(...)` / `urlopen(...)`
calls on 96 instances were never classified at all (attempted 59 → 103). `PY_CODE_TOOLS` now gates the
rule on the tool name (`ipython`, `python`, `code_interpreter`, `jupyter`, …). Model-side exception lines
(`ERR ConnectionError`, `NET FAIL TypeError: … unexpected keyword argument 'timeout'`), tail-cut caret
lines and `#` comment segments no longer count as fetched content; `%s:%d` / `{host}` / `$HOST`
placeholder hosts resolve as loopback (requests' own test-server fixture). Unit coverage:
`scripts/eval/test_audit_leakage_segments.py` gained 11 prime cases (39/39 pass); regression on the two
netopen exposure cells is unchanged at 186 / 182 and the four closed v4 lanes are unchanged.

Loopback channel: django-12286 BRIDGE 3 (`curl 127.0.0.1:23334/health`, `/v1/models` — port enumeration,
no completion call), requests-863 / requests-3362 LOCAL (the project's own test servers). **0 `SELF`
calls** — prime did not find the self-oracle that pytest-5103 (lc) and sympy-16281 (rtk) did.

## Empties are a scaffold × model interaction: one strike on `length`, and the model spends it on recall

34 sessions end on an assistant message with `stopReason: length` (`length-stops-prime-v4.json`); in all 34
that message is the **last** one in the session — prime-agent 0.8.1 has no retry/nudge after a length stop
(pi nudges three times, opencode re-prompts). Every one of the 34 is thinking-only (0 text characters,
`output` = 32,000 = the budget), and 31 of the 32 rc-0 empties are exactly these rows (the 32nd,
matplotlib-23476, ended normally with no diff after 22 turns). Two of them (django-13448,
matplotlib-24265) burned the whole budget on the *first* turn — zero tool calls, the model never looked at
the repository.

What the budget went on — 33/34 thinks explicitly mention SWE-bench; the final think's shape:

| tail pattern | n | what it looks like |
|---|---:|---|
| instance-id enumeration | 18 | 474–2,826 `<repo>__<repo>-NNNNN` tokens, almost all distinct — the model walks candidate benchmark ids in sequence trying to place the task |
| ticket-# enumeration | 1 | django-11905 — cycles Trac ticket numbers |
| counter loop | 2 | django-10924, sympy-17022 — a bare integer sequence (`43253, 43254, …`) to the budget |
| reasoning (cut) | 13 | genuine long analysis cut at 32K — 12 of these also name SWE-bench on the way |

This is the same benchmark-recall behaviour R9700's recall audit tied to their wall hits and that our
opencode-family cells showed as 15–30K-token thinks; under prime it converts into empties instead of walls
because the scaffold gives up on the first `length`. Cycle-boundary check: does prime-agent 0.10.0 retry
on `length` (and if so, is the re-pin a fairness fix or a confound for the 0.8.1 cell)? Either way the
0.8.1 cell stands as rolled — it is what the scaffold does.

## Sawtooth (`sawtooth-prime-v4.json`)

294 sessions, 370 compaction events in 166 (56 % — lc 41 %, rtk 34 %), max peak prompt 225,362, 279
sessions ever above 32K. Compacted vs never: patched 124/166 vs 108/128, empty 12 vs 20, walls 30 vs 0,
median wall 1,251 vs 680 s, median peak prompt 92.7K vs 93.0K. The 128K-window hypothesis for the high
rate was tested and rejected — the peak-prompt histogram is evenly spread from <50K to 225K and rtk has the
same shape; prime compacts on its own cadence. Walls compact a median 4×.

## Where the time goes (`lane_overhead.py --run`)

rc-0 n=261 (pins only): `rollout_seconds` median 869 s; pre (build + boot → first event) median **179 s**
(≈20 %); session median 671 s; post (last event → exit) median 0 s; cleanup session present on 261/261.
Walls silent ≥100 s before the kill: 9/30, none at a finished step. `lane_overhead.py` and
`audit_wall_causes.py` now read prime's flat `.prime/agent/sessions/*.jsonl` alongside pi's
`.pi/agent/sessions/*/*.jsonl` (both printed "no snapshots" on prime before this close).

## Walls (`walls-prime-v4.json`, 30, all rc 124, 28 diff-bearing)

Classes: step_boundary 16, generating 6, tool_running 4, tool_stuck 4.

| class | instances |
|---|---|
| step_boundary 16 | django-11001†, -11564, -14016, -15252, matplotlib-25079, -25311, xarray-3364, -4248, pytest-11148, scikit-learn-10949, sphinx-8627, sympy-14317, -16988, -18057, -18698 (empty), -20212 |
| generating 6 | django-16229, seaborn-2848, sympy-12236 (2,784 s — the build ran inside the wall), -13773, -18087, -23191 |
| tool_running 4 | django-11019, matplotlib-25433†, sphinx-8474†, sympy-17655 |
| tool_stuck 4 | django-16820 (polls a background PID for the full suite), pytest-5103 (`while os.path.exists('/proc/399')` — same), pytest-9359 (empty; 12 calls, AST analysis in-kernel), sympy-11870 (`timeout 1200 python bin/test …` on four test files) |

† = the wall landed in the cleanup pass. Three of the four `tool_stuck` are the model deliberately parking
on a full-test-suite run — a time-budget choice, not a kernel hang. Capture-at-wall kept the diff on 28/30.

## Paired against the other closed v4 lanes

| | vs opencode-v4 (290 paired) | vs little-coder-v4 (268 paired) |
|---|---|---|
| walls | 30 vs 33 | 29 vs 24 |
| empties | 32 vs 35 | 33 vs 4 |
| median s | 912 vs 799 | 913 vs 759 |
| sum h | 84.9 vs 77.0 | 78.8 vs 67.3 |
| per-instance speedup (median) | **0.85×** | **0.80×** |
| flips wall→done / done→wall | 17 / 14 | 14 / 19 |
| flips empty→patch / patch→empty | 27 / 24 | 3 / 32 |

The `patch_to_empty` 32 against little-coder is the length-stop class above, one for one.

## Cycle-boundary items raised or re-confirmed by this close

- prime-agent length-stop retry (0.10.0 behaviour; fairness vs confound) — see above.
- once-per-cycle scaffold image / bounded `docker build` retry — three more rows lost to tools this lane
  never runs.
- `audit_benchmark_recall.py` reads only `opencode.db`; prime/pi session formats carry the same recall
  thinks (33/34 here) — port the reader so the Phase 7b table covers all six scaffolds.
- bridge off `ps` argv + self-dispatch policy (unchanged; prime found the bridge port but not the oracle).
