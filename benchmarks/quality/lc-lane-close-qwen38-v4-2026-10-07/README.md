# qwen38 little-coder v4 — lane close (2026-10-07, 300/300, rc 0 at 05:35 PDT)

DSpark spec on, 256K window, 32K output budget, network=none, docker-served. little-coder 1.1.0 = pi-ai
0.68.1 (the rtk lane's control). Cross-scaffold reference = `qwen38-opencode-v4` (same server, same
cycle). Scored verdict comes from Phase 7 (`scores-docker-summary.json`); everything here is pre-score
rollout shape.

Rows: **260 rc-0 patches, 1 rc-0 empty (`sympy-15011`), 16 walls (rc 124, all 0 B), 23 rc −1 infra**
(`docker build` failed before the scaffold ran: 10 prime-installer HTTP 500, 4 rtk-installer, 9 npm E404 —
registry flakiness paid per instance because every rollout image reinstalls all six scaffolds; Phase 6
re-rolls them, the multi-stage Dockerfile is the cycle-boundary fix).

## Leak gate — PASS on exposure (`leak-little-coder-v4.json`, HEAD classifier)

`audit_leakage.py --require-isolation`: n=300, transcripts 277, attempted 224 (81 % — pi exposes
`websearch`/`webfetch`, so 16 instances also ran a web search; all dead under network=none), **EXPOSED 0**,
isolation proof 277/277 on all four lines. The 23 transcript-less rows are the infra rows; the gate re-runs
after the re-roll. Gold overlap on the isolated set: n=256, median 0.1, ≥80 % on 56.

The first pass read EXPOSED 4 (`django-14667`, `django-15388`, `pytest-8365`, `sympy-13471`) — all four were
pi's own ShellSession scaffolding misread as fetched content, fixed in the classifier (`1486ee9`,
attribution-only; the 186/182 netopen regression and the three opencode-family v3/v4 arms unchanged):

| output after a silent `curl` under network=none | what it is | rule |
|---|---|---|
| `[exit=0 cwd=/testbed timed_out=false backend=subprocess]` | ShellSession's status trailer on every result | `SCAFFOLD_LINE_RE` |
| `exit=6` / `EXIT=0` / `curl_exit=7` | the model's own `echo exit=$?` after the fetch | `SCAFFOLD_LINE_RE` (prefixed exit markers) |

## Paired vs opencode-v4 (274 same-ID instances, infra rows excluded; `paired-little-coder-v4-vs-opencode-v4.json`)

|  | walls (rc 124) | empty | median s | mean s | sum h |
|---|---|---|---|---|---|
| opencode-v4 | 30 | 31 | 792.0 | 943.1 | 71.78 |
| little-coder-v4 | 16 | 17 | 668.6 | 837.2 | 63.72 |

Per-instance speedup (opencode/lc, median) **1.11×**. Flips: wall→done 19, done→wall 5, empty→patch 20,
patch→empty 6. Same model, same server: the pi scaffold walls half as often as opencode on this model.

## Where the time goes (`lane_overhead.py --run`, rc-0 rows under the wall)

| lane | rollout med | pre (build+boot→first row) | session | post (last row→exit) | cleanup pass present |
|---|---|---|---|---|---|
| little-coder-v4 | 613 s | 183 s | 428 s | 0 s | 252/252 |
| opencode-v4 | 730 s | 193 s | 537 s | 0 s | 250/250 |
| opencode-dcp-v4 | 836 s | 265 s | 450 s | 120 s | 11/256 |

Context shape (`prompt_sawtooth.py`, `sawtooth-little-coder-v4.json`): 191/275 sessions compacted (69 %),
median peak prompt 94,215 tokens, max 197,568 — vs 79 % / 112.5K on opencode-v4. Compacted sessions: 175
patched / 15 walled, median 867 s; never-compacted: 83 patched / 1 walled, median 441 s.

## Walls — 16 rc 124 (`walls-little-coder-v4.json`, `audit_wall_causes.py`, pi reader `0f1595d`)

| class | n | instances |
|---|---|---|
| tool_running (test suite inside the command budget at the kill) | 10 | django-13757, xarray-4248, sklearn-15535 25747, sympy-11870 13146 16988 18057 19254 23191 |
| step_boundary (tool result in, next request not yet made) | 6 | django-15252 (in the cleanup pass), matplotlib-23299, seaborn-2848, pytest-5103, sympy-12236 18698 |

Seven of the ten tool_running walls are sympy full-suite / multi-module `bin/test` runs with 300–600 s
budgets; the model re-runs them after each edit. **django-15252 is finished-but-walled**: the snapshot
holds the cleanup session, i.e. the fix was in and the kill landed during the 120 s cleanup pass — the
wall-hit path captures no diff (same capture defect as the dcp cell's 2/300). 1/300 complete fix lost to the
harness in this cell. Walled on every v4 lane so far (0/4 across opencode, dcp, lc + both v3 arms where
present): django-15252, xarray-4248, sympy-11870, sympy-13146, sympy-23191.

The one rc-0 empty, `sympy-15011`: the model tried `webfetch` for the upstream file, got nothing, and
ended the session without an edit.

## pi 0.68 scaffold quirks surfaced by this cell

**Phantom nameless tool call — a serving-side ordering defect, fixed as patch 065.** In 168 of 1,898
multi-call turns (8.9 %; 110/300 instances) the transcript shows `toolCall(real), text("\n"),
toolCall{id:'', name:'', arguments:{}}, toolCall(real)`. Mechanism (CPU-reproduced at random chunkings,
285/1000 splits of a two-call turn → 0/1000 with 065): one streaming increment spanning `</tool_call>\n<tool_call>`
returns a single `StreamingParseResult`, and `serving_chat` yields its `normal_text` (the `\n`) BEFORE the
result's tool-call deltas, so the client sees a content delta between call N's name and its closing `}`.
pi-ai 0.68 tracks the *current block* rather than `index`, so the displaced deltas open a fresh nameless
tool call that the agent executes (`Your tool call had an empty name`); in 5 turns the whole argument body
was displaced — the real call went out with `{}` and the phantom carried its arguments (victims: websearch 2,
bash 2, read 1). opencode (AI SDK) and every pi-ai from 0.83.0 through 1.0.4 (latest, 2026-10-05 — `ensureToolCallBlock`
keys on `index`, one text block per message; verified in the 1.0.4 tarball) resolve by `index` and are
immune; our rtk + prime lanes run 0.83.0, and any published `little-coder` ≤1.20.0 pins
`pi-coding-agent ^0.83.0` → 0.83.0, so only the 1.1.0 control lane (`@mariozechner/pi-ai` 0.68.1) pays it — **an A/B confound against the rtk lane** (control pays it, rtk does not); DSpark's multi-token
increments make the span routine. 0 non-whitespace inter-call prose in 1,939 sandwiched text blocks, so the
whitespace drop covers the whole observed class. Unit test cases 7–8, `patches/README.md` 065. Live tree +
serving image at the cycle boundary.

**149 `Tool 'X' is not defined in the tools list` rejections** (grep 60, Read 38, Glob 13, Bash 7, Grep 5 —
the model naming tools pi does not expose or in the wrong case). pi's own empty-name nudge lists
`Read, Write, Edit, Bash, Glob, Grep` (capitalised) while its tools are `bash, read, write, edit, glob,
grep`; `grep` appears in the nudge but pi 0.68 rejected it here. Each costs a turn; none killed a session.

**`bash whitelist: "curl" is not in SAFE_PREFIXES`** — little-coder's `bash` tool refuses `curl`/`timeout`
prefixes; the model routes them through `ShellSession` instead (where the fetch dies on network=none).

## Cycle-boundary items this cell adds (never mid-cycle)

1. Patch 065 → live tree (`git apply`, 3-gate replay) → serving image rebuild (image carries its own tree).
2. Multi-stage `Dockerfile.rollout` (scaffold installs once, per-instance stage only overlays) — the 23
   infra rows here, 9 + 2 + 8 on the earlier lanes, are all registry blips paid per instance.
3. Wall-hit capture (`git add -A && git diff --cached` + `git stash` on the kill, tag `model_timeout`) —
   django-15252 here, django-14999 + sympy-17022 on dcp.
