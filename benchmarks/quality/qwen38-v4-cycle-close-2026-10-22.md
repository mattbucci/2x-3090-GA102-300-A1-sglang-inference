# qwen38 v4 cycle close — the five-scaffold harness bake-off (2026-10-22)

The first SWE-bench Lite cycle scored under the isolated harness (no network, stdin prompt, served
262,144 window, template-max thinking, 32,768 output budget, Docker-served `qwen38` under DSpark at
true 256K / fp8 KV — contract: [`swebench-harness-isolation-2026-09-20.md`](swebench-harness-isolation-2026-09-20.md)).
Five scaffolds rolled the same 300 instances against the same served model; this receipt is the
table, the pairwise statistics, the harness decision, and the per-lane chronology that used to sit
in the README's Status item 1. Cell JSONs: `bakeoff-qwen38-{opencode,opencode-dcp,little-coder,little-coder-rtk,prime}.json`;
run dirs `evals/swebench/runs/qwen38-<scaffold>-v4/` (scores under `scores-docker/`, recall audits in
`recall-audit.json`, first draws of re-rolled rows in `predictions.jsonl.pre-reroll`).

Cycle wall: 2026-09-27 → 2026-10-22 (re-rolls, two mid-cycle stops, the 2026-10-19 → 10-20 stack
pause for SGLang v0.5.22 + prime-agent 0.10.0 — rows carry the served config and the scaffold pin
they ran under); scoring pass 11 h 26 min for the five lanes, `swebench==4.1.0`, ≈2 h 20 min per lane.

## Table (full 300, isolated)

| Scaffold | pin | resolved | % | walls (rc 124) | empty patches | median rollout | vs opencode (McNemar) |
|---|---|:--:|:--:|:--:|:--:|:--:|:--|
| opencode | opencode-ai 1.14.25 | **196** | 65.3 | 13 | 6 | 804 s | — |
| opencode + DCP | + opencode-dcp 3.1.15 | **203** | 67.7 | 14 | 10 (+1 scorer error) | 923 s | 25 vs 18 discordant, p = 0.36 |
| little-coder | 1.19.0 / pi 0.83.0 | **193** | 64.3 | 25 | 5 | 749 s | 16 vs 19, p = 0.74 |
| little-coder + RTK | + rtk 0.46.0 | **202** | 67.3 | 15 | 4 | 734 s | 22 vs 16, p = 0.42 |
| prime | prime-agent 0.8.1 (297 rows + 3 never-rolled build-crash rows) | **187** | 62.3 | 30 | 37 | 912 s | 13 vs 22, p = 0.18 |

Full pairwise matrix (discordant instances only-A / only-B, two-sided exact McNemar):

| | opencode + DCP | little-coder | little-coder + RTK | prime |
|---|---|---|---|---|
| opencode | 18 / 25 · p 0.36 | 19 / 16 · p 0.74 | 16 / 22 · p 0.42 | 22 / 13 · p 0.18 |
| opencode + DCP | | 28 / 18 · p 0.18 | 22 / 21 · p 1.00 | 30 / 14 · **p 0.023** |
| little-coder | | | 13 / 22 · p 0.18 | 27 / 21 · p 0.47 |
| little-coder + RTK | | | | 29 / 14 · **p 0.032** |

Union of resolved ids across the five cells: 234/300; resolved by all five: 151. The only pairs
that separate at p < 0.05 are prime against the two plugin lanes. **Every other pair is a tie** — the
four opencode / pi cells span 193–203 on 300 instances, inside one standard error (≈ ±8 at 65 %).

Walls / empties are harness-level reads, not model reads: the opencode-family walls are mid-think or
inside a running test suite (`audit_wall_causes.py`), pi's are the same classes plus its larger
compaction cadence, and prime 0.8.1's 37 empties are its single-strike `length` policy (no retry
after a thinking-only response hits the 32K budget — 33/34 of those cut thinks are benchmark recall;
0.10.0 keeps the policy; see the prime lane receipt).

## Harness decision — prune to plain opencode

The user's call (2026-10-22): once the table names the best harness, every later cycle rolls that one
lane. The table does not name one on resolve rate — both plugin A/Bs are null (DCP p = 0.36 with the
pruning engaged on 110/300 instances; RTK p = 0.18 with rtk engaged on 100 % and 2.6K tokens saved per
~100K-token session), and opencode vs little-coder is p = 0.74 — so the pick is on everything else:

1. **Fewest moving parts.** Plain opencode is a single pinned npm package; DCP adds a plugin whose
   boot tax (offline `npm install` awaited by plugin load) cost ~70 s per instance and a cleanup-ceiling
   class, RTK adds a shell shim + an extension that only loads via pi's `-e` flag, and the pi lanes
   needed the `developer`-role template patch, the package model-profile pin and the pi-0.68 → 0.83
   re-pin to produce a clean cell at all.
2. **Lowest wall rate at the top of the table:** 13 walls vs little-coder's 25 at the same resolve
   rate; empties 6 vs 5.
3. **Longitudinal comparability.** opencode is the one scaffold with a cell on every historical
   preset (netopen cells, the v3 no-spec qwen38 reference, the netopen-vs-isolated exposure study), the
   recall audit, the wall-cause audit and the DCP/RTK receipts all read the opencode session format,
   and R9700's matrix keeps opencode(+DCP) as its paired reference — one opencode column per model is
   the comparison both rigs can make.
4. **Tooling target.** Flint (next model, user request) ships vendor presets for OpenCode and pi;
   opencode keeps that bring-up on the same lane.

If the literal top cell is preferred, `opencode-dcp` (203) is one token in `SCAFFOLDS_FOR`; nothing
else in the queue changes. Dropped with the pruning: the `-v5` one-off qwen38 cycle (dcode + omp +
prime 0.10.0) and the `-respsplit` dcode scoring — dcode's defect (patch 066) and omp's title fork are
documented in their receipts, and neither lane rolls again. The DCP / RTK engagement receipts stay in
`benchmarks/quality/{dcp,rtk}-engagement*/` (untracked, local).

## Exposure effect, restated on the isolated cell

opencode isolated 196/300 vs the same scaffold under `--network=host` 250/300: on the 186 instances
the netopen cell fetched upstream for, it resolved 94.6 % and the isolated cell resolves 55.9 %; on the
114 it never fetched, 64.9 % vs **80.7 %** isolated. 75 of the 76 instances lost between the cells were
exposed ones, 19 of the 22 gained were clean. The historical table was measuring exposure, not models.

## Benchmark recall (Phase 7b, opencode-family lanes)

| lane | sessions with recall | long thinks with recall | walls | recall hits 0 / 1–9 / 10–29 / 30+ → n | median s | resolved |
|---|:--:|:--:|:--:|---|---|---|
| opencode | 277/300 | 962 / 1223 | 13 | 23 / 56 / 69 / 152 | 291 / 418 / 761 / 1139 | 19 / 49 / 50 / 78 |
| opencode + DCP | 277/300 | 935 / 1296 | 14 | 23 / 77 / 84 / 116 | 529 / 604 / 986 / 1171 | 19 / 64 / 58 / 62 |

Same shape as the v3 pre-score read: 92 % of sessions recognise the benchmark, session duration
tracks recall 4× across buckets, resolve rate by bucket tracks instance difficulty (the 0-hit bucket
is the easy tail, 19/23). The trigger is the task contract ("Do not modify tests" + a diff capture),
not harness paths — every isolation proof holds on these cells. R9700's comparable table is in their
`FP8_BAKEOFF_SETUP.md`; compare shapes, not rates (their cues differ).

## Lane chronology (moved from README Status item 1)

* **v3 no-spec reference cells:** opencode 300/300 2026-09-24 (0 exposed, 53 walls / 1 `length`),
  opencode-dcp 201/300. The v4 cycle restarted 2026-09-27 with `qwen38` under DSpark at the same
  256K window (`SPEC_FOR=qwen38:1`; receipts `dspark-cuda-2026-09-27/`).
* **opencode** first pass 300/300 2026-09-30 (0 exposed, 33 walls / 2 `length`, 8 infra rows); paired
  vs v3: walls 33 vs 52, empties 33 vs 53, median rollout 801 vs 1108 s, per-instance 1.18×, session
  phase 537 vs 735 s with build + boot unchanged, accept length 3.02 / 88 tok/s steady.
* **opencode-dcp** 300/300 2026-10-04 (0 exposed on 152 attempts; 35 walls vs 33 on 290 paired ids,
  median 940 vs 802 s — ~70 s plugin boot + the 120 s cleanup ceiling on 245/256 instances while model
  sessions were 87 s shorter; DCP engaged on 44 %; two verified fixes lost at the wall to the capture
  path; [receipt](dcp-lane-close-qwen38-v4-2026-10-04/README.md)). Root cause of the boot tax
  (opencode's startup `npm install @opencode-ai/plugin` awaited by plugin load) fixed 2026-10-21
  (`667e3da`, Dockerfile pre-install + `npm_config_offline`).
* **little-coder** 300/300 2026-10-11 at **pi 0.83** (control re-pinned to the rtk lane's 1.19.0
  prefix 2026-10-09 and rolled from scratch; the pi-0.68 roll is quarantined `-pi068`,
  [receipt](lc-lane-close-qwen38-v4-2026-10-07/README.md) carries **patch 065**, the inter-call
  whitespace delta pi 0.68 turned into phantom nameless tool calls on 8.9 % of multi-call turns; 0
  phantom calls / 0 tool-name rejections in 271 transcripts on 0.83). 23 walls vs opencode's 32 on 263
  paired ids, median 755 vs 807 s, 41 % of sessions compacted at ~100K median peak. **Bridge
  self-dispatch** found here (`sympy-13773`: six raw `/v1/chat/completions` self-calls asking the served
  model to recall the upstream PR; rtk `pytest-5103` eight) — not exposure (nothing but the server is
  behind the bridge), routed to a `loopback_*` channel by the classifier (`071c642`); hiding the bridge
  from `ps` and a self-dispatch policy stay boundary items;
  [receipt](lc-lane-close-qwen38-v4-pi083-2026-10-11/README.md). Every row and `meta.json` carries a
  `scaffold_pin`; a resume refuses a cell whose rows ran under another pin.
* **little-coder-rtk** 300/300 2026-10-12 (rows 0–175 before capture-at-wall, 176–299 after; 20 infra
  rows from a five-minute npm `ETARGET` blip → the bounded build retry `533b2f9`; 0 exposed on 229
  attempts). A/B vs the pi-0.83 control on 250 paired ids: rtk engaged on 100 % (median 17 rewrites,
  2.6K tokens saved per session) and moved nothing — 24 walls vs 21, 225 patches vs 229, median 781 vs
  772 s; [receipt](rtk-lane-close-qwen38-v4-2026-10-12/README.md).
* **prime** 300/300 2026-10-16 (prime-agent 0.8.1 vendored; 263 patches / 32 rc-0 empties / 30 walls /
  3 infra rows; 0 exposed on 103 attempts; 0.85× vs opencode on 290 paired ids). The empties are a
  scaffold × model interaction: 34 sessions end on a thinking-only `length` stop and 0.8.1 makes no
  retry; 33/34 of the cut thinks are benchmark recall (18 enumerate 474–2,826 candidate instance ids).
  Leak-classifier fix at this close: python-executor tools gated by name (`ipython` reports kernel
  status, not fetch outcome); [receipt](prime-lane-close-qwen38-v4-2026-10-16/README.md). The 0.8.1
  release was pruned upstream 2026-10-09 (29 crash rows on little-coder, vendored tgz + HF mirror since);
  the pin moved to 0.10.0 at the 2026-10-19 boundary, so this cell scores on its 297 0.8.1 rows + 3
  never-rolled crash rows and prime never re-rolled under 0.10.0 (dropped with the pruning).
* **dcode** 300/300 2026-10-18 but **defect-confounded — quarantined `-respsplit`, unscored:** 213
  rc-0 empties (71 %) were a server-side Responses-API history split (`phase` tag merge) that taught the
  model in-context to stop after an announce-only turn; **patch 066** fixes the merge, gated on the wire
  by `probe_responses_api.py`; [receipt](dcode-lane-close-qwen38-v4-2026-10-18/README.md). Not re-rolled
  (pruning).
* **Phase 4 re-roll** 2026-10-21 (resumed under v0.5.22 + prime 0.10.0 after the 10-19 → 10-20 pause):
  142 infra / pre-capture-wall rows re-drew (opencode 41, DCP 37, little-coder 29, rtk 35). **Leak gate
  5/5** after two attribution false-FAILs (a DCP instance probing the bridge as an HTTP proxy from python;
  sixteen prime `websearch` tracebacks; then prime's three never-rolled rows in the isolation
  denominator — [receipt](leak-audit-bridge-proxy-and-websearch-2026-10-21.md)). Scoring 2026-10-21
  21:39 → 2026-10-22 09:03; recall audit; aggregate; the queue holds on PAUSE before `qwen36-dense`.
* **DCP A/B receipt:** [`dcp-ab-qwen38-v4-2026-10-22/`](dcp-ab-qwen38-v4-2026-10-22/README.md)
  (18 vs 25 discordant, per-repo within ±2 except sympy +6, one scorer-error row `psf__requests-863` =
  pre-`b66d27f` `build/` residue, harness defect counted unresolved).
