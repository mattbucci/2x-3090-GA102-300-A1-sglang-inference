# qwen38 little-coder v4 (pi 0.83 control) — lane close (2026-10-11, 300/300, rc 0 at 20:59 PDT)

DSpark spec on, 256K window, 32K output budget, network=none, docker-served. The lane was re-pinned
mid-cycle on 2026-10-09 (user: the pi-0.68 phantom-tool-call defect is a defect, not a convention) to
little-coder 1.19.0 = `@earendil-works/pi-coding-agent` 0.83.0 (`/opt/lc-rtk` prefix **without** the rtk
`-e` flag) and rolled from scratch; the pi-0.68 cell is quarantined as `qwen38-little-coder-v4-pi068`
(10-07 receipt one directory over). This is the rtk lane's control. Cross-scaffold reference =
`qwen38-opencode-v4` (same server, same cycle). Scored verdict comes from Phase 7; everything here is
pre-score rollout shape.

Rows: **247 rc-0 patches, 0 rc-0 empties, 24 walls (rc 124 — 20 diff-bearing via capture-at-wall, 4 empty),
29 rc −1 infra** (rows 0–86 of the resumed roster: `docker build` 404'd on the prime-agent 0.8.1 installer
after upstream pruned every pre-0.10.0 release on 2026-10-09; vendored tarball since `623abdf`; Phase 4
re-rolls them). Pins 271 × `little-coder@1.19.0/@earendil-works/pi-coding-agent@0.83.0`. rc-0
`rollout_seconds` median 705.6 s, mean 806.4 s, sum 55.33 h.

## Leak gate — PASS on exposure (`leak-little-coder-v4-pi083.json`, HEAD classifier `071c642`)

`audit_leakage.py --require-isolation`: n=300, transcripts 271, attempted 225 (83 % — web_UPSTREAM 212,
web_SEARCH 13, web_OTHER 3, git_LIST 39; all dead under network=none), **EXPOSED 0**, isolation proof
271/271 on all four lines. The 29 transcript-less rows are the infra rows; the gate re-runs after the
re-roll. Gold overlap on the isolated set: n=263, median 0.044, ≥80 % on 46.

The first pass read EXPOSED 1 (`sympy-13773`) and it was a new channel, not a scaffold-trailer misread:

**Bridge self-dispatch.** Inside the sandbox `127.0.0.1:23334` is the container half of `net_bridge.py` —
a loopback socket that reaches the SGLang server and nothing else. The model found the port in `ps aux`
(the per-instance script's `bash -lc` argv carries `net_bridge.py container 23334 /run/bridge.sock` and a
urlopen health loop) and the bearer in pi's `models.json` (pi 0.83 writes the literal key). sympy-13773
then (a) probed the bridge as an HTTP proxy — absolute-URI `GET http://github.com/...` through it, which
uvicorn answers with its own 401 `{"error":"Unauthorized"}` / 404 `{"detail":"Not Found"}`, plus `/` →
`"SGLang is running"` and `/openapi.json` — and (b) made six raw `POST /v1/chat/completions` calls to the
served model (max_tokens 16000, temperature 0, 900 s client timeouts) asking it to *recall the upstream PR*
for the issue. Answers came back empty / `length` / hallucinated; the instance walled at a step boundary
with 1 compaction and a 197K peak prompt. No gold sits behind the bridge, so this is **not exposure** —
but it is (i) GPU time outside scaffold accounting, (ii) a benchmark-recall channel through a self-oracle
(the same instinct R9700's recall audit measures in-session), and (iii) a false positive for a classifier
that treats any URL-bearing fetch with a status line as a web fetch.

Fix (`071c642`, attribution-only; 28/28 offline cases, netopen regression unchanged at 186 / 182, the
opencode-v4 / dcp-v4 / pi068 / rtk arms unchanged at EXPOSED 0): a `loopback` channel —
`LOOPBACK_HOST_RE` / `LOOPBACK_CONN_RE` route any call whose only targets are 127.0.0.1 / localhost / ::1 /
`bridge.sock` to `loopback_SELF` (an OpenAI-shaped path on the bridge port), `loopback_BRIDGE` (any other
touch of the bridge port) or `loopback_LOCAL` (the model's own dev server); `SERVER_LINE_RE` keeps
uvicorn's status / `detail` / `openapi` lines out of the fetched-content byte count so a bridge 404 can no
longer read as `web UPSTREAM ok=True`. Remote URLs in the same call still classify on their own merits.

| instance | loopback | calls | what |
|---|---|---|---|
| sympy-13773 | SELF | 14 | proxy probes + six `/v1/chat/completions` self-calls (the recall attempt) |
| django-13220 | BRIDGE | 1 | `curl 127.0.0.1:23334` while listing listeners |
| requests-2674 | LOCAL | 1 | its own `python -m http.server` on 127.0.0.1 |

Cross-cell "mentions 23334" (almost all `ps` output, not calls): lc-pi083 16/271, rtk 8/176 (so far),
pi068 23/277, opencode-v4 3, dcp-v4 1, v3 arms 0. Self-chat only here and in the rtk lane's pytest-5103
(eight calls, got a hallucinated "PR #11923").

## Paired vs opencode-v4 (263 same-ID instances, infra rows excluded; `paired-little-coder-v4-pi083-vs-opencode-v4.json`)

|  | walls (rc 124) | empty | median s | mean s | sum h |
|---|---|---|---|---|---|
| opencode-v4 | 32 | 34 | 806.7 | 977.8 | 71.43 |
| little-coder-v4 (pi 0.83) | 23 | 4 | 755.0 | 901.8 | 65.88 |

Per-instance speedup (opencode/lc, median) **1.01×**. Flips: wall→done 21, done→wall 12, empty→patch 32,
patch→empty 2. The empty column is the capture-at-wall landing (`b66d27f`): 20 of this cell's 24 walls
carry a diff; opencode-v4 rolled before it and still counts its walls as empties. Same model, same server:
pi 0.83 walls ~30 % less often than opencode and leaves 4 empties to opencode's 34.

## Where the time goes (`lane_overhead.py --run`, rc-0 rows under the wall)

| lane | rollout med | pre (build+boot→first row) | session | post (last row→exit) | cleanup pass present |
|---|---|---|---|---|---|
| little-coder-v4 (pi 0.83) | 664 s | 180 s | 502 s | 0 s | 240/240 |
| little-coder-v4-pi068 | 613 s | 183 s | 428 s | 0 s | 252/252 |
| opencode-v4 | 730 s | 193 s | 537 s | 0 s | 250/250 |

Walls silent ≥100 s before the kill: 4/24; with a finished step: 0/24 (no finished-but-walled instance —
`in_cleanup=False` on all 24, the capture-at-wall path took the diff on 20).

Context shape (`prompt_sawtooth.py --cleanup-drop`, `sawtooth-little-coder-v4-pi083.json`): 112/270
sessions compacted (41 % — vs 69 % on pi 0.68 and 79 % on opencode-v4), 210 compaction events, median
peak prompt 101,940 (compacted) / 98,808 (never), max 247,300, 264 sessions ever above 32K. Compacted:
88 patched / 24 walled, median 1278 s; never-compacted: 158 patched / 0 walled, median 579 s. Walls
median 2.5 compactions; every wall compacted at least once. The four empty walls had 2–4 compactions each
(peaks 73K–191K).

## Walls — 24 rc 124 (`walls-little-coder-v4-pi083.json`, `audit_wall_causes.py`)

| class | n | instances |
|---|---|---|
| step_boundary (tool result in, next request in flight <120 s) | 17 | django-11564 16820 16910, matplotlib-25442 25498, seaborn-2848, sphinx-8474, sympy-11897 13146 13177 13773 13895 13915 16503 17630 18698 19254 |
| tool_running (command open under its own ceiling at the kill) | 6 | django-11964 (774/920 s idle-poll loop), xarray-3364 (160/900), pytest-7220 (48/400), sklearn-25570 (1301/1200 — over its budget, kill landed first), sympy-12236 (14/3600), sympy-16988 (59/900) |
| generating (request in flight >120 s) | 1 | sympy-23191 (144 s) |

Walls here are long sessions that never finished, not test suites hung at the kill: 17/24 are at a step
boundary with 47–174 tool calls behind them, which is the opposite shape from pi 0.68 (10/16 tool_running).
Walled on every v4 lane so far: sympy-13146, sympy-23191 (plus django-15252 / xarray-4248 / sympy-11870 on
the pi-0.68 cell, all three patched here).

## pi 0.83 scaffold notes

- **0 phantom nameless tool calls** — server `Tool 'None' is not defined` 0, empty-name nudges 0 in
  271 transcripts (pi 0.68 cell: 168/1,898 multi-call turns). The A/B confound against the rtk lane is
  gone for the control; patch 065 (the serving-side whitespace-delta fix) stays a cycle-boundary item for
  any client that keys on the current block.
- **0 `Tool 'X' is not defined` rejections** (pi 0.68 cell: 149). The capitalised-name nudge mismatch is
  fixed upstream.
- Tool surface this cell used: bash 14,255 · read 3,200 · glob 1,449 · edit 1,308 · write 632 ·
  webfetch 315 (212 instances) · websearch 249 (207) · ShellStart/ShellLog/ShellList/ShellStop 521 ·
  grep 41 · BrowserNavigate 15 (15 instances — every call fails at launch, the rollout image has no
  playwright chromium; not a network event) · **`dispatch` 6 calls on 5 instances** — pi 0.83's sub-agent
  tool. The sub-agents run through the scaffold against the same served model (inside scaffold
  accounting, unlike the bridge self-dispatch above) but their sessions are **not persisted** — the
  snapshot holds only the parent session, so a sub-agent's `webfetch`/`websearch` calls reach the leak
  audit only as the report text it hands back (all five reports say every route was dead). Four of the
  five dispatch tasks ask the sub-agent to find the upstream issue/PR and its diff — the same recall
  instinct, one hop removed. network=none covers it; the audit's `attempted` count is a floor.
- Two sessions per instance on all 247 rc-0 rows (main + the universal cleanup pass); 24 walls have one.

## Cycle-boundary items this cell adds (never mid-cycle)

1. **Hide the bridge from `ps`**: launch `net_bridge.py` from a staged script file instead of inline
   `bash -lc` argv, and keep the health loop off argv — the port is the only discovery path the model used.
   Then decide policy on self-dispatch: document it as the model's agentic choice (current) or have the
   bridge refuse `POST /v1/*` bodies that did not originate from the scaffold's client (bearer is shared,
   so this needs a second header the scaffold sets).
2. `audit_benchmark_recall.py`: count `loopback_SELF` calls as recall attempts (self-oracle) alongside
   in-session recall thinks; relay the shape to R9700 (their sandbox bridge may be discoverable the same way).
3. ~~Restore the `docker logs -f` server.log follower~~ — done 2026-10-11 21:24 PDT: the follower died with
   the 2026-10-09 re-pin relaunch (the container kept running); restarted with `--since` the last written
   second, gap-free, so `prompt_sawtooth.py --server-log` covers the whole cycle again. Only the sawtooth
   and the boot-timeout tail read `server.log`; nothing in Phases 2–8 does.
4. Patch 065 → live tree + serving image (3-gate, `bench_regression.sh`); upstream PR for 063 + 065.
5. Multi-stage `Dockerfile.rollout` (scaffold installs once) + prime-agent 0.10.0 re-pin; DCP cleanup pass;
   opencode permission block.
