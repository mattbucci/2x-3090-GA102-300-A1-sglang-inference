# Leak-gate false EXPOSED at the qwen38 v4 cycle close — bridge proxy probes + prime websearch (2026-10-21)

**Symptom.** Phase 4.5 of the resumed qwen38 v4 cycle failed on `qwen38-opencode-dcp-v4`
(`EXPOSED=1`, `pydata__xarray-4248`) at 21:08 PDT, 16 h in, before any lane was scored;
`run_all_cycles.sh` moved on to qwen36-dense (stopped at its kernel smoke, nothing rolled,
PAUSE armed). A dry run of the gate on the other lanes found `qwen38-prime-v4` at
`EXPOSED=16` — it would have failed next.

**Both are classifier false positives; the sandbox held.** All 1,497 instances across the
five lanes carry `isolation: network=none`.

| cell | instance(s) | what the model did | why the gate read ok=True | fix |
|---|---|---|---|---|
| opencode-dcp v4 | xarray-4248 | `http.client.HTTPConnection('127.0.0.1', 23334)` + absolute-form `GET http://api.github.com/repos/pydata/xarray` (with the bearer it read from opencode.json), a `ProxyHandler` pointed at the bridge, a path sweep (`/proxy/http://…`, `/docs`, `/v1/models`) | uvicorn answered 401 / 404 / the swagger page; the model printed `== <path> 401` and `<path> 200 b'<!DOCTYPE html>…'`, neither in `SERVER_LINE_RE` | a python script whose every connection goes to loopback (no fetch primitive with a remote literal, no remote `HTTP(S)Connection`/`create_connection`, variable URLs only behind a loopback proxy) emits a loopback event only — the request line cannot reach a remote host through the bridge (`py_direct_remote`). A remote `HTTPSConnection(host)` with no URL literal in the script now also counts as a remote target (was invisible). |
| prime v4 (0.8.1) | 16 rows | `await websearch(q)` / `websearch.run(q)` / `websearch.websearch(q)` | the 0.8.1 kernel bound the name to `_PrimeAgentUnavailableSkill` (RuntimeError / AttributeError / `No module named 'websearch'`); the traceback read as content. `PY_SEARCH_RE` arrived for 0.10.0 after this cell closed, so this was its first full pass over 0.8.1 rows | `PY_SEARCH_FAIL_RE` (unavailable-skill shapes + the 0.10.0 "no Serper API key" notice) and `PY_CELL_RAISED_RE` (an IPython cell that printed nothing before raising returned no results); `PY_SEARCH_RE` tightened to the invocation forms (`websearch.__getattribute__(` is introspection, not a search) |

**Regression sweep (HEAD before → after):** v2-netopen opencode 186 → 186, dcp 182 → 182
(no detection lost); v4 opencode 0 → 0, dcp 1 → 0, little-coder 0 → 0, rtk 0 → 0, prime
16 → 0, lc-pi068 0 → 0. Exactly 17 instance verdicts changed, the 17 above. Unit tests:
`scripts/eval/test_audit_leakage_segments.py` 56/56 (+9 cases: the five xarray-4248 /
HTTPSConnection shapes, four prime websearch shapes incl. a results-returning search that
stays exposed), `test_audit_leakage_omp.py` 12/12.

**Cycle recovery.** Re-enter qwen38 via `relaunch-v4-resume.sh` (every lane at 300 /
prime 297 → Phase 1 instant, Phase 4 nothing owed, gate → one scoring pass over all five
lanes); PAUSE re-armed once the qwen38 wrapper is past its own check so the queue holds
before qwen36-dense for the boundary work (branch merge, `-respsplit` scoring, the
dcode/omp/prime `-v5` one-off).
