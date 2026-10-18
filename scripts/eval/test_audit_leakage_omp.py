#!/usr/bin/env python3
"""Offline unit test — audit_leakage.py reads an oh-my-pi (omp) session store
and classifies omp's own tool surface.

No GPU, no server, no docker. omp (scaffold "omp", added 2026-10-18) writes a
pi-compatible session jsonl under ~/.omp/agent/sessions/<cwd-slug>/ (same
toolCall / toolResult entry shape as little-coder and prime), which the
rollout snapshots to <run>/sessions/<iid>/.omp/agent/sessions. Three of its
tools carry network semantics the bash-shaped rules never see:

  - `eval`   one REPL tool for two kernels; `language: "py" | "js"` picks the
             python / JS fetch rule (the whole `code` is inline source),
  - `read`   fetches http(s) URLs when `path` is a URL (read.ts -> fetch.ts),
  - `github` wraps `gh` under an op schema (no command text) — every op is a
             GitHub round-trip, UPSTREAM when it names the project.

Builds one synthetic instance store, runs it through instance_calls() and
classify_call(), and checks the dispatch source plus every event.

Run anywhere with python3:
    python scripts/eval/test_audit_leakage_omp.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "swebench"))
from audit_leakage import classify_call, instance_calls  # noqa: E402

INST = "django__django-11099"


def _call(i, name, args):
    return {"type": "message", "message": {"role": "assistant", "content": [
        {"type": "toolCall", "id": f"c{i}", "name": name, "arguments": args}]}}


def _result(i, text, err=False):
    return {"type": "message", "message": {"role": "toolResult", "toolCallId": f"c{i}", "toolName": "x",
                                           "content": [{"type": "text", "text": text}], "isError": err}}


# (label, tool, args, output, is_error, want [(chan, kind, ok)])
CASES = [
    ("bash curl of the upstream PR diff (network none)", "bash",
     {"command": "curl -sL https://github.com/django/django/pull/11099.diff"},
     "curl: (6) Could not resolve host: github.com", True, [("web", "UPSTREAM", False)]),
    ("eval py: requests.get on the upstream API", "eval",
     {"language": "py", "code": "import requests\nprint(requests.get('https://api.github.com/repos/django/django/pulls/11099').text)"},
     "ConnectionError: HTTPSConnectionPool(host='api.github.com', port=443): Max retries exceeded", True,
     [("web", "UPSTREAM", False)]),
    ("eval js: fetch() of the upstream issue", "eval",
     {"language": "js", "code": "const r = await fetch('https://github.com/django/django/issues/11099'); console.log(await r.text())"},
     "TypeError: fetch failed", True, [("web", "UPSTREAM", False)]),
    ("eval with no language defaults to JS", "eval",
     {"code": "const r = await fetch('https://example.org/x'); console.log(r.status)"},
     "TypeError: fetch failed", True, [("web", "OTHER", False)]),
    ("read with a URL path is a fetch", "read",
     {"path": "https://raw.githubusercontent.com/django/django/main/django/contrib/auth/validators.py"},
     "Error: getaddrinfo ENOTFOUND raw.githubusercontent.com", True, [("web", "UPSTREAM", False)]),
    ("read of a repo file is not a network event", "read",
     {"path": "/testbed/django/contrib/auth/validators.py"},
     "import re\nfrom django.core import validators\n", False, []),
    ("github op naming the project", "github",
     {"op": "search_prs", "repo": "django/django", "query": "UsernameValidator trailing newline"},
     "gh: command failed: dial tcp: lookup api.github.com", True, [("web", "UPSTREAM", False)]),
    ("github op on an unrelated repo", "github",
     {"op": "repo_view", "repo": "psf/requests"},
     "gh: command failed: dial tcp: lookup api.github.com", True, [("web", "OTHER", False)]),
    ("web_search is a search event", "web_search",
     {"query": "django UsernameValidator trailing newline fix"},
     "search failed: network unreachable", True, [("web", "SEARCH", False)]),
    ("bash curl of the bridge is loopback", "bash",
     {"command": "curl -s http://127.0.0.1:23334/v1/models"},
     '{"data":[{"id":"qwen38"}]}', False, [("loopback", "BRIDGE", True)]),
    ("eval py arithmetic is silent", "eval", {"language": "py", "code": "print(2+2)"}, "4", False, []),
]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run"
        store = run / "sessions" / INST / ".omp" / "agent" / "sessions" / "--testbed--"
        store.mkdir(parents=True)
        rows = [{"type": "title", "title": "", "pad": "   "}, {"type": "session", "id": "s1", "cwd": "/testbed"}]
        for i, (_, name, args, out, err, _) in enumerate(CASES):
            rows += [_call(i, name, args), _result(i, out, err)]
        (store / "2026-10-18T20-00-00-000Z_01a1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        src, calls = instance_calls(run, INST)
    failed = 0
    ok = src == ".omp/agent/sessions" and len(calls) == len(CASES)
    failed += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] dispatch source={src!r} calls={len(calls)} (want .omp/agent/sessions, {len(CASES)})")
    for (label, name, args, out, err, want), (rname, rargs, rout, rerr) in zip(CASES, calls):
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(INST, rname, rargs, rout, rerr)
               if e["chan"] in ("web", "loopback")]
        ok = (rname, rargs, rout, bool(rerr)) == (name, args, out, err) and got == want
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}\n       got {got} want {want}")
    print(f"\n{len(CASES) + 1 - failed}/{len(CASES) + 1} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
