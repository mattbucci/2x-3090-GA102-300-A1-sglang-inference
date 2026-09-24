#!/usr/bin/env python3
"""Offline unit test — audit_leakage.py classifies a compound bash command per
list segment, not on the whole line.

No GPU, no server, no run dir. Reproduces the qwen38 opencode v3 leak-gate
false EXPOSED (pytest-dev__pytest-8365, 2026-09-24): one bash call was

    git -C /testbed show --stat HEAD | head -20; pip download --no-deps 2>/dev/null | head -2;
    ls /testbed/.git 2>/dev/null; find / -name "*.whl" -path "*pytest*" 2>/dev/null | head

Under --network none the (argument-less) pip printed nothing to stdout, the
local segments printed the HEAD stat / .git listing, and the audit read

  - kind UPSTREAM  — the 120-char snippet window after `pip download` reached
    the `find … -path "*pytest*"` glob two commands later, and
  - ok=True        — "some output, no failure signature" on the shared stream.

One such event fails the cell's leak gate and Phase 4.5 of run_model_cycle.sh
exits the whole cycle unscored. Fix: (a) every net tool is classified on its
own list segment (kind from the segment's URLs / project tokens); (b) a pip
download/install that names no requirement is not a network call and emits
no event. The OUTCOME stays a reading of the shared output — a quiet success
(`pip -q … | tail -2; ls /tmp/dj41`, `gh pr diff … | head`) still shows in
the following segments' output, and the v2-netopen cell had five such cells at
gold overlap 1.0 that must keep reading exposed.

Run anywhere with python3:
    python scripts/eval/test_audit_leakage_segments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "swebench"))
from audit_leakage import classify_call, command_segments  # noqa: E402

INST = "pytest-dev__pytest-8365"
HEAD_STAT = "commit 085f7a0\nAuthor: eval <eval@local>\n\n    base\n\n .coveragerc | 30 +\nHEAD\nconfig\nobjects\n"


def web(events):
    return [(e["kind"], e["ok"]) for e in events if e["chan"] == "web"]


CASES = [
    # (name, command, output, expected web events)
    ("pytest-8365 shape: argument-less pip beside local ls/find is not a net call",
     'git -C /testbed show --stat HEAD | head -20; pip download --no-deps 2>/dev/null | head -2; '
     'ls /testbed/.git 2>/dev/null; find / -name "*.whl" -path "*pytest*" 2>/dev/null | head',
     HEAD_STAT, []),
    ("argument-less pip with a dest dir is still a no-op",
     "pip download --no-deps -d /tmp/x -q 2>/dev/null; ls /tmp/x", "", []),
    ("v2-netopen shape: quiet pip success shows in the following ls -> exposed",
     "pip download pytest==5.0.0 --no-deps -d /tmp/p -q 2>&1 | tail -2; ls /tmp/p",
     "pytest-5.0.0-py3-none-any.whl\n", [("UPSTREAM", True)]),
    ("v2-netopen shape: gh pr diff content -> exposed",
     "gh pr view 8365 --repo pytest-dev/pytest 2>&1 | head -50; echo ---; gh pr diff 8365 --repo pytest-dev/pytest | head",
     "title: fix tmpdir\n---\ndiff --git a/src/_pytest/tmpdir.py\n", [("UPSTREAM", True), ("UPSTREAM", True)]),
    ("pip of the project in a compound command with empty output -> not fetched",
     "pip download pytest==5.0.0 --no-deps -d /tmp/p 2>/dev/null | tail -1; ls /tmp/p",
     "", [("UPSTREAM", False)]),
    ("failure signature wins in a compound command",
     "pip download pytest==5.0.0 --no-deps -d /tmp/p 2>&1 | tail -2; ls /tmp/p",
     "ERROR: Could not find a version that satisfies the requirement pytest==5.0.0 (from versions: none)\n",
     [("UPSTREAM", False)]),
    ("curl beside a local ls: shared output stays conservative",
     "ls /testbed/src && curl -s https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py | head -3",
     "__init__.py\n_pytest\n", [("UPSTREAM", True)]),
    ("two net segments are two events, each classified on its own URL",
     "curl -sv https://api.github.com/ 2>&1 | head -3; curl -s https://www.pytest.org 2>&1 | head -3",
     "curl: (6) Could not resolve host: api.github.com\n", [("OTHER", False), ("UPSTREAM", False)]),
    ("a project token in a NEIGHBOURING segment no longer makes a fetch UPSTREAM",
     'curl -s https://pypi.org/simple/ 2>/dev/null | head -1; grep -rn "pytest" /testbed/setup.cfg',
     "setup.cfg:1:[tool:pytest]\n", [("OTHER", True)]),
    ("v2-netopen matplotlib-25442 shape: python heredoc urlopen of the project API -> exposed",
     'timeout 60 python - <<\'EOF\'\nimport json, urllib.request\ndef get(url):\n'
     '    req = urllib.request.Request(url, headers={"User-Agent": "curl"})\n'
     '    with urllib.request.urlopen(req, timeout=30) as r:\n        return r.read().decode()\n'
     'd = get("https://api.github.com/repos/pytest-dev/pytest/compare/a...b")\nprint(d[:100])\nEOF',
     "files changed base->merge:\n  src/_pytest/tmpdir.py modified +2/-2\n",
     [("UPSTREAM", True), ("UPSTREAM", True)]),
    ("python -c urlopen under network none fails with gaierror -> not fetched",
     'python -c "import urllib.request; print(urllib.request.urlopen(\'https://pypi.org\').status)"',
     "urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>\n",
     [("OTHER", False)]),
    ("python urlopen traceback cut off by `| head -5` before the URLError line (django-13964, v3)",
     "timeout 15 python -c \"import urllib.request; print(urllib.request.urlopen("
     "'https://code.djangoproject.com/ticket/34691', timeout=10).read()[:500])\" 2>&1 | head -5",
     'Traceback (most recent call last):\n  File "/opt/miniconda3/envs/testbed/lib/python3.8/urllib/request.py",'
     ' line 1354, in do_open\n    h.request(req.get_method(), req.selector, req.data, headers,\n'
     '  File "/opt/miniconda3/envs/testbed/lib/python3.8/http/client.py", line 1256, in request\n'
     '    self._send_request(method, url, body, headers, encode_chunked)\n',
     [("OTHER", False)]),  # OTHER: the test instance is pytest, so a django URL is off-project
    ("a traceback from the repo's own code is not a network failure signature",
     'python -c "import urllib.request; print(urllib.request.urlopen(\'https://pypi.org\').read()[:80])"',
     'b\'<!doctype html>\\n<html lang="en">\'\nTraceback (most recent call last):\n'
     '  File "/testbed/src/_pytest/x.py", line 3, in <module>\n    x = 1/0\nZeroDivisionError: division by zero\n',
     [("OTHER", True)]),
    ("grep for urlopen in the repo is not a fetch",
     'grep -rn "urlopen(" /testbed/src | head', "src/_pytest/x.py:12: urlopen(url)\n", []),
    ("';' inside a quoted URL does not split the segment",
     'curl -sL "https://api.github.com/search/issues?q=repo:pytest-dev/pytest;x" | head -50',
     "", [("UPSTREAM", False)]),
]


def main() -> int:
    assert command_segments("a; b && c || d\ne") == ["a", "b", "c", "d", "e"]
    assert command_segments("echo 'a;b' && x") == ["echo 'a;b'", "x"]
    failed = 0
    for name, cmd, out, want in CASES:
        got = web(classify_call(INST, "bash", {"command": cmd}, out, None))
        ok = got == want
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
