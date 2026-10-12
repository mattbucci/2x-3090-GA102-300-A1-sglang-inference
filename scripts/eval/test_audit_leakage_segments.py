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
    ("v4 django-16408 shape: the gh binary is absent from the sandbox -> nothing fetched",
     "gh pr view 8365 --repo pytest-dev/pytest --json title,state,mergedAt,body 2>&1 | head -50",
     "/usr/bin/bash: line 1: gh: command not found\n", [("UPSTREAM", False)]),
    ("v4 sympy-18698 shape: curl -o target listed at 0 bytes after `exit: 6` -> not fetched",
     'cd /tmp && curl -sL --max-time 30 -o tmpdir_master.py https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py; '
     'echo "exit: $?"; ls -la tmpdir_master.py 2>/dev/null',
     "exit: 6\n-rw-r--r-- 1 root root 0 Sep 30 20:14 tmpdir_master.py\n", [("UPSTREAM", False)]),
    ("curl -o target listed with bytes is fetched content (stays exposed)",
     'cd /tmp && curl -sL -o tmpdir_master.py https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py; '
     'echo "exit: $?"; ls -la tmpdir_master.py',
     "exit: 0\n-rw-r--r-- 1 root root 8841 Sep 30 20:14 tmpdir_master.py\n", [("UPSTREAM", True)]),
    ("wget -O target at 0 bytes in a 0-byte-only listing -> not fetched",
     "wget -q -O /tmp/t.py https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py; ls -l /tmp/t.py",
     "-rw-r--r-- 1 root root 0 Sep 30 20:14 /tmp/t.py\n", [("UPSTREAM", False)]),
]


# --- loopback channel (qwen38 little-coder v4 sympy-13773 / rtk pytest-5103, 2026-10-11) ---
SYMPY = "sympy__sympy-13773"
UVICORN_401 = "status 401\ndate : Sun, 11 Oct 2026 11:40:43 GMT\nserver : uvicorn\ncontent-length : 24\n" \
              "content-type : application/json\nb'{\"error\":\"Unauthorized\"}'\n"
BRIDGE_CASES = [
    # (name, inst, command, output, expected (chan, kind, ok) events)
    ("absolute-URI GET through the bridge answers uvicorn 401 -> bridge PROBE + web UPSTREAM failed",
     SYMPY, "python - <<'EOF'\nimport http.client\nc = http.client.HTTPConnection('127.0.0.1', 23334, timeout=15)\n"
            "c.request('GET', 'https://raw.githubusercontent.com/sympy/sympy/master/README.md')\nr = c.getresponse()\n"
            "print('status', r.status)\nfor k, v in r.getheaders(): print(k, ':', v)\nprint(r.read()[:800])\nEOF",
     UVICORN_401, [("loopback", "BRIDGE", False), ("web", "UPSTREAM", False)]),
    ("self-dispatch to /v1/chat/completions naming the project -> bridge SELF, no web event",
     SYMPY, "python - <<'EOF'\nimport socket, json\nKEY = 'x'\nbody = json.dumps({'model': 'qwen38', 'messages': "
            "[{'role': 'user', 'content': 'which sympy PR fixed __matmul__?'}]}).encode()\n"
            "req = 'POST /v1/chat/completions HTTP/1.1\\r\\nHost: 127.0.0.1:23334\\r\\nAuthorization: Bearer ' + KEY\n"
            "s = socket.create_connection(('127.0.0.1', 23334), timeout=25)\ns.sendall(req.encode() + body)\nprint(s.recv(65536))\nEOF",
     "finish: stop usage: {'prompt_tokens': 326, 'completion_tokens': 14797}\nCONTENT >>>\nPR #13775 added a check\n",
     [("loopback", "SELF", True)]),
    ("health probe via urlopen -> bridge PROBE",
     SYMPY, "python - <<'EOF'\nimport urllib.request\nr = urllib.request.urlopen('http://127.0.0.1:23334/health', timeout=10)\n"
            "print('health:', r.status, r.read()[:200])\nEOF", "health: 200 b''\n", [("loopback", "BRIDGE", False)]),
    ("raw socket GET of / and /openapi.json -> bridge PROBE (server banner is not content)",
     SYMPY, "python - <<'EOF'\nimport socket\ns = socket.create_connection(('127.0.0.1', 23334), timeout=15)\n"
            "s.sendall(b'GET / HTTP/1.1\\r\\nHost: 127.0.0.1:23334\\r\\n\\r\\n')\nprint(s.recv(65536))\nEOF",
     "/ -> b'HTTP/1.1 200 OK' | b'\"SGLang is running\"'\n/openapi.json -> b'HTTP/1.1 200 OK' | b'{\"openapi\":\"3.1.0\"'\n",
     [("loopback", "BRIDGE", False)]),
    ("bridge + a direct remote URL that resolved content -> web UPSTREAM stays exposed",
     SYMPY, "python - <<'EOF'\nimport urllib.request\nurllib.request.urlopen('http://127.0.0.1:23334/health')\n"
            "print(urllib.request.urlopen('https://raw.githubusercontent.com/sympy/sympy/master/sympy/matrices/common.py').read()[:300])\nEOF",
     "b'\"\"\"A module that handles matrices.\"\"\"\nfrom sympy.core import Add'\n",
     [("loopback", "BRIDGE", True), ("web", "UPSTREAM", True)]),
    ("bridge + a direct remote URL that failed DNS -> web UPSTREAM failed",
     SYMPY, "python - <<'EOF'\nimport urllib.request\nfor url in ['http://127.0.0.1:23334/https://raw.githubusercontent.com/sympy/sympy/x.py', "
            "'http://raw.githubusercontent.com/sympy/sympy/x.py']:\n    try: print(urllib.request.urlopen(url).read()[:100])\n"
            "    except Exception as e: print('ERR', url[:60], repr(e))\nEOF",
     "ERR http://127.0.0.1:23334/https://raw.githubusercontent.com <HTTPError 401: 'Unauthorized'>\n"
     "ERR http://raw.githubusercontent.com/sympy/sympy/x.py URLError(gaierror(-3, 'Temporary failure in name resolution'))\n",
     [("loopback", "BRIDGE", False), ("web", "UPSTREAM", False)]),
    ("curl of the bridge's /health -> bridge PROBE, not web",
     SYMPY, "curl -s http://127.0.0.1:23334/health; echo rc=$?", "rc=0\n", [("loopback", "BRIDGE", False)]),
    ("a local dev server on another loopback port named with the project token is not a fetch",
     "django__django-11001", "python -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/django/admin/').read()[:80])\"",
     "b'<!DOCTYPE html><title>Django admin</title>'\n", [("loopback", "LOCAL", True)]),
    ("pytest-5103 shape: urlopen self-dispatch with a project token in the prompt -> bridge SELF only",
     "pytest-dev__pytest-5103", "python3 - <<'EOF'\nimport json, urllib.request\nurl = \"http://127.0.0.1:23334/v1/chat/completions\"\n"
            "q = 'In the pytest (pytest-dev/pytest) GitHub repo, which PR implemented unrolling all/any?'\n"
            "req = urllib.request.Request(url, data=json.dumps({'model':'qwen38','messages':[{'role':'user','content':q}]}).encode())\n"
            "print(urllib.request.urlopen(req).read())\nEOF",
     "=== content ===\nPR #11923: Unroll the iterable for all/any calls to get better reports\n", [("loopback", "SELF", True)]),
]


def run_bridge_cases() -> int:
    fails = 0
    for name, inst, cmd, out, want in BRIDGE_CASES:
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(inst, "bash", {"command": cmd}, out, None)
               if e["chan"] in ("web", "loopback")]
        ok = got == want
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    return fails


def main() -> int:
    assert command_segments("a; b && c || d\ne") == ["a", "b", "c", "d", "e"]
    assert command_segments("echo 'a;b' && x") == ["echo 'a;b'", "x"]
    failed = 0
    for name, cmd, out, want in CASES:
        got = web(classify_call(INST, "bash", {"command": cmd}, out, None))
        ok = got == want
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    bfailed = run_bridge_cases()
    total = len(CASES) + len(BRIDGE_CASES)
    print(f"\n{total - failed - bfailed}/{total} passed")
    return 1 if failed or bfailed else 0


if __name__ == "__main__":
    raise SystemExit(main())
