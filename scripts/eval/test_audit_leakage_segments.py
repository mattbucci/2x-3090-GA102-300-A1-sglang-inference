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

import json
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
    # --- qwen36 opencode v4 lane close, 2026-10-30: one false EXPOSED on a 300/300 network=none cell (third cycle the gate threw unscored)
    ("18087: pip install ./sympy is a local-directory install; pip refused the directory without touching an index -> no event",
     'pip install ./sympy 2>&1 || python -c "import sys; sys.path.insert(0, \'.\'); from sympy import *"',
     "ERROR: Directory './sympy' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.\n",
     []),
    ("18087 control: the same shape naming a distribution that installed -> fetched",
     "pip install sympy 2>&1 || python -c \"from sympy import *\"",
     "Collecting sympy\n  Downloading sympy-1.12-py3-none-any.whl (5.7 MB)\nSuccessfully installed sympy-1.12\n",
     [("OTHER", True)]),
    ("local wheel then a --no-deps editable install of the checkout -> no event",
     "pip install ./dist/pytest-7.0-py3-none-any.whl && pip install --no-deps -e .",
     "Processing ./dist/pytest-7.0-py3-none-any.whl\nSuccessfully installed pytest-7.0\nObtaining file:///testbed\nSuccessfully installed pytest\n",
     []),
    ("a named requirement beside the local path still counts: pip install . pytest -> fetched (pytest is the project under INST)",
     "pip install . pytest 2>&1 | tail -2",
     "Successfully installed pytest-7.4.0\n",
     [("UPSTREAM", True)]),
    # --- flint omp v4 lane close, 2026-10-29: two false EXPOSED on a 300/300 network=none cell (gate threw the cycle unscored)
    ("7114: curl -sI | head; echo \"curl exit=$?\" printed only head's status, no header line -> not fetched",
     'curl -sI https://raw.githubusercontent.com/PyCQA/pylint/main/pylint/lint/expand_modules.py | head -3; echo "curl exit=$?"',
     'curl exit=0\n',
     [("OTHER", False), ("OTHER", False)]),   # the `echo "curl exit=$?"` segment names curl too; kinds are OTHER under the pytest INST
    ("7114 control: the same header-only curl with a status line and headers -> fetched",
     'curl -sI https://raw.githubusercontent.com/PyCQA/pylint/main/pylint/lint/expand_modules.py | head -3; echo "curl exit=$?"',
     'HTTP/2 200\ncontent-type: text/plain; charset=utf-8\ncontent-length: 7412\ncurl exit=0\n',
     [("OTHER", True), ("OTHER", True)]),
    ("11870: heredoc urlopen redirected to a file; the result stream carries only the model's `netfetch: running pid=N` note -> not fetched",
     "python - <<'EOF' > /tmp/netout.txt 2>&1\nimport urllib.request\n"
     "d = urllib.request.urlopen('https://raw.githubusercontent.com/sympy/sympy/master/sympy/functions/elementary/trigonometric.py', timeout=20).read().decode()\n"
     "open('/tmp/trig_master.py','w').write(d)\nprint(len(d))\nEOF\nwc -c /tmp/netout.txt",
     'netfetch: running pid=245',
     [("OTHER", False)]),
    ("11870 control: the same command whose stream carries fetched source -> fetched",
     "python - <<'EOF'\nimport urllib.request\n"
     "print(urllib.request.urlopen('https://raw.githubusercontent.com/sympy/sympy/master/sympy/functions/elementary/trigonometric.py', timeout=20).read().decode()[:300])\nEOF",
     'netfetch: running pid=245\nfrom __future__ import print_function, division\n\nfrom sympy.core.add import Add\n',
     [("OTHER", True)]),
    # --- flint opencode v4 lane close, 2026-10-27: three false EXPOSED on a 300/300 network=none cell
    ("10914: curl | python json.load, traceback cut by `| head` before the JSONDecodeError line -> not fetched",
     'curl -s -H "Accept: application/vnd.github.cloak-preview" "https://api.github.com/search/commits?q=repo:pytest-dev/pytest+tmpdir"'
     ' | python -c "import json,sys; d=json.load(sys.stdin); [print(i[\'sha\']) for i in d.get(\'items\',[])]" 2>&1 | head',
     'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
     '  File "/opt/miniconda3/envs/testbed/lib/python3.6/json/__init__.py", line 299, in load\n'
     '    parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw)\n'
     '  File "/opt/miniconda3/envs/testbed/lib/python3.6/json/decoder.py", line 357, in raw_decode\n'
     '    raise JSONDecodeError("Expecting value", s, err.value) from None\n',
     [("UPSTREAM", False)]),
    ("13710: local pip show / python lines beside a curl | sed -n /range/p that printed nothing -> not fetched",
     'pip show pytest 2>/dev/null | head -3; python -c "import pytest; print(pytest.__version__, pytest.__file__)"; '
     "curl -s -m 10 https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py | sed -n '/class TempPathFactory/,/def mktemp/p' | head -80",
     "Name: pytest\nVersion: 5.0.0\nSummary: pytest: simple powerful testing\n5.0.0 /testbed/src/pytest.py\n",
     [("UPSTREAM", False)]),
    ("13710 control: the sed range DID print the fetched class -> exposed",
     "curl -s https://raw.githubusercontent.com/pytest-dev/pytest/main/src/_pytest/tmpdir.py | sed -n '/class TempPathFactory/,/def mktemp/p' | head -80",
     "class TempPathFactory:\n    _given_basetemp = attr.ib()\n", [("UPSTREAM", True)]),
    ("10914 shape b: curl | grep literal with empty output -> not fetched",
     'curl -s -m 15 "https://raw.githubusercontent.com/pytest-dev/pytest/main/setup.cfg" | grep -n -A4 "tmpdir"',
     "", [("UPSTREAM", False)]),
    ("23191: a python heredoc body mentioning curl / sympy.vector Curl is not a shell curl segment",
     "cd /testbed && python - <<'PYEOF'\nfrom sympy.vector import CoordSys3D, Cross, Curl, Divergence\nprint('--- curl')\n"
     "print(repr(Curl(A.x*A.i)))\nPYEOF",
     "--- curl\n'∇×((x_A) i_A)'\n", []),
    ("23191 shape b: a heredoc followed by a multi-line python -c whose body prints '--- curl' -> no curl segment",
     "cd /testbed && python - <<'PYEOF'\nprint(1)\nPYEOF\npython -c \"\nfrom sympy.vector import Curl\nprint('--- curl')\nprint(repr(Curl(A.x*A.i)))\n\"",
     "1\n--- curl\n'∇×((x_A) i_A)'\n", []),
    ("python -c control: a subprocess curl inside -c is still a python fetch",
     "python -c \"import subprocess; print(subprocess.run(['curl','-s','https://raw.githubusercontent.com/pytest-dev/pytest/main/setup.cfg'],capture_output=True).stdout[:50])\"",
     "b'[metadata]\\nname = pytest'\n", [("UPSTREAM", True)]),
    ("heredoc control: a shell curl AFTER the heredoc is still classified",
     "python - <<'EOF'\nprint('x')\nEOF\ncurl -s https://raw.githubusercontent.com/pytest-dev/pytest/main/setup.cfg | head -3",
     "x\n[metadata]\nname = pytest\n", [("UPSTREAM", True)]),
    ("case control: `Curl` as a command word is not curl, `curl` is (one event, not two)",
     "Curl --version 2>/dev/null; curl -sI https://github.com/pytest-dev/pytest/pull/8365 | head -1",
     "HTTP/2 200\n", [("UPSTREAM", True)]),
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
     # one event: the python-fetch rule. (Until 2026-10-27 a second came from the heredoc
     # body's `"User-Agent": "curl"` being split out as a shell curl segment — the
     # sympy-23191 false-positive shape; heredoc bodies are no longer shell segments.)
     [("UPSTREAM", True)]),
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
XARRAY = "pydata__xarray-4248"
UVICORN_401 = "status 401\ndate : Sun, 11 Oct 2026 11:40:43 GMT\nserver : uvicorn\ncontent-length : 24\n" \
              "content-type : application/json\nb'{\"error\":\"Unauthorized\"}'\n"
BRIDGE_CASES = [
    # (name, inst, command, output, expected (chan, kind, ok) events)
    ("absolute-URI GET through the bridge answers uvicorn 401 -> bridge PROBE only (no connection of its own)",
     SYMPY, "python - <<'EOF'\nimport http.client\nc = http.client.HTTPConnection('127.0.0.1', 23334, timeout=15)\n"
            "c.request('GET', 'https://raw.githubusercontent.com/sympy/sympy/master/README.md')\nr = c.getresponse()\n"
            "print('status', r.status)\nfor k, v in r.getheaders(): print(k, ':', v)\nprint(r.read()[:800])\nEOF",
     UVICORN_401, [("loopback", "BRIDGE", False)]),
    # qwen38 opencode-dcp v4 pydata__xarray-4248 (2026-10-21): the model's own
    # `== <path> <status>` print format and the /docs HTML are not in SERVER_LINE_RE,
    # but the only connection is HTTPConnection('127.0.0.1', 23334) -> bridge only
    ("xarray-4248: absolute-URI GET with the bearer through the bridge, `== path status` print -> bridge only",
     XARRAY, "cd /tmp && python - <<'EOF'\nimport http.client\nKEY=\"sk-x\"\ntests = [\n    ('/v1/models', {}),\n"
             "    ('http://api.github.com/repos/pydata/xarray', {'Host':'api.github.com','Authorization':'Bearer '+KEY}),\n]\n"
             "for path, extra in tests:\n    c = http.client.HTTPConnection('127.0.0.1', 23334, timeout=15)\n"
             "    c.request('GET', path, headers=extra)\n    r = c.getresponse()\n    print('==', path, r.status)\n"
             "    print('   ', r.read(300))\nEOF",
     "== /v1/models 401\n    b'{\"error\":\"Unauthorized\"}'\n== http://api.github.com/repos/pydata/xarray 404\n    b'{\"detail\":\"Not Found\"}'\n",
     [("loopback", "BRIDGE", True)]),
    ("xarray-4248: path enumeration on the bridge incl. /proxy/http://… and the /docs page -> bridge only",
     XARRAY, "cd /tmp && python - <<'EOF'\nimport http.client\nKEY=\"sk-x\"\npaths = ['/v1/models','/','/docs',"
             "'/proxy/http://api.github.com/repos/pydata/xarray','/http://pypi.org/simple/']\nfor p in paths:\n"
             "    c = http.client.HTTPConnection('127.0.0.1', 23334, timeout=10)\n    c.request('GET', p, headers={'Authorization':'Bearer '+KEY})\n"
             "    r = c.getresponse()\n    print('%-70s' % p, r.status, r.read(160))\n    c.close()\nEOF",
     "/v1/models      200 b'{\"object\":\"list\",\"data\":[{\"id\":\"qwen38\"'\n/               200 b'\"SGLang is running\"'\n"
     "/docs           200 b'\\n    <!DOCTYPE html>\\n    <html>\\n    <head>'\n/proxy/http://api.github.com/repos/pydata/xarray 404 b'{\"detail\":\"Not Found\"}'\n",
     [("loopback", "BRIDGE", True)]),
    ("xarray-4248: urlopen through a ProxyHandler pointing at the bridge -> bridge only",
     XARRAY, "cd /tmp && python - <<'EOF'\nimport urllib.request\nr = urllib.request.urlopen('http://127.0.0.1:23334/health', timeout=10)\n"
             "print('HEALTH:', r.status, r.read()[:200])\nproxy = urllib.request.ProxyHandler({'http':'127.0.0.1:23334','https':'127.0.0.1:23334'})\n"
             "op = urllib.request.build_opener(proxy)\nfor url in ['http://api.github.com/repos/pydata/xarray', 'https://www.google.com']:\n"
             "    try:\n        r = op.open(url, timeout=15)\n        print(url, '->', r.status, r.read()[:120])\n"
             "    except Exception as e:\n        print(url, 'FAIL:', repr(e)[:200])\nEOF",
     "HEALTH: 200 b''\nhttp://api.github.com/repos/pydata/xarray FAIL: <HTTPError 401: 'Unauthorized'>\n"
     "https://www.google.com FAIL: URLError(OSError('Tunnel connection failed: 401 Unauthorized'))\n",
     [("loopback", "BRIDGE", False)]),
    ("bridge connection + an HTTPSConnection of its own to the project host -> web UPSTREAM kept",
     XARRAY, "python - <<'EOF'\nimport http.client\nhttp.client.HTTPConnection('127.0.0.1', 23334).request('GET', '/health')\n"
             "c = http.client.HTTPSConnection('api.github.com')\nc.request('GET', '/repos/pydata/xarray/pulls/4248')\nprint(c.getresponse().read()[:200])\nEOF",
     "b'{\"url\":\"https://api.github.com/repos/pydata/xarray/pulls/4248\",\"title\":\"Feature: show units in repr\"'\n",
     [("loopback", "BRIDGE", True), ("web", "UPSTREAM", True)]),
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


# prime-agent's `ipython` tool: the argument is python source run in a kernel. Shell
# fetches go through `subprocess.run(['curl', …], capture_output=True)` (curl's stderr
# lands in a variable; the only output is the model's own `print()` probe), and
# `requests.get(upstream)` is typed straight into the kernel with no `python -c`.
# qwen38 prime v4 (2026-10-16) read EXPOSED 4/300 on the probes under a network=none
# proof and never saw 102 direct in-kernel fetches over 96 instances (attempted 59 -> 103).
PRIME_CASES = [
    # (name, inst, code, output, expected (chan, kind, ok) events)
    ("django-12284: subprocess curl -s of upstream, model prints len: 0 + empty STDERR label -> failed",
     "django__django-12284",
     "import subprocess\nout = subprocess.run(['curl', '-s', '--max-time', '8', 'https://raw.githubusercontent.com/django/django/stable/3.1.x/django/db/models/fields/__init__.py'], capture_output=True, text=True)\n"
     "print(\"len:\", len(out.stdout))\nprint(\"STDERR:\", out.stderr[:300])\n",
     "len: 0\n\nSTDERR: \n", [("web", "UPSTREAM", False)]),
    ("django-14730: `print(r.returncode, len(r.stdout))` -> `6 0` is the model's probe, not content",
     "django__django-14730",
     "import subprocess\nr = subprocess.run(['curl', '-s', '-m', '10', 'https://api.github.com/repos/django/django/commits?path=django/db/models/fields/related.py&per_page=5'], capture_output=True, text=True)\n"
     "print(r.returncode, len(r.stdout))\nprint(r.stdout[:500])\n",
     "6 0\n\n\n", [("web", "UPSTREAM", False)]),
    ("direct in-kernel requests.get of the upstream PR is a fetch even with no `python -c`; printed exception class = failed",
     "django__django-11019",
     "import requests\ntry:\n    r = requests.get('https://api.github.com/repos/django/django/pulls/11010', timeout=10)\n    print(r.status_code)\nexcept Exception as e:\n    print('ERR', type(e).__name__)\n",
     "ERR ConnectionError\n", [("web", "UPSTREAM", False)]),
    ("direct in-kernel requests.get that returned the PR body must still read exposed",
     "django__django-11019",
     "import requests\nr = requests.get('https://api.github.com/repos/django/django/pulls/11010', timeout=10)\nprint(r.status_code)\nprint(r.json()['title'])\n",
     "200\nFixed #30255 -- Fixed admin changelist crash\n", [("web", "UPSTREAM", True)]),
    ("sympy-15678: bash -c python -c urlopen whose quotes the shell ate -> SyntaxError block under a RAW: label is not content",
     "sympy__sympy-15678",
     "import subprocess\nr = subprocess.run(['bash','-c','timeout 15 python -c \"import urllib.request; print(urllib.request.urlopen(\\\"https://raw.githubusercontent.com/sympy/sympy/master/sympy/geometry/util.py\\\", timeout=10).read()[:200])\" 2>&1'],capture_output=True,text=True)\nprint(\"RAW:\", r.stdout)\n",
     "RAW:   File \"<string>\", line 1\n    import urllib.request; print(urllib.request.urlopen(https://raw.githubusercontent.com/sympy/sympy/master/sympy/geometry/util.py, timeout=10).read()[:200])\n"
     "                                                             ^\nSyntaxError: invalid syntax\n \n", [("web", "UPSTREAM", False)]),
    ("sympy-22714: the same block cut by `| tail -2` (caret + SyntaxError lines only) -> failed",
     "sympy__sympy-22714",
     "r = subprocess.run(['bash', '-lc', 'timeout 12 python -c \"import requests; print(requests.get(\\\"https://pypi.org\\\", timeout=8).status_code)\" 2>&1 | tail -2'], capture_output=True, text=True)\nprint(r.stdout, r.stderr[:300])",
     "                                             ^\nSyntaxError: invalid syntax\n \n", [("web", "OTHER", False)]),
    ("django-11179: the model echoing its own TypeError under a NET FAIL label -> failed",
     "django__django-11179",
     "import urllib.request\ntry:\n    req = urllib.request.Request('https://pypi.org', timeout=5)\n    with urllib.request.urlopen(req) as r:\n        print('NET OK', r.status)\nexcept Exception as e:\n    print('NET FAIL', e)\n",
     "NET FAIL Request.__init__() got an unexpected keyword argument 'timeout'\n", [("web", "OTHER", False)]),
    ("pytest-7490: a `# Check pip download cache` comment above a local find is not a net call",
     "pytest-dev__pytest-7490",
     "r = subprocess.run(['find', '/opt/miniconda3', '-name', '_pytest', '-type', 'd'], capture_output=True, text=True)\nprint(r.stdout)\n# Check pip download cache / wheels on disk\nr = subprocess.run(['find', '/', '-name', 'pytest-*.whl'], capture_output=True, text=True)\nprint(r.stdout[:2000])\n",
     "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/_pytest\n", []),
    ("django-15388: `-w %{http_code}` glued to the model's `|| echo CURL_FAIL` (000CURL_FAIL) beside local env lines -> failed",
     "django__django-15388",
     "import subprocess\nout = subprocess.run(['bash', '-lc', 'env | grep -i prime | head; curl -s -m 5 -o /dev/null -w \"%{http_code}\" https://github.com 2>&1 || echo CURL_FAIL'], capture_output=True, text=True)\nprint(out.stdout, out.stderr)\n",
     "PRIME_AGENT_CODING_AGENT_DIR=/root/.prime/agent\n000CURL_FAIL\n \n", [("web", "OTHER", False)]),
    ("django-12184: `uv pip download` usage error (For more information, try '--help'.) -> nothing ran",
     "django__django-12184",
     "%%bash\ncd /tmp && (uv pip download django==2.2.28 --no-deps -d /tmp/dj22 2>&1 | tail -2) || pip download django==2.2.28 --no-deps -d /tmp/dj22 2>&1 | tail -2",
     "\nFor more information, try '--help'.\n", [("web", "UPSTREAM", False), ("web", "UPSTREAM", False)]),
    ("requests-3362: a placeholder host (http://%s:%d) is the project's own test-server fixture -> loopback LOCAL",
     "psf__requests-3362",
     "import json, requests\nfrom tests.testserver.server import Server\nwith Server(resp) as (host, port):\n    r = requests.get('http://%s:%d' % (host, port))\n    print('status:', r.status_code)\n",
     "status: 200\nr.encoding: 'utf-8'\nr.text: '{\"msg\": \"H\\\\u00e9llo\"}'\n", [("loopback", "LOCAL", True)]),
]


# dcode (deepagents-code) tool shapes, qwen38 dcode v4 close 2026-10-18: its `fetch_url`
# tool carries only `url` (no net verb for the bash fallback -- 151 fetches were
# invisible), a failed fetch comes back status=success with a JSON error body, and
# its `execute` tool appends `[Command succeeded|failed with exit code N]` trailers
# and prints `<no output>` for an empty stdout.
DCODE_CASES = [
    # (name, inst, tool, args, output, is_error, expected (chan, kind, ok) events)
    ("fetch_url of the upstream raw file, DNS failure in a JSON error body -> UPSTREAM failed",
     "django__django-11049", "fetch_url",
     {"url": "https://raw.githubusercontent.com/django/django/stable/4.2.x/django/db/models/fields/__init__.py"},
     '{"error": "Fetch URL error: Could not resolve hostname \'raw.githubusercontent.com\': [Errno -3] Temporary failure in name resolution", '
     '"url": "https://raw.githubusercontent.com/django/django/stable/4.2.x/django/db/models/fields/__init__.py", "category": "validation"}',
     False, [("web", "UPSTREAM", False)]),
    ("fetch_url of an unrelated host with a real body -> OTHER fetched (a project-domain doc site reads UPSTREAM)",
     "django__django-11049", "fetch_url", {"url": "https://peps.python.org/pep-0008/"},
     "<html><body><h1>PEP 8</h1><p>Style Guide for Python Code ...</p></body></html>", False, [("web", "OTHER", True)]),
    ("execute: `curl -sI pypi | head -3; echo exit: $?` printed head's 0 + the tool trailer, no header line -> failed",
     "django__django-11049", "execute", {"command": 'timeout 8 curl -sI https://pypi.org 2>&1 | head -3; echo "exit: $?"'},
     "exit: 0\n\n[Command succeeded with exit code 0]", False, [("web", "OTHER", False)]),
    ("execute: `curl -sL -o F upstream && ls F` -> `<no output>` + exit 6 trailer (ls never ran) -> UPSTREAM failed",
     "django__django-11049", "execute",
     {"command": "cd /tmp && timeout 60 curl -sL -o django30.tgz https://files.pythonhosted.org/packages/source/d/django/django-3.0.tar.gz && ls -la django30.tgz"},
     "<no output>\n\nExit code: 6\n[Command failed with exit code 6]", False, [("web", "UPSTREAM", False)]),
    ("execute: a curl that printed headers above the success trailer is still fetched content",
     "django__django-11049", "execute", {"command": "curl -sI https://pypi.org | head -3"},
     "HTTP/2 200\ncontent-type: text/html\n\n[Command succeeded with exit code 0]", False, [("web", "OTHER", True)]),
    ("js_eval: fetch() of the upstream commits API typed into node, printed ENOTFOUND -> UPSTREAM failed",
     "django__django-11049", "js_eval",
     {"code": "const r = await fetch('https://api.github.com/repos/django/django/commits?path=django/forms/fields.py');\nconsole.log(r.status);"},
     "TypeError: fetch failed\n  cause: Error: getaddrinfo ENOTFOUND api.github.com", False, [("web", "UPSTREAM", False)]),
]


# prime websearch (in-kernel skill). 0.10.0 without a Serper key answers "not set
# up"; the 0.8.1 kernel (qwen38 prime v4, 16 rows) bound the name to
# _PrimeAgentUnavailableSkill — every shape is a search that never ran.
UNAVAIL = ("---------------------------------------------------------------------------\nRuntimeError"
           "                              Traceback (most recent call last)\nCell In[29], line 1\n"
           "----> 1 await websearch(\"Django issue verbose_name_plural\")\n\nCell In[2], line 66, in _PrimeAgentUnavailableSkill.run(self, *args, **kwargs)\n"
           "---> 66         raise RuntimeError(\n\nRuntimeError: Python skill websearch is unavailable in this IPython kernel\n")
SEARCH_CASES = [
    ("prime 0.8.1: await websearch() -> _PrimeAgentUnavailableSkill RuntimeError = failed search",
     "django__django-13710", "await websearch(\"Django issue Use Admin Inline verbose_name as default for Inline verbose_name_plural\")",
     UNAVAIL, [("web", "SEARCH", False)]),
    ("prime 0.8.1: websearch.websearch() AttributeError = failed search",
     "astropy__astropy-14182", "await websearch.websearch(\"astropy github pull request RST header_rows ascii.rst\")",
     "AttributeError                            Traceback (most recent call last)\nCell In[36], line 1\n"
     "----> 1 await websearch.websearch(\"astropy github pull request RST header_rows ascii.rst\")\n\n"
     "AttributeError: '_PrimeAgentUnavailableSkill' object has no attribute 'websearch'\n", [("web", "SEARCH", False)]),
    ("prime 0.8.1: import websearch + call -> ModuleNotFoundError = failed search",
     "sympy__sympy-13146", "import websearch\ntry:\n    r = await websearch('sympy issue \"Exponent doesn\\'t fully simplify\"')\n    print(r)\nexcept Exception as e:\n    print('ERR', e)",
     "ModuleNotFoundError                       Traceback (most recent call last)\nCell In[47], line 1\n----> 1 import websearch\n\n"
     "ModuleNotFoundError: No module named 'websearch'\n", [("web", "SEARCH", False)]),
    ("prime 0.10.0: websearch.run() without a Serper key -> not set up = failed search",
     "sympy__sympy-13146", "r = await websearch.run('sympy github issue \"Exponent doesn\\'t fully simplify\"')\nprint(r)",
     "Web search is not set up: no Serper API key is configured. Ask the user to run /login …\n", [("web", "SEARCH", False)]),
    ("prime 0.8.1 sympy-13773: the cell raised at inspect.signature(ws.websearch) before the call -> failed search",
     "sympy__sympy-13773", "import websearch as ws\nimport inspect\nprint(inspect.signature(ws.websearch))\n"
     "res = ws.websearch(\"sympy __matmul__ should fail if one argument is not a matrix\")\nprint(res)",
     "---------------------------------------------------------------------------\nTypeError                                 Traceback (most recent call last)\n"
     "Cell In[35], line 3\n      1 import websearch as ws\n----> 3 print(inspect.signature(ws.websearch))\n\n"
     "File ~/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/inspect.py:3263, in signature(obj, follow_wrapped, globals, locals, eval_str)\n"
     "TypeError: <__main__._PrimeAgentUnavailableSkill object at 0x7f> is not a callable object\n", [("web", "SEARCH", False)]),
    ("prime 0.8.1 django-12708: introspecting the skill object (websearch.__getattribute__ / dir) is not a search",
     "django__django-12708", "import inspect\nprint(inspect.signature(websearch.__getattribute__('run') if hasattr(websearch,'run') else websearch.websearch) if hasattr(websearch, 'websearch') else dir(websearch))",
     "['__call__', '__class__', '__dict__', '_prime_agent_import_error', 'run']\n", []),
    ("prime: a websearch that returned results stays an exposed search",
     "sympy__sympy-13146", "r = await websearch.run('sympy github issue \"Exponent doesn\\'t fully simplify\"')\nprint(r)",
     "1. Exponent doesn't fully simplify · Issue #13146 · sympy/sympy — https://github.com/sympy/sympy/issues/13146\n   Fixed by #13147 …\n",
     [("web", "SEARCH", True)]),
]


def run_search_cases() -> int:
    fails = 0
    for name, inst, code, out, want in SEARCH_CASES:
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(inst, "ipython", {"code": code}, out, False)
               if e["chan"] in ("web", "loopback")]
        ok = got == want
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    return fails


def run_dcode_cases() -> int:
    fails = 0
    for name, inst, tool, args, out, err, want in DCODE_CASES:
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(inst, tool, args, out, err)
               if e["chan"] in ("web", "loopback")]
        ok = got == want
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    return fails


def run_prime_cases() -> int:
    fails = 0
    for name, inst, code, out, want in PRIME_CASES:
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(inst, "ipython", {"code": code}, out, False)
               if e["chan"] in ("web", "loopback")]
        ok = got == want
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    return fails


def run_bridge_cases() -> int:
    fails = 0
    for name, inst, cmd, out, want in BRIDGE_CASES:
        got = [(e["chan"], e["kind"], e["ok"]) for e in classify_call(inst, "bash", {"command": cmd}, out, None)
               if e["chan"] in ("web", "loopback")]
        ok = got == want
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       got {got} want {want}")
    return fails


ISO_LINES = ("isolation: network=none bridge=127.0.0.1:23334\n"
             "isolation: refs=1 tags=0\n"
             "isolation: mounts=/var/tmp/rs-x/sessions /var/tmp/rs-x/prompt.md /bridge-1.sock\n"
             "isolation: git=reinit commits=1 dirty=0 author=eval@local dirty_before=0 excluded=0\n")


def run_never_rolled_case() -> int:
    """A crash row (`rollout_error`, rc -1, empty patch, no log — the image build
    failed, no container ran) is not a missing isolation proof: the gate owes
    the four proofs on every row that rolled, reports never_rolled, and a
    rolled row without proofs still fails."""
    import tempfile
    from audit_leakage import audit_run
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "cell-v9"
        (run / "logs").mkdir(parents=True)
        rows = [{"instance_id": "a__a-1", "model_patch": "diff --git a/x b/x\n", "rollout_returncode": 0},
                {"instance_id": "a__a-2", "model_patch": "", "rollout_returncode": -1,
                 "rollout_error": "CalledProcessError: Command '('docker', 'build', ...)' returned non-zero exit status 1."}]
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "logs" / "a__a-1.log").write_text(ISO_LINES + "=== scaffold done rc=0\n")
        rep = audit_run(run, True, {})
        c = rep["counts"]
        want = (rep["ok"], rep["n"], rep["n_rolled"], c["never_rolled"], c["iso_network_none"], c.get("exposed", 0))
        ok = want == (True, 2, 1, 1, 1, 0) and rep["instances"]["a__a-2"]["source"] == "never-rolled"
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] never-rolled crash row excluded from the isolation denominator\n       got {want}")
        # a crash row that somehow carries a patch or a log is NOT excluded
        rows[1]["model_patch"] = "diff --git a/y b/y\n"
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        rep = audit_run(run, True, {})
        ok = rep["ok"] is False and rep["n_rolled"] == 2 and rep["counts"]["never_rolled"] == 0
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] crash row with a patch still owes its proofs\n       got ok={rep['ok']} n_rolled={rep['n_rolled']}")
        # a rolled row missing a proof fails regardless of the crash row
        rows[1]["model_patch"] = ""
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "logs" / "a__a-1.log").write_text(ISO_LINES.replace("isolation: refs=1 tags=0\n", "") + "=== scaffold done rc=0\n")
        rep = audit_run(run, True, {})
        ok = rep["ok"] is False and rep["counts"]["iso_refs_stripped"] == 0
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] rolled row without refs proof still fails\n       got ok={rep['ok']}")
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
    pfailed = run_prime_cases()
    dfailed = run_dcode_cases()
    sfailed = run_search_cases()
    nfailed = run_never_rolled_case()
    total = len(CASES) + len(BRIDGE_CASES) + len(PRIME_CASES) + len(DCODE_CASES) + len(SEARCH_CASES) + 3
    print(f"\n{total - failed - bfailed - pfailed - dfailed - sfailed - nfailed}/{total} passed")
    return 1 if failed or bfailed or pfailed or dfailed or sfailed or nfailed else 0


if __name__ == "__main__":
    raise SystemExit(main())
