#!/usr/bin/env python3
"""Audit a SWE-bench rollout cell for answer leakage and prove its isolation.

Every bake-off cell rolled before 2026-09-19 ran its container with
`--network=host`: the scaffolds' web tools (opencode `webfetch`, pi/prime
`webfetch` + `websearch`) and bash (`curl`, `gh`, `pip download`) reached the
internet, so the model could — and did — fetch the upstream fix: 54 % of qwen38
opencode instances touched the project's own sites, 21 % fetched their own PR
diff (`pull/<num>.diff`). R9700 found 45–61 % on their lanes first. The git
channel is closed on our side: the official `sweb.eval` images hold no ref past
the base commit and no unreachable object (django/sympy checked 2026-09-19),
so a `git log --all` finds nothing — attempts are reported, not counted.

docker_rollout.py now runs `--network none` (loopback bridge to the server
only), strips refs to HEAD and snapshots each scaffold's session store to
`<run>/sessions/<iid>/`. This script closes the loop at lane close:

  * proves every instance ran isolated (`isolation: network=none`,
    `isolation: refs=1 tags=0`, `isolation: git=reinit commits=1 dirty=N
    dirty_before=N` (the tree's history is one `eval@local` commit — the
    official image's `SWE-bench` HEAD commit is gone — and the re-added tracked
    set left the tree exactly as dirty as the image shipped it: 0 everywhere
    but psf__requests-863, whose image carries an untracked `build/`; since
    2026-10-08 the prelude writes that baseline to `.git/info/exclude` and
    reports `excluded=K`, so the check is `dirty + excluded == dirty_before`;
    logs from before the `dirty_before=` field fall back to
    image_untracked_baseline.json) and
    an `isolation: mounts=` line whose bind-source paths name neither the
    benchmark nor the instance; `--require-isolation`)
  * scans the transcript for web / network / git-peek attempts and whether
    they returned content (a fetch under `--network none` fails — an attempt
    is a model behaviour note, a success is an infra alarm)
  * reports the added-line overlap between each model_patch and the gold
    patch, exposed vs isolated, so copying is measured rather than inferred

Transcript sources per instance (first that exists wins):
  sessions/<iid>/.local/share/opencode/opencode.db     opencode (sqlite `part`)
  sessions/<iid>/.local/share/opencode/storage/**.json opencode (older layout)
  sessions/<iid>/.pi/agent/sessions/**/*.jsonl         little-coder / -rtk (pi)
  sessions/<iid>/.prime/agent/sessions/*.jsonl         prime (pi format)
  sessions/<iid>/.deepagents/.state/sessions.db        dcode (langgraph msgpack)
  logs/<iid>.log                                       opencode `tool_use` events
                                                       (the only v2 transcript)

Classification per instance:
  web_UPSTREAM  fetch (tool or bash) of the project's own repo / tracker /
                docs, or `gh pr|api`
  web_SEARCH    web search (results carry PR titles + snippets)
  web_OTHER     fetch of an unrelated site
  git_READ / git_LIST   command that would read / enumerate refs beyond HEAD
  loopback_SELF / _BRIDGE / _LOCAL   traffic to 127.0.0.1: a self-dispatch — the
                model's own `POST /v1/chat/completions` through the harness
                bridge (127.0.0.1:23334 = the inference server, nothing else)
                asking the served model to recall the fix —, a health / proxy
                probe of the bridge, or the model's own local test server.
                Reported, never a web fetch: no gold sits behind any of them
                (sympy-13773, 2026-10-11)
  exposed       web_UPSTREAM or web_SEARCH that RETURNED content (unknown
                outcome counts as exposed — conservative)

Usage:
  audit_leakage.py --run evals/swebench/runs/<cell> [--require-isolation] [--json out]
  exit 0 = clean, 1 = exposed instances (or isolation proof missing), 2 = usage
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

# --- classifiers (web channel ported from R9700 audit_git_peek.py) -----------
# dcode (deepagents-code) names its fetch tool `fetch_url` and its args carry only
# `url` — no net verb for NET_BASH_RE, so the bash-like fallback dropped all 151 of
# qwen38 dcode v4's fetches (2026-10-18). Add any new scaffold's fetch tool here.
WEB_FETCH_TOOLS = {"webfetch", "web_fetch", "fetch", "fetch_url", "codesearch"}
WEB_SEARCH_TOOLS = {"websearch", "web_search", "web-search", "search"}
# Tools whose argument IS python source run in a kernel (prime-agent `ipython`). The
# whole `code` is inline python: `requests.get(upstream)` typed straight into the
# kernel has no `python -c` around it, and qwen38 prime v4 (2026-10-16) made 102 such
# calls over 96 instances that the `python -c` / heredoc rule never saw (attempted
# read 59/300; the shell-shaped ones through `subprocess.run(['curl', …])` were caught).
PY_CODE_TOOLS = {"ipython", "python", "python_repl", "execute_python", "run_python",
                 "execute_code", "code_interpreter", "jupyter"}
# Same for a JS kernel (dcode `js_eval`): `fetch(` / `https.get(` typed straight into node.
JS_CODE_TOOLS = {"js_eval", "node", "javascript", "execute_js", "run_js"}
# omp's single `eval` REPL tool carries the kernel in its `language` arg ('py' | 'js');
# classify_call routes it to the python / JS rule above by that value.
EVAL_LANG_TOOLS = {"eval"}
# Tools whose `path` doubles as a URL: omp's `read` fetches http(s) targets (read.ts
# -> fetch.ts), so a URL path is a web fetch, not a file read.
URL_PATH_TOOLS = {"read"}
# Tools that wrap `gh` under their own schema (omp `github`: op + repo/pr/query, no
# command text for NET_BASH_RE) — every op is a GitHub round-trip.
GH_WRAPPER_TOOLS = {"github"}
JS_FETCH_RE = re.compile(r"\b(?:fetch|https?\.(?:get|request)|axios(?:\.(?:get|post))?|got|undici\.request)\(")
URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+", re.I)
NET_BASH_RE = re.compile(
    r"\b(curl|wget|gh\s+(api|pr|issue)|pip3?\s+(download|install)(?!\s+-e)"
    r"|python\S*\s+-m\s+pip\s+(download|install)(?!\s+-e)|git\s+(clone|fetch|pull|ls-remote)\b"
    r"|npm\s+(install|view)|apt(-get)?\s+install)\b", re.I)
# Inline python (`python -c`, `python - <<EOF`, `python <<EOF`) that opens a URL. The
# command text is the unit: a heredoc's URL sits lines away from its urlopen().
# v2-netopen matplotlib-25442 fetched the upstream compare API this way and was only
# caught because its User-Agent header said "curl".
PY_EXEC_RE = re.compile(r"\bpython\S*\s+(?:-c\b|-?\s*<<|-\s*$)", re.M)
PY_FETCH_RE = re.compile(
    r"\b(?:urlopen|urlretrieve|requests\.(?:get|post|head|request)|httpx\.(?:get|post|Client|AsyncClient)"
    r"|aiohttp\.ClientSession|http\.client\.HTTPS?Connection|socket\.create_connection)\(")
# prime-agent 0.10.0 moved web search INTO the kernel: `websearch.run(query)` (the
# module is callable too) is a Python skill, not a model-facing tool, so it only
# shows up inside `ipython` code. It calls Serper over httpx when a key is
# configured; the rollout configures none, so the skill returns its "not set up"
# notice without touching the network — that output is a failed search, not content.
PY_SEARCH_RE = re.compile(r"\bwebsearch(?:\.(?:run|websearch))?\(")
# A websearch call that never reached the network: 0.10.0 without a Serper key
# answers "not set up"; the 0.8.1 kernel bound the name to
# `_PrimeAgentUnavailableSkill` (RuntimeError "Python skill websearch is
# unavailable in this IPython kernel" / AttributeError on `.websearch`), and
# `import websearch` / a shell `websearch` never existed. The qwen38 prime v4 cell
# (0.8.1, 297 rows) carried 16 of these as SEARCH ok=True — a traceback reads as
# content — once PY_SEARCH_RE arrived for 0.10.0 (2026-10-21, first full audit of
# that cell after the regex landed); every one is a failed search.
PY_SEARCH_FAIL_RE = re.compile(
    r"no Serper API key is configured|Python skill websearch is unavailable|_PrimeAgentUnavailableSkill"
    r"|_prime_agent_import_error|No module named ['\"]websearch['\"]|name ['\"]websearch['\"] is not defined"
    r"|websearch: command not found|command not found: websearch|cannot import name ['\"]websearch['\"]", re.I)
# an IPython cell that printed nothing before raising (output opens with the
# traceback rule / header and ends on the exception line) returned no results
PY_CELL_RAISED_RE = re.compile(r"\A\s*(?:-{20,}\s*\n)?\s*\w+\s+Traceback \(most recent call last\)[\s\S]*\n\s*[\w.]+(?:Error|Exception)\b[^\n]*\s*\Z")
NET_FAIL_RE = re.compile(
    r"Could not resolve host|Temporary failure in name resolution|Network is unreachable"
    r"|ENOTFOUND|ECONNREFUSED|EAI_AGAIN|Name or service not known|No address associated"
    r"|fetch failed|Failed to fetch|Connection refused|NewConnectionError|Max retries exceeded"
    r"|URLError|gaierror|\[Errno -?\d+\]"
    r"|Could not fetch URL|No matching distribution|network error|unable to access"
    # pip swallows the DNS failure and prints an empty version list; opencode's
    # webfetch surfaces a blocked socket as "Transport error" (R9700 `1ea2167`)
    r"|\(from versions: none\)|Transport error"
    # a kernel-side fetch wrapped in `try/except … print(type(e).__name__)` prints only
    # the exception class (prime ipython, 2026-10-16: `ERR ConnectionError`,
    # `requests.exceptions.ConnectionError: …`, `ConnectError`, `NameResolutionError`);
    # anchored to a line start (after an optional label) so a fetched source file that
    # merely mentions the class mid-line is not read as a failure
    r"|^\s*(?:(?:ERR(?:OR)?|EXC(?:EPTION)?|FAIL(?:ED|URE)?|Exception|caught|net(?:work)?)\b\s*[:=-]?\s*)?"
    r"(?:[\w.]*\.)?(?:ConnectionError|ConnectError|NameResolutionError|ConnectTimeout|ReadTimeout"
    r"|HTTPError\b[^\n]*\b(?:503|502|504)|SSLError|ProxyError|InvalidURL|URLError|gaierror)\b"
    # the fetch binary itself is absent from the sandbox image -- nothing ran
    # (qwen38 opencode v4 django-16408: `gh pr view … 2>&1 | head` -> "gh: command not found")
    r"|\b(?:gh|curl|wget|aria2c|pip3?|npm): command not found|command not found: (?:gh|curl|wget|aria2c|pip3?|npm)\b", re.I | re.M)
# A Python traceback whose frames pass through the network stack (urllib / http.client /
# socket / requests / urllib3 / httpx / aiohttp) is a fetch that raised before returning
# content. The exception line itself is often cut off by the model's `| head -5`
# (django-13964, v3: `python -c "urlopen(...)" 2>&1 | head -5` ended at h.request()).
PY_NET_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):(?:\n.*){0,24}?\n\s*File \"[^\"\n]*"
    r"(?:urllib/request\.py|http/client\.py|/socket\.py|urllib3/|requests/(?:adapters|sessions|api)\.py"
    r"|httpx/|aiohttp/)[^\"\n]*\"")
# Output lines that are the model's own scaffolding around a silenced fetch (`curl -s`,
# `wget -q`) rather than fetched content: `echo "=== $tag ==="` banners, `EXIT: $?` /
# `rc=…` trailers, curl's `-w %{http_code}` printing 000 (no connection), opencode's
# `(no output)` placeholder, bare separators. Under `--network none` every one of the
# qwen38 opencode v3 lane's five "ok=True" bash fetches (2026-09-22) was one of these.
# pi's ShellSession tool appends its own status trailer to every result --
# `[exit=N cwd=/testbed timed_out=false backend=subprocess]` (also `output_truncated=true`,
# `timed_out=true`) -- and the model's `echo curl_exit=$?` prefixes the exit marker:
# qwen38 little-coder v4 (2026-10-07) read EXPOSED 4/300 on trailer-only results of
# silenced curls (`curl -s … | sed …; echo EXIT=$?` echoes sed's 0; `curl -s -o F …;
# echo exit=$?` printed `exit=6` with no F), every one under a network=none proof.
# A kernel tool (prime `ipython`) runs the fetch through `subprocess.run(…,
# capture_output=True)` and prints its own probe of the result: `len: 0`, `6 0`
# (`print(r.returncode, len(r.stdout))`), `rc=6 len=0`, an empty `STDERR:` label.
# curl's stderr went into a Python variable, so nothing shell-shaped survives; qwen38
# prime v4 (2026-10-16) read EXPOSED 4/300 on exactly these under a network=none proof
# (django-12284 `len: 0`, django-14730 `6 0`). A line made only of label:number
# tokens, of one or two small integers, or of a bare upper-case label is the model
# talking to itself; a CLI usage error (`uv pip download` -> `For more information,
# try '--help'.`) is the tool refusing to run. dcode's `execute` tool appends its own
# trailer -- `[Command succeeded with exit code 0]` / `Exit code: 6` + `[Command failed
# with exit code 6]` -- and prints `<no output>` for an empty stdout: qwen38 dcode v4
# (2026-10-18) read EXPOSED 1/300 on `curl -sI https://pypi.org | head -3; echo "exit: $?"`
# (head's 0, no header line) and `curl -sL -o F … && ls -la F` (exit 6, ls never ran).
SCAFFOLD_LINE_RE = re.compile(
    r"^\s*(?:=+[^=\n]*=+|-{2,}|\*+|\(no output\)|<no output>|\[Command (?:succeeded|failed) with exit code \d+\]"
    r"|0{3}|(?:EXIT|RC|STATUS|CODE|HTTP(?:_CODE)?)"
    r"(?: CODE)?\s*[:=]?\s*\d+|(?:\w+[_-])?exit(?:[ _-]?code)?\s*[:=]?\s*\d+|curl: \(\d+\).*"
    r"|\[exit=\d+ cwd=[^\]\n]*\]"
    r"|(?:(?:rc|ret(?:code|urncode)?|exit(?:code)?|code|status|len(?:gth)?|size|bytes|n|count|lines)"
    r"\s*[:=]\s*-?\d+\s*[,;]?\s*){1,4}"
    r"|-?\d{1,3}(?:\s+\d{1,7})?"
    r"|[A-Z][A-Z_ ]{1,24}:"
    r"|For more information, try '--help'\.|error: (?:unrecognized subcommand|unexpected argument|no such command)\b.*"
    r"|usage: \S.*)\s*$", re.I)
# A `python -c` / `bash -c` snippet that died parsing never ran its fetch: the
# traceback echo (`File "<string>", line 1` + the source line + caret + `SyntaxError:`)
# is not fetched content even when the model prefixed it with a `RAW:` label
# (prime v4 sympy-15678: the shell ate the quotes around the urlopen URL).
PY_SYNTAX_BLOCK_RE = re.compile(
    r"^[^\n]*(?:File \"<(?:string|stdin|ipython[^\">]*)>\"|Cell In\[\d+\]), line \d+\n(?:[^\n]*\n){0,8}?"
    r"\s*(?:Syntax|Indentation|Tab)Error:[^\n]*", re.M)
# The model echoing its own exception under a FAIL/ERR label (`NET FAIL Request.__init__()
# got an unexpected keyword argument 'timeout'`, `FAIL: ModuleNotFoundError No module named
# 'requests'`) is the kernel talking; so are the caret + `SyntaxError:` lines that survive a
# `| tail -2` (prime v4 django-11179, sympy-22714).
MODEL_EXC_LINE_RE = re.compile(
    r"^\s*(?:(?:[A-Z][A-Z_]*\s+)?(?:FAIL(?:ED|URE)?|ERR(?:OR)?|EXC(?:EPTION)?)\s*[:=-]?\s*"
    r"[^\n]*?(?:\b\w*(?:Error|Exception|Timeout)\b|got an unexpected|No module named|is not defined|Traceback"
    r"|has no attribute|not supported|timed out)[^\n]*"
    r"|\^+|(?:Syntax|Indentation|Tab)Error:.*)\s*$")


# `curl -o F` / `wget -O F` targets named in a command; a following `ls -l` that lists
# that file at 0 bytes is the fetch failing (curl -s swallows "(6) Could not resolve",
# the model's `echo "exit: $?"` trailer is neutral scaffolding) -- qwen38 opencode v4
# sympy-18698: `curl -sL -o polytools_master.py …; echo "exit: $?"; ls -la …` printed
# `exit: 6` + `-rw-r--r-- 1 root root 0 Sep 30 20:14 polytools_master.py`.
DL_TARGET_RE = re.compile(r"(?:^|\s)(?:-[a-zA-Z]*[oO]|--output(?:-document)?)[\s=]+([^\s;|&]+)")
# A local tool complaining that a file is missing is the sandbox talking, never the
# network answering: `diff: X: No such file or directory`, `ls: cannot access 'X': …`.
# Under `--network none` a `curl -s -o X … && diff X …` chain prints only these
# (qwen38 opencode-dcp v4 astropy-14995, 2026-10-04: `-o` target never created, so
# the 0-byte `ls -l` rule below had nothing to match).
LOCAL_ERR_LINE_RE = re.compile(
    r"^\s*[\w./+-]+: (?:cannot (?:access|open|stat|read) )?'?[^\n]*?'?: No such file or directory\s*$", re.I)
# curl's own `-w %{http_code}` reporting 000 is a definitive "no connection" even when a
# compound command's other (local) segments printed real output around it
# (django-11999: `pip show django | head -3; python -c …; curl -s -o /dev/null -w "%{http_code}" … | tail -1` -> `000`).
HTTP_CODE_W_RE = re.compile(r"%\{http_code\}")
# `-w "%{http_code}"` prints no newline, so `… || echo CURL_FAIL` glues on: `000CURL_FAIL`
HTTP_000_RE = re.compile(r"(?:^|\s)000(?=\s|$|[A-Z_]{2,})", re.M)
# A fetch piped into `python … json.load(sys.stdin)` that dies at char 0 received an
# empty body — the network returned nothing (matplotlib-25311). Only honoured when
# the traceback is the sole non-scaffold output, so a fetch that printed anything
# else still reads as content.
PIPED_PY_RE = re.compile(r"\|\s*python\S*\b")
JSON_EMPTY_RE = re.compile(r"JSONDecodeError: Expecting value: line 1 column 1 \(char 0\)")
# `git log --oneline` lines of the sandbox's own (re-initialised) history are local
# output even when they share a compound command with a silent fetch
# (pytest-5103: `git remote -v; git log --oneline --all | head; curl -s … | head` ->
# `9c17c3d base`). Not applied when the same command also fetches through git
# (`git fetch/pull/clone/ls-remote`): there the log lines CAN be the network answering.
GIT_LOG_CMD_RE = re.compile(r"\bgit\b[^;|&\n]*\blog\b")
GIT_NET_CMD_RE = re.compile(r"\bgit\b[^;|&\n]*\b(?:fetch|pull|clone|ls-remote|remote\s+add|submodule)\b")
GIT_ONELINE_RE = re.compile(r"^[0-9a-f]{7,40}(?: \([^)]*\))? \S.*$")
# The harness bridge: inside the sandbox 127.0.0.1:23334 is the loopback half of
# net_bridge.py, which reaches the inference server and nothing else (the server is
# not a proxy -- an absolute-URI GET through it answers 401/404 from uvicorn). A
# command whose only network target is loopback is the model talking to the server:
# a `/health`, `/openapi.json` or absolute-URI probe, or a self-dispatch -- its own
# `POST /v1/chat/completions` asking the served model (itself) to recall the
# upstream PR. qwen38 little-coder v4 sympy-13773 (pi 0.83, 2026-10-11) learned the
# port from `ps aux` (the bridge command sits in the instance script's argv) and the
# key from pi's models.json, probed the bridge as a proxy (uvicorn 401 read as
# UPSTREAM ok=True -> a false EXPOSED at the gate), then made six 16K-token
# self-calls; the rtk lane's pytest-5103 made eight. Neither channel carries gold,
# so loopback traffic is reported on its own channel, never as a web fetch (kinds:
# SELF = a completions call to the bridge port, BRIDGE = any other call to it,
# LOCAL = another loopback port — the model's own dev/test server); a command
# that ALSO names a remote URL keeps its web event, with the server's own response
# lines (status, uvicorn headers, 401/404 bodies, completion JSON) not read as
# fetched content.
# a placeholder host (`http://%s:%d`, `http://{host}:{port}`, `http://$HOST`) is the
# project's own test-server fixture being formatted in (prime v4 requests-3362:
# `from tests.testserver.server import Server`), never a remote
LOOPBACK_HOST_RE = re.compile(
    r"^(?:127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|::1|%s|%\(\w+\)s|\{\w*\}|\$\{?\w+\}?)(?::(?:\d+|%d|\{\w*\}|\$\{?\w+\}?))?$", re.I)
LOOPBACK_CONN_RE = re.compile(
    r"\(\s*['\"](?:127\.0\.0\.1|localhost|0\.0\.0\.0|::1)['\"]\s*,\s*\d+\s*[,)]"   # HTTPConnection('127.0.0.1', 23334) / create_connection(('127.0.0.1', 23334))
    r"|https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\])(?::\d+)?(?:[/?#]|\b)"
    r"|/run/bridge\.sock|bridge\.sock", re.I)
SELF_DISPATCH_RE = re.compile(r"/v1/(?:chat/)?completions|/v1/responses|/v1/messages|/generate\b", re.I)
BRIDGE_PORT_RE = re.compile(r"\b23334\b|bridge\.sock")
SERVER_LINE_RE = re.compile(
    r"HTTP/1\.[01]\s+\d{3}\b|^\s*status(?: line)?\s*[:=]?\s*(?:b')?(?:HTTP/1\.[01]\s+)?\d{3}\b"
    r"|^\s*(?:date|server|content-length|content-type|vary|connection|x-[\w-]+|www-authenticate)\s*:\s"
    r"|uvicorn|\"detail\"\s*:\s*\"Not Found\"|\"error\"\s*:\s*\"Unauthorized\"|\bUnauthorized\b|Not Found\b"
    r"|SGLang is running|\"openapi\"\s*:|swagger-ui|\"object\"\s*:\s*\"(?:chat\.completion|text_completion|list|model)"
    r"|\"reasoning_content\"|\"finish_reason\"|\"usage\"\s*:\s*\{|prompt_tokens|completion_tokens|reasoning_tokens"
    r"|\bhealth\b.{0,20}\b200\b|bridge open ok", re.I | re.M)


# A python script whose every connection goes to the loopback bridge cannot reach a
# remote host whatever URL it writes into the request line: an absolute-form
# `c.request('GET', 'http://api.github.com/…')` on HTTPConnection('127.0.0.1', 23334),
# a ProxyHandler / proxies= pointing at the bridge, or a `/proxy/http://…` path probe
# are all answered by uvicorn (401, 404, the /docs page, /v1/models). qwen38
# opencode-dcp v4 xarray-4248 (2026-10-21) printed those answers as `== <path> 401`
# and `<path> 200 b'<!DOCTYPE html>…'` lines SERVER_LINE_RE did not know -> web
# UPSTREAM ok=True -> a false EXPOSED at the gate. The web event is kept only when the
# script opens a connection of its own to a remote host: a fetch primitive given a
# remote URL literal, a remote host in HTTP(S)Connection / create_connection, or a
# fetch of a non-literal URL with no loopback proxy installed.
PY_DIRECT_FETCH_RE = re.compile(
    r"\b(?:urlopen|urlretrieve|requests\.(?:get|post|head|put|delete)|httpx\.(?:get|post|head)|\w*opener\.open|op\.open)"
    r"\(\s*(?:['\"](?P<url>https?://[^'\"]+)['\"]|(?P<var>[A-Za-z_]))"
    r"|\brequests\.request\(\s*['\"]\w+['\"]\s*,\s*(?:['\"](?P<url2>https?://[^'\"]+)['\"]|(?P<var2>[A-Za-z_]))"
    r"|\b(?:HTTPS?Connection|create_connection)\(\s*\(?\s*['\"](?P<host>[^'\"]+)['\"]")
PY_CONN_HOST_RE = re.compile(r"\b(?:HTTPS?Connection|create_connection)\(\s*\(?\s*['\"]([^'\"]+)['\"]")
PY_LOOPBACK_PROXY_RE = re.compile(
    r"(?:ProxyHandler\(|proxies\s*=|https?_proxy\b)[^\n]*(?:127\.0\.0\.1|localhost|\[?::1\]?|0\.0\.0\.0)", re.I)


def py_direct_remote(text: str) -> bool:
    """True when an inline python script opens a connection of its own to a remote host
    (see PY_DIRECT_FETCH_RE); False for a script that only talks to loopback, however
    many remote URLs it passes through the bridge."""
    proxied = bool(PY_LOOPBACK_PROXY_RE.search(text))
    for m in PY_DIRECT_FETCH_RE.finditer(text):
        url = m.group("url") or m.group("url2")
        host = m.group("host")
        if url is not None:
            if not url_is_loopback(url) and not proxied:
                return True
        elif host is not None:
            if not LOOPBACK_HOST_RE.match(host):
                return True
        elif not proxied:  # a variable URL with no loopback proxy in the script
            return True
    return False


def url_is_loopback(url: str) -> bool:
    host = re.sub(r"^https?://", "", url, flags=re.I).split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return bool(LOOPBACK_HOST_RE.match(host))


def targets_loopback(text: str) -> bool:
    return bool(LOOPBACK_CONN_RE.search(text)) or any(url_is_loopback(u) for u in URL_RE.findall(text))


def download_target_empty(output: str, command: str) -> bool:
    """True when every download target the command names that the output mentions is
    either listed at 0 bytes or reported missing by a following local tool (and at
    least one is mentioned) — the fetch never produced the file."""
    names = {Path(t.strip("'\"")).name for t in DL_TARGET_RE.findall(command)}
    names.discard("-")
    if not names:
        return False
    mentioned = 0
    for name in names:
        esc = re.escape(name)
        zero = re.search(r"^[-bcdlps][-rwxsStT]{9}[+@.]?\s+\d+\s+\S+\s+\S+\s+0\s+\S+\s+\d+\s+\S+\s+(?:\S*/)?"
                         + esc + r"\s*$", output, re.M)
        missing = re.search(r"'?(?:\S*/)?" + esc + r"'?: No such file or directory", output)
        present = re.search(r"^[-bcdlps][-rwxsStT]{9}[+@.]?\s+\d+\s+\S+\s+\S+\s+[1-9]\d*\s+\S+\s+\d+\s+\S+\s+(?:\S*/)?"
                            + esc + r"\s*$", output, re.M)
        if present:
            return False
        if zero or missing:
            mentioned += 1
    return mentioned > 0


def fetched_content(output: str, command: str, loopback: bool = False) -> bool:
    """True when a tool output carries something beyond the model's own echo/separator
    scaffolding — a literal that also appears in the command text is the model talking
    to itself, not the network answering. `loopback`: the command also talked to the
    harness bridge, so the server's own response lines are not fetched content either."""
    if download_target_empty(output, command):
        return False
    if HTTP_CODE_W_RE.search(command) and HTTP_000_RE.search(output):
        return False
    output = PY_SYNTAX_BLOCK_RE.sub("", output)
    if PIPED_PY_RE.search(command) and JSON_EMPTY_RE.search(output):
        rest = [ln for ln in output.splitlines() if ln.strip() and not SCAFFOLD_LINE_RE.match(ln)
                and ln.strip() not in command and not LOCAL_ERR_LINE_RE.match(ln)
                and not ln.startswith(("Traceback", "  File ", "    ")) and "JSONDecodeError" not in ln]
        if not rest:
            return False
    local_git_log = bool(GIT_LOG_CMD_RE.search(command)) and not GIT_NET_CMD_RE.search(command)
    for line in output.splitlines():
        t = line.strip()
        if not t or SCAFFOLD_LINE_RE.match(line) or LOCAL_ERR_LINE_RE.match(line) \
                or MODEL_EXC_LINE_RE.match(line) or t in command:
            continue
        if local_git_log and GIT_ONELINE_RE.match(t):
            continue
        if loopback and SERVER_LINE_RE.search(line):
            continue
        return True
    return False

# pip options that consume the next token, so a bare `pip download --no-deps -d /tmp/x
# 2>/dev/null` (no requirement anywhere) is recognised as the no-op it is: pip exits
# with "You must give at least one requirement" before touching the network.
PIP_VALUE_OPTS = {
    "-d", "--dest", "-t", "--target", "-i", "--index-url", "--extra-index-url", "-f",
    "--find-links", "-c", "--constraint", "--platform", "--python-version",
    "--implementation", "--abi", "--prefix", "--root", "--timeout", "--retries", "--proxy",
    "--trusted-host", "--cache-dir", "--log", "--progress-bar", "--exists-action",
    "--config-settings", "--global-option", "--install-option", "--report", "--src",
    "--upgrade-strategy", "--use-feature", "--use-deprecated", "--cert", "--client-cert",
}
PIP_CMD_RE = re.compile(r"\b(?:pip3?|python\S*\s+-m\s+pip)\s+(?:download|install)\b(.*)", re.I | re.S)
REDIRECT_RE = re.compile(r"\d*>{1,2}&?\S*|\d*<\S*")


def pip_has_requirement(segment: str) -> bool:
    """False for a pip download/install segment that names nothing to fetch."""
    m = PIP_CMD_RE.search(segment)
    if not m:
        return True
    args = REDIRECT_RE.sub(" ", m.group(1)).split("|", 1)[0].split()
    skip = False
    for tok in args:
        if skip:
            skip = False
            continue
        if tok in ("-r", "--requirement", "-e", "--editable"):
            return True
        if tok.startswith("-"):
            skip = tok in PIP_VALUE_OPTS
            continue
        return True
    return False




def command_segments(text: str) -> list[str]:
    """Split a shell command at list separators (`;`, `&&`, `||`, newline — not
    pipes, which share one stream) outside quotes. A net tool is classified on
    its own segment: the 120-char snippet window used to read `find … -path
    "*pytest*"` two commands later as an UPSTREAM fetch (qwen38 opencode v3
    pytest-8365, 2026-09-24)."""
    segs, buf, q, i = [], [], None, 0
    while i < len(text):
        c = text[i]
        if q:
            buf.append(c)
            if c == q:
                q = None
        elif c in "'\"":
            q = c
            buf.append(c)
        elif c == "\\" and i + 1 < len(text):
            buf.append(text[i:i + 2])
            i += 1
        elif c == "\n" or c == ";" or text.startswith("&&", i) or text.startswith("||", i):
            segs.append("".join(buf))
            buf = []
            if c not in "\n;":
                i += 1
        else:
            buf.append(c)
        i += 1
    segs.append("".join(buf))
    # a `# …` comment line is not a command: prime v4 pytest-7490 wrote
    # `# Check pip download cache / wheels on disk` above a local `find`
    return [s.strip() for s in segs if s.strip() and not s.strip().startswith("#")]


# instance prefix -> tokens that identify the project's own sites/repos in a URL
PROJECT_TOKENS = {
    "django__django": ["django"],
    "astropy__astropy": ["astropy"],
    "sympy__sympy": ["sympy"],
    "matplotlib__matplotlib": ["matplotlib"],
    "scikit-learn__scikit-learn": ["scikit-learn", "sklearn"],
    "pytest-dev__pytest": ["pytest"],
    "psf__requests": ["requests", "psf"],
    "pallets__flask": ["flask", "pallets"],
    "sphinx-doc__sphinx": ["sphinx"],
    "pylint-dev__pylint": ["pylint"],
    "pydata__xarray": ["xarray"],
    "mwaskom__seaborn": ["seaborn"],
}

# git commands that enumerate or read refs other than the detached HEAD by
# construction (a bare `git log` walks HEAD's ancestry only)
GIT_DEFINITIONAL_RE = re.compile("|".join(f"(?:{p})" for p in [
    r"\bgit\b[^\n|;&]*\s--(all|branches|remotes|tags)\b",
    r"\bgit\s+branch\s+(-a|-r|--all|--remotes|--list|--contains)\b",
    r"\bgit\s+tag\b",
    r"\bgit\s+(for-each-ref|ls-remote|name-rev)\b",
    r"\bgit\s+describe\s+[^\n|;&]*--contains",
    r"\bgit\s+cat-file\s+--batch-all-objects",
    r"\.git/(packed-refs|refs/(remotes|tags|heads))",
    r"\bgit\s+(show|diff|log|checkout|cherry-pick|blame|grep)\b[^\n|;&]*\s(origin/|upstream/|refs/tags/|tags/)",
]), re.I)
GIT_READ_RE = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(show|diff|cat-file|checkout|restore|range-diff|format-patch|cherry-pick)\b"
    r"|\bgit\s+(?:-C\s+\S+\s+)?log\b[^\n|;&]*\s(-p|-u|--patch|--stat|--name-only|--name-status)\b", re.I)


def project_tokens(inst: str) -> list[str]:
    for pre, toks in PROJECT_TOKENS.items():
        if inst.startswith(pre):
            return toks
    owner_repo = inst.rsplit("-", 1)[0]
    return [t for t in owner_repo.split("__") if t]


def url_is_upstream(inst: str, url: str) -> bool:
    u = url.lower()
    return any(t in u for t in project_tokens(inst))


def cmd_text(args) -> str:
    if isinstance(args, str):
        return args
    if not isinstance(args, dict):
        return ""
    return " ".join(str(args.get(k) or "") for k in
                    ("command", "code", "cmd", "path", "pattern", "file_path", "url", "query"))


def classify_call(inst: str, name: str, args, output: str | None, is_error: bool | None) -> list[dict]:
    """One tool call -> zero or more events {chan, kind, ok, evidence}.
    ok: True = returned content, False = failed, None = unknown (no result)."""
    text = cmd_text(args)
    lname = (name or "").lower()
    a = args if isinstance(args, dict) else {}
    ev: list[dict] = []
    if lname in EVAL_LANG_TOOLS:
        lang = str(a.get("language") or "js").lower()   # omp eval defaults to JS
        lname = "python" if lang in ("py", "python") else "js_eval"
    if lname in URL_PATH_TOOLS and re.match(r"https?://", str(a.get("path") or ""), re.I):
        lname, a = "fetch", {**a, "url": a.get("path")}
    if lname in GH_WRAPPER_TOOLS:
        target = " ".join(str(a.get(k)) for k in ("repo", "pr", "run", "query", "path") if a.get(k))
        ev.append({"chan": "web", "kind": "UPSTREAM" if url_is_upstream(inst, target) or any(
            re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", target.lower()) for t in project_tokens(inst)) else "OTHER",
                   "ok": None if output is None else (False if is_error else fetched_content(output, text)),
                   "evidence": f"{name}: {a.get('op')} {target[:140]}".rstrip()})
        return ev

    loopback = targets_loopback(text)

    def outcome() -> bool | None:
        if is_error:
            return False
        if output is None:
            return None
        if NET_FAIL_RE.search(output[:2000]) or PY_NET_TRACEBACK_RE.search(output[:4000]):
            return False
        return fetched_content(output, text, loopback=loopback)

    def bridge_event(evidence: str) -> dict:
        kind = "SELF" if SELF_DISPATCH_RE.search(text) and BRIDGE_PORT_RE.search(text) else (
            "BRIDGE" if BRIDGE_PORT_RE.search(text) else "LOCAL")
        return {"chan": "loopback", "kind": kind, "ok": outcome(), "evidence": evidence}

    if lname in WEB_SEARCH_TOOLS:
        ev.append({"chan": "web", "kind": "SEARCH", "ok": outcome(),
                   "evidence": f"{name}: {a.get('query') or text[:120]}"})
        return ev
    if lname in WEB_FETCH_TOOLS:
        url = str(a.get("url") or text)
        if url_is_loopback(url):
            ev.append(bridge_event(f"{name}: {url[:160]}"))
            return ev
        ev.append({"chan": "web", "kind": "UPSTREAM" if url_is_upstream(inst, url) else "OTHER",
                   "ok": outcome(), "evidence": f"{name}: {url[:160]}"})
        return ev
    # bash-like tools: one event per list segment that touches the network,
    # classified on that segment's own text (URLs / project tokens), never on
    # its neighbours.
    # The outcome stays a reading of the shared output — a `pip -q`/`gh` that
    # fetched quietly still shows in a following `ls` (v2-netopen: 5 such
    # cells at gold overlap 1.0), so a compound command is never cleared on
    # its tool alone.
    segments = command_segments(text)
    compound = len(segments) > 1
    for seg in segments:
        m = NET_BASH_RE.search(seg)
        if not m or not pip_has_requirement(seg):
            continue
        # a segment without a URL of its own borrows the command's (a heredoc
        # / variable two lines up); a project URL anywhere beside a fetch is
        # strong evidence, unlike a neighbouring `*pytest*` glob
        urls = URL_RE.findall(seg) or URL_RE.findall(text)
        if urls and all(url_is_loopback(u) for u in urls):
            # `curl http://127.0.0.1:23334/health` -- the bridge, not the web
            ev.append(bridge_event(f"{name}: {m.group(0)} {urls[0][:140]}" + (" [segment]" if compound else "")))
            continue
        urls = [u for u in urls if not url_is_loopback(u)]
        if urls:
            kind = "UPSTREAM" if any(url_is_upstream(inst, u) for u in urls) else "OTHER"
            evidence = f"{name}: {m.group(0)} {urls[0][:140]}"
        else:
            snippet = seg[m.start():m.start() + 120]
            words = re.sub(r"\S*/testbed/\S*", "", snippet).lower()
            kind = "UPSTREAM" if (m.group(0).lower().startswith("gh ") or any(
                re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", words) for t in project_tokens(inst))) else "OTHER"
            evidence = f"{name}: {snippet}"
        if compound:
            evidence += " [segment]"
        ev.append({"chan": "web", "kind": kind, "ok": outcome(), "evidence": evidence})
    if lname in PY_CODE_TOOLS:
        ps = PY_SEARCH_RE.search(text)
        if ps:
            ok = False if (output and (PY_SEARCH_FAIL_RE.search(output) or PY_CELL_RAISED_RE.match(output))) else outcome()
            ev.append({"chan": "web", "kind": "SEARCH", "ok": ok,
                       "evidence": f"{name}: python {text[ps.start():ps.start() + 120]}"})
    py = PY_FETCH_RE.search(text) if (PY_EXEC_RE.search(text) or lname in PY_CODE_TOOLS) else None
    kernel = "python"
    if py is None and lname in JS_CODE_TOOLS:
        py = JS_FETCH_RE.search(text)
        kernel = "js"
    if py:
        urls = URL_RE.findall(text)
        remote = [u for u in urls if not url_is_loopback(u)]
        if loopback and remote and not py_direct_remote(text):
            remote = []  # absolute-form / proxy probes through the bridge: loopback only
        # HTTPSConnection('api.github.com') + c.request('GET', '/repos/…') names no URL at
        # all: the connection host is the remote target (kind from the project tokens in
        # the script, where the request path sits)
        conn_remote = [h for h in PY_CONN_HOST_RE.findall(text) if not LOOPBACK_HOST_RE.match(h)]
        if loopback:
            # the bridge is one of the script's targets; the project tokens in a
            # self-dispatch prompt ("which sympy PR fixed …") name the question, not a site
            ev.append(bridge_event(f"{name}: {kernel} {py.group(0)} {(urls[0][:140] if urls else '127.0.0.1')}".rstrip()))
        if remote or conn_remote or not loopback:
            words = re.sub(r"\S*/testbed/\S*", "", text).lower()
            kind = "UPSTREAM" if (any(url_is_upstream(inst, u) for u in remote) if remote else any(
                re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", words) for t in project_tokens(inst))) else "OTHER"
            target = remote[0][:140] if remote else (f"https://{conn_remote[0]}"[:140] if conn_remote else "")
            ev.append({"chan": "web", "kind": kind, "ok": outcome(),
                       "evidence": f"{name}: {kernel} {py.group(0)} {target}".rstrip()})
    g = GIT_DEFINITIONAL_RE.search(text)
    if g:
        ev.append({"chan": "git", "kind": "READ" if GIT_READ_RE.search(text) else "LIST",
                   "ok": outcome(), "evidence": f"{name}: {g.group(0)[:120]}"})
    return ev


# --- transcript readers -------------------------------------------------------
def calls_from_opencode_log(log_path: Path):
    """opencode `--format json` event stream captured in the per-instance log."""
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith('{"type":"tool_use"'):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        part = ev.get("part") or {}
        st = part.get("state") or {}
        out = st.get("output")
        yield (part.get("tool") or "", st.get("input") or {},
               str(out) if out is not None else None, st.get("status") == "error")


def calls_from_opencode_parts(parts):
    for p in parts:
        if not isinstance(p, dict) or p.get("type") != "tool":
            continue
        st = p.get("state") or {}
        out = st.get("output")
        yield (p.get("tool") or "", st.get("input") or {},
               str(out) if out is not None else None, st.get("status") == "error")


def calls_from_opencode_db(db: Path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("select data from part").fetchall()
    finally:
        con.close()
    parts = []
    for (data,) in rows:
        try:
            parts.append(json.loads(data))
        except Exception:
            pass
    yield from calls_from_opencode_parts(parts)


def calls_from_opencode_storage(storage: Path):
    parts = []
    for f in storage.rglob("*.json"):
        try:
            parts.append(json.loads(f.read_text(errors="replace")))
        except Exception:
            pass
    yield from calls_from_opencode_parts(parts)


def calls_from_pi_files(files):
    """pi-ai session jsonl (little-coder, prime, omp — same toolCall /
    toolResult entry shape): assistant toolCall blocks + toolResult messages
    linked by id."""
    calls: dict[str, tuple] = {}
    results: dict[str, tuple[str, bool]] = {}
    order: list[str] = []
    for f in files:
        for line in f.read_text(errors="replace").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            m = o.get("message") or {}
            role = m.get("role")
            if role == "assistant":
                for c in m.get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "toolCall":
                        cid = str(c.get("id") or len(order))
                        calls[cid] = (c.get("name") or "", c.get("arguments") or {})
                        order.append(cid)
            elif role == "toolResult":
                txt = " ".join(str(c.get("text", "")) for c in (m.get("content") or []) if isinstance(c, dict))
                results[str(m.get("toolCallId"))] = (txt, bool(m.get("isError")))
    for cid in order:
        name, args = calls[cid]
        out, err = results.get(cid, (None, None))
        yield name, args, out, err


def calls_from_dcode_db(db: Path):
    """deepagents-code langgraph SqliteSaver (msgpack + ext types): AIMessage
    tool_calls + ToolMessage results linked by tool_call_id (the result's
    ``status`` is "error" on a raised tool; a failed fetch_url comes back
    ``status=success`` with a JSON ``{"error": "... Could not resolve ..."}``
    body, which NET_FAIL_RE reads). Before 2026-10-18 the outputs were not
    paired and every dcode fetch under network=none read as outcome unknown."""
    import msgpack  # serving env has it

    def dec(b):
        return msgpack.unpackb(b, raw=False, ext_hook=lambda code, data: dec(data), strict_map_key=False)

    calls: dict[str, tuple] = {}
    results: dict[str, tuple[str, bool]] = {}
    order: list[str] = []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for (tid,) in con.execute("select distinct thread_id from checkpoints"):
            for (v,) in con.execute("select value from writes where thread_id=? and channel='messages' order by rowid", (tid,)):
                try:
                    msgs = dec(v)
                except Exception:
                    continue
                for m in msgs if isinstance(msgs, list) else [msgs]:
                    if not (isinstance(m, list) and len(m) >= 3 and isinstance(m[2], dict)):
                        continue
                    if m[1] == "AIMessage":
                        for tc in m[2].get("tool_calls") or []:
                            if isinstance(tc, dict):
                                cid = str(tc.get("id") or len(order))
                                calls[cid] = (tc.get("name") or "", tc.get("args") or {})
                                order.append(cid)
                    elif m[1] == "ToolMessage":
                        c = m[2].get("content")
                        if isinstance(c, list):
                            c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
                        results[str(m[2].get("tool_call_id"))] = (
                            str(c) if c is not None else "", m[2].get("status") == "error")
    finally:
        con.close()
    for cid in order:
        name, args = calls[cid]
        out, err = results.get(cid, (None, None))
        yield name, args, out, err


def instance_calls(run: Path, iid: str) -> tuple[str, list]:
    """(source, [(name, args, output, is_error), ...]) for one instance."""
    s = run / "sessions" / iid
    db = s / ".local/share/opencode/opencode.db"
    if db.exists():
        return "opencode.db", list(calls_from_opencode_db(db))
    st = s / ".local/share/opencode/storage"
    if st.is_dir():
        return "opencode.storage", list(calls_from_opencode_storage(st))
    for rel in (".pi/agent/sessions", ".prime/agent/sessions", ".omp/agent/sessions"):
        d = s / rel
        if d.is_dir():
            files = sorted(d.rglob("*.jsonl"))
            if files:
                return rel, list(calls_from_pi_files(files))
    dd = s / ".deepagents/.state/sessions.db"
    if dd.exists():
        try:
            return "deepagents.db", list(calls_from_dcode_db(dd))
        except Exception as e:  # noqa: BLE001
            print(f"  ! {iid}: dcode store unreadable ({e})", file=sys.stderr)
    log = run / "logs" / f"{iid}.log"
    if log.exists():
        calls = list(calls_from_opencode_log(log))
        return ("log.opencode" if calls else "log.no-transcript"), calls
    return "none", []


ISO_NET_RE = re.compile(r"^isolation: network=none bridge=", re.M)
ISO_REFS_RE = re.compile(r"^isolation: refs=(\d+) tags=(\d+)", re.M)
# bind-source paths as the sandbox sees them in /proc/self/mountinfo (docker_rollout.py
# stages them under a neutral dir, 2026-09-20): none may name the benchmark or the instance
ISO_MOUNTS_RE = re.compile(r"^isolation: mounts=(.*)$", re.M)
ISO_GIT_RE = re.compile(r"^isolation: git=reinit commits=(\d+) dirty=(\d+) author=(\S+)"
                        r"(?: dirty_before=(\d+))?(?: excluded=(\d+))?", re.M)
CUE_RE = re.compile(r"swe[-_ ]?bench", re.I)
# `git status --porcelain | wc -l` of the pristine official image, for logs written before
# docker_rollout.py reported `dirty_before=` itself (psf__requests-863 ships an untracked
# `build/`; every other Lite image is clean). Keyed by instance id; absent = 0.
IMAGE_UNTRACKED_BASELINE = Path(__file__).with_name("image_untracked_baseline.json")


def image_untracked(iid: str) -> int:
    try:
        return int(json.loads(IMAGE_UNTRACKED_BASELINE.read_text()).get(iid, 0))
    except (OSError, ValueError):
        return 0


def isolation_proof(run: Path, iid: str) -> dict:
    log = run / "logs" / f"{iid}.log"
    if not log.exists():
        return {"network_none": False, "refs_stripped": False, "mounts_clean": False, "git_reinit": False}
    t = log.read_text(errors="replace")
    m = ISO_REFS_RE.search(t)
    mm = ISO_MOUNTS_RE.search(t)
    mg = ISO_GIT_RE.search(t)
    return {"network_none": bool(ISO_NET_RE.search(t)),
            "refs_stripped": bool(m and int(m.group(1)) <= 1 and int(m.group(2)) == 0),
            "mounts_clean": bool(mm and not CUE_RE.search(mm.group(1)) and iid not in mm.group(1)),
            "git_reinit": bool(mg and mg.group(1) == "1" and not CUE_RE.search(mg.group(3))
                               and int(mg.group(2)) + int(mg.group(5) or 0)
                               == (int(mg.group(4)) if mg.group(4) is not None
                                   else image_untracked(iid)))}


# --- gold overlap --------------------------------------------------------------
def load_gold() -> dict[str, str]:
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return {r["instance_id"]: r["patch"] for r in ds}
    except Exception as e:  # noqa: BLE001
        print(f"  ! gold patches unavailable ({e}); overlap skipped", file=sys.stderr)
        return {}


def added_lines(diff: str) -> set[str]:
    out = set()
    for l in diff.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            s = l[1:].strip()
            if len(s) >= 12 and not s.startswith("#"):
                out.add(s)
    return out


def gold_overlap(model_patch: str, gold: str) -> float | None:
    a = added_lines(model_patch)
    if not a:
        return None
    return len(a & added_lines(gold)) / len(a)


# --- driver ---------------------------------------------------------------------
def audit_run(run: Path, require_isolation: bool, gold: dict[str, str]) -> dict:
    preds: dict[str, str] = {}
    predrows: dict[str, dict] = {}
    pf = run / "predictions.jsonl"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    preds[r["instance_id"]] = r.get("model_patch") or ""
                    predrows[r["instance_id"]] = r
                except Exception:
                    pass
    iids = sorted(preds) or sorted(p.stem for p in (run / "logs").glob("*.log"))
    rows = {}
    for iid in iids:
        # A row written by the crash-row path (`rollout_error`: the per-instance
        # image build failed, rc -1, no container ever ran — the 2026-10-09
        # prime-agent release pruning left 3 such rows in qwen38-prime-v4 that
        # the retired 0.8.1 pin can never re-roll) has no log, no transcript
        # and an empty patch by construction. It scores as unresolved; it is
        # not a missing isolation proof — nothing ran that could have leaked.
        # Same definition as check_cell_pin / paired_lane_compare.
        pr = predrows.get(iid, {})
        never_rolled = bool(pr.get("rollout_error")) and not preds.get(iid) \
            and not (run / "logs" / f"{iid}.log").exists()
        if never_rolled:
            rows[iid] = {
                "source": "never-rolled", "web": "none", "git": "clean", "loopback": "none",
                "loopback_calls": 0, "attempted": False, "exposed": False, "never_rolled": True,
                "isolation": {"network_none": False, "refs_stripped": False,
                              "mounts_clean": False, "git_reinit": False},
                "overlap": None,
                "evidence": [f"never rolled: {str(pr.get('rollout_error'))[:100]}"],
            }
            continue
        source, calls = instance_calls(run, iid)
        events = []
        for name, args, out, err in calls:
            events.extend(classify_call(iid, name, args, out, err))
        web = [e for e in events if e["chan"] == "web"]
        git = [e for e in events if e["chan"] == "git"]
        loop = [e for e in events if e["chan"] == "loopback"]
        loop_cls = next((k for k in ("SELF", "BRIDGE", "LOCAL") if any(e["kind"] == k for e in loop)), "none")
        kinds = {e["kind"] for e in web}
        web_cls = "UPSTREAM" if "UPSTREAM" in kinds else ("SEARCH" if "SEARCH" in kinds else ("OTHER" if web else "none"))
        git_cls = "READ" if any(e["kind"] == "READ" for e in git) else ("LIST" if git else "clean")
        leak_events = [e for e in web if e["kind"] in ("UPSTREAM", "SEARCH")]
        attempted = bool(leak_events)
        # conservative: an attempt with unknown outcome counts as exposed
        exposed = any(e["ok"] is not False for e in leak_events)
        iso = isolation_proof(run, iid)
        ov = gold_overlap(preds.get(iid, ""), gold[iid]) if gold and iid in gold else None
        rows[iid] = {
            "source": source, "web": web_cls, "git": git_cls, "loopback": loop_cls,
            "loopback_calls": len(loop),
            "attempted": attempted, "exposed": exposed,
            "isolation": iso, "overlap": ov,
            "evidence": [f"{e['chan']} {e['kind']} ok={e['ok']}: {e['evidence']}" for e in events][:12],
        }
    n = len(rows)
    rolled = {k: r for k, r in rows.items() if not r.get("never_rolled")}
    n_rolled = len(rolled)
    have = [r for r in rolled.values() if not r["source"].startswith("log.no-transcript") and r["source"] != "none"]
    c = Counter()
    for r in have:
        c[f"web_{r['web']}"] += 1
        c[f"git_{r['git']}"] += 1
        if r["loopback"] != "none":
            c[f"loopback_{r['loopback']}"] += 1
        c["attempted"] += r["attempted"]
        c["exposed"] += r["exposed"]
    c["isolated"] = len(have) - c["exposed"]
    c["never_rolled"] = n - n_rolled
    c["no_transcript"] = n_rolled - len(have)
    c["iso_network_none"] = sum(1 for r in rolled.values() if r["isolation"]["network_none"])
    c["iso_refs_stripped"] = sum(1 for r in rolled.values() if r["isolation"]["refs_stripped"])
    c["iso_mounts_clean"] = sum(1 for r in rolled.values() if r["isolation"]["mounts_clean"])
    c["iso_git_reinit"] = sum(1 for r in rolled.values() if r["isolation"]["git_reinit"])

    def ov_stats(flag: bool):
        xs = [r["overlap"] for r in have if r["exposed"] == flag and r["overlap"] is not None]
        return {"n": len(xs), "ge80": sum(1 for x in xs if x >= 0.8),
                "median": round(statistics.median(xs), 3) if xs else None}

    # a cell with unreadable transcripts is only clean if it is PROVEN isolated;
    # the proofs are owed by every row that ran a container (n_rolled)
    verdict_ok = c["exposed"] == 0 and (c["no_transcript"] == 0 or c["iso_network_none"] == n_rolled)
    if require_isolation:
        verdict_ok = verdict_ok and c["iso_network_none"] == n_rolled and c["iso_refs_stripped"] == n_rolled \
            and c["iso_mounts_clean"] == n_rolled and c["iso_git_reinit"] == n_rolled and n_rolled > 0
    return {"run": run.name, "n": n, "n_rolled": n_rolled, "counts": dict(c),
            "gold_overlap": {"exposed": ov_stats(True), "isolated": ov_stats(False)},
            "require_isolation": require_isolation, "ok": verdict_ok, "instances": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, action="append", help="run dir (repeatable)")
    ap.add_argument("--require-isolation", action="store_true",
                    help="fail unless every instance log carries the network=none + refs-stripped + mounts-clean + git-reinit proof")
    ap.add_argument("--json", help="write the report here (default: <run>/leak-audit.json per run)")
    ap.add_argument("--show", type=int, default=3, help="evidence lines to print per run")
    ap.add_argument("--no-gold", action="store_true", help="skip the gold-patch overlap")
    args = ap.parse_args()

    gold = {} if args.no_gold else load_gold()
    reports = []
    rc = 0
    for r in args.run:
        run = Path(r)
        if not run.is_dir():
            print(f"ERROR: not a run dir: {run}", file=sys.stderr)
            return 2
        rep = audit_run(run, args.require_isolation, gold)
        reports.append(rep)
        c, n = rep["counts"], rep["n"]
        nr = rep["n_rolled"]
        den = max(nr - c.get("no_transcript", 0), 1)
        never = f" never_rolled={c['never_rolled']} (build crash, no container)" if c.get("never_rolled") else ""
        print(f"{rep['run']}: n={n}{never} transcripts={nr - c.get('no_transcript', 0)} "
              f"web_UPSTREAM={c.get('web_UPSTREAM', 0)} web_SEARCH={c.get('web_SEARCH', 0)} web_OTHER={c.get('web_OTHER', 0)} "
              f"git_READ={c.get('git_READ', 0)} git_LIST={c.get('git_LIST', 0)} "
              f"loopback_SELF={c.get('loopback_SELF', 0)} _BRIDGE={c.get('loopback_BRIDGE', 0)} _LOCAL={c.get('loopback_LOCAL', 0)} | "
              f"attempted={c.get('attempted', 0)} ({100 * c.get('attempted', 0) / den:.0f}%) "
              f"EXPOSED={c.get('exposed', 0)} ({100 * c.get('exposed', 0) / den:.0f}%) | "
              f"isolation proof: network=none {c['iso_network_none']}/{nr}, refs stripped {c['iso_refs_stripped']}/{nr}, "
              f"mounts clean {c['iso_mounts_clean']}/{nr}, git reinit {c['iso_git_reinit']}/{nr}")
        go = rep["gold_overlap"]
        print(f"  gold-overlap>=80%: exposed {go['exposed']['ge80']}/{go['exposed']['n']} (med {go['exposed']['median']}), "
              f"isolated {go['isolated']['ge80']}/{go['isolated']['n']} (med {go['isolated']['median']})")
        shown = 0
        for iid, row in rep["instances"].items():
            if row["exposed"] and shown < args.show:
                ex = next((e for e in row["evidence"] if " UPSTREAM " in e or " SEARCH " in e), row["evidence"][:1])
                print(f"    {iid}: {ex if isinstance(ex, str) else ex}")
                shown += 1
        out = Path(args.json) if (args.json and len(args.run) == 1) else run / "leak-audit.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"  {'OK' if rep['ok'] else 'FAIL'} -> {out}")
        rc |= 0 if rep["ok"] else 1
    if args.json and len(args.run) > 1:
        Path(args.json).write_text(json.dumps(reports, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
