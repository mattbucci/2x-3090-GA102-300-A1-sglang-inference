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
    `isolation: refs=1 tags=0`, `isolation: git=reinit commits=1 dirty=0` (the
    tree's history is one `eval@local` commit — the official image's `SWE-bench`
    HEAD commit is gone — and the re-added tracked set left the tree clean) and
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
WEB_FETCH_TOOLS = {"webfetch", "web_fetch", "fetch", "codesearch"}
WEB_SEARCH_TOOLS = {"websearch", "web_search", "web-search", "search"}
URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+", re.I)
NET_BASH_RE = re.compile(
    r"\b(curl|wget|gh\s+(api|pr|issue)|pip3?\s+(download|install)(?!\s+-e)"
    r"|python\S*\s+-m\s+pip\s+(download|install)(?!\s+-e)|git\s+(clone|fetch|pull|ls-remote)\b"
    r"|npm\s+(install|view)|apt(-get)?\s+install)\b", re.I)
NET_FAIL_RE = re.compile(
    r"Could not resolve host|Temporary failure in name resolution|Network is unreachable"
    r"|ENOTFOUND|ECONNREFUSED|EAI_AGAIN|Name or service not known|No address associated"
    r"|fetch failed|Failed to fetch|Connection refused|NewConnectionError|Max retries exceeded"
    r"|Could not fetch URL|No matching distribution|network error|unable to access"
    # pip swallows the DNS failure and prints an empty version list; opencode's
    # webfetch surfaces a blocked socket as "Transport error" (R9700 `1ea2167`)
    r"|\(from versions: none\)|Transport error", re.I)

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

    def outcome() -> bool | None:
        if is_error:
            return False
        if output is None:
            return None
        if NET_FAIL_RE.search(output[:2000]):
            return False
        return len(output.strip()) > 0

    if lname in WEB_SEARCH_TOOLS:
        ev.append({"chan": "web", "kind": "SEARCH", "ok": outcome(),
                   "evidence": f"{name}: {a.get('query') or text[:120]}"})
        return ev
    if lname in WEB_FETCH_TOOLS:
        url = str(a.get("url") or text)
        ev.append({"chan": "web", "kind": "UPSTREAM" if url_is_upstream(inst, url) else "OTHER",
                   "ok": outcome(), "evidence": f"{name}: {url[:160]}"})
        return ev
    # bash-like tools
    m = NET_BASH_RE.search(text)
    if m:
        urls = URL_RE.findall(text)
        if urls:
            kind = "UPSTREAM" if any(url_is_upstream(inst, u) for u in urls) else "OTHER"
            evidence = f"{name}: {m.group(0)} {urls[0][:140]}"
        else:
            snippet = text[m.start():m.start() + 120].replace("\n", "⏎")
            words = re.sub(r"\S*/testbed/\S*", "", snippet).lower()
            kind = "UPSTREAM" if (m.group(0).lower().startswith("gh ") or any(
                re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", words) for t in project_tokens(inst))) else "OTHER"
            evidence = f"{name}: {snippet}"
        ev.append({"chan": "web", "kind": kind, "ok": outcome(), "evidence": evidence})
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
    """pi-ai session jsonl (little-coder, prime): assistant toolCall blocks +
    toolResult messages linked by id."""
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
    """deepagents-code langgraph SqliteSaver (msgpack + ext types)."""
    import msgpack  # serving env has it

    def dec(b):
        return msgpack.unpackb(b, raw=False, ext_hook=lambda code, data: dec(data), strict_map_key=False)

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
                                yield tc.get("name") or "", tc.get("args") or {}, None, None
    finally:
        con.close()


def instance_calls(run: Path, iid: str) -> tuple[str, list]:
    """(source, [(name, args, output, is_error), ...]) for one instance."""
    s = run / "sessions" / iid
    db = s / ".local/share/opencode/opencode.db"
    if db.exists():
        return "opencode.db", list(calls_from_opencode_db(db))
    st = s / ".local/share/opencode/storage"
    if st.is_dir():
        return "opencode.storage", list(calls_from_opencode_storage(st))
    for rel in (".pi/agent/sessions", ".prime/agent/sessions"):
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
ISO_GIT_RE = re.compile(r"^isolation: git=reinit commits=(\d+) dirty=(\d+) author=(\S+)", re.M)
CUE_RE = re.compile(r"swe[-_ ]?bench", re.I)


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
            "git_reinit": bool(mg and mg.group(1) == "1" and mg.group(2) == "0"
                               and not CUE_RE.search(mg.group(3)))}


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
    pf = run / "predictions.jsonl"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    preds[r["instance_id"]] = r.get("model_patch") or ""
                except Exception:
                    pass
    iids = sorted(preds) or sorted(p.stem for p in (run / "logs").glob("*.log"))
    rows = {}
    for iid in iids:
        source, calls = instance_calls(run, iid)
        events = []
        for name, args, out, err in calls:
            events.extend(classify_call(iid, name, args, out, err))
        web = [e for e in events if e["chan"] == "web"]
        git = [e for e in events if e["chan"] == "git"]
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
            "source": source, "web": web_cls, "git": git_cls,
            "attempted": attempted, "exposed": exposed,
            "isolation": iso, "overlap": ov,
            "evidence": [f"{e['chan']} {e['kind']} ok={e['ok']}: {e['evidence']}" for e in events][:12],
        }
    n = len(rows)
    have = [r for r in rows.values() if not r["source"].startswith("log.no-transcript") and r["source"] != "none"]
    c = Counter()
    for r in have:
        c[f"web_{r['web']}"] += 1
        c[f"git_{r['git']}"] += 1
        c["attempted"] += r["attempted"]
        c["exposed"] += r["exposed"]
    c["isolated"] = len(have) - c["exposed"]
    c["no_transcript"] = n - len(have)
    c["iso_network_none"] = sum(1 for r in rows.values() if r["isolation"]["network_none"])
    c["iso_refs_stripped"] = sum(1 for r in rows.values() if r["isolation"]["refs_stripped"])
    c["iso_mounts_clean"] = sum(1 for r in rows.values() if r["isolation"]["mounts_clean"])
    c["iso_git_reinit"] = sum(1 for r in rows.values() if r["isolation"]["git_reinit"])

    def ov_stats(flag: bool):
        xs = [r["overlap"] for r in have if r["exposed"] == flag and r["overlap"] is not None]
        return {"n": len(xs), "ge80": sum(1 for x in xs if x >= 0.8),
                "median": round(statistics.median(xs), 3) if xs else None}

    # a cell with unreadable transcripts is only clean if it is PROVEN isolated
    verdict_ok = c["exposed"] == 0 and (c["no_transcript"] == 0 or c["iso_network_none"] == n)
    if require_isolation:
        verdict_ok = verdict_ok and c["iso_network_none"] == n and c["iso_refs_stripped"] == n \
            and c["iso_mounts_clean"] == n and c["iso_git_reinit"] == n and n > 0
    return {"run": run.name, "n": n, "counts": dict(c),
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
        den = max(n - c.get("no_transcript", 0), 1)
        print(f"{rep['run']}: n={n} transcripts={n - c.get('no_transcript', 0)} "
              f"web_UPSTREAM={c.get('web_UPSTREAM', 0)} web_SEARCH={c.get('web_SEARCH', 0)} web_OTHER={c.get('web_OTHER', 0)} "
              f"git_READ={c.get('git_READ', 0)} git_LIST={c.get('git_LIST', 0)} | "
              f"attempted={c.get('attempted', 0)} ({100 * c.get('attempted', 0) / den:.0f}%) "
              f"EXPOSED={c.get('exposed', 0)} ({100 * c.get('exposed', 0) / den:.0f}%) | "
              f"isolation proof: network=none {c['iso_network_none']}/{n}, refs stripped {c['iso_refs_stripped']}/{n}, "
              f"mounts clean {c['iso_mounts_clean']}/{n}, git reinit {c['iso_git_reinit']}/{n}")
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
