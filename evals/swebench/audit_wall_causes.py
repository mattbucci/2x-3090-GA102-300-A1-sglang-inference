#!/usr/bin/env python3
"""Classify what a rolled instance was doing when its rollout ended, from the
session snapshot: opencode's `<run>/sessions/<iid>/.local/share/opencode/
opencode.db` (sqlite since opencode 1.14) or pi's (little-coder / -rtk)
`<run>/sessions/<iid>/.pi/agent/sessions/*/*.jsonl`. Built for wall hits: a wall is
"model still working" only if the last part is an open reasoning/text/tool
part. An open tool part that has been open LONGER than its own execution
ceiling (bash: the requested `timeout`, default 120 s; other tools ~30 s) never
executed: that is opencode's `external_directory` permission prompt (default
`ask` even under --dangerously-skip-permissions in 1.14.25; the check runs
before the command) — headless `opencode run` never answers it, so the
instance idles to the wall with the server empty. A harness hang, not a model
wall. An open part still inside its ceiling at the wall is the model running
out of budget (test suite, `sleep`) — it would have walled with or without the
prompt, so it is NOT charged to the harness even when an outside path is
present (`permission_possible` records the ambiguity).

Terminal classes (last part across all sessions of the instance, incl. task
subagents):
  permission_hang   open tool part, open > ceiling + slack, outside path
  tool_stuck        open tool part, open > ceiling + slack, no outside path
  tool_running      open tool part inside its ceiling (time exhausted)
  generating        open reasoning/text part (model mid-response at the wall)
  step_boundary     last part is step-start (request in flight / no part yet),
                    or the last part completed within `slack` of the kill
                    (12471: step-finish reason=tool-calls 0.6 s before the
                    wall -- the next tool part was never created)
  finished          last part completed > slack before the kill (the model was
                    done; whatever ran on after it was not the model)
  env_destroyed     the container userland is gone: every bash call after
                    some point fails to spawn (`NotFound: ChildProcess.spawn`).
                    sphinx-8474 (v4): `for base in ('/tmp/mt_lt'): shutil.rmtree
                    (base, ignore_errors=True)` -- a str, not a tuple -- rmtree'd
                    '/' from its first character; /testbed, /opt, /usr and
                    /etc/passwd went with it (docker exec fails too, so no
                    snapshot), 12.6 min of probing to the wall. Read from the
                    opencode json stream in the per-instance log, so it works
                    without a db. Model-side; nothing left the sandbox.
  no_snapshot       no db / no jsonl (pre-capture-at-wall cells are blind on
                    walls; the `snapshot` field carries the wall-hit exec
                    verdict if any)

pi lanes (`format: pi`): one jsonl per session, the scaffold runs a main
session then a `timeout 120` cleanup session (`in_cleanup` marks a kill that
landed in the second one -- the model was done with the task). pi writes an
assistant row only when the response completes, so a trailing toolResult /
user row at the kill means a request was in flight (`generating`, with
`in_flight_s` instead of chars); a toolCall with no toolResult is the open
leaf (bash / ShellSession ceiling = its `timeout` arg in seconds, default
120). pi has no permission prompt (its bash whitelist rejects synchronously),
so an over-ceiling open call is `tool_stuck`, never `permission_hang`;
`outside_paths` is still recorded. `n_empty_name_calls` counts the model's
empty-name tool calls (pi answers them with a nudge user row).

Usage: audit_wall_causes.py <run_dir> [--walls-only] [--project /testbed]
       [--wall 1795] [--json out]
Informational only — never a scoring input; never changes wall semantics.
"""
import argparse, json, pathlib, re, shutil, sqlite3, tempfile

PATH_RE = re.compile(r"(?<![\w.\-])/(?:[\w.\-+]+/)*[\w.\-+]*")
SHELL_BUILTIN_PATHS = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/fd/", "/proc/self")


def real_patch(r) -> str:
    """model_patch unless it is not a diff (wall-hit marker collision with a
    `ps`-ed inner script — see paired_lane_compare.patch): then "" (empty)."""
    p = (r.get("model_patch") or "").strip()
    return p if p.startswith(("diff ", "--- ", "Index: ")) else ""

def load_preds(run):
    rows = {}
    p = pathlib.Path(run) / "predictions.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows[r["instance_id"]] = r
    return rows

def is_wall(r, wall_s):
    """rc 124 = the host timeout killed the scaffold. rollout_seconds spans the
    per-instance image build, so it is only the fallback for rows without rc."""
    rc = r.get("rollout_returncode")
    if rc is not None:
        return rc == 124
    return (r.get("rollout_seconds") or 0) >= wall_s

def outside_paths(tool, inp, project):
    """Paths the tool input touches that resolve outside the project dir."""
    cands = []
    tool = {"shellsession": "bash"}.get(str(tool).lower(), str(tool).lower())
    if tool in ("read", "edit", "write", "glob", "grep", "list", "ls"):
        for k in ("filePath", "path", "file_path"):
            if inp.get(k): cands.append(str(inp[k]))
    if tool == "bash":
        cmd = str(inp.get("command", ""))
        cands += PATH_RE.findall(cmd)
    out = []
    for c in cands:
        if c in ("/",) or c.startswith(SHELL_BUILTIN_PATHS):
            continue
        if not (c == project or c.startswith(project.rstrip("/") + "/")):
            out.append(c)
    return sorted(set(out))

def last_parts(db_dir):
    """Copy the sqlite trio out (the -wal holds the newest rows) and return
    all parts ordered by time_created."""
    with tempfile.TemporaryDirectory() as td:
        for f in ("opencode.db", "opencode.db-wal", "opencode.db-shm"):
            src = db_dir / f
            if src.exists(): shutil.copy2(src, pathlib.Path(td) / f)
        con = sqlite3.connect(f"file:{td}/opencode.db?mode=ro", uri=True)
        try:
            sessions = {sid: parent for sid, parent in con.execute("select id,parent_id from session")}
            parts = [(tc, sid, json.loads(d)) for tc, sid, d in
                     con.execute("select time_created,session_id,data from part order by time_created")]
        finally:
            con.close()
    return sessions, parts

def ceiling_s(tool, inp):
    """Longest the tool could legitimately stay open once it started executing."""
    if tool == "bash":
        try:
            return float(inp.get("timeout", 120000)) / 1000.0
        except (TypeError, ValueError):
            return 120.0
    return 30.0

DESTROYED_RE = re.compile(r"^NotFound: (ChildProcess\.spawn|FileSystem\.access)")

def stream_events(run, iid):
    """The opencode `--format json` event stream captured in logs/<iid>.log
    (tool_use events carry the completed/error state of the part)."""
    lg = pathlib.Path(run) / "logs" / f"{iid}.log"
    if not lg.exists():
        return []
    ev = []
    for line in lg.read_text(errors="replace").splitlines():
        if line.startswith('{"type":'):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return ev

def env_destroyed(run, iid):
    """First bash part whose spawn failed with every later bash call failing
    the same way (>= 2 in the streak), plus the last command that ran before
    it -- the destroyer. None when the shell stayed alive to the end."""
    last_ok, first_bad, n_bad = None, None, 0
    for e in stream_events(run, iid):
        p = e.get("part") or {}
        if p.get("type") != "tool" or p.get("tool") != "bash":
            continue
        st = p.get("state") or {}
        out = str(st.get("output") or st.get("error") or "")
        if st.get("status") == "error" and DESTROYED_RE.match(out):
            if first_bad is None:
                first_bad = (e.get("timestamp") or 0) / 1000.0
            n_bad += 1
        else:
            first_bad, n_bad = None, 0          # shell came back: not destroyed
            if st.get("status") == "completed":
                last_ok = (e.get("timestamp"), str((st.get("input") or {}).get("command", "")))
    if first_bad is None or n_bad < 2:
        return None
    return {"destroyed_at": first_bad, "n_failed_bash": n_bad,
            "destroyer": (last_ok[1] if last_ok else "")[:240]}

def snapshot_verdict(run, iid):
    lg = pathlib.Path(run) / "logs" / f"{iid}.log"
    if not lg.exists():
        return None
    m = re.search(r"^# wall-hit session snapshot: (.*)$", lg.read_text(errors="replace"), re.M)
    return m.group(1).strip() if m else None


def _iso_s(ts):
    """pi row timestamps are ISO-8601 with a Z suffix -> epoch seconds."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None

def pi_session_files(run, iid):
    return sorted((pathlib.Path(run) / "sessions" / iid / ".pi/agent/sessions").glob("*/*.jsonl"))

def pi_rows(files):
    """All `message` rows across the instance's pi sessions, time-ordered:
    (ts, session_index, message)."""
    rows = []
    for si, f in enumerate(files):
        for line in f.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "message" or not isinstance(d.get("message"), dict):
                continue
            ts = _iso_s(d.get("timestamp"))
            if ts is not None:
                rows.append((ts, si, d["message"]))
    rows.sort(key=lambda x: x[0])
    return rows

def pi_ceiling_s(tool, args):
    if str(tool).lower() in ("bash", "shellsession"):
        try:
            return float(args.get("timeout", 120))
        except (TypeError, ValueError):
            return 120.0
    return 30.0

def classify_pi(files, project, end_ts=None, slack_s=120.0, wall=False):
    rows = pi_rows(files)
    if not rows:
        return {"class": "no_snapshot", "note": "pi jsonl has no message rows"}
    calls, results, n_empty = {}, {}, 0
    for ts, si, m in rows:
        if m.get("role") == "assistant":
            for c in m.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "toolCall":
                    if not c.get("name"):
                        n_empty += 1
                    calls[c.get("id")] = {"name": c.get("name") or "", "args": c.get("arguments") or {},
                                          "ts": ts, "si": si}
        elif m.get("role") == "toolResult":
            results[m.get("toolCallId")] = ts
    open_calls = [(cid, c) for cid, c in calls.items() if cid not in results]
    ts, si, m = rows[-1]
    info = {"format": "pi", "n_sessions": len(files), "n_rows": len(rows), "n_tool_calls": len(calls),
            "n_open_tool_parts": len(open_calls), "n_empty_name_calls": n_empty,
            "in_cleanup": si > 0, "last_part_type": m.get("role"), "last_part_ms": int(ts * 1000)}
    if open_calls:
        cid, c = max(open_calls, key=lambda x: x[1]["ts"])
        info.update({"last_part_type": "tool", "tool": c["name"], "status": "running",
                     "input": json.dumps(c["args"])[:240], "in_cleanup": c["si"] > 0})
        ext = outside_paths(c["name"], c["args"], project)
        ceil = pi_ceiling_s(c["name"], c["args"])
        open_s = (end_ts - c["ts"]) if end_ts else None
        info.update({"open_s": round(open_s) if open_s is not None else None, "ceiling_s": ceil})
        if ext:
            info["outside_paths"] = ext[:6]
        if open_s is not None and open_s > ceil + slack_s:
            info["class"] = "tool_stuck"
        else:
            info["class"] = "tool_running"
        return info
    gap = (end_ts - ts) if end_ts else None
    if m.get("role") in ("toolResult", "user"):
        # the next request was in flight: pi only writes the assistant row on completion
        info["in_flight_s"] = round(gap) if gap is not None else None
        if not wall:
            info["class"] = "finished"
            info["note"] = "cleanup session cut by its own timeout" if si > 0 else "exit mid-request"
        elif gap is not None and gap <= slack_s:
            info["class"] = "step_boundary"
        else:
            info["class"] = "generating"
        return info
    info["stop_reason"] = m.get("stopReason")
    info["class"] = "finished"
    if wall and gap is not None:
        info["ended_before_kill_s"] = round(gap)
        if gap <= slack_s:
            info["class"] = "step_boundary"
    return info

def classify(run, iid, project, end_ts=None, slack_s=120.0, wall=False):
    """end_ts: when the rollout ended (the per-instance log's mtime — written
    at exit; the wall-hit snapshot is taken up to ~90 s earlier, hence slack)."""
    db_dir = pathlib.Path(run) / "sessions" / iid / ".local/share/opencode"
    destroyed = env_destroyed(run, iid)
    if not (db_dir / "opencode.db").exists():
        pi_files = pi_session_files(run, iid)
        if pi_files:
            info = classify_pi(pi_files, project, end_ts, slack_s, wall)
            if info["class"] == "no_snapshot":
                info["snapshot"] = snapshot_verdict(run, iid)
            return _mark_destroyed(info, destroyed, end_ts)
    if not (db_dir / "opencode.db").exists() or not last_parts(db_dir)[1]:
        info = {"class": "no_snapshot", "snapshot": snapshot_verdict(run, iid)}
        if (db_dir / "opencode.db").exists():
            info["note"] = "db has no parts"
        return _mark_destroyed(info, destroyed, end_ts)
    sessions, parts = last_parts(db_dir)
    tc, sid, d = parts[-1]
    # A step can issue parallel tool calls; the newest part may then be a
    # completed sibling while an earlier-created one is still open (23562: a
    # finished `read` hid the hung `bash` beside it). Judge by the open leaf
    # tool part, not the newest part. `task` parts are subagent containers --
    # the subagent's own open part is the leaf.
    open_leaf = [(pc, ps, pd) for pc, ps, pd in parts
                 if pd.get("type") == "tool" and pd.get("tool") != "task"
                 and (pd.get("state") or {}).get("status") in ("running", "pending")]
    if open_leaf:
        tc, sid, d = max(open_leaf, key=lambda x: ((x[2].get("state") or {}).get("time") or {}).get("start") or x[0])
    t = d.get("type")
    info = {"last_part_type": t, "last_part_ms": tc, "in_subagent": bool(sessions.get(sid)),
            "n_parts": len(parts), "n_sessions": len(sessions), "n_open_tool_parts": len(open_leaf)}
    if t == "tool":
        st = d.get("state", {})
        info["tool"] = d.get("tool"); info["status"] = st.get("status")
        info["input"] = json.dumps(st.get("input", {}))[:240]
        if st.get("status") in ("running", "pending"):
            ext = outside_paths(d.get("tool"), st.get("input", {}), project)
            start = ((st.get("time") or {}).get("start") or tc) / 1000.0
            ceil = ceiling_s(d.get("tool"), st.get("input", {}))
            open_s = (end_ts - start) if end_ts else None
            info.update({"open_s": round(open_s) if open_s is not None else None, "ceiling_s": ceil})
            if ext:
                info["outside_paths"] = ext[:6]
            if open_s is not None and open_s > ceil + slack_s:
                info["class"] = "permission_hang" if ext else "tool_stuck"
            else:
                info["class"] = "tool_running"
                if ext:
                    info["permission_possible"] = True
        else:
            info["class"] = "finished"
    elif t in ("reasoning", "text"):
        tm = d.get("time") or {}
        info["class"] = "generating" if not tm.get("end") else "finished"
        info["chars"] = len(d.get("text") or "")
    elif t == "step-start":
        info["class"] = "step_boundary"
    else:
        info["class"] = "finished"
        if t == "step-finish":
            info["finish_reason"] = d.get("reason")
    # On a wall, a completed last part only means "finished" if the model had
    # actually stopped: a part that ended within `slack` of the kill is a step
    # boundary (the next request / tool part had not been written yet). A
    # clean exit ends right after its last part by construction -- not a wall.
    if wall and info["class"] == "finished" and end_ts:
        ended = d.get("state", {}).get("time", {}).get("end") if t == "tool" else (d.get("time") or {}).get("end")
        ended_s = (ended or tc) / 1000.0
        info["ended_before_kill_s"] = round(end_ts - ended_s)
        if end_ts - ended_s <= slack_s:
            info["class"] = "step_boundary"
    return _mark_destroyed(info, destroyed, end_ts)

def _mark_destroyed(info, destroyed, end_ts):
    if destroyed:
        info["class"] = "env_destroyed"
        info["destroyer"] = destroyed["destroyer"]
        info["n_failed_bash"] = destroyed["n_failed_bash"]
        if end_ts:
            info["destroyed_before_end_s"] = round(end_ts - destroyed["destroyed_at"])
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--walls-only", action="store_true")
    ap.add_argument("--project", default="/testbed")
    ap.add_argument("--wall", type=float, default=1795.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    preds = load_preds(a.run)
    ids = list(preds) or sorted(p.name for p in (pathlib.Path(a.run) / "sessions").glob("*"))
    rows = []
    for iid in ids:
        r = preds.get(iid, {})
        w = is_wall(r, a.wall) if r else None
        if a.walls_only and not w:
            continue
        lg = pathlib.Path(a.run) / "logs" / f"{iid}.log"
        end_ts = lg.stat().st_mtime if lg.exists() else None
        c = classify(a.run, iid, a.project, end_ts, wall=bool(w))
        c.update({"instance_id": iid, "wall": w, "rc": r.get("rollout_returncode"),
                  "rollout_seconds": r.get("rollout_seconds"),
                  "empty": not real_patch(r)})
        rows.append(c)
    tally = {}
    for c in rows:
        tally[c["class"]] = tally.get(c["class"], 0) + 1
    print(f"{a.run}: n={len(rows)} walls_only={a.walls_only}  classes={json.dumps(tally)}")
    for c in rows:
        if c["class"] in ("permission_hang", "tool_stuck", "tool_running", "generating", "step_boundary", "env_destroyed") or c["wall"]:
            extra = ""
            if c["class"] in ("permission_hang", "tool_stuck"):
                extra = f" open={c.get('open_s')}s ceiling={c.get('ceiling_s')}s outside={c.get('outside_paths')}"
            elif c["class"] in ("tool_running",):
                extra = f" open={c.get('open_s')}s/{c.get('ceiling_s')}s{' (outside path)' if c.get('permission_possible') else ''} {c.get('tool')} {c.get('input','')[:90]}"
            elif c["class"] == "generating":
                extra = (f" request in flight {c.get('in_flight_s')}s" if c.get("format") == "pi"
                         else f" {c.get('last_part_type')} chars={c.get('chars')}")
            elif c["class"] == "env_destroyed":
                extra = f" {c.get('destroyed_before_end_s')}s before the end, {c.get('n_failed_bash')} dead bash calls; destroyer: {c.get('destroyer','')[:100]!r}"
            elif c["class"] == "no_snapshot" and c.get("snapshot"):
                extra = f" ({c['snapshot'][:80]})"
            sub = " (subagent)" if c.get("in_subagent") else (" (cleanup)" if c.get("in_cleanup") else "")
            print(f"  {c['instance_id']:<34} wall={str(c['wall']):<5} rc={str(c['rc']):<4} s={str(c['rollout_seconds']):<7} {c['class']}{sub}{extra}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"run": a.run, "wall_s": a.wall, "tally": tally, "rows": rows}, indent=1))

if __name__ == "__main__":
    main()
