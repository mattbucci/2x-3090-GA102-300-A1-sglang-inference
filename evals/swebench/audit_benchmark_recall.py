#!/usr/bin/env python3
"""Measure how much of an opencode lane's reasoning budget goes to *benchmark
recall* -- the model recognising the task as a SWE-bench instance and trying to
remember the gold patch instead of working the issue.

Port of the R9700 audit of the same name (their `1f8ec11`, 2026-09-20: on their
v3 opencode lane every wall hit ended in one 15-30K-token think dominated by
this; 24/25 long thinks carried it, short turns rarely did). Same patterns and
the same summary shape so the two rigs' receipts read side by side. The 3090
lanes carry a different cue set -- work dir `/testbed` from the official
sweb.eval image (no instance id in the path, env or hostname), the official
image's own `SWE-bench` HEAD commit, and the prompt's "Do not modify tests" --
so the rates are not expected to match; the point is to see what our v3 budget
buys.

Reads the per-instance session snapshots docker_rollout.py writes for
audit_leakage.py (`<run>/sessions/<iid>/.local/share/opencode/opencode.db`),
so there is no directory/time join: every session in an instance's db belongs
to that instance (a `run` may open a child session; all are summed). Only
opencode is covered -- the other scaffolds keep their own session formats.
v2 (argv-era, --format json) cells have no snapshot and their logs carry no
reasoning parts, so they cannot be audited retroactively. Wall hits in the
qwen38 opencode v3 cell have an EMPTY snapshot (the in-script copy sat after
the scaffold; the 1800 s SIGKILL never reached it -- `no_snapshot` counts
them, and that cell's wall column reads 0 by construction); from the next
lane on docker_rollout.py snapshots the store live before the kill.

Usage:
    python evals/swebench/audit_benchmark_recall.py \\
        --run evals/swebench/runs/<cell> [--out receipt.json]
    (resolved verdicts are attached when the cell's scores-docker-summary.json exists;
    run_model_cycle.sh runs it after scoring on every opencode-family lane -- informational,
    it never gates a cell)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import statistics
import tempfile
from datetime import datetime
from pathlib import Path

RECALL = re.compile(
    r"swe.?bench|gold.?patch|try to remember|recall the (?:real|actual|upstream|original)"
    r"|from memory|the (?:fix|actual) (?:pr|commit) (?:was|is|should be)", re.I)
HUNT = re.compile(
    r"cached .{0,40}dataset|search all of /opt|/root/\.cache|\.cache/huggingface"
    r"|find / -name|grep -v swebench-work", re.I)
LONG_THINK_TOKENS = 3000  # one turn's output tokens
# A wall hit is the harness's SIGKILL at the 1800 s cap (rc=124). `rollout_seconds`
# is NOT a proxy for it: it starts before the per-instance image build (~1-3 min), so
# a session that finishes just under the wire can carry 1983 s with rc=0 and a full
# diff (qwen38 opencode v3 django-15996: 1785 s session, 1983 s elapsed, patched).
WALL_RC = 124
SNAPSHOT_DB = ".local/share/opencode/opencode.db"


def open_snapshot(db_path: Path, tmp: Path) -> sqlite3.Connection:
    # Copy db + -wal + -shm aside and open the copy: honours un-checkpointed WAL
    # pages (a read-only open of the snapshot cannot) and never touches the receipt.
    for f in db_path.parent.glob(db_path.name + "*"):
        shutil.copy2(f, tmp / f.name)
    return sqlite3.connect(tmp / db_path.name)


def reasoning_text(db: sqlite3.Connection, message_id: str) -> str:
    return " ".join(
        json.loads(pd).get("text", "")
        for (pd,) in db.execute("select data from part where message_id=?", (message_id,))
        if '"reasoning"' in pd)


def instance_turns(db: sqlite3.Connection) -> list[dict]:
    turns = []
    for mid, data in db.execute("select id, data from message order by time_created"):
        m = json.loads(data)
        if m.get("role") != "assistant":
            continue
        txt = reasoning_text(db, mid)
        turns.append({
            "output_tokens": (m.get("tokens") or {}).get("output") or 0,
            "reasoning_chars": len(txt),
            "recall_hits": len(RECALL.findall(txt)),
            "hunt_hits": len(HUNT.findall(txt)),
            "finished": bool(m.get("finish")),
        })
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="cell dir holding predictions.jsonl + sessions/<iid>/")
    ap.add_argument("--out", help="write the JSON receipt here")
    args = ap.parse_args()

    run = Path(args.run)
    pred_path = run / "predictions.jsonl"
    preds = {}
    for line in open(pred_path):
        r = json.loads(line)
        preds[r["instance_id"]] = r
    scores = {}
    summary_path = run / "scores-docker-summary.json"   # score_docker.py: per_instance iid -> status
    if summary_path.exists():
        per = json.load(open(summary_path)).get("per_instance") or {}
        scores = {iid: st == "resolved" for iid, st in per.items()}

    rows, no_snapshot = [], 0
    for iid, p in preds.items():
        db_path = run / "sessions" / iid / SNAPSHOT_DB
        if not db_path.exists():
            no_snapshot += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            db = open_snapshot(db_path, Path(td))
            n_sessions = db.execute("select count(*) from session").fetchone()[0]
            turns = instance_turns(db)
            db.close()
        chars = sum(t["reasoning_chars"] for t in turns)
        if chars < 500:
            continue
        long_t = [t for t in turns if t["output_tokens"] >= LONG_THINK_TOKENS
                  or (not t["finished"] and t["reasoning_chars"] >= 12_000)]
        secs = p.get("rollout_seconds")
        rows.append({
            "instance_id": iid,
            "sessions": n_sessions,
            "rollout_seconds": secs,
            "rollout_returncode": p.get("rollout_returncode"),
            "empty_patch": not p.get("model_patch"),
            "resolved": scores.get(iid),
            "turns": len(turns),
            "reasoning_chars": chars,
            "recall_hits": sum(t["recall_hits"] for t in turns),
            "hunt_hits": sum(t["hunt_hits"] for t in turns),
            "long_thinks": len(long_t),
            "long_thinks_with_recall": sum(1 for t in long_t if t["recall_hits"]),
            "long_think_chars": sum(t["reasoning_chars"] for t in long_t),
            "short_turns_with_recall": sum(1 for t in turns if t not in long_t and t["recall_hits"]),
        })

    n = len(rows)
    if not n:
        print(f"no instances with reasoning in snapshots ({no_snapshot} of {len(preds)} predictions had no snapshot)")
        return 1

    def rate(sel):
        return f"{sum(1 for r in rows if sel(r))}/{n}"

    buckets = [("0", lambda r: r["recall_hits"] == 0), ("1-9", lambda r: 1 <= r["recall_hits"] <= 9),
               ("10-29", lambda r: 10 <= r["recall_hits"] <= 29), ("30+", lambda r: r["recall_hits"] >= 30)]
    summary = {
        "predictions": len(preds),
        "no_snapshot": no_snapshot,
        "sessions": n,
        "sessions_with_recall": rate(lambda r: r["recall_hits"] > 0),
        "sessions_with_dataset_hunt": rate(lambda r: r["hunt_hits"] > 0),
        "long_thinks": sum(r["long_thinks"] for r in rows),
        "long_thinks_with_recall": sum(r["long_thinks_with_recall"] for r in rows),
        "short_turns_with_recall": f"{sum(r['short_turns_with_recall'] for r in rows)}/"
                                   f"{sum(r['turns'] - r['long_thinks'] for r in rows)}",
        "reasoning_share_in_long_thinks": round(
            sum(r["long_think_chars"] for r in rows) / max(1, sum(r["reasoning_chars"] for r in rows)), 3),
        "wall_hits": rate(lambda r: r["rollout_returncode"] == WALL_RC),
        "empty_patches": rate(lambda r: r["empty_patch"]),
        "by_recall_bucket": {},
    }
    for name, sel in buckets:
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        b = {"n": len(g),
             "wall_hits": sum(1 for r in g if r["rollout_returncode"] == WALL_RC),
             "empty_patches": sum(1 for r in g if r["empty_patch"]),
             "median_seconds": statistics.median(r["rollout_seconds"] or 0 for r in g),
             "median_reasoning_chars": statistics.median(r["reasoning_chars"] for r in g)}
        if scores:
            b["resolved"] = sum(1 for r in g if r["resolved"])
        summary["by_recall_bucket"][name] = b

    print(f"benchmark-recall audit: {run}")
    for k, v in summary.items():
        if k != "by_recall_bucket":
            print(f"  {k}: {v}")
    print("  recall hits | n | wall | empty | median s | median reasoning chars" + (" | resolved" if scores else ""))
    for name, b in summary["by_recall_bucket"].items():
        line = f"  {name:11s} | {b['n']:3d} | {b['wall_hits']:4d} | {b['empty_patches']:5d} | {b['median_seconds']:8.0f} | {b['median_reasoning_chars']:8.0f}"
        if scores:
            line += f" | {b['resolved']}"
        print(line)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run": str(run), "scored": bool(scores),
            "patterns": {"recall": RECALL.pattern, "hunt": HUNT.pattern, "long_think_tokens": LONG_THINK_TOKENS},
            "summary": summary, "instances": rows}, indent=1) + "\n")
        print(f"  receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
