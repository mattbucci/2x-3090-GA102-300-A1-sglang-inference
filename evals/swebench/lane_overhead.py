#!/usr/bin/env python3
"""Where does a rollout lane's wall clock go? GPU duty cycle + harness overhead.

Three read-only views, any subset (receipt:
benchmarks/quality/gpu-utilization-harness-overhead-2026-09-26.md):

  --telemetry CSV [--since T --until T]   nvidia-smi log from scripts/gpu_telemetry.sh:
        per GPU busy % (util >= 50 %), SM clock / power while busy, throttle
        reasons, temps, and the idle-gap histogram (a 5-10 min gap per instance
        is the image build + container boot + cleanup pass; longer gaps are
        stalls).
  --rollout-log LOG                       the cycle's rollout-<scaffold>.log: buildkit
        `#N DONE t` per instance -> build seconds per instance, slowest steps,
        build share of `rollout_seconds`.
  --run DIR                               a runs/<cell>: from the opencode JSON event
        timestamps in logs/<iid>.log, per instance: pre (build+boot -> first
        event), session, post (last event -> exit). post ~0 s means the
        cleanup pass ran; a post pinned at 120 s means it was killed by its
        `timeout 120` (opencode+DCP lane, 2026-09-26: the plugin loader's
        offline npm attempt eats the budget). Non-opencode scaffolds emit no
        epoch-ms events; only pre/post via other markers are not attempted.

Usage:
    python evals/swebench/lane_overhead.py --run evals/swebench/runs/qwen38-opencode-dcp-v3 \\
        --rollout-log /tmp/run-model-cycle-logs/qwen38/rollout-opencode-dcp.log \\
        --telemetry /var/tmp/gpu-telemetry/nvidia-smi-20260922T201336.csv --since 2026-09-24T13:38
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime

BUSY_UTIL = 50
WALL_S = 1795


def telemetry(path: str, since: str | None, until: str | None) -> None:
    a = datetime.fromisoformat(since) if since else datetime.min
    b = datetime.fromisoformat(until) if until else datetime.max
    rows = []
    for r in csv.reader(open(path)):
        if not r or r[0].startswith("timestamp"):
            continue
        try:
            ts = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            continue
        if not (a <= ts < b):
            continue
        num = lambda s: float(s.split()[0])  # "259.45 W" -> 259.45
        rows.append((ts, int(r[1]), int(r[2]), num(r[3]), num(r[4]), num(r[6]), r[10].strip(), num(r[11])))
    if not rows:
        print("telemetry: no samples in window")
        return
    step = (rows[2][0] - rows[0][0]).total_seconds() or 30  # two GPUs per sample
    for gpu in sorted({r[1] for r in rows}):
        w = [r for r in rows if r[1] == gpu]
        busy = [r for r in w if r[5] >= BUSY_UTIL]
        print(f"GPU{gpu}: samples={len(w)} ({len(w) * step / 3600:.1f} h)  busy(util>={BUSY_UTIL}%)={100 * len(busy) / len(w):.1f}%  "
              f"mean util={st.mean(r[5] for r in w):.0f}%  busy SM med={st.median([r[4] for r in busy] or [0]):.0f} MHz  "
              f"busy power med={st.median([r[3] for r in busy] or [0]):.0f} W  temp max={max(r[2] for r in w)} "
              f"med={st.median(r[2] for r in w):.0f}  fan max={max(r[7] for r in w):.0f}%  "
              f"throttle(busy)={dict(Counter(r[6] for r in busy).most_common(2))}")
    w = [r for r in rows if r[1] == rows[0][1]]
    gaps, cur = [], 0
    for r in w:
        if r[5] < BUSY_UTIL:
            cur += 1
        elif cur:
            gaps.append(cur * step)
            cur = 0
    if cur:
        gaps.append(cur * step)
    edges = [(0, "<1m"), (60, "1-2m"), (120, "2-5m"), (300, "5-10m"), (600, ">=10m")]
    bucket = lambda s: [n for lo, n in edges if s >= lo][-1]
    by_n, by_h = Counter(), defaultdict(float)
    for s in gaps:
        by_n[bucket(s)] += 1
        by_h[bucket(s)] += s / 3600
    print(f"idle gaps (GPU{w[0][1]}): n={len(gaps)} total={sum(gaps) / 3600:.1f} h  "
          f"count={dict(by_n)}  hours={ {k: round(v, 1) for k, v in by_h.items()} }")


def rollout_log(path: str) -> None:
    inst, builds, steps, elapsed = None, {}, {}, {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[(\d+)/\d+\] (\S+)", line)
        if m:
            inst = m.group(2)
            builds[inst], steps[inst] = 0.0, {}
            continue
        if inst is None:
            continue
        m = re.match(r"#(\d+) DONE ([\d.]+)s", line)
        if m:
            builds[inst] += float(m.group(2))
            steps[inst][m.group(1)] = float(m.group(2))
            continue
        m = re.search(r"done rc=\d+ elapsed=([\d.]+)s", line)
        if m:
            elapsed[inst] = float(m.group(1))
    done = [k for k in builds if k in elapsed]
    if not done:
        print("rollout log: no completed instances")
        return
    b = [builds[k] for k in done]
    agg = defaultdict(list)
    for k in done:
        for s, v in steps[k].items():
            agg[s].append(v)
    top = sorted(agg.items(), key=lambda kv: -st.median(kv[1]))[:6]
    print(f"image build: instances={len(b)} per-instance median={st.median(b):.0f}s mean={st.mean(b):.0f}s "
          f"max={max(b):.0f}s sum={sum(b) / 3600:.1f}h  share of rollout_seconds (median)="
          f"{100 * st.median(builds[k] / elapsed[k] for k in done):.1f}%  slowest steps (median s): "
          f"{ {s: round(st.median(v)) for s, v in top} }")


def run_dir(path: str) -> None:
    rows = []
    for p in glob.glob(os.path.join(path, "logs", "*.log")):
        txt = open(p, errors="replace").read()
        m = re.search(r"# elapsed ([\d.]+)s\s+rc=(\d+)", txt)
        ts = [int(x) / 1000 for x in re.findall(r'"timestamp":(\d{13})', txt)]
        if not m or not ts:
            continue
        el, rc = float(m.group(1)), int(m.group(2))
        end = os.path.getmtime(p)
        start = end - el
        rows.append(dict(el=el, rc=rc, pre=min(ts) - start, sess=max(ts) - min(ts), post=end - max(ts),
                         sessions=len(set(re.findall(r'"sessionID":"([^"]+)"', txt))),
                         steps=len(re.findall(r'"reason":"stop"', txt))))
    if not rows:
        print(f"{path}: no opencode-style event logs")
        return
    ok = [r for r in rows if r["rc"] == 0 and r["el"] < WALL_S]
    walls = [r for r in rows if r["rc"] == 124]
    med = lambda k, s: st.median(r[k] for r in s) if s else float("nan")
    print(f"{os.path.basename(path)}: n={len(rows)} rc0={len(ok)} walls={len(walls)}  "
          f"rollout_seconds med={med('el', ok):.0f}s  pre(build+boot->first event) med={med('pre', rows):.0f}s  "
          f"session med={med('sess', ok):.0f}s  post(last event->exit) med={med('post', ok):.0f}s  "
          f"post>=100s on {sum(1 for r in ok if r['post'] >= 100)}/{len(ok)}  "
          f"cleanup session present on {sum(1 for r in ok if r['sessions'] >= 2)}/{len(ok)}")
    if walls:
        # >=100 s of silence before the kill: a long think that never finished a
        # step (the recall pattern) or, on a lane whose cleanup pass hangs, a
        # session that did finish and then lost its diff to the dead pass.
        print(f"  walls silent >=100s before the kill: {sum(1 for r in walls if r['post'] >= 100)}/{len(walls)}; "
              f"of those with a finished step: "
              f"{sum(1 for r in walls if r['post'] >= 100 and r['sessions'] >= 1 and r['steps'])}/{len(walls)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telemetry")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--rollout-log")
    ap.add_argument("--run", action="append", default=[])
    args = ap.parse_args()
    if not (args.telemetry or args.rollout_log or args.run):
        ap.error("give at least one of --telemetry / --rollout-log / --run")
    if args.telemetry:
        telemetry(args.telemetry, args.since, args.until)
    if args.rollout_log:
        rollout_log(args.rollout_log)
    for d in args.run:
        run_dir(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
