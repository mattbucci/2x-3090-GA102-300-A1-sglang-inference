#!/usr/bin/env python3
"""Detect scaffold-side context resets (compaction) from the SGLang server log.

For every instance of a rollout lane, reconstruct the sequence of prompt sizes
the server saw (`Prefill batch ... #new-token + #cached-token`, windowed by the
instance's `# elapsed` and log mtime — lanes run one container at a time, so
the window is unambiguous) and count *drops*: a prompt that falls below half
of the running peak after the peak passed 20K. A drop means the scaffold sent a
much smaller context than a moment ago — an auto-compaction, a sub-agent's
fresh session (opencode's task tool), a DCP prune, or the cleanup prompt's
fresh session. For headless pi (little-coder) with no sub-agents, drops ARE
compactions. Lanes whose instance script runs a cleanup pass get one free drop
per completed instance (timeouts are killed before cleanup, so none).

This is how the little-coder 32K-fallback loop was found (qwen38 rtk lane,
2026-09-11): pi falls back to the provider's first models.json entry for an
unknown model id (contextWindow 32768), compacts when the session nears it,
and can loop compaction until the timeout. Use it as the passive context-budget
check for any lane: a healthy lane on a 262K server shows ~0 compactions.

Usage:
  prompt_sawtooth.py --lane evals/swebench/runs/<preset>-<scaffold>-v2 \
      --server-log /tmp/run-model-cycle-logs/<preset>/server.log [--cleanup-drop] [--json out.json]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

PREFILL_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*#new-token: (\d+), #cached-token: (\d+)")
ELAPSED_RE = re.compile(r"^# elapsed ([0-9.]+)s\s+rc=(-?\d+)", re.M)


def load_prefills(path: Path) -> list[tuple[datetime, int]]:
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Prefill batch" not in line:
                continue
            m = PREFILL_RE.match(line)
            if m:
                out.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)) + int(m.group(3))))
    return out


def drops(series: list[int], peak_min: int = 20000) -> tuple[int, int]:
    run = peak = 0
    n = 0
    for v in series:
        if run > peak_min and v < run * 0.5 and v > 200:
            n += 1
        run = max(run, v) if v >= run * 0.5 else v
        peak = max(peak, v)
    return n, peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True, type=Path)
    ap.add_argument("--server-log", required=True, type=Path)
    ap.add_argument("--cleanup-drop", action="store_true",
                    help="lane runs a cleanup prompt after the main session: discount one drop per non-timeout instance")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    pre = load_prefills(a.server_log)
    ts = [t for t, _ in pre]
    rows = {}
    for log in sorted((a.lane / "logs").glob("*.log")):
        head = log.read_text(errors="replace")[:4000]
        m = ELAPSED_RE.search(head)
        if not m:
            continue
        el, rc = float(m.group(1)), int(m.group(2))
        end = datetime.fromtimestamp(log.stat().st_mtime)
        start = end - timedelta(seconds=el)
        seg = [n for _, n in pre[bisect.bisect_left(ts, start):bisect.bisect_right(ts, end)]]
        if not seg:
            continue
        nd, peak = drops(seg)
        timeout = rc == 124
        if a.cleanup_drop and not timeout:
            nd = max(0, nd - 1)
        pred = a.lane / "predictions" / f"{log.stem}.diff"
        size = pred.stat().st_size if pred.exists() else None
        rows[log.stem] = {
            "compactions": nd, "peak_prompt": peak, "n_prefill": len(seg), "elapsed": el,
            "outcome": "timeout" if timeout else "patched" if size else "empty" if size == 0 else "no_prediction",
        }
    if not rows:
        print("no instances overlap the server log window")
        return 1

    def summ(vs):
        return {
            "n": len(vs),
            "patched": sum(1 for v in vs if v["outcome"] == "patched"),
            "empty": sum(1 for v in vs if v["outcome"] == "empty"),
            "timeout": sum(1 for v in vs if v["outcome"] == "timeout"),
            "median_wall_s": round(statistics.median(v["elapsed"] for v in vs)),
            "median_peak_prompt": round(statistics.median(v["peak_prompt"] for v in vs)),
        }

    comp = [v for v in rows.values() if v["compactions"] > 0]
    noc = [v for v in rows.values() if v["compactions"] == 0]
    by_outcome = collections.defaultdict(list)
    for v in rows.values():
        by_outcome[v["outcome"]].append(v)
    rep = {
        "lane": str(a.lane), "instances": len(rows),
        "compaction_events": sum(v["compactions"] for v in rows.values()),
        "sessions_compacted": len(comp),
        "max_peak_prompt": max(v["peak_prompt"] for v in rows.values()),
        "sessions_peak_over_32k": sum(1 for v in rows.values() if v["peak_prompt"] > 32768),
        "compacted": summ(comp) if comp else None,
        "never_compacted": summ(noc) if noc else None,
        "by_outcome": {k: {**summ(vs), "sessions_compacted": sum(1 for v in vs if v["compactions"] > 0),
                           "median_compactions_when_any": statistics.median([v["compactions"] for v in vs if v["compactions"] > 0] or [0])}
                       for k, vs in by_outcome.items()},
        "per_instance": rows,
    }
    print(f"{a.lane.name}: {len(rows)} instances, {rep['compaction_events']} compaction events in "
          f"{len(comp)} sessions ({100 * len(comp) / len(rows):.0f}%); max prompt {rep['max_peak_prompt']}, "
          f"{rep['sessions_peak_over_32k']} sessions ever above 32K")
    print("subset            n  patched  empty  timeout  median wall  median peak prompt")
    for nm, s in (("compacted", rep["compacted"]), ("never compacted", rep["never_compacted"])):
        if s:
            print(f"{nm:16s} {s['n']:3d}  {s['patched']:5d}  {s['empty']:5d}  {s['timeout']:5d}    {s['median_wall_s']:5d} s   {s['median_peak_prompt']:7d}")
    print("by outcome:")
    for k, s in rep["by_outcome"].items():
        print(f"  {k:10s} n={s['n']:3d} compacted={s['sessions_compacted']:3d} median compactions(when any)={s['median_compactions_when_any']}")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
