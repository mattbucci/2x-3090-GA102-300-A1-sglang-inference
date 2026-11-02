#!/usr/bin/env python
"""Top CPU-side operators in an SGLang torch-profiler trace window (companion to trace_step_anatomy.py, which
attributes GPU kernels). Lists the longest CPU events (cpu_op / python_function / cuda_runtime) by total duration,
so a step whose GPU timeline is mostly idle gap can be read from the host side.
    python scripts/bench/trace_cpu_ops.py <trace.json.gz> [--top 30]"""
import argparse, collections, gzip, json, sys

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--top", type=int, default=30); a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    with op(a.trace, "rt") as f: j = json.load(f)
    ev = j["traceEvents"] if isinstance(j, dict) else j
    by_cat = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    span = [None, None]
    for e in ev:
        if e.get("ph") != "X": continue
        cat = e.get("cat", ""); dur = float(e.get("dur", 0)); name = e.get("name", "")[:110]
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ts = float(e["ts"]); span[0] = ts if span[0] is None else min(span[0], ts); span[1] = ts + dur if span[1] is None else max(span[1], ts + dur); continue
        by_cat[cat][name][0] += dur; by_cat[cat][name][1] += 1
    if span[0] is not None: print(f"GPU timeline span: {(span[1]-span[0])/1000:.1f} ms")
    for cat in ("cpu_op", "python_function", "cuda_runtime", "user_annotation", "cpu_instant_event"):
        rows = sorted(by_cat.get(cat, {}).items(), key=lambda kv: -kv[1][0])[: a.top if cat in ("cpu_op", "python_function") else 10]
        if not rows: continue
        print(f"\n== {cat}: top by total CPU time (ms, count)")
        for name, (dur, cnt) in rows: print(f"  {dur/1000:9.2f} ms  x{cnt:<6} {name}")

if __name__ == "__main__":
    sys.exit(main())
