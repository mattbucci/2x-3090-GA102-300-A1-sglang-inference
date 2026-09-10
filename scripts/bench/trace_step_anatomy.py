#!/usr/bin/env python3
"""Per-step kernel anatomy of an SGLang torch-profiler trace.

Reads one `*.trace.json(.gz)` written by `POST /start_profile` (SGLang wraps
every scheduler step in a `step[DECODE bs=N]` / `step[EXTEND bs=N toks=M]`
user_annotation) and prints, per step kind:

  * CPU step wall, GPU span (first kernel start -> last kernel end), and the
    kernel-time sum (sum > span means multi-stream overlap);
  * an EXPOSED-time table from a sweep line over the GPU timeline — time when
    only one category is running is charged to it, overlapped time is reported
    as `a+b`, and gaps with nothing running are `idle-gap`. This is the number
    that matters for a lever estimate: a kernel hidden behind another stream
    costs nothing (the Qwen3-Next GDN `in_proj_ba` fp16 gemv looks like 10% of
    the step by kernel-sum but is 2% exposed — it rides the alt stream under
    the Marlin qkvz GEMM);
  * the top kernels by ms/step.

Kernels are attributed to a step by the launch timestamp of their cuda_runtime
correlation record (graph-replayed kernels carry the cudaGraphLaunch record).

Usage:
    scripts/bench/trace_step_anatomy.py <trace.json.gz> [--top N] [--skip K]
        --skip K   ignore the first K steps of each kind (profiler warm-up)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import re
import sys


def category(name: str) -> str:
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "marlin" in n:
        return "marlin(int4)"
    if "gemvx" in n or "splitk" in n or "cublas" in n or "sm80_xmma" in n or "cutlass_kernel_gemm" in n:
        return "gemm/gemv(fp16)"
    if "gated_delta" in n or "causal_conv" in n or "gdn" in n or "chunk_fwd" in n or "recompute_w_u" in n or "kkt_solve" in n:
        return "gdn"
    if "flashinfer::batch" in n or "mergestates" in n or "attention" in n or "attn" in n:
        return "attention"
    if "norm" in n:
        return "norm"
    if "act_and_mul" in n or "sigmoid" in n or "silu" in n or "gelu" in n:
        return "act"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if any(k in n for k in ("sampl", "top_k", "top_p", "softmax", "argmax", "multinomial", "sort", "radix")):
        return "sampling"
    return "misc"


def load(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as fh:
        return json.load(fh)["traceEvents"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--skip", type=int, default=3)
    args = ap.parse_args()

    ev = load(args.trace)
    steps = sorted(
        (e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")),
        key=lambda e: e["ts"],
    )
    if not steps:
        print("no step[...] annotations in trace", file=sys.stderr)
        return 1
    launch_ts = {
        e["args"]["correlation"]: e["ts"]
        for e in ev
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    starts = [s["ts"] for s in steps]

    def step_of(ts: float):
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0:
            return None
        s = steps[i]
        return i if ts <= s["ts"] + s["dur"] else None

    per_step: dict[int, list[dict]] = collections.defaultdict(list)
    for e in gpu:
        ts = launch_ts.get(e.get("args", {}).get("correlation"), e["ts"])
        i = step_of(ts)
        if i is not None:
            per_step[i].append(e)

    kinds: dict[str, list[int]] = collections.defaultdict(list)
    for i, s in enumerate(steps):
        kinds[re.sub(r" toks=\d+", "", s["name"])].append(i)

    for kind, idx in kinds.items():
        idx = [i for i in idx[args.skip:] if per_step.get(i)]
        if not idx:
            continue
        n = len(idx)
        cpu = sum(steps[i]["dur"] for i in idx)
        span = ksum = 0.0
        exposed: dict[str, float] = collections.defaultdict(float)
        names: dict[str, float] = collections.defaultdict(float)
        for i in idx:
            ks = per_step[i]
            t0 = min(e["ts"] for e in ks)
            t1 = max(e["ts"] + e["dur"] for e in ks)
            span += t1 - t0
            ksum += sum(e["dur"] for e in ks)
            for e in ks:
                names[e["name"]] += e["dur"]
            # sweep line: charge each elementary interval to the set of running categories
            pts = sorted({e["ts"] for e in ks} | {e["ts"] + e["dur"] for e in ks})
            for a, b in zip(pts, pts[1:]):
                run = {category(e["name"]) for e in ks if e["ts"] <= a and e["ts"] + e["dur"] >= b}
                exposed["idle-gap" if not run else "+".join(sorted(run))] += b - a
        print(f"\n=== {kind} × {n} steps (after skipping {args.skip}) ===")
        print(
            f"  per step: CPU wall {cpu / n / 1e3:.2f} ms | GPU span {span / n / 1e3:.2f} ms | "
            f"kernel-sum {ksum / n / 1e3:.2f} ms"
        )
        tot = sum(exposed.values())
        print("  exposed time (sweep line):")
        for c, v in sorted(exposed.items(), key=lambda x: -x[1]):
            if v / tot >= 0.002:
                print(f"    {v / n / 1e3:7.3f} ms/step  {100 * v / tot:5.1f}%  {c}")
        print(f"  top {args.top} kernels (ms/step, count/step):")
        counts: dict[str, int] = collections.Counter(e["name"] for i in idx for e in per_step[i])
        for nm, v in sorted(names.items(), key=lambda x: -x[1])[: args.top]:
            print(f"    {v / n / 1e3:7.3f}  {counts[nm] / n:6.1f}  {nm[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
