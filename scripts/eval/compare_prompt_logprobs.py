#!/usr/bin/env python3
"""Compare two per-position next-token distributions over the same prompt ids.

Inputs are the JSON files written by ``flint_cpu_reference.py`` (reference forward, BF16 or
dequantized INT4) and ``served_prompt_logprobs.py`` (live SGLang server, ``return_logprob``).
Both carry ``next_lp[t]`` (logprob of ``input_ids[t+1]`` under the distribution at position t)
and ``top[t]`` (top-20 ``[id, lp]`` pairs at t).

Reports, overall and by position band:
  * top-1 agreement
  * mean / p95 / max |Δ next-token logprob|
  * truncated KL(A||B) over A's top-20 (B's mass for an id A ranks but B does not is bounded
    by B's 20th logprob — a lower bound on the divergence, flagged as such)
  * mean top-20 mass under each (1 - mass = tail that sampling at T=1/top-p .95 can reach)

Usage: compare_prompt_logprobs.py A.json B.json [--label-a X] [--label-b Y] [--bands 256,1024]
"""
import argparse
import json
import math
import sys


def load(path):
    d = json.load(open(path))
    return d


def mass(top):
    return sum(math.exp(lp) for _, lp in top)


def trunc_kl(ta, tb):
    """KL(A||B) restricted to A's top list; ids B lacks get B's floor (its last lp)."""
    b = {i: lp for i, lp in tb}
    floor = tb[-1][1]
    kl = 0.0
    missing = 0
    for i, lpa in ta:
        lpb = b.get(i)
        if lpb is None:
            lpb = floor
            missing += 1
        kl += math.exp(lpa) * (lpa - lpb)
    return kl, missing


def stats(vals):
    if not vals:
        return (float("nan"),) * 3
    s = sorted(vals)
    return sum(s) / len(s), s[int(0.95 * (len(s) - 1))], s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--bands", default="256,1024")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    if A["input_ids"] != B["input_ids"]:
        sys.exit("input_ids differ between the two files")
    T = len(A["input_ids"])
    edges = [0] + [int(x) for x in args.bands.split(",") if x] + [T]
    rows = []
    out = {"a": args.a, "b": args.b, "T": T, "bands": []}
    for lo, hi in list(zip(edges, edges[1:])) + [(0, T)]:
        agree = n = 0
        dl, kls, ma, mb, miss = [], [], [], [], 0
        for t in range(lo, min(hi, T - 1)):
            ta, tb = A["top"][t], B["top"][t]
            la, lb = A["next_lp"][t], B["next_lp"][t]
            if not ta or not tb or la is None or lb is None:
                continue
            n += 1
            agree += ta[0][0] == tb[0][0]
            dl.append(abs(la - lb))
            k, m = trunc_kl(ta, tb)
            kls.append(k)
            miss += m
            ma.append(mass(ta))
            mb.append(mass(tb))
        if n == 0:
            continue
        dm, dp95, dmax = stats(dl)
        km, kp95, kmax = stats(kls)
        band = {"lo": lo, "hi": min(hi, T - 1), "n": n, "top1_agree": agree / n,
                "dlp_mean": dm, "dlp_p95": dp95, "dlp_max": dmax,
                "kl_mean": km, "kl_p95": kp95, "kl_max": kmax, "kl_missing_ids": miss,
                "top20_mass_a": sum(ma) / n, "top20_mass_b": sum(mb) / n}
        out["bands"].append(band)
        label = "all" if (lo, hi) == (0, T) else f"{lo}-{min(hi, T - 1)}"
        rows.append(f"| {label} | {n} | {agree / n:.3f} | {dm:.3f} / {dp95:.3f} / {dmax:.2f} | "
                    f"{km:.4f} / {kp95:.4f} / {kmax:.3f} | {sum(ma) / n:.4f} / {sum(mb) / n:.4f} |")
    print(f"A = {args.label_a}: {args.a}\nB = {args.label_b}: {args.b}\nT = {T}\n")
    print(f"| positions | n | top-1 agree | abs dlp(next) mean / p95 / max | KL(A||B) top-20 mean / p95 / max | top-20 mass A / B |")
    print("|---|---:|---:|---|---|---|")
    print("\n".join(rows))
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
