#!/usr/bin/env python3
"""Exposure study: resolved rate of exposed vs isolated instances in a
--network=host era cell (never a bake-off number — a harness-defect receipt).

Joins <run>/leak-audit.json (audit_leakage.py: per-instance `exposed`,
`attempted`, `overlap`) with <run>/scores-docker-summary.json (score_docker.py
per_instance status). Prints one table row per run and writes
<run>/exposure-study.json.

Usage: exposure_study.py --run <dir> [--run <dir> ...] [--json OUT]
"""
import argparse
import json
import statistics
from pathlib import Path


def resolved_set(summary):
    per = summary.get("per_instance", {})
    out = {}
    for iid, v in per.items():
        status = v if isinstance(v, str) else (v.get("status") or v.get("result") or "")
        out[iid] = status.lower() in ("resolved", "resolved_full") or status is True or v is True
    return out


def study(run: Path):
    leak = json.loads((run / "leak-audit.json").read_text())
    summ = json.loads((run / "scores-docker-summary.json").read_text())
    res = resolved_set(summ)
    groups = {"exposed": [], "attempted_not_exposed": [], "isolated": []}
    overlap = {"exposed": [], "isolated": []}
    for iid, inst in leak["instances"].items():
        if iid not in res:
            continue
        g = "exposed" if inst["exposed"] else ("attempted_not_exposed" if inst["attempted"] else "isolated")
        groups[g].append(res[iid])
        if g in overlap and inst.get("overlap") is not None:
            overlap[g].append(inst["overlap"])
    rows = {}
    for g, xs in groups.items():
        n = len(xs)
        k = sum(xs)
        rows[g] = {"n": n, "resolved": k, "rate": round(100 * k / n, 1) if n else None}
    for g, xs in overlap.items():
        rows[g]["overlap_median"] = round(statistics.median(xs), 2) if xs else None
    out = {
        "run": run.name,
        "scored": len(res),
        "resolved_total": sum(res.values()),
        "rate_total": round(100 * sum(res.values()) / len(res), 1) if res else None,
        "groups": rows,
    }
    (run / "exposure-study.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    results = [study(Path(r)) for r in a.run]
    print("| cell | scored | resolved (all) | exposed: resolved/n | isolated: resolved/n | attempted-not-exposed | overlap med exposed / isolated |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        g = r["groups"]
        e, i, t = g["exposed"], g["isolated"], g["attempted_not_exposed"]
        print(f"| {r['run']} | {r['scored']} | {r['resolved_total']} ({r['rate_total']} %) | "
              f"**{e['resolved']}/{e['n']} ({e['rate']} %)** | **{i['resolved']}/{i['n']} ({i['rate']} %)** | "
              f"{t['resolved']}/{t['n']} ({t['rate']} %) | {e['overlap_median']} / {i['overlap_median']} |")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
