#!/usr/bin/env python3
"""Markdown tables for a stack flip's status doc (patches/v<ver>-rebase-status.md).

    flip_tables.py --old v0520 --new v0522 [--since 2026-10-19]

Table 1 — quality, one row per preset with a benchmarks/quality/<preset>-<new>.json
receipt: `MMLU · HE · needle · think-clean` as counts for the new stack, the same
for the old stack, and the capability-matrix pass count from cap-<preset>-<new>.json.
Table 2 — decode tok/s for the tripwire presets: benchmarks/baselines.json (the kept
baseline, old) → the newest benchmarks/regression/<preset>-<date>.json dated on/after
--since (the flip campaign's phase-2 run, new), at 1K / 32K / deep with the % delta.
Read-only; pass/fail is compare_flip_receipts.py's job, this only renders.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QDIR = ROOT / "benchmarks" / "quality"
RDIR = ROOT / "benchmarks" / "regression"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def counts(d):
    if not d:
        return "—"
    m, h = d.get("mmlu") or {}, d.get("humaneval") or {}
    n = (d.get("needle") or {}).get("results") or []
    t = (d.get("thinking") or {}).get("clean_answer_rate")
    needle = f"{sum(1 for r in n if r.get('found'))}/{len(n)}" if n else "—"
    think = f"{t:.1f}" if isinstance(t, (int, float)) else "—"
    return f"{m.get('correct', '—')}/{m.get('total', '—')} · {h.get('passed', '—')}/{h.get('total', '—')} · {needle} · {think}"


def caps(tag):
    d = load(QDIR / f"cap-{tag}.json")
    if not d:
        return "—"
    inner = (d.get(tag) or next(iter(d.values()))).get("checks", {})
    return f"{sum(1 for c in inner.values() if c.get('passed'))}/{len(inner)}"


def quality_table(old, new):
    presets = sorted(p.name[: -len(f"-{new}.json")] for p in QDIR.glob(f"*-{new}.json")
                     if not p.name.startswith("cap-"))
    print(f"| preset | {new} (MMLU · HE · needle · think-clean) | {old} | caps |")
    print("|---|---|---|---|")
    for p in presets:
        print(f"| {p} | {counts(load(QDIR / f'{p}-{new}.json'))} | {counts(load(QDIR / f'{p}-{old}.json'))} | {caps(f'{p}-{new}')} |")


def newest_run(preset, since):
    best = None
    for p in RDIR.glob(f"{preset}-20??-??-??.json"):
        m = re.search(r"(\d{4}-\d\d-\d\d)\.json$", p.name)
        if m and m.group(1) >= since and (best is None or m.group(1) > best[0]):
            best = (m.group(1), p)
    return best


def speed_table(since):
    base = load(ROOT / "benchmarks" / "baselines.json") or {}
    rows = []
    for preset, b in base.items():
        if preset.startswith("_"):
            continue
        run = newest_run(preset, since)
        if not run:
            rows.append(f"| {preset} | (no regression receipt since {since}) | | |")
            continue
        sweep = {str(s["actual_input_tokens"]): s for s in (load(run[1]) or {}).get("context_sweep", [])}
        cells = []
        for key in ("1024", "32768", "deep"):
            ob = b.get(key) or {}
            o = ob.get("tok_per_sec")
            n = (sweep.get(str(ob.get("actual_input_tokens"))) or {}).get("tok_per_sec")
            if o and n:
                cells.append(f"{o} → {n} ({(n - o) / o * 100:+.1f} %)".replace("(-", "(−"))
            else:
                cells.append(f"{o} → —")
        rows.append(f"| {preset} | " + " | ".join(cells) + f" |  <!-- {run[1].name} -->")
    deep_label = next((b.get("deep", {}).get("label") for k, b in base.items() if not k.startswith("_")), "deep")
    print(f"| preset (kept baseline → newest run since {since}) | 1K tok/s | 32K tok/s | deep ({deep_label:,}) tok/s |")
    print("|---|---|---|---|")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--since", default="1970-01-01", help="earliest regression-receipt date for the new column")
    a = ap.parse_args()
    quality_table(a.old, a.new)
    print()
    speed_table(a.since)


if __name__ == "__main__":
    main()
