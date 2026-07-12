#!/usr/bin/env python3
"""Compare v0.5.15 validation receipts against the v0.5.14 baselines.

For every benchmarks/quality/<preset>-v0515.json with a matching -v0514.json,
print MMLU / HumanEval / needle / thinking side-by-side plus the capability
summary, and flag regressions. Exit 1 if any regression beyond tolerance.

Tolerances: MMLU +-2 questions (sampling noise at N=30); HE +-2 (N=25);
needle any drop flags; caps any newly-failing check flags.
"""
import json
import sys
from pathlib import Path

QDIR = Path(__file__).resolve().parents[2] / "benchmarks" / "quality"

MMLU_TOL = 2 / 30
HE_TOL = 2 / 25


def needle_rate(d):
    res = d.get("needle", {}).get("results", [])
    if not res:
        return None
    return sum(1 for r in res if r.get("found")) / len(res)


def load(tag):
    p = QDIR / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def caps(tag):
    p = QDIR / f"cap-{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    inner = d.get(tag) or next(iter(d.values()))
    return inner.get("checks", {})


def main():
    presets = sorted(
        p.name[: -len("-v0515.json")]
        for p in QDIR.glob("*-v0515.json")
        if not p.name.startswith("cap-")
    )
    if not presets:
        print("no *-v0515.json receipts yet")
        return 0

    regressions = []
    print(f"{'preset':<16} {'MMLU 14→15':<13} {'HE 14→15':<13} {'needle':<13} {'think':<11} caps")
    for preset in presets:
        old, new = load(f"{preset}-v0514"), load(f"{preset}-v0515")
        if not new:
            continue
        row = [f"{preset:<16}"]

        def cell(get, tol, name, fmt="{:.2f}"):
            def safe(d):
                try:
                    return get(d) if d else None
                except (KeyError, TypeError):
                    return None
            ov, nv = safe(old), safe(new)
            s = f"{fmt.format(ov) if ov is not None else '—'}→{fmt.format(nv) if nv is not None else '—'}"
            if ov is not None and nv is not None and nv < ov - tol - 1e-9:
                regressions.append(f"{preset}: {name} {ov:.3f}->{nv:.3f}")
                s += " ⚠"
            return s

        row.append(f"{cell(lambda d: d['mmlu']['accuracy'], MMLU_TOL, 'mmlu'):<13}")
        row.append(f"{cell(lambda d: d['humaneval']['pass_rate'], HE_TOL, 'humaneval'):<13}")
        row.append(f"{cell(needle_rate, 0.0, 'needle'):<13}")
        row.append(
            f"{cell(lambda d: d['thinking'].get('clean_answer_rate'), 0.0, 'think'):<11}"
        )

        oc, nc = caps(f"{preset}-v0514"), caps(f"{preset}-v0515")
        if nc:
            passed = sum(1 for c in nc.values() if c.get("passed"))
            row.append(f"{passed}/{len(nc)}")
            if oc:
                for name, c in nc.items():
                    if not c.get("passed") and oc.get(name, {}).get("passed"):
                        regressions.append(f"{preset}: capability '{name}' newly failing")
                        row.append(f"⚠{name}")
        else:
            row.append("—")
        print(" ".join(row))

    print()
    if regressions:
        print("REGRESSIONS:")
        for r in regressions:
            print(" -", r)
        return 1
    print("NO REGRESSIONS (within tolerance) across", len(presets), "presets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
