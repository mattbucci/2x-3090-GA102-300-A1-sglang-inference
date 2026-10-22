#!/usr/bin/env python3
"""Paired resolve comparison of two scored cells (A/B lane vs its control).

`paired_lane_compare.py` reads the rollout shape (walls, empties, elapsed) and
`ab_lane_receipt.py` the plugin engagement; this is the post-score half: the
same-instance resolve table from the two swebench reports, the discordant
pairs and McNemar's exact test on them, per-repo deltas, and the wall / empty /
scorer-error counts — the numbers the README and the cross-team relay quote.

Usage:
  python evals/swebench/paired_resolve.py <control-cell> <treatment-cell> [--out receipt.json]
Cells are run dirs under evals/swebench/runs/ holding predictions.jsonl and
scores-docker/*.report.json. Exit 2 when either cell has no report.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from math import comb
from pathlib import Path


def load_report(cell: Path) -> dict:
    reps = sorted((cell / "scores-docker").glob("*.report.json"))
    if not reps:
        sys.exit(f"no scores-docker/*.report.json under {cell}")
    return json.loads(reps[-1].read_text())


def load_rows(cell: Path) -> dict[str, dict]:
    rows = {}
    for line in (cell / "predictions.jsonl").read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[d["instance_id"]] = d
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on the discordant counts (b = only-control,
    c = only-treatment): binomial test of b against n = b + c at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def shape(rows: dict[str, dict], rep: dict) -> dict:
    return {
        "n_pred": len(rows),
        "resolved": len(rep.get("resolved_ids", [])),
        "walls": sum(1 for d in rows.values() if d.get("rollout_returncode") == 124 or d.get("model_timeout")),
        "empties": sum(1 for d in rows.values() if not (d.get("model_patch") or "").strip()),
        "errors": len(rep.get("error_ids", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--out", help="write the JSON receipt here (stdout table always printed)")
    a = ap.parse_args()
    ctl, trt = Path(a.control), Path(a.treatment)
    rc, rt = load_report(ctl), load_report(trt)
    rows_c, rows_t = load_rows(ctl), load_rows(trt)
    paired = sorted(set(rows_c) & set(rows_t))
    res_c, res_t = set(rc.get("resolved_ids", [])), set(rt.get("resolved_ids", []))
    both = [i for i in paired if i in res_c and i in res_t]
    only_c = [i for i in paired if i in res_c and i not in res_t]
    only_t = [i for i in paired if i not in res_c and i in res_t]
    neither = [i for i in paired if i not in res_c and i not in res_t]
    per_repo = collections.defaultdict(lambda: [0, 0, 0])
    for i in paired:
        repo = i.rsplit("-", 1)[0]
        per_repo[repo][0] += 1
        per_repo[repo][1] += i in res_c
        per_repo[repo][2] += i in res_t
    out = {
        "control": ctl.name, "treatment": trt.name, "paired": len(paired),
        "control_shape": shape(rows_c, rc), "treatment_shape": shape(rows_t, rt),
        "resolved_control_paired": len(both) + len(only_c),
        "resolved_treatment_paired": len(both) + len(only_t),
        "both": len(both), "only_control": only_c, "only_treatment": only_t, "neither": len(neither),
        "mcnemar_exact_p": round(mcnemar_exact(len(only_c), len(only_t)), 4),
        "per_repo": {r: {"n": v[0], "control": v[1], "treatment": v[2], "delta": v[2] - v[1]}
                     for r, v in sorted(per_repo.items())},
    }
    sc, st = out["control_shape"], out["treatment_shape"]
    print(f"{'':28s} {'control':>10s} {'treatment':>10s}")
    print(f"{'cell':28s} {ctl.name[:10]:>10s} {trt.name[:10]:>10s}")
    for k in ("n_pred", "resolved", "walls", "empties", "errors"):
        print(f"{k:28s} {sc[k]:10d} {st[k]:10d}")
    print(f"{'paired':28s} {out['paired']:10d}")
    print(f"{'both / only-c / only-t / neither':28s} {len(both)} / {len(only_c)} / {len(only_t)} / {len(neither)}")
    print(f"{'McNemar exact p':28s} {out['mcnemar_exact_p']:10.4f}")
    for r, v in out["per_repo"].items():
        print(f"  {r:36s} n={v['n']:3d}  {v['control']:3d} -> {v['treatment']:3d}  ({v['delta']:+d})")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
