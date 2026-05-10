#!/usr/bin/env python3
"""Aggregate bake-off scores into a per-model × per-scaffold resolved-rate
table. Reads scores-docker-summary.json from every
evals/swebench/runs/<preset>-<scaffold>-v2/ directory and writes the
combined Markdown table to evals/swebench/bake-off-<date>.md.

Also computes per-instance scaffold-disagreement (instances where one
scaffold solves it but another doesn't) for the same model — useful for
"is the model failing or is the scaffold failing" diagnosis.

Usage:
    python evals/swebench/aggregate_bakeoff.py
    python evals/swebench/aggregate_bakeoff.py --runs-dir evals/swebench/runs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="evals/swebench/runs",
                   help="Where the per-(preset, scaffold) result dirs live.")
    p.add_argument("--out", default=None,
                   help="Output Markdown file. Default: evals/swebench/bake-off-<date>.md")
    return p.parse_args()


SCAFFOLDS = ("opencode", "little-coder", "claw-code")


def discover_runs(runs_dir: Path):
    """Yield (preset, scaffold, summary_dict) tuples for each completed run."""
    if not runs_dir.exists():
        return

    pat = re.compile(r"^(.*)-(opencode|little-coder|claw-code)-v2$")
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        preset, scaffold = m.group(1), m.group(2)
        summary_path = d / "scores-docker-summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                summary = None
        else:
            summary = None
        yield preset, scaffold, d, summary


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()

    # Group: results[preset][scaffold] = summary
    results = defaultdict(dict)
    paths = defaultdict(dict)
    for preset, scaffold, run_dir, summary in discover_runs(runs_dir):
        results[preset][scaffold] = summary
        paths[preset][scaffold] = run_dir

    # Special-case: the legacy `coder-30b-docker-v2` dir is opencode but
    # doesn't follow the new naming. Pick it up if present.
    legacy = runs_dir / "coder-30b-docker-v2"
    if legacy.exists() and "coder-30b-eval" not in results:
        summary_path = legacy / "scores-docker-summary.json"
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass
        results.setdefault("coder-30b-eval", {})["opencode"] = summary
        paths.setdefault("coder-30b-eval", {})["opencode"] = legacy

    rows = []
    for preset in sorted(results.keys()):
        cells = [preset]
        for scaffold in SCAFFOLDS:
            s = results[preset].get(scaffold)
            if s is None:
                if scaffold in results[preset]:
                    cells.append("queued")
                else:
                    cells.append("—")
                continue
            r = s.get("resolved", 0)
            t = s.get("total_predictions", 0)
            rate = s.get("resolve_rate_pct")
            if rate is None and t:
                rate = round(100.0 * r / t, 1)
            cells.append(f"{r}/{t} = {rate}%" if t else "0/0")
        rows.append(cells)

    out_path = Path(args.out) if args.out else (
        runs_dir.parent / f"bake-off-{time.strftime('%Y-%m-%d')}.md"
    )

    lines = []
    lines.append(f"# SWE-bench Lite bake-off — {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("Per-model × per-scaffold resolved-rate. Each cell is "
                 "`<resolved>/<total> = <rate>%` from the official SWE-bench "
                 "Docker harness (`scores-docker-summary.json` per run dir).")
    lines.append("")
    lines.append("| Model preset | opencode | little-coder | claw-code |")
    lines.append("|--------------|:--------:|:------------:|:---------:|")
    for cells in rows:
        lines.append("| `" + cells[0] + "` | " + " | ".join(cells[1:]) + " |")
    lines.append("")

    # Scaffold disagreement: same model, different verdict per scaffold.
    lines.append("## Scaffold disagreement (same model, different verdict)")
    lines.append("")
    disagreements = []
    for preset, by_sc in results.items():
        per_inst = defaultdict(dict)
        for sc, s in by_sc.items():
            if not s:
                continue
            for iid, label in (s.get("per_instance") or {}).items():
                per_inst[iid][sc] = label
        for iid, by_sc_label in per_inst.items():
            labels = set(by_sc_label.values())
            if len(labels) > 1:
                disagreements.append((preset, iid, by_sc_label))
    if disagreements:
        lines.append(f"{len(disagreements)} disagreements found.")
        lines.append("")
        lines.append("| Model | Instance | opencode | little-coder | claw-code |")
        lines.append("|-------|----------|:--------:|:------------:|:---------:|")
        for preset, iid, by_sc_label in disagreements[:50]:
            cells = [preset, iid]
            for sc in SCAFFOLDS:
                cells.append(by_sc_label.get(sc, "—"))
            lines.append("| `" + cells[0] + "` | `" + cells[1] + "` | "
                         + " | ".join(cells[2:]) + " |")
        if len(disagreements) > 50:
            lines.append(f"\n_({len(disagreements)-50} more truncated; see raw "
                         f"per-instance jsonls.)_")
    else:
        lines.append("None yet — most likely no scoring has run for any pair "
                     "of scaffolds on the same model.")
    lines.append("")
    lines.append("## Per-run paths")
    lines.append("")
    for preset, by_sc in sorted(paths.items()):
        for sc in SCAFFOLDS:
            p = by_sc.get(sc)
            if p:
                lines.append(f"- `{preset}-{sc}`: `{p.relative_to(runs_dir.parent.parent)}`")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}", flush=True)
    for cells in rows:
        print("  " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
