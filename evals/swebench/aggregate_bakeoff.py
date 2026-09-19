#!/usr/bin/env python3
"""Aggregate bake-off scores into a per-model × per-scaffold resolved-rate
table. Reads scores-docker-summary.json from every
evals/swebench/runs/<preset>-<scaffold>-v<N>/ directory (highest N per cell) and writes:

  - `benchmarks/quality/bakeoff-<preset>-<scaffold>.json` — one tracked
    JSON per (preset, scaffold) cell. Matches the existing 1-file-per-
    artifact convention used for MMLU / HumanEval snapshots.
  - `evals/swebench/bake-off-<date>.md` — combined Markdown view across
    all cells, plus per-instance scaffold-disagreement table ("same
    model, different verdict per scaffold" — diagnoses model-failure
    vs scaffold-failure).

The raw per-instance rollouts under evals/swebench/runs/ stay
gitignored; the cell JSONs in benchmarks/quality/ are the persisted
record.

Usage:
    python evals/swebench/aggregate_bakeoff.py
    python evals/swebench/aggregate_bakeoff.py --runs-dir evals/swebench/runs
    python evals/swebench/aggregate_bakeoff.py --no-json   # markdown only
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
    p.add_argument("--quality-dir", default="benchmarks/quality",
                   help="Where per-cell JSONs land (default: benchmarks/quality).")
    p.add_argument("--no-json", action="store_true",
                   help="Skip per-cell JSON writes; markdown only.")
    return p.parse_args()


def first_model_path(run_dir: Path) -> str | None:
    """Pull model_name_or_path off the first prediction line, if present."""
    pred = run_dir / "predictions.jsonl"
    if not pred.exists():
        return None
    try:
        with pred.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return obj.get("model_name_or_path")
    except Exception:
        return None
    return None


def scaffold_context_window(run_dir: Path):
    """Smallest `context_window` recorded across the run's meta.json entries.

    docker_rollout.py records the token budget it handed the scaffold since
    2026-09-11 (read from the server's /v1/models max_model_len). The
    minimum matters: a run assembled from several invocations is only as
    good as its most-truncated one. None for runs older than the field —
    those ran at whatever the scaffold's own default was (pi: 32K, prime:
    128K, dcode: 170K, opencode: the static opencode.json entry).
    """
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
    except Exception:
        return None
    runs = m.get("runs") if isinstance(m, dict) else m
    vals = [r.get("context_window") for r in (runs or []) if isinstance(r, dict)]
    vals = [v for v in vals if isinstance(v, int) and v > 0]
    return min(vals) if vals else None


def scaffold_output_budget(run_dir: Path):
    """Smallest `output_budget` across the run's meta.json entries — the
    per-request output cap docker_rollout.py configured in the scaffold
    (32768 since 2026-09-13; every scaffold also runs max thinking from that
    date). None for older runs: opencode's packaged 8192 / pi's 16384 with
    `reasoning_effort: medium`."""
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
    except Exception:
        return None
    runs = m.get("runs") if isinstance(m, dict) else m
    vals = [r.get("output_budget") for r in (runs or []) if isinstance(r, dict)]
    vals = [v for v in vals if isinstance(v, int) and v > 0]
    return min(vals) if vals else None


def rollout_network_mode(run_dir: Path):
    """`network_mode` of the run's rollouts: "none" = isolated (docker_rollout.py
    default since 2026-09-19), "host" = the scaffold could reach the internet
    (every run before that date — 54 % of qwen38 opencode instances touched
    upstream; audit_leakage.py). A run mixing both reports "host"; runs older
    than the field report "host" too (that is what they ran with)."""
    meta = run_dir / "meta.json"
    if not meta.exists():
        return "host"
    try:
        m = json.loads(meta.read_text())
    except Exception:
        return "host"
    runs = m.get("runs") if isinstance(m, dict) else m
    modes = {r.get("network_mode", "host") for r in (runs or []) if isinstance(r, dict)}
    return "none" if modes == {"none"} else "host"


def write_cell_json(preset: str, scaffold: str, run_dir: Path,
                    summary: dict, quality_dir: Path,
                    repo_root: Path) -> Path:
    """Write benchmarks/quality/bakeoff-<preset>-<scaffold>.json for one cell."""
    quality_dir.mkdir(parents=True, exist_ok=True)
    out = quality_dir / f"bakeoff-{preset}-{scaffold}.json"

    total = summary.get("total_predictions", 0)
    resolved = summary.get("resolved", 0)
    rate = summary.get("resolve_rate_pct")
    if rate is None and total:
        rate = round(100.0 * resolved / total, 1)

    counts = {"resolved": resolved}
    for label in ("unresolved", "error", "empty_patch",
                  "incomplete", "submitted", "completed"):
        # Prefer direct count from the summary (schema v2); fall back to
        # per_instance iteration for older summaries that only have ids.
        n = summary.get(label)
        if not isinstance(n, int):
            n = 0
            for v in (summary.get("per_instance") or {}).values():
                if v == label or v == label.split("_")[0]:
                    n += 1
        counts[label] = n

    payload = {
        "preset": preset,
        "scaffold": scaffold,
        "model_path": first_model_path(run_dir),
        "total_predictions": total,
        "resolve_rate_pct": rate,
        **counts,
        "harness_returncode": summary.get("harness_returncode"),
        "scaffold_context_window": scaffold_context_window(run_dir),
        "scaffold_output_budget": scaffold_output_budget(run_dir),
        "rollout_network_mode": rollout_network_mode(run_dir),
        "run_dir": str(run_dir.relative_to(repo_root))
                   if str(run_dir).startswith(str(repo_root))
                   else str(run_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


SCAFFOLDS = ("opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "claw-code", "prime", "dcode")  # claw-code deprecated 2026-08-30; historical cells still parse


def discover_runs(runs_dir: Path):
    """Yield (preset, scaffold, summary_dict) tuples for each completed run."""
    if not runs_dir.exists():
        return

    # Built from SCAFFOLDS so a roster change cannot silently drop lanes at
    # Phase 6 (the hand-written 3-scaffold alternation would have skipped the
    # 2026-08-31 lanes). `-v<N>$` anchoring disambiguates opencode vs
    # opencode-dcp regardless of alternation order. Only the highest version
    # per (preset, scaffold) is a live cell: v3 (2026-09-19, network-isolated
    # rollouts) supersedes v2 (--network=host, exposure-confounded) as each
    # lane lands. Quarantined dirs (`-v2-ctx32k`, `-v2-netopen`, ...) do not
    # match and never aggregate.
    pat = re.compile(r"^(.*)-(" + "|".join(re.escape(sc) for sc in SCAFFOLDS) + r")-v(\d+)$")
    latest: dict[tuple[str, str], tuple[int, Path]] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        key, ver = (m.group(1), m.group(2)), int(m.group(3))
        if key not in latest or ver > latest[key][0]:
            latest[key] = (ver, d)
    for (preset, scaffold), (_ver, d) in sorted(latest.items()):
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
    # runs_dir = <repo>/evals/swebench/runs → repo root is 3 levels up
    repo_root = runs_dir.parent.parent.parent
    quality_dir = (repo_root / args.quality_dir).resolve()

    # Group: results[preset][scaffold] = summary
    results = defaultdict(dict)
    paths = defaultdict(dict)
    cell_jsons = []
    for preset, scaffold, run_dir, summary in discover_runs(runs_dir):
        results[preset][scaffold] = summary
        paths[preset][scaffold] = run_dir
        if summary and not args.no_json:
            cell_jsons.append(
                write_cell_json(preset, scaffold, run_dir, summary,
                                quality_dir, repo_root)
            )

    # Special-case: the legacy `coder-30b-docker-v2` dir is the opencode
    # cell for coder-30b-eval but doesn't follow the
    # `<preset>-<scaffold>-v2` naming. Pick it up whenever the opencode
    # slot for coder-30b-eval is empty.
    legacy = runs_dir / "coder-30b-docker-v2"
    if legacy.exists() and "opencode" not in results.get("coder-30b-eval", {}):
        summary_path = legacy / "scores-docker-summary.json"
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass
        results.setdefault("coder-30b-eval", {})["opencode"] = summary
        paths.setdefault("coder-30b-eval", {})["opencode"] = legacy
        if summary and not args.no_json:
            cell_jsons.append(
                write_cell_json("coder-30b-eval", "opencode", legacy, summary,
                                quality_dir, repo_root)
            )

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
    if cell_jsons:
        print(f"\nWrote {len(cell_jsons)} per-cell JSON(s) to "
              f"{quality_dir.relative_to(repo_root) if str(quality_dir).startswith(str(repo_root)) else quality_dir}/",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
