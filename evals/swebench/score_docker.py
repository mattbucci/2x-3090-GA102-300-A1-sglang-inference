#!/usr/bin/env python3
"""Score a predictions.jsonl file with the official SWE-bench Docker harness.

Wraps `swebench.harness.run_evaluation` so each (model, scaffold) output dir
gets its own scoring artifacts under `scores-docker/` — never collides
with sibling scaffolds even when they target the same model.

Output dir layout (per the bake-off convention):

    evals/swebench/runs/<preset>-<scaffold>-v2/
        predictions.jsonl                  # rollout output (input here)
        scores-docker/
            <run_id>.<model>.report.json   # the official harness report
            run_evaluation/<run_id>/...    # per-instance test outputs
        scores-docker-summary.json         # condensed: resolved/unresolved/empty/total

run_id is derived from the output dir name to keep the scoring tag aligned
with the rollout directory.

Usage:
    python evals/swebench/score_docker.py \\
        --predictions evals/swebench/runs/coder-30b-opencode-v2/predictions.jsonl

Auto-derives:
    - run_id from the parent dir name (e.g. coder-30b-opencode-v2)
    - report_dir at scores-docker/ next to predictions.jsonl
    - dataset/split from the predictions (or override via --dataset/--split)

Optional flags:
    --max-workers N    Pass through to the harness. Defaults to 1 to match the
                       single-user serving constraint and avoid Docker
                       contention with the host SGLang server.
    --timeout N        Per-instance test timeout (default 1800s).
    --rewrite-reports  Skip instance test runs; only re-aggregate from existing
                       per-instance test outputs (cheap re-score after a
                       previous full run).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.jsonl (the rollout output)")
    p.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite",
                   help="HF dataset id passed to the harness")
    p.add_argument("--split", default="test")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Parallel test runs. Default 1 — the box hit a kernel "
                        "BUG (vfs_getattr_nosec page fault) under sustained "
                        "concurrent docker I/O on 2026-05-10. Single-worker "
                        "scoring + sequenced (non-concurrent) rollout/score "
                        "minimizes VFS pressure on a system with the NVIDIA "
                        "OOT modules + Arch kernel that's been unstable under "
                        "heavy concurrent fs ops.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-instance test timeout (seconds).")
    p.add_argument("--rewrite-reports", action="store_true",
                   help="Re-aggregate from existing per-instance test outputs "
                        "without rerunning tests.")
    p.add_argument("--cache-level", default="env",
                   choices=["none", "base", "env", "instance"],
                   help="Harness cache level — see swebench docs.")
    p.add_argument("--run-id", default=None,
                   help="Override run_id (default: parent dir name of predictions).")
    return p.parse_args()


def _find_reports(report_dir: Path, run_id: str) -> list[Path]:
    """Locate ONE harness report file (the source of truth for this run_id).

    The current swebench harness writes `<model_sanitized>.<run_id>.json`
    to CWD and ignores `--report_dir` for the summary file. Prefer the
    CWD copy; only fall back to an archived `<run_id>.<model>.report.json`
    under report_dir if no CWD/repo-root original exists. Returning more
    than one would double-count when we re-summarize a cell whose archive
    has already been written from a prior run.
    """
    repo_root = report_dir.parents[3] if len(report_dir.parents) >= 4 else report_dir.parent
    # Prefer the harness's CWD original (or anywhere outside report_dir).
    for d in (Path.cwd(), repo_root, report_dir.parent):
        for p in d.glob(f"*.{run_id}.json"):
            if p.resolve().parent != report_dir.resolve():
                return [p]
    # Fall back to archived copies (re-summarize from a prior run).
    archived = sorted(report_dir.glob(f"{run_id}.*.report.json"))
    if archived:
        return [archived[0]]  # one is enough; we don't dual-count
    return []


def summarize(report_dir: Path, run_id: str, predictions: list[dict]) -> dict:
    """Aggregate the harness's per-model report into a flat summary.

    Handles schema v2 (counts as `*_instances`, ids as plain string lists).
    Falls back to id-list counting if counts are absent.
    """
    reports = _find_reports(report_dir, run_id)
    by_instance: dict[str, str] = {}
    counts = {"resolved": 0, "unresolved": 0, "error": 0,
              "empty_patch": 0, "incomplete": 0, "submitted": 0,
              "completed": 0}
    archived = []
    for rp in reports:
        try:
            r = json.loads(rp.read_text())
        except Exception as e:
            print(f"  warn: could not parse {rp}: {e}", flush=True)
            continue
        # Schema v2: explicit counts
        for ck, vk in [("resolved", "resolved_instances"),
                       ("unresolved", "unresolved_instances"),
                       ("error", "error_instances"),
                       ("empty_patch", "empty_patch_instances"),
                       ("incomplete", "incomplete_instances"),
                       ("submitted", "submitted_instances"),
                       ("completed", "completed_instances")]:
            if isinstance(r.get(vk), int):
                counts[ck] += r[vk]
        # ID lists (v1 and v2 both expose these; v2 entries are plain strings)
        for key, label in [
            ("resolved_ids", "resolved"),
            ("unresolved_ids", "unresolved"),
            ("error_ids", "error"),
            ("empty_patch_ids", "empty_patch"),
            ("incomplete_ids", "incomplete"),
            ("submitted_ids", "submitted"),
        ]:
            for iid in r.get(key, []) or []:
                if isinstance(iid, str):
                    by_instance.setdefault(iid, label)
        # Archive a copy into report_dir so the aggregator can find it
        # later even if CWD has moved.
        target = report_dir / f"{run_id}.{rp.stem.split('.')[0]}.report.json"
        try:
            if rp.resolve() != target.resolve():
                report_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(rp.read_text())
                archived.append(str(target.relative_to(report_dir.parent)))
        except Exception:
            pass

    # Fallback: derive counts from id lists if counts weren't present
    if not any(counts.values()):
        for label in by_instance.values():
            if label in counts:
                counts[label] += 1

    total = len(predictions)
    return {
        "run_id": run_id,
        "total_predictions": total,
        "resolved": counts["resolved"],
        "unresolved": counts["unresolved"],
        "error": counts["error"],
        "empty_patch": counts["empty_patch"],
        "incomplete": counts["incomplete"],
        "submitted": counts["submitted"],
        "completed": counts["completed"],
        "per_instance": by_instance,
        "resolve_rate_pct": round(100.0 * counts["resolved"] / total, 1) if total else 0.0,
        "report_files": [str(p) for p in reports],
        "archived_reports": archived,
    }


def main():
    args = parse_args()

    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
        return 2

    out_dir = pred_path.parent
    run_id = args.run_id or out_dir.name
    report_dir = out_dir / "scores-docker"
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            predictions.append(json.loads(line))
        except Exception:
            pass
    print(f"Read {len(predictions)} predictions from {pred_path}", flush=True)

    print(f"Run ID:    {run_id}", flush=True)
    print(f"Report:    {report_dir}", flush=True)
    print(f"Dataset:   {args.dataset}/{args.split}", flush=True)
    print(f"Workers:   {args.max_workers}   Timeout: {args.timeout}s   "
          f"Cache: {args.cache_level}", flush=True)

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset,
        "--split", args.split,
        "--predictions_path", str(pred_path),
        "--run_id", run_id,
        "--max_workers", str(args.max_workers),
        "--timeout", str(args.timeout),
        "--cache_level", args.cache_level,
        "--report_dir", str(report_dir),
    ]
    if args.rewrite_reports:
        cmd.append("--rewrite_reports")
        cmd.append("True")

    print(f"\n+ {' '.join(cmd)}\n", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n  harness exited rc={rc}; trying to summarize anyway",
              file=sys.stderr, flush=True)

    summary = summarize(report_dir, run_id, predictions)
    summary["harness_returncode"] = rc
    summary_path = out_dir / "scores-docker-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {run_id} ===", flush=True)
    print(f"  resolved:       {summary['resolved']} / {summary['total_predictions']} "
          f"= {summary['resolve_rate_pct']}%", flush=True)
    print(f"  unresolved:     {summary['unresolved']}", flush=True)
    print(f"  error:          {summary['error']}", flush=True)
    print(f"  summary file:   {summary_path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
