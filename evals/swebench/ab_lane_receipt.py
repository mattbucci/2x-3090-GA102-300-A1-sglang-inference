#!/usr/bin/env python3
"""Pre-score receipt for an A/B scaffold lane vs its control lane.

Produces the same-ID rollout comparison we publish before the scored cell lands
(patched / empty / timeout / other-rc counts, median wall time, agreement
matrix) plus the plugin engagement rate read from the poller snapshots
(`evals/swebench/{dcp,rtk}_engagement_poller.sh`). The scored cell is the
verdict; this receipt says whether the plugin was actually *doing* anything and
whether it changed the rollout's shape.

Engagement sources:
  rtk  : <engagement>/<iid>.json  (`rtk gain -f json`, summary.total_commands)
         <engagement>/<iid>.db    (rtk history.db, per-command detail)
  dcp  : <engagement>/<iid>.json  (DCP session state, stats.totalPruneTokens)

Usage:
  ab_lane_receipt.py --lane runs/<p>-little-coder-rtk-v2 --control runs/<p>-little-coder-v2 \
      --engagement benchmarks/quality/rtk-engagement --kind rtk --out benchmarks/quality/rtk-lane-receipt
Writes <out>.md and <out>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import statistics
from pathlib import Path

ELAPSED_RE = re.compile(r"^# elapsed ([0-9.]+)s\s+rc=(-?\d+)", re.M)


def lane_rows(run_dir: Path) -> dict[str, dict]:
    rows = {}
    for log in sorted((run_dir / "logs").glob("*.log")):
        iid = log.stem
        head = log.read_text(errors="replace")[:4000]
        m = ELAPSED_RE.search(head)
        pred = run_dir / "predictions" / f"{iid}.diff"
        size = pred.stat().st_size if pred.exists() else None
        rc = int(m.group(2)) if m else None
        rows[iid] = {
            "elapsed": float(m.group(1)) if m else None,
            "rc": rc,
            "patch_bytes": size,
            "outcome": (
                "timeout" if rc == 124 else
                "patched" if size else
                "empty" if size == 0 else
                "no_prediction"
            ),
        }
    return rows


def summarize(rows: dict[str, dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ("patched", "empty", "timeout", "no_prediction"):
        out[k] = sum(1 for r in rows.values() if r["outcome"] == k)
    out["other_rc"] = sum(1 for r in rows.values() if r["rc"] not in (0, 124, None))
    el = [r["elapsed"] for r in rows.values() if r["elapsed"] is not None]
    out["median_elapsed_s"] = round(statistics.median(el), 1) if el else None
    out["p90_elapsed_s"] = round(sorted(el)[int(0.9 * (len(el) - 1))], 1) if el else None
    return out


def rtk_engagement(eng: Path, ids: set[str]) -> dict:
    per = {}
    cmd_families: collections.Counter = collections.Counter()
    parse_failures = 0
    for js in sorted(eng.glob("*.json")):
        iid = js.stem
        if ids and iid not in ids:
            continue
        try:
            s = json.loads(js.read_text())["summary"]
        except Exception:
            continue
        per[iid] = {
            "commands": s.get("total_commands", 0),
            "saved_tokens": s.get("total_saved", 0),
            "input_tokens": s.get("total_input", 0),
            "avg_savings_pct": s.get("avg_savings_pct"),
        }
        db = eng / f"{iid}.db"
        if db.exists():
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
                for (orig,) in con.execute("select original_cmd from commands"):
                    cmd_families[orig.strip().split()[0] if orig.strip() else "?"] += 1
                parse_failures += con.execute("select count(*) from parse_failures").fetchone()[0]
                con.close()
            except sqlite3.Error:
                pass
    seen = {p.name for p in (eng / ".seen").glob("*")} if (eng / ".seen").exists() else set(per)
    if ids:
        seen &= ids
    engaged = {i for i, v in per.items() if v["commands"] > 0}
    cmds = [v["commands"] for v in per.values() if v["commands"] > 0]
    saved = [v["saved_tokens"] for v in per.values() if v["commands"] > 0]
    return {
        "kind": "rtk",
        "observed": len(seen),
        "snapshots": len(per),
        "engaged": len(engaged),
        "engaged_ids": sorted(engaged),
        "engagement_rate": round(len(engaged) / len(seen), 3) if seen else None,
        "commands_per_session_median": statistics.median(cmds) if cmds else 0,
        "commands_per_session_p90": sorted(cmds)[int(0.9 * (len(cmds) - 1))] if cmds else 0,
        "saved_tokens_median": statistics.median(saved) if saved else 0,
        "saved_tokens_total": sum(saved),
        "command_families_top": cmd_families.most_common(12),
        "parse_failures": parse_failures,
    }


def dcp_engagement(eng: Path, ids: set[str]) -> dict:
    per = {}
    for js in sorted(eng.glob("*.json")):
        iid = js.stem
        if ids and iid not in ids:
            continue
        try:
            d = json.loads(js.read_text())
        except Exception:
            continue
        pruned = (d.get("stats") or {}).get("totalPruneTokens", 0) or 0
        per[iid] = {"pruned_tokens": pruned}
    seen = {p.name for p in (eng / ".seen").glob("*")} if (eng / ".seen").exists() else set(per)
    if ids:
        seen &= ids
    engaged = {i for i, v in per.items() if v["pruned_tokens"] > 0}
    pr = [v["pruned_tokens"] for v in per.values() if v["pruned_tokens"] > 0]
    return {
        "kind": "dcp",
        "observed": len(seen),
        "snapshots": len(per),
        "engaged": len(engaged),
        "engaged_ids": sorted(engaged),
        "engagement_rate": round(len(engaged) / len(seen), 3) if seen else None,
        "pruned_tokens_median": statistics.median(pr) if pr else 0,
        "pruned_tokens_total": sum(pr),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True, type=Path)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--engagement", required=True, type=Path)
    ap.add_argument("--kind", choices=["rtk", "dcp"], required=True)
    ap.add_argument("--out", required=True, type=Path, help="prefix; writes .md and .json")
    ap.add_argument("--label", default=None, help="lane label for the markdown (default: lane dir name)")
    a = ap.parse_args()

    lane, ctl = lane_rows(a.lane), lane_rows(a.control)
    common = sorted(set(lane) & set(ctl))
    L = {i: lane[i] for i in common}
    C = {i: ctl[i] for i in common}
    eng = (rtk_engagement if a.kind == "rtk" else dcp_engagement)(a.engagement, set(common))
    engaged = set(eng["engaged_ids"])

    agree = collections.Counter()
    for i in common:
        lp, cp = L[i]["outcome"] == "patched", C[i]["outcome"] == "patched"
        agree["both_patched" if lp and cp else "lane_only" if lp else "control_only" if cp else "neither"] += 1

    split = {
        "engaged": summarize({i: L[i] for i in common if i in engaged}),
        "not_engaged": summarize({i: L[i] for i in common if i not in engaged}),
        "control_on_engaged_ids": summarize({i: C[i] for i in common if i in engaged}),
    }

    rep = {
        "lane": str(a.lane), "control": str(a.control), "kind": a.kind,
        "same_id_n": len(common),
        "lane_summary": summarize(L), "control_summary": summarize(C),
        "agreement": dict(agree),
        "engagement": {k: v for k, v in eng.items() if k != "engaged_ids"},
        "split_by_engagement": split,
        "lane_all": summarize(lane), "control_all": summarize(ctl),
        "per_instance": {i: {"lane": L[i], "control": C[i], "engaged": i in engaged} for i in common},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))

    lab = a.label or a.lane.name
    ls, cs = rep["lane_summary"], rep["control_summary"]
    md = [f"# {lab} vs {a.control.name} — pre-score rollout receipt", ""]
    md += [f"Same-ID instances: **{len(common)}** (lane {len(lane)} logs, control {len(ctl)} logs).", ""]
    md += ["| | lane | control |", "|---|---|---|"]
    for k, nm in (("patched", "patched (non-empty diff)"), ("empty", "empty diff"), ("timeout", "timeout (rc=124)"),
                  ("other_rc", "other non-zero rc"), ("median_elapsed_s", "median wall (s)"), ("p90_elapsed_s", "p90 wall (s)")):
        md.append(f"| {nm} | {ls[k]} | {cs[k]} |")
    md += ["", "Agreement (patched or not): " + ", ".join(f"{k} {v}" for k, v in sorted(agree.items())), ""]
    e = rep["engagement"]
    if a.kind == "rtk":
        md += [f"## rtk engagement", "",
               f"- Observed containers: {e['observed']}; snapshots with a ledger: {e['snapshots']}; "
               f"**engaged (≥1 executed rewrite): {e['engaged']} = {100*(e['engagement_rate'] or 0):.0f}%**",
               f"- Executed rewrites per engaged session: median {e['commands_per_session_median']}, p90 {e['commands_per_session_p90']}",
               f"- Tokens rtk reports saved: median {e['saved_tokens_median']} / session, total {e['saved_tokens_total']:,}; parse failures {e['parse_failures']}",
               "- Rewritten command families: " + ", ".join(f"`{c}` {n}" for c, n in e["command_families_top"]), ""]
    else:
        md += [f"## DCP engagement", "",
               f"- Observed containers: {e['observed']}; snapshots: {e['snapshots']}; "
               f"**engaged (pruned >0 tokens): {e['engaged']} = {100*(e['engagement_rate'] or 0):.0f}%**",
               f"- Pruned tokens when engaged: median {e['pruned_tokens_median']:,}, total {e['pruned_tokens_total']:,}", ""]
    md += ["## Lane split by engagement", "", "| subset | n | patched | empty | timeout | median wall (s) |", "|---|---|---|---|---|---|"]
    for k, nm in (("engaged", "lane, engaged"), ("not_engaged", "lane, not engaged"), ("control_on_engaged_ids", "control, same engaged IDs")):
        s = split[k]
        md.append(f"| {nm} | {s['n']} | {s['patched']} | {s['empty']} | {s['timeout']} | {s['median_elapsed_s']} |")
    md += ["", "Verdict on quality is the scored cell (`scores-docker-summary.json`), not this receipt.", ""]
    a.out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
