#!/usr/bin/env python3
"""Paired read of two lanes of the same (preset, scaffold) — the v4 spec-decode
cell against its v3 no-spec reference while v4 is still rolling.

For every instance present in BOTH predictions.jsonl files: wall hits
(rc 124; rollout_seconds >= WALL_S only for rows without a returncode), empty patches, rollout_seconds
(median / mean / sum) and the per-instance speed ratio. Pairing on instance id
is what makes a partial cell readable at all: the first N v4 instances are
compared with the SAME N instances of v3, not with v3's 300-instance average
(the queue rolls instances in dataset order, so the first 30 are astropy/django
— easier than the tail; a naive "30 vs 300" read would over-credit spec).

Infra rows (the rollout never reached the scaffold: `rollout_error` set or a
non-zero / -1 `rollout_returncode` with no patch — e.g. the per-instance image
build failing) are excluded from the pairing on EITHER side and counted
separately; the cycle re-rolls them at lane close (audit_predictions.py), so
reading them as empties would charge a registry blip to the model. rc 124 is
a wall, not infra.

Usage: paired_lane_compare.py runs/qwen38-opencode-v3 runs/qwen38-opencode-v4 [--wall 1795] [--json out]
Informational only — never a scoring input. Full-300 rule still applies to any
resolved-rate claim.
"""
import argparse, json, pathlib, statistics as st

def load(run):
    rows = {}
    p = pathlib.Path(run) / "predictions.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["instance_id"]] = r
    return rows

def wall(r, wall_s):
    """rc 124 is the wall (the host `timeout` killing the scaffold).
    `rollout_seconds` spans the per-instance image build (~175-300 s), so a
    run that FINISHED (rc 0) after a slow build + long session reads >= 1795 s
    without ever hitting the cap — 2 such rows per arm at 107 pairs
    (django-16229 in both). The seconds threshold is only the fallback for
    rows that carry no returncode."""
    rc = r.get("rollout_returncode")
    if rc is not None:
        return rc == 124
    return (r.get("rollout_seconds") or 0) >= wall_s

def patch(r) -> str:
    """model_patch, or "" when the field is not a diff. A wall-hit row never
    prints the real `=== DIFF ===` marker, and if the session `ps`-ed the inner
    script (the marker sits in its `bash -lc` argv) the extractor's rfind lands
    on that echoed copy and captures the JSON event tail after it instead
    (qwen38 v4 walls pytest-8365, sympy-13915, sympy-20212; dcp-v3 pytest-5103).
    Such a "patch" cannot apply — it is an empty for every reading here."""
    p = (r.get("model_patch") or "").strip()
    return p if p.startswith(("diff ", "--- ", "Index: ")) else ""

def infra(r):
    """Rollout never reached the scaffold (same shape audit_predictions.py
    classes infra_rollout_nonzero_rc and re-rolls). rc 124 = wall, kept."""
    if patch(r):
        return False
    if r.get("rollout_error"):
        return True
    rc = r.get("rollout_returncode")
    return rc not in (0, None, 124)

def summarize(rows, ids, wall_s):
    secs = [rows[i].get("rollout_seconds") or 0 for i in ids]
    return {
        "n": len(ids),
        "walls": sum(wall(rows[i], wall_s) for i in ids),
        "empty": sum(not patch(rows[i]) for i in ids),
        "median_s": round(st.median(secs), 1) if secs else None,
        "mean_s": round(st.mean(secs), 1) if secs else None,
        "sum_h": round(sum(secs) / 3600, 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--wall", type=float, default=1795.0, help="rollout_seconds at/over which a row counts as a wall hit")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref, new = load(a.ref), load(a.new)
    both = [i for i in new if i in ref]       # new-lane order (= roll order)
    infra_ids = {"ref": [i for i in both if infra(ref[i])], "new": [i for i in both if infra(new[i])]}
    skip = set(infra_ids["ref"]) | set(infra_ids["new"])
    ids = [i for i in both if i not in skip]
    if not ids:
        print("no paired instances yet"); return
    sr, sn = summarize(ref, ids, a.wall), summarize(new, ids, a.wall)
    ratios = []
    for i in ids:
        r0, r1 = ref[i].get("rollout_seconds") or 0, new[i].get("rollout_seconds") or 0
        if r0 > 0 and r1 > 0:
            ratios.append(r0 / r1)
    out = {"ref": str(a.ref), "new": str(a.new), "paired": len(ids), "wall_s": a.wall,
           "ref_summary": sr, "new_summary": sn,
           "speedup_median": round(st.median(ratios), 2) if ratios else None,
           "unpaired_new": len(new) - len(both),
           "infra_excluded": infra_ids,
           "flips": {
               "wall_to_done": sum(wall(ref[i], a.wall) and not wall(new[i], a.wall) for i in ids),
               "done_to_wall": sum(not wall(ref[i], a.wall) and wall(new[i], a.wall) for i in ids),
               "empty_to_patch": sum((not patch(ref[i])) and bool(patch(new[i])) for i in ids),
               "patch_to_empty": sum(bool(patch(ref[i])) and not patch(new[i]) for i in ids),
           }}
    name = lambda p: pathlib.Path(p).name
    print(f"paired {len(ids)} instances  ({name(a.ref)} vs {name(a.new)}; wall = rc 124)"
          + (f"  [infra excluded: ref {len(infra_ids['ref'])}, new {len(infra_ids['new'])} — re-rolled at lane close]" if skip else ""))
    print(f"{'':14}{'walls':>7}{'empty':>7}{'median s':>10}{'mean s':>9}{'sum h':>8}")
    for lab, s_ in (("ref", sr), ("new", sn)):
        print(f"{lab:14}{s_['walls']:>7}{s_['empty']:>7}{s_['median_s']:>10}{s_['mean_s']:>9}{s_['sum_h']:>8}")
    print(f"per-instance speedup (ref/new, median): {out['speedup_median']}   flips: {out['flips']}")
    print("\n  instance                                   ref s   new s  ref  new")
    for i in ids:
        r0, r1 = ref[i], new[i]
        tag = lambda r: ("WALL" if wall(r, a.wall) else ("empty" if not patch(r) else "patch"))
        print(f"  {i:42}{(r0.get('rollout_seconds') or 0):>7.0f}{(r1.get('rollout_seconds') or 0):>8.0f}  {tag(r0):5}{tag(r1):5}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
