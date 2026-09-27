#!/usr/bin/env python3
"""Paired read of two lanes of the same (preset, scaffold) — the v4 spec-decode
cell against its v3 no-spec reference while v4 is still rolling.

For every instance present in BOTH predictions.jsonl files: wall hits
(rollout_seconds >= WALL_S or rc 124), empty patches, rollout_seconds
(median / mean / sum) and the per-instance speed ratio. Pairing on instance id
is what makes a partial cell readable at all: the first N v4 instances are
compared with the SAME N instances of v3, not with v3's 300-instance average
(the queue rolls instances in dataset order, so the first 30 are astropy/django
— easier than the tail; a naive "30 vs 300" read would over-credit spec).

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
    return r.get("rollout_returncode") == 124 or (r.get("rollout_seconds") or 0) >= wall_s

def summarize(rows, ids, wall_s):
    secs = [rows[i].get("rollout_seconds") or 0 for i in ids]
    return {
        "n": len(ids),
        "walls": sum(wall(rows[i], wall_s) for i in ids),
        "empty": sum(not (rows[i].get("model_patch") or "").strip() for i in ids),
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
    ids = [i for i in new if i in ref]        # new-lane order (= roll order)
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
           "unpaired_new": len(new) - len(ids),
           "flips": {
               "wall_to_done": sum(wall(ref[i], a.wall) and not wall(new[i], a.wall) for i in ids),
               "done_to_wall": sum(not wall(ref[i], a.wall) and wall(new[i], a.wall) for i in ids),
               "empty_to_patch": sum((not (ref[i].get("model_patch") or "").strip()) and bool((new[i].get("model_patch") or "").strip()) for i in ids),
               "patch_to_empty": sum(bool((ref[i].get("model_patch") or "").strip()) and not (new[i].get("model_patch") or "").strip() for i in ids),
           }}
    name = lambda p: pathlib.Path(p).name
    print(f"paired {len(ids)} instances  ({name(a.ref)} vs {name(a.new)}; wall >= {a.wall:.0f}s or rc 124)")
    print(f"{'':14}{'walls':>7}{'empty':>7}{'median s':>10}{'mean s':>9}{'sum h':>8}")
    for lab, s_ in (("ref", sr), ("new", sn)):
        print(f"{lab:14}{s_['walls']:>7}{s_['empty']:>7}{s_['median_s']:>10}{s_['mean_s']:>9}{s_['sum_h']:>8}")
    print(f"per-instance speedup (ref/new, median): {out['speedup_median']}   flips: {out['flips']}")
    print("\n  instance                                   ref s   new s  ref  new")
    for i in ids:
        r0, r1 = ref[i], new[i]
        tag = lambda r: ("WALL" if wall(r, a.wall) else ("empty" if not (r.get("model_patch") or "").strip() else "patch"))
        print(f"  {i:42}{(r0.get('rollout_seconds') or 0):>7.0f}{(r1.get('rollout_seconds') or 0):>8.0f}  {tag(r0):5}{tag(r1):5}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
