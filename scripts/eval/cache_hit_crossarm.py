#!/usr/bin/env python3
"""Cross-run comparison for cache_hit_consistency.py receipts (issue #10 discriminator).

Given two receipts of the SAME preset and prompts served under different configurations (e.g.
--chunked-prefill-size 4096 vs 2048), compare the FRESH outputs across runs the same way the
instrument compares a cache hit against a fresh fill within a run: greedy common-prefix length
and the first-token top-5 max |Δlogprob|. If cross-run fresh-vs-fresh divergence matches the
within-run hit-vs-fresh divergence, a prefix-cache hit is no less exact than any other change of
chunk alignment (rounding class); if hits diverge where cross-run fresh fills stay exact, the
restored recurrent state itself is the defect.

  python scripts/eval/cache_hit_crossarm.py benchmarks/quality/cache-hit-consistency-qwen38-chunk4096-D.json \
      benchmarks/quality/cache-hit-consistency-qwen38-chunk2048-D.json
"""
import json, sys

def pref(x, y):
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
    return k

def top_delta(d1, d2):
    common = set(d1) & set(d2)
    return round(max(abs(d1[t] - d2[t]) for t in common), 3) if common else None

def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    print(f"{'tokens':>7} {'cell':>4} | within-run A: hit-prefix / floor  dlogp hit / floor | cross-run A fresh: prefix  dlogp | within-run B: hit / floor  dlogp | cross-run B fresh: prefix  dlogp")
    agg = {"A_hit_lt_floor": 0, "A_cross_lt_floor": 0, "B_hit_lt_floor": 0, "B_cross_lt_floor": 0, "n": 0}
    for ra, rb in zip(a["results"], b["results"]):
        assert (ra["tokens"], ra["cell"]) == (rb["tokens"], rb["cell"])
        ta, tb, la, lb = ra["texts"], rb["texts"], ra["first_top5"], rb["first_top5"]
        xa = pref(ta["A_fresh"], tb["A_fresh"]); xb = pref(ta["B_fresh"], tb["B_fresh"])
        da = top_delta(la["A_fresh"], lb["A_fresh"]); db = top_delta(la["B_fresh"], lb["B_fresh"])
        lp = ra["first_token_top5_max_abs_logprob_delta"]
        print(f"{ra['tokens']:>7} {ra['cell']:>4} | {ra['A_hit_prefix']:>5}/{len(ta['A_fresh']):<5} floor {ra['A_fresh_fresh_prefix']:<5} {lp['A_hit_vs_fresh']} / {lp['A_fresh_vs_fresh']} | "
              f"{xa:>5}  {da} | {ra['B_hit_prefix']:>5}/{len(ta['B_fresh']):<5} floor {ra['B_fresh_fresh_prefix']:<5} {lp['B_hit_vs_fresh']} / {lp['B_fresh_vs_fresh']} | {xb:>5}  {db}")
        agg["n"] += 1
        agg["A_hit_lt_floor"] += ra["A_hit_prefix"] < ra["A_fresh_fresh_prefix"]
        agg["A_cross_lt_floor"] += xa < ra["A_fresh_fresh_prefix"]
        agg["B_hit_lt_floor"] += ra["B_hit_prefix"] < ra["B_fresh_fresh_prefix"]
        agg["B_cross_lt_floor"] += xb < ra["B_fresh_fresh_prefix"]
    print(f"\ncells {agg['n']}: hit diverges before the within-run floor A {agg['A_hit_lt_floor']} / B {agg['B_hit_lt_floor']}; "
          f"cross-run fresh diverges before that floor A {agg['A_cross_lt_floor']} / B {agg['B_cross_lt_floor']}")

if __name__ == "__main__":
    main()
