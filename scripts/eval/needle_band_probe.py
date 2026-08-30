#!/usr/bin/env python3
"""Targeted needle-in-a-haystack re-probe: arbitrary context lengths x depths,
against a live server, written as a quality-style receipt.

Use it to bracket a single flaky needle cell instead of re-running the whole
quality suite — e.g. the qwen36 131,072 @ depth-0.5 knife-edge (misses on
v0.5.16 and v0.5.18, passes on v0.5.15/v0.5.17; every neighbouring length
passes): a band probe around the cell shows whether it is a point artifact
or a depth regression.

    python scripts/eval/needle_band_probe.py --port 23334 --tag qwen36-v0518-needle2 \
        --lengths 131072 --depths 0.1,0.5,0.9
    python scripts/eval/needle_band_probe.py --port 23334 --tag qwen36-v0518-needle-band \
        --lengths 114688,122880,131072,139264,147456 --depths 0.5

Receipt: benchmarks/quality/<tag>.json with {tag, timestamp, needle{results, score}}
(needle-only, so compare_flip_receipts / the README chart ignore it by design).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_and_chart import needle_eval  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "quality"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", required=True, help="comma list of context lengths (tokens)")
    ap.add_argument("--depths", default="0.5", help="comma list of needle depths (fractions)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=1, help="repeat the whole grid N times")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}/v1/chat/completions"
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = tuple(float(x) for x in a.depths.split(","))
    results = []
    for _ in range(a.repeat):
        results += needle_eval(url, lengths, depths=depths, max_tokens=a.max_tokens)["results"]
    score = sum(r["found"] for r in results) / len(results) if results else 0
    out = {"tag": a.tag, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "needle": {"results": results, "score": score}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    for r in results:
        print(f"  {r['context']:>7} @ {r['depth']:<4} {'found' if r['found'] else 'MISS '} actual={r['actual_prompt_tokens']}")
    print(f"score {score:.2f} -> {path}")


if __name__ == "__main__":
    main()
