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
    ap.add_argument("--temperature", default="0",
                    help="sampling temperature (default 0 = greedy, as the quality suite); "
                         "'default' omits the field so the checkpoint's generation_config applies")
    ap.add_argument("--haystack", default=None,
                    help="path to a natural-text corpus to tile as the haystack instead of the "
                         "repeated-sentence filler (separates a repetition attractor from a retrieval miss)")
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="chars/token of the haystack under the served tokenizer (filler 4.5; "
                         "default 3.4 with --haystack, 4.5 without) — sizes the prompt")
    ap.add_argument("--flush", action="store_true",
                    help="POST /flush_cache before every cell (no radix-cache prefix reuse)")
    a = ap.parse_args()
    haystack = open(a.haystack).read() if a.haystack else None
    chars_per_token = a.chars_per_token or (3.4 if haystack else 4.5)
    temperature = None if a.temperature == "default" else float(a.temperature)
    url = f"http://localhost:{a.port}/v1/chat/completions"
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = tuple(float(x) for x in a.depths.split(","))
    results = []
    for _ in range(a.repeat):
        results += needle_eval(url, lengths, depths=depths, budget=a.max_tokens,
                               temperature=temperature, haystack=haystack,
                               chars_per_token=chars_per_token, flush=a.flush)["results"]
    # `found` is the fleet rule (needle text anywhere in reasoning + content).
    # `answered` is the strict read: the turn finished (`stop`) AND the needle is
    # in the final content. A model that quotes the haystack until the budget
    # (Flint's NoPE attractor, 2026-10-31 e5m2 gate: 131K@0.9 greedy, 2048 tok,
    # empty content) scores `found` without ever answering — keep both.
    for r in results:
        r["answered"] = bool(r["found"] and r.get("finish_reason") == "stop"
                             and "BANANA42" in (r.get("content_tail") or ""))
    score = sum(r["found"] for r in results) / len(results) if results else 0
    answered = sum(r["answered"] for r in results) / len(results) if results else 0
    out = {"tag": a.tag, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "temperature": "default" if temperature is None else temperature,
           "haystack": a.haystack or "filler", "flush": a.flush,
           "needle": {"results": results, "score": score, "answered_score": answered}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    for r in results:
        print(f"  {r['context']:>7} @ {r['depth']:<4} {'found' if r['found'] else 'MISS '}{'' if r['answered'] or not r['found'] else ' (unanswered)'} actual={r['actual_prompt_tokens']} "
              f"finish={r.get('finish_reason')} out={r.get('completion_tokens')}"
              + ("" if r["found"] else f"  tail={r.get('content_tail') or r.get('reasoning_tail')!r}"))
    print(f"score {score:.2f} answered {answered:.2f} -> {path}")


if __name__ == "__main__":
    main()
