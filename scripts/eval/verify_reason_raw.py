#!/usr/bin/env python3
"""verify_reason_raw.py — raw-dump companion to probe_256k_quality.py.

When a reasoning-probe cell fails (e.g. gemma4 passes `multikey` but fails
`vartrack`/`aggregate` at >=32K while passing all three at 1K), this dumps the
FULL raw response so you can tell WHY:

  - finish_reason == "length"  -> truncated (bump --max-tokens), an artifact
  - answer present but check missed it -> extraction artifact (fix the check)
  - reasoning_content shows a wrong computation -> genuine reasoning failure

It replays probe_256k_quality's EXACT task instances: same `random.Random(42)`
reset per length and the SAME task iteration order, so the rng state that
produces each task's facts is identical to the probe. So the prompt this sends
is byte-for-byte the prompt that failed in the sweep.

Usage:
  python scripts/eval/verify_reason_raw.py --port 23334 \
      --lengths 32768,256000 --tasks vartrack,aggregate
"""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_256k_quality import TASKS, _answer, fill, plant, CHARS_PER_TOKEN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--lengths", default="32768,256000")
    ap.add_argument("--tasks", default="vartrack,aggregate")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    ap.add_argument("--show-chars", type=int, default=1200)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    want = set(args.tasks.split(","))
    for L in [int(x) for x in args.lengths.split(",")]:
        rng = random.Random(42)  # reset per length, identical to the probe
        for tname, tfn in TASKS.items():  # SAME order as the probe -> identical rng draws
            facts, q, check = tfn(rng)
            if tname not in want:
                continue
            prompt = plant(fill(L, args.chars_per_token), facts) + "\n\n" + q
            t0 = time.time()
            try:
                r = requests.post(url, json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens, "temperature": 0,
                }, timeout=max(600, L // 150)).json()
            except Exception as e:
                print(f"\n===== {tname} @ approx={L} =====\n  REQUEST ERROR: {e}")
                continue
            if isinstance(r, dict) and r.get("error"):
                print(f"\n===== {tname} @ approx={L} =====\n  SERVER ERROR: {str(r['error'])[:200]}")
                continue
            choice = (r.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            usage = r.get("usage") or {}
            rc = msg.get("reasoning_content") or ""
            ct = msg.get("content") or ""
            ans = _answer(msg)
            n = args.show_chars
            print(f"\n===== {tname} @ approx={L} actual_tokens={usage.get('prompt_tokens')} =====")
            print(f"  facts planted : {facts}")
            print(f"  question      : {q}")
            print(f"  finish_reason : {choice.get('finish_reason')}")
            print(f"  usage         : completion={usage.get('completion_tokens')} "
                  f"reasoning={usage.get('reasoning_tokens')} total={usage.get('total_tokens')}")
            print(f"  check passes  : {check(ans)}")
            print(f"  elapsed       : {time.time() - t0:.1f}s")
            print(f"  reasoning_content ({len(rc)} chars):\n    {rc[:n].replace(chr(10), chr(10) + '    ')}")
            print(f"  content ({len(ct)} chars):\n    {ct[:n].replace(chr(10), chr(10) + '    ')}")


if __name__ == "__main__":
    main()
