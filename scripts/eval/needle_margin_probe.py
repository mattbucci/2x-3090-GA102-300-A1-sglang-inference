#!/usr/bin/env python3
"""Teacher-forced needle margin probe: a continuous deep-retrieval instrument.

The hit/miss needle grid moves by 1-2 cells between passes on identical code at
the edge of a model's reliable window, so it cannot rank serving knobs there.
This probe forces the assistant turn up to the point of recall and reads the
first answer token's distribution off the prefill: per cell it records the
log-probability of the correct first token, the top competitor and the margin.
Several samples per cell vary the password and the haystack offset.

    python scripts/eval/needle_margin_probe.py --port 23334 --tag flint-margin-base \
        --tokenizer ~/AI/models/hf-mattbucci/Flint-AWQ-INT4-g32-gptq/tokenizer.json \
        --lengths 196000,229000,250000 --depths 0.1,0.3,0.5 --samples 3 --mode direct

Receipt: benchmarks/quality/<tag>.json. Prompt framing defaults to Flint's
template (`--user-open`, `--assistant-open`).
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
CODES = ["BANANA42", "ORCHID17", "MARBLE58", "COBALT93", "WALNUT26", "SAFFRON71", "GRANITE35", "LANTERN84"]
QUESTION = "\n\nWhat is the secret password? Answer with just the password."
# echo: the forced thinking ends on the needle's own phrase (a verbatim copy cue, the easy case).
# direct: the thinking is closed without recalling anything, so the answer token must be found from the question alone.
MODES = {"echo": ("The user wants the secret password from the text. The secret password is:", " "),
         "direct": ("The user asks for the secret password. I will answer with just the password.</think>", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the served checkpoint")
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--depths", default="0.1,0.3,0.5")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--haystack", default=str(REPO / "benchmarks/quality/haystack-platypus.txt"))
    ap.add_argument("--chars-per-token", type=float, default=3.4)
    ap.add_argument("--user-open", default="<|user|>")
    ap.add_argument("--assistant-open", default="<|assistant|><think>")
    ap.add_argument("--mode", choices=sorted(MODES), default="echo")
    ap.add_argument("--no-flush", action="store_true")
    a = ap.parse_args()

    FORCED, sep = MODES[a.mode]
    base = f"http://127.0.0.1:{a.port}"
    tok = Tokenizer.from_file(str(Path(a.tokenizer).expanduser()))
    hay = Path(a.haystack).read_text(errors="replace")
    cpt = a.chars_per_token
    cells = []
    for ctx in [int(x) for x in a.lengths.split(",")]:
        for depth in [float(x) for x in a.depths.split(",")]:
            for s in range(a.samples):
                code = CODES[s % len(CODES)]
                # first answer token as it appears after the forced prefix
                ids = tok.encode(FORCED + sep + code, add_special_tokens=False).ids
                n_forced = len(tok.encode(FORCED, add_special_tokens=False).ids)
                want = tok.decode(ids[n_forced:n_forced + 1])
                off = (s * 3_000_011 + ctx) % (len(hay) // 2)
                rec = {"context": ctx, "depth": depth, "sample": s, "code": code, "want_token": want}
                for _ in range(4):
                    n_chars = int(ctx * cpt)
                    body = hay[off:off + n_chars]
                    pos = int(len(body) * depth)
                    prompt = (a.user_open + body[:pos] + f"\nThe secret password is: {code}.\n" + body[pos:]
                              + QUESTION + a.assistant_open + FORCED)
                    if not a.no_flush:
                        requests.post(base + "/flush_cache", timeout=60)
                    t0 = time.time()
                    try:
                        r = requests.post(base + "/v1/completions", json={
                            "model": "default", "prompt": prompt, "max_tokens": 8, "temperature": 0,
                            "logprobs": 20}, timeout=1800).json()
                    except Exception as e:  # noqa: BLE001
                        r = {"error": {"message": f"request failed: {e}"}}
                    if "choices" in r:
                        break
                    err = str((r.get("error") or {}).get("message", r))[:300]
                    m = re.search(r"input \((\d+) tokens\)", err) or re.search(r"(\d+) tokens from the input", err)
                    if not m:
                        rec["error"] = err
                        break
                    cpt = cpt * ctx / int(m.group(1)) * 0.995
                if "choices" in r:
                    ch = r["choices"][0]
                    lp = ch.get("logprobs") or {}
                    top = (lp.get("top_logprobs") or [{}])[0] or {}
                    want_lp = next((v for k, v in top.items() if k.strip() == want.strip() and k.strip()), None)
                    rivals = sorted((v for k, v in top.items() if k.strip() != want.strip()), reverse=True)
                    floor = min(top.values()) if top else None
                    rec.update({
                        "actual_tokens": r.get("usage", {}).get("prompt_tokens"),
                        "text": ch.get("text", "")[:60], "found": code in ch.get("text", ""),
                        "want_logprob": want_lp, "want_in_top20": want_lp is not None,
                        "rival_logprob": rivals[0] if rivals else None,
                        # a correct token outside the top-20 is scored at the top-20 floor (an upper bound)
                        "margin": ((want_lp if want_lp is not None else floor) - rivals[0]) if rivals and floor is not None else None,
                        "top1": max(top, key=top.get) if top else None, "seconds": round(time.time() - t0, 1)})
                cells.append(rec)
                print(json.dumps(rec), flush=True)

    summary = {}
    for ctx in sorted({c["context"] for c in cells}):
        for depth in sorted({c["depth"] for c in cells}):
            g = [c for c in cells if c["context"] == ctx and c["depth"] == depth and c.get("margin") is not None]
            if g:
                summary[f"{ctx}@{depth}"] = {"n": len(g), "found": sum(c["found"] for c in g),
                                             "mean_margin": round(statistics.mean(c["margin"] for c in g), 3),
                                             "min_margin": round(min(c["margin"] for c in g), 3)}
    ok = [c for c in cells if c.get("margin") is not None]
    out = {"tag": a.tag, "timestamp": datetime.now().isoformat(timespec="seconds"), "mode": a.mode, "forced_prefix": FORCED,
           "haystack": Path(a.haystack).name, "flush": not a.no_flush,
           "overall": {"n": len(ok), "found": sum(c["found"] for c in ok),
                       "mean_margin": round(statistics.mean(c["margin"] for c in ok), 3) if ok else None},
           "summary": summary, "cells": cells}
    p = REPO / "benchmarks" / "quality" / f"{a.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}  overall {out['overall']}")


if __name__ == "__main__":
    main()
