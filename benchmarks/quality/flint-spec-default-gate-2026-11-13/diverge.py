#!/usr/bin/env python3
"""Greedy token streams with logprobs for the reasoning-probe prompts; one JSON per arm. Compare with diverge_cmp.py."""
import sys, json, random, requests
sys.path.insert(0, "/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/eval")
import probe_256k_quality as P
from pathlib import Path
arm, out = sys.argv[1], sys.argv[2]
P._HAYSTACK = Path("/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/benchmarks/quality/haystack-platypus.txt").read_text(encoding="utf-8", errors="replace")
url = "http://localhost:23334/v1/chat/completions"
res = {}
for L in (2048, 32768):
    rng = random.Random(42)
    for tname, tfn in P.TASKS.items():
        facts, q, check = tfn(rng)
        prompt = P.plant(P.fill(L, 3.1), facts) + "\n\n" + q
        for rep in (1, 2, 3):
            r = requests.post(url, json={"model": "default", "messages": [{"role": "user", "content": prompt}], "max_tokens": 700, "temperature": 0, "logprobs": True, "top_logprobs": 5}, timeout=900).json()
            ch = r["choices"][0]
            lp = (ch.get("logprobs") or {}).get("content") or []
            res[f"{L}/{tname}/{rep}"] = {"finish": ch.get("finish_reason"), "ct": r["usage"]["completion_tokens"], "pt": r["usage"]["prompt_tokens"],
                "toks": [[t["token"], round(t["logprob"], 3), [[x["token"], round(x["logprob"], 3)] for x in t.get("top_logprobs", [])]] for t in lp]}
            print(arm, L, tname, rep, ch.get("finish_reason"), r["usage"]["completion_tokens"], len(lp), flush=True)
json.dump(res, open(out, "w"))
