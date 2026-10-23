#!/usr/bin/env python3
"""Collect the served model's per-position prompt logprobs for a fixed token-id prompt.

Usage: served_prompt_logprobs.py <ids.json> <out.json> [url]
  ids.json  {"input_ids": [...]} (e.g. from flint_cpu_reference.py --save-ids)
  url       default http://127.0.0.1:23334

POSTs /generate with return_logprob + top_logprobs_num=20 + max_new_tokens=1 and writes
{"input_ids", "served", "next_lp": [lp of token t+1 at position t], "top": [[(id, lp) ...] per position]}
— the same shape flint_cpu_reference.py emits, so compare_prompt_logprobs.py can read both.
Position t holds the distribution predicting input_ids[t+1]; position 0 has no logprob (None).
"""
import json, sys, urllib.request
ids = json.load(open(sys.argv[1]))["input_ids"]; out = sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:23334"
served = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 20}
req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
mi = r["meta_info"]
tok = mi["input_token_logprobs"]      # [(lp, id, text|None)] aligned to prompt tokens, first is None-lp
top = mi["input_top_logprobs"]        # per prompt token: [(lp, id, text)] or None
# SGLang reports, at prompt index i, the logprob of token i given tokens < i. Re-key to position t = i-1.
next_lp = [None] * len(ids); tops = [None] * len(ids)
for i in range(1, len(ids)):
    if tok[i] and tok[i][0] is not None:
        next_lp[i - 1] = tok[i][0]
    if top[i]:
        tops[i - 1] = [(int(t[1]), float(t[0])) for t in top[i]]
json.dump({"input_ids": ids, "served": served, "url": base, "next_lp": next_lp, "top": tops}, open(out, "w"))
n = sum(1 for x in next_lp if x is not None)
print(f"served_prompt_logprobs: {served} positions={n}/{len(ids)-1} -> {out}")
