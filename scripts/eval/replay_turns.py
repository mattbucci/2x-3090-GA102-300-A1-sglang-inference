#!/usr/bin/env python3
"""Replay captured agent turns N times against a running server and count
degenerate-repetition runaways.

Usage: replay_turns.py <label> <out.jsonl> <n> <max_tokens> <extra-json> <replay.json[.gz]>...
  label       free text recorded on every row (the serving config under test)
  extra-json  per-request sampling overrides merged into the body, e.g. '{}' or
              '{"repetition_penalty":1.05}' (omit → the server's --sampling-defaults)
  replay.json {messages, tools, input_tokens} as written by build_opencode_replay.py
Env: REPLAY_URL (default http://127.0.0.1:23334), REPLAY_MODEL (default: first /v1/models id).

A sample counts as a loop when finish_reason is 'length' or any of reasoning /
content / tool-call arguments contains a 12–80-char n-gram repeated 5× back to
back; loop rows carry a 350-char window around the first derail (sample_*).
Prints REPLAY_DONE at the end. Receipt: benchmarks/quality/flint/runaway-repetition-replay-2026-10-22.md
"""
import gzip, os, sys, json, re, time, urllib.request
label, out, n, mx, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5])
files = sys.argv[6:]
base = os.environ.get("REPLAY_URL", "http://127.0.0.1:23334")
model = os.environ.get("REPLAY_MODEL") or json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
LOOP = re.compile(r'(.{12,80}?)\1\1\1\1', re.S)
def loopy(txt):
    m = LOOP.search(txt)
    return (m.start() if m else None)
def load(f):
    with (gzip.open(f, 'rt') if f.endswith('.gz') else open(f)) as fh:
        return json.load(fh)
fo = open(out, 'a')
for f in files:
    r = load(f); tag = os.path.basename(f).replace('.json.gz', '').replace('.json', '')
    for i in range(n):
        body = {"model": model, "messages": r['messages'], "tools": r['tools'], "tool_choice": "auto",
                "max_tokens": mx, "stream": False}
        body.update(extra)
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=3600))
        except Exception as e:
            row = {"label": label, "replay": tag, "i": i, "error": str(e)[:300]}
            print(json.dumps(row)); fo.write(json.dumps(row) + "\n"); fo.flush(); continue
        ch = resp["choices"][0]; m = ch["message"]; u = resp.get("usage", {})
        reasoning = m.get("reasoning_content") or ""; content = m.get("content") or ""
        tcs = m.get("tool_calls") or []
        args = "".join(tc["function"].get("arguments", "") for tc in tcs)
        texts = {"reasoning": reasoning, "content": content, "args": args}
        derail = {k: loopy(v) for k, v in texts.items() if v}
        looped = ch.get("finish_reason") == "length" or any(v is not None for v in derail.values())
        row = {"label": label, "replay": tag, "ctx": r['input_tokens'], "i": i, "finish": ch.get("finish_reason"),
               "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
               "reasoning_chars": len(reasoning), "content_chars": len(content), "tool_calls": [tc["function"]["name"] for tc in tcs],
               "args_chars": len(args), "derail": derail, "loop": looped, "secs": round(time.time() - t0, 1), "extra": extra}
        if looped:
            for k, v in texts.items():
                p = derail.get(k)
                if p is not None: row["sample_" + k] = v[max(0, p - 200):p + 150]
        print(json.dumps(row)[:400], flush=True); fo.write(json.dumps(row) + "\n"); fo.flush()
print("REPLAY_DONE", label, out)
