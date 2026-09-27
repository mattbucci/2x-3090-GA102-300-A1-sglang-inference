#!/usr/bin/env python3
"""DSpark A/B depth bench — R9700 6b59279 protocol, our CUDA stack.

Greedy chat/completions at each depth, flush_cache before each request, N runs/arm;
decode tok/s = (completion_tokens-1)/(t_last-t_first) client-side streaming.
Accept-len is read from the server log's 'Decode batch ... accept' lines separately.

  python depth_bench.py --arm nospec|dspark --port 23399 [--depths 60,49000]
"""
import argparse, json, time, urllib.request, sys

SHORT = "Write a complete Python module: binary search, merge sort, quicksort, a min-heap class, and Dijkstra shortest path. Full docstrings."

_TOK = None
def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(
            "/data/models/hf-mattbucci/Qwen3.8-27B-AWQ", trust_remote_code=True)
    return _TOK

def filler(ntok):
    # Build a deterministic context and TRUNCATE to exactly ~ntok tokens with the
    # real tokenizer, then append a short question (its tokens included).
    unit = ("Paxos and Raft are consensus protocols; LSM trees back many key-value "
            "stores; write-ahead logging ensures durability; MVCC gives snapshot isolation. ")
    q = "\n\nQuestion: Explain in about 250 words how write-ahead logging, MVCC, and consensus interact in a distributed database. Be thorough."
    tok = _tokenizer()
    qn = len(tok(q)["input_ids"])
    body = unit * (max(1, ntok // 8))          # overshoot, then trim
    ids = tok(body)["input_ids"][: max(1, ntok - qn)]
    return tok.decode(ids) + q

def run_once(port, prompt, max_tokens):
    # flush radix cache
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/flush_cache", method="POST"), timeout=30).read()
    except Exception as e:
        print("  (flush_cache warn:", e, ")")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; ntok = 0; usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta", {}) or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now; ntok += 1
    comp = (usage or {}).get("completion_tokens", ntok)
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (comp - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first) else float("nan")
    return dict(ttft=round(ttft, 2), decode_toks=round(dec, 1), completion_tokens=comp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", type=int, default=23399)
    ap.add_argument("--depths", default="60,49000")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    results = []
    for d in depths:
        prompt = SHORT if d <= 200 else filler(d)
        maxt = 400 if d <= 200 else 500
        for i in range(a.runs):
            r = run_once(a.port, prompt, maxt)
            r.update(arm=a.arm, depth=d, run=i)
            print(f"  {a.arm:7s} depth~{d:6d} run{i}: decode={r['decode_toks']:6} tok/s  "
                  f"ttft={r['ttft']:7}s  comp={r['completion_tokens']}")
            results.append(r)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print("wrote", a.out)

if __name__ == "__main__":
    main()
