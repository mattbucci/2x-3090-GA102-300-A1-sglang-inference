#!/usr/bin/env python
"""Cache-hit extend latency: the agentic turn shape (a long cached prefix + a few new tokens + a 1-token answer).
For each context point: prefill the natural prompt once (warm), then time requests that reuse the prefix and append
N new tokens (N = 1, 16, 128) with max_tokens=1 — TTFT there is the extend step over the cached prefix. The fresh
prefill TTFT is recorded alongside. R9700 (patch 107) found their <=16-token cache-hit extends ran through the full
extend kernel at 523 ms per turn at 158K and halved it by routing them through the split-KV decode/verify kernel;
this measures where our backends (flashinfer presets, triton presets) sit before touching anything.

  python scripts/bench/bench_cache_hit_extend.py --port 23334 --name qwen38 --contexts 32768 131072 244000 \
      --out benchmarks/quality/cache-hit-extend-<preset>-<date>.json
"""
import argparse, json, os, re, time, requests

HAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "quality", "haystack-platypus.txt")
AUTH = ({"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"} if os.environ.get("OPENAI_API_KEY") else {})

def chat(base, model, prompt, max_tokens, timeout=1800):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                      timeout=timeout, headers=AUTH)
    dt = time.perf_counter() - t0
    j = r.json()
    return dt, j

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=23334); ap.add_argument("--name", required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072, 244000]); ap.add_argument("--new", type=int, nargs="+", default=[1, 16, 128])
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--out", required=True)
    a = ap.parse_args(); base = f"http://127.0.0.1:{a.port}"
    model = requests.get(f"{base}/v1/models", headers=AUTH).json()["data"][0]["id"]
    text = open(HAY).read()
    while len(text) < 262144 * 4: text += "\n\n" + text
    cpt = 3.4; rows = []
    for ctx in a.contexts:
        # size the prefix to ctx tokens under the served tokenizer (rescale on the server's own count)
        for _ in range(4):
            prefix = text[: int(ctx * cpt)]
            requests.post(f"{base}/flush_cache", headers=AUTH, timeout=60)
            dt, j = chat(base, model, prefix + "\n\nReply with the single word OK.", 1)
            if "usage" in j: break
            m = re.search(r"input \((\d+) tokens\)", str(j)) or re.search(r"(\d+) tokens from the input", str(j))
            if not m: raise SystemExit(f"prefill rejected: {str(j)[:200]}")
            cpt = cpt * ctx / int(m.group(1)) * 0.99
        actual = j["usage"]["prompt_tokens"]; fresh_ms = dt * 1000
        row = {"context": ctx, "actual_prefix_tokens": actual, "fresh_prefill_ms": round(fresh_ms, 1), "extend_ms": {}}
        for n in a.new:
            times = []
            for k in range(a.repeats):
                # new tokens differ per repeat so the suffix itself is never cached; the prefix stays a radix hit
                suffix = f"\n\n[turn {k}] " + ("tool result line %d. " % k) * max(1, n // 6)
                suffix = suffix[: int(n * 4.0)] if n > 1 else "\n\nX"
                dt, j = chat(base, model, prefix + suffix, 1)
                times.append(dt * 1000)
            u = j.get("usage", {})
            row["extend_ms"][str(n)] = {"median": round(sorted(times)[len(times) // 2], 1), "all": [round(t, 1) for t in times],
                                        "prompt_tokens": u.get("prompt_tokens")}
            print(f"{a.name} ctx={actual:>7} new={n:>4}: extend {row['extend_ms'][str(n)]['median']:.0f} ms (fresh prefill {fresh_ms/1000:.1f} s)", flush=True)
        rows.append(row)
    json.dump({"name": a.name, "model": model, "timestamp": time.strftime("%Y-%m-%d %H:%M"), "rows": rows}, open(a.out, "w"), indent=1)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
