#!/usr/bin/env python
"""Drive repeated 1-token follow-up turns on a cached prefix so a sampler (py-spy) can profile the server's CPU side.
Primes one natural-text prefix (~--tokens), then sends --turns chat completions that reuse it with a different short
suffix each time (max_tokens 1), printing each TTFT. Pair with: sudo py-spy record --pid <tokenizer-manager|scheduler>.
"""
import argparse, json, os, sys, time, requests

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=23334); ap.add_argument("--tokens", type=int, default=197000)
    ap.add_argument("--turns", type=int, default=40); ap.add_argument("--settle", type=float, default=0.0, help="seconds to wait after priming (start the sampler meanwhile)")
    a = ap.parse_args(); base = f"http://127.0.0.1:{a.port}"
    model = requests.get(base + "/v1/models").json()["data"][0]["id"]
    hay = open(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "quality", "haystack-platypus.txt")).read()
    while len(hay) < a.tokens * 5: hay += hay
    prefix = hay[: int(a.tokens * 3.6)]
    def turn(suffix):
        t0 = time.perf_counter()
        r = requests.post(base + "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": prefix + suffix}], "max_tokens": 1, "temperature": 0, "stream": False}, timeout=3600)
        dt = time.perf_counter() - t0; j = r.json()
        if "usage" not in j: print("ERROR", str(j)[:200]); sys.exit(1)
        return dt, j["usage"]["prompt_tokens"]
    dt, pt = turn("\n\nPrime."); print(f"primed {pt} tokens in {dt:.1f}s", flush=True)
    if a.settle: time.sleep(a.settle)
    ts = []
    for k in range(a.turns):
        dt, pt = turn(f"\n\n[turn {k}] reply briefly."); ts.append(dt * 1000)
        print(f"turn {k:3}: {dt*1000:7.1f} ms  prompt_tokens={pt}", flush=True)
    ts.sort(); print(f"median turn {ts[len(ts)//2]:.1f} ms over {a.turns} turns at {pt} tokens")

if __name__ == "__main__":
    main()
