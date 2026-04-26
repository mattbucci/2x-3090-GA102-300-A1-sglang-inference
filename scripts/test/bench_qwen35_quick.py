#!/usr/bin/env python3
"""Quick single-user decode benchmark for Qwen3.5-27B optimization cycles.

Sends a short prompt and measures pure decode tok/s by requesting a long
completion. Uses streaming to get time-to-first-token and per-token timing.

Usage:
    python scripts/test/bench_qwen35_quick.py
    python scripts/test/bench_qwen35_quick.py --port 23334 --max-tokens 200
    python scripts/test/bench_qwen35_quick.py --prompt "Write hello world in Python"
"""
import argparse
import json
import time
import requests


def bench_decode(port, prompt, max_tokens, temperature=0.0):
    """Send a completion request and measure decode throughput."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = []

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                # Count tokens from both content and reasoning_content
                content = delta.get("content", "") or ""
                reasoning = delta.get("reasoning_content", "") or ""
                text = content + reasoning
                if text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
                    full_text.append(text)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    end = time.perf_counter()
    total_time = end - start
    ttft = first_token_time - start if first_token_time else total_time
    decode_time = end - first_token_time if first_token_time else total_time

    # tok/s is decode tokens / decode time (excludes TTFT)
    decode_toks = max(token_count - 1, 1)  # first token is part of TTFT
    toks_per_sec = decode_toks / decode_time if decode_time > 0 else 0

    return {
        "tokens": token_count,
        "total_time": total_time,
        "ttft": ttft,
        "decode_time": decode_time,
        "tok_s": toks_per_sec,
        "ms_per_token": (decode_time / decode_toks * 1000) if decode_toks > 0 else 0,
        "text_preview": "".join(full_text)[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="Count from 1 to 500, one number per line.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=3, help="Benchmark iterations")
    args = parser.parse_args()

    print(f"Benchmarking Qwen3.5-27B decode @ localhost:{args.port}")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Max tokens: {args.max_tokens}  Warmup: {args.warmup}  Iters: {args.iters}")
    print()

    # Warmup
    for i in range(args.warmup):
        print(f"  Warmup {i+1}/{args.warmup}...", end=" ", flush=True)
        r = bench_decode(args.port, args.prompt, args.max_tokens)
        print(f"{r['tok_s']:.1f} tok/s ({r['tokens']} tokens)")

    # Benchmark
    results = []
    for i in range(args.iters):
        print(f"  Run {i+1}/{args.iters}...", end=" ", flush=True)
        r = bench_decode(args.port, args.prompt, args.max_tokens)
        results.append(r)
        print(f"{r['tok_s']:.1f} tok/s  TTFT={r['ttft']*1000:.0f}ms  "
              f"TPOT={r['ms_per_token']:.1f}ms  ({r['tokens']} tokens)")

    # Summary
    avg_toks = sum(r["tok_s"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results) * 1000
    avg_tpot = sum(r["ms_per_token"] for r in results) / len(results)
    best_toks = max(r["tok_s"] for r in results)

    print(f"\n{'='*50}")
    print(f"  Average: {avg_toks:.1f} tok/s  TTFT={avg_ttft:.0f}ms  TPOT={avg_tpot:.1f}ms")
    print(f"  Best:    {best_toks:.1f} tok/s")
    print(f"  Target:  ~22.7 tok/s (theoretical)")
    print(f"{'='*50}")

    # Show text quality
    print(f"\nOutput preview: {results[0]['text_preview'][:150]}...")


if __name__ == "__main__":
    main()
