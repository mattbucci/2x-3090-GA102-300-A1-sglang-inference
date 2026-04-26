#!/usr/bin/env python3
"""Context-length sweep benchmark for Qwen3.5-27B.

Measures tok/s at different input context lengths to show how DeltaNet
performance scales (DeltaNet should be constant time regardless of context).

Usage:
    python scripts/test/bench_qwen35_context_sweep.py --port 23334
"""
import argparse
import json
import time
import requests
import os


def bench_decode(port, prompt, max_tokens, temperature=0.0):
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

    with requests.post(url, json=payload, stream=True, timeout=600) as resp:
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
                content = delta.get("content", "") or ""
                reasoning = delta.get("reasoning_content", "") or ""
                if content or reasoning:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    end = time.perf_counter()
    ttft = first_token_time - start if first_token_time else end - start
    decode_time = end - first_token_time if first_token_time else end - start
    decode_toks = max(token_count - 1, 1)
    toks_per_sec = decode_toks / decode_time if decode_time > 0 else 0

    return {
        "tokens": token_count,
        "ttft_ms": ttft * 1000,
        "tpot_ms": (decode_time / decode_toks * 1000) if decode_toks > 0 else 0,
        "tok_s": toks_per_sec,
    }


def make_prompt(target_tokens):
    """Create a prompt that approximates target_tokens input tokens."""
    # ~0.75 words per token for English text
    base = "Here is a list of random words for context padding: "
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
             "november", "oscar", "papa", "quebec", "romeo", "sierra",
             "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]
    # ~1.3 tokens per word on average
    num_words = int(target_tokens / 1.3)
    padding = " ".join(words[i % len(words)] for i in range(num_words))
    return base + padding + "\n\nNow, briefly say hello."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/qwen35-27b-awq/results.json")
    args = parser.parse_args()

    context_lengths = [128, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"Qwen3.5-27B Context Sweep — {args.output_tokens} output tokens, {args.iters} iters")
    print(f"{'Context':>8s} {'tok/s':>8s} {'TTFT':>8s} {'TPOT':>8s} {'Tokens':>8s}")
    print("-" * 50)

    results = []
    for ctx in context_lengths:
        prompt = make_prompt(ctx)
        best_toks = 0
        avg_toks = 0
        avg_ttft = 0
        avg_tpot = 0

        for i in range(args.iters):
            r = bench_decode(args.port, prompt, args.output_tokens)
            avg_toks += r["tok_s"]
            avg_ttft += r["ttft_ms"]
            avg_tpot += r["tpot_ms"]
            best_toks = max(best_toks, r["tok_s"])

        avg_toks /= args.iters
        avg_ttft /= args.iters
        avg_tpot /= args.iters

        print(f"{ctx:>8d} {avg_toks:>8.1f} {avg_ttft:>7.0f}ms {avg_tpot:>7.1f}ms {r['tokens']:>8d}")

        results.append({
            "context": ctx,
            "tok_per_sec": round(avg_toks, 1),
            "ttft_ms": round(avg_ttft, 1),
            "tpot_ms": round(avg_tpot, 1),
            "tokens": r["tokens"],
        })

    # Save
    all_results = {
        "model": "Qwen3.5-27B-AWQ-DeltaNet",
        "engine": "SGLang v0.5.10 + patches",
        "hardware": "2x RTX 3090 TP=2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_tokens": args.output_tokens,
        "context_sweep": results,
        "throughput_sweep": [],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    best = max(results, key=lambda r: r["tok_per_sec"])
    print(f"\nBest: {best['tok_per_sec']} tok/s at context={best['context']}")


if __name__ == "__main__":
    main()
