#!/usr/bin/env python3
"""Long-context single-user sweep (1K → 256K).

Uses sglang.bench_serving with random prompts to measure TPOT/TTFT at long
context. Writes JSON matching the format consumed by generate_charts.py so
results drop straight into README tables.

Usage: python bench_long_context.py --port 23334 --name "Devstral-24B 262K" \\
           --max-context 262144 --output benchmarks/devstral-24b/long-context.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests


DEFAULT_CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144]


def run_bench(base_url, model, input_len, output_len):
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", "1",
        "--request-rate", "1",
        "--disable-tqdm",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    text = out.stdout + out.stderr
    metrics = {}
    # bench_serving prints lines like "Mean TPOT (ms):                 11.26"
    NUM = r"([\d.]+)"
    for line in text.splitlines():
        if "Mean TPOT" in line:
            m = re.search(NUM + r"\s*$", line)
            if m:
                metrics["tpot_ms"] = float(m.group(1))
        elif "Mean TTFT" in line:
            m = re.search(NUM + r"\s*$", line)
            if m:
                metrics["ttft_ms"] = float(m.group(1))
        elif "Output token throughput" in line:
            m = re.search(NUM + r"\s*$", line)
            if m:
                metrics["throughput"] = float(m.group(1))
    if "tpot_ms" in metrics and metrics["tpot_ms"] > 0:
        metrics["tok_per_sec"] = round(1000.0 / metrics["tpot_ms"], 1)
    return metrics if metrics else {"error": "parse failed", "raw": text[-2000:]}


def server_model(base_url):
    r = requests.get(f"{base_url}/v1/models", timeout=5)
    return r.json()["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--output-tokens", type=int, default=100)
    ap.add_argument("--contexts", type=int, nargs="*", default=None,
                    help="Override context list")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    try:
        requests.get(f"{base}/health", timeout=5)
    except Exception as e:
        print(f"Server not reachable on port {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    model = server_model(base)
    contexts = args.contexts or [c for c in DEFAULT_CONTEXTS if c <= args.max_context]

    print(f"=== {args.name} ({model}) ===")
    print(f"Contexts: {contexts}  (output tokens: {args.output_tokens})\n")

    print("Warmup (3x 128-in 10-out)...")
    for _ in range(3):
        run_bench(base, model, 128, 10)

    results = []
    for ctx in contexts:
        print(f"  ctx={ctx:>6}: ", end="", flush=True)
        m = run_bench(base, model, ctx, args.output_tokens)
        if "error" in m:
            print(f"ERROR ({m.get('error')})")
            results.append({"context": ctx, **m})
            break
        print(f"TPOT={m['tpot_ms']:.1f}ms  TTFT={m['ttft_ms']:.0f}ms  {m['tok_per_sec']:.1f} tok/s")
        results.append({"context": ctx, **m})

    payload = {
        "model": args.name,
        "engine": "SGLang",
        "hardware": "2x RTX 3090 TP=2",
        "benchmark_tool": "sglang.bench_serving",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_tokens": args.output_tokens,
        "context_sweep": results,
    }
    out_path = args.output or f"benchmarks/{args.name.replace(' ', '_').lower()}_long_context.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
