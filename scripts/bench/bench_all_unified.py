#!/usr/bin/env python3
"""Unified benchmark for all SGLang models using sglang.bench_serving.

Runs single-user context sweep + concurrency throughput sweep.
Measures proper TPOT (decode latency) and TTFT (prefill latency) separately.
Outputs results to stdout and JSON file.

Usage: python bench_all_unified.py --port 23334 --name "Model Name" --output benchmarks/out.json
"""
import argparse
import json
import subprocess
import sys
import os
import re
import time
import requests


def run_bench_serving(base_url, model, input_len, output_len, num_prompts, request_rate="inf"):
    """Run sglang.bench_serving and parse TPOT/TTFT/throughput."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--disable-tqdm",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None

    metrics = {}
    for line in output.split("\n"):
        if "Mean TPOT" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["tpot_ms"] = float(m.group(1))
        elif "Mean TTFT" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["ttft_ms"] = float(m.group(1))
        elif "Output token throughput" in line:
            m = re.search(r"([\d.]+)\s*tok", line)
            if m:
                metrics["throughput"] = float(m.group(1))
        elif "Mean E2E" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["e2e_ms"] = float(m.group(1))

    if "tpot_ms" in metrics:
        metrics["tok_per_sec"] = round(1000.0 / metrics["tpot_ms"], 1) if metrics["tpot_ms"] > 0 else 0
    return metrics if metrics else None


def get_model_name(base_url):
    """Get model name from server."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        return r.json()["data"][0]["id"]
    except:
        return "unknown"


def bench_context_sweep(base_url, model, context_lengths, output_tokens=100):
    """Single-user TPOT at various context lengths."""
    results = []
    for ctx in context_lengths:
        print(f"  ctx={ctx:>6}: ", end="", flush=True)
        metrics = run_bench_serving(base_url, model, ctx, output_tokens, num_prompts=1, request_rate=1)
        if metrics:
            results.append({
                "context": ctx,
                "tpot_ms": metrics.get("tpot_ms", 0),
                "ttft_ms": metrics.get("ttft_ms", 0),
                "tok_per_sec": metrics.get("tok_per_sec", 0),
                "throughput": metrics.get("throughput", 0),
            })
            print(f"TPOT={metrics.get('tpot_ms', 0):.1f}ms  "
                  f"TTFT={metrics.get('ttft_ms', 0):.0f}ms  "
                  f"{metrics.get('tok_per_sec', 0):.1f} tok/s")
        else:
            print("ERROR")
            results.append({"context": ctx, "error": "bench_serving failed"})
            if ctx > 4096:
                print("  Stopping context sweep (likely OOM)")
                break
    return results


def bench_throughput(base_url, model, concurrency_levels, output_tokens=256):
    """Throughput at various concurrency levels."""
    results = []
    for conc in concurrency_levels:
        num_prompts = max(conc * 2, 4)
        print(f"  conc={conc:>3}: ", end="", flush=True)
        metrics = run_bench_serving(base_url, model, 256, output_tokens, num_prompts, request_rate="inf")
        if metrics:
            results.append({
                "concurrency": conc,
                "tpot_ms": metrics.get("tpot_ms", 0),
                "throughput": metrics.get("throughput", 0),
                "tok_per_sec": metrics.get("throughput", 0),  # compat with chart generator
            })
            print(f"TPOT={metrics.get('tpot_ms', 0):.1f}ms  "
                  f"throughput={metrics.get('throughput', 0):.1f} tok/s")
        else:
            print("ERROR")
            results.append({"concurrency": conc, "error": "bench_serving failed"})
            break
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--name", required=True, help="Model name for output")
    p.add_argument("--output", default=None)
    p.add_argument("--context-max", type=int, default=32768)
    p.add_argument("--output-tokens", type=int, default=100)
    p.add_argument("--concurrency-max", type=int, default=32)
    args = p.parse_args()

    base = f"http://localhost:{args.port}"

    # Verify server
    try:
        requests.get(f"{base}/health", timeout=5)
    except:
        print(f"Server not responding on port {args.port}")
        sys.exit(1)

    model = get_model_name(base)
    print(f"=== {args.name} ===")
    print(f"Port: {args.port}, Model: {model}")
    print()

    # Warmup
    print("Warming up (3 requests)...")
    for _ in range(3):
        run_bench_serving(base, model, 128, 10, 1, request_rate=1)
    print()

    # Context sweep
    ctx_levels = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ctx_levels = [c for c in ctx_levels if c <= args.context_max]

    print(f"--- Single-user context sweep ({args.output_tokens} output tokens, TPOT via bench_serving) ---")
    context_results = bench_context_sweep(base, model, ctx_levels, args.output_tokens)

    # Concurrency sweep
    conc_levels = [1, 2, 4, 8, 16, 32]
    conc_levels = [c for c in conc_levels if c <= args.concurrency_max]

    print()
    print(f"--- Concurrent throughput (256 in / 256 out, bench_serving) ---")
    throughput_results = bench_throughput(base, model, conc_levels, 256)

    # Save
    all_results = {
        "model": args.name,
        "engine": "SGLang",
        "hardware": "2x RTX 3090 TP=2",
        "benchmark_tool": "sglang.bench_serving",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_tokens": args.output_tokens,
        "context_sweep": context_results,
        "throughput_sweep": throughput_results,
    }
    out_path = args.output or f"benchmarks/{args.name.replace(' ', '_').lower()}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Regenerate charts
    chart_script = os.path.join(os.path.dirname(__file__), "generate_charts.py")
    if os.path.exists(chart_script):
        print("\nRegenerating benchmark charts...")
        subprocess.run([sys.executable, chart_script], check=False)


if __name__ == "__main__":
    main()
