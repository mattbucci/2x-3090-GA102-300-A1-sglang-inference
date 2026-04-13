#!/usr/bin/env python3
"""Benchmark long-context performance using sglang.bench_serving.

Tests TPOT and TTFT at various context lengths with proper decode measurement.
"""

import argparse
import json
import re
import subprocess
import sys
import requests


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
    return metrics if metrics else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    try:
        requests.get(f"{base_url}/health", timeout=5)
    except:
        print(f"Server not ready at {base_url}")
        sys.exit(1)

    try:
        model = requests.get(f"{base_url}/v1/models", timeout=5).json()["data"][0]["id"].split("/")[-1]
    except:
        model = "unknown"

    out = args.output_tokens
    print(f"Model: {model}")
    print(f"Output tokens: {out}")
    print(f"{'Context':>10s}  {'TPOT':>8s}  {'TTFT':>10s}  {'Decode tok/s':>12s}")
    print("-" * 50)

    tests = [
        (256, "256"),
        (1024, "1K"),
        (4096, "4K"),
        (8192, "8K"),
        (16384, "16K"),
        (32768, "32K"),
    ]

    for input_len, label in tests:
        metrics = run_bench(base_url, model, input_len, out)
        if metrics:
            tpot = metrics.get("tpot_ms", 0)
            ttft = metrics.get("ttft_ms", 0)
            tok_s = 1000.0 / tpot if tpot > 0 else 0
            print(f"{label:>10s}  {tpot:>7.1f}ms  {ttft:>9.0f}ms  {tok_s:>11.1f}")
        else:
            print(f"{label:>10s}  ERROR")


if __name__ == "__main__":
    main()
