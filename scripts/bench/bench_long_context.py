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


def run_bench(base_url, model, input_len, output_len, tokenizer=None):
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
        # range_ratio=0 (upstream default) draws the prompt length UNIFORM in
        # [1, input_len] — a "250K" point can silently measure 60K. Pin exact.
        # (Found 2026-07-14: server-side prefill ground truth showed 121,922
        # total tokens across a 2048+131072+250000 sweep.)
        "--random-range-ratio", "1",
        "--disable-tqdm",
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    text = out.stdout + out.stderr
    metrics = {}
    # bench_serving prints lines like "Mean TPOT (ms):                 11.26"
    NUM = r"([\d.]+)"
    for line in text.splitlines():
        if "Total input tokens" in line:
            m = re.search(NUM + r"\s*$", line)
            if m:
                metrics["actual_input_tokens"] = int(float(m.group(1)))
        elif "Mean TPOT" in line:
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


def server_context_length(base_url):
    """Declared --context-length from the live server. A point at exactly this
    value leaves no room for output tokens: input+output overflows the window,
    the request degenerates (TTFT 0.0) and reports artifact tok/s (the 2026-07-14
    second depth bug — surfaced once range_ratio=1 made prompts hit full length)."""
    for ep in ("/server_info", "/get_server_info"):
        try:
            j = requests.get(f"{base_url}{ep}", timeout=10).json()
            v = j.get("context_length") or j.get("max_context_len")
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            continue
    return None


def server_max_tokens(base_url):
    """Real KV-pool capacity (max_total_num_tokens) from the live server, so we
    never bench a context the pool can't hold. An over-cap prompt is rejected /
    never decodes at that length, and bench_serving then logs an artifact tok/s
    (gemma4-26b "75 tok/s @262K" vs 34 @1K). Returns None if unavailable."""
    for ep in ("/server_info", "/get_server_info"):   # /server_info current; other deprecated
        try:
            j = requests.get(f"{base_url}{ep}", timeout=5).json()
            v = j.get("max_total_num_tokens")
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--output-tokens", type=int, default=100)
    ap.add_argument("--contexts", type=int, nargs="*", default=None,
                    help="Override context list")
    ap.add_argument("--output", default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer path. Required when the served model name "
                         "(--served-model-name on launch.sh) is not a valid HF "
                         "repo id or local path — bench_serving uses --model to "
                         "load the tokenizer.")
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    try:
        requests.get(f"{base}/health", timeout=5)
    except Exception as e:
        print(f"Server not reachable on port {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    model = server_model(base)
    requested = args.contexts or [c for c in DEFAULT_CONTEXTS if c <= args.max_context]
    max_kv = server_max_tokens(base)
    ctx_len = server_context_length(base)
    if max_kv:
        usable = max_kv - args.output_tokens - 256   # headroom: output + scaffolding
        if ctx_len:
            # input + output must fit the declared context window too
            usable = min(usable, ctx_len - args.output_tokens - 128)
        contexts = sorted({min(c, usable) for c in requested if min(c, usable) > 0})
        dropped = [c for c in requested if c > usable]
        if dropped:
            print(f"NOTE: KV pool max_total_num_tokens={max_kv}; capping input sweep "
                  f"at {usable} usable tokens. Dropping over-cap contexts {dropped} "
                  f"(they can't fit the pool and would log artifact tok/s).")
    else:
        contexts = requested
        print("WARN: couldn't read max_total_num_tokens (no /get_server_info); "
              "benching the full requested range — distrust any non-falling tok/s.")

    print(f"=== {args.name} ({model}) ===")
    print(f"Contexts: {contexts}  (output tokens: {args.output_tokens}, KV pool: {max_kv})\n")

    print("Warmup (3x 128-in 10-out)...")
    for _ in range(3):
        run_bench(base, model, 128, 10, tokenizer=args.tokenizer)

    results = []
    for ctx in contexts:
        print(f"  ctx={ctx:>6}: ", end="", flush=True)
        m = run_bench(base, model, ctx, args.output_tokens, tokenizer=args.tokenizer)
        if m.get("ttft_ms", 1) == 0.0 or "tpot_ms" not in m:
            print(f"  WARN: ctx={ctx} produced a degenerate measurement (ttft=0 / "
                  f"no TPOT) — marking invalid", flush=True)
            m["invalid"] = True
        actual = m.get("actual_input_tokens")
        if actual is not None and actual < 0.95 * ctx:
            print(f"  WARN: requested ctx={ctx} but server saw {actual} input tokens "
                  f"({100*actual/ctx:.0f}%) — depth label unreliable", flush=True)
            m["depth_shortfall"] = True
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
        "max_total_num_tokens": max_kv,
        "context_sweep": results,
    }
    out_path = args.output or f"benchmarks/{args.name.replace(' ', '_').lower()}_long_context.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
