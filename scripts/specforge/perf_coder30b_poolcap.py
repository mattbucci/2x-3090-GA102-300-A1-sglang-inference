#!/usr/bin/env python3
"""EAGLE3 pool-cap A/B instrument — coder-30b, depth-parameterized (lane 3090-E).

SERVER-CLIENT harness (drive a running launch.sh server over HTTP; NOT the
in-process sgl.Engine delta tool). Per depth rung it builds a real-code context
pad + a NOVEL-code generation instruction (EAGLE3-fair: output is novel, not
copied — copy-heavy content is the NGRAM lane's test, and would make this
result redundant), fires one measured request, and reports:

  - server-log `gen throughput` (AUTHORITATIVE decode tok/s — client TPOT
    under-measures bursty spec ~2x; parsed from --server-log between per-request
    file offsets, median of steady-state lines)
  - client tok/s + usage.prompt_tokens (correlation + true-depth verification)
  - accept_len from /get_server_info (spec runs; recursive key search for
    *spec_accept*)
  - per-card VRAM via nvidia-smi

Donors: NOVEL prompt set from perf_devstral_spec.py; HTTP/server-log discipline
from copyheavy_decode_bench.py (padding glob repointed at the LIVE v0.5.15 tree,
not the donor's stale v0512 path).

Usage:
  python scripts/specforge/perf_coder30b_poolcap.py --port 23334 \
      --server-log /tmp/coder30b-spec-logs/serve.log \
      --depths 14000 41000 61000 94000 --output-tokens 512 \
      --tag armB --out benchmarks/quality/coder30b-poolcap-armB.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

import requests

SRC_GLOB = "/data/sglang-rebase-v0515/python/sglang/srt/**/*.py"
# Fixed chars/tok estimates lie on this corpus (3.6 book-value vs 1.46 measured
# — the alphabetically-first srt files are number/symbol-dense). The ratio is
# CALIBRATED at startup via a 1-token probe against the live server, and
# re-derived from any "input (N tokens)" 400 as a safety net.
CHARS_PER_TOK = 2.9  # pre-calibration seed only

NOVEL_PROMPTS = [
    "Write a Python LRU cache class (dict + doubly linked list, O(1) get/put, type hints).",
    "Implement Dijkstra shortest path in Python with a heap, with a docstring and example.",
    "Write a Python function to merge overlapping intervals, with doctests.",
]

GEN_TPUT_RE = re.compile(r"gen throughput \(token/s\):\s*([\d.]+)")


def read_source_files(max_chars):
    files = sorted(glob.glob(SRC_GLOB, recursive=True))
    blob, used = [], 0
    for fp in files:
        try:
            t = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        if len(t) < 400 or len(t) > 30000:
            continue  # skip stubs and huge generated tables
        digits = sum(c.isdigit() for c in t)
        if digits / len(t) > 0.15:
            continue  # number-dense files tokenize <1 char/tok (arg_groups/overrides.py bomb)
        blob.append(f"# ===== file: {os.path.basename(fp)} =====\n{t}\n")
        used += len(t)
        if used >= max_chars:
            break
    return "".join(blob)


def server_gen_tputs(log_path, offset):
    """gen-throughput values logged after byte `offset`; returns (list, new_offset)."""
    with open(log_path, "rb") as f:
        f.seek(offset)
        chunk = f.read().decode("utf-8", errors="replace")
    vals = [float(v) for v in GEN_TPUT_RE.findall(chunk)]
    return vals, offset + len(chunk.encode("utf-8", errors="replace"))


def accept_len(base):
    """Recursive *spec_accept* search over /get_server_info (layout-robust)."""
    try:
        j = requests.get(f"{base}/get_server_info", timeout=10).json()
    except Exception:
        return None
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "spec_accept" in str(k) and isinstance(v, (int, float)):
                    found.append(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)
    return found[-1] if found else None


def vram():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


_TOKENS_400_RE = re.compile(r"input \((\d+) tokens\)")


def probe_tokens(base, model, content):
    """Exact server-side prompt_tokens for `content` via a 1-token request.
    Uses the 400 error body as the count when over-length (rejected pre-prefill,
    so an oversized probe costs nothing)."""
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": model, "max_tokens": 1, "temperature": 0.0,
        "messages": [{"role": "user", "content": content}]}, timeout=1200)
    if r.status_code == 400:
        m = _TOKENS_400_RE.search(r.text)
        if m:
            return int(m.group(1))
        r.raise_for_status()
    r.raise_for_status()
    return r.json().get("usage", {}).get("prompt_tokens")


def measure_depth(base, model, depth, output_tokens, log_path, log_offset, prompt_idx, ratio):
    instr = NOVEL_PROMPTS[prompt_idx % len(NOVEL_PROMPTS)]

    def build(chars):
        pad = read_source_files(chars) if chars > 0 else ""
        return (
            (f"Below is a reference codebase for context. Read it, then follow the "
             f"instruction at the end.\n\n{pad}\n\n" if pad else "")
            + f"Instruction: {instr}\nWrite NEW code — do not copy from the reference."
        )

    # Iterative sizing via server-verified token counts — a fixed chars/tok
    # ratio cannot survive a non-uniform corpus (fleet lesson: depth labels are
    # only trustworthy server-side). Proportional update, monotonic, ~2-3 iters.
    chars = max(0, int((depth - 300) * ratio))
    user = build(chars)
    for _ in range(4):
        actual = probe_tokens(base, model, user)
        if actual and abs(actual - depth) / depth <= 0.05:
            break
        if not actual:
            break
        chars = max(1000, int(chars * depth / actual))
        user = build(chars)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    log_offset = os.path.getsize(log_path)  # sizing probes stay out of the window
    t0 = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=3600)
    dt = time.time() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    tputs, new_offset = server_gen_tputs(log_path, log_offset)
    steady = [v for v in tputs if v >= 1.0]
    return {
        "approx_depth": depth,
        "actual_prompt_tokens": pt,
        "completion_tokens": ct,
        "wall_s": round(dt, 2),
        "client_tok_s": round(ct / dt, 1) if (ct and dt) else None,
        "server_gen_tok_s_median": round(median(steady), 1) if steady else None,
        "server_gen_tok_s_n": len(steady),
        "accept_len": accept_len(base),
        "vram_mib": vram(),
    }, new_offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--server-log", required=True,
                    help="the launch.sh server log (authoritative gen-throughput source)")
    ap.add_argument("--depths", type=int, nargs="+", required=True)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    requests.get(f"{base}/health", timeout=5).raise_for_status()
    model = requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]
    if not glob.glob(SRC_GLOB, recursive=True):
        raise SystemExit(f"padding glob matches nothing: {SRC_GLOB}")
    if not os.path.exists(args.server_log):
        raise SystemExit(f"server log not found: {args.server_log}")

    print(f"=== {args.tag}: {model} depths={args.depths} out={args.output_tokens} ===")
    # warmup (absorb graph capture); throw away its log window
    requests.post(f"{base}/v1/chat/completions", json={
        "model": model, "max_tokens": 32, "temperature": 0.0,
        "messages": [{"role": "user", "content": "Say OK."}]}, timeout=600)
    ratio = CHARS_PER_TOK  # seed only; per-depth sizing is server-verified
    offset = os.path.getsize(args.server_log)

    results = []
    for i, depth in enumerate(args.depths):
        print(f"  depth~{depth}: ", end="", flush=True)
        try:
            rec, offset = measure_depth(base, model, depth, args.output_tokens,
                                        args.server_log, offset, i, ratio)
        except requests.HTTPError as e:
            rec = {"approx_depth": depth, "error": str(e)}
            print(f"HTTP ERROR {e}")
            results.append(rec)
            continue
        print(f"actual={rec['actual_prompt_tokens']}  server={rec['server_gen_tok_s_median']} tok/s "
              f"(client {rec['client_tok_s']})  accept={rec['accept_len']}  vram={rec['vram_mib']}")
        results.append(rec)

    payload = {
        "tag": args.tag, "model": model, "engine": "sglang-v0.5.15",
        "output_tokens": args.output_tokens,
        "note": "server_gen_tok_s_median is authoritative (client under-measures bursty spec)",
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
