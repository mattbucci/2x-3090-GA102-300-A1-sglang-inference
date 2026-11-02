#!/usr/bin/env python3
"""Fresh-prefill TTFT per preset from bench_long_context natural-text sweeps (cache flushed per point,
server-verified actual_input_tokens). The random-prompt results.json TTFT is a prefix-cache hit and must
not be charted; this reads only prompt_source == "natural" files.

  python scripts/bench/ttft_chart.py --inputs 'benchmarks/fleet-w4a8-2026-11-01/*-arm-*.json' \
      --pick qwen38=2048,qwen36-dense=2048,gemma4-31b=2048,qwen35-moe=2048,qwen36=2048 --out benchmarks/all_models_ttft_natural.png
--pick selects which arm file stands for a preset (its shipped W4A8 setting); presets not listed use arm-0.
"""
import argparse, glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmarks/fleet-w4a8-2026-11-01/*-arm-*.json")
    ap.add_argument("--pick", default="qwen38=2048,qwen36-dense=2048,gemma4-31b=2048,qwen35-moe=2048,qwen36=2048")
    ap.add_argument("--contexts", default="32768,131072")
    ap.add_argument("--extra", action="append", default=["qwen38=benchmarks/qwen38/w4a8-prefill-2026-11-01/arm-2048.json"],
                    help="preset=path for natural sweeps whose filename lacks the preset (repeatable)")
    ap.add_argument("--out", default="benchmarks/all_models_ttft_natural.png")
    ap.add_argument("--title", default="Fresh-prefill TTFT, natural text, cache flushed (2x RTX 3090, 300 W, shipped presets)")
    a = ap.parse_args()
    pick = dict(kv.split("=") for kv in a.pick.split(",") if kv)
    ctxs = [int(c) for c in a.contexts.split(",")]
    rows = {}
    for path in sorted(glob.glob(a.inputs)):
        m = re.match(r".*/([a-z0-9-]+)-arm-(\d+)\.json$", path)
        if not m:
            continue
        preset, arm = m.group(1), m.group(2)
        if pick.get(preset, "0") != arm:
            continue
        d = json.load(open(path))
        if d.get("prompt_source") != "natural":
            continue
        pts = {p["context"]: p for p in d.get("context_sweep", []) if "ttft_ms" in p}
        rows[preset] = {c: (pts[c]["ttft_ms"] / 1000.0, pts[c].get("actual_input_tokens")) for c in ctxs if c in pts}
    for kv in a.extra:
        preset, path = kv.split("=", 1)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        if d.get("prompt_source") != "natural":
            continue
        pts = {p["context"]: p for p in d.get("context_sweep", []) if "ttft_ms" in p}
        rows[preset] = {c: (pts[c]["ttft_ms"] / 1000.0, pts[c].get("actual_input_tokens")) for c in ctxs if c in pts}
    if not rows:
        raise SystemExit("no natural-text sweeps matched")
    order = sorted(rows, key=lambda p: rows[p].get(ctxs[-1], (0,))[0])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(order) + 2))
    y = range(len(order)); h = 0.38
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for i, c in enumerate(ctxs):
        vals = [rows[p].get(c, (0,))[0] for p in order]
        ax.barh([yy + (i - (len(ctxs) - 1) / 2) * h for yy in y], vals, height=h, color=colors[i % 3], label=f"{c // 1024}K prompt")
        for yy, v in zip(y, vals):
            if v:
                ax.text(v + max(vals) * 0.01, yy + (i - (len(ctxs) - 1) / 2) * h, f"{v:.1f} s", va="center", fontsize=8)
    ax.set_yticks(list(y)); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel("time to first token, seconds (fresh prefill, server-verified token count)")
    ax.set_title(a.title, fontsize=11); ax.legend(loc="upper right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    for p in order:
        print(p, {c: f"{rows[p][c][0]:.1f}s @ {rows[p][c][1]} tok" for c in rows[p]})
    print("wrote", a.out)

if __name__ == "__main__":
    main()
