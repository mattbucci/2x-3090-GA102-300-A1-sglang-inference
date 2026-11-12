#!/usr/bin/env python3
"""Fresh-prefill TTFT per preset from bench_long_context natural-text sweeps (cache flushed per point,
server-verified actual_input_tokens). The random-prompt results.json TTFT is a prefix-cache hit and must
not be charted; this reads only prompt_source == "natural" files.

  python scripts/bench/ttft_chart.py --inputs 'benchmarks/fleet-w4a8-2026-11-01/*-arm-*.json' \
      --pick qwen38=2048,qwen36-dense=2048,gemma4-31b=2048,qwen35-moe=2048,qwen36=2048 --out benchmarks/all_models_ttft_natural.png
--pick selects which arm file stands for a preset (its shipped W4A8 setting); presets not listed use arm-0.
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_charts as fleet   # the fleet style: dark rcParams, per-model colour, class sections
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# sweep-file preset -> the launch.sh preset MODELS files it under
ALIAS = {"coder-30b-eval": "coder-30b"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmarks/fleet-w4a8-2026-11-01/*-arm-*.json")
    ap.add_argument("--pick", default="qwen38=2048,qwen36-dense=2048,gemma4-31b=2048,qwen35-moe=2048,qwen36=2048")
    ap.add_argument("--contexts", default="32768,131072")
    ap.add_argument("--extra", action="append", default=["qwen38=benchmarks/qwen38/w4a8-prefill-2026-11-01/arm-2048.json",
                             "flint=benchmarks/profiling/prefill-flashinfer-route-2026-11-11/ttft-fi.json"],
                    help="preset=path for natural sweeps whose filename lacks the preset (repeatable)")
    ap.add_argument("--out", default="benchmarks/all_models_ttft_natural.png")
    ap.add_argument("--title", default="Cold prefill — time to first token, natural text, cache flushed (TP=2, shipped presets)")
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
    by_preset = {m["preset"]: m for m in fleet.MODELS.values()}
    members = [{"preset": p, "meta": by_preset[ALIAS.get(p, p)], "pts": rows[p]} for p in rows]
    sections = [(title, sorted(group, key=lambda r: r["pts"].get(ctxs[-1], (0,))[0]))
                for title, group in fleet.fleet_groups(members)]
    order = [r["preset"] for _, section in sections for r in section]
    fig, ax = plt.subplots(figsize=(11, 0.46 * len(order) + 0.45 * len(sections) + 2.0))
    top = max(v[0] for r in members for v in r["pts"].values())
    h, y, ticks, labels = 0.74 / len(ctxs), 0.0, [], []
    alphas = [0.45, 0.95, 0.7]
    for title, section in sections:
        ax.axhline(y, color="#484f58", linewidth=1.0, zorder=1)
        ax.text(-0.012, y, f"{title} ({len(section)})", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=11.5, fontweight="bold", color="#f0f6fc")
        y += 0.85
        for r in section:
            for i, c in enumerate(ctxs):
                v = r["pts"].get(c, (0,))[0]
                if not v:
                    continue
                yy = y + (i - (len(ctxs) - 1) / 2) * h
                ax.barh(yy, v, h, color=r["meta"]["color"], alpha=alphas[i % 3], zorder=5)
                ax.text(v + top * 0.008, yy, f"{v:.1f} s", va="center", fontsize=9, color=fleet.INK)
            ticks.append(y); labels.append(r["meta"]["label"])
            y += 1.0
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9.5, color=fleet.INK)
    ax.set_ylim(y - 0.35, -0.75)
    ax.set_xlim(0, top * 1.1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("seconds to first token (lower is better; fastest at the deepest prompt first)")
    ax.set_title(a.title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[Patch(facecolor="#8b949e", alpha=alphas[i % 3], label=f"{c // 1024}K-token prompt")
                       for i, c in enumerate(ctxs)],
              loc="upper right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22", fontsize=9)
    ax.grid(True, axis="x", linestyle="--")
    fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")
    for p in order:
        print(p, {c: f"{rows[p][c][0]:.1f}s @ {rows[p][c][1]} tok" for c in rows[p]})
    print("wrote", a.out)

if __name__ == "__main__":
    main()
