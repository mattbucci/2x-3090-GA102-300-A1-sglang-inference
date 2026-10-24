#!/usr/bin/env python
"""Power-cap sweep -> receipt JSON + chart (benchmarks/power-sweep-<date>/).

Inputs (written by the sweep driver, see benchmarks/power-sweep-2026-10-24/README.md):
  serve-results.json    {"<cap_w>": {"<ctx>": {ttft_ms, tpot_ms, tok_per_sec, actual_input_tokens, ...}}}
  tensor-results.jsonl  one tensor_rate.py row per GPU per cap, with "pl_w" prepended
  samples.csv           2 s nvidia-smi + DIMM samples tagged with the sweep phase

Usage:
  python scripts/bench/power_sweep_chart.py <sweep_dir> [--pick <cap_w>]

Prints the per-cap table, writes summary.json and power-sweep.png next to the inputs.
Chart rules: one y-axis per panel (small multiples share x), fixed series colours,
direct labels at the right edge, recessive grid, chosen cap marked as a reference line.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {"1024": ("1K", "#2a78d6"), "32768": ("32K", "#eb6834"), "131072": ("128K", "#1baf7a")}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def load(d):
    serve = json.load(open(os.path.join(d, "serve-results.json")))
    tensor = defaultdict(list)
    for line in open(os.path.join(d, "tensor-results.jsonl")):
        line = line.strip()
        if line:
            r = json.loads(line)
            tensor[int(r["pl_w"])].append(r)
    samp = defaultdict(lambda: defaultdict(list))  # cap -> gpu -> rows (serve phases only)
    dimm = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(d, "samples.csv"))):
        ph = r["phase"]
        if not ph.startswith("pl") or "-ctx" not in ph:
            continue
        cap = int(ph[2:].split("-")[0])
        samp[cap][r["gpu"]].append(r)
        for k in ("dimm50_c", "dimm52_c"):
            if r[k]:
                dimm[cap].append(float(r[k]))
    return serve, tensor, samp, dimm


def summarize(serve, tensor, samp, dimm):
    rows = []
    for cap in sorted(int(c) for c in serve):
        s = serve[str(cap)]
        row = {"cap_w": cap}
        for ctx, (lab, _) in SERIES.items():
            m = s.get(ctx, {})
            row[f"decode_{lab}"] = m.get("tok_per_sec")
            row[f"ttft_{lab}_ms"] = m.get("ttft_ms")
        for g in ("0", "1"):
            rs = samp.get(cap, {}).get(g, [])
            if rs:
                row[f"gpu{g}_max_temp_c"] = max(int(r["temp_c"]) for r in rs)
                row[f"gpu{g}_mean_power_w"] = round(statistics.mean(float(r["power_w"]) for r in rs), 1)
                row[f"gpu{g}_mean_sm_mhz"] = round(statistics.mean(int(r["sm_mhz"]) for r in rs))
        if dimm.get(cap):
            row["dimm_max_c"] = max(dimm[cap])
        t = tensor.get(cap, [])
        if t:
            row["fp16_tflops_mean"] = round(statistics.mean(float(r["fp16_fp32acc_tflops"]) for r in t), 1)
            row["bf16_tflops_mean"] = round(statistics.mean(float(r["bf16_tflops"]) for r in t), 1)
            row["tensor_sm_mhz_mean"] = round(statistics.mean(int(r["sm_clock_during_mhz"]) for r in t))
        rows.append(row)
    return rows


def pick(rows):
    """Knee: the lowest cap within 2 % of the sweep-best decode at 1K AND 32K, and
    within 5 % of the best 128K TTFT. The user picks on this + the temperature column."""
    best = {k: max(r[k] for r in rows if r.get(k)) for k in ("decode_1K", "decode_32K")}
    best_ttft = min(r["ttft_128K_ms"] for r in rows if r.get("ttft_128K_ms"))
    for r in rows:
        ok = all(r.get(k) and r[k] >= 0.98 * best[k] for k in best)
        ok = ok and r.get("ttft_128K_ms") and r["ttft_128K_ms"] <= 1.05 * best_ttft
        if ok:
            return r["cap_w"]
    return rows[-1]["cap_w"]


def chart(rows, out_png, chosen, title):
    caps = [r["cap_w"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=160, facecolor=SURFACE, sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.26, left=0.07, right=0.97, top=0.88, bottom=0.09)

    def style(ax, ylabel, title_):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(BASE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
        ax.set_title(title_, loc="left", color=INK, fontsize=10, fontweight="semibold", pad=8)
        ax.axvline(chosen, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(caps[0] - 5, caps[-1] + 48)  # room for the right-edge direct labels

    def series(ax, key_fmt, fmt):
        for ctx, (lab, col) in SERIES.items():
            ys = [r.get(key_fmt.format(lab)) for r in rows]
            ax.plot(caps, ys, color=col, linewidth=2, marker="o", markersize=3.5, zorder=3)
            ax.annotate(lab, (caps[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
            ax.annotate(fmt(ys[-1]), (caps[-1], ys[-1]), xytext=(5, -9), textcoords="offset points",
                        color=MUTED, fontsize=7, va="center")

    ax = axes[0][0]
    style(ax, "tok/s", "Decode, qwen38 single-user (fresh prefill)")
    series(ax, "decode_{}", lambda v: f"{v:.0f}")
    ax = axes[0][1]
    style(ax, "seconds", "Time to first token (prefill)")
    for ctx, (lab, col) in SERIES.items():
        ys = [r.get(f"ttft_{lab}_ms") / 1000 for r in rows]
        ax.plot(caps, ys, color=col, linewidth=2, marker="o", markersize=3.5, zorder=3)
        ax.annotate(f"{lab}  {ys[-1]:.2f} s", (caps[-1], ys[-1]), xytext=(5, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10]); ax.set_yticklabels(["0.3", "1", "3", "10"])
    ax = axes[1][0]
    style(ax, "TFLOPS (mean of both cards)", "Tensor rate, 8192² GEMM (unlocked)")
    for k, lab, col, dy in (("bf16_tflops_mean", "bf16", "#2a78d6", 5), ("fp16_tflops_mean", "fp16/fp32-acc", "#eb6834", -6)):
        ys = [r.get(k) for r in rows]
        if any(ys):
            ax.plot(caps, ys, color=col, linewidth=2, marker="o", markersize=3.5, zorder=3)
            ax.annotate(f"{lab}  {ys[-1]:.0f}", (caps[-1], ys[-1]), xytext=(5, dy),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax = axes[1][1]
    style(ax, "°C (max during the serving points)", "Temperature")
    for k, lab, col in (("gpu0_max_temp_c", "GPU 0", "#2a78d6"), ("gpu1_max_temp_c", "GPU 1", "#eb6834"),
                        ("dimm_max_c", "DDR5 DIMM", "#1baf7a")):
        ys = [r.get(k) for r in rows]
        if any(ys):
            ax.plot(caps, ys, color=col, linewidth=2, marker="o", markersize=3.5, zorder=3)
            ax.annotate(f"{lab}  {ys[-1]:.0f}°", (caps[-1], ys[-1]), xytext=(5, 0),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
    for ax in axes[1]:
        ax.set_xlabel("power cap per card (W)", color=INK2, fontsize=8.5)
        ax.set_xticks(caps[::2])
    axes[0][0].annotate(f"chosen {chosen} W", (chosen, axes[0][0].get_ylim()[1]), xytext=(4, -2),
                        textcoords="offset points", color=MUTED, fontsize=7.5, va="top")
    fig.suptitle(title, x=0.07, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.savefig(out_png, facecolor=SURFACE)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--pick", type=int, default=None)
    ap.add_argument("--title", default="RTX 3090 ×2 power-cap sweep, 200–400 W per card, gpu-unlock on")
    a = ap.parse_args()
    serve, tensor, samp, dimm = load(a.sweep_dir)
    rows = summarize(serve, tensor, samp, dimm)
    chosen = a.pick or pick(rows)
    cols = ["cap_w", "decode_1K", "decode_32K", "decode_128K", "ttft_128K_ms", "fp16_tflops_mean",
            "bf16_tflops_mean", "gpu0_mean_sm_mhz", "gpu0_max_temp_c", "gpu1_max_temp_c", "dimm_max_c",
            "gpu0_mean_power_w"]
    print("  ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):>14}" for c in cols))
    print("chosen cap:", chosen)
    json.dump({"chosen_cap_w": chosen, "rule": pick.__doc__.strip(), "rows": rows},
              open(os.path.join(a.sweep_dir, "summary.json"), "w"), indent=1)
    chart(rows, os.path.join(a.sweep_dir, "power-sweep.png"), chosen, a.title)


if __name__ == "__main__":
    main()
