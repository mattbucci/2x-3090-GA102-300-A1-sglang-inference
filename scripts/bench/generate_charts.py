#!/usr/bin/env python3
"""Generate context-length vs performance charts for each model.

Reads benchmark data from benchmarks/{model}/results.json and produces PNG charts.
All context charts share a unified **256K x-axis** for direct comparison — every
preset in our fleet now serves at 256K (or model-card max) per the 2026-05-31
preset cleanup. Existing results.json files were measured at the prior 16-32K
throughput-tuned defaults; re-benchmark sweeps at 256K are queued.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO, "benchmarks")

# --- Style ---
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
})

MODELS = {
    # slug = benchmarks/<slug>/results.json. Keep in sync with run_v0512_fleet_eval.sh.
    "qwen3.6-35b-a3b":     {"label": "Qwen3.6-35B-A3B AWQ (MoE, think)", "color": "#58a6ff"},
    "qwen3.6-ream":        {"label": "Qwen3.6-REAM-A3B AWQ (MoE, think)", "color": "#79c0ff"},
    "qwen3.6-27b":         {"label": "Qwen3.6-27B AWQ (dense, think)",    "color": "#1f6feb"},
    "qwen3-30b-ream":      {"label": "Qwen3-30B-REAM AWQ (MoE)",          "color": "#3fb950"},
    "devstral-24b-awq":    {"label": "Devstral-24B AWQ (dense)",          "color": "#56d364"},
    "gemma4-31b":          {"label": "Gemma 4 31B AWQ (dense, think)",    "color": "#d2a8ff"},
    "gemma4-26b-awq":      {"label": "Gemma 4 26B AWQ (MoE, think)",      "color": "#bc8cff"},
    "coder-30b-awq":       {"label": "Coder-30B AWQ (MoE)",              "color": "#f0883e"},
    "qwen35-27b-awq":      {"label": "Qwen3.5-27B AWQ",                  "color": "#e3b341"},
}

# Unified x-axis: 128 to 256K (matches R9700's chart format for cross-stack comparison).
# Our presets serve at 256K natively; benchmark data sweeps will be re-run at this range.
UNIFIED_XLIM = (96, 300_000)
UNIFIED_XTICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Standard concurrency levels for bar charts
STD_CONC = [1, 2, 4, 8, 16, 32]


def fmt_ctx(x, _):
    if x >= 1024:
        return f"{x / 1024:.0f}K"
    return f"{x:.0f}"


def load_results(model_key):
    path = os.path.join(BENCH_DIR, model_key, "results.json")
    with open(path) as f:
        return json.load(f)


def make_context_chart(model_key, meta, results, out_dir):
    """Single-user tok/s vs context length."""
    sweep = [p for p in results["context_sweep"] if "error" not in p and p.get("tok_per_sec", 0) > 0]
    if not sweep:
        print(f"  SKIP context chart (no valid data)")
        return
    ctx = [p["context"] for p in sweep]
    toks = [p["tok_per_sec"] for p in sweep]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ctx, toks, "o-", color=meta["color"], linewidth=2, markersize=5, zorder=5)
    ax.fill_between(ctx, toks, alpha=0.12, color=meta["color"])

    # Annotate peak
    peak_idx = int(np.argmax(toks))
    ax.annotate(f"{toks[peak_idx]:.1f}", (ctx[peak_idx], toks[peak_idx]),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=10, fontweight="bold", color=meta["color"])
    if peak_idx != len(toks) - 1:
        ax.annotate(f"{toks[-1]:.1f}", (ctx[-1], toks[-1]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color="#8b949e")

    ax.set_xscale("log", base=2)
    ax.set_xlim(*UNIFIED_XLIM)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("tok/s (single user)")
    ax.set_title(meta["label"], fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "context_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_concurrency_chart(model_key, meta, results, out_dir):
    """Total throughput vs concurrency."""
    sweep = results.get("throughput_sweep") or []
    if not sweep:
        print(f"  SKIP concurrency chart (single-user-only data)")
        return
    measured = {p["concurrency"]: p["tok_per_sec"] for p in sweep}

    conc_levels = sorted(set(STD_CONC) & set(measured.keys()))
    labels = [str(c) for c in conc_levels]
    values = [measured[c] for c in conc_levels]
    colors = [meta["color"]] * len(conc_levels)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, width=0.6, zorder=5)

    y_max = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(i, v + y_max * 0.02, f"{v:.0f}", ha="center", fontsize=10,
                fontweight="bold", color=meta["color"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Total tok/s")
    ax.set_title(f"{meta['label']} — Throughput Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_ylim(bottom=0, top=y_max * 1.15)

    fig.tight_layout()
    path = os.path.join(out_dir, "concurrency_vs_toks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_combined_context_chart(all_data):
    """All models on one context chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, (meta, results) in all_data.items():
        sweep = results["context_sweep"]
        ctx = [p["context"] for p in sweep]
        toks = [p["tok_per_sec"] for p in sweep]
        ax.plot(ctx, toks, "o-", color=meta["color"], linewidth=2, markersize=5,
                label=meta["label"], zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xlim(*UNIFIED_XLIM)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
    ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("tok/s (single user)")
    ax.set_title("All Models — Context Length vs Decode Speed", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.5,
              edgecolor="#30363d", facecolor="#161b22")

    fig.tight_layout()
    path = os.path.join(BENCH_DIR, "all_models_context.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_combined_concurrency_chart(all_data):
    """All models on one concurrency chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    plotted = 0
    for key, (meta, results) in all_data.items():
        sweep = results.get("throughput_sweep") or []
        if not sweep:
            continue
        conc = [p["concurrency"] for p in sweep]
        toks = [p["tok_per_sec"] for p in sweep]
        ax.plot(conc, toks, "o-", color=meta["color"], linewidth=2, markersize=5,
                label=meta["label"], zorder=5)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print("  SKIP combined concurrency chart (single-user-only fleet)")
        return

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Total tok/s")
    ax.set_title("All Models — Throughput Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="both", linestyle="--")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.5,
              edgecolor="#30363d", facecolor="#161b22")

    fig.tight_layout()
    path = os.path.join(BENCH_DIR, "all_models_concurrency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def make_specdec_comparison_chart():
    """AWQ baseline vs AWQ + spec-decode (single-user decode tok/s) across the
    spec-validated fleet. Data from benchmarks/specdec-comparison.json.

    Adapted from R9700's `make_fp8_comparison_chart` (commit a668c39, "+draft"
    series) but stripped to the AWQ + spec-decode bars only — FP8 doesn't
    accelerate on Ampere sm_86 and the relevant models don't fit at FP8 on 24GB.
    """
    path_in = os.path.join(BENCH_DIR, "specdec-comparison.json")
    if not os.path.exists(path_in):
        print(f"  (no {path_in} — skipping spec-decode chart)")
        return
    with open(path_in) as f:
        data = json.load(f)
    models = data["models"]
    if not models:
        return
    x = np.arange(len(models))
    w = 0.36
    AWQC, SPECC = "#58a6ff", "#3fb950"
    xlabels = [f'{m["name"]}\n{m["kind"]}  •  ctx {m["ctx_k"]}K' for m in models]

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    seen = set()
    def _lbl(key, text):
        if key in seen:
            return None
        seen.add(key); return text

    for i, m in enumerate(models):
        awq = m["awq_toks"]; spec = m["spec_toks"]
        ax.bar(x[i] - w / 2, awq, w, color=AWQC, zorder=5,
               label=_lbl("awq", "AWQ int4 (no spec)"))
        ax.text(x[i] - w / 2, awq + 2.0, f'{awq:.0f}',
                ha="center", fontsize=9, color=AWQC, fontweight="bold")
        ax.bar(x[i] + w / 2, spec, w, color=SPECC, zorder=5,
               label=_lbl("spec", "+ draft (spec-decode)"))
        ax.text(x[i] + w / 2, spec + 2.0,
                f'{spec:.0f}\n{m["spec_draft"]}\n{m["speedup_x"]:.2f}×',
                ha="center", fontsize=8, color=SPECC, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("tok/s (single user)")
    ax.set_title("Single-user decode — AWQ vs +draft (spec-decode)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper left", framealpha=0.5, edgecolor="#30363d",
              facecolor="#161b22", fontsize=9)
    ax.grid(True, axis="y", linestyle="--")
    y_top = max(m["spec_toks"] for m in models) * 1.30
    ax.set_ylim(bottom=0, top=y_top)

    fig.suptitle(f'{data["title"]}\n{data["subtitle"]}',
                 fontsize=12, fontweight="bold", y=1.02)
    if data.get("footnote"):
        fig.text(0.5, -0.04, data["footnote"], ha="center", fontsize=8,
                 color="#8b949e", style="italic")
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "specdec_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    print("Generating benchmark charts...\n")

    all_data = {}
    for key, meta in MODELS.items():
        out_dir = os.path.join(BENCH_DIR, key)
        if not os.path.exists(os.path.join(out_dir, "results.json")):
            print(f"{meta['label']}: no results.json, skipping")
            continue
        os.makedirs(out_dir, exist_ok=True)
        results = load_results(key)
        all_data[key] = (meta, results)
        print(f"{meta['label']}:")
        make_context_chart(key, meta, results, out_dir)
        make_concurrency_chart(key, meta, results, out_dir)
        print()

    if all_data:
        print("Combined:")
        make_combined_context_chart(all_data)
        make_combined_concurrency_chart(all_data)

    print("Spec-decode:")
    make_specdec_comparison_chart()

    print("\nDone!")
