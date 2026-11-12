#!/usr/bin/env python3
"""Generate context-length vs performance charts for each model.

Reads benchmark data from benchmarks/{model}/results.json and produces PNG charts.
All context charts share a unified **256K x-axis** for direct comparison — every
preset in our fleet now serves at 256K (or model-card max) per the 2026-05-31
preset cleanup. Existing results.json files were measured at the prior 16-32K
throughput-tuned defaults; re-benchmark sweeps at 256K are queued.
"""
import os
import glob
import json
import re
import textwrap
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

# --- Fleet styling (shared with the R9700 stack's generate_charts.py) ---
# A fleet panel holds up to ten curves, so a model is identified by two channels that are both
# unique inside its class: a FLEET_PALETTE colour and a marker shape. Marker fill carries the
# class (filled = MoE, hollow = dense) and the dash carries the quantization (solid = FP8,
# dashed = 4-bit; every preset here is AWQ int4, so every curve is dashed and reads the same as
# the 4-bit curves on the R9700 charts). A model both stacks serve keeps the same colour + marker.
FLEET_PALETTE = {
    "blue": "#4ea1ff", "orange": "#ff9a3c", "green": "#3ddc84", "red": "#ff5c5c",
    "violet": "#b48cff", "yellow": "#ffe45e", "cyan": "#3fd6e0", "pink": "#ff79c6",
    "white": "#f0f6fc", "tan": "#b89b7a",
}
MODEL_CLASSES = {"moe": "MoE", "dense": "Dense"}
INK = "#c9d1d9"


def _model(preset, label, cls, color, marker, quant="4bit"):
    return {"preset": preset, "label": label, "cls": cls, "quant": quant,
            "color": FLEET_PALETTE[color], "marker": marker}


MODELS = {
    # slug = benchmarks/<slug>/results.json (context sweep, bench_long_context.py);
    # preset = the launch.sh name, which is also how the throughput tripwire files
    # its runs (benchmarks/regression/<preset>-<date>.json) — the decode bar chart
    # takes whichever of the two is newer. Keep in sync with launch.sh presets.
    "qwen3.6-35b-a3b":  _model("qwen36",          "Qwen3.6-35B-A3B AWQ (MoE+DeltaNet)",        "moe",   "white",  "X"),
    "qwen3.6-ream":     _model("qwen36-ream",     "Qwen3.6-REAM-A3B AWQ (MoE+DeltaNet)",       "moe",   "blue",   "<"),
    "qwen3.5-28b-moe":  _model("qwen35-moe",      "Qwen3.5-28B-A3B REAP AWQ (MoE+DeltaNet)",   "moe",   "violet", "P"),
    "coder-30b-awq":    _model("coder-30b",       "Coder-30B AWQ (MoE)",                       "moe",   "green",  "^"),
    "coder-reap-25b":   _model("coder-reap-25b",  "Coder-REAP-25B AWQ (MoE)",                  "moe",   "pink",   "s"),
    "qwen3-30b-ream":   _model("qwen3-ream",      "Qwen3-30B-REAM AWQ (MoE)",                  "moe",   "orange", "*"),
    "flint":            _model("flint",           "Flint 40B-A2.5B AWQ (MoE + hybrid SWA + sinks)", "moe", "tan", ">"),
    "nemotron3-omni":   _model("nemotron3-omni",  "Nemotron-3-Nano-Omni AWQ (Mamba2 hybrid MoE)", "moe", "cyan",  "o"),
    "gemma4-26b-awq":   _model("gemma4",          "Gemma 4 26B AWQ (MoE)",                     "moe",   "yellow", "D"),
    "gemma4-21b-reap":  _model("gemma4-21b-reap", "Gemma 4 21B REAP AWQ (MoE)",                "moe",   "red",    "v"),
    "qwen38":           _model("qwen38",          "Qwen3.8-27B AWQ (Dense DeltaNet VL)",       "dense", "blue",   "o"),
    "qwen3.6-27b":      _model("qwen36-dense",    "Qwen3.6-27B AWQ (Dense)",                   "dense", "orange", "P"),
    "qwen35-27b-awq":   _model("qwen35-dense",    "Qwen3.5-27B AWQ (DeltaNet)",                "dense", "white",  "<"),
    "devstral-24b-awq": _model("devstral",        "Devstral-24B AWQ",                          "dense", "yellow", "^"),
    "gemma4-31b":       _model("gemma4-31b",      "Gemma 4 31B AWQ (Dense)",                   "dense", "tan",    "X"),
    "gemma4-12b":       _model("gemma4-12b",      "Gemma 4 12B AWQ (omni)",                    "dense", "cyan",   "D"),
    "qwen3-vl-32b":     _model("qwen3-vl-32b",    "Qwen3-VL-32B AWQ (Dense VL)",               "dense", "violet", "*"),
}


def latest_regression_run(preset):
    """Newest benchmarks/regression/<preset>-<date>.json (the throughput
    tripwire's depth-verified 1K / 32K / deep sweep), or None."""
    # Anchor on the date suffix: `qwen36-*.json` would also match
    # `qwen36-dense-2026-08-30.json`, and from 2026-08-30 the dense sweep
    # (05:03) was newer than qwen36's own (04:56) — the decode bar chart showed
    # the 35B-A3B MoE at the dense 27B's 70/48 tok/s instead of 215/122.
    pat = re.compile(rf"^{re.escape(preset)}-\d{{4}}-\d{{2}}-\d{{2}}\.json$")
    runs = [p for p in glob.glob(os.path.join(BENCH_DIR, "regression", f"{preset}-*.json"))
            if pat.match(os.path.basename(p))]
    best = None
    for path in runs:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if d.get("context_sweep") and (best is None or str(d.get("timestamp", "")) > str(best.get("timestamp", ""))):
            best = d
    return best


def newest_decode_sweep(key, meta):
    """The most recent depth-verified sweep for a model: its results.json or a
    later tripwire run. Anything before 2026-05 is the old 16-32K-capped
    instrument and would read as a fake KV cap — dropped."""
    cands = []
    rpath = os.path.join(BENCH_DIR, key, "results.json")
    if os.path.exists(rpath):
        cands.append(load_results(key))
    reg = latest_regression_run(meta["preset"])
    if reg:
        cands.append(reg)
    cands = [c for c in cands if str(c.get("timestamp", "")) >= "2026-05"]
    return max(cands, key=lambda c: str(c.get("timestamp", ""))) if cands else None

# Unified x-axis: 128 to 256K (matches R9700's chart format for cross-stack comparison).
# Our presets serve at 256K natively; benchmark data sweeps will be re-run at this range.
UNIFIED_XLIM = (800, 300_000)
UNIFIED_XTICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Standard concurrency levels for bar charts
STD_CONC = [1, 2, 4, 8, 16, 32]

TARGET_256K = 262_144  # the single-user target: 256 * 1024 tokens

# Real KV-pool cap = max_total_num_tokens from the serve log, read straight out
# of each results.json (the bench records it). This is authoritative and current
# by construction — NEVER hardcode it (the old hardcoded dict went stale: it
# still said gemma4-26b=118K after the SWA-ratio sprint took it to 652K, which
# silently truncated its 128K–256K sweep points). FALLBACK_CAP only covers the
# pre-2026-06 results.json files that predate the max_total_num_tokens field.
FALLBACK_CAP = {
    "coder-30b-awq":  900_000,   # A3B MoE, light KV — old-format file, no field
    "qwen35-27b-awq":  32_000,   # replicated-DeltaNet 27B — context-limited
}


def real_cap(model_key, results):
    """Authoritative KV pool size for the model, from the serve log."""
    mt = results.get("max_total_num_tokens")
    if mt:
        return int(mt)
    return FALLBACK_CAP.get(model_key, 10 ** 12)


def reaches_256k(model_key, results, deep_ctx=None):
    """True only when the pool holds 256K AND the sweep's deepest honest point sits
    there — a preset served at a smaller window (qwen35-dense: CTX=32768 over a
    610K pool) has a 256K-sized pool but never reaches 256K, and must draw hatched."""
    if real_cap(model_key, results) < TARGET_256K:
        return False
    return deep_ctx is None or deep_ctx >= 0.95 * TARGET_256K


def honest_sweep(model_key, results):
    """context_sweep filtered to points the model's real KV pool can actually
    hold, with a valid tok/s — drops the over-cap bench artifacts (a prompt that
    can't fit the pool never decodes there, so its logged tok/s is garbage)."""
    cap = real_cap(model_key, results)
    # Floor at 1K: sub-1K points are noise on a 256K x-axis and add nothing.
    return [p for p in (results.get("context_sweep") or [])
            if "error" not in p and (p.get("tok_per_sec") or 0) > 0
            and 1024 <= p.get("context", 0) <= cap]


def _tag_to_slug(tag):
    """Classify a tool-use receipt's internal `tag` to a chart slug. Order
    matters (qwen36-ream / qwen36-dense before bare qwen36; gemma4-12b/31b
    before bare gemma4)."""
    t = tag.lower()
    if t.startswith("qwen36-ream"):  return "qwen3.6-ream"
    if t.startswith("qwen36-dense"): return "qwen3.6-27b"
    if t.startswith("qwen36"):       return "qwen3.6-35b-a3b"
    if t.startswith("qwen3-ream"):   return "qwen3-30b-ream"
    if t.startswith("gemma4-12b"):   return "gemma4-12b"
    if t.startswith("gemma4-31b"):   return "gemma4-31b"
    if t.startswith("gemma4-21b"):   return "gemma4-21b-reap"
    if t.startswith("gemma4"):       return "gemma4-26b-awq"   # incl. gemma4-swa*, gemma4-fp8e5
    if t.startswith("devstral"):     return "devstral-24b-awq"
    return None


def _build_verified_depths():
    """Scan EVERY tool-use receipt under benchmarks/ (canonical quality/ +
    sprint dirs) and record, per slug, the deepest TRUE-token length with a
    correct tool call. Self-maintaining: new probe runs are picked up by their
    internal `tag`, so the metric never goes stale against a hardcoded list.

    'Verified-retrieval depth' is a STRICTER metric than KV pool size — a pool
    can hold tokens the model can't actually retrieve from."""
    # per slug: deepest correct + deepest ATTEMPTED (to know if it was deep-probed
    # at all — a model probed only to 147K that passed has an UNKNOWN ceiling, not
    # a 147K one; don't draw a false ceiling for it).
    best_ok, attempted = {}, {}
    pat = os.path.join(BENCH_DIR, "**", "*tooluse*.json")
    for path in glob.glob(pat, recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        slug = _tag_to_slug(str(d.get("tag", "")))
        if not slug:
            continue
        for r in d.get("results", []):
            tok = r.get("actual_prompt_tokens") or 0
            if tok > attempted.get(slug, 0):
                attempted[slug] = int(tok)
            if r.get("correct_action") and tok > best_ok.get(slug, 0):
                best_ok[slug] = int(tok)
    # gate: only report a depth where the probe actually reached deep (≥230K true);
    # else None = "not deep-probed", so the chart omits the bar instead of lying.
    DEEP_MIN = 230_000
    return {s: best_ok.get(s, 0) for s in attempted if attempted[s] >= DEEP_MIN}


_VERIFIED_DEPTHS = None


def verified_depth(model_key):
    global _VERIFIED_DEPTHS
    if _VERIFIED_DEPTHS is None:
        _VERIFIED_DEPTHS = _build_verified_depths()
    return _VERIFIED_DEPTHS.get(model_key, 0)


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
    sweep = honest_sweep(model_key, results)
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


def fleet_groups(members):
    """[(title, [member, ...])] in MODEL_CLASSES order, empty groups dropped; a member is any
    tuple or dict-row whose meta carries `cls`."""
    meta_of = lambda m: m["meta"] if isinstance(m, dict) else m[1]
    groups = [(title, [m for m in members if meta_of(m)["cls"] == value])
              for value, title in MODEL_CLASSES.items()]
    return [(title, group) for title, group in groups if group]


def fleet_line_style(meta):
    """ax.plot keywords of one fleet curve (see the fleet styling note above MODELS)."""
    panel = plt.rcParams["axes.facecolor"]
    dense = meta["cls"] == "dense"
    return {
        "color": meta["color"],
        "linewidth": 2.0,
        "linestyle": "-" if meta["quant"] == "fp8" else (0, (5, 2)),
        "marker": meta["marker"],
        "markersize": 9.5 if meta["marker"] in ("*", "P", "X") else 7.5,
        "markerfacecolor": panel if dense else meta["color"],
        "markeredgecolor": meta["color"] if dense else panel,
        "markeredgewidth": 1.6 if dense else 0.7,
        "zorder": 5,
    }


def _fleet_style_key(fig, quants):
    """Figure legend for the channels that are not a model's identity (only the ones drawn)."""
    from matplotlib.lines import Line2D
    panel = plt.rcParams["axes.facecolor"]
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7.5, markerfacecolor=INK,
               markeredgecolor=panel, label="filled marker: MoE"),
        Line2D([], [], linestyle="none", marker="o", markersize=7.5, markerfacecolor=panel,
               markeredgecolor=INK, markeredgewidth=1.6, label="hollow marker: dense"),
    ]
    if "fp8" in quants:
        handles.append(Line2D([], [], color=INK, linewidth=2.0, linestyle="-", label="solid line: FP8"))
    if "4bit" in quants:
        handles.append(Line2D([], [], color=INK, linewidth=2.0, linestyle=(0, (5, 2)),
                              label="dashed line: 4-bit (AWQ int4)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9.5,
               frameon=False, handlelength=3.2)


def make_combined_context_chart(all_data):
    """All models on one context chart, unified 256K x-axis, one panel per class (MoE / Dense)
    on shared axes — the R9700 fleet layout. A panel's legend lists its curves in the order they
    leave the left edge. Sweeps from the current instrument only (pre-2026-05 results.json files
    are the old 16-32K-capped runs), filtered to points the real KV pool holds."""
    members = []
    for key, (meta, results) in all_data.items():
        if str(results.get("timestamp", "")) < "2026-05":
            continue
        sweep = honest_sweep(key, results)
        if sweep:
            members.append((key, meta, [p["context"] for p in sweep], [p["tok_per_sec"] for p in sweep]))
    panels = [(title, sorted(group, key=lambda m: m[3][0], reverse=True))
              for title, group in fleet_groups(members)]
    if not panels:
        print("  SKIP all_models_context.png (no fleet data)")
        return
    tallest = max(len(curves) for _, curves in panels)
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, sharey=True, squeeze=False, layout="constrained",
        figsize=(12, max(3.8, 0.27 * tallest + 1.1) * len(panels) + 1.0))
    for ax, (title, curves) in zip(axes[:, 0], panels):
        for _, meta, xs, ys in curves:
            ax.plot(xs, ys, label=meta["label"], **fleet_line_style(meta))
        ax.set_xscale("log", base=2)
        ax.set_xlim(*UNIFIED_XLIM)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ctx))
        ax.xaxis.set_major_locator(ticker.FixedLocator(UNIFIED_XTICKS))
        ax.set_ylabel("tok/s (single user)")
        ax.grid(True, axis="both", linestyle="--")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{title} · {len(curves)} models", loc="left", fontsize=11.5, fontweight="bold")
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=9,
                           handlelength=3.4, title="fastest at the left edge first", title_fontsize=8,
                           framealpha=0.5, edgecolor="#30363d", facecolor="#161b22")
        legend.get_title().set_color("#8b949e")
    axes[-1, 0].set_xlabel("Context Length")
    fig.suptitle("All Models — Context Length vs Decode Speed (single user, TP=2)",
                 fontsize=13, fontweight="bold")
    _fleet_style_key(fig, {m[1]["quant"] for m in members})
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
    """Plain AWQ vs AWQ + draft (single-user decode tok/s) for every spec-validated
    (model, depth) row of benchmarks/specdec-comparison.json, as horizontal paired
    bars in the fleet layout. The speed-up is the label; the provenance sentence
    (stack, date, prompt source) is the wrapped footnote, not a title."""
    from matplotlib.patches import Patch
    path_in = os.path.join(BENCH_DIR, "specdec-comparison.json")
    if not os.path.exists(path_in):
        print(f"  (no {path_in} — skipping spec-decode chart)")
        return
    with open(path_in) as f:
        data = json.load(f)
    models = data["models"]
    if not models:
        return
    PLAIN, SPEC = FLEET_PALETTE["blue"], FLEET_PALETTE["green"]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(models) + 2.4))
    top = max(max(m["spec_toks"], m["awq_toks"]) for m in models)
    h, pad, labels = 0.37, top * 0.008, []
    for y, m in enumerate(models):
        ax.barh(y - h / 2, m["awq_toks"], h, color=PLAIN, alpha=0.95, zorder=5)
        ax.text(m["awq_toks"] + pad, y - h / 2, f'{m["awq_toks"]:.0f}', va="center", fontsize=9, color=INK)
        ax.barh(y + h / 2, m["spec_toks"], h, color=SPEC, alpha=0.95, zorder=5)
        ax.text(m["spec_toks"] + pad, y + h / 2,
                f'{m["spec_toks"]:.0f}  ·  {m["speedup_x"]:.2f}×  ·  {m["spec_draft"]}',
                va="center", fontsize=9, color=INK, fontweight="bold")
        ctx = m.get("ctx_label") or f'ctx {m["ctx_k"]}K'
        labels.append(f'{m["name"]} ({m["kind"]})\n{ctx}' + (f' · {m["measured"]}' if m.get("measured") else ""))
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(len(models) - 0.4, -0.6)
    ax.set_xlim(0, top * 1.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("tok/s (single user, M=1)")
    ax.set_title("Speculative decoding — plain AWQ vs + draft, single-user decode (TP=2)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[Patch(facecolor=PLAIN, label="AWQ int4, no draft"),
                       Patch(facecolor=SPEC, label="+ draft (speculative decoding)")],
              loc="lower right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22", fontsize=9)
    ax.grid(True, axis="x", linestyle="--")
    note = data.get("chart_note") or ("natural-text prompts, greedy, cache flushed per point, plain arm on the same "
                                      "boot; each row is labelled with the stack and date it was measured on — "
                                      "provenance and receipts in benchmarks/specdec-comparison.json")
    ax.annotate(textwrap.fill(note, 150), xy=(0, 0), xycoords="axes fraction", xytext=(0, -40),
                textcoords="offset points", va="top", fontsize=7.5, color="#8b949e")
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "specdec_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def make_decode_bar_chart(all_data):
    """Per-model single-user decode tok/s — peak (short ctx) vs deep (256K where
    the KV pool reaches it, else the model's real max ctx), in the R9700 fleet
    layout: horizontal bars, one section per class (MoE / Dense), fastest first
    inside each. The deep bar is hatched for models whose KV pool or served
    window caps below 256K, so they show their honest limit instead of being
    read as 256K-depth throughput."""
    from matplotlib.patches import Patch
    rows = []
    for key, meta in MODELS.items():
        results = newest_decode_sweep(key, meta)
        if not results:
            continue
        sweep = honest_sweep(key, results)
        if len(sweep) < 2:
            continue
        peak = max(sweep, key=lambda p: p["tok_per_sec"])
        deep = sweep[-1]
        rows.append({"meta": meta, "peak": peak["tok_per_sec"], "deep": deep["tok_per_sec"],
                     "deep_ctx": deep["context"], "reaches": reaches_256k(key, results, deep["context"]),
                     "date": str(results.get("timestamp", ""))[:10]})
    if not rows:
        print("  SKIP decode bar chart (no fresh fleet data)")
        return
    sections = [(title, sorted(group, key=lambda r: r["peak"], reverse=True))
                for title, group in fleet_groups(rows)]

    panel = plt.rcParams["axes.facecolor"]
    fig, ax = plt.subplots(figsize=(11, 0.46 * len(rows) + 0.45 * len(sections) + 2.0))
    top = max(max(r["peak"], r["deep"]) for r in rows)
    h, pad, y, ticks, labels = 0.37, top * 0.008, 0.0, [], []
    for title, section in sections:
        ax.axhline(y, color="#484f58", linewidth=1.0, zorder=1)
        ax.text(-0.012, y, f"{title} ({len(section)})", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=11.5, fontweight="bold", color="#f0f6fc")
        y += 0.85
        for r in section:
            c = r["meta"]["color"]
            ax.barh(y - h / 2, r["peak"], h, color=c, alpha=0.45, zorder=5)
            ax.text(r["peak"] + pad, y - h / 2, f'{r["peak"]:.0f}', va="center",
                    fontsize=9, color=INK, fontweight="bold")
            ax.barh(y + h / 2, r["deep"], h, color=c, alpha=0.95, zorder=5,
                    hatch=None if r["reaches"] else "////", edgecolor=panel, linewidth=0.6)
            ctx_lbl = "256K" if r["reaches"] else f'{r["deep_ctx"] // 1024}K'
            ax.text(r["deep"] + pad, y + h / 2, f'{r["deep"]:.0f} @{ctx_lbl}',
                    va="center", fontsize=9, color=INK)
            ticks.append(y)
            labels.append(r["meta"]["label"])
            y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(y - 0.35, -0.75)   # inverted: the first row is on top
    ax.set_xlim(0, top * 1.16)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("tok/s (single user, M=1)")
    ax.set_title("Single-user decode — short input vs 256K or the model's own cap (TP=2)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[
        Patch(facecolor="#8b949e", alpha=0.42, label="fastest point of the sweep (short input)"),
        Patch(facecolor="#8b949e", alpha=0.95, label="deep: true 256K window, server-verified depth"),
        Patch(facecolor="#8b949e", alpha=0.95, hatch="////", edgecolor="#0d1117",
              label="at the model's own cap (< 256K)"),
    ], loc="lower right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22", fontsize=9)
    ax.grid(True, axis="x", linestyle="--")
    dates = sorted({r["date"] for r in rows})
    note = ("newest depth-verified sweep per model (results.json or a later tripwire run; streaming "
            "TPOT, decode-only, server-verified actual tokens), every preset as shipped at the 300 W cap; "
            f"measured {dates[0]}" + (f" .. {dates[-1]}" if len(dates) > 1 else ""))
    ax.annotate(textwrap.fill(note, 150), xy=(0, 0), xycoords="axes fraction", xytext=(0, -40),
                textcoords="offset points", va="top", fontsize=7.5, color="#8b949e")
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "all_models_decode.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")



# Scaffold display order for the bake-off chart. The harness was pruned to
# plain opencode on 2026-10-22 (qwen38 five-scaffold bake-off = statistical
# tie; benchmarks/quality/qwen38-v4-cycle-close-2026-10-22.md): landed cells
# of every scaffold still plot (the qwen38 bake-off row), but only the ACTIVE
# lane gets "pending" placeholders on the presets still in the queue.
# claw-code (retired 2026-08-31) is not plotted — its cell JSONs stay as receipts.
BAKEOFF_SCAFFOLDS = [
    ("opencode",         "#58a6ff", "opencode"),
    ("opencode-dcp",     "#1f6feb", "opencode + DCP"),
    ("little-coder",     "#3fb950", "little-coder"),
    ("little-coder-rtk", "#2ea043", "little-coder + RTK"),
    ("prime",            "#d2a8ff", "prime"),
    ("dcode",            "#f0883e", "dcode"),
    ("omp",              "#56d4dd", "oh-my-pi"),
]


def make_bakeoff_chart():
    """SWE-bench Lite resolve rate per preset × scaffold, from the per-cell
    JSONs `benchmarks/quality/bakeoff-<preset>-<scaffold>.json` that
    `evals/swebench/aggregate_bakeoff.py` writes — the R9700 layout: only cells
    with data are drawn, bars of a preset sit side by side in the fixed scaffold
    colours, a cell under 300 predictions is hatched with its (n) under the value.
    Superseded cells (`"superseded"` set; the pre-isolation receipts) and retired
    scaffolds are not plotted."""
    from matplotlib.patches import Patch
    roster = [sc for sc, _, _ in BAKEOFF_SCAFFOLDS]
    cells = {}
    for path in glob.glob(os.path.join(BENCH_DIR, "quality", "bakeoff-*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not d.get("preset") or not d.get("scaffold") or d.get("resolved") is None:
            continue
        if d.get("superseded") or d["scaffold"] not in roster:
            continue
        cells[(d["preset"], d["scaffold"])] = d
    if not cells:
        print("  SKIP bake-off chart (no cells)")
        return
    best = lambda p: max(c["resolve_rate_pct"] for (pp, _), c in cells.items() if pp == p)
    presets = sorted({p for p, _ in cells}, key=best, reverse=True)
    used = [t for t in BAKEOFF_SCAFFOLDS if any(sc == t[0] for _, sc in cells)]
    w = 0.17
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(presets)), 6))
    top = max(c["resolve_rate_pct"] for c in cells.values()) * 1.32
    partial = False
    for i, p in enumerate(presets):
        mine = [(sc, color) for sc, color, _ in used if (p, sc) in cells]
        for j, (sc, color) in enumerate(mine):
            c = cells[(p, sc)]
            xx = i + (j - (len(mine) - 1) / 2) * w
            n = c.get("total_predictions", 0)
            full = n >= 300
            partial |= not full
            rate = c["resolve_rate_pct"]
            ax.bar(xx, rate, w * 0.92, color=color, alpha=0.95 if full else 0.55,
                   hatch=None if full else "////", edgecolor="#0d1117", linewidth=0.6, zorder=5)
            ax.text(xx, rate + top * 0.012, f"{rate:.1f}" + ("" if full else f"\n({n})"),
                    ha="center", va="bottom", fontsize=7.5, color=INK, fontweight="bold")
    handles = [Patch(facecolor=color, label=label) for _, color, label in used]
    if partial:
        handles.append(Patch(facecolor="#8b949e", alpha=0.55, hatch="////", edgecolor="#0d1117",
                             label="partial cell (< 300 instances)"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.5, edgecolor="#30363d",
              facecolor="#161b22", fontsize=9)
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels(presets, fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(presets) - 0.4)
    ax.set_ylabel("SWE-bench Lite resolved (%)")
    ax.set_ylim(0, top)
    ax.set_title("SWE-bench Lite (300 instances) — resolve rate per preset × scaffold, 256K single-user",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--")
    notes = ["isolated Docker harness: network=none, served 262K window, 32K output budget, template-default "
             "thinking; official swebench scorer, one attempt per instance (resolved / 300)",
             "oh-my-pi is the default lane; the other scaffolds were rolled on qwen38 only and tied within noise"]
    width = int(fig.get_figwidth() * 17)
    ax.text(0.0, -0.1, "\n".join(textwrap.fill("• " + t, width, subsequent_indent="  ") for t in notes),
            transform=ax.transAxes, va="top", fontsize=7.5, color="#8b949e")
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "bakeoff_swebench_lite.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def _fmt_tok(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1000:.0f}K"


def make_kv_capacity_chart(all_data):
    """NEW METRIC — single-user KV reach. Two bars per model: the real KV pool
    (max_total_num_tokens, what the cache can physically hold) and the verified-
    retrieval depth (deepest TRUE-token prompt with a correct tool call). The
    256K target is a reference line. Capacity ≥ target + a retrieval bar that
    also clears it = a genuine 256K single-user model; a tall pool with a short
    retrieval bar = holds the tokens but can't use them. Log-x (pools span
    24K–2.4M). Fleet counterpart to the decode bar chart."""
    rows = []
    for key, (meta, results) in all_data.items():
        cap = real_cap(key, results)
        if not results.get("max_total_num_tokens"):   # old-format file, no real pool — skip
            continue
        rows.append({"meta": meta, "cap": cap, "depth": verified_depth(key),
                     "reaches": cap >= TARGET_256K})
    if not rows:
        print("  SKIP kv-capacity chart (no pool data)")
        return
    rows.sort(key=lambda r: r["cap"])
    y = np.arange(len(rows)); h = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    GREEN, AMBER = "#3fb950", "#e3b341"
    for i, r in enumerate(rows):
        cap_c = GREEN if r["reaches"] else AMBER
        ax.barh(y[i] + h / 2, r["cap"], h, color=cap_c, alpha=0.85, zorder=5)
        ax.text(r["cap"] * 1.04, y[i] + h / 2, _fmt_tok(r["cap"]), va="center",
                fontsize=9, color=cap_c, fontweight="bold")
        if r["depth"] > 0:
            dep_c = GREEN if r["depth"] >= TARGET_256K * 0.98 else AMBER
            ax.barh(y[i] - h / 2, r["depth"], h, color=dep_c, alpha=0.45, zorder=5,
                    hatch="////", edgecolor="#0d1117", linewidth=0.5)
            ax.text(r["depth"] * 1.04, y[i] - h / 2, _fmt_tok(r["depth"]), va="center",
                    fontsize=8, color=dep_c, fontweight="bold")
    ax.axvline(TARGET_256K, color="#f85149", linestyle="--", linewidth=1.6, zorder=8)
    ax.text(TARGET_256K, len(rows) - 0.3, " 256K target", color="#f85149",
            fontsize=9, fontweight="bold", va="top")
    ax.set_xscale("log", base=10)
    ax.set_xlim(8_000, 4_000_000)
    ax.set_yticks(y)
    ax.set_yticklabels([r["meta"]["label"].replace(" AWQ ", " ").replace(" AWQ", "") for r in rows], fontsize=9)
    ax.set_xlabel("tokens (log scale)")
    ax.set_title("Single-user KV reach — pool capacity vs verified-retrieval depth",
                 fontsize=13, fontweight="bold", pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#8b949e", alpha=0.85, label="KV pool (max_total_num_tokens)"),
        Patch(facecolor="#8b949e", alpha=0.45, hatch="////", edgecolor="#0d1117",
              label="verified-correct tool-call depth (true tokens)"),
        Patch(facecolor=GREEN, label="≥ 256K target"),
        Patch(facecolor=AMBER, label="below target"),
    ], loc="lower right", framealpha=0.6, edgecolor="#30363d", facecolor="#161b22", fontsize=8.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "all_models_kv_capacity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def make_power_cap_chart(all_data):
    """300 W vs 400 W per-card cap, same boot, same 10-point sweep (2026-10-25 refresh,
    gpu-unlock on, CPU boost off). Three small multiples sharing the model axis, one
    y-scale each: decode at 1K, decode at the deepest honest point, and TTFT at that
    deepest point (prefill is the compute-bound half that the extra tensor rate feeds).
    Reads benchmarks/<slug>/results-400w.json next to results.json; models without a
    400 W sweep are skipped. The 300 W side is results-300w.json when present — the
    same-boot companion of the 400 W sweep — so a later preset change that re-sweeps
    results.json (flint on fp8_e5m2 KV, 2026-10-31) does not masquerade as a power
    effect; otherwise results.json."""
    from matplotlib.patches import Patch
    rows = []
    for key, (meta, r300) in all_data.items():
        p400 = os.path.join(BENCH_DIR, key, "results-400w.json")
        if not os.path.exists(p400):
            continue
        r400 = json.load(open(p400))
        p300 = os.path.join(BENCH_DIR, key, "results-300w.json")
        if os.path.exists(p300):
            r300 = json.load(open(p300))
        s300, s400 = honest_sweep(key, r300), honest_sweep(key, r400)
        if len(s300) < 2 or len(s400) < 2:
            continue
        by400 = {p["context"]: p for p in s400}
        deep = s300[-1]
        mid = max((p for p in s300 if p["context"] <= 32768), key=lambda p: p["context"], default=None)
        if mid is None or deep["context"] not in by400 or s300[0]["context"] not in by400 or mid["context"] not in by400:
            continue
        rows.append({
            "meta": meta,
            "d1k": (s300[0]["tok_per_sec"], by400[s300[0]["context"]]["tok_per_sec"]),
            "dmid": (mid["tok_per_sec"], by400[mid["context"]]["tok_per_sec"]),
            "ddeep": (deep["tok_per_sec"], by400[deep["context"]]["tok_per_sec"]),
            "deep_ctx": deep["context"], "reaches": reaches_256k(key, r300, deep["context"]),
        })
    if not rows:
        print("  SKIP power-cap chart (no results-400w.json)")
        return
    sections = [(title, sorted(group, key=lambda r: r["d1k"][0], reverse=True))
                for title, group in fleet_groups(rows)]
    panels = [("d1k", "Decode at 1K"), ("dmid", "Decode at 32K"),
              ("ddeep", "Decode at 256K (or the KV cap)")]
    fig, axes = plt.subplots(1, 3, sharey=True, layout="constrained",
                             figsize=(14, 0.5 * len(rows) + 0.45 * len(sections) + 2.2))
    h = 0.37
    for ax, (k, title) in zip(axes, panels):
        top = max(max(r[k]) for r in rows)
        y, ticks, labels = 0.0, [], []
        for sec_title, section in sections:
            ax.axhline(y, color="#484f58", linewidth=1.0, zorder=1)
            if ax is axes[0]:
                ax.text(-0.02, y, f"{sec_title} ({len(section)})", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=11.5, fontweight="bold", color="#f0f6fc")
            y += 0.85
            for r in section:
                c = r["meta"]["color"]; lo, hi = r[k]
                ax.barh(y - h / 2, lo, h, color=c, alpha=0.45, zorder=5)
                ax.barh(y + h / 2, hi, h, color=c, alpha=0.95, zorder=5)
                delta = (hi / lo - 1) * 100 if lo else 0
                ax.text(max(lo, hi) + top * 0.015, y, f"{lo:.0f} → {hi:.0f}  {delta:+.0f} %",
                        va="center", fontsize=8.5, color=INK)
                ticks.append(y); labels.append(r["meta"]["label"])
                y += 1.0
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9.5, color=INK)
        ax.set_ylim(y - 0.35, -0.75)
        ax.set_xlim(0, top * 1.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("tok/s (single user, M=1)")
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
        ax.grid(True, axis="x", linestyle="--")
    fig.legend(handles=[Patch(facecolor="#8b949e", alpha=0.45, label="300 W per card (shipped cap)"),
                        Patch(facecolor="#8b949e", alpha=0.95, label="400 W per card (card maximum)")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Power cap 300 W vs 400 W — single-user decode per model (TP=2, same boot per model)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(BENCH_DIR, "all_models_power_300v400.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["bakeoff"]:          # regenerate only the bake-off chart
        make_bakeoff_chart(); sys.exit(0)
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
        make_kv_capacity_chart(all_data)
    print("Decode bars (results.json vs newer tripwire runs):")
    make_decode_bar_chart(all_data)
    print("Power cap 300 vs 400 W:")
    make_power_cap_chart(all_data)

    print("Spec-decode:")
    make_specdec_comparison_chart()

    print("SWE-bench Lite bake-off:")
    make_bakeoff_chart()

    print("\nDone!")
