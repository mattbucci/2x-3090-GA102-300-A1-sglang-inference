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
    "gemma4-21b-reap":     {"label": "Gemma 4 21B REAP AWQ (MoE, think)", "color": "#a371f7"},
    "gemma4-12b":          {"label": "Gemma 4 12B AWQ (omni, think)",     "color": "#8957e5"},
    "coder-30b-awq":       {"label": "Coder-30B AWQ (MoE)",              "color": "#f0883e"},
    "qwen35-27b-awq":      {"label": "Qwen3.5-27B AWQ",                  "color": "#e3b341"},
}

# Unified x-axis: 128 to 256K (matches R9700's chart format for cross-stack comparison).
# Our presets serve at 256K natively; benchmark data sweeps will be re-run at this range.
UNIFIED_XLIM = (1024, 300_000)
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


def reaches_256k(model_key, results):
    return real_cap(model_key, results) >= TARGET_256K


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


def make_combined_context_chart(all_data):
    """All models on one context chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, (meta, results) in all_data.items():
        sweep = honest_sweep(key, results)
        if not sweep:
            continue
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


def make_decode_bar_chart(all_data):
    """Per-model single-user decode tok/s — peak (short ctx) vs deep (256K where
    the KV pool reaches it, else the model's real max ctx). The deep bar is
    hatched for models whose KV pool caps below 256K, so gemma/devstral show
    their honest sub-256K limit instead of the over-cap bench artifacts. Fleet
    counterpart to R9700's per-model decode bar chart."""
    from matplotlib.patches import Patch
    rows = []
    for key, (meta, results) in all_data.items():
        # fresh v0512 fleet only — stale April sweeps stop at 16K (an old short
        # sweep, not a KV limit), which would read as a fake cap.
        if not str(results.get("timestamp", "")).startswith(("2026-05", "2026-06")):
            continue
        sweep = honest_sweep(key, results)
        if len(sweep) < 2:
            continue
        peak = max(sweep, key=lambda p: p["tok_per_sec"])
        deep = sweep[-1]
        rows.append({"meta": meta, "peak": peak["tok_per_sec"], "deep": deep["tok_per_sec"],
                     "deep_ctx": deep["context"], "reaches": reaches_256k(key, results)})
    if not rows:
        print("  SKIP decode bar chart (no fresh fleet data)")
        return
    rows.sort(key=lambda r: r["peak"], reverse=True)

    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, r in enumerate(rows):
        c = r["meta"]["color"]
        ax.bar(x[i] - w / 2, r["peak"], w, color=c, alpha=0.42, zorder=5)
        ax.text(x[i] - w / 2, r["peak"] + 1.5, f'{r["peak"]:.0f}', ha="center",
                fontsize=9, color=c, fontweight="bold")
        hatch = None if r["reaches"] else "////"
        ax.bar(x[i] + w / 2, r["deep"], w, color=c, alpha=0.95, zorder=5,
               hatch=hatch, edgecolor="#0d1117", linewidth=0.6)
        ctx_lbl = "256K" if r["reaches"] else f'{r["deep_ctx"] // 1024}K'
        ax.text(x[i] + w / 2, r["deep"] + 1.5, f'{r["deep"]:.0f}\n@{ctx_lbl}',
                ha="center", fontsize=8, color=c, fontweight="bold")
    labels = [r["meta"]["label"].replace(" AWQ ", "\n").replace(" AWQ", "") for r in rows]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("tok/s (single user, M=1)")
    ax.set_title("Single-user decode — peak vs 256K (or real KV cap)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[
        Patch(facecolor="#8b949e", alpha=0.42, label="peak (short ctx)"),
        Patch(facecolor="#8b949e", alpha=0.95, label="at true 256K KV pool"),
        Patch(facecolor="#8b949e", alpha=0.95, hatch="////", edgecolor="#0d1117",
              label="at KV cap (< 256K)"),
    ], loc="upper right", framealpha=0.5, edgecolor="#30363d", facecolor="#161b22", fontsize=9)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_ylim(bottom=0, top=max(r["peak"] for r in rows) * 1.18)
    fig.tight_layout()
    out = os.path.join(BENCH_DIR, "all_models_decode.png")
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
        make_decode_bar_chart(all_data)
        make_kv_capacity_chart(all_data)

    print("Spec-decode:")
    make_specdec_comparison_chart()

    print("\nDone!")
