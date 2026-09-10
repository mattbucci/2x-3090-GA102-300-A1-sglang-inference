# Quality evals — methodology and findings

The README keeps the three tables (static evals, 256K tool-use probe, 256K reasoning probe). This file holds the methodology, the footnotes, and the per-probe findings.

## Fleet integrity and static evals

**Fleet integrity: every shipped AWQ model is scale-integrity clean** (fleet-wide `check_awq_scales.py --base` audit — zero real zero-over-live defects; all flags were benign MoE dead-channel sparsity; receipt [`benchmarks/quality/fleet-integrity-audit-2026-05-31.json`](../benchmarks/quality/fleet-integrity-audit-2026-05-31.json)), and every capability (thinking/image/video/audio/tool as applicable) passes on the current stack — per-preset receipts `cap-*-v0517.json`.

Run with `scripts/eval/eval_quality.py` (or `eval_and_chart.py` / the fleet orchestrators): MMLU, HumanEval pass@1, [LAB-Bench](https://github.com/Future-House/LAB-Bench), Needle-in-Haystack (**1K → 250K**). Treat ±a few points as sampling noise. The **Needle column is the deepest length where all 3 depths (0.1 / 0.5 / 0.9) retrieve** and reflects each model's KV pool *at measurement time* (server-verified: 11 of 12 presets retrieve 3/3 at a true ~250,035 actual tokens; nemotron3-omni is 2/3 — the miss is depth-0.1 only, consistent with Mamba recurrent-state fade of oldest context; devstral is 3/3 at its 131K pool cap. Receipts: `*-v0515-needle2.json`) — rows measured before the KV-pool unwalling under-report vs current pools (see the README [VRAM context limits](../README.md#vram-context-limits) table); the tool-use probe (next section) is the current-depth agentic instrument. Current-stack fleet receipts are `*-v0517.json` (flip table in `patches/v0.5.17-rebase-status.md`). Thinking-model MMLU/LAB read correctly because the eval reads `reasoning_content` and gives budget to close `</think>`.

| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | **✓250K** | `gemma4.json` |
| Gemma 4 12B Unified AWQ | 77.2% | 92.5% | 29.3% | **✓250K** | `gemma4-12b.json` |

† **Gemma 4 21B REAP HumanEval 0%** is a known calibration artifact — the v3b ship serves cleanly per the audit but the REAP prune lost coding capability. Use `gemma4-31b` (the in-house dense AWQ rebuild) for code workloads.
‡ **Qwen3.5-28B-A3B-REAP LAB-Bench 15.9%** is on a partial 333-question subset (the eval timed out on the full 1786). Most rows are the full LAB-Bench (1786); the v0.5.12 rows use 20-per-subbench (140).

## 256K tool-use probe

◊ **qwen3-ream HumanEval 27.5%** — a non-coder text generalist on a code task (raw completion flattered it to 47%; the consistent no-think chat number is 27.5%). Use the Coder / qwen36 ships for code.



## 256K reasoning probe







## Ship gate

**SWE-bench Lite** scores live in the [bake-off table in the README](../README.md#coding-eval-bake-off-swe-bench-lite) — that's the end-to-end agentic eval (opencode/claw-code/little-coder scaffolds × v2 Docker harness), not part of this static-eval table.
