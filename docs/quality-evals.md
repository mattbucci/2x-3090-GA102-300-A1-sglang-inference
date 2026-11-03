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

## Current tables


Every shipped AWQ model is scale-integrity clean (fleet `check_awq_scales.py --base` audit, [receipt](benchmarks/quality/fleet-integrity-audit-2026-05-31.json)) and passes every applicable capability probe (thinking / image / video / audio / tool) on the current stack (`cap-*-v0522.json`). Methodology, footnotes, and per-probe findings: [`docs/quality-evals.md`](docs/quality-evals.md).

**Static evals** (`scripts/eval/eval_quality.py`, 2026-06 refresh: MMLU, HumanEval pass@1 no-think chat, [LAB-Bench](https://github.com/Future-House/LAB-Bench), Needle 1K → 250K; ±a few points is noise). Since the thinking-on serving defaults (2026-09-13) the per-flip gate is `scripts/eval/eval_and_chart.py` (MMLU / HumanEval at a thinking-sized budget, answer read from reasoning + content; receipts `benchmarks/quality/*-v0522.json`) — the two harnesses are not comparable, and `eval_quality.py`'s 200-token MMLU is a knife-edge on thinking models:

| Model | MMLU | HumanEval | LAB-Bench | Needle | Source |
|-------|:----:|:---------:|:---------:|:------:|:------:|
| Qwen3-VL-32B AWQ | **91.2%** | 83.3% | **39.8%** | 100% | `Qwen3-VL-32B-v0511.json` |
| Qwen3-Coder-30B-A3B AWQ | **91.2%** | **96.7%** | 33.3% | 100% | `Coder-30B-v0511.json` |
| Gemma 4 21B REAP AWQ | 80.7% | 0.0% † | — | — | `Gemma4-21B-REAP-v0511.json` |
| Qwen3-Coder-REAP-25B-A3B AWQ | 77.2% | **96.7%** | 30.5% | 100% | `Coder-REAP-25B-v0511.json` |
| Qwen3.6-35B-A3B AWQ-CT | 73.7% | 80.0% | — | — | `Qwen3.6-35B-A3B-CT-v0511.json` |
| Qwen3.5-28B-A3B-REAP AWQ | 69.6% | 80.0% | 15.9% ‡ | 100% | `REAP-28B.json` |
| Qwen3.6-REAM-A3B AWQ | 84.2% | **97.5%** | 24.3% | **✓ 250K** | `qwen36-ream.json` |
| Qwen3-30B-Instruct-2507 REAM AWQ | 80.7% | 27.5% ◊ | 35.0% | **✓ 250K** | `qwen3-ream.json` |
| Qwen3.6-35B-A3B AWQ-Marlin | 93.0% | **97.5%** | 21.4% | **✓ 250K** | `qwen36.json` |
| Qwen3.6-27B Dense AWQ | **98.2%** | **97.5%** | 27.1% | **✓ 250K** | `qwen36-dense.json` |
| Devstral-Small-2-24B AWQ | 77.2% | 80.0% | 33.6% | ✓131K (pool cap at measurement) | `devstral.json` |
| Gemma 4 31B Dense AWQ | 93.0% | **97.5%** | **42.9%** | **✓250K** | `gemma4-31b.json` |
| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | **✓250K** | `gemma4.json` |
| Gemma 4 12B Unified AWQ | 77.2% | 92.5% | 29.3% | **✓250K** | `gemma4-12b.json` |
| Flint 40B-A2.5B AWQ | 68.4% ◆ | 70.0% | — | **✓ 195K** natural ◆◆ | `flint.json` |

† REAP prune lost coding (use `gemma4-31b` for code). ◆ 57-question chat-mode MMLU under a 200-token cap on a thinking model — a same-instrument comparison, not a score. ◆◆ natural Open-Platypus haystack (the stock filler is a repetition attractor on Flint's NoPE layers); retrieval is reliable to 195K and text-dependent from ≈214K at depth ≤ 0.5 on either KV dtype ([issue #7](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/issues/7)). ‡ partial 333-question LAB subset. ◊ non-coder text generalist on a code task.

**256K tool-use probe** (`scripts/eval/probe_256k_tooluse.py`) — plants a needle deep in filler and measures whether the model emits a valid, correctly-argumented tool call with the planted value, bucketed by TRUE `prompt_tokens`. The agentic 256K signal SWE-bench Lite (tops ~128K) never reaches:

| Preset | valid tool call | correct args | max TRUE tokens still correct |
|---|:---:|:---:|:---:|
| `qwen36` (MoE-thinking) | **1.0** | **1.0** | **255,889** |
| `qwen36-ream` (MoE-thinking) | **1.0** | **1.0** | **255,889** |
| `qwen36-dense` (dense-thinking) | **1.0** | **1.0** | **258K** |
| `gemma4` (26B MoE) | **1.0** | **1.0** | **255,957** |
| `gemma4-12b` / `21b-reap` / `31b` | **1.0** | **1.0** | **258K** |
| `qwen35-moe` (DeltaNet MoE) | **1.0** | **1.0** | **255,889** |
| `qwen3-ream` (text generalist) | **1.0** | **1.0** | 148K single-turn (multi-turn agentic fails — bake-off-excluded) |
| `coder-30b-eval` / `coder-reap-25b` | 0.4 | 0.4 | **~64K agentic ceiling** — prose-stop instead of calls at ≥131K true |
| `devstral` (dense, tool) | **1.0** | **1.0** | **132K** firm; 178K flaky (measured at the pre-062 202K pool) |
| `nemotron3-omni` (Mamba2 AVLM) | 0.6 | 0.6 | **131K @8K budget** (76K @2K); ≥196K spirals past 8K, 253K prose-stops |
| `flint` (MoE, SWA + NoPE) | **1.0** | **1.0** | **234,430** on a natural haystack at depths 0.1 / 0.5 / 0.9 (`--haystack`, the probe's own calibration stops there); the stock filler is a repetition attractor on Flint (0.4 / 0.4, length-stops from 131K on either KV dtype) |

The flagship MoE thinkers and gemma4 hold perfect tool-calling at TRUE 256K, at every needle depth (0.1 / 0.5 / 0.9 — no lost-in-the-middle). The Coder-30B family does not: route agentic work beyond ~64K to the qwen36 family.

**256K reasoning probe** (`scripts/eval/probe_256k_quality.py`) — multi-key retrieval + variable-tracking + aggregation over a full context, identical task instances at every length (temp=0), measured at TRUE actual-token lengths (top point 255,800):

| Preset | 1K | 32K | 65K | 131K | 200K | 256K | overall |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemma4` (26B MoE) | 100 | 67§ | 100 | 100 | 100 | **100** | 94 |
| `gemma4-31b` (dense) | 100 | 100 | 100 | 100 | 100 | **100** | **100** |
| `gemma4-21b-reap` (MoE) | 100 | 100 | 100 | 100 | 100 | 67§ | 94 |
| `gemma4-12b` (unified omni, int4 RTN) | 100 | 67◊ | 67◊ | 33◊ | 33◊ | 67◊ | 61 |
| `qwen36` / `qwen36-dense` / `qwen36-ream` | 100 | 100 | 100 | 100 | 100 | **100** | **100** |

§ isolated single-cell misses, non-monotonic (noise, not a depth cliff). ◊ `gemma4-12b` chain-reasons to 256K but its multi-needle retrieval saturates at ~3 needles at depth. `qwen35-moe` is genuinely soft at depth (0.72 overall — `multikey` fails ≥131K); at 178K `qwen3-ream` 100%, `devstral` ~80%.

**Every new AWQ ship must pass `scripts/eval/validate_capabilities.py`** (basic + thinking + image + video + audio + tool, per applicable modality) before entering these tables.

