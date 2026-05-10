# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Cross-team activity (last ~2 weeks)

R9700 (RDNA4) and M4 (Apple) sister teams ship findings into our repo. Per-day forensic narrative + closed-item history in [`patches/README.md`](patches/README.md) and `git log -- README.md`.

- **Both stacks at v0.5.11** (3090 patch rebase commit `1655e46`, R9700 commit `3466816`, 2026-05-07; 3090 env upgrade 2026-05-09). 3090: 17 patches, R9700: 15; 8 share content cross-stack.
- **R9700 ships `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`** (2026-05-09) — first in-house REAM merge from upstream BF16 via Samsung SAIL `merge.py` (128→96 experts, ~30B→~23B params). Ampere cross-validation 2026-05-09: 1/1 PASS + STRONG 8/8 codegen-probe at TP=1 / 4K via `coder-30b` preset (`QUANT=moe_wna16 DTYPE=bfloat16`).
- **3090 ships `mattbucci/gemma-4-21B-REAP-AWQ`** (2026-05-09, commit `6bc20c5`) — 4/4 PASS at TP=1 / 4K via `gemma4` preset under v0.5.11 + patches 023+028. Audit clean. R9700 cross-validation pending their v0.5.11 env upgrade.

> **Recommended for coding tasks: `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% on SWE-bench Lite** (`./scripts/launch.sh coder-reap-25b`). **2026-05-09 v0.5.11 quality matrix** (eval framework MMLU runs 1-per-subject = 57 questions across MMLU's 57 subjects; `benchmarks/quality/*-v0511.json`):
>
> | Model | MMLU | HumanEval | LAB-Bench | Needle |
> |-------|:----:|:---------:|:---------:|:------:|
> | `coder-30b` | **91.2%** | 96.7% | 33.3% | ✓ to 4K |
> | `qwen3-vl-32b` | **91.2%** | 83.3% | **39.8%** | ✓ to 4K |
> | `mattbucci/gemma-4-21B-REAP-AWQ` | 80.7% | extractor-broken* | (n/a) | (n/a) |
> | `coder-reap-25b` | 77.2% | 96.7% | (n/a) | ✓ to 65K |
> | `qwen36-tp1` (CT default) | 73.7% | 80.0% | (partial) | (partial) |
>
> *Gemma 4 HumanEval extractor-broken; codegen-probe STRONG 8/8 is the cleaner signal there. Codegen-probe (`probe_codegen.py`) STRONG 8/8 across 7/7 production presets on v0.5.11; PARTIAL on `qwen3-ream` (generalist, not coder-tuned). Harness: opencode v1.14.25 headless, 256K, scored locally. Resolved-rate where tests actually ran: 88/236 = 37.3%. Artifacts: `evals/swebench/runs/coder-reap-25b-lite/`. Bake-off continues — see Suggested next §F.*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**End goal: Docker-harness coding-eval rollouts at 256K context, single-user 1-request-at-a-time, across all coder-class models, scored against opencode AND little-coder scaffolds.** Everything below is a milestone on that path: serve correctly at 256K (sweep done), preserve calibration quality (audit + recal), unblock loader bugs (CT TP=2 narrow, Qwen3-VL-30B MoE), then ship Coder-30B / Qwen3.6-35B / Devstral-24B / Qwen3-30B-REAM through the Docker harness on the v0.5.11 stack. Final number: SWE-bench Verified on the top 1-2 finalists.

**Primary serving target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 107 tok/s** (9.3 ms TPOT, fresh prefill, **measured at TP=2** — 2026-05-09 v0.5.11 sweep at `benchmarks/qwen3-30b-ream/long-context-v0511.json`; matches the 2026-04-18 baseline). For thinking + vision at 256K on the same GPU budget: **Qwen3.6-35B-A3B AWQ-native — 31 tok/s flat across 1K-250K** (TPOT 31.1-32.3 ms, 2026-05-09 v0.5.11 TP=2 sweep at `benchmarks/qwen3.6-35b-a3b/v0511-tp2-flashinfer.json`). The historical decode regression that dropped 33 → 2.6 tok/s @ 250K closed on v0.5.11.

### Cold-launch matrix (TP=1 / 24 GB single-card)

All `✅` rows are 4/4 PASS at TP=1 from the 2026-05-09 v0.5.11 sweep (`benchmarks/quality/capability_check.json`).

| Preset | TP=1 cold | Note |
|--------|:---------:|------|
| `qwen35` | ❌ | OOM at preset CTX=32K. Use `qwen35-tp1` (CTX=4K). |
| `qwen35-tp1` | ✅ | Qwen3.6-27B-AWQ R9700 recal. CTX=4K. |
| `qwen36` | ❌ | OOM at preset CTX=262K. Use `qwen36-tp1` for single-card; `qwen36` is for TP=2 / 256K. |
| `qwen36-tp1` | ✅ | CT default. CTX=2K, MAX_MAMBA_CACHE=4. |
| `qwen3-ream` | ✅ | Bakes `--disable-piecewise-cuda-graph` (TP=1 awq_marlin MoE protection). |
| `coder-30b` | ✅ | Same piecewise bake as `qwen3-ream`. STRONG 8/8 codegen-probe. |
| `coder-reap` | ✅ | Text-only (W4A16 auto-round). Bakes `--disable-piecewise-cuda-graph` for the cold-launch detokenizer hang. STRONG 8/8 codegen-probe. |
| `qwen3-vl-32b` | ✅ | R9700 self-cal `mattbucci/Qwen3-VL-32B-AWQ`. CTX=4K @ TP=1; TP=2 unlocks 131K. |
| `gemma4-31b` | ⚠️ | Validator-passes but vision content-degraded ("scattered red and black pixels" instead of "red circle"). Same Gemma-4 calibration limitation as 26B; recipe-side recal in §B / §E1. |
| `gemma4` (26B MoE) | ✅ | v0.5.11 + patches 023/024/025/026/028. Content-aware vision/video as of 2026-05-09. |
| `qwen3-vl-moe` | ❌ | SGLang `Qwen3VLMoeForConditionalGeneration` loader broken; see Known Issues. |
| `devstral` / `devstral-long` | ❌ | TP=2 only — Dense 24B AWQ create_weights prealloc OOMs on TP=1. |

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json`.

## Known Issues (open)

- **AWQ scales audit (2026-05-07) — 8 suspicious-class rare-expert under-cal findings.** Affects `qwen36` native (144 flagged) and `qwen35-moe`; CT variant of Qwen3.6-35B-A3B is clean (0/31010). Quality-degraded but serves. Recipe-side fixes in §B. Full report: [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** 164 ALL-ZERO scale tensors from an empty `ignore` list. v3b build (regex `ignore`) is the shipping checkpoint at [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` produces gibberish across 4 distinct sources → upstream weight-mapping bug. Narrative in [`patches/README.md`](patches/README.md).
- **Qwen3.5-27B DeltaNet stuck at 32K.** DeltaNet TP replication forces 19 GB/GPU. Use `qwen3-ream` for long-context DeltaNet workloads.
- **60B+ models don't fit.** Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed the 48 GB total at MoE-AWQ.
- **Devstral-24B Dense OOMs on TP=1.** ~23.48 GiB resident before `create_weights` + per-layer int32 destination push past 24 GB. Needs an upstream lazy/streamed loader. TP=2 path is fine.
- **Per-preset piecewise CUDA graph disables.** `coder-reap` / `coder-reap-25b` (cold-launch detokenizer hang); `qwen35-moe` / `qwen36` (DeltaNet+MoE+mamba_cache); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced); `qwen3-ream` / `coder-30b` (TP=1 awq_marlin MoE protection — see §G for TP=2 piecewise-on retest). Reasons in launch.sh comments.

## Suggested next

Backlog grouped by area. Headline destination: **Docker-harness coding-eval rollouts at 256K / single-user / 1-req-at-a-time** (§F). Closed items live in [patches/README.md](patches/README.md) and `git log`.

### A. Loader patches

- **A1.** Qwen3VLMoe `*ForConditionalGeneration` weight-mapping fix — gibberish across 4 distinct sources, source-independent. Non-coder, lower priority.
- **A2.** Devstral-24B Dense lazy/streamed `create_weights` (unblocks TP=1). Upstream PR. TP=2 already works.

### B. Recalibration backlog

Targets the 8 audit-flagged rare-expert findings ([`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md)). Recipe template: `NUM_CALIBRATION_SAMPLES≥1024`, regex `ignore` for `vision_tower` / `router` / `shared_expert_gate`, `drop_images=False` for multimodal, `force_route_all_experts` for MoE.

- **B1.** Qwen3.6-35B-A3B-AWQ native (l1.exp.214, 144 flagged) — CT clone is already clean; recal would either match or beat.
- **B2.** Qwen3.6-VL-REAP-26B-A3B-AWQ — better fixed by §C1 (vision-tower-retained REAP from upstream BF16).
- **B3.** Qwen3-Coder-REAP-25B-A3B-AWQ (l1.exp.22) — SWE-bench Lite leader; quality lift affects bake-off.
- **B4.** Qwen3-Coder-Next-REAM-AWQ — won't fit at 48 GB, deprioritized.

### C. In-house rebuilds (build-from-scratch rule)

Three shipped models source from 3rd-party prunes — rebuild via `run_ream_qwen3moe.sh` from upstream BF16 keeps quality knobs (vision tower, router, ignore lists) under our control.

- **C1.** Qwen3.6-35B-A3B clean text-only REAP (R9700 task #60) — Samsung SAIL prune with vision tower retained.
- **C2.** Gemma 4 26B REAM-or-REAP (R9700 task #61) — no in-house pruned variant yet.
- **C3.** Coder-REAP-25B from upstream BF16 — currently Cerebras prune. Multi-week; keep current ship live until in-house validates.

### D. Cross-team / portability

- **D1.** R9700 v0.5.11 env upgrade — verify long-ctx regression also closes on RDNA4 (Ampere lift: `qwen36` TP=2 went 33→2.6 → flat 31 tok/s @ 250K).
- **D2.** R9700 cross-validation of `mattbucci/gemma-4-21B-REAP-AWQ` after they port patch 023.
- **D3.** Devstral-24B-AWQ HF mirror ship at `mattbucci/Devstral-24B-AWQ`. Validate at TP=2 first.

### E. Quality improvements

- **E1.** Gemma 4 vision quality — R9700 `drop_images=False` recal in flight; A/B their CT artifacts on 3090 when they land. Currently keyword-grep PASS but content-degraded ("scattered red pixels" instead of "red circle").
- **E2.** Audio modality for Gemma 4 — upstream-blocked (Google BF16 base has zero audio_tower keys). Track for future releases.

### F. Coding-eval bake-off — Docker harness, 256K, single-user

> **Running now (2026-05-10):** `bake_off.sh` phases p1–p10 detached via `setsid`. Score workers tightened to 8 (was 24) and scorers `flock`-serialized so only one runs at a time across phases — preserves the "1 rollout + 1 score concurrent" design. Tail `/tmp/loop-bakeoff-logs/bakeoff.log`.

- **F1.** ✅ Coder-30B v2 — 300/300 predictions on disk (294 rc=0, 13 EMPTY), `evals/swebench/runs/coder-30b-docker-v2/`. Driver rebuilt 2026-05-09 (commit `d11bde7`). Next action: score against the official SWE-bench Docker harness.
- **F2.** Coder-REAP-25B v2 — Docker rollout + Docker scoring; compares cleanly against F1 on the same driver.
- **F3.** Qwen3.6-35B-A3B v2 — TP=2 / 256K serving validated; rollout unblocked.
- **F4.** Devstral-24B v2 — text-only at TP=2.
- **F5.** Qwen3-30B-REAM v2 — fastest decode in the lineup.
- **F6.** little-coder scaffold A/B on each F1-F5 finalist (`npm i -g little-coder`, OpenAI-compat against `:23334`).
- **F7.** claw-code scaffold smoke (Rust `claw`, `cargo build --workspace`).
- **F8.** SWE-bench Verified (500-task) on top 1-2 finalists from F1-F5 — final headline number.

### G. Performance / optimization (post-bake-off)

- **G1.** `qwen3-ream` + `coder-30b` at TP=2 with piecewise CUDA graph re-enabled — current presets bake `--disable-piecewise-cuda-graph` for TP=1 awq_marlin MoE protection; test if conditional-on-`$TP` lifts the headline (107 tok/s @ 250K and 180 tok/s @ 16K respectively today).
- **G2.** Multi-attention-backend Gemma 4 A/B at TP=2 — currently triton-attn forced (head_dim=256 + Ampere FP8 incompat). Revisit when v0.5.12+ lands.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.11, apply patches, create conda env

# TP=1 / 24 GB friendly (single-card cold-fit baseline):
./scripts/launch.sh qwen3-ream              # fastest 256K — reference model (MoE active params fit cold)
./scripts/launch.sh qwen35-tp1              # Qwen3.6-27B-AWQ R9700 recal — TP=1 cold-fit variant (CTX=4K), 4/4 PASS (2026-05-07 strict sweep)
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput
./scripts/launch.sh coder-reap              # Coder-REAP-25B — SWE-bench Lite leader (29.3%)
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — TP=1 defaults boot cold (4K/MAX_RUNNING=1)
./scripts/launch.sh qwen36-tp1              # Qwen3.6-35B-A3B AWQ-native — TP=1 cold-fit variant (CTX=2K), 4/4 PASS (2026-05-07 strict sweep)

# TP=2:
./scripts/launch.sh devstral-long           # Devstral-24B at 217K — TP=2 only (Dense AWQ create_weights prealloc OOMs on TP=1)
./scripts/launch.sh devstral                # Devstral-24B 131K default

python scripts/eval/validate_capabilities.py --port 23334                 # auto-skips thinking/vision/video per preset
python scripts/eval/test_capabilities_all.sh                              # sweep across all AWQ presets
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

- 2x NVIDIA RTX 3090 (24 GB each, 48 GB total) with NVLink bridge
- NVIDIA driver 595+ / CUDA 13.x
- Miniforge3 or Conda
- ~150 GB disk for models

## Model Support

Single-user tok/s measured at the max-context value in the table. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **107** | 9.3 ms | `qwen3-ream` | TP=2 / 262K, 183 tok/s @ 1K → 107 tok/s @ 250K. Receipt: `benchmarks/qwen3-30b-ream/long-context-v0511.json`. |
| **Qwen3.6-35B-A3B AWQ-CT** | DeltaNet+MoE (256 exp, VL) | **262K** | **31** | 32 ms | `qwen36` | TP=2 / 256K, decode flat 30.3-31.5 tok/s across 1K-250K, 4/4 PASS via patch 030. CT is calibration-clean (0/31010 vs native's 144 flagged). Native override: `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ QUANT=awq_marlin`. Receipt: `benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`. |
| **Qwen3.6-27B AWQ** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` | R9700 self-cal at `mattbucci/Qwen3.6-27B-AWQ`. 4/4 PASS at `qwen35-tp1` TP=1 / 4K. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **131K** | **40** | 25 ms | `qwen3-vl-32b` | R9700 self-cal at `mattbucci/Qwen3-VL-32B-AWQ`. TP=2: 68 → 50 → 40 tok/s @ 1K/65K/131K, 3/3 PASS. TP=1 / 4K cold-fit also passes. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **50** | 19.9 ms | `devstral-long` | TP=2 / 217K, basic PASS, decode 59 → 50 tok/s @ 1K/200K. Vision OOMs at MEM=0.97 (preset bakes `--skip-server-warmup`). Text-only path clean. |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | TP=2 only — same Dense 24B prealloc constraint. |
| Coder-REAP-25B AWQ-Marlin | MoE (103 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | TP=2 / 256K, 183 → 109 tok/s @ 1K/250K. SWE-bench Lite: 29.3% on v1 host harness; v2 Docker queued (§F2). |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Auto-round W4A16 build of the same Cerebras REAP source. |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | TP=2 / 16K, 187 → 180 tok/s @ 1K/16K. SWE-bench Lite v2: 300/300 predictions ready for scoring (§F1). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | TP=2 / 262K, decode flat 30.5 tok/s across 1K-250K, 4/4 PASS. R9700 cross-validated 4/4 on RDNA4. Cerebras REAP base + `balanced_thinking_vision` recal (333 vision tensors retained). |
| Gemma 4 31B Dense | Dense | 16K* | 22 | ~50 ms | `gemma4-31b` | basic+thinking real; vision validator-passes-but-degraded ("scattered red pixels" — see §E1 recal). *KV tight at TP=1 / 16K — drop to CTX=4K or run TP=2. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | basic+thinking real; vision content-aware on v0.5.11 (`'a solid red circle with a black outline'`) — earlier "scattered pixels" output was the v0.5.10 SGLang serving gap. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | 4K* | — | — | — | Shipped 2026-05-09 to [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). 4/4 PASS at TP=1 / 4K via `gemma4` preset. Audit clean. v2 build is a calibration disaster — see Known Issues. |

### VRAM context limits (KV dtype varies, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-native | 9.87 GB | ~8 KB hybrid | 262K |
| Coder-30B AWQ | 8.0 GB | 36 KB | 262K |
| Devstral-24B AWQ (long preset) | 7.0 GB | 80 KB | **217K** (true 3090 ceiling for 24B dense @ TP=2) |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 131K |
| Qwen3.5-27B AWQ | 19.0 GB | 24 KB | 32K (weights replicated for DeltaNet TP) |

## Benchmarks

Per-model long-context sweep JSON in `benchmarks/<model>/`. Reference: Qwen3-30B REAM AWQ at `benchmarks/qwen3-30b-ream/long-context-262k.json`. Qwen3.6-35B-A3B AWQ-native detailed curve + tuning experiments in `benchmarks/qwen3.6-35b-a3b/awq-native-thinking-vision.json`.

### Quality (REAP vs REAM vs original)

![Quality Comparison](benchmarks/quality/quality_comparison.png)

| Model | MMLU | HumanEval | Needle (65K) |
|-------|:----:|:---------:|:------------:|
| Coder-30B (128 exp) | 73% | 100% | 100% |
| REAP-28B DeltaNet (205 exp) | 70% | 80% | 100% |
| REAM-30B (96 exp) | 63% | 80% | 100% |

Methodology: `scripts/eval/eval_and_chart.py` — MMLU (200 samples), HumanEval pass@1 (30 samples), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 × 50), needle-in-a-haystack (1K→65K). Temperature=0, full context as reasoning budget.

**Still TODO:** [RULER](https://github.com/NVIDIA/RULER) (4K→256K synthetic), [LongBench Pro](https://arxiv.org/html/2601.02872v1), [LiveCodeBench](https://livecodebench.github.io/).

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
cd components/sglang && git checkout v0.5.11
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e ".[srt]"
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.11 + 17 local patches (`ls patches/*.patch \| wc -l`) |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.58) / cu130 wheel |
| NCCL | bundled with torch 2.11 (P2P over NVLink) |
| FlashInfer | 0.6.8.post1 (v0.5.11 pin) |
| transformers | 5.6.0 (v0.5.11 pin) |
| sglang-kernel | 0.4.2 |
| compressed-tensors | 0.15.0.1 |

## Patches

17 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.11. Originally rebased from a v0.5.10 set 24→13 (2026-05-07 commit `1655e46`); 002 cross-team port of R9700's qwen3_next AWQ weight_loader fix; 028 v0.5.11 gemma4_mm per-expert AWQ loader (R9700 `gemma4_causal.py` port); 029 qwen35 shared_expert_gate CT dequant; **030 fused_moe_triton presharded-w2 detection** (unblocks CT MoE at TP≥2 — qwen36 default switched to CT 2026-05-09). Per-patch narratives in [`patches/README.md`](patches/README.md).

## Quantization

Self-calibrated models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                       # MoE CT→native AWQ
```

For thinking+vision-preserving calibration: `scripts/quantize/calibration_datasets.py` builds `thinking_text` / `thinking_vision` / `code_vision` / `code_thinking` recipes (drawing from AM-Thinking-v1-Distilled, NuminaMath-CoT, LLaVA-Instruct-150K, UltraChat, the-stack). See [rules-for-agents.md](rules-for-agents.md) and [REAM.md](scripts/quantize/REAM.md).

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
RAM:    96 GB (92 usable)
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24 GB GDDR6X each)
NVLink: NV4 — 4 lanes × 14 GB/s = 56 GB/s bidirectional
Driver: 595.58.03   CUDA 13.2   Python 3.12
```

## Repo layout

```
patches/                  # SGLang v0.5.11 patches — see patches/README.md for full narratives
benchmarks/               # Per-model benchmark JSON + charts
  quality/                #   MMLU / HumanEval / LAB-Bench / Needle
  <model>/                #   throughput + long-context sweeps
scripts/
  launch.sh               # unified launcher (launch.sh <preset>)
  common.sh               # shared conda + NVIDIA env setup
  setup.sh                # full setup (conda, SGLang install, patch apply)
  bench/                  # throughput benchmarks
  eval/                   # quality evals + chat template validator
  quantize/               # GPTQ → CT → AWQ pipeline + calibration recipes
  test/                   # kernel microbenchmarks + profiling
components/sglang/        # SGLang v0.5.11 + patches (cloned by setup.sh)
components/sglang.v0.5.10-backup-2026-05-09/  # rollback safety net (delete after v0.5.11 stable burn-in)
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Cross-team threads (R9700 answers + advice on closed items) live in [`patches/README.md`](patches/README.md).
