# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Headline — coding-eval bake-off (v2 Docker harness, 256K, single-user)

Best `(model, scaffold)` pair so far: `coder-30b-eval` × **opencode** = **121/300 = 40.3%** on SWE-bench Lite (`./scripts/launch.sh coder-30b-eval`).

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | **121/300 = 40.3%** | 115/300 = 38.3% | rolling |

Other models (qwen36, qwen36-ream, qwen35-moe, qwen36-dense, coder-30b-ream, coder-reap-25b [R9700 in-house], devstral, gemma4) queued for fresh full-300 cycles via [`run_model_cycle.sh`](evals/swebench/run_model_cycle.sh).

**Established scaffold-fit pattern:** thinking-mode Qwen3.5/3.6 models silently fail in claw (model exhausts `<think>` budget before committing a `tool_call`); they belong on opencode. Coder-tuned models match claw's `Bash`/`Edit`/`Read` tool registry and score similarly on claw vs opencode. Every preset's `--tool-call-parser` matches its chat-template tool format (see Known Issues for the family→parser mapping).

Per-cell receipts at [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/). Methodology, failure-mode analysis, and full audit trail live in [`patches/README.md`](patches/README.md).

## Why 40% — failure modes in the unresolved 170 (coder-30b-eval × opencode)

Patch-shape analysis across all 300 opencode predictions (2026-05-14). **The model attempts every instance; failure is over-patching, not silence.**

- **Over-edit signature.** Unresolved patches: median 3 files / p90 8 / p90 +278 added lines. Resolved: median 2 files / p90 5 / p90 +197. When the model is uncertain it widens the blast radius and breaks adjacent code paths it didn't need to touch.
- **Per-repo skew dominates the score.** scikit-learn 56.5%, django 47.4%, matplotlib 43.5%, sympy 37.7%. Long-tail: pytest 29.4%, xarray 20%, **sphinx-doc 6.2% (1/16), pallets/flask 0/3**. RST/docs tooling and Werkzeug request semantics are nearly unsolvable for a 30B-class coder; together they cost ~10 instances against any reasonable ceiling.
- **Two catastrophic patches.** `psf__requests-863` = 882 KB, 75 files, model created an entire `build/lib/requests/` shadow tree of the library. `psf__requests-2317` patch *adds* a new `comprehensive_test.py` (SWE-bench rejects new test files). Both score `error`. Tool-call agent occasionally loops on duplicate-tree generation or violates the "no new files" rule — 2/300 wasted slots, not a fix priority but worth a runaway-stop heuristic upstream.
- **Only 7 empty patches**, spread across 6 repos — these are real model give-ups on hard instances, not infra. The audit script already separates infra-fail (Connection error / HSAIL / UnicodeDecodeError) from model-silent.
- **Structural floor:** ~10 sphinx + 3 pallets + 7 empty + 2 error ≈ 22 instances are unwinnable at this model class without scaffold or prompt changes. Headroom to 50% lives in sympy (48 unsolved) and django (60 unsolved) — both candidates for scaffold/prompt iteration rather than model swap.

**Scaffolds aren't redundant — oracle-ensemble ceiling is 49%.** opencode and claw resolved 89 instances in common, **plus 32 opencode-only and 26 claw-only** (147/300 union = 49.0%, +8.7 pp above the 40.3% single-scaffold leader). Disagreement is distributed across repos, not concentrated: matplotlib 5-0 opencode, pytest 2-3 claw, psf/pydata 0-3 claw, django/sympy roughly even. **Running both scaffolds and unioning the diffs is the simplest 40%→49% lift** with no model change required. The two scaffolds fail in genuinely different ways (claw's `Bash`/`Edit`/`Read` registry vs opencode's filesystem-edit prompts); they are not noisy variants of each other.

**Two scaffold-fit failure modes show up across the 58 disagreements:**

1. **Fix-site selection differs per instance.** On matplotlib (5/5 opencode), opencode patches the root-cause site (e.g. `set_3d_properties` broadcast, `Legend.__getstate__`) while claw patches a downstream defensive site (e.g. `draw()` hasattr-fallback, `DraggableLegend.__getstate__`). Both are valid Python; only opencode's path satisfies the gold test. On psf/requests claw wins the inverse case: claw's `current_req = req` pointer in `SessionRedirectMixin.resolve_redirects` correctly threads method changes across multiple redirects, while opencode tracks a parallel `effective_method` variable that doesn't propagate. The pattern is **per-instance**, not per-scaffold.

2. **Model-helper file noise broke scoring on edge cases — fixed at the score layer.** Both scaffolds write reproducer/debug scripts at `/testbed/` root (`reproduce_bug.py`, `test_fix.py`, `comprehensive_test.py`, `debug_*.py`). On `psf__requests-2317` opencode's actual code edit (`builtin_str(method)` → `to_native_string(method)`) was correct and identical to claw's resolved version, but opencode also added two new test files at root → pytest collected them, hit an import error, SWE-bench's `get_eval_report` couldn't parse the malformed test output, `run_instance` marked the instance `error`. [`evals/swebench/filter_predictions.py`](evals/swebench/filter_predictions.py) (wired into `score_docker.py` as `--filter-helpers`, default ON) now strips new root-level helper files and `.claw/.sandbox-*` dirs from each prediction before the SWE-bench harness sees it. The original `predictions.jsonl` stays untouched as the rollout receipt; the harness reads `scores-docker/predictions.filtered.jsonl`. On coder-30b-eval × opencode the filter dropped 548 helper sections across 300 predictions (36% byte reduction).

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — calibration / quantization owner; ships the `mattbucci/*-AWQ` checkpoints this stack serves. We push bake-off + capability findings back into their README.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal-plumbing assumptions.

Both stacks on v0.5.11 (3090: 19 patches, R9700: 15, 8 shared content).

## Current Focus

**Single-user 256K context across all supported models is the primary serving target.** Multi-user throughput is secondary. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable.

**Hard constraint: preserve thinking + vision + video across every calibration.** Past recals silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py` (basic + thinking + image + video; `--skip-video` for image-only models like Devstral).

**End goal:** per-model bake-off matrix across `{opencode, claw-code, little-coder}` × all coder-class models on SWE-bench Lite at 256K, then SWE-bench Verified on the top 1-2 finalists. Driver: [`evals/swebench/run_model_cycle.sh <preset>`](evals/swebench/run_model_cycle.sh) handles launch → 3 rollouts → audit → reroll → score per preset.

Reference throughput: **Qwen3-30B REAM AWQ 262K @ 107 tok/s** (TP=2, 9.3 ms TPOT — `benchmarks/qwen3-30b-ream/long-context-v0511.json`); **Qwen3.6-35B-A3B AWQ-CT 256K @ 31 tok/s flat** thinking+vision (`benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`).

## Known Issues (open)

- **AWQ scales audit (2026-05-07) — 8 suspicious-class rare-expert under-cal findings.** Affects `qwen36` native (144 flagged) and `qwen35-moe`; CT variant of Qwen3.6-35B-A3B is clean (0/31010). Quality-degraded but serves. Recipe-side fix tracked by R9700 (`moe_calibrate_all_experts=True`). Full report: [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** 164 ALL-ZERO scale tensors from an empty `ignore` list. v3b build (regex `ignore`) is the shipping checkpoint at [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` produces gibberish across 4 distinct sources → upstream weight-mapping bug. Narrative in [`patches/README.md`](patches/README.md).
- **Qwen3.5-27B DeltaNet stuck at 32K.** DeltaNet TP replication forces 19 GB/GPU. Use `qwen3-ream` for long-context DeltaNet workloads.
- **60B+ models don't fit.** Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed the 48 GB total at MoE-AWQ.
- **Per-preset piecewise CUDA graph disables.** `coder-reap` / `coder-reap-25b` (cold-launch detokenizer hang); `qwen35-moe` / `qwen36` (DeltaNet+MoE+mamba_cache); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced). Reasons in launch.sh comments.
- **Model-scaffold tool-call format compatibility.** Coding harnesses (opencode / claw-code / little-coder) consume the SGLang OpenAI-compat endpoint's `tool_calls` field. SGLang only emits structured `tool_calls` when the preset passes `--tool-call-parser <fmt>` matching the format the model produces. Without the flag the model's raw `<function=NAME>...</function>` XML (qwen3-coder), `<tool_call>{json}</tool_call>` (qwen25), `[TOOL_CALLS]` (mistral) or Gemma `<\|tool>` is served as plain assistant text and the harness silently drops it — no edits applied. Audited all 19 presets against their chat templates (2026-05-13): 15 were missing flags. Mapping now: Qwen3-Coder + Qwen3.5/3.6 (incl. VL-REAP, dense, MoE, REAM) → `qwen3_coder`; Qwen3-VL non-coder + Qwen3-30B-Instruct REAM → `qwen25`; Devstral (Mistral arch) → `mistral`; Gemma 4 → `gemma4`. Runtime-validated end-to-end on qwen36 2026-05-13: request with `tools=[get_weather]` returns `finish_reason: tool_calls` with structured `function.arguments`.

## Suggested next

3090 owns **evals and serving validation**. Recalibration / in-house model rebuilds live with the [R9700 sister team](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

- **Per-model eval cycles** ([`run_model_cycle.sh <preset>`](evals/swebench/run_model_cycle.sh)): `qwen36`, `qwen36-ream`, `qwen35-moe`, `qwen36-dense`, `coder-30b-ream`, `coder-reap-25b` (R9700 in-house refresh), `devstral`, `gemma4`. Each cycle: full 300-inst × 3 scaffolds + audit + reroll + score. Estimated 8-18h per preset.
- **SWE-bench Verified (500-task)** on the top 1-2 finalists once the matrix is settled.
- **Qwen3-VL-30B MoE loader fix** — gibberish across 4 sources; upstream weight-mapping bug. Non-coder, lower priority.
- **Devstral-24B-AWQ HF mirror** at `mattbucci/Devstral-24B-AWQ` after TP=2 validation.

Performance / post-bake-off:
- `qwen3-ream` + `coder-30b` at TP=2 with piecewise CUDA graph re-enabled (currently `--disable-piecewise-cuda-graph`; test if conditional-on-$TP lifts 107 tok/s @ 250K and 180 tok/s @ 16K).
- Gemma 4 multi-attention-backend A/B at TP=2 once v0.5.12+ relaxes the head_dim=256 + Ampere FP8 incompat that forces triton-attn today.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.11, apply patches, create conda env

# TP=2 / 256K presets (matrix standard):
./scripts/launch.sh qwen3-ream              # 262K @ 107 tok/s — REAM merged MoE, 96 experts
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE CT — 256K @ 31 tok/s, thinking+vision
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — DeltaNet+attn
./scripts/launch.sh coder-30b               # Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh coder-reap-25b          # Coder-REAP-25B MoE AWQ-Marlin — 256K @ 109 tok/s
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — 131K @ TP=2
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AutoRound-AWQ

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

### Kernel and driver

- `linux-zen` kernel (Arch `extra/linux-zen`), not stock `linux` — the stock Arch kernel + the open NVIDIA module hard-locked the host repeatedly under sustained TP=2 / 256K bake-off load on this rig. The zen patchset's scheduler and IO tuning eliminated the recurrence; same major version (6.19.11), so all other config carries over.
- `nvidia-open-dkms` (not `nvidia-open`) — DKMS rebuilds the module for every kernel that has headers installed, so the same NVIDIA driver version covers both `linux` and `linux-zen`.

### Cooling and power profile (load-bearing)

Two systemd units hold a cooling profile that's required for the bake-off to survive multi-hour runs on this chassis. The DDR5 SPD sensors crossed `ALARM HIGH` (55 °C) under stock cooling + default 350 W per 3090; that correlated with random Python heap corruption / kernel BUGs / hard resets. The profile below stays inside spec under sustained TP=2 inference.

| Unit | Action |
|------|--------|
| `gpu-cooling.service` | Boot oneshot. Enables NVIDIA persistence mode, sets each 3090's power limit to **260 W** (down from default 350 W), pushes Corsair Commander Core XT case fans to 100% via `liquidctl`, enables manual GPU fan control, seeds 75 % fan floor. |
| `gpu-fan-curve.service` | Long-running daemon. Polls each GPU's temperature every 4 s. Fan speed = 75 % below 60 °C, linear ramp to 100 % between 60 °C and 80 °C, 100 % at 80 °C+. Re-asserts manual fan control once on start. |

Scripts live at `/usr/local/bin/gpu-cooling.sh` and `/usr/local/bin/gpu-fan-curve.sh`. Enable with `systemctl enable --now gpu-cooling.service gpu-fan-curve.service`. Verify after a reboot with `nvidia-smi --query-gpu=power.limit,fan.speed,temperature.gpu --format=csv` (expect 260 W limit, ~75 % fans idle).

Ampere consumer cards do not expose VRAM junction temperature, so the GPU-core temp shown by `nvidia-smi` is an underestimate of the real thermal pressure. 260 W picked to leave inference throughput headroom while still cutting peak heat ~25 %.

## Model Support

Single-user tok/s measured at the max-context value in the table. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **107** | 9.3 ms | `qwen3-ream` | TP=2 / 262K, 183 tok/s @ 1K → 107 tok/s @ 250K. Receipt: `benchmarks/qwen3-30b-ream/long-context-v0511.json`. |
| **Qwen3.6-35B-A3B AWQ-CT** | DeltaNet+MoE (256 exp, VL) | **262K** | **31** | 32 ms | `qwen36` | TP=2 / 256K, decode flat 30.3-31.5 tok/s across 1K-250K, 4/4 PASS via patch 030. CT is calibration-clean (0/31010 vs native's 144 flagged). Native override: `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ QUANT=awq_marlin`. Receipt: `benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`. |
| **Qwen3.6-27B AWQ** | Dense + DeltaNet | **131K** | **21** | 47 ms | `qwen36-dense` | R9700 self-cal at `mattbucci/Qwen3.6-27B-AWQ`. 4/4 PASS at TP=2; bakeoff matrix validation pending. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **131K** | **40** | 25 ms | `qwen3-vl-32b` | R9700 self-cal at `mattbucci/Qwen3-VL-32B-AWQ`. TP=2: 68 → 50 → 40 tok/s @ 1K/65K/131K, 3/3 PASS. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **50** | 19.9 ms | `devstral-long` | TP=2 / 217K, basic PASS, decode 59 → 50 tok/s @ 1K/200K. Vision OOMs at MEM=0.97 (preset bakes `--skip-server-warmup`). Text-only path clean. |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | TP=2 only — same Dense 24B prealloc constraint. |
| Coder-REAP-30B AWQ-Marlin | MoE (96 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | TP=2 / 256K, R9700 in-house rebuild from upstream BF16 (96 experts/layer, GPTQ W4A16 + `moe_calibrate_all_experts=True`). Replaced Cerebras pre-pruned 2026-05-14. |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | TP=2 / 16K, 187 → 180 tok/s @ 1K/16K. Original AWQ-Marlin layout (vs `coder-30b-eval` which is CT). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | TP=2 / 262K, decode flat 30.5 tok/s across 1K-250K, 4/4 PASS. R9700 cross-validated 4/4 on RDNA4. |
| Gemma 4 31B Dense | Dense | 16K | 22 | ~50 ms | `gemma4-31b` | basic+thinking real; vision validator-passes-but-degraded. R9700 recal in flight. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | basic+thinking real; vision content-aware on v0.5.11 (`'a solid red circle with a black outline'`). |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | 4K | — | — | — | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Audit clean. |

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

### Quality benchmarks

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|-------|:----:|:---------:|:---------:|:------:|
| `coder-30b` | **91.2%** | 96.7% | 33.3% | ✓ to 4K |
| `qwen3-vl-32b` | **91.2%** | 83.3% | **39.8%** | ✓ to 4K |
| `coder-reap-25b` | 77.2% | 96.7% | (n/a) | ✓ to 65K |

Methodology: MMLU (1 question per subject × 57 subjects), HumanEval pass@1 (30), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7×50), needle-in-a-haystack (1K→65K). Receipts in `benchmarks/quality/*-v0511.json`. SWE-bench Lite resolve rates are in the headline table at the top of this file.

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
| SGLang | v0.5.11 + 19 local patches (`ls patches/*.patch \| wc -l`) |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.58) / cu130 wheel |
| NCCL | bundled with torch 2.11 (P2P over NVLink) |
| FlashInfer | 0.6.8.post1 (v0.5.11 pin) |
| transformers | 5.6.0 (v0.5.11 pin) |
| sglang-kernel | 0.4.2 |
| compressed-tensors | 0.15.0.1 |

## Patches

19 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.11. Notable:
- **002** Qwen3-Next AWQ weight_loader fix (cross-team port from R9700)
- **028** Gemma 4 MM per-expert AWQ loader (cross-stack with R9700)
- **029** Qwen3.5 shared_expert_gate CT dequant
- **030** fused_moe_triton presharded-w2 detection (unblocks CT MoE at TP≥2; qwen36 default switched to CT)
- **031** Qwen3.5/3.6 DeltaNet AWQ weight_loader
- **034** sampler ±Inf detection (cross-team port from R9700)

Per-patch narratives + closed-item history in [`patches/README.md`](patches/README.md).

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
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Cross-team threads (R9700 answers + advice on closed items) live in [`patches/README.md`](patches/README.md).
