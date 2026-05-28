# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

> 📢 **Cross-team from R9700 (2026-05-26): MoE squash SOLVED — your marlin was never implicated.** Root cause was our v0.5.12 patch rebase: `004-rdna4-moe-fixes` rejected *wholesale* (layer.py moved-import cascade) so the RDNA4 `get_default_config` hunk silently dropped, losing `BLOCK_SIZE_N` from the int4 decode config → `dynamic_func` crash on fresh Triton cache / wrong-tile 20× squash on stale. Not the kernel (amdgcn bit-identical v11↔v12), not the prune, not zero-point. Mechanically isolated via per-version Triton-cache diff + clean re-clone abort. **Fix:** rebased all 25 patches clean on v0.5.12; every int4 MoE now coherent at 256K. No action needed on your side — thanks for the marlin reference layout, it ruled out the bind/swap branch fast.
>
> 📢 **R9700 validated your `mattbucci/Qwen3.6-35B-A3B-AWQ` recal (2026-05-26) — PASS on RDNA4.** TP=2 / 262144 / mem0.80: basic ✅ (Paris), thinking ✅ (reasoning_content active), vision ✅ ("a red square"). 256-expert `qwen3_5_moe` multimodal serves clean on our Triton MoE post-fix. `check_awq_scales`: 144 flags but 142 are layer-0 experts gate/up (structural, ~55% zero — same first-layer pattern as our ships) + 2 at layer-1 (audit-tier, matches coder l1.exp). Recipe travels cleanly to RDNA4.

## Headline — coding-eval bake-off (v2 Docker harness, 256K, single-user)

Best `(model, scaffold)` pair so far: `qwen36-ream` × **opencode** = **176/300 = 58.7%** on SWE-bench Lite (`./scripts/launch.sh qwen36-ream`). +15.7 pp over the prior leader (`coder-30b-eval`); thinking-mode REAM-merged MoE pays off when the scaffold lets it think.

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | **176/300 = 58.7%** | 20/123 = 16.3% † | 0/10 = 0.0% † |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% | 107/300 = 35.7% | 74/300 = 24.7% |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% | 122/300 = 40.7% | 101/300 = 33.7% |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% | 109/300 = 36.3% | 73/300 = 24.3% |

† `qwen36-ream` claw-code and little-coder rerolls hit the "10 consecutive empty diffs" safety abort at the partial denominators shown — the same scaffold-fit pattern documented below (thinking-mode models exhaust `<think>` budget before committing a `tool_call` on tool-call-heavy scaffolds). Opencode lets the model think before edit, hence the 58.7% lift.

`coder-reap-25b` remains the most-rounded preset (only 1.3 pp behind `coder-30b-eval` on opencode while leading on claw-code +5.0 pp and little-coder +9.0 pp). `qwen36-ream` is the right pick when the scaffold matches the model (opencode); `coder-reap-25b` is the right pick when scaffold mix is uncertain.

`coder-30b-ream` (REAM-merged coder) lands below both the base CT `coder-30b-eval` and the REAP-pruned `coder-reap-25b` on every scaffold — the merge didn't help here. **Next: the matrix moves to sglang v0.5.12** for a uniform single-stack re-run of all presets — it upstreams most of our patches and clears the `gemma4` head_dim=256 blocker — adding the not-yet-run `qwen36`, `qwen36-dense`, `devstral`, `gemma4`. `qwen35-moe` is unblocked (its "missing checkpoint" was a stale HF-cache snapshot symlink, now repointed; boots 4/4 PASS basic+thinking+vision+video on v0.5.12, no fuse-convert needed). `qwen36` is being **rebuilt from scratch as AWQ-Marlin** (the CT build hits a v0.5.12 MoE-loader bug — see Current Focus). The 0.5.11 numbers above hold until that re-run lands.

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

**Rollout self-clean is live (qwen36 onwards).** After the main scaffold invocation finishes, the same scaffold re-runs with a short CLEANUP_PROMPT asking the model to `rm` its own reproducer / debug / analysis helpers. Validated against the first 25 qwen36 × opencode predictions vs coder-30b-eval's pre-clean ones on the same instances: **0 helper files in qwen36 patches vs 40 in coder-30b-eval's**. Median patch size dropped 3-5×. Cost: per-instance time roughly tripled (mean 275s vs ~150s baseline). Score impact won't be visible until the qwen36 cycle finishes — the design hypothesis is that cleaner patches reduce SWE-bench `error` (pytest collection failures from helpers) without losing resolved cases (`filter_predictions.py`'s empty-fallback isn't applicable here because the model rather than a regex decides what's a helper).

**Why little-coder lags 17 pp behind (22% vs 39%) — diagnosis in progress.** little-coder wraps [`pi`](https://github.com/earendil-works/pi), which is a coding agent (full tool registry: `edit`/`write`/`bash`/`read`/`grep`/`find`/`ls`; sends OpenAI-structured `tools=[...]` and parses streaming `delta.tool_calls[]`). So this is NOT a chat-vs-agent gap. But ~13% of pi's predictions (40/300) come back with `git diff --cached` empty — the model output never resulted in a file modification. Another ~20% are only new files at testbed root (`reproduce_bug.py`, `test_fix.py`) with no real source edit. When pi engages tool-use it works fine — 65/66 of its resolved cases touched existing source, identical to opencode/claw. Real cause not yet pinned: candidates are (a) SGLang's `qwen3_coder` tool-call parser producing chunks that pi's streaming accumulator drops, (b) pi's tool descriptions less effective at engaging Qwen3-Coder vs opencode's prompt, or (c) pi's system prompt putting less emphasis on edit-tool use. Needs a small smoke capturing pi's outgoing request + SGLang's actual streamed response to disambiguate.

**Two scaffold-fit failure modes show up across the 58 disagreements:**

1. **Fix-site selection differs per instance.** On matplotlib (5/5 opencode), opencode patches the root-cause site (e.g. `set_3d_properties` broadcast, `Legend.__getstate__`) while claw patches a downstream defensive site (e.g. `draw()` hasattr-fallback, `DraggableLegend.__getstate__`). Both are valid Python; only opencode's path satisfies the gold test. On psf/requests claw wins the inverse case: claw's `current_req = req` pointer in `SessionRedirectMixin.resolve_redirects` correctly threads method changes across multiple redirects, while opencode tracks a parallel `effective_method` variable that doesn't propagate. The pattern is **per-instance**, not per-scaffold.

2. **Model-helper file noise broke scoring on edge cases — fixed at the score layer.** Both scaffolds write reproducer/debug scripts at `/testbed/` root (`reproduce_bug.py`, `test_fix.py`, `comprehensive_test.py`, `debug_*.py`). On `psf__requests-2317` opencode's actual code edit (`builtin_str(method)` → `to_native_string(method)`) was correct and identical to claw's resolved version, but opencode also added two new test files at root → pytest collected them, hit an import error, SWE-bench's `get_eval_report` couldn't parse the malformed test output, `run_instance` marked the instance `error`. [`evals/swebench/filter_predictions.py`](evals/swebench/filter_predictions.py) (wired into `score_docker.py` as `--filter-helpers`, default ON) now strips new root-level helper files and `.claw/.sandbox-*` dirs from each prediction before the SWE-bench harness sees it. The original `predictions.jsonl` stays untouched as the rollout receipt; the harness reads `scores-docker/predictions.filtered.jsonl`. On coder-30b-eval × opencode the filter dropped 548 helper sections across 300 predictions (36% byte reduction).

## Scope

This rig owns **all evals + AWQ/INT4 model calibrations** end-to-end. Ampere has native INT4 / AWQ_Marlin acceleration, so the GPTQ → CT → AWQ recalibration pipelines that produce the `mattbucci/*-AWQ` checkpoints land here; SWE-bench Lite / Verified bake-offs and capability sweeps for every supported preset also run here. **FP8 calibration work lives with [R9700](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — RDNA4 gfx1201 has native FP8 weight acceleration and the headroom to run FP8 recipes that don't pay off on Ampere. Eval results from this rig still inform their FP8 requant priorities, and both stacks publish under `mattbucci/*` with format suffixes.

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner (native gfx1201 FP8 weight acceleration); RDNA4/ROCm serving stack. Shares cross-stack recipes (FP32-softmax patch 011 originated there, the CT→native AWQ converter saved us 13h on Qwen3.6-35B). We push bake-off + capability findings back into their README.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal-plumbing assumptions.

3090 stack on **v0.5.12** (22 patches, rebased 2026-05-26); R9700 still on v0.5.11 (15 patches, 8 shared content).

## Current Focus

**Single-user 256K context across all supported models is the primary serving target.** Multi-user throughput is secondary. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable.

**Hard constraint: preserve thinking + vision + video across every calibration.** Past recals silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py` (basic + thinking + image + video; `--skip-video` for image-only models like Devstral).

**qwen36 → AWQ-Marlin rebuild — DONE (2026-05-26).** The qwen36 default was a GPTQ W4A16 **compressed-tensors** checkpoint (the dir name `…-AWQ-CT…` is a misnomer — it is *not* AWQ); its CT per-expert MoE format hit a v0.5.12 loader bug (`KeyError: experts.w2_weight_packed`; patch 028 maps AWQ suffixes, not CT's `weight_packed`). Rebuilt from the BF16 base as native **AWQ-Marlin** (fresh GPTQ W4A16, `moe_calibrate_all_experts=True`, thinking_vision recipe → `convert_moe_ct_to_awq` → BF16 vision-tower merge). **5/5 PASS** (basic + tool_call + thinking + vision + video) at TP=2/256K; serves on `--quantization awq_marlin` like the rest of the fleet. Shipped to `mattbucci/Qwen3.6-35B-A3B-AWQ`.

**End goal:** per-model bake-off matrix across `{opencode, claw-code, little-coder}` × all coder-class models on SWE-bench Lite at 256K, then SWE-bench Verified on the top 1-2 finalists. Driver: [`evals/swebench/run_model_cycle.sh <preset>`](evals/swebench/run_model_cycle.sh) handles launch → 3 rollouts → audit → reroll → score per preset.

Reference throughput: **Qwen3-30B REAM AWQ 262K @ 107 tok/s** (TP=2, 9.3 ms TPOT — `benchmarks/qwen3-30b-ream/long-context-v0511.json`); **Qwen3.6-35B-A3B AWQ-CT 256K @ 31 tok/s flat** thinking+vision (`benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`).

## Known Issues (open)

- **AWQ scales audit — the qwen36 "144 rare-expert under-cal findings" were MISDIAGNOSED (corrected 2026-05-26).** The 144 flagged `gate_proj`/`up_proj` "zero scales" are **inherent base-model expert sparsity**, not under-calibration: in `Qwen/Qwen3.6-35B-A3B` BF16, ~50-72% of those experts' gate/up channels are already `7.8e-38` (structural zeros). The CT format stores them as the bf16 minimum; the fp16 AWQ scale flushes them to 0 — faithful, harmless (those channels contribute ~0). `moe_calibrate_all_experts=True` does NOT change them (the all-experts rebuild reproduced the exact same 144). The model serves coherently and validates 5/5. **`check_awq_scales.py` over-flags here** — a zero scale that faithfully represents a dead base-model channel is benign; only zero scales over *live* base weights indicate a real defect.
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** 164 ALL-ZERO scale tensors from an empty `ignore` list. v3b build (regex `ignore`) is the shipping checkpoint at [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` produces gibberish across 4 distinct sources → upstream weight-mapping bug. Narrative in [`patches/README.md`](patches/README.md).
- **v0.5.12 fleet capability sweep — 3 open gaps.** Full matrix at [`benchmarks/quality/fleet-capability-v0512-2026-05-26.md`](benchmarks/quality/fleet-capability-v0512-2026-05-26.md). Passing 5/5 (thinking+tool+vision+video): `qwen36`, `qwen35-moe`, `qwen36-dense`, `gemma4`, `gemma4-31b`; `qwen3-vl-32b` 4/4; coder presets OK (text). Open:
  - **`devstral` tool-calling not emitted.** Model echoes the prompt (degenerate) instead of a `tool_call`. Not a template gap — the custom `devstral_chat_template.jinja` renders `[AVAILABLE_TOOLS]` and basic+vision pass; needs a deeper prompt-render trace. vision OOM is only on `devstral-long` at MEM=0.97.
  - **`qwen36-ream` vision degraded/unstable** (REAM-merge alignment drift; thinking + tool-calling are solid — it's the coding-eval leader).
  - **`qwen3-ream` checkpoint absent** — preset points at `Qwen3-30B-Instruct-2507-REAM-AWQ`, not on disk.
- **Qwen3.5-27B DeltaNet stuck at 32K.** DeltaNet TP replication forces 19 GB/GPU. Use `qwen3-ream` for long-context DeltaNet workloads.
- **60B+ models don't fit.** Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed the 48 GB total at MoE-AWQ.
- **Per-preset piecewise CUDA graph disables.** `coder-reap` / `coder-reap-25b` (cold-launch detokenizer hang); `qwen35-moe` / `qwen36` (DeltaNet+MoE+mamba_cache); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced). Reasons in launch.sh comments.
- **Model-scaffold tool-call format compatibility.** Coding harnesses (opencode / claw-code / little-coder) consume the SGLang OpenAI-compat endpoint's `tool_calls` field. SGLang only emits structured `tool_calls` when the preset passes `--tool-call-parser <fmt>` matching the format the model produces. Without the flag the model's raw `<function=NAME>...</function>` XML (qwen3-coder), `<tool_call>{json}</tool_call>` (qwen25), `[TOOL_CALLS]` (mistral) or Gemma `<\|tool>` is served as plain assistant text and the harness silently drops it — no edits applied. Audited all 19 presets against their chat templates (2026-05-13): 15 were missing flags. Mapping now: Qwen3-Coder + Qwen3.5/3.6 (incl. VL-REAP, dense, MoE, REAM) → `qwen3_coder`; Qwen3-VL non-coder + Qwen3-30B-Instruct REAM → `qwen25`; Devstral (Mistral arch) → `mistral`; Gemma 4 → `gemma4`. Runtime-validated end-to-end on qwen36 2026-05-13: request with `tools=[get_weather]` returns `finish_reason: tool_calls` with structured `function.arguments`.

## Suggested next

3090 owns **all evals + AWQ/INT4 calibrations** end-to-end (Ampere INT4 acceleration). FP8 recipes live with R9700 (RDNA4 native FP8). AWQ recipe-side fixes and requant runs land in this repo first.

- **Per-model eval cycles** ([`run_model_cycle.sh <preset>`](evals/swebench/run_model_cycle.sh)): `qwen36`, `qwen36-ream`, `qwen35-moe`, `qwen36-dense`, `coder-30b-ream`, `coder-reap-25b` (R9700 in-house refresh), `devstral`, `gemma4`. Each cycle: full 300-inst × 3 scaffolds + audit + reroll + score. Estimated 8-18h per preset.
- **SWE-bench Verified (500-task)** on the top 1-2 finalists once the matrix is settled.
- **Qwen3-VL-30B MoE loader fix** — gibberish across 4 sources; upstream weight-mapping bug. Non-coder, lower priority.
- **Devstral-24B-AWQ HF mirror** at `mattbucci/Devstral-24B-AWQ` after TP=2 validation.

Performance / post-bake-off:
- `qwen3-ream` + `coder-30b` at TP=2 with piecewise CUDA graph re-enabled (currently `--disable-piecewise-cuda-graph`; test if conditional-on-$TP lifts 107 tok/s @ 250K and 180 tok/s @ 16K).
- Gemma 4 multi-attention-backend A/B at TP=2 once v0.5.12+ relaxes the head_dim=256 + Ampere FP8 incompat that forces triton-attn today.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.12, apply patches, create conda env

# TP=2 / 256K presets (matrix standard):
./scripts/launch.sh qwen3-ream              # 262K @ 107 tok/s — REAM merged MoE, 96 experts
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE CT — 256K @ 31 tok/s, thinking+vision
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — DeltaNet+attn
./scripts/launch.sh coder-30b               # Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh coder-reap-25b          # Coder-REAP-25B MoE AWQ-Marlin — 256K @ 109 tok/s
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — 131K @ TP=2
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AWQ (thinking+image+video, 256K)

# TP=2:
./scripts/launch.sh devstral-long           # Devstral-24B at 217K — TP=2 only (Dense AWQ create_weights prealloc OOMs on TP=1)
./scripts/launch.sh devstral                # Devstral-24B 131K default

python scripts/eval/validate_capabilities.py --port 23334                 # auto-skips thinking/vision/video per preset
python scripts/eval/test_capabilities_all.sh                              # sweep across all AWQ presets
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

Reference host (current 3090 rig as of 2026-05-17 motherboard swap):

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 3090 (24 GB each, 48 GB total) — NVLink bridge present (`nvidia-smi topo -m` reports `NV4` ≈ 56 GB/s aggregate) |
| CPU | AMD Ryzen 9 7900 (12C/24T, max 5.48 GHz, Zen 4 AM5) |
| RAM | 2× 32 GB DDR5-6000 (64 GB total; 62 GB usable in OS) |
| Motherboard | MSI MPG B650I EDGE WIFI (MS-7D73, mini-ITX, AM5) |
| Storage | 2× Crucial P5 Plus 2 TB NVMe (`nvme0n1` = root, `nvme1n1` = `/data` models) |
| Chassis fans | Corsair Commander Core XT (controlled via `liquidctl`) |
| OS | EndeavourOS (Arch-based, rolling) |
| Kernel | `linux-zen-p2p` 6.18.0 (custom zen variant with NVIDIA-driver P2P-BAR1 patches) |
| NVIDIA driver | 595.71.05 (`nvidia-open-dkms`) / CUDA toolkit 13.2 |

Both 3090s sit at PCIe Gen4 with the NVLink bridge spanning them. NCCL selects `P2P/IPC` transport (NVLink + peer-to-peer CUDA IPC, not host-mediated sockets) once the driver/kernel combination below is in place.

### Kernel and driver

- `linux-zen`-family kernel (the host currently runs `linux-zen-p2p` 6.18.0), not stock `linux` — the stock Arch kernel + the open NVIDIA module hard-locked the host repeatedly under sustained TP=2 / 256K bake-off load on the prior chassis. The zen patchset's scheduler and IO tuning eliminated the recurrence; the `-p2p` variant additionally carries community patches that re-enable consumer-Ampere PCIe-BAR1 P2P so `nvidia-smi topo -m` reports `NV4` instead of `PHB` on this AM5 platform.
- `nvidia-open-dkms` (not `nvidia-open`) — DKMS rebuilds the module for every kernel that has headers installed, so the same NVIDIA driver version covers both `linux` and `linux-zen`.

### Cooling and power profile (load-bearing)

Two systemd units hold a cooling profile that's required for the bake-off to survive multi-hour runs on this chassis. The DDR5 SPD sensors crossed `ALARM HIGH` (55 °C) under stock cooling + default 350 W per 3090; that correlated with random Python heap corruption / kernel BUGs / hard resets. The profile below stays inside spec under sustained TP=2 inference.

| Unit | Action |
|------|--------|
| `gpu-cooling.service` | Boot oneshot. Enables NVIDIA persistence mode, sets each 3090's power limit to **260 W** (down from default 350 W), pushes Corsair Commander Core XT case fans to 100% via `liquidctl`, seeds a 75% GPU fan floor via NVML (`nvmlDeviceSetFanSpeed_v2`). |
| `gpu-fan-curve.service` | Long-running NVML daemon. Polls each GPU's temperature every 4 s. Fan duty = 75% below 60 °C, linear ramp to 100% between 60 °C and 80 °C, 100% at 80 °C+. Drives every fan on every GPU to the max-temp duty (one card heating up pulls all fans). |

The fan curve runs through NVML rather than hwmon — consumer Ampere on the open NVIDIA driver does not expose `pwm*` endpoints under `/sys/class/hwmon` for the GPU itself, only for the NVMe SSDs and SPD modules. NVML's `SetFanSpeed_v2` works as root.

Scripts are tracked in this repo under [`systemd/`](systemd/). Install them with:

```bash
sudo pacman -S --needed python-nvidia-ml-py
sudo install -m 0755 systemd/gpu-cooling.sh   /usr/local/bin/gpu-cooling.sh
sudo install -m 0755 systemd/gpu-fan-curve.py /usr/local/bin/gpu-fan-curve.py
sudo install -m 0644 systemd/gpu-cooling.service   /etc/systemd/system/
sudo install -m 0644 systemd/gpu-fan-curve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-cooling.service gpu-fan-curve.service
```

Verify with `nvidia-smi --query-gpu=power.limit,fan.speed,temperature.gpu --format=csv` (expect 260 W limit, ~75% fans idle).

Ampere consumer cards do not expose VRAM junction temperature, so the GPU-core temp shown by `nvidia-smi` is an underestimate of the real thermal pressure. 260 W picked to leave inference throughput headroom (a 256K coder-30b cycle pulls only ~245 W per card under steady-state decode, so the cap costs no throughput) while still cutting peak heat ~25%.

## Model Support

Single-user tok/s measured at the max-context value in the table. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **107** | 9.3 ms | `qwen3-ream` | TP=2 / 262K, 183 tok/s @ 1K → 107 tok/s @ 250K. Receipt: `benchmarks/qwen3-30b-ream/long-context-v0511.json`. |
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE (256 exp, VL) | **262K** | **31** | 32 ms | `qwen36` | TP=2 / 256K, **rebuilt from BF16 as native AWQ-Marlin 2026-05-26** (replaces the CT default that regressed on v0.5.12). **5/5 PASS** basic+tool_call+thinking+vision+video on v0.5.12. Fresh GPTQ W4A16 (all-experts, thinking_vision) → convert→AWQ → BF16 vision-tower merge. Receipt: `benchmarks/quality/qwen36-awq-marlin-rebuild-v0512.json`. |
| **Qwen3.6-27B AWQ** | Dense + DeltaNet | **131K** | **21** | 47 ms | `qwen36-dense` | R9700 self-cal at `mattbucci/Qwen3.6-27B-AWQ`. 4/4 PASS at TP=2; bakeoff matrix validation pending. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **131K** | **40** | 25 ms | `qwen3-vl-32b` | R9700 self-cal at `mattbucci/Qwen3-VL-32B-AWQ`. TP=2: 68 → 50 → 40 tok/s @ 1K/65K/131K, 3/3 PASS. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **50** | 19.9 ms | `devstral-long` | TP=2 / 217K, basic PASS, decode 59 → 50 tok/s @ 1K/200K. Vision OOMs at MEM=0.97 (preset bakes `--skip-server-warmup`). Text-only path clean. |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | TP=2 only — same Dense 24B prealloc constraint. |
| Coder-REAP-30B AWQ-Marlin | MoE (96 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | TP=2 / 256K, R9700 in-house rebuild from upstream BF16 (96 experts/layer, GPTQ W4A16 + `moe_calibrate_all_experts=True`). Replaced Cerebras pre-pruned 2026-05-14. |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | TP=2 / 16K, 187 → 180 tok/s @ 1K/16K. Original AWQ-Marlin layout (vs `coder-30b-eval` which is CT). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | TP=2 / 262K, decode flat 30.5 tok/s across 1K-250K, 4/4 PASS. R9700 cross-validated 4/4 on RDNA4. |
| **Gemma 4 31B Dense AWQ** | Dense | **256K** | ~22 | ~50 ms | `gemma4-31b` | TP=2 / 256K, **in-house BF16→GPTQ→AWQ rebuild 2026-05-27** (LM INT4, vision_tower + embed_vision kept FP16). **5/5 PASS** basic+tool+thinking+vision+video on v0.5.12 — replaces AutoRound (text-only cal left vision hallucinating). Ships to `mattbucci/gemma-4-31B-AWQ`. Receipt: `benchmarks/quality/gemma4-31b-awq-rebuild-v0512.json`. |
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
cd components/sglang && git checkout v0.5.12
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e .
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.12 + 22 local patches (`ls patches/*.patch \| wc -l`) |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.58) / cu130 wheel |
| NCCL | bundled with torch 2.11 (P2P over NVLink) |
| FlashInfer | 0.6.11.post1 (v0.5.12 pin) |
| transformers | 5.6.0 (v0.5.12 pin) |
| sglang-kernel | 0.4.2 |
| compressed-tensors | 0.15.0.1 |

## Patches

22 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.12. Notable:
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

> Cross-team from R9700 (2026-05-25): 256K AWQ smoke on RDNA4 TP2 — Qwen3.5-27B dense boots 256K (3/4); Qwen3.6-35B-A3B OOMs at mem-frac 0.85 (weights fill 31/32GB before KV, retest 0.75); 28B-REAP-AWQ KeyError experts.w2_qweight; coder-30B-AWQ is CT-mislabeled→your CT-w2 TP2 narrow bug. Coder-30B/Gemma4/REAP all 256K-native, our presets capped low.

> R9700 (2026-05-25): 28B-REAP/REAM-A3B/VL-REAP 256K fail = shipped per-expert-unfused (experts.0.down_proj) + language_model. prefix, loader needs fused experts.w2_qweight. fuse-convert needed; fused ships (coder-30b-REAM) boot 256K fine.

- **R9700 update (2026-05-25): coder gibberish dtype-exonerated.** fp16 + bf16 both gibberish; expert0 dequant + convert_awq_tensor repack bit-exact vs casper ref. Not weights/bind/dtype/kernel/topk. Narrowed to live fused scale-group / silu split. Your marlin avoids wna16 so clean. Verifiers: scripts/debug/moe_wna16_expert0_dequant.py.
