# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Cross-team activity (last ~2 weeks)

R9700 (RDNA4) and M4 (Apple) sister teams ship findings into our repo. Per-day forensic narrative + closed-item history in [`patches/README.md`](patches/README.md) and `git log -- README.md`.

> **📢 Cross-team note from R9700 (2026-05-13) — `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` rebuilt in-house, replaces broken 2026-04-29 ship.** Built end-to-end from upstream `Qwen/Qwen3-Coder-30B-A3B-Instruct` BF16 via a new homegrown pure-pytorch REAP implementation ([`scripts/quantize/run_reap.py`](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/scripts/quantize/run_reap.py) — no vLLM dependency, sidesteps Cerebras's vllm==0.10 pin and accelerate single-GPU placement OOM on R9700). Saliency `S_{L,E} = Σ_t (gate_t × ‖down_proj_E(x)‖₂)` accumulated over 1024 evol-codealpaca samples; 128 → 96 experts per layer (per-layer survivor lists, NOT uniform). Then GPTQ W4A16 group_size=128 calibrated 1024 × 2048 with `moe_calibrate_all_experts=True` on code+thinking mix (40% evol-codealpaca, 25% AM-Thinking, 20% NuminaMath, 15% UltraChat). 2 audit-class scale flags at `l1.exp.18.{gate,up}_proj` (~52% zero — same structural pattern as the shipped Coder-30B-REAM-AWQ). Smoke PASS: basic Q&A + clean memoized Fibonacci code-gen. 13 GB / 7 shards. **Same `coder-30b-eval` Ampere preset works** (just swap MODEL=). Cross-team validation request: run your SWE-bench Lite or opencode bakeoff against this fresh REAP variant vs the existing `Coder-REAP-25B-A3B-AWQ` to see if the 30B→96-expert (in-house) variant matches or beats the Cerebras 25B at the bakeoff matrix. HF commit `d09a18c`.
>
> **📢 Cross-team request from R9700 (2026-05-12) — validate `mattbucci/gemma-4-31B-AWQ` on Ampere.** New in-house build: end-to-end calibration of `google/gemma-4-31b-it` BF16 via `balanced_thinking_vision` recipe (replaces the AutoRound repack). Phase 2 audit clean (0/410 scale flags); R9700 validator: basic ✅, thinking ✅ (460 tok finish=stop), **vision ❌ HSAIL 0x1016 in `torch_native_backend.py:332 forward_decode` mid-decode**. Need Ampere validation to determine if the vision crash is RDNA-specific (HSAIL is ROCm-only error code, but root cause may be Gemma 4 31B Dense long-decode + SDPA interaction that affects both stacks). Run `MODEL=<your-models-dir>/hf-mattbucci/gemma-4-31B-AWQ ./scripts/launch.sh gemma4-31b` (or your equivalent preset) and `python scripts/eval/validate_capabilities.py --port <port>`. If 3090 vision PASSES → RDNA4-only crash; if 3090 also crashes → shared upstream issue; if 3090 hallucinates → Gemma 4 31B Dense vision is upstream-degraded. Reply via push to R9700 README's Cross-team activity section.

- **Both stacks at v0.5.11.** 3090: 19 patches, R9700: 15; 8 share content cross-stack.
- **M4 (2026-05-13):** Ported our probe trio (`probe_thinking`/`probe_vision`/`probe_codegen`) and surfaced a **VLM image regression** on the v0.5.11 MLX stack — Devstral now fabricates content for a red-circle-on-white prompt (`"A diagram of a circular flow chart..."`) where the April 18 stack returned `"I see a red circle with a black outline."`. Qwen3.5-9B-8bit reports `"a solid image of a pale pink color"` — different signature, same conclusion (vision tower features degraded). Important cross-stack data point: `validate_capabilities.py:check_vision` keyword grep passed both fabricated responses because "circle"/color words appeared — the **probe trio is the gate**, the keyword validator is not sufficient. M4 cross-stack codegen confirm: `coder-30b-DWQ` STRONG 8/8 matches our `coder-reap-25b` baseline.
- **M4 (2026-05-13 evening) — VLM regression FIXED + Gemma 4 radix-cache root-caused.** (a) The VLM image regression was a v0.5.11 rebase footgun: the original Apr-18 patch 010 (`pixel_values` plumbing tp_worker → prefill → shim) didn't re-apply and the patch slot was silently reused for an unrelated change. M4's new [patch 013](https://github.com/mattbucci/m4-sglang-inference/blob/main/patches/013-mlx-vlm-pixel-values.patch) restores the plumbing + also forwards `mm_kwargs` (image_sizes for Pixtral/Mistral3, image_grid_thw for Qwen-VL, etc.) — Devstral + Qwen3.5-9B-8bit both probe_vision STRONG end-to-end. Same lesson as our own patch 030 (presharded-w2 detection) — read the patch number, not the file count, when validating a rebase. (b) Gemma 4 first-prefill crash root-caused to `MlxKVPool` assuming homogeneous attention shapes — Gemma 4 has 25 sliding @ (8,256) + 5 full @ (2,512) per-layer, pool was sized for the sliding shape but only full layers actually write to it. Workaround: `--disable-radix-cache` baked into both gemma4 presets. Fix-A/B/C documented at [`patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md`](https://github.com/mattbucci/m4-sglang-inference/blob/main/patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md). Worth flagging because 3090 + R9700 Gemma 4 calibrations are both downstream of this latent assumption — if your own pool layouts ever go heterogeneous, sample a real-write-path layer in `_get_attn_config`.

> **Recommended for coding tasks (scaffold-dependent):** `qwen36` (Qwen3.6-35B-A3B-AWQ-CT) via **opencode**. Post-fix partial 23-instance run (2026-05-13): **14/23 = 60.9%** vs Coder-30B's same-23 = 11/23 = 47.8% (+13pp on identical subset). qwen36 strict superset — resolves 3 instances coder-30b cannot (astropy-12907 separability_matrix gold patch, django-10914, django-10924); coder-30b resolves 0 that qwen36 cannot. Run was manually stopped at 23/300 because pace had slowed to ~7-10 min/pred (full-300 ETA ~30-40h). Headline 300-instance leader remains `coder-30b-eval` at **121/300 = 40.3%** opencode (`./scripts/launch.sh coder-30b-eval`) pending a complete qwen36 × opencode run.
>
> **Round-1 Docker-harness bake-off:** Coder-30B 40.3% (opencode) / 38.3% (claw) / 26.3% (little-coder, partial); REAP-25B 33.0% (claw); Qwen3.6-35B 0.3% (claw — round-1, root-caused 2026-05-13 to missing `--tool-call-parser qwen3_coder`). Parser audit fixed 15 presets ([commit `5fa80fb`](scripts/launch.sh)).
>
> **Post-fix smoke matrix (2026-05-13, same 5 astropy instances both scaffolds; partial 23-inst for qwen36):**
>
> | Preset | tool-call parser | claw-code | opencode |
> |--------|------------------|:---------:|:--------:|
> | `qwen36` (Qwen3.6-35B-A3B MoE, thinking) | `qwen3_coder` | 0/5 | **4/5 smoke · 14/23 partial = 60.9%** |
> | `qwen36-dense` (Qwen3.6-27B Dense, thinking) | `qwen3_coder` | 0/5 | 1/5 |
> | `qwen35-moe` (Qwen3.5-28B-A3B-REAP, thinking) | `qwen3_coder` | not run | 2/5 |
> | `qwen36-ream` (Qwen3.6-REAM-A3B MoE, thinking) | `qwen3_coder` | not run | 2/5 |
> | `qwen3-ream` (Qwen3-30B-Instruct REAM, non-thinking) | `qwen25` | 0/5 | 0/5 |
> | `coder-30b-ream` (Qwen3-Coder-30B REAM, coder-tuned) | `qwen3_coder` | 1/5 | 1/5 |
>
> **Findings:**
> - **Thinking-mode Qwen3.6 + qwen3_coder format → opencode works, claw fails.** Both qwen36 sizes show the same scaffold swing. Two qwen36 × opencode resolves are canonical gold patches (`cright[...] = right` for separability_matrix; `output_field[:] = chararray.replace(...)` for FITS D-exponent). Claw fails because the model spends tokens in `<think>` before committing a tool_call. Smaller dense variant resolves fewer absolute (1/5) but the same scaffold pattern.
> - **Qwen3-30B-Instruct REAM (non-thinking) fails BOTH scaffolds.** claw: `assistant stream produced no content` — model never streams back. opencode: model emits 97-8192 tokens of text but no tool_call extracted. The qwen25 chat template advertises tools but the Instruct-REAM model isn't trained on tool-call output for either scaffold. **This is a model-capability issue, not a scaffold or parser issue.**
> - **Cross-cell consistency check:** instance 14365 (qdp.py case-insensitivity) was unresolved across **all 4 qwen36-family cells** using the same incomplete `re.IGNORECASE` fix → model-understanding limit on that specific bug, scaffold-independent.
> - **Recommendation:** Qwen3.6 thinking-mode models → opencode. Coder-tuned models (Coder-30B at 40.3% opencode / 38.3% claw, REAP-25B) → either scaffold but claw has trained tool registry match. Qwen3-30B-Instruct REAM → not viable for tool-call-driven codegen evals.
>
> Per-cell receipts at [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/).
>
> **2026-05-09 v0.5.11 quality matrix** (eval framework MMLU runs 1-per-subject = 57 questions across MMLU's 57 subjects; `benchmarks/quality/*-v0511.json`):
>
> | Model | MMLU | HumanEval | LAB-Bench | Needle |
> |-------|:----:|:---------:|:---------:|:------:|
> | `coder-30b` | **91.2%** | 96.7% | 33.3% | ✓ to 4K |
> | `qwen3-vl-32b` | **91.2%** | 83.3% | **39.8%** | ✓ to 4K |
> | `mattbucci/gemma-4-21B-REAP-AWQ` | 80.7% | extractor-broken* | (n/a) | (n/a) |
> | `coder-reap-25b` | 77.2% | 96.7% | (n/a) | ✓ to 65K |
>
> *Gemma 4 HumanEval extractor-broken; codegen-probe STRONG 8/8 is the cleaner signal there. Codegen-probe (`probe_codegen.py`) STRONG 8/8 across 7/7 production presets on v0.5.11; PARTIAL on `qwen3-ream` (generalist, not coder-tuned). Harness: opencode v1.14.25 headless, 256K, scored locally. Resolved-rate where tests actually ran: 88/236 = 37.3%. Artifacts: `evals/swebench/runs/coder-reap-25b-lite/`. Bake-off continues — see Suggested next §B.*

**Note on `reasoning_tokens` reporting (2026-05-14):** opencode's session logs show `tokens.reasoning: 0` for qwen36 rollouts. **Thinking is still engaged** — the model emits `<think>` blocks and SGLang extracts them into `delta.reasoning_content` in every streaming chunk. The `0` is a display artifact: SGLang returns `usage.reasoning_tokens` as a flat field; opencode parses OpenAI's o1 nested path (`completion_tokens_details.reasoning_tokens`). Direct curl confirms: qwen36 returns 256+ reasoning tokens by default on the same server opencode hit. The 40-65% headline rates reflect the model's real capability with full 256K + thinking.

**Note on common failure modes (2026-05-14):** Sampled 6 unresolved qwen36 × opencode partial-23 instances. Pattern: model identifies the right file/function and iterates heavily (10-54 bash calls, 2-5 edits, 30-96K session tokens), but **the fix is incomplete or symptom-only**. Example astropy-14182 (RST `header_rows`): model adds the kwarg to `__init__` correctly but leaves the `write()` method's `lines[1]` separator assumption unchanged → output RST has wrong separators when `header_rows` expands. Example astropy-14365 (qdp.py): `re.IGNORECASE` fix solves casing but misses the bad-continue-line handling. Industry baseline: 30B-class on SWE-bench Lite hits 35-50%; we're in that band, not under-performing infrastructure.

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**End goal: Docker-harness coding-eval rollouts at 256K context, single-user 1-request-at-a-time, across all coder-class models, scored against opencode AND little-coder scaffolds.** Everything below is a milestone on that path: serve correctly at 256K (sweep done), preserve calibration quality (audit + recal), unblock loader bugs (CT TP=2 narrow, Qwen3-VL-30B MoE), then ship Coder-30B / Qwen3.6-35B / Devstral-24B / Qwen3-30B-REAM through the Docker harness on the v0.5.11 stack. Final number: SWE-bench Verified on the top 1-2 finalists.

**Primary serving target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 107 tok/s** (9.3 ms TPOT, fresh prefill, TP=2 — `benchmarks/qwen3-30b-ream/long-context-v0511.json`). For thinking + vision at 256K on the same GPU budget: **Qwen3.6-35B-A3B AWQ-CT — 31 tok/s flat across 1K-250K** (TPOT 31.1-32.3 ms — `benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`).

### Active presets (TP=2 / 256K)

Both 3090s are online and all matrix work runs at TP=2 / 256K. Per-preset launch
notes (chat template, reasoning parser, piecewise-CUDA-graph disables) live in
the case comments inside [`scripts/launch.sh`](scripts/launch.sh). The model
table further down has measured tok/s + TPOT for each.

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json`.

## Known Issues (open)

- **AWQ scales audit (2026-05-07) — 8 suspicious-class rare-expert under-cal findings.** Affects `qwen36` native (144 flagged) and `qwen35-moe`; CT variant of Qwen3.6-35B-A3B is clean (0/31010). Quality-degraded but serves. Recipe-side fix tracked by R9700 (`moe_calibrate_all_experts=True`). Full report: [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** 164 ALL-ZERO scale tensors from an empty `ignore` list. v3b build (regex `ignore`) is the shipping checkpoint at [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` produces gibberish across 4 distinct sources → upstream weight-mapping bug. Narrative in [`patches/README.md`](patches/README.md).
- **Qwen3.5-27B DeltaNet stuck at 32K.** DeltaNet TP replication forces 19 GB/GPU. Use `qwen3-ream` for long-context DeltaNet workloads.
- **60B+ models don't fit.** Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed the 48 GB total at MoE-AWQ.
- **Per-preset piecewise CUDA graph disables.** `coder-reap` / `coder-reap-25b` (cold-launch detokenizer hang); `qwen35-moe` / `qwen36` (DeltaNet+MoE+mamba_cache); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced). Reasons in launch.sh comments.
- **Model-scaffold tool-call format compatibility.** Coding harnesses (opencode / claw-code / little-coder) consume the SGLang OpenAI-compat endpoint's `tool_calls` field. SGLang only emits structured `tool_calls` when the preset passes `--tool-call-parser <fmt>` matching the format the model produces. Without the flag the model's raw `<function=NAME>...</function>` XML (qwen3-coder), `<tool_call>{json}</tool_call>` (qwen25), `[TOOL_CALLS]` (mistral) or Gemma `<\|tool>` is served as plain assistant text and the harness silently drops it — no edits applied. Audited all 19 presets against their chat templates (2026-05-13): 15 were missing flags. Mapping now: Qwen3-Coder + Qwen3.5/3.6 (incl. VL-REAP, dense, MoE, REAM) → `qwen3_coder`; Qwen3-VL non-coder + Qwen3-30B-Instruct REAM → `qwen25`; Devstral (Mistral arch) → `mistral`; Gemma 4 → `gemma4`. Runtime-validated end-to-end on qwen36 2026-05-13: request with `tools=[get_weather]` returns `finish_reason: tool_calls` with structured `function.arguments`.

## Suggested next

3090 team focus is **evals and serving validation**. Recalibration and in-house model rebuilds live with the [R9700 sister team](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference) — see their Active work for the calibration/REAM/REAP backlog. Cross-stack ship validation status: [`SHIP_VALIDATION_REPORT_2026-05-11.md`](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/benchmarks/quality/SHIP_VALIDATION_REPORT_2026-05-11.md).

### A. Loader patches

- **A1.** Qwen3VLMoe `*ForConditionalGeneration` weight-mapping fix — gibberish across 4 distinct sources, source-independent. Non-coder, lower priority.

### B. Coding-eval bake-off — Docker harness, 256K, single-user

Bake-off runs detached via `systemd-run --user --scope`. Score runs in foreground per phase at 1 worker — no concurrent rollout+score (see Cooling and power profile for why).

**Round 1 (complete, 2026-05-12) — 5 cells, all 300-prediction except little-coder-partial 38:**

| Preset | opencode | little-coder | claw-code |
|--------|:--------:|:------------:|:---------:|
| `coder-30b-eval` | **40.3%** | 26.3% (38/300) | 38.3% |
| `coder-reap-25b` | — | — | 33.0% |
| `qwen36` | — | — | 0.3% ⚠️ scaffold-bug, fixed 2026-05-13 |

- **B1.** Re-roll `qwen36` × claw-code with `--tool-call-parser qwen3_coder` (fix in launch.sh 2026-05-13). Expected lift: 0.3% → 30%+ if it's purely scaffold-routing.
- **B2.** Expand opencode + little-coder columns across remaining dense presets (`qwen36-dense`, `qwen35-dense`, `qwen36-dense-ct`) and MoE presets (`qwen3-ream`, `qwen35-moe`, `coder-reap-25b`, `coder-30b-ream`, `qwen36-ream`, `qwen36-vl-reap`).
- **B3.** Diagnose why little-coder rollout stops at 38/300 (pipeline halts mid-run, not all-instances scored).
- **B4.** SWE-bench Verified (500-task) on top 1-2 finalists from B1+B2 — final headline number.

Per-cell receipts at `benchmarks/quality/bakeoff-<preset>-<scaffold>.json`.

### C. Cross-team / portability

- **C1.** Devstral-24B-AWQ HF mirror ship at `mattbucci/Devstral-24B-AWQ`. Validate at TP=2 first.

### D. Performance / optimization (post-bake-off)

- **D1.** `qwen3-ream` + `coder-30b` at TP=2 with piecewise CUDA graph re-enabled — current presets bake `--disable-piecewise-cuda-graph`; test if conditional-on-`$TP` lifts the headline (107 tok/s @ 250K and 180 tok/s @ 16K respectively today).
- **D2.** Multi-attention-backend Gemma 4 A/B at TP=2 — currently triton-attn forced (head_dim=256 + Ampere FP8 incompat). Revisit when v0.5.12+ lands.

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
| Coder-REAP-25B AWQ-Marlin | MoE (103 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | TP=2 / 256K, 183 → 109 tok/s @ 1K/250K. SWE-bench Lite: 29.3% on v1 host harness; v2 Docker matrix in flight (§B1). |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Auto-round W4A16 build of the same Cerebras REAP source. |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | TP=2 / 16K, 187 → 180 tok/s @ 1K/16K. |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | TP=2 / 262K, decode flat 30.5 tok/s across 1K-250K, 4/4 PASS. R9700 cross-validated 4/4 on RDNA4. Cerebras REAP base + `balanced_thinking_vision` recal (333 vision tensors retained). |
| Gemma 4 31B Dense | Dense | 16K | 22 | ~50 ms | `gemma4-31b` | basic+thinking real; vision validator-passes-but-degraded ("scattered red pixels"). R9700 recal in flight. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | basic+thinking real; vision content-aware on v0.5.11 (`'a solid red circle with a black outline'`) — earlier "scattered pixels" output was the v0.5.10 SGLang serving gap. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | 4K | — | — | — | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Audit clean. v2 build is a calibration disaster — see Known Issues. |

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

19 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.11. Originally rebased from a v0.5.10 set 24→13 (2026-05-07 commit `1655e46`); 002 cross-team port of R9700's qwen3_next AWQ weight_loader fix; 028 v0.5.11 gemma4_mm per-expert AWQ loader (R9700 `gemma4_causal.py` port); 029 qwen35 shared_expert_gate CT dequant; **030 fused_moe_triton presharded-w2 detection** (unblocks CT MoE at TP≥2 — qwen36 default switched to CT 2026-05-09); 034 sampler +/-Inf detection (R9700 backport, 2026-05-13). Per-patch narratives in [`patches/README.md`](patches/README.md).

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
