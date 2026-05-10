# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Cross-team activity (rolling, last ~2 weeks)

R9700 (RDNA4) and M4 (Apple) sister teams ship findings into our repo. Compact summary; per-day forensic narrative + closed item history live in [`patches/README.md`](patches/README.md) and `git log -- README.md`.

- **Gemma 4 vision lifted on v0.5.11 — fix was SGLang serving-side, not upstream model** (3090 sweep 2026-05-09 post env-upgrade). 2026-05-09 `probe_vision.py` content-quality matrix: 5/5 STRONG across `gemma4` (26B MoE HF mirror), `gemma4-31b` (Dense AutoRound), `qwen3-vl-32b`, `qwen35-tp1`, `qwen36-tp1`. Vision response examples: gemma4 26B `'A red circle with a black outline is centered on a white background.'`, gemma4-31b `'A red circle with a black outline on a white background.'`. The R9700 BF16-base probe that closed #66 as upstream Google was on v0.5.10 ROCm — one of the v0.5.11 deltas (patch 028 native, transformers 5.6, flashinfer 0.6.8.post1) restores content recognition. Cross-team alert pushed to R9700.
- **Qwen-family vision works content-aware end-to-end** across self-cal Dense (Qwen3.6-27B), self-cal MoE-REAP (Qwen3.5-28B), self-cal Dense (Qwen3-VL-32B), community Dense (QuantTrio Qwen3-VL-32B). All 4 multimodal Qwen builds + both Gemma 4 builds now pass content-aware on v0.5.11.
- **R9700 ships `mattbucci/Qwen3-VL-32B-AWQ`** (2026-05-05): self-calibrated from `Qwen/Qwen3-VL-32B-Instruct` BF16 base, balanced_thinking_vision recipe, 4/4 PASS on RDNA4 + 3/3 PASS on Ampere TP=1 / 4K. Preset `qwen3-vl-32b` repointed to this HF mirror.
- **`balanced_thinking_vision` is the production calibration recipe** (40% thinking + 60% non-thinking + 25% LLaVA images). Replaces 70%-thinking which triggered `</think>\nX\n</think>…` repetition loops. R9700 originator; defined in `scripts/quantize/calibration_datasets.py`.
- **`scripts/eval/check_awq_scales.py`** scans `*.scales` / `*.qweight` for all-zero / NaN / Inf. R9700 forensic tool (commit `e4aa012`); took 30s to find a 16h calibration disaster the validator missed. Run after every CT→native AWQ conversion. All locally-mirrored mattbucci AWQs pass; 5 have minor rare-expert flags (single MoE up_proj 50–83% zero; quality-degraded but serves).
- **Calibration regex-ignore rule** (both stacks lost ~16h on 2026-05-06): bare-string `ignore=["model.embed_vision"]` does NOT exclude descendant Linears. Always use `r"re:.*vision_tower.*"` / `r"re:.*embed_vision.*"` / `r"re:.*multi_modal_projector.*"`. Codified in CLAUDE.md.
- **Both stacks at v0.5.11 source + patches** (3090 patch rebase commit `1655e46`, R9700 commit `3466816`, both 2026-05-07; 3090 env upgrade 2026-05-09). 3090 24→13 patches at rebase, R9700 22→15. 11 dropped on 3090 as upstreamed. 8 patches share content cross-stack (003/011/012/018/023/024/025/026).
- **R9700 sweep of all 9 presets** (2026-05-08): 6/9 fully PASS, gemma4 26B is the only failure class on RDNA4 (sampler HSAIL — Ampere unaffected). Receipt: `evals/awq-audit-2026-05-08-r9700-sweep.md` in their repo.
- **Qwen3MoeExperts unfused-experts patch** (R9700 2026-05-08, commits `8c6ad38`/`46a417a`/`82070e2`): transformers 5.x silently drops per-expert checkpoint keys for Qwen3MoeForCausalLM models — port `patches/qwen3moe_unfused_experts.py` from R9700 if Ampere-side recalibration of Coder-30B-A3B / Qwen3-30B-A3B is queued.
- **R9700 ships `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`** (2026-05-09): first **in-house REAM merge from upstream BF16** (128 → 96 experts via Samsung SAIL `merge.py`, ~30B → ~23B params, 256×1024 code_thinking calib). 2 audit-class flags at `l1.exp.25.{gate,up}_proj`. **Ampere cross-validation 2026-05-09: validator 1/1 PASS + STRONG 8/8 codegen-probe** at TP=1 / 4K (`MODEL=$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ QUANT=moe_wna16 DTYPE=bfloat16 ./scripts/launch.sh coder-30b`). Cross-stack parity with R9700 ship confirmed.
- **R9700 build-from-scratch rule extension** (2026-05-09): we now also prune ourselves — REAM/REAP from upstream BF16 via our `run_ream_qwen3moe.sh`, not from third-party pre-pruned BF16. Three currently-shipped models source from 3rd-party prunes (Coder-REAP-25B = Cerebras, Qwen3.5-28B-REAP = Cerebras, VL-REAP-26B = atbender); rebuilding those in-house is now a tracked workstream — keep current versions live until in-house replacements validate. New prunes start from upstream only. No immediate Ampere-side action.
- **REAM merger broken for Qwen3MoeForCausalLM** (R9700 2026-04-30, **resolved 2026-05-08** via `patches/qwen3moe_unfused_experts.py`): `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` flagged DO NOT USE — Samsung SAIL `merge.py --merging none --saliency reap` produced gibberish on Qwen3MoE arch despite working on Qwen3_5MoeForConditionalGeneration. Working REAM merge now shipping at `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ` (above bullet). For REAP, the published Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` remains the recommendation until our in-house rebuild lands.
- **HF upload operational rule:** plain `hf upload <repo> <dir>` for repos ≤25 GB; `hf upload-large-folder` past 50 GB. Both stalled (XET worker deadlock at `committed: 0/X`) on prior pushes. Workaround: R9700's `scripts/quantize/upload_repo_per_file.py` does per-file uploads via `HfApi.upload_file()` so each commit is small + idempotent on retry.

For older detail (Qwen3.6-VL-REAP vision-tower-stripped warning, Qwen3.6-27B v3 vision regression history, LLaVA `data_files` pinning fix, Qwen3.5-28B REAP cross-validation, qwen36-27b DECODE_STEPS=8 fix on RDNA4, coder-30b DTYPE+QUANT fix on RDNA4): `git log --since=2026-04-25 --until=2026-05-08 -- README.md`.

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
> *Gemma 4 chat-template not extractable by HumanEval framework — codegen-probe STRONG 8/8 is the cleaner content-quality signal for that model. The Dense models (coder-30b, qwen3-vl-32b) lead on MMLU; REAP-pruned/MoE-DeltaNet variants trade ~10-18pts of MMLU for shorter weight footprint. All 5 models cleared the production-quality gate (validator + codegen-probe + vision-probe per the matrices above). **2026-05-09 v0.5.11 codegen-probe matrix (`probe_codegen.py` — 5/5 paren-balance + 3/3 interval-merge):** STRONG 8/8 across `coder-reap-25b` (HF mirror AWQ-Marlin) / `coder-reap` (local W4A16 auto-round) / `coder-30b` / `qwen3-vl-32b` / `gemma4` (26B MoE HF mirror) / `gemma4-31b` (Dense AutoRound) / `qwen35-moe` (DeltaNet+MoE-REAP thinking). PARTIAL on `qwen3-ream` (generalist text — STRONG is the coder-tuned bar). Production-quality content synthesis confirmed across 7 of 7 production-recommended models on the new v0.5.11 stack. The probe was extended to fall back to `reasoning_content` for thinking-mode models (commit `ad22c8f`).
>
> *Disclaimer: agent harness was [opencode](https://github.com/anomalyco/opencode) v1.14.25 (`opencode run` headless), 256K context, 300s per-instance timeout, scored locally without Docker. Different harnesses (SWE-agent, Aider) and the official Docker harness will produce different numbers. 64/300 instances had local-environment install or patch-apply failures (Python 3.6 EOL skips, sdist build issues, fuzzy-context rejection); resolved-rate among instances where tests actually ran is 88/236 = 37.3%. See `evals/swebench/runs/coder-reap-25b-lite/` for raw artifacts. This is the first model in a four-way bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B / Qwen3-30B-REAM still queued).*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**End goal: Docker-harness coding-eval rollouts at 256K context, single-user 1-request-at-a-time, across all coder-class models, scored against opencode AND little-coder scaffolds.** Everything below is a milestone on that path: serve correctly at 256K (sweep done), preserve calibration quality (audit + recal), unblock loader bugs (CT TP=2 narrow, Qwen3-VL-30B MoE), then ship Coder-30B / Qwen3.6-35B / Devstral-24B / Qwen3-30B-REAM through the Docker harness on the v0.5.11 stack. Final number: SWE-bench Verified on the top 1-2 finalists.

**Primary serving target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 107 tok/s** (9.3 ms TPOT, fresh prefill, **measured at TP=2** — 2026-05-09 v0.5.11 sweep at `benchmarks/qwen3-30b-ream/long-context-v0511.json`; matches the 2026-04-18 baseline). For thinking + vision at 256K on the same GPU budget: **Qwen3.6-35B-A3B AWQ-native — 31 tok/s flat across 1K-250K** (TPOT 31.1-32.3 ms, 2026-05-09 v0.5.11 TP=2 sweep at `benchmarks/qwen3.6-35b-a3b/v0511-tp2-flashinfer.json`). The historical decode regression that dropped 33 → 2.6 tok/s @ 250K closed on v0.5.11.

## In Flight

0. **Gemma 4 multimodal AWQ — pad-only bug fully resolved 2026-05-09.** Required two complementary fixes on v0.5.11: patch 028 (per-expert loader mapping) and the patch 023 detection upgrade (parses `quantization_config.ignore` to decide whether dense MLP gets quant_config or stays BF16). With both: v3b 21B-REAP and `mattbucci/gemma-4-26B-AWQ` HF mirror BOTH validate 4/4 PASS at TP=1 / 4K with content-aware vision/video. The earlier pad-only diagnosis was correct that patch 028 closed the loader path, but missed that the dense MLP was getting BF16 placeholders which never matched the AWQ checkpoint keys. `gemma4` preset repointed to HF mirror. v3b ready to ship to `mattbucci/gemma-4-21B-REAP-AWQ`. Patch 029 extended-CTX validation: 4/4 PASS at TP=1 / 4K against `hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT` (2026-05-08, 38.1s).
1. **SWE-bench Lite bake-off — Coder-REAP-25B done, three coders queued.** Coder-REAP-25B baseline shipped (29.3% / 37.3% on tests-ran, see banner). Next up: **Coder-30B → Qwen3.6-35B-A3B → Devstral-24B → Qwen3-30B-REAM**. Each rollout ~22h at 300s/instance × 256K ctx; scoring is ~30 min on the existing per-instance venv cache. Final pick → SWE-bench Verified (500 task) for the headline number on the top 1-2 finalists.
2. **Rollout v2: Docker-backed test-edit-test harness — Coder-30B 245/300 predictions written, scoring pending.** v1 was read-edit-pray (model couldn't `pytest` mid-iteration; 64/300 instances also failed local-env install/patch-apply scoring, marked unresolved). v2 runs opencode INSIDE the official swebench eval container (FROM `swebench/sweb.eval.x86_64.<inst>` + Node + opencode + ripgrep, host SGLang reachable via `--network=host`), so the model can run `pytest` against the exact env its fix is graded in, AND we score with the official Docker harness. **Rollout state on disk (last touched 2026-04-30, paused since):** `evals/swebench/runs/coder-30b-docker-v2/predictions.jsonl` has 245 entries (last: `sympy__sympy-14308`); 55 of 300 not yet rolled out. **Earlier partial scoring** — 25/62 = 40.3% (first 62) → **51/108 = 47.2% (first 111)** — projecting ~142/300 (47%) final, but final-number scoring against the Docker harness on the full 245 hasn't been re-run. **Comparison vs REAP-25B v1 (29.3%) is muddled by 3 axes** — model swap, rollout backend (host→Docker), AND scoring backend (local→Docker harness). Apples-to-apples requires both models on v2 (Docker rollout + Docker scoring); REAP-25B v2 will run after Coder-30B finishes. Until then, claims about "harness uplift" are not yet supported. Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout`.
3. **Scaffold A/B vs opencode.** Two challengers to bench on REAP-25B's 0/5 failing cluster after the bake-off: (a) [**little-coder**](https://github.com/itayinbarr/little-coder) — small-model-tuned harness (skill injection, thinking-budget cap, write-vs-edit invariant; built on `pi`) claiming **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1 / 23.82% TBench 2.0** — install `npm i -g little-coder`, OpenAI-compat against our `:23334`. (b) [**claw-code**](https://github.com/ultraworkers/claw-code) — Rust implementation of the `claw` CLI harness (build from source via `cargo build --workspace`; the crates.io stub is deprecated); takes `OPENAI_API_KEY` env var, no published benchmarks but the parity-harness/mock-service design is interesting. If either lifts ≥2/5 on REAP-25B's 0/5, promote to a second harness column in the bake-off.

### Cold-launch matrix (TP=1 / 24 GB single-card)

| Preset | TP=1 cold | Note |
|--------|:---------:|------|
| `qwen35` | ❌ | Bare command OOMs on TP=1 (`RuntimeError: Not enough memory` at preset CTX=32K + MEM=0.80 + 17.5 GB weights). Use `qwen35-tp1` variant instead — same Qwen3.6-27B-AWQ recal, CTX=4K, 4/4 PASS (2026-05-07 strict sweep). |
| `qwen35-tp1` | ✅ | TP=1-tuned: CTX=4K, MEM=0.85, MAX_RUNNING=1. **2026-05-07 strict-validator re-sweep (post-validator-readiness fix `a51c29d`): 4/4 PASS in 36.9s.** basic + thinking (1182 tok finish=stop) + vision (`'(reasoning)the user wants a short description of the image. 1. **identify the subject:** it's a red circle.'`) + video (`'a red circle moves from the left to the right.'`). Genuine content-aware on both modalities. (Confirms 2026-05-04 prior claim.) |
| `qwen36` | ❌ | Bare command OOMs on TP=1 at preset default CTX=262K. Use `qwen36-tp1` variant for single-card; `qwen36` is for TP=2 / 256K. |
| `qwen36-tp1` | ✅ | TP=1-tuned: CTX=2K, MAX_RUNNING=1, MAX_MAMBA_CACHE=4 (must be ≥4 to satisfy SGLang's mamba ratio division). **2026-05-07 strict-validator re-sweep: 4/4 PASS in 54.3s.** basic + thinking (1528 tok finish=stop) + vision (`'(reasoning)... it's a circle.'`) + video (`'a red circle moves to the right.'`). Genuine content-aware on both modalities. (Confirms 2026-05-04 prior claim.) |
| `qwen3-ream` | ✅ | **2026-05-07 sweep + same-day fix: 1/1 PASS in 0.0s** post-`--disable-piecewise-cuda-graph` bake-in. Was 0/1 / 120s timeout pre-patch — piecewise CUDA graph + awq_marlin MoE on Ampere TP=1 corrupts decode dispatch (GPU 100% util but ~1 tok/s). Preset now bakes `--disable-piecewise-cuda-graph` (TP=1 cold-fit only — TP=2 deployments still get piecewise for the headline ~74 tok/s @256K). |
| `coder-30b` | ✅ | **2026-05-07 sweep + same-day fix: 1/1 PASS in 0.2s** post-`--disable-piecewise-cuda-graph` bake-in. **2026-05-08 codegen-probe: STRONG 8/8 PASS** (`probe_codegen.py` — 5/5 paren-balance + 3/3 interval-merge with hand-rolled unit tests against generated Python; matches coder-reap-25b baseline). Same regression + fix as `qwen3-ream`. NB: `coder-reap` (auto-round quant on similar arch) was always fine, narrowing the failure to the awq_marlin MoE replay path on Ampere TP=1. |
| `coder-reap` | ✅ | **2026-05-07 sweep: 1/1 PASS in 11.6s** (basic only — text-only model, vision/video/thinking auto-skipped per validator's `TEXT_ONLY_MODELS` + `NON_THINKING_MODELS` lists). Qwen3-Coder-REAP-25B-A3B-AWQ on auto-round quant. Preset bakes `--disable-piecewise-cuda-graph` (line 104 — detokenizer hang at first prefill cold; ~5-10% TPOT cost). **2026-05-08 codegen-probe re-ran: STRONG 8/8 PASS** (5/5 paren-balance + 3/3 interval-merge — re-confirms the 2026-05-04 baseline). |
| `qwen3-vl-32b` MODEL=`mattbucci/Qwen3-VL-32B-AWQ` | ✅ | **2026-05-06 repointed from community QuantTrio to R9700's self-cal** (`hf-mattbucci/Qwen3-VL-32B-AWQ`, their task #58 ship `62fa459`) — `balanced_thinking_vision` recipe, native AWQ, 11 shards × ~2 GB = 20 GB. **2026-05-07 strict-validator re-sweep: 3/3 PASS** (thinking auto-skipped per non-thinking design). Vision: `'a solid red circle with a black outline is centered on a white background'`. Video: `'a red circle moves from the left side of the screen to the right side'`. Replaces community QuantTrio reference per build-from-scratch rule; override with `MODEL=$MODELS_DIR/Qwen3-VL-32B-Instruct-AWQ-4bit` if you want A/B against the original. |
| `gemma4-31b` MODEL=`gemma-4-31B-it-AutoRound-AWQ` | ⚠️ | **2026-05-03 repointed to R9700's HF mirror** (`hf-mattbucci/gemma-4-31B-it-AutoRound-AWQ`) — native AWQ + AWQ-Marlin on Ampere, `architectures: Gemma4ForConditionalGeneration` (R9700 task #63 metadata flip already shipped 2026-04-29). **5.2s weight load vs 30s CT** (~6× faster cold). **2026-05-07 strict-validator re-sweep: 4/4 loose-PASS in 15.2s — but vision DEGRADED.** Validator's keyword grep finds `red+round` in `'a small, fragmented red shape against a white background'` (NOT "red circle"); video says `'identical and do not show a video'` (it IS a moving circle). Same upstream-limitation pattern as 26B MoE — closed as Gemma 4 calibration / vision-tower limitation per task #66. Patches 024+025+026 still apply. Preset bakes triton-attn + KV_DTYPE=auto + disable-cuda-graph (head_dim=256 + Ampere FP8 incompat). |
| `qwen36-tp1` MODEL=`Qwen3.6-REAM-A3B-AWQ` | ✅ | **R9700 ship 2026-04-30**, pulled + validated 2026-05-02: **2/2 PASS** basic+thinking (vision auto-skipped — REAM stripped tower). Now launches cleanly via `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-REAM-A3B-AWQ ./scripts/launch.sh qwen36-tp1` (the new TP=1 variant bakes in the 2K-ctx + max-mamba-cache=4 settings). |
| `qwen36` MODEL=`Qwen3.6-VL-REAP-26B-A3B-AWQ` | ⚠️ | **R9700 ship 2026-05-02 (recal)**, pulled + validated same day: **2/3 PASS** basic+thinking; vision PARTIAL — saw 'circle','round' but missed 'red' (model said "white circles" when shown red circle). Confirmed 0 vision tensors in safetensors via `safe_open`; the partial keyword match is hallucination, not real vision processing. R9700 reported HSAIL on this; on Ampere it doesn't crash, just hallucinates — same outcome (vision broken by REAP-stripped tower), different failure surface. |
| `gemma4` (26B MoE) | ✅ | **2026-05-09 v0.5.11 sweep: 4/4 PASS in 15.0s — vision/video now CONTENT-AWARE.** Vision: `'a solid red circle with a black outline is centered on a white background.'` (was `'a red and black pixelated gradient fades into a white background'` on v0.5.10). Video: `'a red ball moves diagonally from the top left to the bottom right.'` (was `'this video is a static image'` on v0.5.10). Big improvement on env upgrade — see Known Issues entry "Gemma 4 vision quality" for full reframing. Patches 023 + 024 + 025 + 026 all still apply. |
| `qwen3-vl-moe` | ❌ | Closed: SGLang loader broken |
| `devstral` / `devstral-long` | ❌ | OOM at AWQ create_weights eager prealloc — TP=2 only |

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json`.

## Known Issues (open)

- **AWQ scales audit (2026-05-07) — 8 suspicious-class rare-expert under-calibration findings.** Affects `qwen36` (native, 144 flagged), `qwen36-tp1` source, and `qwen35-moe` — single rare expert per layer with `up_proj`/`gate_proj` scales 50-83% zero. Quality-degraded but serves; ~1-3% degraded tokens. CT variant of Qwen3.6-35B-A3B is calibration-clean (0/31010). Mitigation paths in **Suggested next §B** (recalibration backlog). Full report: [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** 164 ALL-ZERO scale tensors (vision_tower + MoE experts) — empty `ignore` list silently produced zero scales. v3b build at `~/AI/models/gemma-4-21b-REAP-AWQ-thinking-vision-v3b-2026-05-08` is the shipping checkpoint (regex `ignore` covering `vision_tower` + `embed_vision` + `multi_modal_projector`).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` produces multilingual gibberish across 3 calibration variants + community vLLM AWQ + self-cal native. Source-independent → upstream loader fix or weight-mapping trace required. Narrative in [`patches/README.md`](patches/README.md).
- **Qwen3.5-27B DeltaNet stuck at 32K.** DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **60B+ models don't fit.** Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed the 48 GB total VRAM budget at MoE-AWQ.
- **Devstral-24B Dense OOMs on TP=1.** Source weights load to ~23.48 GiB resident before `create_weights` runs, then the per-layer `torch.empty(input_size, output_size // pack_factor, dtype=int32)` allocation pushes past the 24 GB budget (`awq.py:518` / `compressed_tensors_wNa16.py:135`). `mem-fraction=0.97` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` don't close the gap. Needs an upstream loader change (lazy/streamed dest allocation) or `Devstral-Small`. TP=2 path serves cleanly via `devstral` / `devstral-long`.
- **Per-preset piecewise CUDA graph disables.** `coder-reap` / `coder-reap-25b` (cold-launch detokenizer hang); `qwen35-moe` / `qwen36` (DeltaNet+MoE+mamba_cache interaction); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced); `qwen3-ream` / `coder-30b` (TP=1 cold-fit awq_marlin MoE protection — costs ~5-10% TPOT at TP=2, see **Suggested next §G** for the TP=2 piecewise-on retest). Each launch.sh comment documents the specific reason.

## Suggested next

Comprehensive backlog grouped by area. Final destination is **Docker-harness coding-eval rollouts at 256K / single-user / 1-req-at-a-time** across opencode + little-coder scaffolds; everything below either unblocks that target or extends model coverage. Resolved entries from prior loops live in [patches/README.md](patches/README.md) under "Recent resolved items".

### A. Loader patches (unblocks current bugs)

- **A1. Qwen3VLMoeForConditionalGeneration loader fix** (Qwen3-VL-30B MoE AWQ). Loader produces multilingual gibberish across 4 distinct sources — source-independent → upstream weight-mapping bug. Either patch `Qwen3VLMoe*` weight loader or trace the divergence between `*ForCausalLM` (works) and `*ForConditionalGeneration` (broken) paths. Effort: ~3 days. Lower priority — non-coder model.
- **A2. Devstral-24B Dense lazy/streamed weight loader** (TP=1 OOM). Upstream change: stream `create_weights` per layer instead of pre-allocating the full int32 destination at load time. Effort: ~2-3 days, upstream PR. Lower priority — TP=2 path works.

### B. Recalibration backlog (closes audit findings)

Targets the 8 suspicious-class rare-expert under-calibration findings from `evals/awq-audit-2026-05-07.md`. Recipe template: `NUM_CALIBRATION_SAMPLES=1024+`, regex `ignore` for vision/router/shared_expert_gate, image-text fraction ≥25% with `drop_images=False` for multimodal models, `force_route_all_experts` for MoE (needs port to Qwen3.6 / Qwen3-Coder calibrators per R9700 task #5(b)).

- **B1. Qwen3.6-35B-A3B-AWQ recal** (l1.exp.214, 144 flagged native). Production `qwen36` model. CT clone is calibration-clean — recal would either match CT or beat it.
- **B2. Qwen3.6-VL-REAP-26B-A3B-AWQ recal** (l1.exp.166). Vision-stripped REAP base; rebuild from upstream BF16 (§C2) is the better fix.
- **B3. Qwen3-Coder-REAP-25B-A3B-AWQ recal** (l1.exp.22 gate AND up). SWE-bench Lite leader — quality lift directly affects the bake-off.
- **B4. Qwen3-Coder-Next-REAM-AWQ recal** (l47.exp.81). Larger model — won't fit at 48 GB anyway, deprioritized.
- **B5. Qwen3.5-28B-A3B-REAP-AWQ recal** if check_awq_scales flags any rare-expert findings on the v0.5.11 sweep build.

### C. In-house rebuilds (build-from-scratch rule)

Three currently-shipped models source from third-party prunes; rebuild from upstream BF16 via our `run_ream_qwen3moe.sh` keeps quality knobs (vision tower, router, DeltaNet ignore lists) under our control. R9700 owns the prune pipeline; 3090 owns Ampere-stack validation.

- **C1. Qwen3.6-35B-A3B clean text-only REAP** (R9700 task #60). Current `Qwen3.6-VL-REAP-26B-A3B-AWQ` has zero vision tensors at the BF16 layer (atbender already stripped). Rebuild via Samsung SAIL with vision tower retained.
- **C2. Gemma 4 26B REAM-or-REAP** (R9700 task #61). No in-house pruned variant exists; current `gemma4` preset uses the unpruned 26B-A4B-it base.
- **C3. Coder-REAP-25B from upstream BF16** — currently sources from Cerebras prune. Build-from-scratch rule extension 2026-05-09 puts this in scope; tracked as multi-week workstream. Keep current Cerebras-sourced ship live until in-house validates.

### D. Cross-team / portability

- **D1. R9700 v0.5.11 env upgrade — verify long-ctx regression also closes on RDNA4.** Already pushed advisory (R9700 README entry 8). When their upgrade lands, expect `qwen36` TP=2 to flip from 33→2.6 tok/s @ 250K to flat 30+ tok/s on flashinfer (Ampere 2026-05-09 receipt).
- **D2. R9700 cross-validation of `mattbucci/gemma-4-21B-REAP-AWQ`** — shipped 2026-05-09 (3090 commit `6bc20c5`). After R9700 ports patch 023 detection upgrade.
- **D3. Devstral-24B-AWQ HF mirror repoint (`mattbucci/Devstral-24B-AWQ`).** `devstral` preset currently points at local `Devstral-24B-AWQ-Marlin`. Validate at TP=2 + push HF mirror.

### E. Quality improvements

- **E1. Gemma 4 vision quality lift** — R9700 task #66 in flight (`drop_images=False` recalibration; expected complete ~2026-05-06+). On 3090: when their CT artifacts upload, run convert_moe_ct_to_awq + serve A/B against current. Quality target: content-aware vision (currently keyword-grep PASS but degraded — model says "scattered red pixels" not "red circle").
- **E2. Audio modality calibration for Gemma 4** — Google's BF16 base has zero audio_tower keys (per R9700 audit 2026-05-02 #4) so audio is upstream-blocked regardless. Track for future Gemma releases that ship audio_encoder weights.
- **E3. coder-reap (W4A16) decode regression at TP=2** — historical 46 tok/s @ 131K headline; v0.5.11 sweep deferred since `coder-reap-25b` (AWQ-Marlin variant of the same Cerebras source) already validates 109 tok/s @ 250K. Useful only if a deployed downstream needs the W4A16 format specifically.

### F. Coding-eval bake-off — Docker harness, 256K, single-user (the headline destination)

- **F1. Resume Coder-30B Docker rollout v2** (`evals/swebench/runs/coder-30b-docker-v2/`). 245/300 predictions written 2026-04-30, paused since. Score the existing 245 against the official Docker harness; finish the remaining 55 instances; report final number. Apples-to-apples comparison against REAP-25B v1's 29.3% requires REAP-25B to also run on v2 (next).
- **F2. Coder-REAP-25B v2** (Docker rollout + Docker scoring). Compares cleanly against F1 once both are on the v2 harness.
- **F3. Qwen3.6-35B-A3B Docker rollout v2.** Now that TP=2 / 256K serves at flat 31 tok/s decode + 4/4 PASS validator (2026-05-09 v0.5.11 ship), this is unblocked.
- **F4. Devstral-24B Docker rollout v2** (text-only at MEM=0.97 / TP=2, vision check skipped per the pixtral OOM Known Issue).
- **F5. Qwen3-30B-REAM Docker rollout v2.** Reference model — 107 tok/s @ 250K, fastest decode in the lineup.
- **F6. little-coder scaffold A/B on each F1-F5 finalist** — `npm i -g little-coder`, OpenAI-compat against `:23334`. Compare against opencode v1.14.25 baseline. If little-coder lifts ≥2/5 on the 0/5 failing cluster, promote to a permanent harness column.
- **F7. claw-code scaffold smoke** — Rust `claw` CLI from source (`cargo build --workspace`), `OPENAI_API_KEY` env. No published baselines; if it differentiates from opencode/little-coder on REAP-25B's failing cluster, evaluate further.
- **F8. SWE-bench Verified on the top 1-2 finalists from F1-F5.** 500-task. Final headline number.

### G. Performance / optimization (post-bake-off)

- **G1. qwen3-ream piecewise CUDA graph at TP=2** — current preset bakes `--disable-piecewise-cuda-graph` for TP=1 cold-fit safety. At TP=2 / 256K we measured 107 tok/s with piecewise off; the historical pre-bake number was the same. Test with piecewise on at TP=2 to see if the bake costs anything; if so, make the flag conditional on `$TP`.
- **G2. coder-30b same investigation** — 180 tok/s at TP=2 / 16K vs historical 193. Same piecewise bake. Same conditional fix candidate.
- **G3. Multi-attention-backend Gemma 4 A/B** at TP=2 — currently triton-attn forced (head_dim=256 + Ampere FP8 incompat). Upstream may add FlashInfer head_dim=256 support; revisit when v0.5.12+ lands.

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
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **107** | 9.3 ms | `qwen3-ream` | **Hits 256K target** — 1/1 PASS at TP=2 / 262K (2026-05-09 v0.5.11). 183 tok/s @ 1K → 107 tok/s @ 250K (`benchmarks/qwen3-30b-ream/long-context-v0511.json`). |
| **Qwen3.6-35B-A3B AWQ-CT** | DeltaNet+MoE (256 exp, VL) | **262K** | **31** | 32 ms | `qwen36` | **4/4 PASS** at TP=2 / 256K on the calibration-clean CT variant (2026-05-09 v0.5.11 + patch 030, 137.4s). **Decode flat 30.3-31.5 tok/s across 1K-250K** (`benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json`), within 1-3% of the native-AWQ TP=2 number on the same grid (`v0511-tp2-flashinfer.json` 30.9-32.2 tok/s; triton-attn `v0511-tp2-triton.json` 30.2-31.8 tok/s). CT is preferred default — 0/31010 audit-flagged scales vs native's 144 (rare-expert under-cal). Override to native: `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ QUANT=awq_marlin ./scripts/launch.sh qwen36`. |
| **Qwen3.6-27B AWQ (R9700 recal Apr-29)** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` (preset → `mattbucci/Qwen3.6-27B-AWQ`) | **4/4 PASS basic+thinking+vision+video** at `qwen35-tp1` TP=1 / 4K (2026-05-07 strict-validator sweep, 36.9s). R9700's `balanced_thinking_text` recal. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **131K** | **40** | 25 ms | `qwen3-vl-32b` | **3/3 PASS** at TP=2 / 131K (2026-05-09 v0.5.11 sweep). Decode 68 tok/s @ 1K → 50 tok/s @ 65K → 40 tok/s @ 131K (`benchmarks/qwen3-vl-32b/long-context-v0511-tp2.json`). Vision: `'a solid red circle with a black outline is centered on a white background'`. Video: `'a red circle moves from the left side of the screen to the right side'`. Preset repointed 2026-05-06 from community QuantTrio to R9700's self-cal `mattbucci/Qwen3-VL-32B-AWQ`. TP=1 / 4K cold-fit also passes (3/3, qwen3-vl-32b row in capability_check). |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **50** | 19.9 ms | `devstral-long` | **basic PASS** at TP=2 / 217K (2026-05-09 v0.5.11; vision OOMs at MEM=0.97 — text-only inference clean). 1/2 validator + decode 59 tok/s @ 1K → 50 tok/s @ 200K (`benchmarks/devstral-24b-awq/long-context-v0511-tp2.json`). Preset bakes `--skip-server-warmup` (added 2026-05-09 to dodge pixtral image_processor OOM during MEM=0.97 warmup). |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly. **TP=2 only** (same Dense 24B prealloc constraint as devstral-long). |
| Coder-REAP-25B AWQ-Marlin | MoE (103 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | **1/1 PASS** at TP=2 / 256K (2026-05-09 v0.5.11). Decode 183 tok/s @ 1K → 109 tok/s @ 250K (`benchmarks/coder-reap-25b/long-context-v0511-tp2.json`). SWE-bench Lite: 29.3% (88/300) on opencode v1 host harness; v2 Docker harness queued. AWQ-Marlin variant of the W4A16 build. |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working — auto-round W4A16 build of the same Cerebras REAP source. |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | **1/1 PASS** at TP=2 / 16K (2026-05-09 v0.5.11). Decode 187 tok/s @ 1K, 200 tok/s @ 4K, 180 tok/s @ 16K (`benchmarks/coder-30b-awq/long-context-v0511-tp2.json`). |
| Gemma 4 31B Dense | Dense | 16K* | 22 | ~50 ms | `gemma4-31b` | **basic + thinking VERIFIED (2026-05-03) at preset-default 16K**: `gemma4-31b-ctx16k-May03` entry. Thinking confirmed real (structured channel + correct arithmetic). **Vision validator-passes-but-degraded** — model says "scattered black and red pixels on a white background" instead of "red circle"; saw=red,round is a keyword-grep hit, not real recognition. *KV pool tight at 16K — only 1947 tokens after 20 GB weights at TP=1 / mem-fraction 0.92, so per-request input cap is ~1.9K. Drop to `--context-length 4096` to get a more usable KV budget, or run at TP=2 to unlock real 16K+ ctx. Triple-fix: patches 023+024 + BF16 default + local config arch flip to `Gemma4ForConditionalGeneration` (R9700 task #63 will retire the local edit). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | **4/4 PASS** at TP=2 / 262K (2026-05-09 v0.5.11, 272s). Decode flat 30.5 tok/s across 1K-250K (`benchmarks/qwen3.5-28b-moe/long-context-v0511.json`). R9700 cross-validated 4/4 PASS on RDNA4 (2026-05-03 18:51 PDT). Built from Cerebras's REAP base via `balanced_thinking_vision` recipe (256 samples × 2K, 18.22h CPU GPTQ + CT→AWQ). 333 vision tensors retained. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | **basic + thinking VERIFIED (2026-05-03) at 16K ctx** post-patches 023 + 024 + BF16 default. `gemma4-26b-ctx16k-May03` validator entry. Thinking probed via `skip_special_tokens=False`: separated `<\|channel>thought` channel, real step-by-step scratch work, correct answer 11 with intermediate 22. **Vision validator-passes-but-degraded** — "gradient of black and red pixels fades into a white background" is a keyword-grep hit on red/round, not real recognition of the red circle. Tower loads cleanly (no NaN); quality just isn't there yet vs Qwen3-VL baseline. 25880 KV tokens budget at TP=1 / mem-fraction 0.92. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | 4K* | — | — | — | **Shipped 2026-05-09** to [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ) (commit `6bc20c5`). Validates 4/4 PASS at TP=1 / 4K (12.0s) via the `gemma4` preset under v0.5.11 + patches 023 (detection upgrade) + 028 (per-expert loader). Content-aware vision (`'a solid red circle with a black outline on a white background.'`) + video (`'a red circle moves in a clockwise direction.'`). Audit clean (0/11725 scales + 0/11725 qweight). v2 build is a separate calibration disaster (164 ALL-ZERO scales) — DO NOT USE. |

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
