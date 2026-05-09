# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

## Cross-team activity (rolling, last ~2 weeks)

R9700 (RDNA4) and M4 (Apple) sister teams ship findings into our repo. Compact summary; per-day forensic narrative + closed item history live in [`patches/README.md`](patches/README.md) and `git log -- README.md`.

- **Gemma 4 vision lifted on v0.5.11 — fix was SGLang serving-side, not upstream model** (3090 sweep 2026-05-09 post env-upgrade). Vision: `'a solid red circle with a black outline is centered on a white background.'` (26B MoE) / `'a red circle with a black outline on a white background.'` (31B Dense). Video: content-aware ("moves diagonally", "from left to right"). The R9700 BF16-base probe that closed #66 as upstream Google was on v0.5.10 ROCm — one of the v0.5.11 deltas (patch 028 native, transformers 5.6, flashinfer 0.6.8.post1) restores content recognition. Cross-team alert pushed to R9700.
- **Qwen-family vision works content-aware end-to-end** across self-cal Dense (Qwen3.6-27B), self-cal MoE-REAP (Qwen3.5-28B), self-cal Dense (Qwen3-VL-32B), community Dense (QuantTrio Qwen3-VL-32B). All 4 multimodal Qwen builds + both Gemma 4 builds now pass content-aware on v0.5.11.
- **R9700 ships `mattbucci/Qwen3-VL-32B-AWQ`** (2026-05-05): self-calibrated from `Qwen/Qwen3-VL-32B-Instruct` BF16 base, balanced_thinking_vision recipe, 4/4 PASS on RDNA4 + 3/3 PASS on Ampere TP=1 / 4K. Preset `qwen3-vl-32b` repointed to this HF mirror.
- **`balanced_thinking_vision` is the production calibration recipe** (40% thinking + 60% non-thinking + 25% LLaVA images). Replaces 70%-thinking which triggered `</think>\nX\n</think>…` repetition loops. R9700 originator; defined in `scripts/quantize/calibration_datasets.py`.
- **`scripts/eval/check_awq_scales.py`** scans `*.scales` / `*.qweight` for all-zero / NaN / Inf. R9700 forensic tool (commit `e4aa012`); took 30s to find a 16h calibration disaster the validator missed. Run after every CT→native AWQ conversion. All locally-mirrored mattbucci AWQs pass; 5 have minor rare-expert flags (single MoE up_proj 50–83% zero; quality-degraded but serves).
- **Calibration regex-ignore rule** (both stacks lost ~16h on 2026-05-06): bare-string `ignore=["model.embed_vision"]` does NOT exclude descendant Linears. Always use `r"re:.*vision_tower.*"` / `r"re:.*embed_vision.*"` / `r"re:.*multi_modal_projector.*"`. Codified in CLAUDE.md.
- **Both stacks at v0.5.11 source + patches** (3090 patch rebase commit `1655e46`, R9700 commit `3466816`, both 2026-05-07; 3090 env upgrade 2026-05-09). 3090 24→13 patches at rebase, R9700 22→15. 11 dropped on 3090 as upstreamed. 8 patches share content cross-stack (003/011/012/018/023/024/025/026).
- **R9700 sweep of all 9 presets** (2026-05-08): 6/9 fully PASS, gemma4 26B is the only failure class on RDNA4 (sampler HSAIL — Ampere unaffected). Receipt: `evals/awq-audit-2026-05-08-r9700-sweep.md` in their repo.
- **Qwen3MoeExperts unfused-experts patch** (R9700 2026-05-08, commits `8c6ad38`/`46a417a`/`82070e2`): transformers 5.x silently drops per-expert checkpoint keys for Qwen3MoeForCausalLM models — port `patches/qwen3moe_unfused_experts.py` from R9700 if Ampere-side recalibration of Coder-30B-A3B / Qwen3-30B-A3B is queued.
- **R9700 ships `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`** (2026-05-09): first **in-house REAM merge from upstream BF16** ([`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), 128 → 96 experts via Samsung SAIL `merge.py` saliency=reap+grouping=ream+merging=logits+weights, ~30B → ~23B params). 256-sample × 1024-tok code_thinking calib; `check_awq_scales.py` returned 2 audit-class flags at `l1.exp.25.{gate,up}_proj` (~52% zero, validator + code-gen still PASS). Receipt commit `ce9a92b`. Ampere cross-validation requested — same flow as Qwen3.5-28B-REAP. Use `--quantization moe_wna16 --dtype bfloat16` (per-Linear AWQ + fp16 routes 128 experts × 48 layers × 3 projs as individual GEMVs and is much slower).
- **R9700 build-from-scratch rule extension** (2026-05-09): we now also prune ourselves — REAM/REAP from upstream BF16 via our `run_ream_qwen3moe.sh`, not from third-party pre-pruned BF16. Three currently-shipped models source from 3rd-party prunes (Coder-REAP-25B = Cerebras, Qwen3.5-28B-REAP = Cerebras, VL-REAP-26B = atbender); rebuilding those in-house is now a tracked workstream — keep current versions live until in-house replacements validate. New prunes start from upstream only. No immediate Ampere-side action.
- **REAM merger broken for Qwen3MoeForCausalLM** (R9700 2026-04-30, **resolved 2026-05-08** via `patches/qwen3moe_unfused_experts.py`): `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` flagged DO NOT USE — Samsung SAIL `merge.py --merging none --saliency reap` produced gibberish on Qwen3MoE arch despite working on Qwen3_5MoeForConditionalGeneration. Working REAM merge now shipping at `mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ` (above bullet). For REAP, the published Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` remains the recommendation until our in-house rebuild lands.
- **HF upload operational rule:** plain `hf upload <repo> <dir>` for repos ≤25 GB; `hf upload-large-folder` past 50 GB. Both stalled (XET worker deadlock at `committed: 0/X`) on prior pushes. Workaround: R9700's `scripts/quantize/upload_repo_per_file.py` does per-file uploads via `HfApi.upload_file()` so each commit is small + idempotent on retry.

For older detail (Qwen3.6-VL-REAP vision-tower-stripped warning, Qwen3.6-27B v3 vision regression history, LLaVA `data_files` pinning fix, Qwen3.5-28B REAP cross-validation, qwen36-27b DECODE_STEPS=8 fix on RDNA4, coder-30b DTYPE+QUANT fix on RDNA4): `git log --since=2026-04-25 --until=2026-05-08 -- README.md`.

> **Recommended for coding tasks: `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% on SWE-bench Lite** (`./scripts/launch.sh coder-reap-25b`). **2026-05-04/05 quick code-gen probe receipts (`scripts/eval/probe_codegen.py` 8 algorithmic test cases):** `coder-reap-25b` → STRONG 8/8 (TP=1 / 4K port 23358); `qwen3-vl-32b` (community QuantTrio Dense, multimodal) → STRONG 8/8 (TP=1 / 4K port 23365). Real code synthesis on Ampere across both a coder-tuned Cerebras-REAP build AND a generalist multimodal Dense model — coder-reap-25b is the recommended for code workflows, but qwen3-vl-32b is a viable backup that also handles vision in the same context.
>
> *Disclaimer: agent harness was [opencode](https://github.com/anomalyco/opencode) v1.14.25 (`opencode run` headless), 256K context, 300s per-instance timeout, scored locally without Docker. Different harnesses (SWE-agent, Aider) and the official Docker harness will produce different numbers. 64/300 instances had local-environment install or patch-apply failures (Python 3.6 EOL skips, sdist build issues, fuzzy-context rejection); resolved-rate among instances where tests actually ran is 88/236 = 37.3%. See `evals/swebench/runs/coder-reap-25b-lite/` for raw artifacts. This is the first model in a four-way bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B / Qwen3-30B-REAM still queued).*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill, **measured at TP=2** — `benchmarks/qwen3-30b-ream/long-context-262k.json`). At TP=1 / single-card the model still boots cold and serves cleanly (cold-launch matrix below) but headline tok/s and 256K context numbers depend on TP=2; second 3090 reinstall pending. For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## In Flight

0. **Gemma 4 21B-REAP — multimodal pad-only output bug resolved on v0.5.11.** v3b build is calibration-clean (audit passes 0/11725 scales+qweight, post-converter-fix `839e44b`). v0.5.10's `expert_params_mapping` only handled HF-fused 3D tensors so per-expert AWQ keys silently fell through to default-init. v0.5.11 + patch 028 closes the per-expert path. Validation pending post-rebuild; ship gate is `mattbucci/gemma-4-21B-REAP-AWQ` to HF + cross-stack RDNA4 cross-validation. The `gemma4` preset will repoint to `mattbucci/gemma-4-26B-AWQ` HF mirror once validated. Patch 029 extended-CTX validation already done: 4/4 PASS at TP=1 / 4K against `hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT` (2026-05-08, 38.1s).
1. **SWE-bench Lite bake-off — Coder-REAP-25B done, three coders queued.** Coder-REAP-25B baseline shipped (29.3% / 37.3% on tests-ran, see banner). Next up: **Coder-30B → Qwen3.6-35B-A3B → Devstral-24B → Qwen3-30B-REAM**. Each rollout ~22h at 300s/instance × 256K ctx; scoring is ~30 min on the existing per-instance venv cache. Final pick → SWE-bench Verified (500 task) for the headline number on the top 1-2 finalists.
2. **Rollout v2: Docker-backed test-edit-test harness — Coder-30B 245/300 predictions written, scoring pending.** v1 was read-edit-pray (model couldn't `pytest` mid-iteration; 64/300 instances also failed local-env install/patch-apply scoring, marked unresolved). v2 runs opencode INSIDE the official swebench eval container (FROM `swebench/sweb.eval.x86_64.<inst>` + Node + opencode + ripgrep, host SGLang reachable via `--network=host`), so the model can run `pytest` against the exact env its fix is graded in, AND we score with the official Docker harness. **Rollout state on disk (last touched 2026-04-30, paused since):** `evals/swebench/runs/coder-30b-docker-v2/predictions.jsonl` has 245 entries (last: `sympy__sympy-14308`); 55 of 300 not yet rolled out. **Earlier partial scoring** — 25/62 = 40.3% (first 62) → **51/108 = 47.2% (first 111)** — projecting ~142/300 (47%) final, but final-number scoring against the Docker harness on the full 245 hasn't been re-run. **Comparison vs REAP-25B v1 (29.3%) is muddled by 3 axes** — model swap, rollout backend (host→Docker), AND scoring backend (local→Docker harness). Apples-to-apples requires both models on v2 (Docker rollout + Docker scoring); REAP-25B v2 will run after Coder-30B finishes. Until then, claims about "harness uplift" are not yet supported. Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout`.
3. **Scaffold A/B vs opencode.** Two challengers to bench on REAP-25B's 0/5 failing cluster after the bake-off: (a) [**little-coder**](https://github.com/itayinbarr/little-coder) — small-model-tuned harness (skill injection, thinking-budget cap, write-vs-edit invariant; built on `pi`) claiming **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1 / 23.82% TBench 2.0** — install `npm i -g little-coder`, OpenAI-compat against our `:23334`. (b) [**claw-code**](https://github.com/ultraworkers/claw-code) — Rust implementation of the `claw` CLI harness (build from source via `cargo build --workspace`; the crates.io stub is deprecated); takes `OPENAI_API_KEY` env var, no published benchmarks but the parity-harness/mock-service design is interesting. If either lifts ≥2/5 on REAP-25B's 0/5, promote to a second harness column in the bake-off.

### Cold-launch matrix (TP=1 / 24 GB, current single-card rig)

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

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json` (16 tagged runs as of 2026-05-02).

**Open blocker:** Second 3090 reinstall pending (PCIe adapter swap). Unblocks long-context benches (TP=2 256K), Devstral-24B serving (TP=1 OOMs at AWQ create_weights), Qwen3.6-35B-A3B full-context revalidation, multi-attention-backend Gemma 4 A/B.

## Known Issues (open)

- **NEW 2026-05-07: comprehensive AWQ scales audit across 30+ checkpoints — full report at [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).** Ran `check_awq_scales.py` (post-bf16 fix `6cfcd21`) over every local AWQ + every `hf-mattbucci/` mirror. Findings: **4 disaster-class** (Gemma 4 21B-REAP all variants — DO NOT USE, see entry below); **8 suspicious-class** (Qwen3.5/3.6 family rare-expert under-calibration in early layers — quality-degraded but serves; affects production `qwen36` + `qwen36-tp1` + `qwen35-moe`); **3 minor-flag** (Qwen3-Coder family single-expert finding, negligible inference impact); **rest clean**. **Notable:** `hf-mattbucci/Qwen3.6-35B-A3B-AWQ` (native, used by `qwen36` preset) has 144 flagged scales, while `Qwen3.6-35B-A3B-AWQ-CT` (different calibration build of same model) is fully clean. **2026-05-07 follow-up: CT variant now serves cleanly thanks to [patch 029](patches/029-qwen35-shared-expert-gate-ct-dequant.patch).** Earlier-blocking bug (`shared_expert_gate.weight_packed not found in params_dict` × 120 + serves multilingual gibberish) was that the model constructs `shared_expert_gate` as plain `torch.nn.Linear` so CT's quantized triplet had nowhere to land. Patch buffers the 3-tuple per layer + dequants int4 → bf16 via `compressed_tensors.unpack_from_int32` and writes to `.weight`. Verified: TP=1 / 2K → **4/4 PASS** (was 1/4 gibberish), 0 missing-param warnings, content-aware vision/video output (`'a red circle moves to the right.'`). Switching `qwen36` preset to CT for the calibration-clean win is the next iteration's preset-side action; left for separate validation under TP=2 / 256K once the second 3090 returns. See report for inference-impact analysis + mitigation paths for the rest of the suspicious-class set.
- ~~**Gemma 4 multimodal MoE per-expert AWQ unloadable on v0.5.10**~~ — RESOLVED 2026-05-09 by env upgrade to v0.5.11. v0.5.10's `expert_params_mapping` only handled HF-fused 3D tensors; per-expert AWQ keys silently fell through to the catch-all branch and got dropped. Patch 028 on v0.5.11 closes the per-expert path; `mattbucci/gemma-4-26B-AWQ` HF mirror + the local v3b 21B-REAP build should now serve correctly. Validation pending re-test under the new stack.
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE.** Calibration breakage: `scripts/eval/check_awq_scales.py` flags **164 ALL-ZERO scale tensors** (vision_tower + MoE experts). Empty `ignore` list silently produced zero scales for vast portions of the model. v3b at `~/AI/models/gemma-4-21b-REAP-AWQ-thinking-vision-v3b-2026-05-08` is the corrected build (regex `ignore` patterns + key-order fix). Fix recipe: regex `ignore` covering `vision_tower` + `embed_vision` + `multi_modal_projector`.
- ~~**2026-05-07: awq_marlin Qwen3MoE inference timeout at TP=1 (`coder-30b` + `qwen3-ream`).**~~ **RESOLVED** via `--disable-piecewise-cuda-graph` baked into both presets (commit `26c53bd`). Both went from 0/1 timeout 120s to 1/1 PASS in <1s. Open follow-up: file upstream once we have a tighter repro than "Qwen3MoE preset on awq_marlin TP=1".
- ~~**2026-05-07: `qwen35-moe` thinking truncates at 3840 tok at CTX=4K.**~~ **RESOLVED** via two validator fixes (commit `a37c450`): bumped `check_thinking` default `max_tokens` 4096→8192, refined the gate from `(closed or not truncated)` to plain `has_reasoning AND answer_correct`. Re-validator: 3/4 → 4/4 PASS. Model was genuinely thinking + answering correctly; the failure was validator over-strictness on verbose recipes.
- **Qwen3.6-35B-A3B long-context decode regression** (TP=2 / 256K observation, can't reproduce at TP=1) — historical: 33 tok/s short → 2.6 tok/s @250K on flashinfer with native AWQ. **2026-05-09 v0.5.11 / CT-format / TP=1 sweep**: TPOT flat 25.0→25.4ms across 1K/4K/8K/16K (39-40 tok/s consistent), see `benchmarks/qwen3.6-35b-a3b/v0511-flashinfer.json`. Past 16K can't be tested at TP=1 / 24 GB (KV pool capped at 13539 tokens after 17.5 GB weights at MEM=0.85 + 0.5 GB Mamba). Whether v0.5.11 + CT format closes the regression past 16K is undetermined until second 3090 returns and TP=2 / 256K can be re-run. If the regression persists, next test is `--attention-backend triton` + verify patch 011 (FP32 online-softmax accumulation) is applied.
- **Qwen3-VL-30B MoE AWQ** — closed: SGLang `Qwen3VLMoeForConditionalGeneration` loader is broken (3 calibration variants + community vLLM AWQ all produce identical garbage). Needs upstream loader fix or weight-mapping trace. **2026-05-07 receipt — self-cal native AWQ confirmed broken too.** Found `qwen3-vl-moe` preset was pointing at a non-existent path (`Qwen3-VL-30B-A3B-Instruct-AWQ-4bit`); fixed to local self-cal `Qwen3-VL-30B-A3B-AWQ-native-thinking-vision` (full metadata, single-file safetensors, arch `Qwen3VLMoeForConditionalGeneration`, 3090 commit `aeb13a4`). Smoke-tested at TP=1 / 4K — server fires up cleanly via awq_marlin runtime conversion + triton_attn multimodal backend, but output is multilingual gibberish identical to the prior failure mode (basic: `'1볼 nicol大量大量腰腰ធ historicpackagepackage▾ וה vistawebsocket》，p'`; vision: empty saw, `'可以说 알ู่-yaly应注意洗涤洗涤...'`; video same). 0/3 PASS. **Confirms loader bug is source-independent** — community vLLM AWQ + 3 calibration variants + self-cal native all produce the same garbage. Narrows fix to either an upstream patch on `Qwen3VLMoe*` weight loading or a separate native-AWQ + `ConditionalGeneration` weight-mapping issue. Narrative in `patches/README.md`.
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **Stale local checkpoints vs HF — `coder-30b` repointed 2026-05-01.** The `coder-30b` preset now points at `hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT` (the Apr-29 HF mirror, run through `convert_moe_ct_to_awq.py --group-size 128` since the HF upload is CT-format). Smoke test passes: clean Python lambda output, `finish=stop` at 15 tokens. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE in its own HF README (validation failed on the upload — REAP merger broken for Qwen3MoeForCausalLM, R9700 cross-team). Other presets audited and clean.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- ~~**Gemma 4 26B video OOMs in `_position_embeddings` on Ampere TP=1**~~ — RESOLVED 2026-05-04 via [patch 026](patches/026-gemma4-mm-video-per-frame-batching.patch). Processes frames one-at-a-time through the vision tower instead of batching, dropping the one_hot allocation peak from ~1.24 GB to ~100 MB. Validated 4/4 PASS across both 26B + 31B; structural fix only — quality remains validator-passes-but-degraded per the closed Gemma 4 vision Known Issue below. Patch portable — cross-team advisory shipped to R9700.
- ~~**Gemma 4 vision quality degraded vs Qwen3-VL/Qwen3.5/Qwen3.6 baseline**~~ — REOPENED-AND-CLOSED 2026-05-09 — actually a **SGLang serving-side gap on v0.5.10 → fixed in v0.5.11**, NOT upstream Google as the 2026-05-07 R9700 BF16 probe suggested. Post-env-upgrade sweep: gemma4 26B vision returns `'a solid red circle with a black outline is centered on a white background.'` (content-aware) and gemma4-31b returns `'a red circle with a black outline on a white background.'`, vs the v0.5.10 outputs `'a red and black pixelated gradient fades into a white background'` / `'a small, fragmented red shape against a white background'`. Video also flipped from "static image" to "moves diagonally / from the left side to the right". The R9700 BF16-base probe was on v0.5.10 ROCm path; one of the v0.5.10 → v0.5.11 changes (patch 028 multimodal per-expert loader landing natively + transformers 5.6 + flashinfer 0.6.8.post1) restores Gemma 4 content-recognition. Pushed cross-team alert to R9700 — they should expect this to lift on their env rebuild too. Patches 023 + 024 + 025 + 026 still apply (closed structural bugs, complementary fix path).
- **Piecewise CUDA graph disabled on certain presets** — `coder-reap` / `coder-reap-25b` (cold-launch capture hang, `--disable-piecewise-cuda-graph` clears it); `qwen35-moe` and `qwen36` (DeltaNet+MoE+mamba_cache interaction); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn + disable-cuda-graph combo). The reference `qwen3-ream` preset DOES enable piecewise and captures 50 graphs cleanly at TP=2/256K — so piecewise itself isn't broken, it's per-preset. Old "quant_type=None" claim from 2026-04-22 era no longer reproduces (likely fixed upstream). Each disable-comment in launch.sh documents its specific reason.
- **TP=2 currently degraded — one 3090 offline (PCIe adapter swap pending).** Single-GPU rig at TP=1 / 24 GB. AWQ MoE models (qwen3-ream, coder-30b, coder-reap-25b) fit at 4K–8K context. Long-context (256K) and TP=2 throughput benchmarks paused until both cards return. **Devstral 24B Dense OOMs on TP=1** — source weights load to ~23.48 GiB resident before `create_weights` runs, then the per-layer `torch.empty(input_size, output_size // pack_factor, dtype=int32)` allocation pushes past the 24 GB budget. Fails at `awq.py:518` (awq_marlin) and `compressed_tensors_wNa16.py:135` (CT). `mem-fraction=0.97` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` don't close the gap. Fix needs an upstream loader change (lazy/streamed dest allocation) or `Devstral-Small`. Workaround until second card returns: MoE-AWQ models (active params << 14 GB).

## Suggested next

Open items only.  Resolved entries from prior loops live in [patches/README.md](patches/README.md) under "Recent resolved items" / per-patch sections.

- **Per-expert AWQ Gemma 4 MoE pad-only on v0.5.11** — affects both `mattbucci/gemma-4-26B-AWQ` HF mirror AND local 21B-REAP-v3b build. Server boots clean (0 not-found warnings, 0 expert0-zero warnings via patch 028); audit clean (0/11725 scales + qweight); validator returns `<pad>` for every prompt. Tried `QUANT=moe_wna16 DTYPE=bfloat16` (R9700's Coder-30B Qwen3MoE fix axis) — same pad output, so the bug isn't the per-Linear AWQ kernel routing. Local CT build (`gemma-4-26B-A4B-it-AWQ-4bit`, current `gemma4` preset default) works 4/4 PASS — v0.5.11 vision lift is specific to the CT loader path. The per-expert AWQ MoE Gemma 4 dequant/forward path produces zeros somewhere downstream of patch 028's expert mapping. Local CT remains the only serviceable Gemma 4 MoE format on Ampere; v3b ship to HF blocked. Likely needs upstream SGLang fix or alternative kernel path.
- **Ship 21B-REAP-v3b to `mattbucci/gemma-4-21B-REAP-AWQ`** (post-env-rebuild validation gate). Build is calibration-clean (audit passes 0/11725 scales + 0/11725 qweight), key-format-correct (post-converter-fix `839e44b`). Generation-quality verification on v0.5.11 + cross-stack RDNA4 cross-validation are the remaining gates.
- **Devstral-24B-AWQ HF mirror repoint (`mattbucci/Devstral-24B-AWQ`)** — `devstral` preset still points at local `Devstral-24B-AWQ-Marlin`. Model OOMs at TP=1 anyway (Devstral 24B Dense + AWQ create_weights eager prealloc, see Known Issues), so the repoint is moot until the second 3090 returns. When TP=2 is back, switching to the HF mirror gives canonical traceability + `git pull` auto-update. Single-line repoint pending validation that R9700's calibration is functionally equivalent to the local Marlin-prepacked build.
- **`qwen36` preset switch from native AWQ to CT variant** (post-patch-029 cleanup). Patch 029 unblocked `hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT` (calibration-clean: 0/31010 flagged scales) on NVIDIA. **Validation extended 2026-05-08: 4/4 PASS at TP=1 / 4K** (38.1s) — basic + thinking (1423 tok finish=stop) + vision (saw=red+circle+round) + video. Patch 029 dequant scales hold past the 2K ship window. The native AWQ at `hf-mattbucci/Qwen3.6-35B-A3B-AWQ` has 144 flagged rare-expert scales — switching the default to CT closes that quality gap. Remaining unknowns gating the headline-throughput preset swap: (a) TP=2 / 256K parity (need 2nd 3090), (b) any CT-format regressions on awq_marlin TP=2 path that we haven't seen at TP=1. Override available now: `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT QUANT=compressed-tensors ./scripts/launch.sh qwen36-tp1 --tp 1 --context-length 4096 --mem-fraction 0.85`.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.11, apply patches, create conda env

# TP=1 / 24 GB friendly (current rig — second 3090 offline):
./scripts/launch.sh qwen3-ream              # fastest 256K — reference model (MoE active params fit cold)
./scripts/launch.sh qwen35-tp1              # Qwen3.6-27B-AWQ R9700 recal — TP=1 cold-fit variant (CTX=4K), 4/4 PASS (2026-05-07 strict sweep)
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput
./scripts/launch.sh coder-reap              # Coder-REAP-25B — SWE-bench Lite leader (29.3%)
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — TP=1 defaults boot cold (4K/MAX_RUNNING=1)
./scripts/launch.sh qwen36-tp1              # Qwen3.6-35B-A3B AWQ-native — TP=1 cold-fit variant (CTX=2K), 4/4 PASS (2026-05-07 strict sweep)

# TP=2 only (second 3090 needed):
./scripts/launch.sh devstral-long           # Devstral-24B at 217K — OOMs on TP=1 (eager weight prealloc, see Known Issues)
./scripts/launch.sh devstral                # Devstral-24B 131K default — same TP=2-only constraint

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
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **74** | 13.5 ms | `qwen3-ream` | **Hits 256K target** |
| **Qwen3.6-35B-A3B AWQ-native** | DeltaNet+MoE (256 exp, VL) | **262K** | 2.6 | 385 ms | `qwen36` | **4/4 PASS** at `qwen36-tp1` TP=1 / 2K (2026-05-07 strict sweep, 54.3s — same model). Native AWQ has 144 flagged rare-expert scales per audit; CT variant clean (see Suggested-next for CT-switch plan). 33 @ short / 5.8 @160K / 2.6 @250K. |
| **Qwen3.6-27B AWQ (R9700 recal Apr-29)** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` (preset → `mattbucci/Qwen3.6-27B-AWQ`) | **4/4 PASS basic+thinking+vision+video** at `qwen35-tp1` TP=1 / 4K (2026-05-07 strict-validator sweep, 36.9s). Originally 3/3 at 2026-05-01 ship before video-check existed; R9700's `balanced_thinking_text` recal resolved the vision regression seen on the prior CT v3 self-cal. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **150K** | **40** | 25 ms | `qwen3-vl-32b` | **3/3 PASS basic+vision+video** at TP=1 / 4K (2026-05-07 strict-validator sweep). Vision: `'a solid red circle with a black outline is centered on a white background'`. Video: `'a red circle moves from the left side of the screen to the right side'`. Thinking auto-skipped (upstream Qwen3-VL-32B-Instruct is non-thinking by design — the `-Thinking` edition is separate). 21 GB weights leave 1.33 GB free at 4K KV; 4699 KV tokens budget. Preset repointed 2026-05-06 from community QuantTrio to R9700's self-cal `mattbucci/Qwen3-VL-32B-AWQ`. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling. **Currently TP=2 only** — OOMs on TP=1 / 24 GB at all loader paths (see Known Issues). |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly. **TP=2 only** until the second 3090 returns. |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Gemma 4 31B Dense | Dense | 16K* | 22 | ~50 ms | `gemma4-31b` | **basic + thinking VERIFIED (2026-05-03) at preset-default 16K**: `gemma4-31b-ctx16k-May03` entry. Thinking confirmed real (structured channel + correct arithmetic). **Vision validator-passes-but-degraded** — model says "scattered black and red pixels on a white background" instead of "red circle"; saw=red,round is a keyword-grep hit, not real recognition. *KV pool tight at 16K — only 1947 tokens after 20 GB weights at TP=1 / mem-fraction 0.92, so per-request input cap is ~1.9K. Drop to `--context-length 4096` to get a more usable KV budget, or wait for the second 3090 (TP=2) to unlock real 16K+ ctx. Triple-fix: patches 023+024 + BF16 default + local config arch flip to `Gemma4ForConditionalGeneration` (R9700 task #63 will retire the local edit). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | **4/4 PASS basic+thinking+vision+video** at `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` commit `2cf434c8` (2026-05-07 strict-validator re-sweep, TP=1 / 12K, 202s). Originally 3/3 at 2026-05-02 ship before video check existed; R9700 cross-validated 4/4 PASS on RDNA4 (2026-05-03 18:51 PDT). Built from Cerebras's REAP base via `balanced_thinking_vision` recipe (256 samples × 2K, 18.22h CPU GPTQ + CT→AWQ). 333 vision tensors retained. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | **basic + thinking VERIFIED (2026-05-03) at 16K ctx** post-patches 023 + 024 + BF16 default. `gemma4-26b-ctx16k-May03` validator entry. Thinking probed via `skip_special_tokens=False`: separated `<\|channel>thought` channel, real step-by-step scratch work, correct answer 11 with intermediate 22. **Vision validator-passes-but-degraded** — "gradient of black and red pixels fades into a white background" is a keyword-grep hit on red/round, not real recognition of the red circle. Tower loads cleanly (no NaN); quality just isn't there yet vs Qwen3-VL baseline. 25880 KV tokens budget at TP=1 / mem-fraction 0.92. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | — | — | — | — | **PENDING v0.5.11 env-rebuild validation (in flight).** v3b output at `~/AI/models/gemma-4-21b-REAP-AWQ-thinking-vision-v3b-2026-05-08` audits clean (0/11725 scales + 0/11725 qweight via `check_awq_scales.py`); calibration recipe + converter normalize-key fix landed `839e44b`. v0.5.10 lacked per-expert AWQ multimodal support so all 3 multimodal Gemma 4 builds (this v3b, the CT-format v3 build, R9700's `mattbucci/gemma-4-26B-AWQ` HF mirror) returned `<pad>`-only output. Patch 028 on v0.5.11 closes the gap. Re-validate post-rebuild. v2 build is a separate calibration disaster (164 ALL-ZERO scales) — DO NOT USE. |

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
| SGLang | v0.5.11 + 16 local patches (`ls patches/*.patch \| wc -l`) |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.58) / cu130 wheel |
| NCCL | bundled with torch 2.11 (P2P over NVLink) |
| FlashInfer | 0.6.8.post1 (v0.5.11 pin) |
| transformers | 5.6.0 (v0.5.11 pin) |
| sglang-kernel | 0.4.2 |
| compressed-tensors | 0.15.0.1 |

## Patches

16 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.11. Originally rebased from a v0.5.10 set 24→13 (2026-05-07 commit `1655e46`); 002 cross-team port of R9700's qwen3_next AWQ weight_loader fix; 028 v0.5.11 gemma4_mm per-expert AWQ loader (R9700 `gemma4_causal.py` port); 029 qwen35 shared_expert_gate CT dequant (Ampere-only). Per-patch narratives in [`patches/README.md`](patches/README.md).

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
