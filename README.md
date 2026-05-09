# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

> 🚧 **Active session 2026-05-09 (single-card autonomous loop):** One 3090 offline (PCIe adapter swap pending — confirmed via `nvidia-smi` showing GPU 0 only, 1 MiB / 24576 MiB used, no running processes). Iteration plan in priority order:
> 1. **Switch `qwen36` / `qwen36-tp1` preset default to CT variant.** Patch 029 + 4/4 PASS validation already in hand (2026-05-08). Native AWQ has 144 flagged rare-expert scales; CT is calibration-clean. TP=1 / 4K validates clean. The remaining "TP=2 / 256K parity" gate doesn't unblock here, but the TP=1 default switch is independent and safe.
> 2. **Re-run TP=1 capability sweep** across all viable presets to confirm no regressions after patches 002 / 029 + validator improvements.
> 3. **Investigate qwen36 long-context decode regression** at TP=1 with `--attention-backend triton` — the issue manifests at 250K but we can probe at 32K-64K within 24 GB and look for the breakpoint.
> 4. **Probe v0.5.11 env-rebuild path** — preserved at `components/sglang.v0.5.11-prepped`; smoke-test what breaks under torch 2.11 / sglang-kernel 0.4.2 / transformers 5.6 to scope the rebuild before committing 1+ hour to it.
> 5. **Cross-team push to R9700 README** — all findings shareable.
>
> User check-in style: reads README each return, interrupts via prompt with new ideas. No need to ask for confirmation between iterations.

## Cross-team activity (rolling, last ~2 weeks)

R9700 (RDNA4) and M4 (Apple) sister teams ship findings into our repo. Compact summary; per-day forensic narrative + closed item history live in [`patches/README.md`](patches/README.md) and `git log -- README.md`.

- **Gemma 4 vision shape-recognition is upstream Google** (R9700 BF16-base probe 2026-05-06). `gemma-4-26B-A4B-it-BF16` returned `'a black and red graphic pattern on a white background'` on the red-circle test — same "validator-passes-but-degraded" pattern as our AWQ. Color sub-channel works, shape sub-channel doesn't. #66 closed as upstream-fix-only-axis. Patches 023+024+025+026 stay applied (real plumbing fixes) but don't lift content recognition.
- **Qwen-family vision works content-aware end-to-end** across self-cal Dense (Qwen3.6-27B), self-cal MoE-REAP (Qwen3.5-28B), self-cal Dense (Qwen3-VL-32B), community Dense (QuantTrio Qwen3-VL-32B), and via balanced thinking_vision recipe. Gemma 4 is the single outlier.
- **R9700 ships `mattbucci/Qwen3-VL-32B-AWQ`** (2026-05-05): self-calibrated from `Qwen/Qwen3-VL-32B-Instruct` BF16 base, balanced_thinking_vision recipe, 4/4 PASS on RDNA4 + 3/3 PASS on Ampere TP=1 / 4K. Preset `qwen3-vl-32b` repointed to this HF mirror.
- **`balanced_thinking_vision` is the production calibration recipe** (40% thinking + 60% non-thinking + 25% LLaVA images). Replaces 70%-thinking which triggered `</think>\nX\n</think>…` repetition loops. R9700 originator; defined in `scripts/quantize/calibration_datasets.py`.
- **`scripts/eval/check_awq_scales.py`** scans `*.scales` / `*.qweight` for all-zero / NaN / Inf. R9700 forensic tool (commit `e4aa012`); took 30s to find a 16h calibration disaster the validator missed. Run after every CT→native AWQ conversion. All locally-mirrored mattbucci AWQs pass; 5 have minor rare-expert flags (single MoE up_proj 50–83% zero; quality-degraded but serves).
- **Calibration regex-ignore rule** (both stacks lost ~16h on 2026-05-06): bare-string `ignore=["model.embed_vision"]` does NOT exclude descendant Linears. Always use `r"re:.*vision_tower.*"` / `r"re:.*embed_vision.*"` / `r"re:.*multi_modal_projector.*"`. Codified in CLAUDE.md.
- **Both stacks at v0.5.11 patches** (3090 commit `1655e46`, R9700 commit `3466816`, both 2026-05-07): 3090 24→13 patches, R9700 22→15. 11 dropped on 3090 as upstreamed. 8 patches share content cross-stack (003/011/012/018/023/024/025/026). Env-rebuild to v0.5.11 in progress this session.
- **R9700 sweep of all 9 presets** (2026-05-08): 6/9 fully PASS, gemma4 26B is the only failure class on RDNA4 (sampler HSAIL — Ampere unaffected). Receipt: `evals/awq-audit-2026-05-08-r9700-sweep.md` in their repo.
- **Qwen3MoeExperts unfused-experts patch** (R9700 2026-05-08, commits `8c6ad38`/`46a417a`/`82070e2`): transformers 5.x silently drops per-expert checkpoint keys for Qwen3MoeForCausalLM models — port `patches/qwen3moe_unfused_experts.py` from R9700 if Ampere-side recalibration of Coder-30B-A3B / Qwen3-30B-A3B is queued.
- **REAM merger broken for Qwen3MoeForCausalLM** (R9700 2026-04-30): `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` flagged DO NOT USE — Samsung SAIL `merge.py --merging none --saliency reap` produces gibberish on Qwen3MoE arch despite working on Qwen3_5MoeForConditionalGeneration. Use the published Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` instead.
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

0. **Gemma 4 21B-REAP — pad-only output on v0.5.10 multimodal MoE.** Local v3b build is calibration-clean (audit passes 0/11725 scales+qweight, post-converter-fix `839e44b`) and patch 030 lands 30/30 expert0 nonzero. But the running v0.5.10 source lacks proper per-expert AWQ multimodal support (patch 028 targets v0.5.11). Three builds all return 1/4 PASS with `(reasoning)` placeholder + `<pad>` tokens: 21B-REAP-AWQ-v3b, 21B-REAP-CT-v3, `mattbucci/gemma-4-26B-AWQ`. Local CT `gemma-4-26B-A4B-it-AWQ-4bit` still serves loose-4/4 via WNA16 fallback (the current `gemma4` preset default). **Unblock: env-rebuild to v0.5.11** (in flight this session). Post-rebuild todo: re-validate v3b + ship to `mattbucci/gemma-4-21B-REAP-AWQ`; switch `gemma4` preset to `mattbucci/gemma-4-26B-AWQ` HF mirror (currently blocked by same loader gap). Patch 029 extended-CTX validation already done: 4/4 PASS at TP=1 / 4K against `hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT` (2026-05-08, 38.1s).
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
| `gemma4` (26B MoE) | ⚠️ | **2026-05-07 strict-validator re-sweep: 4/4 loose-PASS in 11.5s — vision/video CONTENT DEGRADED.** Validator's keyword grep finds `red+round` in `'a red and black pixelated gradient fades into a white background'` (NOT "red circle") and `move` in `'this video is a static image that does not contain any movement or action'` (it IS a moving circle). Closed as Gemma 4 upstream/calibration limitation per task #66 — R9700's BF16 base probe also returned the same degenerate output, so the issue predates AWQ calibration. **basic + thinking pass real content checks** (484 tok reasoning, finish=stop). Patches 023 + 024 + BF16 default + 025 + 026 all still apply (closed structural bugs); the remaining gap is recipe/architecture-side and won't move via patches. |
| `qwen3-vl-moe` | ❌ | Closed: SGLang loader broken |
| `devstral` / `devstral-long` | ❌ | OOM at AWQ create_weights eager prealloc — TP=2 only |

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json` (16 tagged runs as of 2026-05-02).

**Open blocker:** Second 3090 reinstall pending (PCIe adapter swap). Unblocks long-context benches (TP=2 256K), Devstral-24B serving (TP=1 OOMs at AWQ create_weights), Qwen3.6-35B-A3B full-context revalidation, multi-attention-backend Gemma 4 A/B.

## Known Issues (open)

- **NEW 2026-05-07: comprehensive AWQ scales audit across 30+ checkpoints — full report at [`evals/awq-audit-2026-05-07.md`](evals/awq-audit-2026-05-07.md).** Ran `check_awq_scales.py` (post-bf16 fix `6cfcd21`) over every local AWQ + every `hf-mattbucci/` mirror. Findings: **4 disaster-class** (Gemma 4 21B-REAP all variants — DO NOT USE, see entry below); **8 suspicious-class** (Qwen3.5/3.6 family rare-expert under-calibration in early layers — quality-degraded but serves; affects production `qwen36` + `qwen36-tp1` + `qwen35-moe`); **3 minor-flag** (Qwen3-Coder family single-expert finding, negligible inference impact); **rest clean**. **Notable:** `hf-mattbucci/Qwen3.6-35B-A3B-AWQ` (native, used by `qwen36` preset) has 144 flagged scales, while `Qwen3.6-35B-A3B-AWQ-CT` (different calibration build of same model) is fully clean. **2026-05-07 follow-up: CT variant now serves cleanly thanks to [patch 029](patches/029-qwen35-shared-expert-gate-ct-dequant.patch).** Earlier-blocking bug (`shared_expert_gate.weight_packed not found in params_dict` × 120 + serves multilingual gibberish) was that the model constructs `shared_expert_gate` as plain `torch.nn.Linear` so CT's quantized triplet had nowhere to land. Patch buffers the 3-tuple per layer + dequants int4 → bf16 via `compressed_tensors.unpack_from_int32` and writes to `.weight`. Verified: TP=1 / 2K → **4/4 PASS** (was 1/4 gibberish), 0 missing-param warnings, content-aware vision/video output (`'a red circle moves to the right.'`). Switching `qwen36` preset to CT for the calibration-clean win is the next iteration's preset-side action; left for separate validation under TP=2 / 256K once the second 3090 returns. See report for inference-impact analysis + mitigation paths for the rest of the suspicious-class set.
- **NEW 2026-05-08: Gemma 4 multimodal MoE per-expert AWQ format unloadable on v0.5.10 (current dev rig source) — patch 028 not applied.** Diagnosis tracking the 21B-REAP v3 → v3b → final-root-cause path: (a) v3 had llmcompressor's proj-first key order, fixed in converter commit `839e44b`. (b) v3b reconverted with expert-first keys verified clean. (c) v3b STILL fails — `components/sglang/` is v0.5.10, not v0.5.11; patches/028 targets v0.5.11 `gemma4_mm.py:714+` lines; never applied here. v0.5.10's `expert_params_mapping` (line 802+) only handles HF-fused 3D tensors via `for i in range(num_experts)`. Per-expert AWQ keys silently fall through to the catch-all `name = orig_name` branch → dropped → MoE block stays at default-init → `expert0_first4=[0,0,0,0]` server diagnostic for 27/30 layers → validator returns `(reasoning)` placeholder. Same gap applies to `mattbucci/gemma-4-26B-AWQ` HF mirror format (also per-expert) — currently the only multimodal Gemma 4 builds that work on v0.5.10 are CT-format (loaded via WNA16 fallback in `awq.py`); the `gemma4` preset's local `gemma-4-26B-A4B-it-AWQ-4bit` is CT and serves degraded-but-passing. Fix paths: (i) backport patch 028 to v0.5.10 source (small targeted edit, no env rebuild — easiest); (ii) full v0.5.11 env rebuild per top-of-readme header. Audit tooling (`check_awq_scales.py` now checks both `.scales` and `.qweight`) is correct; the safetensors are correct; only the loader's per-expert support is missing on v0.5.10.
- **NEW 2026-05-07: `gemma-4-21b-REAP-AWQ-thinking-vision-v2` (real calibration breakage, distinct from the v3/v3b loader-gap entry above) — DO NOT USE.** Audit via `scripts/eval/check_awq_scales.py` flags **164 ALL-ZERO scale tensors** (across vision_tower + MoE experts). Validator at TP=1 / 4K returns 0/4 PASS with empty content (`response='(reasoning)'`, `saw=[]`). Server log shows `expert0_nonzero=False expert0_first4=[0, 0, 0, 0]` for **27 of 30 layers** during AWQ MoE pre-repack. Note: the v0.5.10 source can't actually load per-expert AWQ keys at all (see entry above), so the v2 server log expert-0=zero pattern is partially the loader gap and partially the calibration breakage — the calibration breakage is the load-bearing reason it's tagged DO NOT USE. Pattern matches the documented v3 calibration disaster — empty `ignore` list silently produced zero scales for vast portions of the model. Local 26B AWQ at `gemma-4-26B-A4B-it-AWQ-4bit` audits **0 flagged / 115 scale tensors** (clean), so the issue is specific to the 21B-REAP-v2 build. Fix path: full re-calibration with proper `ignore` regex (vision_tower + embed_vision + multi_modal_projector). Not in flight this iteration; tracked here so the model isn't accidentally launched as production. (CLAUDE.md rule: run `check_awq_scales.py` after every CT→native AWQ conversion before ship — this build was uploaded before the rule existed.) Side-finding: audit script crashed on bf16 scale tensors before today; fixed in commit `6cfcd21` (switched safetensors `framework="np"` → `"pt"`).
- ~~**2026-05-07: awq_marlin Qwen3MoE inference timeout at TP=1 (`coder-30b` + `qwen3-ream`).**~~ **RESOLVED** via `--disable-piecewise-cuda-graph` baked into both presets (commit `26c53bd`). Both went from 0/1 timeout 120s to 1/1 PASS in <1s. Open follow-up: file upstream once we have a tighter repro than "Qwen3MoE preset on awq_marlin TP=1".
- ~~**2026-05-07: `qwen35-moe` thinking truncates at 3840 tok at CTX=4K.**~~ **RESOLVED** via two validator fixes (commit `a37c450`): bumped `check_thinking` default `max_tokens` 4096→8192, refined the gate from `(closed or not truncated)` to plain `has_reasoning AND answer_correct`. Re-validator: 3/4 → 4/4 PASS. Model was genuinely thinking + answering correctly; the failure was validator over-strictness on verbose recipes.
- **Qwen3.6-35B-A3B long-context decode regression** — 33 tok/s short → 2.6 tok/s @250K on flashinfer (vs R9700's flat 20 @131K on ROCm-triton). A/B'd CHUNKED/DECODE_STEPS/MAMBA_CACHE/triton attention; none help. **Next test:** `--attention-backend triton` + port patch 011 (FP32 online-softmax accumulation) — R9700 hit the same bug class on RDNA4 and Blackwell sm12.x; flashinfer might already do FP32 internally but worth confirming.
- **Qwen3-VL-30B MoE AWQ** — closed: SGLang `Qwen3VLMoeForConditionalGeneration` loader is broken (3 calibration variants + community vLLM AWQ all produce identical garbage). Needs upstream loader fix or weight-mapping trace. **2026-05-07 receipt — self-cal native AWQ confirmed broken too.** Found `qwen3-vl-moe` preset was pointing at a non-existent path (`Qwen3-VL-30B-A3B-Instruct-AWQ-4bit`); fixed to local self-cal `Qwen3-VL-30B-A3B-AWQ-native-thinking-vision` (full metadata, single-file safetensors, arch `Qwen3VLMoeForConditionalGeneration`, 3090 commit `aeb13a4`). Smoke-tested at TP=1 / 4K — server fires up cleanly via awq_marlin runtime conversion + triton_attn multimodal backend, but output is multilingual gibberish identical to the prior failure mode (basic: `'1볼 nicol大量大量腰腰ធ historicpackagepackage▾ וה vistawebsocket》，p'`; vision: empty saw, `'可以说 알ู่-yaly应注意洗涤洗涤...'`; video same). 0/3 PASS. **Confirms loader bug is source-independent** — community vLLM AWQ + 3 calibration variants + self-cal native all produce the same garbage. Narrows fix to either an upstream patch on `Qwen3VLMoe*` weight loading or a separate native-AWQ + `ConditionalGeneration` weight-mapping issue. Narrative in `patches/README.md`.
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **Stale local checkpoints vs HF — `coder-30b` repointed 2026-05-01.** The `coder-30b` preset now points at `hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT` (the Apr-29 HF mirror, run through `convert_moe_ct_to_awq.py --group-size 128` since the HF upload is CT-format). Smoke test passes: clean Python lambda output, `finish=stop` at 15 tokens. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE in its own HF README (validation failed on the upload — REAP merger broken for Qwen3MoeForCausalLM, R9700 cross-team). Other presets audited and clean.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- ~~**Gemma 4 26B video OOMs in `_position_embeddings` on Ampere TP=1**~~ — RESOLVED 2026-05-04 via [patch 026](patches/026-gemma4-mm-video-per-frame-batching.patch). Processes frames one-at-a-time through the vision tower instead of batching, dropping the one_hot allocation peak from ~1.24 GB to ~100 MB. Validated 4/4 PASS across both 26B + 31B; structural fix only — quality remains validator-passes-but-degraded per the closed Gemma 4 vision Known Issue below. Patch portable — cross-team advisory shipped to R9700.
- ~~**Gemma 4 vision quality degraded vs Qwen3-VL/Qwen3.5/Qwen3.6 baseline**~~ — CLOSED 2026-05-07 as **upstream Google model limitation** (R9700 BF16-base probe: `gemma-4-26B-A4B-it-BF16` returned `'a simple black and red graphic pattern on a white background'` on the red-circle test, same "validator-passes-but-degraded" pattern as our AWQ). Color sub-channel works (`'Red'`), shape sub-channel doesn't (`'None'` for circle/square/triangle forced choice). Calibration-side fix is not the right axis — patches 023 + 024 + 025 + 026 stay applied because they fix real upstream-divergence bugs in SGLang Gemma 4 plumbing, but they don't lift content recognition. STOP-WORK signal at top of README has the full data table + four-prompt probe; investigation history (suspects ruled out 2026-05-03 → 04, drop_images=True audit, V-SmoothQuant disqualification, `<image_pad>` plumbing trace) lives in commit history `dd7ad76` → `0758258` → `7e75703` → `c069288`.
- **Piecewise CUDA graph disabled on certain presets** — `coder-reap` / `coder-reap-25b` (cold-launch capture hang, `--disable-piecewise-cuda-graph` clears it); `qwen35-moe` and `qwen36` (DeltaNet+MoE+mamba_cache interaction); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn + disable-cuda-graph combo). The reference `qwen3-ream` preset DOES enable piecewise and captures 50 graphs cleanly at TP=2/256K — so piecewise itself isn't broken, it's per-preset. Old "quant_type=None" claim from 2026-04-22 era no longer reproduces (likely fixed upstream). Each disable-comment in launch.sh documents its specific reason.
- **One 3090 offline (PCIe adapter swap pending)** — repo currently runs TP=1, single-GPU. AWQ MoE models (qwen3-ream, coder-30b, coder-reap-25b) still fit on 24 GB at 4K–8K context. Long-context evals and TP=2 benches paused until the second card returns. Quick capability sweeps via `./scripts/eval/test_capabilities_all.sh`. **Devstral 24B Dense OOMs on 24 GB at TP=1** — confirmed 2026-05-01 across all three loader paths (AWQ-Marlin, AWQ direct, compressed-tensors): the source weights load to ~23.48 GiB resident before the AWQ `create_weights` step runs, and that step's `torch.empty(input_size, output_size // pack_factor, dtype=int32)` per-layer destination allocation pushes past the 24 GB budget on the very next 160 MB request. Fails at `awq.py:518` with awq_marlin, at `compressed_tensors_wNa16.py:135` with CT — same root cause: weight-buffer prealloc isn't chunked, and Devstral's 24B-param footprint × 4-bit int32 packing leaves no headroom. `mem-fraction=0.97` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` both tested, neither closes the gap. Original "Marlin repack doubles weight memory" diagnosis was wrong — the bug is in eager dest alloc, not in the repack. Fix would need an upstream SGLang loader change (lazy/streamed dest allocation) or use of `Devstral-Small` once published. Workaround until the second 3090 returns: stay on the MoE-AWQ models (active params << 14 GB).

## Suggested next

Open items only.  Resolved entries from prior loops live in [patches/README.md](patches/README.md) under "Recent resolved items" / per-patch sections.

- **🔝 Highest-leverage: env rebuild to v0.5.11** (header note has the gory torch 2.9 → 2.11 + sglang-kernel 0.4.1 → 0.4.2 + transformers 5.5.3 → 5.6 details). Unlocks: (a) patch 028 actually applies (per-expert AWQ multimodal Gemma 4 loader — currently bridged on v0.5.10 by patch 030 backport but pad-token bug still blocks generation, see In Flight #0); (b) `mattbucci/gemma-4-26B-AWQ` HF mirror serves correctly via `gemma4` preset (currently v0.5.10 forces local CT default); (c) 21B-REAP-v3b output (`~/AI/models/gemma-4-21b-REAP-AWQ-thinking-vision-v3b-2026-05-08`) becomes shippable to HF. Single largest unblocker for the Gemma 4 family.
- **Ship 21B-REAP-v3b to `mattbucci/gemma-4-21B-REAP-AWQ`** (post-env-rebuild). Build is calibration-clean (audit passes 0/11725 scales + 0/11725 qweight), key-format-correct (post-converter-fix `839e44b`), v0.5.10 loader-fixed (patch 030 lands 30/30 expert0 nonzero). Generation-quality verification on v0.5.11 + cross-stack RDNA4 cross-validation are the remaining gates.
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
| Gemma 4 21B REAP AWQ | MoE (128 exp) | — | — | — | — | **DO NOT USE on v0.5.10 — see In Flight #0 + Known Issues 2026-05-08 entries.** v3 calibration recipe + converter normalize-key fix + patch 030 v0.5.10 loader backport landed; v3b output at `~/AI/models/gemma-4-21b-REAP-AWQ-thinking-vision-v3b-2026-05-08` audits clean (0/11725 scales + 0/11725 qweight flagged via `check_awq_scales.py`) and loader path is correct (30/30 layers `expert0_nonzero=True`). But pad-token bug surfaces post-load on v0.5.10 multimodal Gemma 4 — affects this build, the CT-format v3 build, AND R9700's `mattbucci/gemma-4-26B-AWQ` HF mirror. Resolves on v0.5.11 env rebuild (patch 028 already targets v0.5.11). v2 build is a separate calibration disaster (164 ALL-ZERO scales). |

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
| SGLang | running source `components/sglang/` is v0.5.10; `patches/` contains 17 files (13 v0.5.11-targeted from rebase + 028 v0.5.11 multimodal per-expert + 029 qwen35 CT shared_expert_gate + 030 v0.5.10 backport of 028). `setup.sh` `git apply --check` silently skips whichever patches don't match the running source. Forward path is env-rebuild to v0.5.11 — see header `v0.5.10 → v0.5.11 patch upgrade` note. Verify count with `ls patches/*.patch \| wc -l`. |
| PyTorch | 2.9.1 + cu128 (env still on 2.9; v0.5.11 source forward-compatible) |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 (v0.5.11 pin is 0.6.8.post1; patch 005 relaxes to 0.6.8) |
| transformers | 5.5.3 (v0.5.11 pin is 5.6.0; current env still on 5.5.3) |

## Patches

17 patches (`ls patches/*.patch | wc -l`): 13 v0.5.11-targeted from the v0.5.10→v0.5.11 upgrade (2026-05-07 commit `1655e46`); 002 cross-team port of R9700's qwen3_next AWQ weight_loader fix; 028 v0.5.11 gemma4_mm per-expert AWQ loader (R9700 `gemma4_causal.py` port); 029 qwen35 shared_expert_gate CT dequant (Ampere-only); 030 v0.5.10 backport of 028 (this rig — 028 only applies on a v0.5.11 worktree). `setup.sh`'s `git apply --check` silently skips patches that don't match the running source — when env-rebuild lands, 030 will skip and 028 will apply. Per-patch narratives in [`patches/README.md`](patches/README.md).

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
components/sglang.v0.5.10-backup/  # rollback safety net (delete after v0.5.11 stable burn-in)
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Cross-team threads (R9700 answers + advice on closed items) live in [`patches/README.md`](patches/README.md).
