# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

> **Recommended for coding tasks: `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% on SWE-bench Lite** (`./scripts/launch.sh coder-reap-25b`)
>
> *Disclaimer: agent harness was [opencode](https://github.com/anomalyco/opencode) v1.14.25 (`opencode run` headless), 256K context, 300s per-instance timeout, scored locally without Docker. Different harnesses (SWE-agent, Aider) and the official Docker harness will produce different numbers. 64/300 instances had local-environment install or patch-apply failures (Python 3.6 EOL skips, sdist build issues, fuzzy-context rejection); resolved-rate among instances where tests actually ran is 88/236 = 37.3%. See `evals/swebench/runs/coder-reap-25b-lite/` for raw artifacts. This is the first model in a four-way bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B / Qwen3-30B-REAM still queued).*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill). For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## In Flight

1. **SWE-bench Lite bake-off — Coder-REAP-25B done, three coders queued.** Coder-REAP-25B baseline shipped (29.3% / 37.3% on tests-ran, see banner). Next up: **Coder-30B → Qwen3.6-35B-A3B → Devstral-24B → Qwen3-30B-REAM**. Each rollout ~22h at 300s/instance × 256K ctx; scoring is ~30 min on the existing per-instance venv cache. Final pick → SWE-bench Verified (500 task) for the headline number on the top 1-2 finalists.
2. **Rollout v2: Docker-backed test-edit-test harness — Coder-30B partial 47.2%, full run in flight.** v1 was read-edit-pray (model couldn't `pytest` mid-iteration; 64/300 instances also failed local-env install/patch-apply scoring, marked unresolved). v2 runs opencode INSIDE the official swebench eval container (FROM `swebench/sweb.eval.x86_64.<inst>` + Node + opencode + ripgrep, host SGLang reachable via `--network=host`), so the model can run `pytest` against the exact env its fix is graded in, AND we score with the official Docker harness. **Coder-30B partial scoring** — 25/62 = 40.3% (first 62) → **51/108 = 47.2% (first 111)** — projecting ~142/300 (47%) final. **Comparison vs REAP-25B v1 (29.3%) is muddled by 3 axes** — model swap, rollout backend (host→Docker), AND scoring backend (local→Docker harness). Apples-to-apples requires both models on v2 (Docker rollout + Docker scoring); REAP-25B v2 will run after Coder-30B finishes. Until then, claims about "harness uplift" are not yet supported. Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout`.
3. **Scaffold A/B vs opencode.** Two challengers to bench on REAP-25B's 0/5 failing cluster after the bake-off: (a) [**little-coder**](https://github.com/itayinbarr/little-coder) — small-model-tuned harness (skill injection, thinking-budget cap, write-vs-edit invariant; built on `pi`) claiming **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1 / 23.82% TBench 2.0** — install `npm i -g little-coder`, OpenAI-compat against our `:23334`. (b) [**claw-code**](https://github.com/ultraworkers/claw-code) — Rust implementation of the `claw` CLI harness (build from source via `cargo build --workspace`; the crates.io stub is deprecated); takes `OPENAI_API_KEY` env var, no published benchmarks but the parity-harness/mock-service design is interesting. If either lifts ≥2/5 on REAP-25B's 0/5, promote to a second harness column in the bake-off.
4. **Qwen3.5-28B REAP thinking recalibration (paused — env blocker).** Re-cal with `thinking_vision_video` recipe paused after the calibration+eval-concurrent system crash. Recipe choice updated 2026-05-01: ported R9700's new `balanced_thinking_vision` (40/60 thinking/non-thinking, vs the existing 70/30 `thinking_vision_video`) to `scripts/quantize/calibration_datasets.py`, and made `quantize_qwen35_28b_moe_reap_thinking.py` accept `RECIPE` env var. Detached calibration kicked off via setsid (PPID=1, ~89 GB RAM free, BF16 base 54 GB) immediately died on import: `ModuleNotFoundError: No module named 'compressed_tensors.distributed'` — the vendored `components/llmcompressor` (commit 30845208) imports from `compressed_tensors.distributed`, but the module only exists on `vllm-project/compressed-tensors@main`, not in our pinned 0.15.0.1. Unblocking needs `pip install git+https://github.com/vllm-project/compressed-tensors.git@main` in the sglang env (or pinning llmcompressor to a pre-distributed-refactor commit). Holding for explicit go-ahead before changing env packages mid-loop.

### Loop status (2026-05-01, 8-iteration self-paced run)

After the second 3090 came offline, an autonomous loop ran 8 substantive iterations. Most menu items are now blocked on hardware or explicit user authorization — surfacing the unblockers here so the next session has a clear re-entry point.

**What landed (committed + pushed):**
- patches **020 v2** (upstream `clippable_linear.py` port — fixes shim signature mismatch), **021** (Marlin MoE GeLU activation dispatch — drops 4 hard `silu`-only asserts), **022** (`gemma4_causal.py` EntryClass dedup — unblocks any model launch once `gemma4_mm.py` imports cleanly).
- `validate_capabilities.py` extended with `--save`/`--tag` JSON output; `test_capabilities_all.sh` orchestrator with `NON_THINKING` auto-skip.
- 4× capability sweeps logged in `benchmarks/quality/capability_check.json` (`coder-30b`, `coder-reap-25b`, `qwen3-ream`, `coder-30b-mattbucci-HF-Apr29` all 1/1 basic PASS; `qwen36-27b-ct-v3-revalidate` 3/4 — vision regression vs prior pre-validator-patch "4/4"; `gemma4-clippable-fix` 1/4 — server boots clean, decode emits `<pad>` garbage).
- Cross-team notes posted to R9700 (`f6da90b`) and M4 (`d039667`) READMEs.
- `balanced_thinking_vision` + `balanced_thinking_text` calibration recipes ported from R9700.

**Unblockers that need user input to make progress:**
1. **`pip install git+https://github.com/vllm-project/compressed-tensors.git@main`** in the sglang env. Unblocks the Qwen3.5-28B REAP recal (In Flight #4) and any future llmcompressor work. Risk: bumps `compressed-tensors` ahead of the released 0.15.0.1 we serve with — could affect serving paths for already-shipped CT-format models. Recommended: snapshot the env first, then install, then run a smoke validate on a CT-format model (e.g. qwen36-27b CT v3) to confirm serving still works.
2. **Second 3090 reinstalled.** Unblocks: long-context benches (TP=2 256K targets), Devstral-24B serving (AWQ-Marlin repack overshoots 24 GB on TP=1), Qwen3.6-35B-A3B / Qwen3-VL-32B re-validation with the patched validator, and crucially the Gemma 4 attention-backend A/B (triton-only fits at TP=1; need `flex_attention`/`torch_native` to actually disprove the precision hypothesis).
3. **Gemma 4 21B-REAP variant smoke test on TP=1.** ~14 GB weights (vs 26B's ~17 GB) might leave headroom for `flex_attention`/`torch_native`. If A/B works on the 21B and the alternate backend produces clean output, the precision hypothesis is confirmed and patch 011-style FP32-accumulation needs to extend into Gemma 4's attention path. Doable on current hardware; just needs the explicit "test 21B on torch_native" prompt.

**What's still in scope on the current GPU (haven't tackled):**
- `coder-30b` preset still points at the local Apr-17 self-built AWQ-Marlin instead of the Apr-29 `mattbucci/Qwen3-Coder-30B-A3B-AWQ` HF mirror. Repointing requires a CT→AWQ-Marlin conversion of the HF version (`scripts/quantize/convert_moe_ct_to_awq.py`, ~30 min) before the preset swap. Documented as version drift in Known Issues.
- Final logits collapse on Gemma 4 hasn't been traced past "attention precision suspect." Could probe by intercepting the lm_head output on the next prompt and checking whether logits are uniform/zero. Higher-effort, more code.

## Cross-team updates

- **R9700 building REAM-pruned Qwen3.6-35B-A3B** (256→192 experts via Samsung SAIL `merge.py`, c4+math+code calibration). ETA ~24-28h CPU. Output → `Qwen3.6-35B-A3B-REAM-BF16` then quant → ~27B AWQ. First self-built REAM variant of a multimodal MoE on either rig; will join the SWE-bench eval queue when shipped. **Update 2026-04-30:** shipped at `mattbucci/Qwen3.6-REAM-A3B-AWQ`. Coder-Next-REAM (60B effective) also shipped at `mattbucci/Qwen3-Coder-Next-REAM-AWQ`.
- **HF upload rule (cross-team):** plain `hf upload <repo> <dir>` for repos ≤25 GB; `hf upload-large-folder` only past 50 GB (the latter stalled 11h at `committed: 0/9` on a 19 GB push due to XET worker deadlock). **Update 2026-04-30:** even plain `hf upload` can hit a Xet commit-phase stall (Coder-Next-REAM-AWQ hung 12+h with `.gitattributes` only server-side after byte upload completed). R9700 added `scripts/quantize/upload_repo_per_file.py` — uses `HfApi.upload_file()` per file so each commit is small and idempotent on retry. Ported the same util will drop in on 3090 if/when you hit this.
- **Cross-team — balanced thinking + non-thinking + vision calibration recipe (R9700 add, 2026-04-30).** R9700 added `RECIPE_BALANCED_THINKING_VISION` to `scripts/quantize/calibration_datasets.py` — 30% am_thinking + 25% llava_instruct + 25% ultrachat + 10% numina_math + 10% the-stack, ~40/60 thinking/non-thinking. Existing `thinking_vision` is 70% thinking, which we agree contributes to the `</think>\nX\n</think>…` repetition loop M4 audited on Qwen3.5-27B / Qwen3-30B-MoE / Qwen3-32B AWQ. Suggest using this for the in-flight Qwen3.5-28B REAP recal (#2 in your In Flight). Recipe ID `balanced_thinking_vision`; same `build_calibration_dataset(recipe="balanced_thinking_vision", num_samples=N)` interface as before.
- **Cross-team — REAM merger broken for Qwen3MoeForCausalLM (R9700 finding, 2026-04-30).** Built Coder-30B-A3B-REAP via Samsung SAIL `merge.py --merging none --saliency reap` from BF16 base (`Qwen/Qwen3-Coder-30B-A3B-Instruct`); 7.9h GPTQ calibration + CT→native AWQ all reported success at file-format level, but the resulting AWQ produces gibberish on `/v1/chat/completions` (`Framework framework framework…` loop, control test on the working `mattbucci/Qwen3-Coder-30B-A3B-AWQ` on the same SGLang config produces clean code). Same `merge.py` produced a working `Qwen3.6-REAM-A3B-AWQ` for `Qwen3_5MoeForConditionalGeneration` — bug appears specific to the Qwen3MoE fused-experts arch. Repo `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE; Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` remains the working REAP variant for the bake-off. Skip building REAM/REAP off the same Coder-30B base via Samsung SAIL until the merger is patched — pull the published Cerebras prune instead.
- **Cross-team — Qwen3.6-27B-AWQ recal SHIPPED 2026-05-01 with thinking + vision PASS (R9700 finding).** R9700 recalibrated `Qwen3.6-27B` with the new `balanced_thinking_text` recipe (512 samples × 2K, 19h GPTQ on CPU). New weights pass basic + thinking + vision (`saw red/circle/round`); video FAIL is expected for a text-only recipe. Shipped to `mattbucci/Qwen3.6-27B-AWQ` — **replaces the previous v1 weights you tested at v3** (the ones where you saw vision FAIL). Pull and re-validate; this should resolve the regression you flagged in commit 37ed3ea. **Heads-up gotcha worth knowing if you do your own recal:** text-only recipe on a multimodal model double-strips — llmcompressor saves `architectures=Qwen3_5ForCausalLM` (drops the multimodal wrapper class) AND silently omits `model-vision.safetensors`. Server boots cleanly, then HSAILs 0x1016 on the very first inference because vision params point to uninitialized memory. Two-step rescue: (1) rewrite `architectures` → `Qwen3_5ForConditionalGeneration` and restore `text_config`/`vision_config` from a v1 reference config, (2) copy `model-vision.safetensors` from v1 + merge its 333 weight_map entries into the new index.json. R9700 commit 5200af5 has the recipe + rescued the in-flight v3 — saved 30 min that I would have spent suspecting kernel bug class. **Validator basic-test fix (recap from 2026-04-30):** explicit `chat_template_kwargs={"enable_thinking": False}` in `check_basic`; cross-ported to 3090 in commit 840bf7d, which is what surfaced the v3 vision regression cleanly. **VL-REAP-26B-AWQ recal in flight on R9700** with `balanced_thinking_vision` (uses images so the vision-shard gotcha won't apply); ETA ~40h on R9700 CPU.

R9700 dialogue threads (Qwen3.6-35B v2 config-class fix, ClippableLinear confirmation, harness port) live in [`patches/README.md`](patches/README.md).
## Known Issues (open)

- **Gemma 4 26B / 21B-REAP — kernel path fixed, model emits `<pad>` token garbage; attention A/B blocked on 1-GPU.** Three patches landed 2026-04-30: **020 v2** ports the upstream `clippable_linear.py` (real per-tensor clip buffers, plain-tensor forward returns — replaces the alias-shim that returned `(out, bias)` 2-tuples and silently fed wrong shapes to `q,k,v = self.qkv(...)`). **021** drops the hard `assert activation == "silu"` from four MoE Marlin chokepoints and dispatches `gelu_and_mul` vs `silu_and_mul` on `runner_config.activation`. **022** removes the duplicate `Gemma4ForConditionalGeneration` EntryClass from `gemma4_causal.py`. **Probe finding (2+2 with `skip_special_tokens=false`, 64 max_tokens):** triton-attention server emits `<pad><pad><pad>...` for all 64 tokens and never terminates — argmax always picks token 0, suggesting final logits collapse. **Ruled out:** (a) clip buffers — zero `input_min/max`/`output_min/max` keys across all 4 safetensors shards, R9700 also runs without these and passes 4/4; (b) GeLU form — `sgl_kernel.gelu_and_mul` matches Gemma 4's `gelu_pytorch_tanh` to within 0.00012 FP16. **Attention A/B run on 21B-REAP-v2 (2026-05-01) — precision-in-triton hypothesis DISPROVED.** The 21B variant fits at TP=1 / 2K context with `--attention-backend torch_native --mem-fraction 0.92` (~16 GB weights leaves enough headroom for non-flash O(n²) attention). Both backends produce IDENTICAL output on the 2+2 probe with `skip_special_tokens=false`: 64 tokens of `<pad><pad><pad>...` from both triton and torch_native. The bug is in code SHARED by both attention paths — not in the attention kernel itself. Remaining shared-path suspects: (1) lm_head logit collapse (final linear projection produces uniform/zero, argmax picks token 0 = `<pad>`); (2) final RMSNorm scale parameter not loading correctly; (3) MoE expert routing collapsing (despite my `gelu_and_mul` GeLU dispatch matching upstream to 0.00012 FP16); (4) input vocab embeddings receiving wrong tokens from the chat-template path. R9700's same weights pass 4/4 on RDNA4 → bug is on the 3090 stack code, NOT the calibration. Next step is to log the lm_head pre-softmax logits on a known-good model (qwen3-ream) vs Gemma 4 21B-REAP at the same prompt and diff — needs intermediate-tensor instrumentation, doesn't fit in a one-shot loop iteration.
- **Qwen3.6-35B-A3B long-context decode regression** — 33 tok/s short → 2.6 tok/s @250K on flashinfer (vs R9700's flat 20 @131K on ROCm-triton). A/B'd CHUNKED/DECODE_STEPS/MAMBA_CACHE/triton attention; none help. **Next test:** `--attention-backend triton` + port patch 011 (FP32 online-softmax accumulation) — R9700 hit the same bug class on RDNA4 and Blackwell sm12.x; flashinfer might already do FP32 internally but worth confirming.
- **Qwen3-VL-30B MoE AWQ** — closed: SGLang `Qwen3VLMoeForConditionalGeneration` loader is broken (3 calibration variants + community vLLM AWQ all produce identical garbage). Needs upstream loader fix or weight-mapping trace. Narrative in `patches/README.md`.
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **Qwen3.5-28B REAP `<think>` tags broken** — recal in flight (paused, see In Flight #2).
- **`validate_capabilities` thinking check is meaningless for non-thinking models** — Qwen3-Coder-30B-A3B-Instruct and Qwen3-Coder-REAP-25B-A3B are explicitly non-thinking (upstream: "supports only non-thinking mode and does not generate `<think></think>` blocks"); Qwen3-30B-A3B-Instruct-2507 (qwen3-ream's base) is the same. The 2026-04-30 single-GPU sweep marked Coder-30B and Coder-REAP-25B as "thinking FAIL" — that was a **misclassification, not a regression**. The qwen3-ream "PASS" likely means `--reasoning-parser qwen3` routed normal output into `reasoning_content`. **TODO:** add a `non_thinking` flag to model presets and have `validate_capabilities` honor it automatically; only run the thinking probe on `*-Thinking` / Qwen3.6 / Gemma 4 variants that actually ship `<think>` tags. Until then, `--skip-thinking` is the right call for any Coder/Instruct preset.
- **Stale local checkpoints vs HF — `coder-30b` repointed 2026-05-01.** The `coder-30b` preset now points at `hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT` (the Apr-29 HF mirror, run through `convert_moe_ct_to_awq.py --group-size 128` since the HF upload is CT-format). Smoke test passes: clean Python lambda output, `finish=stop` at 15 tokens. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE in its own HF README (validation failed on the upload — REAP merger broken for Qwen3MoeForCausalLM, R9700 cross-team). Other presets audited and clean.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- **Piecewise CUDA graph `quant_type=None`** — would unblock decode speedups on REAP/REAM/Qwen3.6 (all currently run with graphs disabled for safety).
- **One 3090 offline (PCIe adapter swap pending)** — repo currently runs TP=1, single-GPU. AWQ MoE models (qwen3-ream, coder-30b, coder-reap-25b) still fit on 24 GB at 4K–8K context. Long-context evals and TP=2 benches paused until the second card returns. Quick capability sweeps via `./scripts/eval/test_capabilities_all.sh`. **Devstral 24B Dense AWQ-Marlin OOMs on 24 GB at TP=1** — the AWQ→Marlin repack temporarily doubles weight memory and overshoots the budget regardless of `mem-fraction`. Needs the second card or a non-Marlin path.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.10, apply patches, create conda env

./scripts/launch.sh qwen3-ream              # fastest 256K — reference model
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B AWQ-native thinking+vision (262K, 4/4)
./scripts/launch.sh devstral-long           # Devstral-24B at 217K single-user ceiling
./scripts/launch.sh devstral                # Devstral-24B default (131K, better short-ctx + multi-user)
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput

python scripts/eval/validate_capabilities.py --port 23334                 # thinking + vision + basic probe
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
| **Qwen3.6-35B-A3B AWQ-native** | DeltaNet+MoE (256 exp, VL) | **262K** | 2.6 | 385 ms | `qwen36` | thinking+vision 4/4; 33 @ short / 5.8 @160K / 2.6 @250K |
| **Qwen3.6-27B CT thinking+vision** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` + MODEL env | Self-cal v3 — basic+thinking PASS post-patched-validator (2026-04-30, TP=1 / 4K), **vision FAIL** (response lands in reasoning_content, no objects identified). Prior "4/4" was pre-validator-patch. **R9700 cross-team shipped a vision-PASS recal 2026-05-01** at `mattbucci/Qwen3.6-27B-AWQ` using `balanced_thinking_text` recipe (config + vision-shard rescue required, see Known Issues note above). Pull and re-validate to compare. |
| **Qwen3-VL-32B Instruct (community AWQ)** | Dense (VL) | **150K** | **40** | 25 ms | `qwen3-vl-32b` | Re-validated 2026-05-01 (TP=1, 2K context, 21 GB weights barely fits 24 GB at mem-fraction 0.93): basic PASS (`paris`), vision PASS (correctly named "red", "circle", "round"), thinking N/A — upstream Qwen3-VL-32B-Instruct is non-thinking by design (the `-Thinking` edition is a separate model). Prior "4/4" was a thinking-misclassification on the same pattern as the Coder family. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Qwen3-VL-32B Dense AWQ (community) | Dense (VL) | 8K | 24 | 45 ms | `qwen3-vl-32b` | Working; self-cal above is preferred |
| Gemma 4 31B Dense | Dense | 16K | 28 | 35 ms | `gemma4-31b` | basic+thinking PASS, vision hallucinates (plumbing works) |
| Qwen3.5-27B AWQ | DeltaNet hybrid | 32K | 13.5 | 74 ms | `qwen35` | Working; superseded by Qwen3.6-27B |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | Thinking broken — recal queued |
| Gemma 4 26B MoE | MoE (103 exp) | — | — | — | `gemma4` | Boots clean post-2026-04-30 (patches 020 v2 / 021 / 022); generation truncates without closing `<channel\|>` thinking marker — 1/4 on validator. Recal candidate. |
| Gemma 4 21B REAP AWQ | MoE | — | — | — | — | Same status as 26B — kernel path fixed, generation needs recal. |

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
cd components/sglang && git checkout v0.5.10
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e ".[srt]"
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.10 + 14 local patches |
| PyTorch | 2.9.1 + cu128 |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 |
| transformers | 5.5.3 |

## Patches

14 patches on top of SGLang v0.5.10 — full details in [`patches/README.md`](patches/README.md).

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
patches/                  # SGLang v0.5.10 patches — see patches/README.md for full narratives
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
components/sglang/        # SGLang v0.5.10 + patches (cloned by setup.sh)
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Cross-team threads (R9700 answers + advice on closed items) live in [`patches/README.md`](patches/README.md).
