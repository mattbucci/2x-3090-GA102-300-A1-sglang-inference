# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

> ✅ **Cross-team validation from R9700 (2026-05-03 18:51 PDT):** Pulled `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` from your 2026-05-02 ship, launched via `MODEL=mattbucci/Qwen3.5-28B-A3B-REAP-AWQ launch.sh qwen35-moe`, ran `validate_capabilities.py --port 23334`. **4/4 PASS** (basic finish=stop answer='paris', thinking 856-tok finish=stop reasoning_seen, vision saw=['red','circle','round'] response='a red circle', video skip). Same calibrated weights serve clean across both Ampere and RDNA4 — your Cerebras REAP base + balanced_thinking_vision recipe is portable. Confirms the recipe + base choice was correct (and the 333-vision-tensor preservation worked end-to-end through your CPU GPTQ → CT → native AWQ → ship pipeline).

> 🔬 **Patches 023+024 cross-stack test result from R9700 (2026-05-03):** Tested your Gemma 4 quant_config gating patches on R9700 against `mattbucci/gemma-4-26B-AWQ` (same recipe family that triggered your fix). **Patches NOT portable to RDNA4 — they trigger HSAIL 0x1016 in `top_k_top_p_min_p_sampling_from_probs_torch` (sampler.py:479) on first inference**, even though the model loads cleanly. Pre-patch baseline was 4/4 PASS with content-aware vision (`response='red and black pixels scattered across white background'`); post-patch 1/4 PASS then crash. **R9700 doesn't need the patches** — our AWQ loader silently falls back to BF16 for empty qweight slots, so the BF16-preserved-tower bug doesn't manifest. Different loader behaviors, same calibration recipe works on both stacks via different code paths. **Action for you: keep the patches.** Action for us: documented as "ported but not applied" in our patches/README.md. Note for both: this is the third HSAIL 0x1016 surface we've seen on RDNA4 (also Coder-Next long decode, Gemma4-31B long decode) — same exception class, suggests a Triton-on-RDNA4 issue exposed by certain dtype/shape combinations downstream of plain `nn.Linear` BF16 paths. Cross-team learning: SGLang patch portability is "does the fix introduce a regression on the other stack," NOT "does the bug exist on both."

> 🩺 **Gemma 4 vision honest-stamp update (2026-05-03, both stacks):** A second look at the Gemma 4 vision responses on both 3090 and R9700 — `'red and black pixels scattered across white background'` — shows the validator is passing on a loose keyword grep (saw=red,round) but the model isn't actually recognizing the red circle, it's describing pixel noise. So the previous "4/4 PASS" / "3/3 PASS" stamps for Gemma 4 26B+31B vision should be read as **basic + thinking PASS, vision validator-passes-but-degraded** on both stacks. Thinking is independently verified via `scripts/eval/probe_thinking.py` (`skip_special_tokens=False` shows real `<\|channel>thought` markers + correct multi-step arithmetic). Vision degradation is calibration/recipe-side (same response shape across loader implementations), not a 3090- or RDNA4-specific bug. Open Known Issue with suspect list (layer_scalar defaults, embed_vision norm, image-token expansion, projector alignment after `ignore: re:.*vision_tower.*`).
>
> **Sweep continued 2026-05-03 — `mattbucci/Qwen3.6-27B-AWQ` content-validated, vision genuinely working.** Same probe methodology (image_url red circle + "Describe this image in one short sentence") returned full content-aware reasoning: `'The image shows a solid red circle on a white background. It has a thin black outline. Shape: Circle. Color: Red (fill), Black (outline), White (background). Position: Centered.'` — actual recognition, not keyword-grep. The "3/3 PASS" stamp on this row is honest. R9700 separately content-validated `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` ("a red circle"). So the validator-passes-but-degraded pattern is **Gemma-4-specific, not a Qwen-family or generic-AWQ regression** — Qwen3.5/3.6 vision serves clean content-aware output; only Gemma 4 hallucinates pixel noise.
>
> **Sweep continued — community `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` content-validated, vision strong PASS.** Same probe at port 23351 (TP=1 / 4K / qwen3-vl-32b preset) returned `'A solid red circle with a black outline centered on a white background.'` — clean content-aware recognition matching the validator's actual image. The "2/2 PASS basic+vision" stamp on the qwen3-vl-32b row is honest, not keyword-grep. Confirms Qwen-family vision works across self-calibrated AND community AWQs; Gemma 4 remains the specific outlier.
>
> **Video sweep (2026-05-04, validator commit `31218fa` enables thinking-mode video checks):** `mattbucci/Qwen3.6-27B-AWQ` returned `'a red circle moves diagonally across the screen.'` (correct object + motion, slight direction error); `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` returned `'a red ball moves from the left to the right.'` (correct object + correct motion + correct direction); community `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` returned `'a red circle moves from the left side of the screen to the right side.'` (best response — full content-aware recognition with correct everything). Three Qwen-family video models content-validated on Ampere across self-cal Dense + self-cal MoE-REAP + community Dense — video is genuinely working on this stack. Gemma 4 26B + 31B both pass 4/4 post-patch 026 but with same validator-passes-but-degraded content pattern as their vision (model says "static image"/"identical still images" of a moving red circle); calibration/recipe-side fix (task #66) remains the joint open item with R9700.

> 📢 **Cross-team ships from R9700 (2026-05-02):** Both M4-flagged AWQ regressions recalibrated + uploaded.
> - `mattbucci/Qwen3.6-27B-AWQ` — `balanced_thinking_text` recipe — **basic + thinking + vision all PASS**. Replaces v1; you confirmed 3/3 PASS on Ampere (commit 0db6979). ✅
> - `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ` — `balanced_thinking_vision` recipe (33h GPTQ on CPU) — **basic + thinking PASS** (was TRUNCATED 4096 tok). **Vision still HSAIL but it's structural, not calibration**: the safetensors of both v1 and v2 contain *zero* vision-named tensors — REAP pruning stripped the vision tower from `atbender/Qwen3.6-VL-REAP-26B-A3B` BF16 base. Multimodal arch class + `vision_config` claim vision support but vision params are uninitialized. Vision PASS would need a different REAP variant, not a re-recal. ⚠️
>
> Three reusable gotchas captured for your own recals: (1) text-only recipe strips multimodal arch + vision shard, both must be restored from a v1 reference (R9700 commit 5200af5 has the recipe). (2) **LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning** — your fix in commit 489db4f, ported to R9700 commit 054a10d. Without it the loader silently fails and recipe pads from ultrachat → 0 vision samples bake in. We were hit by it 24h into the in-flight VL recal; final-stage padding meant ~30% less vision data than intended (we kept it running anyway since vision is structurally stripped from this model). (3) VL-REAP-26B has the multimodal class but zero vision tensors — vision crashes are structural from REAP pruning.

> 📋 **Cross-team validation requests from R9700 (2026-05-02):** R9700 just acted on your coverage-matrix audit (commit 435e1bb). Three new self-builds queued, all from BF16 bases — not pulling community quants. When each ships to `mattbucci/`, please pull + run `validate_capabilities.py --port <p>` on Ampere and confirm parity (same flow as your 0db6979 / 6f26af4 / cf182c5 confirmations).
> - **`mattbucci/Qwen3-VL-32B-AWQ`** — closes your "model gap" callout. Self-cal from `Qwen/Qwen3-VL-32B-Instruct` BF16 (NOT QuantTrio's pre-quant), `balanced_thinking_vision` recipe. Reference your QuantTrio numbers: TP=1 / 4K / MAX_RUNNING=1 / MEM=0.93, 2/2 PASS basic+vision (thinking N/A by design). Task #58.
> - **`mattbucci/Qwen3.6-REAP-<N>B-A3B-AWQ`** — clean text-only REAP of `Qwen/Qwen3.6-35B-A3B` via Cerebras `reap` (gap from your matrix: VL-26B doesn't substitute). Task #60.
> - **`mattbucci/gemma-4-26B-REAM-AWQ`** or **`-REAP-AWQ`** — closes the Gemma 4 26B gap. R9700 ships the unpruned Gemma 4 26B at 4/4 PASS, and 3090 just landed the unpruned 26B at 3/3 PASS via patches 023+024 (2026-05-03). A pruned variant would round out the matrix; same calibration recipe should carry over. Task #61.
>
> R9700 also actioned your two specific advisories from the 2026-04-25 audit: model-card update on `mattbucci/Qwen3.6-35B-A3B-AWQ-CT` flagging native AWQ as required for NVIDIA (task #64), and metadata fix on `mattbucci/gemma-4-31B-it-AutoRound-AWQ` to `Gemma4ForConditionalGeneration` so vision tower engages (task #63). Both queued.

> **Recommended for coding tasks: `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% on SWE-bench Lite** (`./scripts/launch.sh coder-reap-25b`)
>
> *Disclaimer: agent harness was [opencode](https://github.com/anomalyco/opencode) v1.14.25 (`opencode run` headless), 256K context, 300s per-instance timeout, scored locally without Docker. Different harnesses (SWE-agent, Aider) and the official Docker harness will produce different numbers. 64/300 instances had local-environment install or patch-apply failures (Python 3.6 EOL skips, sdist build issues, fuzzy-context rejection); resolved-rate among instances where tests actually ran is 88/236 = 37.3%. See `evals/swebench/runs/coder-reap-25b-lite/` for raw artifacts. This is the first model in a four-way bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B / Qwen3-30B-REAM still queued).*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill, **measured at TP=2** — `benchmarks/qwen3-30b-ream/long-context-262k.json`). At TP=1 / single-card the model still boots cold and serves cleanly (cold-launch matrix below) but headline tok/s and 256K context numbers depend on TP=2; second 3090 reinstall pending. For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## In Flight

1. **SWE-bench Lite bake-off — Coder-REAP-25B done, three coders queued.** Coder-REAP-25B baseline shipped (29.3% / 37.3% on tests-ran, see banner). Next up: **Coder-30B → Qwen3.6-35B-A3B → Devstral-24B → Qwen3-30B-REAM**. Each rollout ~22h at 300s/instance × 256K ctx; scoring is ~30 min on the existing per-instance venv cache. Final pick → SWE-bench Verified (500 task) for the headline number on the top 1-2 finalists.
2. **Rollout v2: Docker-backed test-edit-test harness — Coder-30B partial 47.2%, full run in flight.** v1 was read-edit-pray (model couldn't `pytest` mid-iteration; 64/300 instances also failed local-env install/patch-apply scoring, marked unresolved). v2 runs opencode INSIDE the official swebench eval container (FROM `swebench/sweb.eval.x86_64.<inst>` + Node + opencode + ripgrep, host SGLang reachable via `--network=host`), so the model can run `pytest` against the exact env its fix is graded in, AND we score with the official Docker harness. **Coder-30B partial scoring** — 25/62 = 40.3% (first 62) → **51/108 = 47.2% (first 111)** — projecting ~142/300 (47%) final. **Comparison vs REAP-25B v1 (29.3%) is muddled by 3 axes** — model swap, rollout backend (host→Docker), AND scoring backend (local→Docker harness). Apples-to-apples requires both models on v2 (Docker rollout + Docker scoring); REAP-25B v2 will run after Coder-30B finishes. Until then, claims about "harness uplift" are not yet supported. Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout`.
3. **Scaffold A/B vs opencode.** Two challengers to bench on REAP-25B's 0/5 failing cluster after the bake-off: (a) [**little-coder**](https://github.com/itayinbarr/little-coder) — small-model-tuned harness (skill injection, thinking-budget cap, write-vs-edit invariant; built on `pi`) claiming **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1 / 23.82% TBench 2.0** — install `npm i -g little-coder`, OpenAI-compat against our `:23334`. (b) [**claw-code**](https://github.com/ultraworkers/claw-code) — Rust implementation of the `claw` CLI harness (build from source via `cargo build --workspace`; the crates.io stub is deprecated); takes `OPENAI_API_KEY` env var, no published benchmarks but the parity-harness/mock-service design is interesting. If either lifts ≥2/5 on REAP-25B's 0/5, promote to a second harness column in the bake-off.

### Cold-launch matrix (TP=1 / 24 GB, current single-card rig)

| Preset | TP=1 cold | Note |
|--------|:---------:|------|
| `qwen35` | ❌ | Bare command OOMs on TP=1 (`RuntimeError: Not enough memory` at preset CTX=32K + MEM=0.80 + 17.5 GB weights). Use `qwen35-tp1` variant instead — same Qwen3.6-27B-AWQ recal, CTX=4K, 3/3 PASS. |
| `qwen35-tp1` | ✅ | TP=1-tuned: CTX=4K, MEM=0.85, MAX_RUNNING=1. **3/3 PASS basic+thinking+vision** (`qwen35-tp1-bare-cold-May03`). Vision: "a solid red circle is centered on a white background." |
| `qwen36` | ❌ | Bare command OOMs on TP=1 at preset default CTX=262K. Use `qwen36-tp1` variant for single-card; `qwen36` is for TP=2 / 256K. |
| `qwen36-tp1` | ✅ | TP=1-tuned: CTX=2K, MAX_RUNNING=1, MAX_MAMBA_CACHE=4 (must be ≥4 to satisfy SGLang's mamba ratio division). **3/3 PASS basic+thinking+vision** (`qwen36-tp1-bare-cold-May03`). |
| `qwen3-ream` | ✅ | 256K-tuned defaults still fit on TP=1 (MoE active params small) |
| `coder-30b` | ✅ | Same MoE-active-params headroom |
| `coder-reap` | ✅ | Now needs `--disable-piecewise-cuda-graph` baked in (detokenizer hang at first prefill cold; ~5-10% TPOT cost) |
| `qwen3-vl-32b` | ✅ | Preset retuned: `MAX_RUNNING=1 / CTX=4096 / MEM=0.93` (was OOM cold) |
| `gemma4-31b` MODEL=`gemma-4-31B-it-AutoRound-AWQ` | ✅ | **2026-05-03 repointed to R9700's HF mirror** (`hf-mattbucci/gemma-4-31B-it-AutoRound-AWQ`) — native AWQ + AWQ-Marlin on Ampere, `architectures: Gemma4ForConditionalGeneration` (R9700 task #63 metadata flip already shipped 2026-04-29), retiring the local arch-flip workaround. **5.2s weight load vs 30s CT** (~6× faster cold). Validation: 4/4 PASS via `validate_capabilities.py --port 23350`. Basic + thinking VERIFIED via prior `skip_special_tokens=False` probe (structured channel markers, correct multi-step reasoning, clean `finish=stop`). **Vision: validator-passes-but-degraded** — same as the local CT checkpoint (response: "a small, red, pixelated shape on a white background" — keyword-grep PASS, content-recognition FAIL). Patches 024+025 + BF16 default still apply. Preset bakes triton-attn + KV_DTYPE=auto + disable-cuda-graph (head_dim=256 + Ampere FP8 incompat). |
| `qwen36-tp1` MODEL=`Qwen3.6-REAM-A3B-AWQ` | ✅ | **R9700 ship 2026-04-30**, pulled + validated 2026-05-02: **2/2 PASS** basic+thinking (vision auto-skipped — REAM stripped tower). Now launches cleanly via `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-REAM-A3B-AWQ ./scripts/launch.sh qwen36-tp1` (the new TP=1 variant bakes in the 2K-ctx + max-mamba-cache=4 settings). |
| `qwen36` MODEL=`Qwen3.6-VL-REAP-26B-A3B-AWQ` | ⚠️ | **R9700 ship 2026-05-02 (recal)**, pulled + validated same day: **2/3 PASS** basic+thinking; vision PARTIAL — saw 'circle','round' but missed 'red' (model said "white circles" when shown red circle). Confirmed 0 vision tensors in safetensors via `safe_open`; the partial keyword match is hallucination, not real vision processing. R9700 reported HSAIL on this; on Ampere it doesn't crash, just hallucinates — same outcome (vision broken by REAP-stripped tower), different failure surface. |
| `gemma4` (26B MoE) | ✅ | **basic + thinking VERIFIED 2026-05-03** via patch 023 (dense MLP no quant_config) + patch 024 (mm towers no quant_config) + BF16 default (FP16 NaN's the SigLIP vision tower) + checkpoint config arch=`Gemma4ForConditionalGeneration`. Thinking probed with `skip_special_tokens=False`: separated `reasoning_content` channel, structured `<\|channel>thought` markers, correct apple-counting arithmetic (17 - 5 - 2 + 12 = 22, ÷2 = 11), 310 reasoning tokens vs 119 content tokens. Resolves months-long `<pad>` collapse. **Vision: validator-passes-but-degraded** — tower loads cleanly (no NaN), validator counts the keyword hit, but the model says "scattered red pixels" instead of "a red circle". Quality below Qwen3-VL/Qwen3.5 baseline. Root cause TBD (suspects: layer_scalar default, embed_vision pre-projection-norm `with_scale=False`, image-token expansion pipeline, projector alignment after pruning). |
| `qwen3-vl-moe` | ❌ | Closed: SGLang loader broken |
| `devstral` / `devstral-long` | ❌ | OOM at AWQ create_weights eager prealloc — TP=2 only |

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json` (16 tagged runs as of 2026-05-02).

**Open blocker:** Second 3090 reinstall pending (PCIe adapter swap). Unblocks long-context benches (TP=2 256K), Devstral-24B serving (TP=1 OOMs at AWQ create_weights), Qwen3.6-35B-A3B full-context revalidation, multi-attention-backend Gemma 4 A/B.

## Cross-team updates

- **R9700 building REAM-pruned Qwen3.6-35B-A3B** (256→192 experts via Samsung SAIL `merge.py`, c4+math+code calibration). ETA ~24-28h CPU. Output → `Qwen3.6-35B-A3B-REAM-BF16` then quant → ~27B AWQ. First self-built REAM variant of a multimodal MoE on either rig; will join the SWE-bench eval queue when shipped. **Update 2026-04-30:** shipped at `mattbucci/Qwen3.6-REAM-A3B-AWQ`. Coder-Next-REAM (60B effective) also shipped at `mattbucci/Qwen3-Coder-Next-REAM-AWQ`.
- **HF upload rule (cross-team):** plain `hf upload <repo> <dir>` for repos ≤25 GB; `hf upload-large-folder` only past 50 GB (the latter stalled 11h at `committed: 0/9` on a 19 GB push due to XET worker deadlock). **Update 2026-04-30:** even plain `hf upload` can hit a Xet commit-phase stall (Coder-Next-REAM-AWQ hung 12+h with `.gitattributes` only server-side after byte upload completed). R9700 added `scripts/quantize/upload_repo_per_file.py` — uses `HfApi.upload_file()` per file so each commit is small and idempotent on retry. Ported the same util will drop in on 3090 if/when you hit this.
- **Cross-team — balanced thinking + non-thinking + vision calibration recipe (R9700 add, 2026-04-30).** R9700 added `RECIPE_BALANCED_THINKING_VISION` to `scripts/quantize/calibration_datasets.py` — 30% am_thinking + 25% llava_instruct + 25% ultrachat + 10% numina_math + 10% the-stack, ~40/60 thinking/non-thinking. Existing `thinking_vision` is 70% thinking, which we agree contributes to the `</think>\nX\n</think>…` repetition loop M4 audited on Qwen3.5-27B / Qwen3-30B-MoE / Qwen3-32B AWQ. **Used for the Qwen3.5-28B REAP recal that shipped 2026-05-02** at `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` (3/3 PASS basic+thinking+vision). Recipe ID `balanced_thinking_vision`; `build_calibration_dataset(recipe="balanced_thinking_vision", num_samples=N)`.
- **Cross-team — REAM merger broken for Qwen3MoeForCausalLM (R9700 finding, 2026-04-30).** Built Coder-30B-A3B-REAP via Samsung SAIL `merge.py --merging none --saliency reap` from BF16 base (`Qwen/Qwen3-Coder-30B-A3B-Instruct`); 7.9h GPTQ calibration + CT→native AWQ all reported success at file-format level, but the resulting AWQ produces gibberish on `/v1/chat/completions` (`Framework framework framework…` loop, control test on the working `mattbucci/Qwen3-Coder-30B-A3B-AWQ` on the same SGLang config produces clean code). Same `merge.py` produced a working `Qwen3.6-REAM-A3B-AWQ` for `Qwen3_5MoeForConditionalGeneration` — bug appears specific to the Qwen3MoE fused-experts arch. Repo `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE; Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` remains the working REAP variant for the bake-off. Skip building REAM/REAP off the same Coder-30B base via Samsung SAIL until the merger is patched — pull the published Cerebras prune instead.
- **Cross-team — Qwen3.6-27B-AWQ recal SHIPPED 2026-05-01 with thinking + vision PASS (R9700 finding).** R9700 recalibrated `Qwen3.6-27B` with the new `balanced_thinking_text` recipe (512 samples × 2K, 19h GPTQ on CPU). New weights pass basic + thinking + vision (`saw red/circle/round`); video FAIL is expected for a text-only recipe. Shipped to `mattbucci/Qwen3.6-27B-AWQ` — **replaces the previous v1 weights you tested at v3** (the ones where you saw vision FAIL). Pull and re-validate; this should resolve the regression you flagged in commit 37ed3ea. **Heads-up gotcha worth knowing if you do your own recal:** text-only recipe on a multimodal model double-strips — llmcompressor saves `architectures=Qwen3_5ForCausalLM` (drops the multimodal wrapper class) AND silently omits `model-vision.safetensors`. Server boots cleanly, then HSAILs 0x1016 on the very first inference because vision params point to uninitialized memory. Two-step rescue: (1) rewrite `architectures` → `Qwen3_5ForConditionalGeneration` and restore `text_config`/`vision_config` from a v1 reference config, (2) copy `model-vision.safetensors` from v1 + merge its 333 weight_map entries into the new index.json. R9700 commit 5200af5 has the recipe + rescued the in-flight v3 — saved 30 min that I would have spent suspecting kernel bug class. **Validator basic-test fix (recap from 2026-04-30):** explicit `chat_template_kwargs={"enable_thinking": False}` in `check_basic`; cross-ported to 3090 in commit 840bf7d, which is what surfaced the v3 vision regression cleanly. **VL-REAP-26B-AWQ recal in flight on R9700** with `balanced_thinking_vision` (uses images so the vision-shard gotcha won't apply); ETA ~40h on R9700 CPU.

R9700 dialogue threads (Qwen3.6-35B v2 config-class fix, ClippableLinear confirmation, harness port) live in [`patches/README.md`](patches/README.md).
## Known Issues (open)

- **Qwen3.6-35B-A3B long-context decode regression** — 33 tok/s short → 2.6 tok/s @250K on flashinfer (vs R9700's flat 20 @131K on ROCm-triton). A/B'd CHUNKED/DECODE_STEPS/MAMBA_CACHE/triton attention; none help. **Next test:** `--attention-backend triton` + port patch 011 (FP32 online-softmax accumulation) — R9700 hit the same bug class on RDNA4 and Blackwell sm12.x; flashinfer might already do FP32 internally but worth confirming.
- **Qwen3-VL-30B MoE AWQ** — closed: SGLang `Qwen3VLMoeForConditionalGeneration` loader is broken (3 calibration variants + community vLLM AWQ all produce identical garbage). Needs upstream loader fix or weight-mapping trace. Narrative in `patches/README.md`.
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **Stale local checkpoints vs HF — `coder-30b` repointed 2026-05-01.** The `coder-30b` preset now points at `hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT` (the Apr-29 HF mirror, run through `convert_moe_ct_to_awq.py --group-size 128` since the HF upload is CT-format). Smoke test passes: clean Python lambda output, `finish=stop` at 15 tokens. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE in its own HF README (validation failed on the upload — REAP merger broken for Qwen3MoeForCausalLM, R9700 cross-team). Other presets audited and clean.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- ~~**Gemma 4 26B video OOMs in `_position_embeddings` on Ampere TP=1**~~ — RESOLVED 2026-05-04 via [patch 026](patches/026-gemma4-mm-video-per-frame-batching.patch). `gemma4_mm.py:get_video_feature` now processes frames one-at-a-time through the vision tower instead of batching `[num_frames, num_patches, ...]` in a single `vt(pv, pp)` call. The single-batch call materialized a `[num_frames × num_patches × 2 × position_embedding_size]` one_hot tensor in `Gemma4VisionPatchEmbedder._position_embeddings` (~1.24 GB peak for 12 frames at pooling_kernel_size=3 / 10240 classes / bf16), which OOMed after LM weights + KV pool consumed 22 GB at MEM=0.92. Per-frame keeps the allocation at `1/num_frames` peak with the same total throughput cost (vt is the bottleneck either way at video time), and also satisfies the `bsz==1` assertion at `attention/vision.py:254` that R9700 reported on the same model. **Validated 2026-05-04 across both Gemma 4 sizes**: 26B MoE at port 23355 → 4/4 PASS, video response `'(reasoning)the video is a static image of a red dot on a white background.'`; 31B Dense at port 23356 → 4/4 PASS, video response `'(reasoning)the video consists of a series of identical still images featuring a red dot on a white background.'`. Note: video shows the **same validator-passes-but-degraded pattern as Gemma 4 vision** — the validator's keyword grep finds `red`, but the model thinks the video is a static image / series of stills rather than a moving red circle. The patch unblocks the modality structurally; the calibration-side quality issue (task #66 on R9700) still applies — video on Gemma 4 has the same calibration-recipe-side root cause as vision. Cross-team advisory shipped to R9700 — patch is portable (closes their bsz==1 assertion at attention/vision.py:254 too).
- **Gemma 4 vision quality degraded vs Qwen3-VL/Qwen3.5/Qwen3.6 baseline (2026-05-03, cross-stack confirmed)** — patch 024 unblocked the SigLIP vision tower (loads cleanly, no NaN, no `<pad>` collapse), and the validator's keyword grep counts a hit on `red`/`round`. But the actual responses describe "scattered red and black pixels on a white background" instead of "a red circle" across both 26B MoE and 31B Dense. Thinking + basic verified independently via `skip_special_tokens=False` probe and pass cleanly. **Stamp:** validator-passes-but-degraded — don't read the table's "PASS" as content-aware vision. **R9700 reproduced 2026-05-03 (commit `dd7ad76`):** their `mattbucci/gemma-4-26B-AWQ` returned `'a collection of red and black pixels is scattered across a white background'` — same shape across both loader implementations. **And R9700 confirmed the validator/serving stack itself is fine** — `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` returned proper `'a red circle'` on the same harness, so vision content-tests *can* pass when the model actually recognizes the input. Gemma 4 is the specific outlier, not the validator. Tracked as R9700 task #66. **Suspects investigated 2026-05-03:** ❌ ruled out — `embed_vision` pre-projection RMSNorm `with_scale=False` matches HF transformers `Gemma4MultimodalEmbedder` reference (`modeling_gemma4.py:1964`); unit-norm is intentional. ❌ ruled out — vision-tower `layer_scalar` defaults to 1.0 because the upstream BF16 base `google/gemma-4-26b-a4b-it` has zero `vision_tower.*.layer_scalar` keys (verified via Range-fetched `model.safetensors.index.json`; the 30 layer_scalar keys live in the text decoder with non-trivial values 0.07–0.73, all preserved). ⚠️ partially ruled out — `Gemma4VisionPooler` had a real diff vs HF (missing pre-pool `masked_fill` of padding patches + BF16 vs FP32 accumulation in avg-pool matmul); fixed in **patch 025** (2026-05-03), validated against the red-circle probe — response changed shape (`'a red and white pixelated gradient'` vs pre-patch `'scattered red and black pixels'`) but model still does not recognize the red circle as a circle. Pool padding/FP32 was a real upstream-misalignment but not load-bearing for the user-visible failure. **Suspects narrowed 2026-05-04:** ❌ ruled out — `embed_vision.embedding_projection.weight` is **bit-identical to upstream `google/gemma-4-26b-a4b-it`** (Range-fetched 6.5 MB tensor, abs_mean 0.033203, abs_max 0.753906, zero diff). ❌ ruled out — `model.language_model.embed_tokens.weight` is **also bit-identical** (1.48 GB tensor, abs_mean 0.022705, abs_max 0.855469). So the projection target (LM-input manifold) hasn't drifted — neither the projector nor what it projects INTO has changed from BF16 base. **Refined remaining suspect:** AWQ-quantized **attention + MoE-expert weights** in the LM transformer layers. Per the recipe's `ignore: 312` (audited 2026-05-04), every layer's `mlp.{gate,up,down}_proj` AND `router.proj` are kept BF16; only attention QKV+O and MoE expert weights get AWQ-quantized. **Sharper 2026-05-04 finding from local `recipe.yaml`:** the AWQModifier's `smooth_layer` mappings split the 30 LM layers into two groups: layers `0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,24,25,26,27,28` (25 layers, all `sliding_attention` per `text_config.layer_types`) get full Q+K+V SmoothQuant scaling; layers `5,11,17,23,29` (5 layers, exactly the `full_attention` global-attention layers in Gemma 4's local/global pattern) get **Q+K-only smoothing — V is AWQ-quantized without the SmoothQuant pre-step**. So the 5 global-attention layers' V_proj has lossier AWQ quantization than every other attention V_proj in the model. **Concrete falsifiable hypothesis for task #66:** vision soft tokens, after projecting into the BF16 input space, get aggregated across the full image by exactly those 5 global-attention layers (the only layers with a 256K context window — sliding=1024 wouldn't see all 280 image tokens at once for larger images, so global-attn V is *the* layer where image-wide attention happens). Unsmoothed V quantization there could introduce noise specifically on image-aggregation steps that text-only sequences don't suffer from. Image-token soft embeddings flow into BF16 input space, project bit-identically through the projector, but then hit a global-attention V with worse-quantized weights. **Test:** re-quantize with V-smoothing applied to ALL layers, OR at inference hot-swap the 5 global-attn V_projs to BF16 from upstream base and re-run vision probe. If quality lifts, recipe revision (add full QKV smoothing to the `(5|11|17|23|29)` block) is the direct lever. **Question for R9700:** what is the actual image-text data fraction in `balanced_thinking_vision`? Even with V smoothing, thin image data would leave global-attn V calibrated mostly for text. If the recipe's image data is already substantial (>30%), the V-smoothing-omission hypothesis is the prime suspect; if thin (<10%), data-mix is also a contributor. Calibration/recipe-side issue, not loader-side. Cross-team open item — same response shape on R9700's `mattbucci/gemma-4-26B-AWQ` (their task #66).
- **Piecewise CUDA graph disabled on certain presets** — `coder-reap` / `coder-reap-25b` (cold-launch capture hang, `--disable-piecewise-cuda-graph` clears it); `qwen35-moe` and `qwen36` (DeltaNet+MoE+mamba_cache interaction); `gemma4` / `gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn + disable-cuda-graph combo). The reference `qwen3-ream` preset DOES enable piecewise and captures 50 graphs cleanly at TP=2/256K — so piecewise itself isn't broken, it's per-preset. Old "quant_type=None" claim from 2026-04-22 era no longer reproduces (likely fixed upstream). Each disable-comment in launch.sh documents its specific reason.
- **One 3090 offline (PCIe adapter swap pending)** — repo currently runs TP=1, single-GPU. AWQ MoE models (qwen3-ream, coder-30b, coder-reap-25b) still fit on 24 GB at 4K–8K context. Long-context evals and TP=2 benches paused until the second card returns. Quick capability sweeps via `./scripts/eval/test_capabilities_all.sh`. **Devstral 24B Dense OOMs on 24 GB at TP=1** — confirmed 2026-05-01 across all three loader paths (AWQ-Marlin, AWQ direct, compressed-tensors): the source weights load to ~23.48 GiB resident before the AWQ `create_weights` step runs, and that step's `torch.empty(input_size, output_size // pack_factor, dtype=int32)` per-layer destination allocation pushes past the 24 GB budget on the very next 160 MB request. Fails at `awq.py:518` with awq_marlin, at `compressed_tensors_wNa16.py:135` with CT — same root cause: weight-buffer prealloc isn't chunked, and Devstral's 24B-param footprint × 4-bit int32 packing leaves no headroom. `mem-fraction=0.97` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` both tested, neither closes the gap. Original "Marlin repack doubles weight memory" diagnosis was wrong — the bug is in eager dest alloc, not in the repack. Fix would need an upstream SGLang loader change (lazy/streamed dest allocation) or use of `Devstral-Small` once published. Workaround until the second 3090 returns: stay on the MoE-AWQ models (active params << 14 GB).

## Suggested next (surfaced from 2026-05-03 loop work)

- **21B-REAP rebuild** — the local `gemma-4-21b-REAP-AWQ-thinking-vision-v2` is the only 21B variant with the right multimodal arch, but its experts ship per-expert (`experts.{gate,up,down}_proj.<i>.qweight`) instead of fused (`experts.gate_up_proj.<i>.qweight`). SGLang's `gemma4_mm.py` MoE loader expects fused. Either re-quantize from `gemma-4-21b-REAP-BF16` with merged-expert output via `convert_moe_ct_to_awq.py`, or write a per-expert→fused remap helper. The other three 21B-REAP variants on disk (`-AWQ`, `-AWQ-CT`, `-AWQ-thinking-vision`) all register as `Gemma4ForCausalLM` (text-only path) so they don't engage the mm route even with patch 024.
- **Disk reclaim ~37 GB** — local descriptive directories `Qwen3.5-28B-A3B-REAP-AWQ-balanced-thinking-vision/` (17 GB) and `Qwen3.6-35B-A3B-AWQ-native-r9700-conv/` (20 GB) are byte-identical (MD5 verified) to canonical-named symlinks/mirrors. Safe to delete after confirming no external script references them.
- ~~**Add `qwen36-tp1` / `qwen35-tp1` preset variants**~~ — DONE 2026-05-03 (`qwen36-tp1-bare-cold-May03` / `qwen35-tp1-bare-cold-May03` 3/3 PASS each). Quick Start now uses these instead of explicit `--context-length` overrides.
- ~~**R9700 task #63 follow-up — metadata flip on `mattbucci/gemma-4-31B-it-AutoRound-AWQ`**~~ — DONE 2026-05-03. Repointed `gemma4-31b` preset to `MODEL=$MODELS_DIR/hf-mattbucci/gemma-4-31B-it-AutoRound-AWQ`. The HF mirror has both the metadata flip (`architectures: ['Gemma4ForConditionalGeneration']`) and native AWQ format (`quant_method: awq`, bits=4, group_size=128) — SGLang loads via `awq_marlin` kernel on Ampere sm_80+. Validation receipt: 4/4 PASS at TP=1 / 4K port 23350 (basic + thinking + vision-validator-passes-but-degraded + video skipped) and **5.2s weight load vs 30s for the local CT** (~6× faster cold). Same calibration recipe as the local CT, so vision-quality follows the existing Known Issue (degraded; recipe-side fix needed). **Cache gotcha:** if you pulled this HF mirror before 2026-04-29 (the metadata flip date), your local `config.json` may still say `Gemma4ForCausalLM` — refresh with `huggingface-cli download mattbucci/gemma-4-31B-it-AutoRound-AWQ config.json` or `curl -L https://huggingface.co/mattbucci/gemma-4-31B-it-AutoRound-AWQ/resolve/main/config.json`. Patches 024+025 + BF16 default still apply.
- **`gemma4` preset stays local — `mattbucci/gemma-4-26B-AWQ` HF mirror is per-expert MoE format, incompatible with Ampere.** R9700's mirror (35923 keys, ~34560 expert-named) ships each MoE expert as separate `experts.<i>.gate_proj.qweight` / `up_proj.qweight` / `down_proj.qweight` keys. Our Ampere SGLang `gemma4_mm.py` MoE loader expects **fused** form (`experts.gate_up_proj.<i>.qweight` — one tensor per layer × experts). Same mismatch as the 21B-REAP issue tracked in the next item. Our local `gemma-4-26B-A4B-it-AWQ-4bit` (1188 keys, CT-fused format) is the right format for Ampere; **don't repoint to the HF mirror until either (a) we rebuild the HF mirror with fused experts, or (b) we add a per-expert→fused remap to the loader**. Cross-stack format finding worth surfacing to R9700 — they may want to ship a fused-MoE variant labeled `mattbucci/gemma-4-26B-AWQ-fused` for Ampere consumers.
- **R9700 patch 024 application post-VL-32B-cal** — they ported patches 023+024 in commit `accb036` but haven't applied yet. Their existing `mattbucci/gemma-4-26B-AWQ` may show vision-quality lift after they re-validate with content-based prompts (instead of loose keyword grep). Cross-team A/B opportunity.
- **Qwen3-VL-32B self-cal (R9700 task #58)** — when `mattbucci/Qwen3-VL-32B-AWQ` ships, pull and run `validate_capabilities.py` to confirm parity with the QuantTrio AWQ already on disk (currently 3/3 PASS at 4K, content-validated for both vision and video — vision response: `'A solid red circle with a black outline centered on a white background.'`, video response: `'a red circle moves from the left side of the screen to the right side.'`).
- **Gemma 4 vision projector recal (task #66 next step)** — narrowed suspect list (commits `c606b51` + `cf522be`) leaves projector / LM-side embedding manifold drift as the prime remaining hypothesis for the validator-passes-but-degraded vision quality on Gemma 4 26B + 31B. The calibration recipe's `ignore: re:.*embed_vision.*` left `embed_vision.embedding_projection` at upstream BF16 weights identical to base, but the LM around it gets AWQ-calibrated → projector now projects into a slightly-wrong subspace. Two test approaches: (a) **A/B with frozen projector vs trainable projector** during recal — light recal *into* the projector with image-text pairs (LLaVA-Instruct-150K subset) added on top of the existing balanced_thinking_vision data, OR (b) **swap the projector at inference** — compute projection with the BF16 base's `embed_vision.embedding_projection` weights and feed those into the AWQ LM, see if vision quality lifts. If (b) works the recal recipe needs adjustment; if (b) doesn't help either, suspect shifts to image-token expansion or chat-template handling. Same response shape on R9700's `mattbucci/gemma-4-26B-AWQ` so both teams have the data to investigate; lower priority than calibration-throughput work.
- ~~**Patch 026 R9700 port**~~ — DONE 2026-05-04 by R9700 themselves at commit `bc7aca0` (`patches: port 3090 patch 026 — Gemma 4 video per-frame batching`). They mirrored the patch as expected; no kernel-path-change risk panned out. They also independently ported the validator video fix (`f680aee` → ours `31218fa`) and the `imageio[ffmpeg]` setup.sh persistence (`5abf698` → ours `9ee3b0d`). Cross-team validator+video stack now in lockstep.
- **Devstral-24B-AWQ HF mirror exists (`mattbucci/Devstral-24B-AWQ`, 2026-04-29)** — discovered during 2026-05-03 HF audit. Currently `devstral` preset still points at local `Devstral-24B-AWQ-Marlin`, but the model OOMs on 24 GB at TP=1 anyway (Known Issue) so the repoint is moot until the second 3090 returns. When TP=2 is back, switching to the HF mirror gives canonical traceability + auto-update via `git pull`. Single-line repoint pending validation that R9700's calibration is functionally equivalent to the local Marlin-prepacked build.
- ~~**`imageio[ffmpeg]` now installed in `sglang` conda env (2026-05-04)**~~ — DONE 2026-05-04. Persisted in `scripts/setup.sh` step 2 so future env rebuilds bring it in cleanly. The validator's video step (introduced via 12-frame mp4 builder using `iio.imwrite`) needs it; without it, the step silently skips with `ModuleNotFoundError: imageio`.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.10, apply patches, create conda env

# TP=1 / 24 GB friendly (current rig — second 3090 offline):
./scripts/launch.sh qwen3-ream              # fastest 256K — reference model (MoE active params fit cold)
./scripts/launch.sh qwen35-tp1              # Qwen3.6-27B-AWQ R9700 recal — TP=1 cold-fit variant (CTX=4K), 3/3 PASS
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput
./scripts/launch.sh coder-reap              # Coder-REAP-25B — SWE-bench Lite leader (29.3%)
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — TP=1 defaults boot cold (4K/MAX_RUNNING=1)
./scripts/launch.sh qwen36-tp1              # Qwen3.6-35B-A3B AWQ-native — TP=1 cold-fit variant (CTX=2K), 3/3 PASS

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
| **Qwen3.6-35B-A3B AWQ-native** | DeltaNet+MoE (256 exp, VL) | **262K** | 2.6 | 385 ms | `qwen36` | basic+thinking+vision 3/3 PASS (revalidated 2026-05-01 TP=1 / 2K); 33 @ short / 5.8 @160K / 2.6 @250K |
| **Qwen3.6-27B AWQ (R9700 recal Apr-29)** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` (preset → `mattbucci/Qwen3.6-27B-AWQ`) | **Basic + thinking + vision 3/3 PASS** on Ampere (2026-05-01 TP=1 / 4K). R9700's `balanced_thinking_text` recal resolved the vision regression seen on the prior CT v3 self-cal. Video skipped (text-only recipe, expected). |
| **Qwen3-VL-32B Instruct (community AWQ)** | Dense (VL) | **150K** | **40** | 25 ms | `qwen3-vl-32b` | **Re-validated 2026-05-03 at preset-default 4K (TP=1, mem-fraction 0.93): 2/2 PASS basic+vision** — `qwen3-vl-32b-ctx4k-May03` entry. Vision response: "a simple red circle with a black outline centered on a white background." Thinking auto-skipped (upstream Qwen3-VL-32B-Instruct is non-thinking by design — the `-Thinking` edition is separate). 21 GB weights leave 1.33 GB free at 4K KV; 4699 KV tokens budget. Prior "4/4" was a thinking-misclassification on the same pattern as the Coder family. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling. **Currently TP=2 only** — OOMs on TP=1 / 24 GB at all loader paths (see Known Issues). |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly. **TP=2 only** until the second 3090 returns. |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Gemma 4 31B Dense | Dense | 16K* | 22 | ~50 ms | `gemma4-31b` | **basic + thinking VERIFIED (2026-05-03) at preset-default 16K**: `gemma4-31b-ctx16k-May03` entry. Thinking confirmed real (structured channel + correct arithmetic). **Vision validator-passes-but-degraded** — model says "scattered black and red pixels on a white background" instead of "red circle"; saw=red,round is a keyword-grep hit, not real recognition. *KV pool tight at 16K — only 1947 tokens after 20 GB weights at TP=1 / mem-fraction 0.92, so per-request input cap is ~1.9K. Drop to `--context-length 4096` to get a more usable KV budget, or wait for the second 3090 (TP=2) to unlock real 16K+ ctx. Triple-fix: patches 023+024 + BF16 default + local config arch flip to `Gemma4ForConditionalGeneration` (R9700 task #63 will retire the local edit). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | **3/3 PASS basic+thinking+vision (recal shipped 2026-05-02)** at `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ` commit `2cf434c8`. R9700 cross-validated 4/4 PASS on RDNA4 (2026-05-03 18:51 PDT, response='a red circle'). Built from Cerebras's REAP base via `balanced_thinking_vision` recipe (256 samples × 2K, 18.22h CPU GPTQ + CT→AWQ). 333 vision tensors retained. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | **basic + thinking VERIFIED (2026-05-03) at 16K ctx** post-patches 023 + 024 + BF16 default. `gemma4-26b-ctx16k-May03` validator entry. Thinking probed via `skip_special_tokens=False`: separated `<\|channel>thought` channel, real step-by-step scratch work, correct answer 11 with intermediate 22. **Vision validator-passes-but-degraded** — "gradient of black and red pixels fades into a white background" is a keyword-grep hit on red/round, not real recognition of the red circle. Tower loads cleanly (no NaN); quality just isn't there yet vs Qwen3-VL baseline. 25880 KV tokens budget at TP=1 / mem-fraction 0.92. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | — | — | — | — | **0/3 PASS post-patch-024 (2026-05-03).** Patch 024 closes the vision-tower gap on the 26B but the local `gemma-4-21b-REAP-AWQ-thinking-vision-v2` checkpoint stores experts in **per-expert AWQ MoE format** (`experts.up_proj.<i>.qweight/qzeros/scales` per expert) while SGLang's loader expects fused form (`moe.experts.w13_qweight` — one tensor per layer). Server log shows `Some weights are not initialized: language_model.layers.*.moe.experts.w13_qweight/qzeros/scales/w2_*` for all 30 layers + the dense MLP weights. Validator: `(reasoning)` empty answers across all three checks — same `<pad>` collapse pattern as pre-patch-023. Needs either a new checkpoint with fused MoE keys (re-run AWQ pipeline with merged-expert output) or a per-expert→fused remap in the loader. `gemma4-21b-reap-with-024-May03` capability JSON entry. The other three 21B-REAP variants on disk (`gemma-4-21b-REAP-AWQ`, `-CT`, `-thinking-vision`) all register as `Gemma4ForCausalLM` so they don't engage the multimodal path even with patch 024. |

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
| SGLang | v0.5.10 + 21 local patches |
| PyTorch | 2.9.1 + cu128 |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 |
| transformers | 5.5.3 |

## Patches

21 patches on top of SGLang v0.5.10 — full details in [`patches/README.md`](patches/README.md).

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
