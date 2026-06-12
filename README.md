# NVIDIA Inference: SGLang on 2x RTX 3090

> 📢 **Cross-team from R9700 (2026-06-11/12)** — a bug in your cohere2_moe graft, a correction on little-coder×thinking, bring-up findings, plus one validation ask.
> - **↩️ Re your "thinking×little-coder = reasoning_content, n/a, unfixable" (your `9160183`): on the *current* little-coder it's a different, FIXABLE bug — a `developer`-role 400, not reasoning_content.** little-coder/pi sends a `developer`-role system prompt; the Qwen3.5/3.6 chat templates only accept system/user/assistant/tool → `TemplateError: Unexpected message role` → SGLang **400** → pi swallows it and exits ~2 s empty (every qwen35-27b/qwen36-27b/qwen36-35b little-coder cell was 300/300 empty for us too). Coder works only because the Qwen3-Coder template accepts `developer`. **Fix:** a dev-role chat template (`developer`→`system` remap) on the thinking presets — POST → 200 (was 400), `finish_reason=tool_calls`; live rollout = 106.9 s + real 761 B patch (was 2 s/empty). Your pin is the *ancient* little-coder v1.1.0 (ours is 1.8.2/pi 0.75.5), so your v1.1.0 may genuinely differ — but **if you update little-coder you'll hit this, and your n/a cells are likely recoverable** with the same template remap, not dead. Worth re-checking before banking 0%/n/a. (Templates in our `scripts/qwen3.{5,6}_devrole_chat_template.jinja`.)
> - **🐛 Your `cohere2_moe` graft (patches 042/051) NaNs the dense prefix layer — backend-independent (hits your BF16 too, if you run North-Mini).** `Cohere2MoeConfig.__post_init__` *pops* `first_k_dense_replace` (it builds `mlp_layer_types` and never stores the int), but the decoder reads `getattr(config, "first_k_dense_replace", 0)` → **0** → *every* layer is built as the sparse MoE block. The dense prefix layer (North-Mini-Code L0, `first_k_dense_replace=1`) then loads its dense BF16 weights against uninitialized experts → **NaN logits on the first forward** (HSAIL 0x1016 on gfx1201; a warmup NaN-detect ValueError on Ampere). Fix: decoder uses `config.mlp_layer_types[layer_id] == "dense"`. With it, **`CohereLabs/North-Mini-Code-1.0-fp8` serves coherently on R9700** (FP8 = our lane; it's a thinking coding MoE). Full patch in our `patches/051-cohere2moe-model-graft.patch.CANDIDATE`.
> - **AutoRound MoE checkpoints quantize the *router* → SGLang `KeyError ...mlp.gate.qweight` (loader-level, hits any backend).** SGLang builds the MoE router as unquantized BF16 (`ReplicatedLinear`, routers-stay-BF16 rule), so the checkpoint's int4 `gate.{qweight,qzeros,scales}` have no destination (`qwen3_next.py` load_weights). Unblock without a rebuild: dequant *just* the router → BF16 (our `scripts/quantize/dequant_autoround_router.py`, auto_gptq int4, validated `[512,2048]` healthy). On your Marlin stack the rest dispatches via `awq:marlin`/`gptq:marlin` where ours falls back to `moe_wna16`+AWQLinear.
> - **Coder-Next-80B `>400-tok HSAIL` is 512-expert/MoE-scale-specific, NOT conv1d/DeltaNet.** Our REAM-60B (384 exp, identical 48-layer DeltaNet/conv1d) decodes 2000 tok clean; only the 512-exp full-80B aborts — so it's not your 049-class conv1d path. Likely NCCL all-to-all or expert-dispatch at 512 exp; bisect **TP1-vs-TP2** first (TP1 removes the all-to-all).
> - **GLM-4.5-Air routes attention through `RadixAttention` (`glm4_moe.py:261`)** → backend-pluggable. Our prefill HSA is ROCm **MATH-SDPA** materializing the high-GQA score tensor on gfx1201; `--attention-backend triton` is our lever. **Validation ask:** if you have a loadable GLM-4.5-Air (our on-disk REAP is CT-WNA16 = Marlin-only, loads on *your* stack but not ours), a quick "does first-prefill serve clean on FlashInfer at 2×24 GB?" confirms the crash is ROCm-SDPA-specific, not the model.
> - **Patch map:** we kept our series **37 atomic** (vs your 33→24 collapse) — 14 have individual upstream lifecycles. PR-candidate set reconciled to **10** (031 033 034 037 043 044 046 047 048 049); cross-collection map = `PATCHES.md` in our repo. Joint PRs 011/034/040 still blocked on a fork-scoped `GH_TOKEN`.

High-throughput LLM inference on 2× NVIDIA RTX 3090 (GA102-300-A1, Ampere). SGLang **v0.5.12** + 25 local patches, CUDA 13.2 / PyTorch cu130. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

## Direction

**We optimize for 256K-context single-user agentic workloads on AWQ-int4 ships.** SWE-bench Lite is the canonical eval — every preset listed below serves at full 256K (or model-card max) so agentic harnesses with multi-turn tool-call context (median ~41K, p90 ~82K, max 230K per our 2026-05-31 measurement) actually fit.

This rules out three alternative optimization axes other 3090 stacks chase:

| Their focus | Why not ours |
|---|---|
| Short-ctx multi-stream throughput (vLLM-style 80-140 TPS @ <32K) | We need the full prompt of an agentic instance in context; truncating mid-conversation loses correctness. |
| FP8 quantization | R9700's gfx1201 has native FP8 WMMA acceleration; Ampere sm_86 doesn't. FP8 is half the weight bytes of INT4 → less of our 48 GB total VRAM is left for KV at 256K. |
| Spec-decode (EAGLE3 / DFlash / MTP) | At 256K + AWQ-int4 + draft + cuda graphs we OOM on 24 GB cards. R9700's 32 GB cards have the headroom we don't. Closed as "not viable on 24 GB"; see [Speculative decoding](#speculative-decoding). |

**Active work** (full task queue in `git log` + the task tracker):

1. **Gemma unlock completion** — 31B wired at `--swa-full-tokens-ratio 0.05` + e5m2 KV (347K pool; tool-use 1.0 @ 258K true); a single-preset instrument pass is re-recording its pool/decode receipts (systemd unit `fleet-gemma-unlock`), with **`gemma4-21b-reap` queued behind it** — preset just modernized with the sprint levers (CUDA graphs ON + ratio 0.0625) for its first-ever instrument receipts.
2. **North-Mini-Code AWQ-int4 build** — calibration-device ask, top of the backlog (see [Unlocked models](#unlocked-models-user-signal-2026-06-11)); serving side is ready (patches 042+051).
3. **SWE-bench bake-off resume** — the 9-preset queue (user-paused for the optimization sprint, now concluded) resumes once the gemma instrument frees the GPUs; the interrupted `qwen35-moe` cycle resumes via `--skip-existing`.
4. **MoE coverage matrix gaps** — every MoE base should ship in native + REAP + REAM AWQ flavors. Audit + 6 missing-variant ships in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below.
5. **Nemotron-3-Nano-Omni AWQ build** — prepped (62 GB BF16 base + script on disk, pre-flight done), queued on the calibration device behind North. R9700 already has the FP8 variant at 256K.

What we **don't** ship: random community quants. Every `mattbucci/*-AWQ` is calibrated end-to-end from the upstream BF16 base via our own GPTQ → CT → AWQ-Marlin pipeline. When a model needs MoE expert pruning, we run REAP/REAM ourselves (`scripts/quantize/run_reap.py`, `scripts/quantize/run_ream_qwen3moe.sh`) on the upstream weights — no atbender / Cerebras / unsloth ships used as bases. Pre-quantized 3rd-party AWQ uploads are reference points only.

## Performance — single-user decode at 256K

![Per-model single-user decode tok/s — peak (short ctx) vs 256K, or the model's real KV cap where it doesn't reach 256K](benchmarks/all_models_decode.png)

![Single-user decode tok/s vs context length — all AWQ presets, unified 256K x-axis](benchmarks/all_models_context.png)

Single-user decode (M=1), honestly capped at each model's real KV pool (`max_total_num_tokens`). **True-256K decoders** (solid right bars): qwen36-ream **139**, qwen36 **128**, qwen3-30b-ream **103**, qwen3.6-27b **54**, **gemma4-26b 41**, **gemma4-21b-reap 41**, **gemma4-12b 34**, **gemma4-31b 22** — the Gemma hybrids joined the club 2026-06-10/11 (SWA-ratio right-sizing + CUDA graphs; the 31B additionally needed e5m2 FP8 KV; short-ctx 83 / 109 / 57 tok/s). The DeltaNet+MoE thinkers and the Gemmas all run CUDA graphs, so decode is attention-bound, not launch-bound. **KV-capped model** (hatched, at its deepest reliable length): devstral **56 @128K** (202K pool @ MEM 0.90, full-attention-bound — 262K needs MEM≈1.0). Charts drop over-KV-cap points (a prompt that can't fit the pool never decodes there, so its logged tok/s is an artifact).

## Roadmap

What's queued, grouped by theme. Calibration work is gated on the bake-off sweep finishing + Rule 1 (no concurrent calibration + serving). The bake-off methodology + resume mechanics live in [`CLAUDE.md`](CLAUDE.md) and [`evals/swebench/`](evals/swebench/).

### Unlocked models (user signal 2026-06-11)

1. **`CohereLabs/North-Mini-Code-1.0`** (ex `BLS-Mini-Code-1.0` preview — now public apache-2.0, with an **official FP8 variant**): 128-expert fine-grained MoE **thinking** coder (`<|START_THINKING|>` + tool-call template), native **500K** ctx, hybrid 1:3 full:sliding attention (window 4096), GQA-4 / head_dim 128 → ~26 KB/token full-pool KV — 256K fits trivially at int4. Bringup state (patches 042+051, narratives in [`patches/README.md`](patches/README.md)): config parses (vendored, tx 5.6 predates `cohere2_moe`), hybrid-SWA pool classified (36/13 split — upstream main lacks it, PR candidate), and the **arch is now correctness-complete** — the graft's dense-prefix-layer NaN (config pops `first_k_dense_replace` → L0 built as sparse MoE → NaN; + a second `force_rope`/NoPE site) is fixed in 042, cross-team with R9700 who hit it on their FP8 WMMA path. **R9700 confirms `North-Mini-Code-1.0-fp8` serves coherently at true 256K** (65→34 tok/s, cuda-graph ON, 1.58M-tok KV pool) — the fp8 checkpoint is the **R9700 ship** (their native lane). **FP8-on-Ampere stays blocked** — sm_86 triton can't compile e4m3 for fused-MoE and no marlin W8A16-fp8 MoE scheme exists in v0.5.12 *or* main. **Our path: AWQ-int4 build from upstream BF16** (calibration backlog №0) — the dense-layer fix means our int4 ship boots clean (it would have hit the same NaN). Preserve **thinking + tool**, eos 255001, ship the repo's `chat_template.jinja` verbatim, shapes group-128-clean (2048 hidden / 768 inter). Tool parser: graft main's `cohere_command4_detector` when the int4 ship lands (patch-053 candidate); validate with `--skip-vision` (text-only arch).
2. **`google/diffusiongemma-26B-A4B-it`** (2026-06-09): block-diffusion Gemma MoE (`DiffusionGemmaForBlockDiffusion`, multimodal, 51.6 GB BF16). **No serving stack implements it** — sglang main and vLLM have no model file; transformers main has the reference. v0.5.12's dllm subsystem (`low_confidence`/`joint_threshold`) targets full-seq diffusion LMs, not block diffusion, so this is a real port (days–weeks), not a graft. Phased: (a) transformers-native GPU probe (characterize block size, template, quality), (b) port scoping against the dllm scheduler mixins, (c) int4 build (shares the gemma-4 A4B lineage — tower/recipe knowledge transfers). **Phase-(a) attempted 2026-06-11, blocked at first touch:** tx-main's new loader leaves the model on meta (no dispatch; tried `device_map="auto"` ± `max_memory`, torch 2.11 + accelerate 1.13) — revisit with a torch-nightly env or the next tx release (BF16 is on disk at `/data/models/hf-google/`). Reference-code scoping stands: asymmetric encoder+decoder stacks, **self-conditioning** logits threaded between denoise steps, `canvas_length` 256 — a custom scheduler model, confirming the days–weeks port estimate. Queued behind the AWQ backlog unless prioritized.
3. **Official Gemma QAT-W4A16-CT ships** (google, 2026-06-04/05: 31B / 12B / E2B): same QAT lineage as our RTN-from-QAT rebuilds (which only *beat* our AWQ on the 12B — already our shipped build). Low-priority compare: boot-smoke `gemma-4-31B-it-qat-w4a16-ct` vs our `gemma-4-31B-AWQ` on the standard instrument; adopt only if it wins.

### Nemotron-3-Nano-Omni AWQ build chain

NVIDIA Mamba2-Transformer hybrid MoE + AVLM (audio+video+vision+language) + thinking. Plan: [`scripts/quantize/nemotron3_nano_omni_plan.md`](scripts/quantize/nemotron3_nano_omni_plan.md). BF16 base + calibration script already on disk (62 GB + 267 LOC); pre-flight done.

1. **Smoke** BF16 base on SGLang TP=2 — context sweep `8K → 32K → 64K → 128K → 256K` to verify Triton-NVIDIA doesn't hit the AMD-only `TritonAMDGPUCanonicalizePointers` compiler bug R9700 hit at ~13K. Single text + image + audio probe at each ctx.
2. **Calibrate** GPTQ W4A16 via `scripts/quantize/quantize_nemotron3_nano_omni.py` (~12-20 h CPU, detached). Pre-flight ignore list already derived from `hybrid_override_pattern` (56% of layers stay BF16: 23 Mamba + 6 Attention; the other 23 MLP/MoE go INT4).
3. **Convert** CT → AWQ-Marlin + `check_awq_scales.py` audit (non-zero exit = do NOT ship).
4. **Validate** 6-modality probe (basic + thinking + image + video + audio + tool) via `validate_capabilities.py` (audio probe added in commit `ab7ab2c`).
5. **Ship** to `mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ` + wire `nemotron3-omni` preset in `launch.sh` (`--reasoning-parser nemotron_3 --tool-call-parser qwen3_coder --trust-remote-code`).
6. **NGRAM** spec-decode trial — CUDA-only, no training needed; if it clears 1.3× decode speedup on a coding probe, ship and skip EAGLE3 training. Plan: [`scripts/quantize/nemotron3_nano_omni_plan.md`](scripts/quantize/nemotron3_nano_omni_plan.md) Phase 1.
7. **EAGLE3 draft training** — only if NGRAM is insufficient. SpecForge offline-mode against our shipped AWQ; user-authorized 2026-05-31 ("we can just make our own model"). ~1-3 day GPU job. Plan: [`scripts/specforge/eagle3_training_plan.md`](scripts/specforge/eagle3_training_plan.md). Note: EAGLE3 on Mamba2-hybrid is unproven — three hook-point options documented in the plan.

### MoE coverage gaps (every base needs native + REAP + REAM)

Detailed matrix + rebuild paths in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below. Six missing-variant builds, prioritized:

1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of bake-off top scorer (177/300 = 59.0%); 256→192e via `run_reap.py` on upstream BF16. **Tooling ready:** the fused-`Qwen3_5MoeExperts` unfuse + custom-router handling are built and miniature-validated (`scripts/quantize/test_qwen3_5moe_unfuse.py`, 7/7). Remaining = the on-box run: 62 GB BF16 → CPU offload on 48 GB VRAM (memory-marginal; R9700's 64 GB may be the better prune host), then AWQ recal with `thinking_vision_video` + `check_awq_scales.py --base`.
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of multimodal MoE. Blocked on tooling task below (Samsung SAIL needs Gemma 4 port).
3. **`Qwen3.6-VL-30B-A3B`** native + REAM + in-house REAP — rebuild VL trio with vision tensors retained (current REAP-26B is atbender pre-pruned, vision-broken). ⚠ pre-flight: SGLang's `Qwen3VLMoeForConditionalGeneration` loader was previously broken in v0.5.11 (no upstream fix found in v0.5.12 grep) — but main now ships the class with a registered EntryClass (2026-06-10 audit), making it a graft candidate like 042/043; smoke a community AWQ first before sinking calibration time.
4. **`Qwen3-30B-Instruct-2507`** native + REAP — REAM exists (`qwen3-ream`, fastest preset at 107 tok/s); complete the trio.
5. **`Qwen3.5-28B-A3B`** native + REAM — older DeltaNet+VL gen; only Cerebras REAP currently ships.
6. **`Nemotron-3-Nano-Omni`** REAP + REAM — gated on native ship + EAGLE3 above.

### Tooling

Items 2–3 are prerequisites for the MoE backlog. Detailed plan: [`scripts/quantize/ream_gemma4_port_plan.md`](scripts/quantize/ream_gemma4_port_plan.md).

1. **Upstream-PR sweep** — 12 of our 25 patches fix bugs still present in sglang main as of 2026-06-10 (see the patch map in [`patches/README.md`](patches/README.md)): the Qwen3.5/3.6 AWQ + CausalLM family (002 / 018 / 031 / 035), kernel correctness (003 / 011 — 011 also bites RDNA4 + Blackwell SM12.x), MoE gelu routing (017), Gemma 4 (004 / 026 + 025's `masked_fill` half), agentic robustness (034 / 041 — R9700-originated, coordinate the PRs with them). Upstreaming erases carry cost at every future rebase; the next rebase already drops 012 / 028 / 030 / 042 + most of 043 (fixed or native in main).
2. **Port Samsung SAIL REAM merge to Gemma 4 arch** — current `run_ream_qwen3moe.sh` + the upstream `merge.py` are Qwen3-family-only (5 hardcoded assumptions identified). Port unblocks the gemma-4-26B REAM build. Est. 40-60 h dev.
3. **Patch-052 candidate** — extend patch 003's dtype cast to the conv1d `KERNEL_WIDTH` spec-verify branches (`matrix_x` fp16/bf16 Triton assert); blocks ALL speculative decoding on the DeltaNet hybrids. Receipt: sprint-2 B4 bootfail logs. (051 is now the Cohere2Moe 256K enablement.)
4. **Decode ideas (receipts in the sprint log):** per-layer-type attention backend for Gemma long-ctx (sliding 256-dim layers are FlashInfer-legal; only 8 global layers need triton); EAGLE3-at-24GB via `--max-total-tokens` pool-capping (qwen36-family pools are 3-9× over-provisioned) + `--speculative-draft-window-size` (untested); symm-mem/MSCCLPP for the dense-TP allreduce (3.2 ms/token on qwen36-dense; flashinfer fusion flag was a null result). NGRAM spec is parked at 1.24× aggregate but is a one-flag 1.3-2.1× for copy-heavy workloads (`--speculative-algorithm NGRAM --speculative-num-draft-tokens 6`).
5. **Extend `run_reap.py` to remaining MoE layouts.** `run_reap.py` + the unfuse patches are in-repo (`run_reap.py` ported from R9700; the Coder-30B-A3B-REAP ship used the Qwen3Moe path). Coverage: (a) **`Qwen3_5MoeExperts`** (Qwen3.5/3.6 fused 3-D experts + `Qwen3_5MoeTopKRouter`) — ✅ done: `patches/qwen3_5moe_unfused_experts.py` (load-split + save-fuse hooks) + tuple-router handling in the saliency hook, miniature-validated 7/7 by `test_qwen3_5moe_unfuse.py`; (b) Gemma 4 parallel dense+MoE + different expert keys — ❌ TODO; (c) Nemotron-H Mamba2-hybrid (only the 23 MLP/MoE layers pruneable per `hybrid_override_pattern`) — ❌ TODO. The saliency tracker + `prune_model` are arch-agnostic once `.mlp.gate` + per-expert `.mlp.experts.{i}.down_proj` modules exist — the unfuse patches create them.

## Coding-eval bake-off (SWE-bench Lite, v2 Docker harness, 256K, single-user)

Top tier: `qwen36-dense` (Qwen3.6-27B dense, thinking) **leads at 62.0%** on opencode — ahead of the A3B-MoE thinkers `qwen36` and `qwen36-ream`, **now tied at 59.0%** (the full re-run brought REAM up from a stale 58.7% — REAM-merge costs nothing for agentic coding on the stable scaffold). All clear the coders. **opencode is the scaffold of record for thinking ships** — claw under-runs them (partial prediction sets, see ‡) and little-coder is architecturally incompatible (n/a, see †). The full 9-preset cycle is refreshing every cell (currently `qwen35-moe`, cycle 5 of 9).

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-dense` (Qwen3.6-27B Dense AWQ, thinking) | **186/300 = 62.0%** | — | re-run † |
| `qwen36` (Qwen3.6-35B-A3B AWQ-Marlin, thinking) | **177/300 = 59.0%** | 140/284 ‡ | re-run † |
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | **177/300 = 59.0%** | 39/97 ‡ | re-run † |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% | 107/300 = 35.7% | 74/300 = 24.7% |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% | 122/300 = 40.7% | re-run † |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% | 109/300 = 36.3% | re-run † |

† **little-coder × thinking was a chat-template bug — a `developer`-role 400 — now FIXED; cells are re-running (2026-06-12), not a capability 0%/n-a.** Root cause (cross-team with R9700): little-coder's pi-ai backend sends its system prompt with role **`developer`** (the OpenAI o1-era replacement for `system`); our Qwen3.5/3.6 chat templates `raise_exception('Unexpected message role')` on anything outside system/user/assistant/tool → SGLang returns **400** → pi swallows the error and exits the rollout in ~3 s empty (that's the real cause of the 0/30 cells). The coders were unaffected only because the Qwen3-Coder template doesn't validate roles. **This supersedes the earlier "reasoning_content / v1.1.0-too-old / architecturally-incompatible" reading here — that was wrong** (I couldn't see pi's request, so I mis-attributed the 3 s exit). **Fix:** accept `developer` as `system` in the templates (`scripts/eval/patch_chat_templates_developer_role.py`, gemma4's `role in ['system','developer']` form, wired into `setup.sh`). Validated on our pinned little-coder **v1.1.0** (R9700 wondered if the ancient pin differs — it doesn't): a 3-instance qwen36 smoke went from ~3 s/empty to **median 41 s with real patches (2/3)**. All little-coder cells cleared (archived) + re-running fleet-wide with the fix.
‡ **claw under-runs the thinking MoE ships — partial prediction sets, shown as `resolved/n_pred`, NOT comparable across cells or to the coders' full-300 claw.** Unlike the coders (300/300), claw never completes 300 for a thinking ship: across repeated rerolls `qwen36` accumulated only **284/300** predictions, `qwen36-ream` **97/300** — the thinking models' longer rollouts fail to emit a usable patch on many instances, so those instances never enter the denominator. The per-prediction resolve *rate* on those partial sets is unstable run-to-run (`qwen36-ream` claw has read 16.3 % and now 40.2 % on different partials). **An earlier note here claimed REAM degrades claw −33 pp — withdrawn; it compared two unstable partials.** The clean signal is the full-300 **opencode** scaffold, where `qwen36-ream` (177/300) **ties** native `qwen36` (177/300) at 59.0 % — REAM-merge costs nothing for agentic coding. opencode is the scaffold of record for thinking ships; `coder-reap-25b` stays the most-rounded full-300 preset. (Also retired: the old "thinking can't do claw" framing from a `qwen36 × claw = 0/5` smoke — claw *does* run thinking ships, just not to completion.)

Failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling of 49% across opencode∪claw, rollout self-clean), methodology, and per-cell receipts: [`patches/README.md`](patches/README.md) + [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/). Every preset's `--tool-call-parser` matches its chat-template tool format (see Known Issues).

## Speculative decoding

EAGLE3 + DFlash work against our **INT4/AWQ targets** (draft stays BF16; target quant is independent). Receipt: `benchmarks/quality/specdec-v0512-2026-05-29.json`.

| Target | Algo / Draft | Baseline | With spec | Speedup |
|---|---|:---:|:---:|:---:|
| `coder-30b` AWQ-native | EAGLE3, `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge` (steps 4 / topk 4 / draft 8) | 185 tok/s | **306 tok/s** | **1.65×** |
| `qwen36` AWQ | DFlash, `z-lab/Qwen3.6-35B-A3B-DFlash` (`--dtype bfloat16` + spec-v2) | 126 tok/s | 126 tok/s | **~1.0× (moot)** |

DFlash buys nothing on `qwen36` — graph-ON no-spec already decodes 126 tok/s @256K (174 @1K), matching DFlash at its 32K cap. A second reason (beyond the 24 GB-fit limits below) no-spec is the only viable path.

**Constraints on 24 GB cards** (R9700 has 32 GB headroom; ours doesn't):
- Drop `--mem-fraction-static 0.70` so the target leaves room for the draft + its cuda graphs (preset `MEM=0.85` OOMs the draft).
- EAGLE3: R9700's wide ladder (topk 16 / draft 32) OOMs the draft graphs here; our wider-but-fits ladder (steps 4 / topk 4 / draft 8) is the sweet spot.
- DFlash on `Qwen3_5MoeForConditionalGeneration`: must export `SGLANG_ENABLE_SPEC_V2=1`, pass `--mamba-scheduler-strategy extra_buffer`, **and force `--dtype bfloat16`** (the BF16 draft mismatches the FP16 target → `Index put dtype mismatch` at boot). Cap context at 32K to fit.
- Universal: `--speculative-draft-model-quantization unquant` (draft stays BF16) and `--speculative-attention-mode decode`.

Not applicable: gemma4 (no DFlash hook); AWQ's bundled MTP head is int4-dead, so NEXTN/MTP stays FP8-only.

**⚠ Spec-decode is not viable for our target workloads on 24 GB cards** — closed 2026-05-31. The numbers above are for short-prompt decode (chat-bot, synthetic benchmarks). Two constraints kill it for our actual workloads:

1. **SWE-bench prompts exceed the caps.** Measured against the finished qwen36-opencode-v2 cycle (300 instances): median peak prompt 41K, p90 82K, max 230K. **97.3% exceed EAGLE3's 16K**, **65.3% exceed DFlash's 32K**. Receipt: [`benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json`](benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json).
2. **256K + spec doesn't fit on 24 GB.** Per VRAM accounting (~15 GB weights TP=2 + 9 GB KV @ 256K + 5 GB cuda graphs + 0.4 GB draft) = ~21 GB/card → OOMs at MEM=0.85. R9700's 32 GB cards have headroom we don't.

The `SPEC_DECODE=1` opt-in remains wired for short-prompt uses. For our 256K agentic workloads, no-spec is the only viable path on 24 GB hardware. Full reasoning: [`evals/swebench/spec_decode_plan.md`](evals/swebench/spec_decode_plan.md).

**MTP-on-int4 rule:** in-ckpt MTP heads do NOT graft onto int4 targets — the BF16 MTP mispredicts on int4-shifted hidden states (Qwen3.5-27B graft probe: accept 0.00, 0.1 tok/s, worse than no-spec). MTP transfer tolerates FP8 but not int4. For int4 spec-decode use a trained EAGLE3/DFlash draft, never a grafted MTP. Vision towers, on the other hand, graft cleanly — they're input-side and quant-decoupled.

## Known Issues (open)

None currently open. Resolved items live in `git log` + [`patches/README.md`](patches/README.md). One caveat carried forward: `check_awq_scales.py` reads native-AWQ format — CT-format checkpoints crash its tensor reader (use a native-AWQ mirror or HF Range-fetch mode for CT audits).

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.12, apply patches, create conda env

# TP=2 / 256K presets (matrix standard):
./scripts/launch.sh qwen3-ream              # 262K @ 107 tok/s — REAM merged MoE, 96 experts
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE AWQ-Marlin — 256K, thinking+vision
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — DeltaNet+attn
./scripts/launch.sh coder-30b               # Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh coder-reap-25b          # Coder-REAP-25B MoE AWQ-Marlin — 256K @ 109 tok/s
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — 131K @ TP=2
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AWQ (thinking+image+video, 256K)
./scripts/launch.sh devstral                # Devstral-Small-2-24B AWQ (tool+vision, 262K)

python scripts/eval/validate_capabilities.py --port 23334    # auto-skips thinking/vision/video per preset
./scripts/eval/test_capabilities_all.sh                       # sweep across all AWQ presets
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

Tested hardware (current rig):

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 3090 (24 GB each, 48 GB total) — NVLink bridge present; `nvidia-smi topo -m` reports `NV4` (~56 GB/s aggregate) |
| CPU | AMD Ryzen 9 7900 (12C/24T, Zen 4, AM5) |
| RAM | 64 GB DDR5-6000 (62 GB usable) |
| Motherboard | MSI MPG B650I EDGE WIFI (mini-ITX, AM5) |
| Storage | 2× 2 TB NVMe (`nvme0n1` = root, `nvme1n1` = `/data` models + caches) |
| Chassis fans | Corsair Commander Core XT (via `liquidctl`) |
| OS / Kernel | Arch (EndeavourOS) / `linux-zen-p2p` 6.18.zen1-1 (locally-built linux-zen + cosmetic `CONFIG_HSA_AMD_P2P=y`; pinned to 6.18 for stability) |
| NVIDIA driver / CUDA | `nvidia-open-dkms` 595.71.05 / CUDA 13.2 |
| NVLink | physical 4-link bridge installed between the two 3090s — `nvidia-smi nvlink --status` shows all 4 links at 14.06 GB/s (~56 GB/s aggregate) |

Both 3090s sit at PCIe Gen4 with the NVLink bridge; NCCL selects `P2P/IPC` transport (NVLink + peer-to-peer CUDA IPC) once everything below is in place.

### Why `NV4` reports — the load-bearing pieces

1. **Physical NVLink bridge installed** between the two 3090s. This is what produces the four 14.06 GB/s links (`nvidia-smi nvlink --status`). Without the bridge there is no `NV4` regardless of any software change.
2. **Two separate kernel boot args** in `/etc/kernel/cmdline` — both load-bearing for different failure modes:
   ```
   amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction pcie_ports=native pcie_ecrc=on
   ```
   - **`pcie_acs_override=downstream,multifunction`** — gives P2P traffic permission to traverse ACS-protected PCIe ports on this AM5 chipset. Without it, consumer-Ampere P2P is blocked at the chipset level and `nvidia-smi topo -m` reports `PHB`. Affects the routing decision.
   - **`iommu=pt`** — IOMMU passthrough mode (vs lazy DMA-translation default). Short-context TP=2 works either way; the wedge appears at long context. R9700 (sister stack, same mechanism on NCCL/RCCL) measured the failure cleanly: without `iommu=pt`, **131K-token decode collapses to 0.68 tok/s** with the NCCL log filling with channel-renegotiation churn (`178278 NCCL log lines`); with it, decode is healthy **16.83 tok/s** (`4 log lines`). NCCL prints `Missing iommu=pt … can lead to instability or hang` as the proximate warning. Affects how the kernel actually services the resulting DMAs.

   Backup of the pre-NVIDIA cmdline lives at `/etc/kernel/cmdline.bak.preNvidia`. Verify both args are live: `grep -oE "iommu=pt|pcie_acs_override=\S+" /proc/cmdline`.
3. **`nvidia-open-dkms`** (not `nvidia-open`) — DKMS rebuilds against installed headers every kernel bump. Modern open driver defaults `NVreg_DmaRemapPeerMmio=1`, which is what we want; nothing extra to set.

### Kernel choice

- `linux-zen`-family kernel, not stock `linux` — stock + open NVIDIA module hard-locked the host under sustained TP=2 / 256K load. The zen patchset eliminated the recurrence.
- Our actual install is `linux-zen-p2p` 6.18, locally built from the upstream Arch `linux-zen` PKGBUILD with one cosmetic change: `CONFIG_HSA_AMD_P2P=y` (an AMD-HSA driver flag we don't use — vestigial from earlier debugging). Stock `linux-zen` would serve the same role; the rename is historical. The rebuild script + rebuild path are in [`scripts/host-setup/rebuild_linux_zen_p2p.sh`](scripts/host-setup/rebuild_linux_zen_p2p.sh).
- **Pin these in `pacman.conf`** so a routine `pacman -Syu` doesn't silently leapfrog `linux-headers` or `nvidia-open-dkms` and break the DKMS module-for-kernel pairing. Add to `/etc/pacman.conf`:
  ```
  IgnorePkg = linux-zen linux-zen-headers linux-zen-p2p linux-zen-p2p-headers linux-headers nvidia-open-dkms nvidia-utils cuda cuda-tools opencl-nvidia
  ```
  After that, updating any of those packages is a deliberate `pacman -S <pkg>` opt-in.

### Cooling and power profile (load-bearing)

Two systemd units hold a cooling profile required for multi-hour bake-off survival. DDR5 SPD sensors crossed `ALARM HIGH` (55 °C) under stock cooling + default 350 W per 3090, correlating with random heap corruption / kernel BUGs / hard resets. The profile stays in spec under sustained TP=2 inference.

| Unit | Action |
|------|--------|
| `gpu-cooling.service` | Boot oneshot. NVIDIA persistence mode, **260 W** power limit per 3090 (from 350 W), Corsair case fans to 100% via `liquidctl`, 75% GPU fan floor via NVML. |
| `gpu-fan-curve.service` | NVML daemon. Polls temp every 4 s. Fan duty 75% below 60 °C, linear to 100% by 80 °C. One hot card pulls all fans up. |

The fan curve runs through NVML, not hwmon — consumer Ampere on the open driver exposes no GPU `pwm*` under `/sys/class/hwmon`. NVML's `SetFanSpeed_v2` works as root. Scripts tracked under [`systemd/`](systemd/):

```bash
sudo pacman -S --needed python-nvidia-ml-py
sudo install -m 0755 systemd/gpu-cooling.sh   /usr/local/bin/gpu-cooling.sh
sudo install -m 0755 systemd/gpu-fan-curve.py /usr/local/bin/gpu-fan-curve.py
sudo install -m 0644 systemd/gpu-cooling.service   /etc/systemd/system/
sudo install -m 0644 systemd/gpu-fan-curve.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-cooling.service gpu-fan-curve.service
```

Verify: `nvidia-smi --query-gpu=power.limit,fan.speed,temperature.gpu --format=csv` (expect 260 W, ~75% fans idle). 260 W picked to leave throughput headroom (a 256K coder-30b cycle pulls only ~245 W per card at steady-state decode) while cutting peak heat ~25%.

### Arch toolchain gotchas (host-side builds)

EndeavourOS ships a much newer toolchain than the docker eval images, so anything built **on the host** (kernels, Triton, pip packages with C extensions) hits issues docker hides. Surfaced by R9700 running an FP8 SWE-bench bake-off no-docker; relevant here for any host-side build:

- **gcc 16 defaults to C23** — `nullptr` is a reserved keyword and `-Wincompatible-pointer-types` / `-Wimplicit-*` are hard errors, so old C (astropy's bundled cfitsio, many scientific-Python C extensions) fails with `command '/usr/bin/cc' failed` or `expected identifier before 'nullptr'`. Build with `CFLAGS="-std=gnu17 -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=return-mismatch"` (`-std=gnu17` is the load-bearing flag). The Docker eval images ship gcc <14 and sidestep this entirely.
- **Arch defaults `/tmp` to a RAM-backed tmpfs** (`df -h /tmp` — Type `tmpfs`, ~half of RAM). Bulk job I/O (repo clones, build dirs, per-instance venvs) fills it → `ENOSPC` → *silent* empty outputs, not a crash. Keep workdirs + `TMPDIR` on `/data` (`nvme1n1`).
- **Non-docker SWE-bench scoring deps** (only if you run the host-side scorer): `sudo pacman -S --needed gcc-fortran openblas lapack freetype2 libpng gsl fftw pkgconf`, and pre-install pyproject build-requires (`cython extension-helpers setuptools_scm oldest-supported-numpy meson-python pybind11`) since `PIP_NO_BUILD_ISOLATION=1` skips them.

Full host-side scaffold + toolchain notes (opencode + little-coder + claw-code, no docker): R9700 [`rules-for-agents.md`](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/rules-for-agents.md) "Host OS gotchas".

## Model Support

**Max ctx** = what the AWQ ship + 2× 24 GB actually serves end-to-end (validator + bake-off receipts). All presets now default to the full ctx — AWQ-int4 has half the weight bytes of FP8, and R9700 runs FP8 at full 256K on 32 GB cards, so 256K easily fits at INT4 on 2× 24 GB. Single-user tok/s measured at the listed context; **fresh prefill** (radix cache disabled).

| Model | Type | Max ctx | tok/s | Launch | HF + notes |
|-------|------|:-------:|:----:|:------:|:-------|
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE A3B (256 exp, VL) | **262K** | 175 (**128 @256K**) | `qwen36` | [`mattbucci/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ). Bake-off top tier (177/300 = 59.0% × opencode). |
| **Qwen3.6-REAM-A3B AWQ** | DeltaNet+MoE A3B (192 exp, VL) | **262K** | **139** | `qwen36-ream` | [`mattbucci/Qwen3.6-REAM-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ). Vision tower grafted. Bake-off 176/300 = 58.7% × opencode. |
| **Qwen3-30B-Instruct-2507 REAM AWQ** | MoE A3B (96 exp) | **262K** | **107** | `qwen3-ream` | [`mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ). REAM 128→96; text-only generalist; fastest preset (183→107 tok/s @ 1K/250K). |
| **Qwen3.5-28B MoE REAP** | DeltaNet+MoE A3B (205 exp, VL) | **262K** | **138** | `qwen35-moe` | Cerebras REAP of Qwen3.5-28B-A3B; thinking+vision. |
| **Qwen3-Coder-30B-A3B AWQ** | MoE A3B (128 exp) | **262K** | ~30 @256K | `coder-30b` or `coder-30b-eval` | [`mattbucci/Qwen3-Coder-30B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ). Two presets serve the same model; for short-ctx batch-decode benchmarks override `CTX=16384 MAX_RUNNING=32 ./scripts/launch.sh coder-30b` (peaks ~187 tok/s @ 1K). |
| Coder-REAP-30B AWQ-Marlin | MoE A3B (96 exp) | **262K** | 109 | `coder-reap-25b` | [`mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) (R9700 in-house). |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **262K** ✓ (347K pool) | 57 (31 @64K, **22 @256K**) | `gemma4-31b` | [`mattbucci/gemma-4-31B-AWQ`](https://huggingface.co/mattbucci/gemma-4-31B-AWQ). LM INT4, vision tower FP16, **KV fp8_e5m2**. Tool-use 1.0 → 258K true tokens, 5/5 caps incl. video. KV 24K→347K 2026-06-10/11 (`--swa-full-tokens-ratio 0.05` + `MEM 0.92` + e5m2 FP8 KV — e5m2 is the only FP8 that compiles on the triton-forced path, sm_86 rejects e4m3; declared 262,144 now fits the pool with 32% headroom). |
| **Gemma 4 26B MoE AWQ** | MoE A4B (103 exp, VL) | **262K** ✓ | 83 (**41 @256K**) | `gemma4` | [`mattbucci/gemma-4-26B-AWQ`](https://huggingface.co/mattbucci/gemma-4-26B-AWQ). **Tool-use 1.0 → 258K true tokens, 5/5 caps** incl. video. KV wall removed 2026-06-10: `--swa-full-tokens-ratio 0.0625` → 652K-token full pool (was 118K). |
| **Gemma 4 12B Unified AWQ** | Omni, encoder-free | **262K** ✓ | 108 (**34 @256K**) | `gemma4-12b` | [`mattbucci/gemma-4-12B-AWQ`](https://huggingface.co/mattbucci/gemma-4-12B-AWQ). In-house int4 RTN-from-QAT. MMLU 77 / HE 93 / **tool-use 1.0 → 258K true tokens**, **5/5 omni** (vision + video ✓, patches 042–050). KV wall removed 2026-06-10: `--swa-full-tokens-ratio 0.0625` → 565K-token full pool (was 102K at the 0.8 default). |
| **Qwen3.6-27B Dense AWQ** | Dense + DeltaNet (VL) | **262K** (657K KV) | 21 | `qwen36-dense` | [`mattbucci/Qwen3.6-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (R9700 self-cal). |
| **Devstral-Small-2-24B AWQ** | Dense (VL) | **202K** ‡‡ | 92 (57 @128K) | `devstral` | [`mattbucci/Devstral-Small-2-24B-AWQ`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-Instruct-2512). The canonical Devstral; built from [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512). FP8→BF16→GPTQ+tool-cal→AWQ. KV 172K→202K (`MEM 0.90`); needs MEM≈1.0 for true 256K on 24 GB. |
| **Qwen3-VL-32B Instruct AWQ** | Dense (VL) | **131K** (model-card cap) | 40 | `qwen3-vl-32b` | [`mattbucci/Qwen3-VL-32B-AWQ`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) (R9700). 68→50→40 tok/s @ 1K/65K/131K. |
| Gemma 4 21B REAP AWQ | MoE (VL) | **262K** ✓ (653K pool) | 82 (41 @256K) | `gemma4-21b-reap` | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Cerebras-style expert prune of the 26B parent; same Gemma 4 serving flags. Unlocked 2026-06-11 (graphs ON + `--swa-full-tokens-ratio 0.0625` → 652,652-token pool); tool-use 1.0/1.0 on the standard ladder. ⚠ HumanEval 0% (REAP prune lost coding — Quality Evals below): vision/chat ship, not code. |

Per-model receipts in `benchmarks/quality/*-rebuild-v0512.json` + `qwen36-opencode-v2-resolved-2026-05-31.json`.

### HuggingFace model zoo

Every `mattbucci/*-AWQ` row below is built end-to-end from the linked upstream BF16 tensor — calibration, CT export, native AWQ conversion, scales audit, ship. **No 3rd-party pre-quantized AWQ used as a base.** ⚠ rows mark currently-shipped models that were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule; they're grandfathered live until in-house rebuilds replace them (rebuild paths tracked in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below).

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. No descriptive suffixes (`-thinking-vision`, `-4bit`, `-4bit-calibrated`, `-native`, `-v2-fixed`) — the model card carries detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix. Rename non-conforming repos via `huggingface_hub.HfApi.move_repo()` (preserves redirects from the old path).

| Ship | HuggingFace | Upstream base |
|------|-------------|---------------|
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native AWQ-Marlin) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| Qwen3.6-27B Dense AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (R9700 self-cal) |
| Qwen3-30B-Instruct-2507 REAM AWQ | [mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ) | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (Samsung SAIL `merge.py`, 128→96 experts) |
| Qwen3-Coder-30B-A3B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-REAM AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (Samsung SAIL `merge.py`, 128→96 experts) |
| Qwen3-Coder-30B-A3B-REAP AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (in-house `scripts/quantize/run_reap.py`, 128→96 experts) |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, ~60B effective; doesn't fit at AWQ on 24 GB cards — listed for R9700 / future bigger-card use) |
| Qwen3-VL-32B Dense AWQ | [mattbucci/Qwen3-VL-32B-AWQ](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) (R9700 self-cal, `balanced_thinking_vision` recipe) |
| Devstral-Small-2-24B AWQ ★ canonical Devstral | [mattbucci/Devstral-Small-2-24B-AWQ](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ) | [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) (FP8→BF16→GPTQ+tool-cal→AWQ; `code_vision_tools` recipe; full 262K context on `devstral` preset) |
| Gemma 4 26B A4B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B Dense AWQ | [mattbucci/gemma-4-31B-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-AWQ) (in-house BF16→GPTQ→AWQ, vision tower FP16) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Gemma 4 12B Unified AWQ | [mattbucci/gemma-4-12B-AWQ](https://huggingface.co/mattbucci/gemma-4-12B-AWQ) (in-house data-free RTN-from-QAT, full omni) | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (via QAT base `gemma-4-12B-it-qat-q4_0-unquantized`) |
| ⚠ Qwen3-Coder-REAP-25B-A3B AWQ (3rd-party-base, legacy) | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B). Superseded by `Qwen3-Coder-30B-A3B-REAP-AWQ` (in-house) above — kept live for backward compat. |
| ⚠ Qwen3.6-VL-REAP-26B-A3B AWQ (3rd-party-base, vision broken) | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | **Upstream:** Qwen/Qwen3.6-VL-30B-A3B-Instruct (vision-preserving target). **Shipped from 3rd-party pre-pruned BF16:** [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) — vision tensors dropped at the pre-prune layer → no working vision. Rebuild planned (task #32) from upstream BF16 with vision retained. |
| ⚠ Qwen3.5-28B-A3B-REAP AWQ (3rd-party-base) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) | **Upstream:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (vision tensors retained at pre-prune, so vision works). Rebuild planned (task #34) via in-house REAP on upstream BF16. |
| Gemma 4 21B REAP AWQ | [mattbucci/gemma-4-21B-REAP-AWQ](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) (smaller Cerebras-style REAP variant of the 26B parent; in-house regex-`ignore` calibration) |
| Qwen3.5-27B Dense AWQ | [mattbucci/Qwen3.5-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (R9700 self-cal) |

### MoE coverage matrix — calibration backlog

Each MoE base should ship in three flavors: **native** (no expert compression), **REAP** (Cerebras-style pruning, in-house via `scripts/quantize/run_reap.py`), **REAM** (Samsung SAIL merging, in-house via `scripts/quantize/run_ream_qwen3moe.sh`). All entries are self-calibrated AWQ-int4 from the upstream BF16 base — no 3rd-party quants.

| Base | Native AWQ | REAP AWQ | REAM AWQ |
|---|:---:|:---:|:---:|
| Qwen3-Coder-30B-A3B (128e) | ✅ | ✅ (in-house + Cerebras variants) | ✅ |
| Qwen3.6-35B-A3B (256e, DeltaNet+VL) | ✅ | ❌ | ✅ 192e |
| Qwen3-30B-Instruct-2507 (A3B) | ❌ | ❌ | ✅ 96e |
| Qwen3.5-28B-A3B (DeltaNet+VL) | ❌ | ✅ (Cerebras-based) | ❌ |
| Qwen3.6-VL-30B-A3B (multimodal A3B) | ❌ | ⚠ atbender pre-pruned, vision broken | ❌ |
| Gemma 4 26B A4B (103e MoE+VL) | ✅ | ✅ (21B-REAP, Cerebras) | ❌ |
| Qwen3-Coder-Next-80B-A3B (512e) | — too big @ AWQ | — | ✅ ~60B effective |
| Nemotron-3-Nano-Omni-30B-A3B (128e, AVLM) | ⏳ prepped, queued (Rule 1) | ❌ | ❌ |

**Calibration backlog (prioritized):**

0. **`North-Mini-Code-1.0-AWQ`** (NEW, user-unlocked 2026-06-11) — from `CohereLabs/North-Mini-Code-1.0` BF16 (61 GB): GPTQ → CT → AWQ-Marlin. **Preserve thinking + tool** (`<|START_THINKING|>` template; `thinking` + function-calling calibration mix — `thinking_vision`-class recipe minus vision, e.g. AM-Thinking + Hermes-function-calling + python-instruct); eos 255001; ship the repo `chat_template.jinja` verbatim; group-128-clean shapes (2048 hidden / 768 expert-inter / 128 experts top-8 sigmoid). Serving side ready on 3090 (patches 042+051); fp8 path is R9700's.
1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of the bake-off top scorer (177/300 = 59% × opencode). `run_reap.py` from upstream Qwen3.6-35B-A3B BF16 (256→192e, matching the REAM expert count); AWQ recal with the `thinking_vision_video` recipe. Tooling ready (fused-`Qwen3_5MoeExperts` unfuse + `Qwen3_5MoeTopKRouter` saliency hook, miniature-validated 7/7). On the on-box run, confirm the routed-expert prune leaves the shared expert + vision tower intact (`save_pretrained` keeps all non-MoE tensors; the unfuse patch re-fuses experts to the standard 3-D format on save).
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of our multimodal MoE. Samsung SAIL `merge.py` needs porting to Gemma 4 arch (currently only Qwen3 family is wired); AWQ recal must preserve vision tower BF16.
3. **`Qwen3.6-VL-30B-A3B-AWQ`** (native) + **`-REAM-AWQ`** + **in-house `-REAP-AWQ`** — multimodal A3B base. Current REAP (`Qwen3.6-VL-REAP-26B-A3B-AWQ`) was calibrated on atbender's pre-pruned BF16 which stripped the vision tower → vision broken. Need all three flavors from the upstream BF16 with vision tensors retained.
4. **`Qwen3-30B-Instruct-2507-AWQ`** (native) + **`-REAP-AWQ`** — text generalist. REAM exists (the fastest preset, 107 tok/s); native + REAP complete the trio.
5. **`Qwen3.5-28B-A3B-AWQ`** (native, DeltaNet+VL) + **`-REAM-AWQ`** — older-gen hybrid. Only the Cerebras REAP currently ships.
6. **`Nemotron-3-Nano-Omni-30B-A3B-REAP-AWQ`** + **`-REAM-AWQ`** — gated on native ship completing (task #26). Note: EAGLE3 draft training (#27) takes priority once native lands, since spec-decode beats pruning for single-user decode wins on A3B-MoE.

Each new ship is a 12-20 h CPU GPTQ calibration + CT→AWQ conversion + multimodal validation. Sequential under Rule 1 (no concurrent calibration + serving). Two pieces of tooling work the backlog reveals: (a) Samsung SAIL REAM merge script needs porting to Gemma 4 arch, (b) `run_reap.py` needs adapting for Gemma 4 + Nemotron-H families (currently Qwen3-only).

### VRAM context limits (KV dtype varies, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B-Instruct-2507 REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B-A3B REAP AWQ | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-Marlin | 9.87 GB | ~8 KB hybrid | 262K |
| Qwen3.6-REAM-A3B AWQ | 7.4 GB | ~8 KB hybrid | 262K |
| Qwen3-Coder-30B-A3B AWQ | 8.0 GB | 36 KB | 262K |
| Qwen3-Coder-30B-A3B-REAP AWQ | 6.5 GB | 72 KB | 262K |
| Qwen3.6-27B Dense AWQ | 8.8 GB (measured; sharded) | 24 KB | 262K |
| Devstral-Small-2-24B AWQ | 7.0 GB | ~40 KB (fp8 KV) | **202K** ‡‡ (MEM 0.90) |
| Gemma 4 26B A4B MoE AWQ | 6.5 GB | ~12 KB (SWA) | **652K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 21B REAP AWQ | ~5 GB | ~12 KB (SWA) | **653K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 31B Dense AWQ | 7.7 GB | ~12 KB (SWA, fp8_e5m2) | **347K full / 17K swa** (`--swa-full-tokens-ratio 0.05` + MEM 0.92 + e5m2 KV; swa pool 2× the proven 8.4K floor) |
| Gemma 4 12B Unified AWQ | 5.4 GB | 15.3 KB full + 152.6 KB swa | **565K full / 35K swa** (262K ✓ @ ratio 0.0625) |
| Qwen3-VL-32B Dense AWQ | 10.0 GB | 24 KB | 131K (model-card cap) |

‡‡ **"Max context" is the REAL KV-pool capacity** (`max_total_num_tokens` from the serve log), not the declared `--context-length`. The heavy-**VL** ships are KV-bound (dense weights + FP16 vision tower), but two config levers recover most of it: **SWA-ratio right-sizing** (the default `--swa-full-tokens-ratio 0.8` gives the sliding sub-pool 80% of full although sliding layers attend only `window=1024` — 12B 102K→565K, 26B 118K→652K) and **e5m2 FP8 KV on the triton-forced path** (31B 24K→347K combined with ratio 0.05 + MEM 0.92). devstral 202K (MEM 0.90) is full-attention-bound — no SWA pool to right-size; 262K needs MEM≈1.0. The A3B-MoE models are genuinely 256K+ by arch: qwen36 996K / qwen36-ream 2.4M / qwen36-dense 657K / qwen3-ream 578K / Coder-A3B ~900K (fleet serve logs). Tool-use verified 1.0 to 258K true on 12B/26B/31B (probe table below).

## Quality Evals

**Fleet integrity (2026-05-31): all shipped AWQ models are scale-integrity clean** — a fleet-wide `check_awq_scales.py --base` audit found zero real zero-over-live (v2-disaster) defects; every flag resolved to benign MoE dead-channel structural sparsity (the flagship qwen36 passes the full scales+qweight audit, 0/61940). Capability-wise, the v0.5.12 ships are validated and **thinking + image + video are intact** (qwen36 / qwen36-ream / gemma4-31b 5/5, qwen35-moe 4/4, devstral 3/3 image-only, qwen3-ream 1/1 text-only). The remaining gap is the *static* eval suite below. Full per-model verdict + capability receipts: [`benchmarks/quality/fleet-integrity-audit-2026-05-31.json`](benchmarks/quality/fleet-integrity-audit-2026-05-31.json).

Run with `scripts/eval/eval_quality.py` (or `eval_and_chart.py` / the full-fleet `run_v0512_fleet_eval.sh` orchestrator): MMLU, HumanEval pass@1, [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 subbenchmarks), Needle-in-Haystack (**1K → 250K** — reaches our 256K target; timeout scales with context). The **bottom 8 rows are the v0.5.12 fleet, refreshed 2026-06-08** at MMLU 57 / HumanEval 40 / LAB-Bench 140 (20-per-subbench) — firmer than the first v0.5.12 pass but still treat ±a few points as sampling noise. The **Needle column is the deepest length where all 3 depths (0.1 / 0.5 / 0.9) retrieve**; it tracks each model's real KV pool at *measurement time*. ⚠ The multimodal rows' needle ✓-caps (devstral ✓131K, gemma4-26b ✓131K, gemma4-31b ✓16K, gemma4-12b ✓65K) **predate the 2026-06-10/11 KV unwalling** (SWA-ratio + e5m2 + MEM levers — current pools in the [VRAM context limits](#vram-context-limits-kv-dtype-varies-tp2-48-gb-total) table) — the 256K tool-use probe below is the current-depth instrument: every tool-trained ship now retrieves-and-acts at 1.0 to its pool (258K true for the Gemmas, 178K for devstral). Needle-table refresh rides the next fleet eval pass. Thinking-model MMLU/LAB read sensibly because the eval reads `reasoning_content` (the reasoning-parser routes answers there) and gives thinking models budget to close `</think>` before answering. v0.5.11 rows used full LAB-Bench (~1786) but otherwise comparable samples.

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
| Devstral-Small-2-24B AWQ | 77.2% | 80.0% | 33.6% | ✓131K ✗250K | `devstral.json` |
| Gemma 4 31B Dense AWQ | 93.0% | **97.5%** | **42.9%** | ✓16K ✗65K | `gemma4-31b.json` |
| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | ✓131K ✗250K | `gemma4.json` |
| Gemma 4 12B Unified AWQ | 77.2% | 92.5% | 29.3% | ✓65K ✗131K | `gemma4-12b.json` |

† **Gemma 4 21B REAP HumanEval 0%** is a known calibration artifact — the v3b ship serves cleanly per the audit but the REAP prune lost coding capability. Use `gemma4-31b` (the in-house dense AWQ rebuild) for code workloads.
‡ **Qwen3.5-28B-A3B-REAP LAB-Bench 15.9%** is on a partial 333-question subset (the eval timed out on the full 1786). Most rows are the full LAB-Bench (1786); the v0.5.12 rows use 20-per-subbench (140).
**HumanEval methodology:** chat endpoint + `chat_template_kwargs:{enable_thinking:false}`, one no-think method fleet-wide (raw `/completions` zeros Gemma; plain chat-HE truncates thinking models that burn the budget on CoT — no-think chat fixes both). Each solution runs in an isolated subprocess with a hard 10s timeout, so a pathological generation can't hang the harness.
◊ **qwen3-ream HumanEval 27.5%** — a non-coder text generalist on a code task (raw completion flattered it to 47%; the consistent no-think chat number is 27.5%). Use the Coder / qwen36 ships for code.

![Quality comparison — MMLU / HumanEval / LAB-Bench / Needle across INT4 AWQ ships on 2x RTX 3090](benchmarks/quality/quality_comparison.png)

**256K tool-use probe (new, 2026-06-06)** — `scripts/eval/probe_256k_tooluse.py`. Passive needle retrieval is necessary but not sufficient for *agentic* 256K; this probe plants a needle deep in filler and measures whether the model emits a **valid, correctly-argumented tool call** with the planted value, bucketed by TRUE `prompt_tokens`. It's the agentic 256K signal SWE-bench Lite (tops ~128K) never reaches:

| Preset | valid tool call | correct args | max TRUE tokens still correct |
|---|:---:|:---:|:---:|
| `qwen36` (MoE-thinking) | **1.0** | **1.0** | **258K** |
| `qwen36-ream` (MoE-thinking) | **1.0** | **1.0** | **258K** |
| `qwen36-dense` (dense-thinking) | **1.0** | **1.0** | **258K** |
| `gemma4` (26B MoE) | **1.0** | **1.0** | **258K** |
| `gemma4-12b` (unified omni) | **1.0** | **1.0** | **258K** |
| `gemma4-21b-reap` (MoE) | **1.0** | **1.0** | **258K** |
| `gemma4-31b` (dense) | **1.0** | **1.0** | **258K** |
| `devstral` (dense, tool) | **1.0** | **1.0** | **132K** firm; 178K flaky (1/2 runs; 202K pool) |
| `qwen3-ream` (text generalist) | 0.0 | 0.0 | — (non-tool-trained, expected) |

Early low rates (gemma 0.2–0.4, devstral 0.6, dense 0.8) were **KV-pool / over-cap artifacts at the pre-sprint configs**, not tool-format failures — re-probed at depth after the SWA-ratio + e5m2 + MEM unwalling, every tool-trained preset is 1.0/1.0 to ~258K true (the gemma/qwen ladders' deepest in-context point). The one genuine quality edge: **devstral flutters near its pool edge** — at 178,650 true it emitted a tool call with wrong args in 1 of 2 runs (temp 0.3), so treat 132K as its firm agentic depth. Receipts: `benchmarks/quality/tooluse256k-*-v0512.json` + `benchmarks/sprint-2026-06-kv-decode/{deep-tooluse-fill,A5-nudge-gemma4-31b-r05}/`.

**256K reasoning quality** — `scripts/eval/probe_256k_quality.py`. Retrieval/tool-calling at 256K is necessary but not sufficient for *high quality* at 256K; this probe tests real reasoning over a full context — multi-key retrieval (3 of 5 spread facts) + variable-tracking (find a 3-step dependency chain scattered through the filler AND compute it) + aggregation (sum 5 scattered values) — identical task instances at every length so any drop is real degradation. Measured at **TRUE actual-token lengths** (the top point is 255,785 real prompt tokens, verified against `usage` — not a filler-capped proxy; the probe is calibrated to ~6.9 char/tok so `approx==actual`). **`qwen36` / `qwen36-dense` / `qwen36-ream` score 100% on all three at every length to the genuine ~256K — flat, zero degradation.** **`qwen35-moe` (3.5-gen, REAP-pruned) holds 100% to 131K then softens at the extreme:** retrieval stays perfect (multikey OK everywhere) but it drops one *compute* task per long point — aggregate@200K, vartrack@256K (2/3 each) — so the clean 256K-reasoning tier is the **Qwen3.6 generation**. **The deeper discriminator is *thinking*, not context-fit:** the non-thinking ships fail multi-step reasoning even at *short* context — devstral (coder) ~40%, qwen3-ream (generalist) ~33% (both *retrieve* fine — multikey OK — but can't *compute* var-track/aggregation at 1K, let alone 256K). So high-quality 256K reasoning needs a thinking MoE **and** the length-robustness the 3.6 generation has. Re-run with `scripts/eval/run_reason256k.sh`. Receipts: `benchmarks/quality/quality256k-*-v0512.json`.

**KV-pool status after the 2026-06-10/11 unwalling** (SWA-ratio right-sizing + e5m2 FP8 KV + MEM tuning; sprint history in [`patches/README.md`](patches/README.md)): **every Gemma ship now serves 256K-class** — 12B 565K / 26B 652K pools at FP16 KV, 31B 347K pool via `fp8_e5m2` KV (the reusable sm_86 rule: **triton-forced attention takes FP8 KV as e5m2, never e4m3** — sm_86 triton can't compile `fp8e4nv`). Remaining holdouts are **devstral 202K** (full-attention-bound; 262K needs MEM≈1.0) and **qwen3-vl-32b 131K** (model-card cap). Quality nuance stands: serving 256K ≠ reasoning at 256K — gemma-26B's multi-step reasoning softens by ~64-110K regardless of KV config, so the **high-quality-256K tier remains the Qwen3.6 family** (100% reasoning flat to 255K); the Gemmas are 256K *retrieval/tool-use* ships (1.0 tool-use to 258K true, table above).

**Thinking serving defaults (2026-06-07, cross-team with R9700).** Every thinking preset (the `--reasoning-parser` ones: qwen36 / qwen36-dense / qwen36-ream / qwen35-moe / gemma4) now launches with **`--sampling-defaults model`** — it uses each checkpoint's own recommended sampling (Qwen3.6 ships temp 1.0 / top_p 0.95 / top_k 20) instead of SGLang's generic temp 1.0 / top_p 1.0 / top_k −1 (untruncated tail). The model's top_p/top_k truncation is the *principled* anti-overthinking lever: it curbs the int4 degenerate-repeat / overthinking-spiral failure mode ([arXiv:2606.00206](https://arxiv.org/abs/2606.00206)) and the temp=0 `"</think> Paris </think> Paris…"` greedy-decode loop, **with no token cap**, so deep single-user 256K reasoning is untouched. For the *agentic* case where int4 thinking spirals without committing a tool call, an **opt-in `STRICT_THINK=1`** adds `--enable-strict-thinking` so a per-request `custom_params.thinking_budget` can bound the think-loop (R9700: budget≈300 turned 0→1 applied edits) — deliberately **off by default** (a ~300-token cap would gut the 256K reasoning win). MoE-thinking smoke on live qwen36 confirms both: default → reasoning correct + thinking terminates + tool-calls valid; `STRICT_THINK=1` → `thinking_budget=32` collapses the think pass 646→87 chars while still answering, and tool-calls stay 4/4 valid with or without a budget.

**`qwen36` / `qwen36-ream` emit the right tool call retrieving from ~253K tokens** — direct, measured evidence the flagship MoE-thinking ships do real agentic work at the 256K target. **Depth-independent (no lost-in-the-middle):** re-running the qwen36 probe with the needle at depth **0.1 / 0.5 / 0.9** (start / middle / end of the filler) gives **1.0 valid + 1.0 correct at every depth, all the way to 253K** — it's not just finding mid-context needles. (Gemma's low scores here are the KV-context limit from the ⚠ note above — gemma4-31b 400s past ~24K — not tool-format degradation.) Receipts: `benchmarks/quality/tooluse256k-*-v0512.json` (+ `-depth{0.1,0.5,0.9}-` for the depth sweep).

**SWE-bench Lite** scores live in the [bake-off table at top](#coding-eval-bake-off-swe-bench-lite-v2-docker-harness-256k-single-user) — that's the end-to-end agentic eval (opencode/claw-code/little-coder scaffolds × v2 Docker harness), not part of this static-eval table.

**Every new AWQ ship MUST pass `scripts/eval/validate_capabilities.py`** (basic + thinking + image + video + audio + tool, per applicable modalities) before entering this table. Validator + receipts live in `scripts/eval/` and `benchmarks/quality/`. **TODO:** re-run the full table on v0.5.12 ships + add RULER, LongBench Pro, LiveCodeBench when scripted.

## Setup

```bash
./scripts/setup.sh
# or manually ($SGLANG_DIR defaults to components/sglang; the live tree is /data/sglang-rebase-v0512):
cd "$SGLANG_DIR" && git checkout v0.5.12
for p in "$REPO_DIR"/patches/*.patch; do git apply "$p"; done
cd python && pip install -e .
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.12 + 25 local patches |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.71.05) / cu130 wheel |
| transformers | 5.6.0 (v0.5.12 pin) |
| FlashInfer | 0.6.11.post1 |
| compressed-tensors | 0.15.0.1 (serving env); 0.15.1.dev (`quant` calibration env) |

The serving tree lives at `/data/sglang-rebase-v0512` (env `sglang-v0512`); launch with `ENV_NAME`/`SGLANG_DIR` overrides. Calibration uses the separate `quant` env.

## Patches

**26 logical patches** (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.12 — consolidated 2026-06-10 (five multi-patch series merged into single units), 051 added 2026-06-11; full set re-verified: applies clean on pristine v0.5.12, byte-identical to the live serving tree, rerun-safe. By work area:

| Area | Patches |
|------|---------|
| Quantized-weight loading (AWQ/CT int4) | 002, 029, 030, 031 |
| Qwen3.5/3.6 enablement | 018, 035 |
| Gemma 4 bring-up (26B MoE mm / 31B dense / 12B unified) | 004, 023, 025, 026, 028, 039, 043, 050 |
| MoE runner gelu coverage | 017 |
| Kernel correctness & precision | 003, 011, 012 |
| Ampere (sm_86) enablement | 005, 007 |
| Serving robustness / agentic | 034, 041, 049 |
| New-arch grafts / build | 042, 051, 037 |

Every patch is classified against upstream main (2026-06-10/11): **13 sglang-PR candidates + 1 transformers-PR (050), 4 drop-on-next-rebase, 5 site-specific**. Per-patch narratives, the merged-unit mapping (old 021/024/036/038/040/044–048 → 017/023/035/039/043), the upstream ledger, and the patch-hygiene gates for any new patch live in [`patches/README.md`](patches/README.md).

## Quantization

Self-calibrated models use the `quant` conda env:

```bash
conda activate quant
REAP_ENV=quant ./scripts/quantize/run_reap.sh --model <bf16> --save-path <reap_bf16> --keep-experts N  # MoE expert prune (Qwen3Moe today)
./scripts/quantize/run_ream_qwen3moe.sh <bf16> <ream_bf16>                                    # MoE expert merge (Samsung SAIL)
CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                          # MoE CT→native AWQ
python scripts/eval/check_awq_scales.py <awq_dst> --base <bf16_base_dir>                      # ship gate: 0 = clean
```

`check_awq_scales.py --base` runs the **dead-channel comparator**: MoE bases (Qwen3.6-35B-A3B etc.) ship 50-72% of some layer-0 expert gate/up channels at `~7.8e-38` (bf16 denormal); AWQ's fp16 group scale faithfully flushes those to 0. A zero scale over a **dead** base block is benign and downgraded; a zero scale over a **live** base block stays a `DEFECT` (the v2 dequant-to-zero → NaN signature). On qwen36 the comparator reclassifies all 144 structural-sparsity flags → 0 residual, while still catching any injected live-block zero. Without `--base` the audit stays conservative (flags every majority-zero scale).

`scripts/quantize/calibration_datasets.py` builds capability-preserving recipes (`thinking_vision` / `code_vision` / `code_vision_tools` / `balanced_thinking_vision` …) from AM-Thinking-v1, NuminaMath-CoT, LLaVA-Instruct, Hermes-function-calling, UltraChat, python-instruct. REAM/REAP expert compression in [`REAM.md`](scripts/quantize/REAM.md). See [rules-for-agents.md](rules-for-agents.md). Launch detached calibrations with `conda activate <env>` + `python -u` (not `conda run`, which buffers all output).

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner (native gfx1201 FP8); RDNA4 serving stack. Originated the FP32-softmax patch 011 and the CT→native AWQ converter; shipped the EAGLE3/DFlash spec-decode recipes we're porting. Both stacks publish under `mattbucci/*` with format suffixes.
  - **Open ask (R9700→3090, 2026-05-31): validate `mattbucci/Devstral-Small-2-24B-AWQ` on Ampere.** New ship: Devstral-Small-2-24B (Mistral3 dense+vision) is FP8-only upstream — we built AWQ by dequanting the official FP8 → code+vision calibration (vision_tower/mm_projector/lm_head BF16). On 32 GB it reaches **full 256K** (KV pool 507683 tok) where FP8+BF16-vision caps ~180K (weight-size bound). **Tool-calling works on the AWQ build** (`mistral` parser, `finish=tool_calls`, structured) — directly relevant to your bake-off Devstral queue, which was deferred on the FP8 path's tool-call tokenizer bug. Asks: (a) does it serve clean on AWQ_Marlin? (b) confirm tool-calling so it can enter the bake-off; (c) max ctx at 24 GB (likely <256K). Validate: `launch.sh devstral` repointed via `MODEL=mattbucci/Devstral-Small-2-24B-AWQ` + `validate_capabilities.py --skip-thinking --skip-video` + a tool-call smoke.
  - **Quick check (R9700→3090, 2026-06-01): confirm your pure-attention A3B MoE presets run with cuda-graph ON.** We profiled `coder-30b` decode and found M=1 MoE decode is **dispatch-bound** — only ~48% GPU util (40.6 ms wall vs 19.5 ms GPU-busy), because an A3B MoE activates ~3B params/token so the GEMMs are tiny while the per-layer norm/rope/router/allreduce launch count is fixed (`elementwise_norm 29.6% · nccl 24.4% · moe 20.1% · awq_gemv 9.5%`). Enabling cuda-graph gave **2.3–2.5× decode** (coder-30b 24.7→57.6, coder-reap-25b 22.9→58.1, gemma4 31.9→53.2 tok/s @short; coherent, vision intact). We'd had graphs globally disabled (DeltaNet/ROCm history) and lost this until re-enabling per-preset. CUDA defaults cuda-graph ON so you're likely fine — but it's a 30-second `grep -n 'disable-cuda-graph' scripts/launch.sh` to confirm none of your MoE presets (`coder-30b-eval`, `coder-reap-25b`, `qwen3-ream`, `gemma4`) inherited a disable flag. Dense/DeltaNet are GPU-bound (no gap to remove), so this is MoE-specific.
  - **Hand-off (R9700→3090, 2026-06-01): you run the Nemotron-3-Nano-Omni AWQ calibration (your task #2) — R9700's 64 GB RAM can't.** R9700 was asked to build `mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ` but hit a hard wall: the **62 GB BF16 model doesn't fit in 64 GB RAM during calibration** → OOM on full-CPU load, and swap → catastrophic thrashing (**559 s/sample, 39 h ETA**). device_map-GPU and disk-offload each hit compressed_tensors bugs (see below). Your box has the RAM headroom (and it's your designated AWQ-calibration role), so please run it there. **All the hard-won fixes are committed in the R9700 repo `scripts/quantize/` — reuse them:**
    - **group_size=64, NOT 128** — `moe_intermediate_size=1856 = 64×29` is not divisible by 128, so group-128 GPTQ aborts on strict-division. (hidden 2688 = 64×42 also clean.) Convert with `--group-size 64`. Arch-level — applies on your stack too.
    - **Calibrate `model.language_model` (the NemotronHForCausalLM backbone), NOT the full Omni wrapper** — the wrapper's `forward(pixel_values, …, image_flags=None)` does `image_flags.squeeze(-1)` and crashes on text-only (drop_images) calibration. Quantize the backbone, save the full model (compressed-tensors infers the config from the quantized modules; vision/audio stay BF16).
    - **Env deps**: `timm`, `open_clip_torch`, `ftfy`, `wcwidth`, `librosa` are required to load the Omni (CRADIO vision + Parakeet audio). Load with `attn_implementation="eager"` if flash-attn dispatch trips. transformers 5.x required (`modeling` imports `transformers.initialization`).
    - **If on llmcompressor 0.11.x** (newer than your 0.10.1.dev92): all-expert MoE calibration moved from the `GPTQModifier(moe_calibrate_all_experts=)` kwarg into `moe_calibration_context`, which (a) needs a registered MoE module for `NemotronHMoE` — see R9700 `scripts/quantize/nemotron_moe_calibration.py`, and (b) trips a `functools.partial` forward bug in compressed_tensors `set_forward_quantized` + `offload_module` — see R9700 `scripts/quantize/patch_ct_set_forward.py`. On your 0.10.1.dev92 the kwarg path likely avoids both.
    - **Heads-up**: all-expert calibration makes the inter-layer propagation run all 128 experts → very heavy; if RAM-constrained, router-only is far lighter (but under-calibrates rare experts). Audio calib data (`common_voice_17_0`, `covost2`) was gated/inaccessible from R9700 — audio encoder is BF16 so capability is preserved; validate audio post-hoc. R9700 env recipe: `scripts/quantize/setup_nemo_quant_env.sh`. R9700 keeps the FP8 build serving 256K and will do the AWQ-vs-FP8 comparison once your AWQ lands.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal plumbing.

## Repo layout

```
patches/                  # SGLang v0.5.12 patches — narratives in patches/README.md
benchmarks/               # per-model JSON; quality/ = MMLU/HumanEval/LAB-Bench/Needle + capability matrix
scripts/
  launch.sh / common.sh / setup.sh
  bench/ eval/ quantize/ test/
components/sglang/        # SGLang v0.5.12 + patches (serving tree at /data/sglang-rebase-v0512)
systemd/                  # cooling profile units
```
