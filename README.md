# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2× NVIDIA RTX 3090 (GA102-300-A1, Ampere). SGLang **v0.5.15** + 24 local patches (default since 2026-07-12; flip receipt + fleet re-validation: [`patches/v0.5.15-rebase-status.md`](patches/v0.5.15-rebase-status.md); prior stacks retained for one-revert rollback), CUDA 13.2 / PyTorch cu130. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

## Direction

**We optimize for 256K-context single-user agentic workloads on AWQ-int4 ships.** SWE-bench Lite is the canonical eval — every preset listed below serves at full 256K (or model-card max) so agentic harnesses with multi-turn tool-call context (median ~41K, p90 ~82K, max 230K per our 2026-05-31 measurement) actually fit.

This rules out three alternative optimization axes other 3090 stacks chase:

| Their focus | Why not ours |
|---|---|
| Short-ctx multi-stream throughput (vLLM-style 80-140 TPS @ <32K) | We need the full prompt of an agentic instance in context; truncating mid-conversation loses correctness. |
| FP8 quantization | **FP8 W8A8 MoE doesn't compile on sm_86** — the Triton fused-MoE kernel needs e4m3 (`fp8e4nv`); Ampere Triton only has e5m2 (same wall as the FP8-KV-e5m2 rule), so the engine SIGQUITs on the first forward. And even where FP8 *is* native (R9700 gfx1201), AWQ-int4 wins single-user M=1 decode — weight-byte-bound, FP8 is 2× int4 (Coder-30B 56 vs 38 tok/s). FP8 is the R9700 lane. Receipt: [`benchmarks/fp8-vs-awq-coder-reap.json`](benchmarks/fp8-vs-awq-coder-reap.json). |
| Spec-decode (EAGLE3 / DFlash / MTP) | At 256K + AWQ-int4 + draft + cuda graphs we OOM on 24 GB cards. R9700's 32 GB cards have the headroom we don't. Closed as "not viable on 24 GB"; see [Speculative decoding](#speculative-decoding). |

**Active work** (full task queue in `git log` + the task tracker):

1. **SWE-bench bake-off — resuming** *(current lever)* — the remaining little-coder re-run cells and the short claw cells (queue in `evals/swebench/run_all_cycles.sh`, `--skip-existing` resumes mid-queue). Was paused for the EAGLE3 draft training; both drafts are shipped (results in [Speculative decoding](#speculative-decoding)).
2. **Single-user 256K decode optimization** — the standing mission. Per-model decode receipts in [Performance](#performance--single-user-decode-at-256k); open levers in [Tooling](#tooling). Continuous iteration loop — cadence in [`CLAUDE.md`](CLAUDE.md).
3. **North-Mini-Code AWQ-int4 build** — top of the calibration backlog (see [Unlocked models](#unlocked-models-user-signal-2026-06-11)); serving side ready (native cohere2_moe + patch 051).
4. **MoE coverage matrix gaps** — every MoE base should ship in native + REAP + REAM AWQ. Audit + missing variants in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below.

What we **don't** ship: random community quants. Every `mattbucci/*-AWQ` is calibrated end-to-end from the upstream BF16 base via our own GPTQ → CT → AWQ-Marlin pipeline. When a model needs MoE expert pruning, we run REAP/REAM ourselves (`scripts/quantize/run_reap.py`, `scripts/quantize/run_ream_qwen3moe.sh`) on the upstream weights — no atbender / Cerebras / unsloth ships used as bases. Pre-quantized 3rd-party AWQ uploads are reference points only.

## Performance — single-user decode at 256K

![Per-model single-user decode tok/s — peak (short ctx) vs 256K, or the model's real KV cap where it doesn't reach 256K](benchmarks/all_models_decode.png)

![Single-user decode tok/s vs context length — all AWQ presets, unified 256K x-axis](benchmarks/all_models_context.png)

Every deep point below is **server-verified at its labeled depth** (`actual_input_tokens`; deepest honest point 255K = 262144 − output − margin; instrument notes: [`benchmarks/bench-depth-bug-2026-07-14.md`](benchmarks/bench-depth-bug-2026-07-14.md)). Single-user decode (M=1), capped at each model's real KV pool.

**True-256K decoders** (solid right bars): **qwen35-moe 144** and **qwen36-ream 144** lead @255K (210/209 short), then **qwen36 121** (209 short), **nemotron3-omni 93** (101 short — the Mamba2-hybrid's near-flat depth curve is genuine: 23 recurrent layers are O(1), only 6 attention layers scale), the 30B-A3B MoE trio **coder-30b / coder-reap-25b / qwen3-ream ≈ 69** (200 short; the three are M=1 twins — same arch class, same active experts), **qwen36-dense 47** (69 short). The Gemma hybrids trail hard at depth — **gemma4 / gemma4-21b-reap 24, gemma4-12b 17.5, gemma4-31b 13** — consistent with their whole quantized stack running fallback kernels (group-32 AWQ + TP-hostile shapes fail Marlin; root cause + fix directions in [Tooling](#tooling)). **KV-capped models** (hatched, at their pool/cap): devstral **42 @196K** (202K pool), qwen3-vl-32b **35 @127K** (model-card cap). REAM/REAP pruning does NOT speed M=1 decode (gemma4-21b-reap ≡ gemma4, coder-reap ≡ coder-30b): activated-expert count and attention dominate, not total expert weights — the pruned variants win on KV headroom and load footprint instead.

## Roadmap

What's queued, grouped by theme. Calibration work is gated on the bake-off sweep finishing + Rule 1 (no concurrent calibration + serving). The bake-off methodology + resume mechanics live in [`CLAUDE.md`](CLAUDE.md) and [`evals/swebench/`](evals/swebench/).

### Unlocked models (user signal 2026-06-11)

1. **`CohereLabs/North-Mini-Code-1.0`** — 128-expert fine-grained MoE **thinking** coder (`<|START_THINKING|>` + tool-call template), native 500K ctx, hybrid 1:3 full:sliding attention. Serving side is correctness-complete on our stack (native cohere2_moe + patch 051; reasoning/tool-call parsers landed via R9700's North-Mini work). **Our path: AWQ-int4 from upstream BF16** (calibration backlog №0) — preserve thinking + tool, eos 255001, ship the repo `chat_template.jinja` verbatim. Agentic expectation ~25% (middling, not leader). ⚠ R9700 measured a **~120K inherent recall horizon** (100% to 116K, 0% at 176K — cohere2's NoPE full-attention design, not a serving bug; their `flagship-recall-depth-2026-07-16.md`), so despite the 500K-ctx card this is a ≤120K-recall ship — calibrate the build's value accordingly. The FP8 variant is R9700's lane (sm_86 can't run FP8 MoE).
2. **`google/diffusiongemma-26B-A4B-it`** (2026-06-09): block-diffusion Gemma MoE (`DiffusionGemmaForBlockDiffusion`, multimodal, 51.6 GB BF16). **No serving stack implements it** — sglang main and vLLM have no model file; transformers main has the reference. v0.5.12's dllm subsystem (`low_confidence`/`joint_threshold`) targets full-seq diffusion LMs, not block diffusion, so this is a real port (days–weeks), not a graft. Phased: (a) transformers-native GPU probe (characterize block size, template, quality), (b) port scoping against the dllm scheduler mixins, (c) int4 build (shares the gemma-4 A4B lineage — tower/recipe knowledge transfers). Phase (a) is blocked on tx-main's meta-device dispatch (model never leaves meta under `device_map="auto"`; retry on the next tx release — BF16 already on disk at `/data/models/hf-google/`). Scoping: asymmetric encoder+decoder stacks, self-conditioning between denoise steps, `canvas_length` 256 → custom scheduler, days–weeks port. Queued behind the AWQ backlog unless prioritized.
3. **Official Gemma QAT-W4A16-CT ships** (google, 2026-06-04/05: 31B / 12B / E2B): same QAT lineage as our RTN-from-QAT rebuilds (which only *beat* our AWQ on the 12B — already our shipped build). Low-priority compare: boot-smoke `gemma-4-31B-it-qat-w4a16-ct` vs our `gemma-4-31B-AWQ` on the standard instrument; adopt only if it wins.

### MoE coverage gaps (every base needs native + REAP + REAM)

Detailed matrix + rebuild paths in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below. Six missing-variant builds, prioritized:

1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of bake-off top scorer (177/300 = 59.0%); 256→192e via `run_reap.py` on upstream BF16. **Tooling ready:** the fused-`Qwen3_5MoeExperts` unfuse + custom-router handling are built and miniature-validated (`scripts/quantize/test_qwen3_5moe_unfuse.py`, 7/7). Remaining = the on-box run: 62 GB BF16 → CPU offload on 48 GB VRAM (memory-marginal; R9700's 64 GB may be the better prune host), then AWQ recal with `thinking_vision_video` + `check_awq_scales.py --base`.
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of multimodal MoE. Blocked on tooling task below (Samsung SAIL needs Gemma 4 port).
3. **`Qwen3.6-VL-30B-A3B`** native + REAM + in-house REAP — rebuild VL trio with vision tensors retained (current REAP-26B is atbender pre-pruned, vision-broken). ⚠ pre-flight: smoke a community AWQ first — sglang ships the `Qwen3VLMoeForConditionalGeneration` EntryClass natively now, but this loader has a history of silent breakage; verify before sinking calibration time.
4. **`Qwen3-30B-Instruct-2507`** native + REAP — REAM exists (`qwen3-ream`, fastest preset at 107 tok/s); complete the trio.
5. **`Qwen3.5-28B-A3B`** native + REAM — older DeltaNet+VL gen; only Cerebras REAP currently ships.
6. **`Nemotron-3-Nano-Omni`** REAP + REAM — native AWQ now ships (6/6 caps, 256K-verified); the REAP/REAM variants still need `run_reap.py` extended to the Mamba2-hybrid layout (only the 23 MLP/MoE layers prune per `hybrid_override_pattern`; the 23 Mamba + 6 attention layers stay BF16).

### Tooling

Items 2–3 are prerequisites for the MoE backlog. Detailed plan: [`scripts/quantize/ream_gemma4_port_plan.md`](scripts/quantize/ream_gemma4_port_plan.md).

1. **Upstream-PR sweep — five packages PREPARED in `scripts/upstream-pr/`, NOT OPENED (user green-light gates all public PRs/issues).** All re-verified live on main `27ad9d1` (2026-07-16): **056** (gdn conv-state dtype crash — fp16 DeltaNet + mamba-track crashes on first decode), **011** (scope narrowed by site audit: PR = decode stage1's fp16 `tl.sum` softmax chain only — the extend `out_dtype` sweep is cosmetic since `tl.dot` already accumulates fp32; the **fp8-KV q-downcast** (`q.to(k.dtype)` quantizes live queries unscaled) rides along as a companion *issue* draft, perf-contested on fp8 tensor-core archs; PV-fp32 stays local — our A/B showed recall clean both arms), **017** (gated-gelu through Marlin/WNA16 MoE — main's own `relu2` relaxation is the precedent), **049** (env override for the hardcoded 480s `UNBALANCED_MODEL_LOADING_TIMEOUT_S`; acceptance retires the patch), **057** (prefer HF backend over MistralCommonBackend when tokenizer.json ships — main's Pixtral `_text_to_ids` marker shim proves the mechanism but misses every instruct-control token). Unpackaged candidates: **030** (upstream has `use_presharded_weights` plumbing at exactly our narrow sites but only Quark sets it), the Qwen3.5/3.6 AWQ + CausalLM family (002 / 018 / 031 / 035), Gemma 4 (004 / 026 + 025's `masked_fill` half), agentic robustness (041, R9700-originated — coordinate).
2. **Port Samsung SAIL REAM merge to Gemma 4 arch** — current `run_ream_qwen3moe.sh` + the upstream `merge.py` are Qwen3-family-only (5 hardcoded assumptions identified). Port unblocks the gemma-4-26B REAM build. Est. 40-60 h dev.
3. **Spec-verify conv1d dtype-cast candidate** — extend patch 003's cast to the conv1d `KERNEL_WIDTH` spec-verify branches (`matrix_x` fp16/bf16 Triton assert); blocks ALL speculative decoding on the DeltaNet hybrids. Receipt: sprint-2 B4 bootfail logs.
4. **Decode ideas (receipts in the sprint log):**
   - **NGRAM spec (opt-in `NGRAM=1`, `coder-30b`/`coder-30b-eval` only).** Draft-model-free (CPU trie → works at 256K where EAGLE3/DFlash OOM) and does not collapse at depth: @172K, no-spec 89 t/s → **235–237 t/s (~2.6×)** on copy-heavy spans (accept 6–7.6), ~42 t/s floor on novel spans. Opt-in because it's gated by copy fidelity; REAP/REAM pruning degrades it and DeltaNet thinkers are excluded (recurrent verify wall). Receipt: [`benchmarks/ngram-copyheavy-at-depth-2026-06-15.md`](benchmarks/ngram-copyheavy-at-depth-2026-06-15.md).
   - **EAGLE3-at-24GB via `--max-total-tokens` pool-capping** (qwen36-family pools are 3-9× over-provisioned) + `--speculative-draft-window-size` — untested.
   - Null/closed levers (one-line findings, receipts under `benchmarks/`): **decode-attention roofline** — sm_86 already runs ~72% of BW roofline at depth (≤+23% ceiling; [`attn-roofline-sm86-2026-07-15.md`](benchmarks/attn-roofline-sm86-2026-07-15.md)); **R9700 MoE-campaign ports** — fused-topk already default-on CUDA, qk-norm-rope ±0.2%, MoE config tuning null/blocked (Ampere heuristics at-optimum; [`nemotron-moe-tune-null-2026-07-14.md`](benchmarks/nemotron-moe-tune-null-2026-07-14.md), [`gemma4-tune-and-flashinfer-close-2026-07-14.md`](benchmarks/gemma4-tune-and-flashinfer-close-2026-07-14.md)), their BF16 collectives reclaim a HIP-only penalty; **per-layer-type FlashInfer for Gemma** — refuted at the model level (backend assert); **dense-TP allreduce acceleration** — null/blocked on sm_86 ([`allreduce-accel-null-2026-06-15.md`](benchmarks/allreduce-accel-null-2026-06-15.md), re-test toggle `ENABLE_CUSTOM_AR=1`). Also null: **bf16-operand PV** (R9700 087's +21%-at-depth) — measured **+1% @244K here, recall clean both arms** (their win fixed a 51%-of-roofline occupancy bind; sm_86 already runs ~72% BW, so there's nothing to collect; 011's fp32-PV precision margin costs ~1% and stays; receipts `pv-precision-ab-{bf16,fp32}-2026-07-16.json`). Still-open from the gemma4 fallback-kernel root cause (group-32 AWQ + TP-hostile shapes → dense MLP on unoptimized AWQ GEMM, experts on wna16 Triton): **Marlin-friendly gemma4 requant** (calibration-device item) — now the sole fix direction: R9700's grid-level split-K for narrow-shape GEMM was implemented, benchmarked, and refuted on their side (their #25, 2026-07-16), so the kernel path is dead on both stacks.
5. **Extend `run_reap.py` to remaining MoE layouts.** `run_reap.py` + the unfuse patches are in-repo (`run_reap.py` ported from R9700; the Coder-30B-A3B-REAP ship used the Qwen3Moe path). Coverage: (a) **`Qwen3_5MoeExperts`** (Qwen3.5/3.6 fused 3-D experts + `Qwen3_5MoeTopKRouter`) — ✅ done: `patches/qwen3_5moe_unfused_experts.py` (load-split + save-fuse hooks) + tuple-router handling in the saliency hook, miniature-validated 7/7 by `test_qwen3_5moe_unfuse.py`; (b) Gemma 4 parallel dense+MoE + different expert keys — ❌ TODO; (c) Nemotron-H Mamba2-hybrid (only the 23 MLP/MoE layers pruneable per `hybrid_override_pattern`) — ❌ TODO. The saliency tracker + `prune_model` are arch-agnostic once `.mlp.gate` + per-expert `.mlp.experts.{i}.down_proj` modules exist — the unfuse patches create them.

## Coding-eval bake-off (SWE-bench Lite, v2 Docker harness, 256K, single-user)

Scope: 10 presets (the queue in `evals/swebench/run_all_cycles.sh`), including `nemotron3-omni` — an AVLM omni ship measured because "it can't code" is an unproven assumption (it has thinking + tool-call); it runs last via `--skip-existing`.

Top tier: `qwen36-dense` (Qwen3.6-27B dense, thinking) **leads at 62.0%** on opencode, ahead of the A3B-MoE thinkers `qwen36` and `qwen36-ream` (59.0%). `qwen36` is a strong, consistent three-scaffold performer — **opencode 59.0% / little-coder 59.0% / claw 53.7%** — with the `developer`-role chat-template fix (†) applied. REAM ties native on opencode (both 59.0%) but **trails ~9 pp on little-coder** (`qwen36-ream` 150/300 = 50.0% vs `qwen36` 59.0%, full-300 both, 64 empty patches in the REAM cell) — the merge is scaffold-sensitive on the thinking ships, so "REAM ties native" holds on opencode/claw but not uniformly. The fleet is re-running every little-coder cell with that fix (cycle 3 of 9).

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-dense` (Qwen3.6-27B Dense AWQ, thinking) | **186/300 = 62.0%** | — | re-run † |
| `qwen36` (Qwen3.6-35B-A3B AWQ-Marlin, thinking) | **177/300 = 59.0%** | **161/300 = 53.7%** | **177/300 = 59.0%** |
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | **177/300 = 59.0%** | 122/270 (partial) | **150/300 = 50.0%** |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% | 107/300 = 35.7% | 74/300 = 24.7% |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% | 122/300 = 40.7% | 107/300 = 35.7% |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% | 109/300 = 36.3% | 76/300 = 25.3% |

† **little-coder/claw × thinking cells are re-running with the `developer`-role chat-template fix** (pi-ai sends its system prompt as role `developer`; the Qwen3.5/3.6 templates 400'd on it → empty thinking rollouts; fix wired into `setup.sh` via `patch_chat_templates_developer_role.py`). `qwen36-ream`/`qwen36-dense` claw cells show stale partials until cycles 4/7 re-run. Full story: `CLAUDE.md`.

Queued next: **`qwen3-ream` re-enters the bake-off for a v0.5.15 cycle** — it was dropped in June as "failed opencode+claw / non-tool-trained", but its June tool-use 0.0 was stack-era (1.0/1.0 to 148K on v0.5.15 — probe table below), so those failures are suspect infra; no June rollouts survive to contaminate a fresh run.

Failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling of 49% across opencode∪claw, rollout self-clean), methodology, and per-cell receipts: [`patches/README.md`](patches/README.md) + [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/). Every preset's `--tool-call-parser` matches its chat-template tool format (see Known Issues).

## Speculative decoding

**EAGLE3 drafts for R9700** (both delivered; SpecForge online training on 2×24 GB). **Devstral-24B** → [`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3) (`LlamaForCausalLMEagle3`). **Measured decode speedup on our 2×3090** (single-user coding, `num-steps=3`): **short 91.9 → 207.5 tok/s = 2.26×** (accept_len 3.32); **~16K 80.2 → 153.4 tok/s = 1.91×** (accept_len 2.86). Receipt: [`benchmarks/quality/devstral-eagle3-speedup.json`](benchmarks/quality/devstral-eagle3-speedup.json). `ttt`/num-steps capped at **3** by 24 GB training memory; spec is a **≤~64K win** (use no-spec at true 256K depth). ⚠ **Serving caveat:** EAGLE3 attaches to the Devstral **text decoder** (`Ministral3ForCausalLM`) — sglang's full-VLM wrapper (`LlavaForConditionalGeneration`) lacks `set_eagle3_layers_to_capture`, so serve the draft against the text-decoder, not the VLM wrapper (a sglang delegation patch would close that). **Qwen3-VL-32B** → [`mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ-EAGLE3) (2026-07-15). Measured here (steps 3 / topk 4 / draft 8): **short 60.4→112.2 tok/s = 1.86×** (accept 2.47), **~16K 52.1→83.2 = 1.60×** (accept 2.16) — trained at **max-length 6144** (the 19 GB 32B target on 24 GB cards can't fit the Devstral 16K recipe: full-vocab logits + target spill OOM both GPUs; the zero-copy reduction-shift refactor for 16K-class targets is documented in `scripts/specforge/launch_qwen3vl_eagle3_realrun.sh`). Same text-decoder attach caveat as Devstral (serve against the extracted `Qwen3ForCausalLM`, not the VL wrapper — extraction: `scripts/specforge/extract_qwen3vl_text_only.py`). Receipt: [`benchmarks/quality/qwen3vl32b-eagle3-speedup.json`](benchmarks/quality/qwen3vl32b-eagle3-speedup.json). Recipe + the 2×24 GB memory fixes: [`scripts/specforge/eagle3_training_plan.md`](scripts/specforge/eagle3_training_plan.md). (Serve on v0.5.13 with `TVM_FFI_GPU_BACKEND=cuda` + `SGLANG_ENABLE_SPEC_V2=0`.)

Below is the **serving** picture (draft stays BF16; target quant is independent). Receipt: `benchmarks/quality/specdec-v0512-2026-05-29.json`.

| Target | Algo / Draft | Baseline | With spec | Speedup |
|---|---|:---:|:---:|:---:|
| `coder-30b` AWQ-native | EAGLE3, `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge` (steps 4 / topk 4 / draft 8) | 185 tok/s | **306 tok/s** | **1.65×** |
| `qwen36` AWQ | DFlash, `z-lab/Qwen3.6-35B-A3B-DFlash` (`--dtype bfloat16` + spec-v2) | 126 tok/s | 126 tok/s | **~1.0× (moot)** |

DFlash buys nothing on `qwen36` — graph-ON no-spec already decodes 126 tok/s @256K (174 @1K), matching DFlash at its 32K cap. A second reason (beyond the 24 GB-fit limits below) no-spec is the only viable path.

**FP8: not a lever on this hardware** — see the [Direction](#direction) table for the sm_86 compile wall + the int4-wins-decode economics; same-weights control decoded **210 tok/s short / 179 @16K** on AWQ-int4 where FP8 won't run at all. Receipt: [`benchmarks/fp8-vs-awq-coder-reap.json`](benchmarks/fp8-vs-awq-coder-reap.json).

**Spec collapses at true 256K depth** (draft acceptance craters + the draft re-attends the full deep KV every micro-step) — confirmed for both EAGLE3 and DFlash, pure-attention and DeltaNet alike. So the documented `@256K` spec bars are short-depth-on-a-256K-server; **at depth, no-spec is the path** and spec is a ≤~64K optimization.

**Constraints on 24 GB cards** (R9700 has 32 GB headroom; ours doesn't):
- Drop `--mem-fraction-static 0.70` so the target leaves room for the draft + its cuda graphs (preset `MEM=0.85` OOMs the draft).
- EAGLE3: R9700's wide ladder (topk 16 / draft 32) OOMs the draft graphs here; our wider-but-fits ladder (steps 4 / topk 4 / draft 8) is the sweet spot.
- DFlash on `Qwen3_5MoeForConditionalGeneration`: must export `SGLANG_ENABLE_SPEC_V2=1`, pass `--mamba-scheduler-strategy extra_buffer`, **and force `--dtype bfloat16`** (the BF16 draft mismatches the FP16 target → `Index put dtype mismatch` at boot). Cap context at 32K to fit.
- Universal: `--speculative-draft-model-quantization unquant` (draft stays BF16) and `--speculative-attention-mode decode`.

Not applicable: gemma4 (no DFlash hook); AWQ's bundled MTP head is int4-dead, so NEXTN/MTP stays FP8-only.

**⚠ Spec-decode is not viable for our target workloads on 24 GB cards.** The numbers above are short-prompt decode. Two constraints kill it for the real workloads:

1. **SWE-bench prompts exceed the caps.** Measured against the finished qwen36-opencode-v2 cycle (300 instances): median peak prompt 41K, p90 82K, max 230K. **97.3% exceed EAGLE3's 16K**, **65.3% exceed DFlash's 32K**. Receipt: [`benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json`](benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json).
2. **256K + spec doesn't fit on 24 GB.** Per VRAM accounting (~15 GB weights TP=2 + 9 GB KV @ 256K + 5 GB cuda graphs + 0.4 GB draft) = ~21 GB/card → OOMs at MEM=0.85. R9700's 32 GB cards have headroom we don't.

The `SPEC_DECODE=1` opt-in remains wired for short-prompt uses. For our 256K agentic workloads, no-spec is the only viable path on 24 GB hardware. Full reasoning: [`evals/swebench/spec_decode_plan.md`](evals/swebench/spec_decode_plan.md).

**MTP-on-int4 rule:** in-ckpt MTP heads do NOT graft onto int4 targets — the BF16 MTP mispredicts on int4-shifted hidden states (Qwen3.5-27B graft probe: accept 0.00, 0.1 tok/s, worse than no-spec). MTP transfer tolerates FP8 but not int4. For int4 spec-decode use a trained EAGLE3/DFlash draft, never a grafted MTP. Vision towers, on the other hand, graft cleanly — they're input-side and quant-decoupled.

## Known Issues (open)

- **`qwen36-ream` × claw-code is a partial cell (122/270 = 45.2% of scored)** — discount per the full-300 rule. The 2026-07-16 reroll recovered 102 predictions (168→270); the last ~30 instances hard-fail in the GLIBC-sensitive claw scaffold (`rc=3` rollout landmine, not a model issue) and resist retry; opencode + little-coder are full-300.
- **Host reboots every ~9–17 h under sustained docker rollout I/O (kernel BUG).** Predictions on disk survive; auto-resume is via `swebench-bakeoff.service` (the boot-ordering cycle that was silently dropping it at every boot is fixed — cooling oneshot now orders after `nvidia-persistenced`, not `multi-user.target`). Full forensic recipe in [`CLAUDE.md`](CLAUDE.md) → Operational Lessons.

One caveat carried forward: `check_awq_scales.py` reads native-AWQ format — CT-format checkpoints crash its tensor reader (use a native-AWQ mirror or HF Range-fetch mode for CT audits). Resolved items live in `git log` + [`patches/README.md`](patches/README.md).

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.15, apply patches, create conda env

# TP=2 / 256K presets (matrix standard):
./scripts/launch.sh qwen3-ream              # 262K @ 69 tok/s — REAM merged MoE, 96 experts
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
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE A3B (256 exp, VL) | **262K** | 209 (**121 @255K**) | `qwen36` | [`mattbucci/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ). Bake-off top tier (177/300 = 59.0% × opencode). |
| **Qwen3.6-REAM-A3B AWQ** | DeltaNet+MoE A3B (192 exp, VL) | **262K** | 209 (**144 @255K**) | `qwen36-ream` | [`mattbucci/Qwen3.6-REAM-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ). Vision tower grafted. Bake-off 176/300 = 58.7% × opencode. |
| **Qwen3-30B-Instruct-2507 REAM AWQ** | MoE A3B (96 exp) | **262K** | 200 (**69 @255K**) | `qwen3-ream` | [`mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ). REAM 128→96; text-only generalist. |
| **Qwen3.5-28B MoE REAP** | DeltaNet+MoE A3B (205 exp, VL) | **262K** | 210 (**144 @255K**) | `qwen35-moe` | Cerebras REAP of Qwen3.5-28B-A3B; thinking+vision. |
| **Nemotron-3-Nano-Omni-30B-A3B AWQ** | Mamba2-hybrid MoE A3B (128 exp, AVLM) | **262K** ✓ (5.25M pool) | 101 (**93 @255K**) | `nemotron3-omni` | [`mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ). int4; **6/6 caps** (basic+thinking+tool+vision+**video**+**audio** — the only audio ship). Serves via **`QUANT=moe_wna16`** (baked into the preset; the auto-detected awq_marlin path fails on this model's TP=2 shard shapes — mechanism in `patches/v0.5.14-rebase-status.md`), enabled by **patches 052** (non-gated squared-ReLU moe_wna16) + **053** (EVS video routing). Mamba2 O(1) recurrent → decode ~flat (101→93 @255K, fixed-instrument re-measure 2026-07-14); beats R9700 FP8 (74.79/49.22). ⚠ **Agentic depth ~76K**: ≥113K actual it thinking-spirals past its token budget and never commits a tool call (tool-use table below) — flat decode does not buy agentic 256K here. |
| **Qwen3-Coder-30B-A3B AWQ** | MoE A3B (128 exp) | **262K** | 200 (**69 @255K**) | `coder-30b` or `coder-30b-eval` | [`mattbucci/Qwen3-Coder-30B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ). Two presets serve the same model; for short-ctx batch-decode benchmarks override `CTX=16384 MAX_RUNNING=32 ./scripts/launch.sh coder-30b`. |
| Coder-REAP-30B AWQ-Marlin | MoE A3B (96 exp) | **262K** | 200 (**69 @255K**) | `coder-reap-25b` | [`mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) (R9700 in-house). |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **262K** ✓ (347K pool) | 54 (30 @64K, **13 @255K**) | `gemma4-31b` | [`mattbucci/gemma-4-31B-AWQ`](https://huggingface.co/mattbucci/gemma-4-31B-AWQ). LM INT4, vision tower FP16, **KV fp8_e5m2**. Tool-use 1.0 → 258K true tokens, 5/5 caps incl. video. 347K pool via `--swa-full-tokens-ratio 0.05` + `MEM 0.92` + e5m2 FP8 KV (the only FP8 that compiles on the triton-forced path — sm_86 rejects e4m3); 262,144 declared fits with 32% headroom. |
| **Gemma 4 26B MoE AWQ** | MoE A4B (103 exp, VL) | **262K** ✓ | 81 (**24 @255K**) | `gemma4` | [`mattbucci/gemma-4-26B-AWQ`](https://huggingface.co/mattbucci/gemma-4-26B-AWQ). **Tool-use 1.0 → 258K true tokens, 5/5 caps** incl. video. 652K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Gemma 4 12B Unified AWQ** | Omni, encoder-free | **262K** ✓ | 100 (**17.5 @255K**) | `gemma4-12b` | [`mattbucci/gemma-4-12B-AWQ`](https://huggingface.co/mattbucci/gemma-4-12B-AWQ). In-house int4 RTN-from-QAT. MMLU 77 / HE 93 / **tool-use 1.0 → 258K true tokens**, **5/5 omni** (vision + video ✓; gemma4_unified is native since transformers 5.12.1). 565K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Qwen3.6-27B Dense AWQ** | Dense + DeltaNet (VL) | **262K** (657K KV) | 69 (**47 @255K**) | `qwen36-dense` | [`mattbucci/Qwen3.6-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (R9700 self-cal). |
| **Devstral-Small-2-24B AWQ** | Dense (VL) | **202K** ‡‡ | 88 (52 @128K, **42 @196K** pool cap) | `devstral` | [`mattbucci/Devstral-Small-2-24B-AWQ`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-Instruct-2512). The canonical Devstral; built from [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512). FP8→BF16→GPTQ+tool-cal→AWQ. KV 172K→202K (`MEM 0.90`); needs MEM≈1.0 for true 256K on 24 GB. |
| **Qwen3-VL-32B Instruct AWQ** | Dense (VL) | **131K** (model-card cap) | 63 (**35 @127K**) | `qwen3-vl-32b` | [`mattbucci/Qwen3-VL-32B-AWQ`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) (R9700). 63→45→35 tok/s @ 1K/64K/127K. |
| Gemma 4 21B REAP AWQ | MoE (VL) | **262K** ✓ (653K pool) | 81 (**24 @255K**) | `gemma4-21b-reap` | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Cerebras-style expert prune of the 26B parent; same Gemma 4 serving flags (graphs ON + `--swa-full-tokens-ratio 0.0625` → 652K pool); tool-use 1.0/1.0 on the standard ladder. ⚠ HumanEval 0% (REAP prune lost coding — Quality Evals below): vision/chat ship, not code. |

Per-model quality + capability receipts for the current stack: `benchmarks/quality/*-v0515.json` + `cap-*-v0515.json` (flip table in [`patches/v0.5.15-rebase-status.md`](patches/v0.5.15-rebase-status.md)).

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
| Nemotron-3-Nano-Omni-30B-A3B (128e, AVLM) | ✅ serves (moe_wna16 + patches 052/053, 6/6 caps) | ❌ | ❌ |

**Calibration backlog (prioritized):**

0. **`North-Mini-Code-1.0-AWQ`** (NEW, user-unlocked 2026-06-11) — from `CohereLabs/North-Mini-Code-1.0` BF16 (61 GB): GPTQ → CT → AWQ-Marlin. **Preserve thinking + tool** (`<|START_THINKING|>` template; `thinking` + function-calling calibration mix — `thinking_vision`-class recipe minus vision, e.g. AM-Thinking + Hermes-function-calling + python-instruct); eos 255001; ship the repo `chat_template.jinja` verbatim; group-128-clean shapes (2048 hidden / 768 expert-inter / 128 experts top-8 sigmoid). Serving side ready on 3090 (native cohere2_moe + patch 051); fp8 path is R9700's.
1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of the bake-off top scorer (177/300 = 59% × opencode). `run_reap.py` from upstream Qwen3.6-35B-A3B BF16 (256→192e, matching the REAM expert count); AWQ recal with the `thinking_vision_video` recipe. Tooling ready (fused-`Qwen3_5MoeExperts` unfuse + `Qwen3_5MoeTopKRouter` saliency hook, miniature-validated 7/7). On the on-box run, confirm the routed-expert prune leaves the shared expert + vision tower intact (`save_pretrained` keeps all non-MoE tensors; the unfuse patch re-fuses experts to the standard 3-D format on save).
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of our multimodal MoE. Samsung SAIL `merge.py` needs porting to Gemma 4 arch (currently only Qwen3 family is wired); AWQ recal must preserve vision tower BF16.
3. **`Qwen3.6-VL-30B-A3B-AWQ`** (native) + **`-REAM-AWQ`** + **in-house `-REAP-AWQ`** — multimodal A3B base. Current REAP (`Qwen3.6-VL-REAP-26B-A3B-AWQ`) was calibrated on atbender's pre-pruned BF16 which stripped the vision tower → vision broken. Need all three flavors from the upstream BF16 with vision tensors retained.
4. **`Qwen3-30B-Instruct-2507-AWQ`** (native) + **`-REAP-AWQ`** — text generalist. REAM exists (69 tok/s @255K); native + REAP complete the trio.
5. **`Qwen3.5-28B-A3B-AWQ`** (native, DeltaNet+VL) + **`-REAM-AWQ`** — older-gen hybrid. Only the Cerebras REAP currently ships.
6. **`Nemotron-3-Nano-Omni-30B-A3B-REAP-AWQ`** + **`-REAM-AWQ`** — native AWQ ships (serves 6/6, 256K-verified); REAP/REAM are unblocked but need `run_reap.py` + the REAM merge extended to the Nemotron-H Mamba2-hybrid layout (only the 23 MLP/MoE layers are pruneable per `hybrid_override_pattern`).

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

‡‡ **"Max context" is the REAL KV-pool capacity** (`max_total_num_tokens` from the serve log), not the declared `--context-length`. The heavy-**VL** ships are KV-bound (dense weights + FP16 vision tower), but two config levers recover most of it: **SWA-ratio right-sizing** (the default `--swa-full-tokens-ratio 0.8` gives the sliding sub-pool 80% of full although sliding layers attend only `window=1024` — 12B 102K→565K, 26B 118K→652K) and **e5m2 FP8 KV on the triton-forced path** (31B 24K→347K combined with ratio 0.05 + MEM 0.92). devstral 202K (MEM 0.90) is full-attention-bound — no SWA pool to right-size; 262K needs MEM≈1.0. **Why the *smaller* devstral-24B caps lower than the dense gemma-4-31B (347K):** "dense" describes the FFN (no MoE routing), not the attention — and KV, the 256K bottleneck, is set by the attention layout. gemma-31B is hybrid-SWA (50/60 layers sliding at `window=1024`, only **10 full-attention**) with **MQA-4** global KV heads (`num_global_key_value_heads=4`) and `attention_k_eq_v=True` (caches K=V *once*) → **~12 KB/token**; devstral runs **full attention on all 40 layers** with GQA-8 and separate K/V → **~40 KB/token** (~3–4× heavier). At 262K that's ≈3 GB vs ≈10.5 GB of KV/GPU, so devstral exhausts the 48 GB ~3× sooner — its 0.7 GB-lighter weights don't compensate. The A3B-MoE models are genuinely 256K+ by arch: qwen36 996K / qwen36-ream 2.4M / qwen36-dense 657K / qwen3-ream 578K / Coder-A3B ~900K (fleet serve logs). Tool-use verified 1.0 to 258K true on 12B/26B/31B (probe table below).

## Quality Evals

**Fleet integrity: every shipped AWQ model is scale-integrity clean** (fleet-wide `check_awq_scales.py --base` audit — zero real zero-over-live defects; all flags were benign MoE dead-channel sparsity; receipt [`benchmarks/quality/fleet-integrity-audit-2026-05-31.json`](benchmarks/quality/fleet-integrity-audit-2026-05-31.json)), and every capability (thinking/image/video/audio/tool as applicable) passes on the current stack — per-preset receipts `cap-*-v0515.json`.

Run with `scripts/eval/eval_quality.py` (or `eval_and_chart.py` / the fleet orchestrators): MMLU, HumanEval pass@1, [LAB-Bench](https://github.com/Future-House/LAB-Bench), Needle-in-Haystack (**1K → 250K**). Treat ±a few points as sampling noise. The **Needle column is the deepest length where all 3 depths (0.1 / 0.5 / 0.9) retrieve** and reflects each model's KV pool *at measurement time* (server-verified: 11 of 12 presets retrieve 3/3 at a true ~250,035 actual tokens; nemotron3-omni is 2/3 — the miss is depth-0.1 only, consistent with Mamba recurrent-state fade of oldest context; devstral is 3/3 at its 131K pool cap. Receipts: `*-v0515-needle2.json`) — rows measured before the KV-pool unwalling under-report vs current pools (see the [VRAM context limits](#vram-context-limits-kv-dtype-varies-tp2-48-gb-total) table); the tool-use probe below is the current-depth agentic instrument. Current-stack fleet receipts are `*-v0515.json` (flip table in `patches/v0.5.15-rebase-status.md`). Thinking-model MMLU/LAB read correctly because the eval reads `reasoning_content` and gives budget to close `</think>`.

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
| Devstral-Small-2-24B AWQ | 77.2% | 80.0% | 33.6% | ✓131K (pool cap) | `devstral.json` |
| Gemma 4 31B Dense AWQ | 93.0% | **97.5%** | **42.9%** | **✓250K** | `gemma4-31b.json` |
| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | **✓250K** | `gemma4.json` |
| Gemma 4 12B Unified AWQ | 77.2% | 92.5% | 29.3% | **✓250K** | `gemma4-12b.json` |

† **Gemma 4 21B REAP HumanEval 0%** is a known calibration artifact — the v3b ship serves cleanly per the audit but the REAP prune lost coding capability. Use `gemma4-31b` (the in-house dense AWQ rebuild) for code workloads.
‡ **Qwen3.5-28B-A3B-REAP LAB-Bench 15.9%** is on a partial 333-question subset (the eval timed out on the full 1786). Most rows are the full LAB-Bench (1786); the v0.5.12 rows use 20-per-subbench (140).
**HumanEval methodology:** chat endpoint + `chat_template_kwargs:{enable_thinking:false}`, one no-think method fleet-wide (raw `/completions` zeros Gemma; plain chat-HE truncates thinking models that burn the budget on CoT — no-think chat fixes both). Each solution runs in an isolated subprocess with a hard 10s timeout, so a pathological generation can't hang the harness.
◊ **qwen3-ream HumanEval 27.5%** — a non-coder text generalist on a code task (raw completion flattered it to 47%; the consistent no-think chat number is 27.5%). Use the Coder / qwen36 ships for code.

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
| `qwen35-moe` (DeltaNet MoE) | **1.0** | **1.0** | ≥**148K** (probe ceiling — deep rung queued) |
| `qwen3-ream` (text generalist) | **1.0** | **1.0** | ≥**148K** (probe ceiling; was 0.0 on v0.5.12 — stack-era failure, retired) |
| `coder-30b-eval` / `coder-reap-25b` | 0.8 | 0.8 | 113K / **148K** (one sporadic miss each, non-monotonic) |
| `devstral` (dense, tool) | **1.0** | **1.0** | **132K** firm; 178K flaky (1/2 runs; 202K pool) |
| `nemotron3-omni` (Mamba2 AVLM) | 0.6 | 0.6 | **76K** — spirals past its token budget ≥113K, never calls |

The **258K** rows are the June deep-fill receipts (v0.5.12 stack: `tooluse256k-*-v0512.json` + `benchmarks/sprint-2026-06-kv-decode/deep-tooluse-fill/`); every other row is the **v0.5.15 fleet re-verification** (13 presets, `tooluse256k-*-v0515.json`), whose deepest rung landed at ~148K actual — the probe's 3.8 chars/token guess under-filled labels 0.58× (fixed 2026-07-16: the probe now self-calibrates fill from `usage.prompt_tokens` each rung and caps rungs at the server window; a full-depth v0.5.15 pass is queued post-bake-off). v0.5.15 findings vs June: **`qwen3-ream` tool-calling works now** (1.0/1.0 to 148K — the June 0.0 was stack-era, not "non-tool-trained"); **`nemotron3-omni` has a firm agentic depth of ~76K** — at ≥113K actual it spends its full 2,048-token budget without committing a tool call (`finish_reason: length`, the deep-context thinking-spiral mode; its depth-0.1 needle fade says old context degrades on the same rungs — larger-budget rung queued to separate the two). Sporadic single-rung flutters, non-monotonic: `coder-30b-eval` at its deepest 148K, `coder-reap-25b` at 113K (deepest correct), `qwen36-dense` at 9.8K (all deeper correct). **devstral** 132K-firm stands (v0.5.15 corroborates: 113K correct, 148K miss).

**256K reasoning quality** — `scripts/eval/probe_256k_quality.py`. Retrieval/tool-calling at 256K is necessary but not sufficient for *high quality* at 256K; this probe tests real reasoning over a full context — multi-key retrieval (3 of 5 spread facts) + variable-tracking (find a 3-step dependency chain scattered through the filler AND compute it) + aggregation (sum 5 scattered values) — identical task instances at every length (temp=0, seed reset per length) so any drop is real degradation, measured at **TRUE actual-token lengths** (top point 255,800 real prompt tokens, `usage`-verified; calibrated to ~6.9 char/tok so `approx==actual`). **The probe elicits chain-of-thought + parses an explicit `ANSWER:` line** — a terse ask-only-the-number form silently measures no-CoT mental arithmetic and under-reads models that reason in `content` (gemma4 has no thinking channel), so CoT elicitation is load-bearing for fair measurement.

| Preset | 1K | 32K | 65K | 131K | 200K | 256K | overall |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemma4` (26B MoE) | 100 | 67§ | 100 | 100 | 100 | **100** | 94 |
| `gemma4-31b` (dense) | 100 | 100 | 100 | 100 | 100 | **100** | **100** |
| `gemma4-21b-reap` (MoE) | 100 | 100 | 100 | 100 | 100 | 67§ | 94 |
| `gemma4-12b` (unified omni, int4 RTN) | 100 | 67◊ | 67◊ | 33◊ | 33◊ | 67◊ | 61 |
| `qwen36` / `qwen36-dense` / `qwen36-ream` ‖ | 100 | 100 | 100 | 100 | 100 | **100** | **100** |

**The 26B / 31B / 21B-REAP Gemmas reason AND retrieve to true 256K (94–100%), joining the Qwen3.6 family (100% flat to 255K) in the high-quality-256K tier.** §Isolated single-task cell misses are non-monotonic (each passes at every *other* depth) → single-instance noise, not a depth cliff. ◊**`gemma4-12b`** chain-reasons to 256K (`vartrack` 100% at every depth) but its multi-needle *retrieval* saturates at ~3 needles at depth, so the 5-needle `aggregate` fails ≥32K (61% overall) — a genuine small-model deep-retrieval limit, confirmed in the raw dumps ("the text only lists three items"). ‖The qwen3.6 family scores **1.00 flat at every length** on the current probe (true 255,798 verified; `quality256k-{qwen36,qwen36-dense,qwen36-ream}-v0515.json`). **`qwen35-moe` is genuinely soft at depth: 0.72 overall** — `multikey` fails ≥131K while short lengths are clean, the same deep multi-needle-saturation class as gemma4-12b's ◊, so it sits below the high-quality-256K tier (`quality256k-qwen35-moe-v0515.json`). At 178K: `qwen3-ream` 100%, `devstral` ~80%. Re-run: `scripts/eval/run_reason256k.sh`.

**KV-pool rule (sm_86): triton-forced attention takes FP8 KV as e5m2, never e4m3** (sm_86 triton can't compile `fp8e4nv`). Per-preset pools live in the [VRAM context limits](#vram-context-limits-kv-dtype-varies-tp2-48-gb-total) table; the remaining sub-256K holdouts are **devstral 202K** (full-attention-bound; 262K needs MEM≈1.0) and **qwen3-vl-32b 131K** (model-card cap). Serving 256K ≠ reasoning at 256K — the reasoning table above is the quality instrument (sprint history: [`patches/README.md`](patches/README.md)).

**Thinking serving defaults.** Every thinking preset (the `--reasoning-parser` ones: qwen36 / qwen36-dense / qwen36-ream / qwen35-moe / gemma4) launches with **`--sampling-defaults model`** — it uses each checkpoint's own recommended sampling (Qwen3.6 ships temp 1.0 / top_p 0.95 / top_k 20) instead of SGLang's generic temp 1.0 / top_p 1.0 / top_k −1 (untruncated tail). The model's top_p/top_k truncation is the *principled* anti-overthinking lever: it curbs the int4 degenerate-repeat / overthinking-spiral failure mode ([arXiv:2606.00206](https://arxiv.org/abs/2606.00206)) and the temp=0 `"</think> Paris </think> Paris…"` greedy-decode loop, **with no token cap**, so deep single-user 256K reasoning is untouched. For the *agentic* case where int4 thinking spirals without committing a tool call, an **opt-in `STRICT_THINK=1`** adds `--enable-strict-thinking` so a per-request `custom_params.thinking_budget` can bound the think-loop (R9700: budget≈300 turned 0→1 applied edits) — deliberately **off by default** (a ~300-token cap would gut the 256K reasoning win). Validated live: a tight budget collapses the think pass while answers + tool-calls stay correct, with or without a budget.


**Depth-independence (no lost-in-the-middle):** the qwen36 tool-use probe with the needle at depth **0.1 / 0.5 / 0.9** gives **1.0 valid + 1.0 correct at every depth to 253K** — the flagship ships aren't just finding mid-context needles. Receipts: `benchmarks/quality/tooluse256k-*-v0512.json` (+ `-depth{0.1,0.5,0.9}-`).

**SWE-bench Lite** scores live in the [bake-off table at top](#coding-eval-bake-off-swe-bench-lite-v2-docker-harness-256k-single-user) — that's the end-to-end agentic eval (opencode/claw-code/little-coder scaffolds × v2 Docker harness), not part of this static-eval table.

**Every new AWQ ship MUST pass `scripts/eval/validate_capabilities.py`** (basic + thinking + image + video + audio + tool, per applicable modalities) before entering this table. Validator + receipts live in `scripts/eval/` and `benchmarks/quality/`. **TODO:** refresh the static table on the current stack + add RULER, LongBench Pro, LiveCodeBench when scripted.

## Setup

```bash
./scripts/setup.sh
# or manually (the live tree is /data/sglang-rebase-v0515):
cd "$SGLANG_DIR" && git checkout v0.5.15
for p in "$REPO_DIR"/patches/*.patch; do git apply "$p"; done
cd python && pip install -e .
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.15 + 24 local patches |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.71.05) / cu130 wheel |
| transformers | 5.12.1 (v0.5.15 pin; ships gemma4_unified natively — retired patch 055; routes Mistral ckpts to MistralCommonBackend — countered by patch 057) |
| FlashInfer | 0.6.12 [cu13] |
| compressed-tensors | 0.15.0.1 (serving env); 0.15.1.dev (`quant` calibration env) |

The serving tree lives at `/data/sglang-rebase-v0515` (env `sglang-v0515`); launch with `ENV_NAME`/`SGLANG_DIR` overrides (v0.5.14 / `/data/sglang-rebase-v0514` / env `sglang-v0514` kept as rollback). Calibration uses the separate `quant` env.

## Patches

**24 logical patches** (`ls patches/*.patch | wc -l`) targeting SGLang **v0.5.15** — cover AWQ/CT int4 weight loading, Qwen3.5/3.6 enablement, Gemma 4 bring-up (26B MoE / 31B dense / 12B unified omni), Nemotron-3-Nano-Omni serving (052/053), MoE gelu coverage, kernel correctness & precision, sm_86 enablement, and serving/agentic robustness. The v0.5.14→v0.5.15 flip (2026-07-12) is the first with a net patch-count reduction — 21 clean, 2 regenerated (011 extend-kernel paging drift / 051 hybrid-SWA set drift), **2 retired as upstreamed** (054 ministral3 keyword-init now native; 055's vendored gemma4_unified stack now ships in transformers 5.12.1), +1 new (**057** MistralCommonBackend opt-out — tx 5.12 silently re-routes Mistral checkpoints to a tokenizer that treats `[INST]`/`[TOOL_CALLS]` as plain text on sglang's render-then-encode path; caught by the fleet probe as devstral needle 0.0 + HE 48% + dead tool-calls, restored to parity by the patch). The 3-gate pristine replay is green and now scripted (`scripts/test_patch_gates.sh`). Per-patch narratives, the upstream-PR ledger, and the patch-hygiene gates live in [`patches/README.md`](patches/README.md); the flip receipt + full-probe results are [`patches/v0.5.15-rebase-status.md`](patches/v0.5.15-rebase-status.md).

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

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner (native gfx1201 FP8) + RDNA4 serving stack. We own evals + AWQ/INT4 + EAGLE3 draft training.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal plumbing.

## Repo layout

```
patches/                  # SGLang v0.5.15 patches (24) — narratives in patches/README.md
benchmarks/               # per-model JSON; quality/ = MMLU/HumanEval/LAB-Bench/Needle + capability matrix
scripts/
  launch.sh / common.sh / setup.sh
  bench/ eval/ quantize/ test/
components/sglang/        # legacy vendored SGLang tree (historical; live serving trees are under /data/)
systemd/                  # cooling profile units
```
