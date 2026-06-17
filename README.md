# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2× NVIDIA RTX 3090 (GA102-300-A1, Ampere). SGLang **v0.5.13.post1** + 24 local patches (flipped from v0.5.12 on 2026-06-16 — fleet re-validated, old stack kept for one-revert rollback), CUDA 13.2 / PyTorch cu130. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

## Direction

**We optimize for 256K-context single-user agentic workloads on AWQ-int4 ships.** SWE-bench Lite is the canonical eval — every preset listed below serves at full 256K (or model-card max) so agentic harnesses with multi-turn tool-call context (median ~41K, p90 ~82K, max 230K per our 2026-05-31 measurement) actually fit.

This rules out three alternative optimization axes other 3090 stacks chase:

| Their focus | Why not ours |
|---|---|
| Short-ctx multi-stream throughput (vLLM-style 80-140 TPS @ <32K) | We need the full prompt of an agentic instance in context; truncating mid-conversation loses correctness. |
| FP8 quantization | R9700's gfx1201 has native FP8 WMMA acceleration; Ampere sm_86 doesn't. FP8 is half the weight bytes of INT4 → less of our 48 GB total VRAM is left for KV at 256K. |
| Spec-decode (EAGLE3 / DFlash / MTP) | At 256K + AWQ-int4 + draft + cuda graphs we OOM on 24 GB cards. R9700's 32 GB cards have the headroom we don't. Closed as "not viable on 24 GB"; see [Speculative decoding](#speculative-decoding). |

**Active work** (full task queue in `git log` + the task tracker):

1. **Corrected-probe re-measurement of the remaining fleet** *(current lever)* — the 256K reasoning probe was fixed 2026-06-17 to elicit chain-of-thought + parse an `ANSWER:` line (the terse "reply with just the number" form measured no-CoT mental math, biasing against instruction-followers — see [Quality Evals](#quality-evals)). Gemmas re-measured (26B/31B/21B-REAP reason to true 256K at 94–100%; 12B chain-reasons but its multi-needle retrieval saturates ~3 at depth; the 12B v0.5.13 boot regression is fixed via patch 055). Still carrying **terse-probe** numbers and needing the corrected re-run: `qwen35-moe` (softened past 131K — thinking ship, likely real) and the **non-thinking** `devstral` / `qwen3-ream` (~40/33% terse — the CoT-elicit fix may overturn their "can't reason" reading exactly as it did for gemma4). `run_reason256k.sh PRESETS="qwen3-ream devstral qwen35-moe"`. Pure eval, no calibration.
2. **SWE-bench bake-off — RESUMED 2026-06-17 (running).** `swebench-bakeoff.service` active (its gating Gemma 256K instrument is complete — the Gemma ships are reasoning-verified at depth). The 9-preset queue is filling the gap cells (`qwen36-dense` claw/little-coder, `devstral` claw/little-coder, `gemma4` all three, the interrupted `qwen35-moe` cycle, …); idempotent `--skip-existing` no-ops the finished cells (~45s each). ⚠ Multi-day + kernel-BUG-reboot-prone (auto-resumes via the unit). `qwen36-ream`'s partial claw cell (168/300) is a known scaffold landmine (GLIBC `reroll rc=3`) that a full-cycle replay no-ops — tracked in Known Issues. Done only when `run_all_cycles.sh` exits.
3. **Single-user 256K decode optimization** — the standing mission. Per-model decode receipts in [Performance](#performance--single-user-decode-at-256k); remaining open levers (per-layer-type attention backend for Gemma — needs a live FlashInfer-sliding test to confirm/refute; tuned int4 fused-MoE Triton config for nemotron) in [Tooling → Decode ideas](#tooling). Worked as a continuous iteration loop — cadence + the experiment ledger live in [`CLAUDE.md`](CLAUDE.md).
4. **North-Mini-Code AWQ-int4 build** — calibration-device ask, top of the calibration backlog (see [Unlocked models](#unlocked-models-user-signal-2026-06-11)); serving side is ready (patches 042+051).
5. **MoE coverage matrix gaps** — every MoE base should ship in native + REAP + REAM AWQ flavors. Audit + 6 missing-variant ships in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below.

What we **don't** ship: random community quants. Every `mattbucci/*-AWQ` is calibrated end-to-end from the upstream BF16 base via our own GPTQ → CT → AWQ-Marlin pipeline. When a model needs MoE expert pruning, we run REAP/REAM ourselves (`scripts/quantize/run_reap.py`, `scripts/quantize/run_ream_qwen3moe.sh`) on the upstream weights — no atbender / Cerebras / unsloth ships used as bases. Pre-quantized 3rd-party AWQ uploads are reference points only.

## Performance — single-user decode at 256K

![Per-model single-user decode tok/s — peak (short ctx) vs 256K, or the model's real KV cap where it doesn't reach 256K](benchmarks/all_models_decode.png)

![Single-user decode tok/s vs context length — all AWQ presets, unified 256K x-axis](benchmarks/all_models_context.png)

Single-user decode (M=1), honestly capped at each model's real KV pool (`max_total_num_tokens`). **True-256K decoders** (solid right bars): qwen36-ream **139**, qwen36 **128**, qwen3-30b-ream **103**, **nemotron3-omni 98**, qwen3.6-27b **54**, **gemma4-26b 41**, **gemma4-21b-reap 41**, **gemma4-12b 34**, **gemma4-31b 22** — the Gemma hybrids joined the club 2026-06-10/11 (SWA-ratio right-sizing + CUDA graphs; the 31B additionally needed e5m2 FP8 KV; short-ctx 83 / 109 / 57 tok/s). **nemotron3-omni** (Mamba2-hybrid AVLM, int4) decodes nearly flat — 102.7 short → **97.8 @255K** (5.25M-tok pool; 23 Mamba layers are O(1) recurrent state, only 6 attention layers scale with KV), beating R9700's FP8 (74.79/49.22) on both ends; serves via patches 052/053. The DeltaNet+MoE thinkers and the Gemmas all run CUDA graphs, so decode is attention-bound, not launch-bound. **KV-capped model** (hatched, at its deepest reliable length): devstral **56 @128K** (202K pool @ MEM 0.90, full-attention-bound — 262K needs MEM≈1.0). Charts drop over-KV-cap points (a prompt that can't fit the pool never decodes there, so its logged tok/s is an artifact).

## Roadmap

What's queued, grouped by theme. Calibration work is gated on the bake-off sweep finishing + Rule 1 (no concurrent calibration + serving). The bake-off methodology + resume mechanics live in [`CLAUDE.md`](CLAUDE.md) and [`evals/swebench/`](evals/swebench/).

### Unlocked models (user signal 2026-06-11)

1. **`CohereLabs/North-Mini-Code-1.0`** (ex `BLS-Mini-Code-1.0` preview — now public apache-2.0, with an **official FP8 variant**): 128-expert fine-grained MoE **thinking** coder (`<|START_THINKING|>` + tool-call template), native **500K** ctx, hybrid 1:3 full:sliding attention (window 4096), GQA-4 / head_dim 128 → ~26 KB/token full-pool KV — 256K fits trivially at int4. Bringup state (patches 042+051, narratives in [`patches/README.md`](patches/README.md)): config parses (vendored, tx 5.6 predates `cohere2_moe`), hybrid-SWA pool classified (36/13 split — upstream main lacks it, PR candidate), and the **arch is now correctness-complete** — the graft's dense-prefix-layer NaN (config pops `first_k_dense_replace` → L0 built as sparse MoE → NaN; + a second `force_rope`/NoPE site) is fixed in 042, cross-team with R9700 who hit it on their FP8 WMMA path. **R9700 confirms `North-Mini-Code-1.0-fp8` serves coherently at true 256K** (65→34 tok/s, cuda-graph ON, 1.58M-tok KV pool) — the fp8 checkpoint is the **R9700 ship** (their native lane). *(R9700 2026-06-14: both 042 fixes now live in their FP8 serving path — they'd only carried the MoE-selection site; the `force_rope`/NoPE site was still open until today — and L0-RoPE is validated by a cheap positional gate: a temp-0 "reply with the 6th then 3rd word, in that order" prompt returns the exact order. Reuse that probe as the int4 ship's positional check.)* **FP8-on-Ampere stays blocked** — sm_86 triton can't compile e4m3 for fused-MoE and no marlin W8A16-fp8 MoE scheme exists in v0.5.12 *or* main. **Our path: AWQ-int4 build from upstream BF16** (calibration backlog №0) — the dense-layer fix means our int4 ship boots clean (it would have hit the same NaN). Preserve **thinking + tool**, eos 255001, ship the repo's `chat_template.jinja` verbatim, shapes group-128-clean (2048 hidden / 768 inter). Parsers — it's **two** separate grafts, not one (R9700 2026-06-14): (1) **reasoning** = `CohereCommand4Detector` in `parser/reasoning_parser.py`, `DetectorMap["cohere_command4"]`, wired via `--reasoning-parser cohere_command4` — R9700 grafted it from upstream main + validated on FP8 (routes `<|START_THINKING|>…<|END_THINKING|>`→`reasoning_content`, strips the `<|START_TEXT|>…<|END_TEXT|>` response wrapper; **note these delimiters are `special=False`, so `skip_special_tokens` does NOT remove them — without the parser they leak verbatim into `content`**); backend-independent, reuse our patch 053 verbatim for your int4 ship. (2) **tool-call** = a *separate* cohere `function_call` detector (`function_call/cohere_command4_detector.py`, `ToolCallParserEnum["cohere_command4"]`, `--tool-call-parser cohere_command4`) for `<|START_ACTION|>[…json…]<|END_ACTION|>`→structured `tool_calls` — R9700 grafted this too (patch 054, normalizes `tool_name`→`name`, drops `tool_call_id`) and validated on FP8: a tools request returns `finish_reason=tool_calls` with the call structured, non-stream + stream. Backend-independent, reuse verbatim — **North-Mini is fully agentic-ready on R9700** (thinking + tool calls + clean content), so both parsers are ready-made for your int4 ship. Validate with `--skip-vision` (text-only arch). **Scored agentic bench (R9700 2026-06-14, two runs):** the model + parsers DO drive real SWE-bench resolution end-to-end, but set int4 expectations to *middling, not leader*: **django-only was 6/9 (66.7%) — django-flattering**; on the **diverse buildable subset it's ~3/12 (25%)** (django 3/3, but seaborn/flask/requests/xarray 0/9 — patches apply, fix-to-pass tests fail), i.e. between your qwen36 (13%) and Coder-30B (40%). ⚠ The buildable run **hung at 12/15 on a large-xarray prompt** (heartbeat stopped 40 min → 1800s watchdog SIGQUIT, leaked VRAM on one TP rank). **CORRECTION (R9700 2026-06-15): this is EMERGENT, NOT a long-ctx serving bug — don't bound context for your int4 ship.** We chased it with 6 isolation/stress vectors against a live North-Mini, all clean: prefill@118K, decode@110K+744-tok, multi-turn radix-reuse@108K, eviction-churn (20×90K distinct), a 25-min/168-req soak, AND **host saturation @ load-avg ~13** (8 CPU burners + 3 disk/page-cache churners) concurrent with 113K-token requests — cold request slowed to 52s but completed; cached repeats held 38 tok/s. So the serving path is robust to deep ctx *and* host contention; the GPU-rank-only VRAM leak points to a **TP-rank/RCCL-collective stall** that only fires under the full opencode harness (tool traffic + concurrent scoring I/O + multi-hour) — same emergent class as the qwen36-27b hang (also irreproducible by host-saturation alone). For your int4 ship: **no context-bounding needed**, just keep an 1800s watchdog; if it recurs, `py-spy dump` the stalled scheduler for the deadlock stack (it's not isolation-reproducible, so a live capture is the only root-cause path). Drain a hung server with `kill -9 -<pgid>` (releases VRAM cleanly; un-drainable VRAM only follows a *partial* kill). Receipts + repro harness: R9700 `benchmarks/hsail/north-mini-hang/`. Both parsers (053/054) still reuse verbatim. Expect ~25% agentic.
2. **`google/diffusiongemma-26B-A4B-it`** (2026-06-09): block-diffusion Gemma MoE (`DiffusionGemmaForBlockDiffusion`, multimodal, 51.6 GB BF16). **No serving stack implements it** — sglang main and vLLM have no model file; transformers main has the reference. v0.5.12's dllm subsystem (`low_confidence`/`joint_threshold`) targets full-seq diffusion LMs, not block diffusion, so this is a real port (days–weeks), not a graft. Phased: (a) transformers-native GPU probe (characterize block size, template, quality), (b) port scoping against the dllm scheduler mixins, (c) int4 build (shares the gemma-4 A4B lineage — tower/recipe knowledge transfers). **Phase-(a) attempted 2026-06-11, blocked at first touch:** tx-main's new loader leaves the model on meta (no dispatch; tried `device_map="auto"` ± `max_memory`, torch 2.11 + accelerate 1.13) — revisit with a torch-nightly env or the next tx release (BF16 is on disk at `/data/models/hf-google/`). Reference-code scoping stands: asymmetric encoder+decoder stacks, **self-conditioning** logits threaded between denoise steps, `canvas_length` 256 — a custom scheduler model, confirming the days–weeks port estimate. Queued behind the AWQ backlog unless prioritized.
3. **Official Gemma QAT-W4A16-CT ships** (google, 2026-06-04/05: 31B / 12B / E2B): same QAT lineage as our RTN-from-QAT rebuilds (which only *beat* our AWQ on the 12B — already our shipped build). Low-priority compare: boot-smoke `gemma-4-31B-it-qat-w4a16-ct` vs our `gemma-4-31B-AWQ` on the standard instrument; adopt only if it wins.

### MoE coverage gaps (every base needs native + REAP + REAM)

Detailed matrix + rebuild paths in the [MoE coverage matrix](#moe-coverage-matrix--calibration-backlog) below. Six missing-variant builds, prioritized:

1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of bake-off top scorer (177/300 = 59.0%); 256→192e via `run_reap.py` on upstream BF16. **Tooling ready:** the fused-`Qwen3_5MoeExperts` unfuse + custom-router handling are built and miniature-validated (`scripts/quantize/test_qwen3_5moe_unfuse.py`, 7/7). Remaining = the on-box run: 62 GB BF16 → CPU offload on 48 GB VRAM (memory-marginal; R9700's 64 GB may be the better prune host), then AWQ recal with `thinking_vision_video` + `check_awq_scales.py --base`.
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of multimodal MoE. Blocked on tooling task below (Samsung SAIL needs Gemma 4 port).
3. **`Qwen3.6-VL-30B-A3B`** native + REAM + in-house REAP — rebuild VL trio with vision tensors retained (current REAP-26B is atbender pre-pruned, vision-broken). ⚠ pre-flight: SGLang's `Qwen3VLMoeForConditionalGeneration` loader was previously broken in v0.5.11 (no upstream fix found in v0.5.12 grep) — but main now ships the class with a registered EntryClass (2026-06-10 audit), making it a graft candidate like 042/043; smoke a community AWQ first before sinking calibration time.
4. **`Qwen3-30B-Instruct-2507`** native + REAP — REAM exists (`qwen3-ream`, fastest preset at 107 tok/s); complete the trio.
5. **`Qwen3.5-28B-A3B`** native + REAM — older DeltaNet+VL gen; only Cerebras REAP currently ships.
6. **`Nemotron-3-Nano-Omni`** REAP + REAM — native AWQ now ships (6/6 caps, 256K-verified); the REAP/REAM variants still need `run_reap.py` extended to the Mamba2-hybrid layout (only the 23 MLP/MoE layers prune per `hybrid_override_pattern`; the 23 Mamba + 6 attention layers stay BF16).

### Tooling

Items 2–3 are prerequisites for the MoE backlog. Detailed plan: [`scripts/quantize/ream_gemma4_port_plan.md`](scripts/quantize/ream_gemma4_port_plan.md).

1. **Upstream-PR sweep** — 12 of our 25 patches fix bugs still present in sglang main as of 2026-06-10 (see the patch map in [`patches/README.md`](patches/README.md)): the Qwen3.5/3.6 AWQ + CausalLM family (002 / 018 / 031 / 035), kernel correctness (003 / 011 — 011 also bites RDNA4 + Blackwell SM12.x), MoE gelu routing (017), Gemma 4 (004 / 026 + 025's `masked_fill` half), agentic robustness (034 / 041 — R9700-originated, coordinate the PRs with them). Upstreaming erases carry cost at every future rebase; the next rebase already drops 012 / 028 / 030 / 042 + most of 043 (fixed or native in main).
2. **Port Samsung SAIL REAM merge to Gemma 4 arch** — current `run_ream_qwen3moe.sh` + the upstream `merge.py` are Qwen3-family-only (5 hardcoded assumptions identified). Port unblocks the gemma-4-26B REAM build. Est. 40-60 h dev.
3. **Patch-052 candidate** — extend patch 003's dtype cast to the conv1d `KERNEL_WIDTH` spec-verify branches (`matrix_x` fp16/bf16 Triton assert); blocks ALL speculative decoding on the DeltaNet hybrids. Receipt: sprint-2 B4 bootfail logs. (051 is now the Cohere2Moe 256K enablement.)
4. **Decode ideas (receipts in the sprint log):**
   - **NGRAM spec — ✅ SHIPPED opt-in `NGRAM=1` on `coder-30b` (copy-heavy decode survives at depth).** Draft-model-free (no draft weights → stays at 256K where EAGLE3/DFlash OOM) and, crucially, it does **NOT collapse at depth** the way model-draft spec does (R9700's 107→0.8 t/s) — the draft is a CPU trie lookup, not a model attending the deep KV. Measured on `coder-30b` (server-log `gen throughput`, copy-heavy real-code task, deterministic temp=0 → lossless): @172K depth no-spec is 89 t/s steady, NGRAM sustains **235–237 t/s (~2.6×) on copy-heavy spans** (accept len 6.1–7.6), floor ~42 t/s on novel spans (never collapses). The win *grows* with depth (memory-bound verify amortizes the deep-KV read). Net is workload-dependent, so opt-in not default (it disables the overlap scheduler → neutral-to-slightly-negative on novel-text decode). Receipt: [`benchmarks/ngram-copyheavy-at-depth-2026-06-15.md`](benchmarks/ngram-copyheavy-at-depth-2026-06-15.md). **Scope: the FULL Qwen3-Coder-30B only (`coder-30b` / `coder-30b-eval`).** The win is gated by the model's **copy fidelity** (= n-gram acceptance), not architecture: the full coder copies verbatim (accept 6–7.6), but **REAP/REAM pruning degrades that** — `coder-reap-25b` accepts only ~2 and measures **net-negative** (153.6s vs no-spec 145.6s @172K, and draft=4 didn't rescue it), so the pruned coders are excluded from the `NGRAM=1` allowlist. Also excluded: DeltaNet thinkers (`qwen36*`, `qwen35-moe`) — their recurrent verify is sequential in num-draft-tokens → net-negative regardless of acceptance ([R9700 verify-wall](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)). Open follow-up: `--speculative-ngram-external-corpus-path` seeded with the edited repo. Note it can only be evaluated on a **real multi-turn agentic rollout**, not a synthetic single-request bench — when the target file is in the prompt, the in-context trie already covers it, so an external corpus is redundant; it only adds value when referenced code has scrolled out of the prompt window (large-repo / multi-turn). So this rides the bake-off, not a standalone probe (corpus format is JSONL, one JSON-string doc per line). **⚠ Cross-stack (R9700 2026-06-15): NGRAM does NOT transfer to RDNA4/triton — net-negative even on the FULL Coder-30B, for a reason orthogonal to your copy-fidelity gate.** Two RDNA4 blockers: (1) our May-25 sgl_kernel lacks the native `reconstruct_indices_from_tree_mask` op (NGRAM crashes on first request) — fixed via a sync-free GPU-vectorized torch fallback (kernel-exact 300/300, lossless); (2) the disqualifier — **profiled at true 244K depth: CPU prepare (trie+reconstruct) = 1.2ms (negligible); the GPU verify forward = ~2.7s, under `cuda_graph=True`** → **~0.6–1.4 t/s vs 12.2 no-spec**, *independent of acceptance*. So it's **NOT CPU and NOT eager** (both measured-out) — the **triton tree-attention verify re-reads the deep KV ~per-draft-token** (overhead 2×@8K → 33×@244K) instead of sharing one read across the draft tree the way your FlashInfer does (which is exactly why it's cheap for you). So **"backend-independent" doesn't hold** — it's a triton attention-kernel limit, and the only lever is an efficient tree-attention verify kernel. Receipt: R9700 `benchmarks/ngram-rdna4-at-depth-2026-06-15.md`. **✅ RESOLVED (R9700 2026-06-17): that verify kernel is BUILT (patch 065, opt-in `SGLANG_TREE_VERIFY_SPLITKV`).** A split-KV (flash-decoding) tree-verify kernel gives the verify the SM-split decode already has — measured **~12.8× faster verify @~53K (split-KV 53.7 vs stock 4.18 t/s, matched accept) → NGRAM net-positive at _mid_-depth (≤~64K)** on copy-heavy coding on RDNA4/triton. **⚠ Correction (R9700 2026-06-17, same-day): mid-depth win ONLY, NOT the 244K fix** — the earlier "net-positive at depth" phrasing overstated the @53K result. At true 244K the split-KV verify's occupancy collapses (0.57 vs stock 0.76) AND acceptance is depth-limited (~1.25), so NGRAM stays **net-negative at 256K**; no-spec remains the 256K path. Ships as a mid-depth opt-in (default-OFF); the 256K occupancy fix is deprioritized (R9700 task #16). Decomposition (verify is `skip_prefix_custom_mask=True`): prefix split-KV (mask-free, one shared deep-KV read for all D draft tokens) + cheap D×D tree-masked suffix + `merge_state`. Doesn't port to your FlashInfer (which already shares the read), but the **decomposition is the general technique** for any backend lacking a shared-read tree-verify. Default-OFF pending a bit-exact v2 (the merge rounds differently than the stock single-pass → near-tie bf16 flips on high-entropy NL tokens; code is exact). Thanks for `copyheavy_decode_bench.py` — we used it verbatim to elicit the copy-heavy decode (our homegrown whole-codebase prompt got refused → accept 1.05). Receipt: R9700 `benchmarks/tree-verify-splitkv-bench-2026-06-17.md`.
   - **Per-layer-type attention backend for Gemma long-ctx** — ⚠ likely blocked on sm_86: FlashInfer rejects head_dim=256 outright on Ampere (`Unsupported max_mma_kv: 0`), which is why every Gemma preset forces `--attention-backend triton`. The "sliding layers are FlashInfer-legal" claim needs a live FlashInfer-sliding-kernel test to confirm/refute before it's worth pursuing.
   - **EAGLE3-at-24GB via `--max-total-tokens` pool-capping** (qwen36-family pools are 3-9× over-provisioned) + `--speculative-draft-window-size` (untested).
   - ~~Allreduce acceleration for the dense-TP path~~ — **CLOSED, null/blocked on 3090 TP=2** (2026-06-15). Custom all-reduce (the only path that targets the per-layer decode allreduce) breaks cuda-graph capture on sm_86 — re-confirmed on v0.5.12 (`custom_all_reduce.cuh:508`), and graphs win more; `--enable-torch-symm-mem` is a null (it touches only the embedding allreduce — qwen36-dense decode 67.4 vs 67.6 @2K); `--enable-symm-mem` won't even link (`-lnccl` missing); flashinfer fusion was already null. Baselines + full table: [`benchmarks/allreduce-accel-null-2026-06-15.md`](benchmarks/allreduce-accel-null-2026-06-15.md). One-flag re-test toggle (`ENABLE_CUSTOM_AR=1`) kept for a future sglang/driver bump.
5. **Extend `run_reap.py` to remaining MoE layouts.** `run_reap.py` + the unfuse patches are in-repo (`run_reap.py` ported from R9700; the Coder-30B-A3B-REAP ship used the Qwen3Moe path). Coverage: (a) **`Qwen3_5MoeExperts`** (Qwen3.5/3.6 fused 3-D experts + `Qwen3_5MoeTopKRouter`) — ✅ done: `patches/qwen3_5moe_unfused_experts.py` (load-split + save-fuse hooks) + tuple-router handling in the saliency hook, miniature-validated 7/7 by `test_qwen3_5moe_unfuse.py`; (b) Gemma 4 parallel dense+MoE + different expert keys — ❌ TODO; (c) Nemotron-H Mamba2-hybrid (only the 23 MLP/MoE layers pruneable per `hybrid_override_pattern`) — ❌ TODO. The saliency tracker + `prune_model` are arch-agnostic once `.mlp.gate` + per-expert `.mlp.experts.{i}.down_proj` modules exist — the unfuse patches create them.

## Coding-eval bake-off (SWE-bench Lite, v2 Docker harness, 256K, single-user)

Top tier: `qwen36-dense` (Qwen3.6-27B dense, thinking) **leads at 62.0%** on opencode, ahead of the A3B-MoE thinkers `qwen36` and `qwen36-ream` (59.0%). `qwen36` is a strong, consistent three-scaffold performer — **opencode 59.0% / little-coder 59.0% / claw 51.7%** — with the `developer`-role chat-template fix (†) applied. REAM ties native on opencode (both 59.0%) but **trails ~9 pp on little-coder** (`qwen36-ream` 150/300 = 50.0% vs `qwen36` 59.0%, full-300 both, 64 empty patches in the REAM cell) — the merge is scaffold-sensitive on the thinking ships, so "REAM ties native" holds on opencode/claw but not uniformly. The fleet is re-running every little-coder cell with that fix (cycle 3 of 9).

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-dense` (Qwen3.6-27B Dense AWQ, thinking) | **186/300 = 62.0%** | — | re-run † |
| `qwen36` (Qwen3.6-35B-A3B AWQ-Marlin, thinking) | **177/300 = 59.0%** | **155/300 = 51.7%** | **177/300 = 59.0%** |
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | **177/300 = 59.0%** | 74/168 † (partial) | **150/300 = 50.0%** |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% | 107/300 = 35.7% | 74/300 = 24.7% |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% | 122/300 = 40.7% | 107/300 = 35.7% |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% | 109/300 = 36.3% | re-run † |

† **little-coder/claw × thinking cells are re-running with the `developer`-role chat-template fix.** little-coder's pi-ai sends its system prompt as role `developer`; the Qwen3.5/3.6 templates 400'd on it, so thinking rollouts exited empty (little-coder 0/30; claw stuck at partial prediction sets). Fix: `scripts/eval/patch_chat_templates_developer_role.py` (in `setup.sh`). Confirmed on the re-run — `qwen36` little-coder **0/30 → 177/300 = 59.0%**, claw **partial → 300/300 = 51.7%**. `qwen36-ream`/`qwen36-dense` claw cells still show stale partials until cycles 4/7 re-run. Root cause + the wrong earlier readings it overturned: `CLAUDE.md` + `git log`.

Failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling of 49% across opencode∪claw, rollout self-clean), methodology, and per-cell receipts: [`patches/README.md`](patches/README.md) + [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/). Every preset's `--tool-call-parser` matches its chat-template tool format (see Known Issues).

## Speculative decoding

EAGLE3 + DFlash work against our **INT4/AWQ targets** (draft stays BF16; target quant is independent). Receipt: `benchmarks/quality/specdec-v0512-2026-05-29.json`.

| Target | Algo / Draft | Baseline | With spec | Speedup |
|---|---|:---:|:---:|:---:|
| `coder-30b` AWQ-native | EAGLE3, `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge` (steps 4 / topk 4 / draft 8) | 185 tok/s | **306 tok/s** | **1.65×** |
| `qwen36` AWQ | DFlash, `z-lab/Qwen3.6-35B-A3B-DFlash` (`--dtype bfloat16` + spec-v2) | 126 tok/s | 126 tok/s | **~1.0× (moot)** |

DFlash buys nothing on `qwen36` — graph-ON no-spec already decodes 126 tok/s @256K (174 @1K), matching DFlash at its 32K cap. A second reason (beyond the 24 GB-fit limits below) no-spec is the only viable path.

> **Cross-team corroboration (R9700 2026-06-14, numbers corrected):** we reach the same "DFlash isn't worth it on the Qwen DeltaNet-MoE family" conclusion from the other side. Tested `Qwen3.5-28B-A3B-REAP` + the parent `z-lab/Qwen3.5-35B-A3B-DFlash` transfer on our RDNA4 triton path (`--speculative-attention-mode decode`, TP2): *functional* — coherent, 4/4 caps incl. vision+video, accepts (len 2–8 workload-dep) — but **no useful speedup**: authoritative **server-log sustained gen-throughput ≈52 tok/s short-ctx vs ~60 no-spec** (net-negative-to-neutral; block-diffusion draft+verify overhead ≈ the acceptance benefit), and decode collapses deep (≤~2 tok/s by ~240K, same wall as Coder-Next-REAM). Matches your qwen36 DFlash *moot* (~1.0×) finding. Net: for both stacks, **don't pursue DFlash on the Qwen3.5/3.6 DeltaNet-MoE ships** — no-spec is the path. ⚠ **Measurement heads-up for your spec benching:** the streaming-client TPOT path (our `measure_decode_curve.py`, likely your equivalent) **under-measures bursty spec decode ~2×** — SSE delivers an accepted block as near-simultaneous deltas, but client-side buffering spreads them, so the inter-token timing reads ~2× slow. It read 25.5@128 here where the server-log `gen throughput` is ~52. **Use the server-log `gen throughput` for spec-decode, not client TPOT** (client TPOT is fine for non-bursty no-spec). Worth auditing any spec tok/s you measured via client streaming.

> **Cross-team follow-up (R9700 2026-06-14): EAGLE3 extends the "DeltaNet spec is moot" finding — and a portable transformers-v5 config bug you'll hit.** We tested EAGLE3 (not DFlash) on the **dense** DeltaNet `Qwen3.6-27B` with a geometry-matched draft (`LlamaForCausalLMEagle3`, 1.25 GB). It **accepts beautifully (accept len 4.0)** yet is still **net-negative** (~18 vs ~25 no-spec, server-log). So acceptance quality was never the problem on DeltaNet: the **recurrent verify is sequential in num-draft-tokens** (verifying an 8-token tree ≈ 8 serial state updates → the spec step is GPU-compute-bound at ~5.5× a plain decode). **cuda-graph gives zero help** (18.3 ON / 18.1 OFF) → compute-bound, not launch-bound; deeper trees only make it worse. Net for both stacks: **don't chase a "better draft" (EAGLE3/DFlash/MTP) for any DeltaNet target** — the only lever is a DeltaNet-aware parallel/chunked verify kernel. ⚠ **Portable bug (bites you on v0.5.12 + transformers-v5, spec or not):** these EAGLE3 drafts use a *decoupled* `head_dim` (hidden 5120 / 24 heads / head_dim 256 → q_proj 6144), and transformers-v5's `@strict` `LlamaConfig.validate_architecture` **aborts config-load** with "hidden size not a multiple of num_attention_heads" even though `head_dim` is explicit. `@strict` snapshots validators into `cls.__class_validators__` at class-decoration, so the fix is to **strip the entry from that list** at import (rebinding the method is a no-op) — monotonic, normal configs pass it anyway. Our recipe is `patches/055-rdna4-eagle3-deltanet-vl-enablement.patch.CANDIDATE` (a `_patch_llama_head_dim_validation()` in our `hf_transformers_patches.py` + a `qwen3_vl` EAGLE3 capture-layer-marking fix). Reuse the validator-strip verbatim the moment you load any decoupled-head-dim draft/model.

> **Cross-team finding (R9700 2026-06-15): spec-decode `@256K` is a SHORT-DEPTH artifact — at true 256K decode depth on real content it COLLAPSES, and it is NOT DeltaNet-specific.** We finally decoded at true ~244K depth on diverse real code (server-log gen-throughput; in-session depth A/B — one server boot, two prompts, *only* prompt depth varies). **Coder-30B EAGLE3 AWQ (pure-attention MoE): 107 tok/s short → 0.8 @244K** (accept 6.12→1.75) = **~15× slower than no-spec @244K (12.3)**; **Qwen3.6-35B DFlash AWQ: ~1.2 @240K** (vs the doc'd 80). So the depth-collapse **generalizes beyond the DeltaNet verify wall (above)** — even pure-attention spec dies at depth, for two reasons: (1) the draft's acceptance craters at depth (drafts are short-ctx-calibrated → can't predict hard real-code continuations at 244K), and (2) the draft attends the *full* 244K KV every micro-step. **The documented `@256K` spec bars — ours AND very likely yours — were measured at SHORT depth on a 256K-*capable* server:** our "97 @256K" commit's own log reads `max_total=817979, huge KV headroom` (a near-empty pool). **Action for the 3090:** re-read every spec `@256K` tok/s as "short-depth on a 256K server", and re-measure at depth on REAL content (not filler — repetitive filler trivially inflates draft acceptance) before claiming a 256K spec win — your cross-stack table's 306 / 126 are likely short-depth too. For the shared 256K mandate, **no-spec is the path at depth; spec is a ≤~32–64K optimization.** Receipt + harness in the R9700 repo: `benchmarks/spec-at-depth-collapse-2026-06-15.md`, `scripts/bench/spec_depth_ab.sh`.

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

- **`qwen36-ream` × claw-code is a partial cell (168/300 predictions = 74/168 = 44.0%).** Discount it per the full-300 rule. The 2026-06-16 resume reran the claw rollout but it stalled at 168 (`reroll rc=3` — the GLIBC-sensitive claw scaffold, a known rollout landmine, not a model issue); little-coder is full (150/300 = 50.0%), opencode full (177/300 = 59.0%). Only claw-code stays short.
- **Host reboots every ~9–17 h under sustained docker rollout I/O (kernel BUG).** Predictions on disk survive; auto-resume is via `swebench-bakeoff.service` (the boot-ordering cycle that was silently dropping it at every boot is fixed — cooling oneshot now orders after `nvidia-persistenced`, not `multi-user.target`). Full forensic recipe in [`CLAUDE.md`](CLAUDE.md) → Operational Lessons.

One caveat carried forward: `check_awq_scales.py` reads native-AWQ format — CT-format checkpoints crash its tensor reader (use a native-AWQ mirror or HF Range-fetch mode for CT audits). Resolved items live in `git log` + [`patches/README.md`](patches/README.md).

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.13.post1, apply patches, create conda env

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
| **Nemotron-3-Nano-Omni-30B-A3B AWQ** | Mamba2-hybrid MoE A3B (128 exp, AVLM) | **262K** ✓ (5.25M pool) | 103 (**98 @256K**) | `nemotron3-omni` | [`mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ). int4; **6/6 caps** (basic+thinking+tool+vision+**video**+**audio** — the only audio ship). Serves on v0.5.13.post1 via **patches 052** (non-gated squared-ReLU `moe_wna16`) + **053** (EVS video routing, ex-R9700 057). Mamba2 O(1) recurrent → decode ~flat (102.7→97.8 @255K); beats R9700 FP8 (74.79/49.22). |
| **Qwen3-Coder-30B-A3B AWQ** | MoE A3B (128 exp) | **262K** | ~30 @256K | `coder-30b` or `coder-30b-eval` | [`mattbucci/Qwen3-Coder-30B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ). Two presets serve the same model; for short-ctx batch-decode benchmarks override `CTX=16384 MAX_RUNNING=32 ./scripts/launch.sh coder-30b` (peaks ~187 tok/s @ 1K). |
| Coder-REAP-30B AWQ-Marlin | MoE A3B (96 exp) | **262K** | 109 | `coder-reap-25b` | [`mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) (R9700 in-house). |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **262K** ✓ (347K pool) | 57 (31 @64K, **22 @256K**) | `gemma4-31b` | [`mattbucci/gemma-4-31B-AWQ`](https://huggingface.co/mattbucci/gemma-4-31B-AWQ). LM INT4, vision tower FP16, **KV fp8_e5m2**. Tool-use 1.0 → 258K true tokens, 5/5 caps incl. video. KV 24K→347K 2026-06-10/11 (`--swa-full-tokens-ratio 0.05` + `MEM 0.92` + e5m2 FP8 KV — e5m2 is the only FP8 that compiles on the triton-forced path, sm_86 rejects e4m3; declared 262,144 now fits the pool with 32% headroom). *(R9700 cross-team 2026-06-14: your SWA-ratio lever moved the dense 31B on gfx1201 from a mislabeled "~105K weight-bound" to a **570K-tok pool @262144** — confirming it's SWA-pool-bound, not weight-bound, on RDNA4 too. Notably **on native `fp8_e4m3` KV, no e5m2 needed** — gfx1201 compiles e4m3 on the triton-forced path, so your "gfx1201 may not need e5m2" open question = confirmed. Ratio 0.0625 + MEM 0.92, needle PASS ~105K.)* |
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
| Nemotron-3-Nano-Omni-30B-A3B (128e, AVLM) | ✅ serves (v0.5.13 + patches 052/053, 6/6 caps) | ❌ | ❌ |

**Calibration backlog (prioritized):**

0. **`North-Mini-Code-1.0-AWQ`** (NEW, user-unlocked 2026-06-11) — from `CohereLabs/North-Mini-Code-1.0` BF16 (61 GB): GPTQ → CT → AWQ-Marlin. **Preserve thinking + tool** (`<|START_THINKING|>` template; `thinking` + function-calling calibration mix — `thinking_vision`-class recipe minus vision, e.g. AM-Thinking + Hermes-function-calling + python-instruct); eos 255001; ship the repo `chat_template.jinja` verbatim; group-128-clean shapes (2048 hidden / 768 expert-inter / 128 experts top-8 sigmoid). Serving side ready on 3090 (patches 042+051); fp8 path is R9700's.
1. **`Qwen3.6-35B-A3B-REAP-AWQ`** — REAP of the bake-off top scorer (177/300 = 59% × opencode). `run_reap.py` from upstream Qwen3.6-35B-A3B BF16 (256→192e, matching the REAM expert count); AWQ recal with the `thinking_vision_video` recipe. Tooling ready (fused-`Qwen3_5MoeExperts` unfuse + `Qwen3_5MoeTopKRouter` saliency hook, miniature-validated 7/7). On the on-box run, confirm the routed-expert prune leaves the shared expert + vision tower intact (`save_pretrained` keeps all non-MoE tensors; the unfuse patch re-fuses experts to the standard 3-D format on save).
2. **`gemma-4-26B-A4B-REAM-AWQ`** — REAM of our multimodal MoE. Samsung SAIL `merge.py` needs porting to Gemma 4 arch (currently only Qwen3 family is wired); AWQ recal must preserve vision tower BF16.
3. **`Qwen3.6-VL-30B-A3B-AWQ`** (native) + **`-REAM-AWQ`** + **in-house `-REAP-AWQ`** — multimodal A3B base. Current REAP (`Qwen3.6-VL-REAP-26B-A3B-AWQ`) was calibrated on atbender's pre-pruned BF16 which stripped the vision tower → vision broken. Need all three flavors from the upstream BF16 with vision tensors retained.
4. **`Qwen3-30B-Instruct-2507-AWQ`** (native) + **`-REAP-AWQ`** — text generalist. REAM exists (the fastest preset, 107 tok/s); native + REAP complete the trio.
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

**Fleet integrity (2026-05-31): all shipped AWQ models are scale-integrity clean** — a fleet-wide `check_awq_scales.py --base` audit found zero real zero-over-live (v2-disaster) defects; every flag resolved to benign MoE dead-channel structural sparsity (the flagship qwen36 passes the full scales+qweight audit, 0/61940). Capability-wise, the v0.5.12 ships are validated and **thinking + image + video are intact** (qwen36 / qwen36-ream / gemma4-31b 5/5, qwen35-moe 4/4, devstral 3/3 image-only, qwen3-ream 1/1 text-only). The remaining gap is the *static* eval suite below. Full per-model verdict + capability receipts: [`benchmarks/quality/fleet-integrity-audit-2026-05-31.json`](benchmarks/quality/fleet-integrity-audit-2026-05-31.json).

Run with `scripts/eval/eval_quality.py` (or `eval_and_chart.py` / the full-fleet `run_v0512_fleet_eval.sh` orchestrator): MMLU, HumanEval pass@1, [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 subbenchmarks), Needle-in-Haystack (**1K → 250K** — reaches our 256K target; timeout scales with context). The **bottom 8 rows are the v0.5.12 fleet, refreshed 2026-06-08** at MMLU 57 / HumanEval 40 / LAB-Bench 140 (20-per-subbench) — firmer than the first v0.5.12 pass but still treat ±a few points as sampling noise. The **Needle column is the deepest length where all 3 depths (0.1 / 0.5 / 0.9) retrieve**; it tracks each model's real KV pool at *measurement time*. ⚠ The multimodal rows' needle ✓-caps (devstral ✓131K, gemma4-26b ✓131K, gemma4-31b ✓16K, gemma4-12b ✓65K) **predate the 2026-06-10/11 KV unwalling** (SWA-ratio + e5m2 + MEM levers — current pools in the [VRAM context limits](#vram-context-limits-kv-dtype-varies-tp2-48-gb-total) table) — the 256K tool-use probe below is the current-depth instrument: every tool-trained ship now retrieves-and-acts at 1.0 to its pool (258K true for the Gemmas, 178K for devstral). The 256K reasoning probe below independently confirms the Gemmas retrieve **and** reason to true 256K (its multikey sub-task is multi-needle retrieval), so the Gemma needle cells here are updated to ✓256K¶ (¶ = via the reasoning probe, 2026-06-17); devstral's single-needle refresh rides the next fleet eval pass. Thinking-model MMLU/LAB read sensibly because the eval reads `reasoning_content` (the reasoning-parser routes answers there) and gives thinking models budget to close `</think>` before answering. v0.5.11 rows used full LAB-Bench (~1786) but otherwise comparable samples.

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
| Gemma 4 31B Dense AWQ | 93.0% | **97.5%** | **42.9%** | ✓256K¶ | `gemma4-31b.json` |
| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | ✓256K¶ | `gemma4.json` |
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

**256K reasoning quality** — `scripts/eval/probe_256k_quality.py`. Retrieval/tool-calling at 256K is necessary but not sufficient for *high quality* at 256K; this probe tests real reasoning over a full context — multi-key retrieval (3 of 5 spread facts) + variable-tracking (find a 3-step dependency chain scattered through the filler AND compute it) + aggregation (sum 5 scattered values) — identical task instances at every length (temp=0, seed reset per length) so any drop is real degradation, measured at **TRUE actual-token lengths** (top point 255,800 real prompt tokens, `usage`-verified; calibrated to ~6.9 char/tok so `approx==actual`). **The probe elicits chain-of-thought + parses an explicit `ANSWER:` line (fixed 2026-06-17).** The earlier terse "reply with just the number" form silently measured *no-CoT mental arithmetic* — it flattered think-by-default models and penalized instruction-followers that obey it. gemma4 reasons in `content` (no thinking channel), so it needs the CoT-eliciting prompt to be measured fairly.

| Preset | 1K | 32K | 65K | 131K | 200K | 256K | overall |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemma4` (26B MoE) | 100 | 67§ | 100 | 100 | 100 | **100** | 94 |
| `gemma4-31b` (dense) | 100 | 100 | 100 | 100 | 100 | **100** | **100** |
| `gemma4-21b-reap` (MoE) | 100 | 100 | 100 | 100 | 100 | 67§ | 94 |
| `gemma4-12b` (unified omni, int4 RTN) | 100 | 67◊ | 67◊ | 33◊ | 33◊ | 67◊ | 61 |
| `qwen36` / `qwen36-dense` / `qwen36-ream` ‖ | 100 | 100 | 100 | 100 | 100 | **100** | **100** |

**The 26B / 31B / 21B-REAP Gemmas reason AND retrieve correctly to true 256K (94–100%)** — this *overturns* the prior reading that the Gemmas' reasoning "softens by ~64–110K" and that the high-quality-256K tier was Qwen3.6-only. That softening was the terse-probe artifact (CoT suppressed → fragile mental math), not a Gemma limitation; root-caused live (`verify_reason_raw.py`: gemma4 @32K vartrack terse→"156" wrong, reasoning-allowed→correct→84). §Isolated single-task cell misses (gemma4 32K multikey; gemma4-21b-reap 256K aggregate) are non-monotonic (each passes at every *other* depth) → single-instance noise, not a depth cliff. ◊**`gemma4-12b`** (smallest ship, int4 RTN-from-QAT, encoder-free omni; restored on v0.5.13 by patch 055) is the exception — it **chain-reasons to 256K (`vartrack` 100% at every depth)** but its multi-needle *retrieval* saturates ~3 needles at depth, so the 5-needle `aggregate` fails consistently ≥32K (61% overall). Raw dump confirms the mechanism is **retrieval, not arithmetic/format**: at 65K it finds only 3 of the 5 planted items and explicitly says "the text only lists three items" — a genuine small-model deep-retrieval limit (vartrack passing proves CoT works, so it's not the terse-probe artifact). ‖The qwen3.6 families were measured on the predecessor terse probe, which they passed (100%) because they think by default; a corrected-probe re-run rides a later full pass. **Corrected-probe re-measure of the rest (2026-06-17, to 178K — the deepest common length fitting devstral's 202K pool):** the CoT-eliciting fix **overturns the "non-thinking can't reason" reading too** — `qwen3-ream` (Instruct, non-thinking) **33→100%**, `qwen35-moe` (thinking) clean **100%** (its terse "softens past 131K" was largely the artifact; the 200K+ tail still to confirm), `devstral` (coder, non-thinking) **~40→80%** (`vartrack` 100% at every depth; the scattered multikey/aggregate misses are non-monotonic noise). So the earlier "thinking is the discriminator — non-thinking ships fail multi-step reasoning even at short context" reading was the terse-prompt artifact, now retired: **every measured ship reasons well to ≥178K once the probe elicits CoT.** Re-run with `scripts/eval/run_reason256k.sh`. Receipts: `benchmarks/quality/quality256k-*-v0513.json`.

**KV-pool status after the 2026-06-10/11 unwalling** (SWA-ratio right-sizing + e5m2 FP8 KV + MEM tuning; sprint history in [`patches/README.md`](patches/README.md)): **every Gemma ship now serves 256K-class** — 12B 565K / 26B 652K pools at FP16 KV, 31B 347K pool via `fp8_e5m2` KV (the reusable sm_86 rule: **triton-forced attention takes FP8 KV as e5m2, never e4m3** — sm_86 triton can't compile `fp8e4nv`). Remaining holdouts are **devstral 202K** (full-attention-bound; 262K needs MEM≈1.0) and **qwen3-vl-32b 131K** (model-card cap). Quality nuance: serving 256K ≠ reasoning at 256K, but **the Gemmas clear both** — corrected 2026-06-17, they reason to true 256K at 94–100% (reasoning table above), joining the Qwen3.6 family (100% flat to 255K) in the high-quality-256K tier. The earlier "Gemmas are retrieval-only / soften by ~64–110K" reading was a terse-prompt probe artifact, now retired. qwen35-moe's past-131K softening still stands pending a corrected-probe re-run.

**Thinking serving defaults (2026-06-07, cross-team with R9700).** Every thinking preset (the `--reasoning-parser` ones: qwen36 / qwen36-dense / qwen36-ream / qwen35-moe / gemma4) now launches with **`--sampling-defaults model`** — it uses each checkpoint's own recommended sampling (Qwen3.6 ships temp 1.0 / top_p 0.95 / top_k 20) instead of SGLang's generic temp 1.0 / top_p 1.0 / top_k −1 (untruncated tail). The model's top_p/top_k truncation is the *principled* anti-overthinking lever: it curbs the int4 degenerate-repeat / overthinking-spiral failure mode ([arXiv:2606.00206](https://arxiv.org/abs/2606.00206)) and the temp=0 `"</think> Paris </think> Paris…"` greedy-decode loop, **with no token cap**, so deep single-user 256K reasoning is untouched. For the *agentic* case where int4 thinking spirals without committing a tool call, an **opt-in `STRICT_THINK=1`** adds `--enable-strict-thinking` so a per-request `custom_params.thinking_budget` can bound the think-loop (R9700: budget≈300 turned 0→1 applied edits) — deliberately **off by default** (a ~300-token cap would gut the 256K reasoning win). MoE-thinking smoke on live qwen36 confirms both: default → reasoning correct + thinking terminates + tool-calls valid; `STRICT_THINK=1` → `thinking_budget=32` collapses the think pass 646→87 chars while still answering, and tool-calls stay 4/4 valid with or without a budget.

**↩ R9700 ANSWER to the int4 × triton-attention isolator — your hypothesis is refuted in-regime (2026-06-15).** We ran your proposed control as a **same-session, single-variable agentic A/B**: same Qwen3.5-27B int4 weights, same 6 SWE-bench Lite instances, same 2048 cap + **64K ctx (exactly the regime our documented 0/6 was measured in)**, only `--attention-backend` varies {torch_native, triton}, both arms back-to-back so the triton arm is an in-session baseline reproduction. **Both arms = 0/6, 6/6 empty diffs.** torch_native ran *clean* (all rc=0, full multi-hundred-second agentic loops — django 279s, pylint 493s — so a real never-commit result, not a serve artifact; servers healthy, no OOM/HIP/500s); triton control reproduced the baseline (3 hard 601s timeouts). Swapping triton→torch_native at the exact 64K where the 0/6 lives **rescues zero instances** → **attention precision is NOT the lever**; it's the int4 weight-noise agentic-correctness ceiling, now confirmed in the multi-turn regime (and corroborated by the earlier short-ctx coherence probe: over-think 2/3 on both backends). *Secondary:* torch_native spiraled **less** (0 timeouts vs triton's 3) — the backend changes the failure *shape*, not the outcome (mildly consistent with your patch-011 thesis that triton erodes a bit more, but not enough to matter for resolve). *Honest caveat:* torch_native's ROCm MATH-SDPA OOMs >~64–150K so we can't probe 64–128K there — but triton is **already 0/6 at 64K**, so the failure is fully present in-regime and a 64K test is decisive for it. **Implication for us:** the lever forward is **recalibration / FP8 for agentic**, not a triton-attn rekernel — your FlashInfer cleanliness is likely the GPU/quant path difference (your int4 is AWQ_Marlin, not the same kernel surface), not attention precision rescuing int4. Receipt: `benchmarks/quality/int4-torch_native-agentic-ab-2026-06-15.md` + harness `scripts/eval/int4_agentic_sweep/torch_native_ab.sh` in the R9700 repo.

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

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner (native gfx1201 FP8); RDNA4 serving stack. Originated the FP32-softmax patch 011 and the CT→native AWQ converter; shipped the EAGLE3/DFlash spec-decode recipes; caught the cohere2_moe dense-layer NaN (042) and the little-coder `developer`-role 400. Both stacks publish under `mattbucci/*`.
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
