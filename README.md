# NVIDIA Inference: SGLang on 2x RTX 3090

Single-user, **256K-context** LLM inference on 2× NVIDIA RTX 3090 (GA102, Ampere, 48 GB total) — SGLang **v0.5.22** + 32 local patches, CUDA 13.2 / PyTorch cu130, every model an **AWQ-int4 ship calibrated in-house** from the upstream BF16 base. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

Long-form material lives in [`docs/`](docs/) — [decode levers](docs/decode-levers.md) · [SWE-bench harness](docs/swebench-bakeoff.md) · [quality-eval methodology](docs/quality-evals.md) · [roadmap detail](docs/roadmap.md) · [host setup](docs/host-setup.md) · [OCI image](docs/oci-image.md) — and the per-patch history in [`patches/README.md`](patches/README.md). Agent operating rules: [`CLAUDE.md`](CLAUDE.md), [`rules-for-agents.md`](rules-for-agents.md).

## Direction

**We optimize for 256K-context single-user agentic workloads on AWQ-int4 ships.** SWE-bench Lite is the canonical eval — every preset serves at full 256K (or model-card max) so agentic harnesses with multi-turn tool-call context (median ~41K, p90 ~82K, max 230K per instance) actually fit. Decode tok/s / TPOT at depth is the primary metric; multi-user throughput is secondary and never bought with single-user latency.

This rules out three axes other 3090 stacks chase:

| Their focus | Why not ours |
|---|---|
| Short-ctx multi-stream throughput (vLLM-style 80-140 TPS @ <32K) | We need the full prompt of an agentic instance in context; truncating mid-conversation loses correctness. |
| FP8 quantization | **FP8 W8A8 MoE doesn't compile on sm_86** (the Triton fused-MoE kernel needs `fp8e4nv`; Ampere Triton only has e5m2), and even where FP8 is native (R9700 gfx1201), AWQ-int4 wins single-user M=1 decode — weight-byte-bound, FP8 is 2× int4 (Coder-30B 56 vs 38 tok/s). FP8 is the R9700 lane. Receipt: [`benchmarks/fp8-vs-awq-coder-reap.json`](benchmarks/fp8-vs-awq-coder-reap.json). |
| Spec-decode (EAGLE3 / DFlash / MTP) | Draft acceptance collapses at depth and 256K + draft + cuda graphs OOM on 24 GB cards; 97% of SWE-bench prompts exceed the drafts' caps. Opt-ins only — the one that holds at true 256K is DSpark on `qwen38` (fp8 KV, 1.2× at 250K) — [`docs/decode-levers.md`](docs/decode-levers.md#speculative-decoding). |

What we **don't** ship: random community quants. Every `mattbucci/*-AWQ` is calibrated end-to-end from the upstream BF16 base via our own GPTQ → CT → AWQ-Marlin pipeline, with thinking + image + video + audio preserved and probed. When a model needs MoE expert compression we run REAP (pruning) or REAM (merging) ourselves on the upstream weights (`scripts/quantize/run_reap.py`, `run_ream_qwen3moe.sh`).

## Results at a glance

![Per-model single-user decode tok/s — peak (short ctx) vs 256K, or the model's real KV cap where it doesn't reach 256K — 2026-10-25 refresh at the 300 W ship cap, gpu-unlock on; Flint re-swept 2026-10-31 on fp8_e5m2 KV](benchmarks/all_models_decode.png)

![Power cap 300 W vs 400 W per card, same boot — decode at 1K / 32K / deepest point for every preset; +0–5 %, the cap barely moves weight-streaming-bound decode](benchmarks/all_models_power_300v400.png)

![Single-user decode tok/s vs context length — all AWQ presets, unified 256K x-axis](benchmarks/all_models_context.png)

![SWE-bench Lite resolve rate per preset × scaffold under the isolated harness — landed full-300 cells as bars, queued/in-test cells marked pending](benchmarks/bakeoff_swebench_lite.png)

![Speculative decoding — AWQ int4 vs +draft single-user decode; Qwen3.8-27B DSpark on the shipped config (fp8 KV, true 256K) at 60-tok / 49K / 130K / 250K](benchmarks/specdec_comparison.png)

![Power-cap sweep 200–400 W per card with gpu-unlock on — qwen38 decode and fresh-prefill TTFT at 1K/32K/128K, 8192² tensor rate, GPU and DDR5 DIMM temperatures; 300 W chosen](benchmarks/power-sweep-2026-10-24/power-sweep.png)

Every deep point is **server-verified at its labeled depth** (`actual_input_tokens`; deepest honest point 255K = 262144 − output − margin), single-user (M=1), fresh prefill, at each model's real KV pool. **True-256K decoders:** the A3B DeltaNet MoEs lead (`qwen35-moe` / `qwen36-ream` **144 @255K**, `qwen36` 121, ~210 short), the Mamba2-hybrid `nemotron3-omni` is near-flat (101 → 93), the 30B-A3B coder trio sits at ~69 @255K (200 short), and the dense DeltaNet 27Bs (`qwen36-dense` / `qwen38`) at ~48 (70 short). The triton-backend presets (Flint, Gemma 4) were under-filling the GPU at depth until 2026-10-22 — the triton decode kernel's split-KV cap defaults to 8, which at M=1 / TP=2 puts 16–64 CTAs on 82 SMs; `--triton-attention-num-kv-splits 64` now ships on all five and moved `flint` 36 → **90 @262K**, the Gemma 4 MoEs and the 12B omni 24 / 17.5 → **58**, and the dense `gemma4-31b` 12.9 → **30** (exact attention; it beats the opt-in decode-topk lever's 26.2) — [flint bracket](benchmarks/flint/kv-splits-bracket-2026-10-22.md), [Gemma bracket](benchmarks/gemma4-kv-splits-2026-10-22.md). Expert pruning/merging does **not** speed M=1 decode (activated experts + attention dominate) — REAP/REAM win on KV headroom and footprint instead. **Quality:** the first isolated SWE-bench Lite cell is in — `qwen38` **196/300 (65.3 %)** on opencode, from a five-scaffold harness bake-off that is a statistical tie across the opencode / pi families (203 / 202 / 196 / 193; prime 187) and settled the eval harness on **plain opencode** ([receipt](benchmarks/quality/qwen38-v4-cycle-close-2026-10-22.md)); every historical cell (including the former `qwen36-dense` 62.3 % lead) ran with network open and under since-fixed scaffold context / output-budget defects — on the same 300 ids the exposed netopen cell scored 250 — so those reads are withdrawn and the remaining presets re-roll on the one lane; the Qwen3.6 family and the Gemma 4 26B/31B reason and tool-call perfectly at true 256K, while the Coder-30B family hits a ~64K agentic ceiling. Receipts and the full ledger: [`docs/decode-levers.md`](docs/decode-levers.md), [Quality evals](#quality-evals). **Speculative decoding (new, 2026-09-27):** DSpark spec-decode on `qwen38` is a validated **1.4–1.5× single-user decode win on our CUDA verify** (60-tok 1.54×, 34K 1.47×, 49K 1.39×; up to 2.79× on predictable output; **1.21× at 250K actual — 58.5 vs 48.4 tok/s — with the true 256K window kept because fp8 KV leaves a 307K-token pool under the draft**) — R9700's 0.24×-at-49K Triton cliff does **not** reproduce on flashinfer verify ([receipt](benchmarks/quality/dspark-cuda-depth-ab-2026-09-27.md), [fp8-KV/250K receipts](benchmarks/quality/dspark-cuda-2026-09-27/)). It serves the bake-off's `qwen38` cycle from 2026-09-27 (`SPEC_DECODE=1`). Drafts exist for `qwen38` (DSpark), `qwen36`-family (DFlash), `coder-30b-eval` + `devstral` (EAGLE3); NGRAM (no draft) covers the rest. **The spec-decode chart above is current (shipped config, 2026-09-27); in the decode/context PNGs the Flint and Gemma 4 curves are current (2026-10-22, kv-splits 64) while the other presets predate the current stack — re-measuring the fleet with spec-decode + current settings is a GPU campaign queued for the next serving-only window (it cannot run while a bake-off lane holds the GPUs).**

## Status & next steps

**Live:** the SWE-bench Lite re-roll is **paused since 2026-10-31** (user call: the GPUs go to the decode experiments in item 3) with `qwen36-ream` at 117/300; the queue resumes from that row (`qwen35-moe`, `coder-30b-eval`, `coder-reap-25b`, `coder-30b-ream`, `devstral`, `gemma4`, `nemotron3-omni` behind it, about a day each) via `swebench-bakeoff.service`. Closed so far: `qwen38` 196, `qwen36-dense` 191, `flint` 182 (`omp` 186), `qwen36` 176 of 300 — table below. How the campaign got here (the 2026-09-20 isolation restart, the DSpark `-v4` restart, capture-at-wall, the harness defects each boundary fixed, the 2026-10-22 pruning to opencode): [`docs/swebench-bakeoff.md`](docs/swebench-bakeoff.md#campaign-history).

**Next steps, in order:**

1. **Resume and finish the re-roll queue (after the experiments).** Each lane closes on `audit_leakage.py --require-isolation` + the recall audit, then its row fills in. Run the leak audit the hour a lane closes: the Phase-4.5 gate has misfired on a new false-positive shape at each of the last three closes (fixed each time — [`test_audit_leakage_segments.py`](scripts/eval/test_audit_leakage_segments.py)), and a misfire leaves the cycle unscored until a PAUSE window. The `devstral` cell doubles as the post-patch-058 observational cell against its pre-058 receipt ([`devstral-058-baseline`](benchmarks/quality/devstral-058-baseline-2026-07-19.txt)). `nemotron3-omni` last.
2. **Flint follow-ups** ([bring-up notes](docs/flint.md)): (a) re-roll `flint:omp` under the fixed tool grammar — the landed 186/300 ran with the strict key-ordered grammar (7.4 % of bash calls swapped, [receipt](benchmarks/quality/flint/tool-grammar-key-order-2026-10-28.md)) and is a floor; (b) levers still untried on it: `--mem-fraction-static` (the full pool is now 4.5× the window), prefill on triton (~1–2.4K tok/s, a 250K fill ≈ 4 min); (c) a calibrated g32 build as a quality A/B — patch 067 serves g32 at g64 speed and data-free g32 already halves the g64 RTN KL; (d) a natural-haystack variant of the 256K tool probe — the stock filler is a repetition attractor on Flint, so its tool-use row is a floor.
3. **Decode levers — ACTIVE, ranked by exposed step time.** Landed 2026-10-31: `fp8_e5m2` KV on `flint` (+11 % @262K, +5 % @131K, −4 % @1K, quality parity; [receipt](benchmarks/flint/kv-dtype-ab-2026-10-31.md)). Next up: 3.1 (the serving-only window is open). Ranked list (qwen38 agentic profile: 57 % INT4 weight streaming, 12 % NCCL, 10 % fp16 `lm_head`, 7 % attention — [ledger](docs/decode-levers.md)), each gated on needle + HumanEval + the tool probe:
   1. **Spec-decode housekeeping (serving-only window):** the `qwen38` DSpark arm cannot be baselined by the random-prompt tripwire — 2026-10-31 it read 183 / 116 / 230 tok/s at 1K / 32K / 262K with accept length 4.1 (random prompts make the output degenerate and the draft near-perfect; the deep request also hit the warmup's prefix cache), so the 2026-09-27 natural-text receipt stays the spec arm's reference and a natural-text prompt source for `bench_long_context.py` is the prerequisite for a spec tripwire; the served spec config's gate ran the same day — needle 12/12 to 250K, tool probe 1.0 to 255,880 true tokens, HumanEval 36/40 under the evaluator's new per-candidate timeout. Validate the other opt-ins at 256K / fp8 KV before any joins `SPEC_FOR` (`qwen36` DFlash still carries v0.5.1x flags and a 32K window; EAGLE3 on `coder-30b-eval` / `devstral` is validated at 16K only; NGRAM covers the REAP/REAM variants).
   2. **`lm_head` INT8 through Marlin — LANDED 2026-10-31 (patch 069, default ON).** The loaded bf16 head is quantized per output channel (symmetric int8) and repacked into Marlin W8A16 at load; `qwen38` **+5.6 / +5.2 / +4.1 % at 1K / 32K / 262K** at quality parity (HumanEval 36/40 on both heads, needle 12/12, tool probe 1.0 to 256K); tied heads (Gemma 4) fall back. Fleet tripwire + per-preset quality under the default, and the INT4 head (−7 % per step, needs a quality check per family): [receipt](benchmarks/quality/lm-head-int8-2026-10-31.md).
   3. **PCIe host-read decisive tests** — 3–4 GB/s of GPU-initiated host reads during decode for no known purpose: 60 s each under `dmon` with `--disable-cuda-graph` and `NCCL_P2P_LEVEL` / `NCCL_WORK_FIFO_DEPTH` variants.
   4. Smaller: `in_proj_ba` fp16 gemv → real kernel (−2 %), P0 memory clock via `-lmc` (~+1.5 %), W4A8 Marlin for prefill (≤−4 % e2e, activation-quant quality gate), prefill-only NCCL `Simple` protocol via a tuner plugin (−0.5 % e2e). Custom allreduce (−6 %) stays blocked on sm_86 graph capture (re-test `ENABLE_CUSTOM_AR=1` at each rebase).
   5. **Thermal headroom.** The ship profile is 300 W cap + CPU boost off + fan floor 100 % ([sweep](benchmarks/power-sweep-2026-10-24/), ledger entry). If a longer lane closes the 5 °C DIMM margin: a `scaling_max_freq` CPU cap traded for 330 W (+4.7 % decode, −17 % 128K TTFT), a fan across the DIMMs, a check of the CPU cooler and the three Commander channels reading 0 rpm. Tripwire baselines re-saved on this profile 2026-10-31 (all 9 presets, `benchmarks/baselines.json`).
4. **Harness lever — stop building per-instance rollout images.** The GPUs are busy 73 % of an opencode lane's wall clock; the per-instance build is 176 s median (14–16 % of `rollout_seconds`) and a network dependency per instance that has cost 85 rows to registry / installer outages this campaign ([receipt](benchmarks/quality/gpu-utilization-harness-overhead-2026-09-26.md); R9700's hub image + bind-mounted toolchain measures 97 % busy with a 5 s scaffold boot). Bind-mount a host-built scaffold stack read-only into the official `swebench/sweb.eval.x86_64.<iid>` image — a harness-version change recorded per cell, after the queue.
5. **Tooling for the MoE backlog:** port the Samsung SAIL REAM merge to the Gemma 4 arch (40–60 h; plan [`scripts/quantize/ream_gemma4_port_plan.md`](scripts/quantize/ream_gemma4_port_plan.md)); extend `run_reap.py` to the Gemma 4 parallel dense+MoE and Nemotron-H Mamba2-hybrid layouts (the fused-`Qwen3_5Moe` unfuse is done, 7/7).
6. **Calibration backlog (calibration device, not this box):** `North-Mini-Code-1.0-AWQ` → `Qwen3.6-35B-A3B-REAP-AWQ` → `gemma-4-26B-A4B-REAM-AWQ` → the `Qwen3.6-VL-30B-A3B` native/REAM/REAP trio with vision retained → `Qwen3-30B-Instruct-2507` native + REAP → `Qwen3.5-28B-A3B` native + REAM → Nemotron REAP/REAM; plus a Marlin-friendly `gemma4` requant (the sole fix for the Gemma fallback-kernel decode). Gated further out: a `Qwen3.8-Flash-Next` REAP/REAM shrink (blocked until sglang ships the QSA model class), the `diffusiongemma-26B-A4B` port, the official Gemma QAT-W4A16 compare. Recipes and rationale: [`docs/roadmap.md`](docs/roadmap.md).
7. **User-gated:** schedule the sudo-domain durable fix for the ~9–17 h docker-I/O kernel BUG (every multi-day cycle eats a crash).

### Known issues (open)

- **`nemotron3-omni` decode −12 % at depth since v0.5.18 on the stock 260 W profile** (93.3 → 82.4 tok/s @261,916; v0.5.20 85.9, v0.5.22 85.2). Bisected to the flashinfer-side decode dispatch (0.6.15.post1 → 0.6.17); preset unchanged (still the fastest path). The 300 W + gpu-unlock profile brings the deep point back to **93.1** (2026-10-31 tripwire baseline), so the gap is masked by the profile, not fixed — a future flashinfer bump is where it would show again; receipts `benchmarks/regression/exp-nemotron-*.json`.
- **Host reboots every ~9–17 h under sustained docker rollout I/O (kernel BUG).** Predictions on disk survive; `swebench-bakeoff.service` auto-resumes. Durable fix is user-gated (item 7). Forensic recipe: [`CLAUDE.md`](CLAUDE.md) → Operational Lessons.
- **Flint retrieval is text-dependent from ≈214K actual tokens at depths ≤ 0.5** (natural haystack; identical on bf16 and `fp8_e5m2` KV, reproduced by R9700 on FP8): 240.8K / 246.5K / 252.4K greedy ×2–3 — depth 0.9 hits 14/14 per arm, depths 0.1 + 0.5 hit 2/14 (e5m2) and 1/12 (bf16) with the misses confident wrong answers; 131K is 8/8. Bracket: 155K and 195K actual 4/4 at depths 0.1 / 0.5, 214K 0/4 (both depths return the same haystack sentence — a text-triggered continuation). The served window is 262,144 — treat its last ~20 % as text-dependent for mid-context retrieval ([receipt](benchmarks/flint/kv-dtype-ab-2026-10-31.md)).
- **Qwen3.6 greedy-needle knife-edge (`qwen36` 131,072@0.5, `qwen36-vl-reap` 1,024@0.1 / 16,384@0.5) — a checkpoint repetition attractor, not the stack.** At T=0 the think starts quoting the haystack filler and never stops (`finish=length`, empty content); reproduced on the retained v0.5.20 env, the neighbouring cells answer 36/36 from a flushed cache, a 4,096 budget changes nothing, and under the checkpoint's own sampling the loop moves to a different cell. Watch item: if a fleet eval drops `qwen36` below 14/15, bisect the full-attention path with the flushed-band recipe in [`patches/v0.5.22-rebase-status.md`](patches/v0.5.22-rebase-status.md); `needle_band_probe.py --temperature default` separates a greedy knife-edge from a sampled miss.

## Model Support

**Max ctx** = what the AWQ ship + 2× 24 GB actually serves end-to-end (validator + bake-off receipts). Every preset defaults to the full ctx; single-user tok/s = the 2026-10-25 fleet refresh (300 W cap, gpu-unlock on, CPU boost off; the 10-point depth sweep in `benchmarks/<model>/results.json`, 400 W companion in `results-400w.json`): short context (32K, **deepest point**).

| Model | Type | Max ctx | tok/s | Launch | HF + notes |
|-------|------|:-------:|:----:|:------:|:-------|
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE A3B (256 exp, VL) | **262K** | 223 (197 @32K, **129 @255K**) | `qwen36` | [`mattbucci/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ). |
| **Qwen3.6-REAM-A3B AWQ** | DeltaNet+MoE A3B (192 exp, VL) | **262K** | 224 (207 @32K, **157 @255K**) | `qwen36-ream` | [`mattbucci/Qwen3.6-REAM-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ). Vision tower grafted. |
| **Qwen3-30B-Instruct-2507 REAM AWQ** | MoE A3B (96 exp) | **262K** | 207 (163 @32K, **73 @255K**) | `qwen3-ream` | [`mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ). REAM 128→96; text-only generalist. |
| **Qwen3.5-28B MoE REAP** | DeltaNet+MoE A3B (205 exp, VL) | **262K** | 222 (205 @32K, **156 @255K**) | `qwen35-moe` | Cerebras REAP of Qwen3.5-28B-A3B; thinking+vision. |
| **Nemotron-3-Nano-Omni-30B-A3B AWQ** | Mamba2-hybrid MoE A3B (128 exp, AVLM) | **262K** ✓ (5.25M pool) | 101 (99 @32K, **94 @255K**) | `nemotron3-omni` | [`mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ). int4; **6/6 caps** (basic+thinking+tool+vision+**video**+**audio** — the only audio ship). Serves via **`QUANT=moe_wna16`** (baked into the preset; the auto-detected awq_marlin path fails on this model's TP=2 shard shapes — mechanism in `patches/v0.5.14-rebase-status.md`), enabled by **patches 052** (non-gated squared-ReLU moe_wna16) + **053** (EVS video routing). Mamba2 O(1) recurrent → decode ~flat (101→93 @255K); beats R9700 FP8 (74.79/49.22). ⚠ **Agentic depth 131K (needs ≥8K token budget; 76K at 2K)**: ≥196K it spirals past even 8K and at 253K stops calling entirely — flat decode does not buy agentic 256K here. |
| **Qwen3-Coder-30B-A3B AWQ** | MoE A3B (128 exp) | **262K** | 210 (165 @32K, **74 @255K**) | `coder-30b` or `coder-30b-eval` | [`mattbucci/Qwen3-Coder-30B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ). Two presets serve the same model; for short-ctx batch-decode benchmarks override `CTX=16384 MAX_RUNNING=32 ./scripts/launch.sh coder-30b`. |
| Coder-REAP-30B AWQ-Marlin | MoE A3B (96 exp) | **262K** | 207 (164 @32K, **74 @255K**) | `coder-reap-25b` | [`mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) (R9700 in-house). |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **262K** ✓ (347K pool) | 58 (52 @32K, **32 @255K**) | `gemma4-31b` | [`mattbucci/gemma-4-31B-AWQ`](https://huggingface.co/mattbucci/gemma-4-31B-AWQ). LM INT4, vision tower FP16, **KV fp8_e5m2**. Tool-use 1.0 → 258K true tokens, 5/5 caps incl. video. 347K pool via `--swa-full-tokens-ratio 0.05` + `MEM 0.92` + e5m2 FP8 KV (the only FP8 that compiles on the triton-forced path — sm_86 rejects e4m3). |
| **Gemma 4 26B MoE AWQ** | MoE A4B (103 exp, VL) | **262K** ✓ | 84 (80 @32K, **61 @255K**) | `gemma4` | [`mattbucci/gemma-4-26B-AWQ`](https://huggingface.co/mattbucci/gemma-4-26B-AWQ). **Tool-use 1.0 → 258K true tokens, 5/5 caps** incl. video. 652K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Gemma 4 12B Unified AWQ** | Omni, encoder-free | **262K** ✓ | 108 (99 @32K, **62 @255K**) | `gemma4-12b` | [`mattbucci/gemma-4-12B-AWQ`](https://huggingface.co/mattbucci/gemma-4-12B-AWQ). In-house int4 RTN-from-QAT. MMLU 77 / HE 93 / **tool-use 1.0 → 258K true tokens**, **5/5 omni** (vision + video ✓; gemma4_unified is native since transformers 5.12.1). 565K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Qwen3.6-27B Dense AWQ** | Dense + DeltaNet (VL) | **262K** (657K KV) | 73 (69 @32K, **52 @255K**) | `qwen36-dense` | [`mattbucci/Qwen3.6-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (R9700 self-cal). |
| **Qwen3.8-27B AWQ** | Dense + DeltaNet (VL **+video**) | **262K** (652K KV @ M=1; 307K under the DSpark draft) | 73 (69 @32K, **52 @255K**) · `SPEC_DECODE=1` **125 / 90 @49K / 75 @130K / 58 @250K** | `qwen38` | [`mattbucci/Qwen3.8-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ) (3090 self-cal 2026-08-18, GPTQ `thinking_vision_video`). MMLU **0.93** · HE **0.96** · needle **1.0** at server-verified 250,077 actual · caps **5/5** incl. video (LAB 0.16 — inside the DeltaNet-family spread 0.11–0.32 on a 56-question probe; watch item). 48 GDN + 16 full-attn layers; ships MTP weights (inert — main loader skips them). ⚠ `--max-running 1` is load-bearing: at 8 the KV pool collapses 652K→32K (18.7 GB ship: untied 248,320 vocab + BF16 lm_head, and DeltaNet state replicates per slot). |
| **Flint 40B-A2.5B AWQ** | MoE A2.5B (256 experts top-6 + shared expert; 35 SWA-513 + 14 **NoPE** full-attention layers, learned sinks; thinking + tools, text-only) | **262K** ✓ (1.18M full / 59K SWA pool) | 159 (148 @32K, **104 @255K**) | `flint` | [`mattbucci/Flint-AWQ`](https://huggingface.co/mattbucci/Flint-AWQ) (3090 calibrated AWQ g64, 2026-10-22 — AWQ scale search + GPTQ on activation Grams from the model's own reference forward, [`quantize_flint_awq_calib.py`](scripts/quantize/quantize_flint_awq_calib.py); the data-free RTN build it replaced looped in agentic turns, [receipt](benchmarks/quality/flint/runaway-repetition-replay-2026-10-22.md)). Tool calls need `FLINT_STRICT_TOOLS=1` (preset default: the model ignores JSON-schema `required` keys without schema-enforced grammar, [receipt](benchmarks/quality/flint/tool-schema-probe-2026-10-22.json)). Needle: natural-text haystack **found** greedy at 131K and 243.5K, both depths — the stock repeated-sentence filler is a NoPE attractor and a known-bad cell for this model ([receipt](benchmarks/quality/flint/needle-haystack-discriminator-2026-10-22.md)); **TP=2 validated against TP=1** ([receipt](benchmarks/quality/flint/tp-check-2026-10-22.md)). Triton attention only on Ampere (sinks + `SWAKVPool`), **`fp8_e5m2` KV** (since 2026-10-31: +11 % @262K over bf16 KV at quality parity, −4 % @1K — [receipt](benchmarks/flint/kv-dtype-ab-2026-10-31.md)), MoE runner `auto` (Marlin). Model code: `plugins/flint_sglang` (vendor `flint_sglang` 0.1.4 ported to v0.5.22). |
| **Devstral-Small-2-24B AWQ** | Dense (VL) | **262K** ✓ (339K pool, patch 062) | 92 (79 @32K, **41 @255K**) | `devstral` | [`mattbucci/Devstral-Small-2-24B-AWQ`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ). The canonical Devstral; built from [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512). FP8→BF16→GPTQ+tool-cal→AWQ. KV 202K → **339K on v0.5.18+062** (the loader-garbage fix recovered ~5.3 GB/rank): true 256K with headroom. |
| **Qwen3-VL-32B Instruct AWQ** | Dense (VL) | **131K** (model-card cap) | 66 (55 @32K, **38 @127K**) | `qwen3-vl-32b` | [`mattbucci/Qwen3-VL-32B-AWQ`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) (R9700). 63→45→35 tok/s @ 1K/64K/127K. |
| Gemma 4 21B REAP AWQ | MoE (VL) | **262K** ✓ (653K pool) | 85 (81 @32K, **62 @255K**) | `gemma4-21b-reap` | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Cerebras-style expert prune of the 26B parent; same Gemma 4 serving flags (graphs ON + `--swa-full-tokens-ratio 0.0625` → 652K pool); tool-use 1.0/1.0 on the standard ladder. ⚠ HumanEval 0% (REAP prune lost coding — Quality Evals below): vision/chat ship, not code. |

Per-preset receipts for the current stack: `benchmarks/quality/*-v0522.json` (flip smoke) + `cap-*-v0522.json` (capability matrix); flip table in [`patches/v0.5.22-rebase-status.md`](patches/v0.5.22-rebase-status.md) (previous stack: [`v0.5.20-rebase-status.md`](patches/v0.5.20-rebase-status.md)).

### HuggingFace model zoo

Every `mattbucci/*-AWQ` row is built end-to-end from the linked upstream BF16 tensor — calibration, CT export, native AWQ conversion, scales audit, ship. **No 3rd-party pre-quantized AWQ used as a base.** ⚠ rows were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule; they're grandfathered live until in-house rebuilds replace them (rebuild paths in the [MoE coverage matrix](#moe-coverage-matrix)).

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. No descriptive suffixes (`-thinking-vision`, `-4bit`, `-native`, `-v2-fixed`) — the model card carries detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix.

| Ship | HuggingFace | Upstream base |
|------|-------------|---------------|
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native AWQ-Marlin) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| Qwen3.6-27B Dense AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (R9700 self-cal) |
| Qwen3.8-27B AWQ | [mattbucci/Qwen3.8-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (3090 self-cal, `thinking_vision_video` recipe) |
| Flint AWQ | [mattbucci/Flint-AWQ](https://huggingface.co/mattbucci/Flint-AWQ) | [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (3090 calibrated AWQ g64, integer zeros — `scripts/quantize/quantize_flint_awq_calib.py`, a reference-forward calibrator: AWQ scale search + GPTQ on activation Grams, expert-Gram shrinkage for the routed experts; the vendor MLX int4 has a free per-group bias and is not AWQ-representable; `quantize_flint_rtn_awq.py` is the data-free fallback) |
| Qwen3-30B-Instruct-2507 REAM AWQ | [mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ) | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (Samsung SAIL `merge.py`, 128→96 experts) |
| Qwen3-Coder-30B-A3B AWQ | [mattbucci/Qwen3-Coder-30B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-REAM AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (Samsung SAIL `merge.py`, 128→96 experts) |
| Qwen3-Coder-30B-A3B-REAP AWQ | [mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (in-house `scripts/quantize/run_reap.py`, 128→96 experts) |
| Qwen3-Coder-Next-REAM AWQ | [mattbucci/Qwen3-Coder-Next-REAM-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-Next-REAM-AWQ) | [Qwen/Qwen3-Coder-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) (Samsung SAIL `merge.py`, 512→384 experts, ~60B effective; doesn't fit at AWQ on 24 GB cards — for R9700 / bigger-card use) |
| Qwen3-VL-32B Dense AWQ | [mattbucci/Qwen3-VL-32B-AWQ](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) (R9700 self-cal, `balanced_thinking_vision` recipe) |
| Devstral-Small-2-24B AWQ ★ canonical Devstral | [mattbucci/Devstral-Small-2-24B-AWQ](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ) | [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) (FP8→BF16→GPTQ+tool-cal→AWQ; `code_vision_tools` recipe) |
| Nemotron-3-Nano-Omni-30B-A3B AWQ | [mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ) | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning) (in-house; audio + video preserved) |
| Gemma 4 26B A4B MoE AWQ | [mattbucci/gemma-4-26B-AWQ](https://huggingface.co/mattbucci/gemma-4-26B-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) |
| Gemma 4 31B Dense AWQ | [mattbucci/gemma-4-31B-AWQ](https://huggingface.co/mattbucci/gemma-4-31B-AWQ) (in-house BF16→GPTQ→AWQ, vision tower FP16) | [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) |
| Gemma 4 12B Unified AWQ | [mattbucci/gemma-4-12B-AWQ](https://huggingface.co/mattbucci/gemma-4-12B-AWQ) (in-house data-free RTN-from-QAT, full omni) | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (via QAT base `gemma-4-12B-it-qat-q4_0-unquantized`) |
| Gemma 4 21B REAP AWQ | [mattbucci/gemma-4-21B-REAP-AWQ](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ) | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) (smaller Cerebras-style REAP variant of the 26B parent; in-house regex-`ignore` calibration) |
| Qwen3.5-27B Dense AWQ | [mattbucci/Qwen3.5-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.5-27B-AWQ) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (R9700 self-cal) |
| ⚠ Qwen3-Coder-REAP-25B-A3B AWQ (3rd-party-base, legacy) | [mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ) | **Upstream:** [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B). Superseded by `Qwen3-Coder-30B-A3B-REAP-AWQ` (in-house) — kept live for backward compat. |
| ⚠ Qwen3.6-VL-REAP-26B-A3B AWQ (3rd-party-base, vision broken) | [mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ) | **Upstream:** Qwen/Qwen3.6-VL-30B-A3B-Instruct. **Shipped from 3rd-party pre-pruned BF16:** [atbender/Qwen3.6-VL-REAP-26B-A3B](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B) — vision tensors dropped at the pre-prune layer → no working vision. Rebuild queued (calibration backlog). |
| ⚠ Qwen3.5-28B-A3B-REAP AWQ (3rd-party-base) | [mattbucci/Qwen3.5-28B-A3B-REAP-AWQ](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) | **Upstream:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). **Shipped from 3rd-party pre-pruned BF16:** [cerebras/Qwen3.5-28B-A3B-REAP](https://huggingface.co/cerebras/Qwen3.5-28B-A3B-REAP) (vision tensors retained at pre-prune, so vision works). Rebuild queued via in-house REAP. |

### MoE coverage matrix

Each MoE base should ship in three flavors: **native** (no expert compression), **REAP** (Cerebras-style pruning, in-house via `scripts/quantize/run_reap.py`), **REAM** (Samsung SAIL merging, in-house via `scripts/quantize/run_ream_qwen3moe.sh`). All entries are self-calibrated AWQ-int4 from the upstream BF16 base. The missing cells are the calibration backlog (Next steps item 6; recipes in [`docs/roadmap.md`](docs/roadmap.md)).

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

### VRAM context limits

TP=2, 48 GB total. **"Max context" is the real KV-pool capacity** (`max_total_num_tokens` from the serve log), not the declared `--context-length`. The heavy-VL ships are KV-bound; the SWA-hybrid Gemmas recover it with `--swa-full-tokens-ratio` right-sizing (sliding layers attend only `window=1024`) and, on the 31B, e5m2 FP8 KV. Devstral's full-attention GQA-8 layout (~40 KB/token) is ~3–4× heavier than gemma-4-31B's MQA-4 SWA hybrid (~12 KB), which is why the smaller model caps lower. Detail: [`docs/decode-levers.md`](docs/decode-levers.md#kv-pools-and-context-limits).

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B-Instruct-2507 REAM AWQ | 6.2 GB | 36 KB | 262K (578K pool) |
| Qwen3.5-28B-A3B REAP AWQ | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-Marlin | 9.87 GB | ~8 KB hybrid | 262K (996K pool) |
| Qwen3.6-REAM-A3B AWQ | 7.4 GB | ~8 KB hybrid | 262K (2.4M pool) |
| Qwen3-Coder-30B-A3B AWQ | 8.0 GB | 36 KB | 262K (~900K pool) |
| Qwen3-Coder-30B-A3B-REAP AWQ | 6.5 GB | 72 KB | 262K |
| Qwen3.6-27B Dense AWQ | 8.8 GB (measured; sharded) | 24 KB | 262K (657K pool) |
| Qwen3.8-27B AWQ | ~9.4 GB (18.7 GB ship) | 24 KB | 262K (652K pool @ M=1) |
| Flint 40B-A2.5B AWQ | 10.6 GB | 14 KB fp8_e5m2 (14 full-attn layers) + 35 KB over a 513-token window (35 SWA layers) | 262K (1.18M full / 59K SWA pool, MEM 0.85) |
| Devstral-Small-2-24B AWQ | 7.0 GB | ~40 KB (fp8 KV) | **339K** (MEM 0.90, patch 062) |
| Gemma 4 26B A4B MoE AWQ | 6.5 GB | ~12 KB (SWA) | **652K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 21B REAP AWQ | ~5 GB | ~12 KB (SWA) | **653K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 31B Dense AWQ | 7.7 GB | ~12 KB (SWA, fp8_e5m2) | **347K full / 17K swa** (`--swa-full-tokens-ratio 0.05` + MEM 0.92 + e5m2 KV) |
| Gemma 4 12B Unified AWQ | 5.4 GB | 15.3 KB full + 152.6 KB swa | **565K full / 35K swa** (262K ✓ @ ratio 0.0625) |
| Qwen3-VL-32B Dense AWQ | 10.0 GB | 24 KB | 131K (model-card cap) |

## Coding-eval bake-off (SWE-bench Lite)

Full 300-instance SWE-bench Lite, isolated Docker harness, one attempt per instance, single-user, `swebench==4.1.0` scorer. **Only full-300 cells are compared**, and every counted cell ran at the served 262K window with template-default thinking and a 32K output budget, with no network (a loopback bridge to the server is the only address), the task on stdin, `/testbed` re-initialised to one neutral commit, and closed on `audit_leakage.py --require-isolation` = 0 exposed on 300/300 ([contract](benchmarks/quality/swebench-harness-isolation-2026-09-20.md)). **Harness: opencode (opencode-ai 1.14.25), one lane per preset since 2026-10-22**, picked from the `qwen38` five-scaffold bake-off below (a statistical tie across the opencode / pi families); the DCP, RTK, pi, prime, dcode and omp lanes are receipts only (claw-code retired 2026-08-31). Mechanics, A/B-lane engagement receipts, exclusions and the campaign history: [`docs/swebench-bakeoff.md`](docs/swebench-bakeoff.md).

| Preset | opencode (resolved/300) |
|--------|:--------:|
| `qwen38` (Qwen3.8-27B AWQ, thinking) | **196/300 (65.3 %)** |
| `qwen36-dense` (Qwen3.6-27B Dense AWQ, thinking) | **191/300 (63.7 %)** |
| `flint` ([Flint](https://huggingface.co/RecursiveSI/flint) 40B-A2.5B, calibrated AWQ, thinking) | **182/300 (60.7 %)** · `omp` **186/300 (62.0 %)** (McNemar tie, p = 0.57; opencode 4 empty / 9 walls, omp 5 empty / 3 walls, 0 errors; the omp cell ran under the strict-ordered tool grammar — a floor, see the Flint item) |
| `qwen36` (Qwen3.6-35B-A3B AWQ-Marlin, thinking) | **176/300 (58.7 %)** (0 empty / 0 walls / 0 errors, median 299 s; the recall audit finds 0/300 sessions mentioning the benchmark) |
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | *pending* |
| `qwen35-moe` (Qwen3.5-28B-A3B-REAP-AWQ, thinking) | *pending* |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | *pending* |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | *pending* |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | *pending* |
| `devstral` (Devstral-Small-2-24B-AWQ) | *pending* |
| `gemma4` (Gemma 4 26B MoE AWQ, thinking) | *pending* |
| `nemotron3-omni` (Nemotron-3-Nano-Omni AWQ, Mamba2-hybrid) | *pending* |

**Harness bake-off — `qwen38`, five scaffolds on the same 300 isolated instances (2026-10-22, [receipt](benchmarks/quality/qwen38-v4-cycle-close-2026-10-22.md)):**

| Scaffold | resolved | % | walls | empty | median rollout | vs opencode (McNemar) |
|---|:--:|:--:|:--:|:--:|:--:|:--|
| opencode 1.14.25 | **196** | 65.3 | 13 | 6 | 804 s | — |
| opencode + DCP 3.1.15 | **203** | 67.7 | 14 | 10 | 923 s | 25 / 18 discordant, p = 0.36 |
| little-coder 1.19.0 (pi 0.83) | **193** | 64.3 | 25 | 5 | 749 s | 16 / 19, p = 0.74 |
| little-coder + RTK 0.46 | **202** | 67.3 | 15 | 4 | 734 s | 22 / 16, p = 0.42 |
| prime-agent 0.8.1 | **187** | 62.3 | 30 | 37 | 912 s | 13 / 22, p = 0.18 |
| dcode 0.1.65 | *quarantined* | | | 213 | | server-side Responses-API defect (patch 066) + no output cap on the wire (R9700 `b79a392`, ported 2026-10-26) — lower bound only |

Only prime separates from the two plugin lanes (p = 0.023 / 0.032); the other four cells are inside one standard error of each other. The 37 prime empties are its single-strike `length` policy on 32K-budget recall thinks, not the model.

Rows are in queue order; *pending* = queued, rolling, or rolled-and-unscored — it fills in with `resolved/300` at the cycle's close. Pre-isolation cells (`--network=host`, scaffold-default context / output budgets) are kept only as receipts — `benchmarks/quality/bakeoff-*-{ctx32k,ctx131k,opencode-out8k,claw-code}.json` — and every read made from them is withdrawn until the re-rolled cell lands. `qwen3-ream` is excluded on model grounds (can't sustain agentic sessions — [verdict](benchmarks/quality/bakeoff-qwen3-ream-verdict-2026-07-17.md)). Per-cell receipts: [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/); failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling): [`patches/README.md`](patches/README.md).

## Quality evals

Every shipped AWQ model is scale-integrity clean (fleet `check_awq_scales.py --base` audit, [receipt](benchmarks/quality/fleet-integrity-audit-2026-05-31.json)) and passes every applicable capability probe (thinking / image / video / audio / tool) on the current stack (`cap-*-v0522.json`). Methodology, footnotes, and per-probe findings: [`docs/quality-evals.md`](docs/quality-evals.md).

**Static evals** (`scripts/eval/eval_quality.py`: MMLU, HumanEval pass@1 no-think chat, [LAB-Bench](https://github.com/Future-House/LAB-Bench), Needle 1K → 250K; ±a few points is noise):

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
| Devstral-Small-2-24B AWQ | 77.2% | 80.0% | 33.6% | ✓131K (pool cap at measurement) | `devstral.json` |
| Gemma 4 31B Dense AWQ | 93.0% | **97.5%** | **42.9%** | **✓250K** | `gemma4-31b.json` |
| Gemma 4 26B MoE AWQ | 82.5% | **97.5%** | 36.4% | **✓250K** | `gemma4.json` |
| Gemma 4 12B Unified AWQ | 77.2% | 92.5% | 29.3% | **✓250K** | `gemma4-12b.json` |
| Flint 40B-A2.5B AWQ | 68.4% ◆ | 70.0% | — | **✓ 195K** natural ◆◆ | `flint.json` |

† REAP prune lost coding (use `gemma4-31b` for code). ◆ 57-question chat-mode MMLU under a 200-token cap on a thinking model — a same-instrument comparison, not a score. ◆◆ natural Open-Platypus haystack (the stock filler is a repetition attractor on Flint's NoPE layers); retrieval is reliable to 195K and text-dependent from ≈214K at depth ≤ 0.5 on either KV dtype (Known issues). ‡ partial 333-question LAB subset. ◊ non-coder text generalist on a code task.

**256K tool-use probe** (`scripts/eval/probe_256k_tooluse.py`) — plants a needle deep in filler and measures whether the model emits a valid, correctly-argumented tool call with the planted value, bucketed by TRUE `prompt_tokens`. The agentic 256K signal SWE-bench Lite (tops ~128K) never reaches:

| Preset | valid tool call | correct args | max TRUE tokens still correct |
|---|:---:|:---:|:---:|
| `qwen36` (MoE-thinking) | **1.0** | **1.0** | **255,889** |
| `qwen36-ream` (MoE-thinking) | **1.0** | **1.0** | **255,889** |
| `qwen36-dense` (dense-thinking) | **1.0** | **1.0** | **258K** |
| `gemma4` (26B MoE) | **1.0** | **1.0** | **255,957** |
| `gemma4-12b` / `21b-reap` / `31b` | **1.0** | **1.0** | **258K** |
| `qwen35-moe` (DeltaNet MoE) | **1.0** | **1.0** | **255,889** |
| `qwen3-ream` (text generalist) | **1.0** | **1.0** | 148K single-turn (multi-turn agentic fails — bake-off-excluded) |
| `coder-30b-eval` / `coder-reap-25b` | 0.4 | 0.4 | **~64K agentic ceiling** — prose-stop instead of calls at ≥131K true |
| `devstral` (dense, tool) | **1.0** | **1.0** | **132K** firm; 178K flaky (measured at the pre-062 202K pool) |
| `nemotron3-omni` (Mamba2 AVLM) | 0.6 | 0.6 | **131K @8K budget** (76K @2K); ≥196K spirals past 8K, 253K prose-stops |
| `flint` (MoE, SWA + NoPE) | 0.4 | 0.4 | **~65K on the stock filler** — length-stops from 131K on the filler attractor, identical on bf16 and e5m2 KV; a natural-haystack tool probe is owed before this row is a capability read |

The flagship MoE thinkers and gemma4 hold perfect tool-calling at TRUE 256K, at every needle depth (0.1 / 0.5 / 0.9 — no lost-in-the-middle). The Coder-30B family does not: route agentic work beyond ~64K to the qwen36 family.

**256K reasoning probe** (`scripts/eval/probe_256k_quality.py`) — multi-key retrieval + variable-tracking + aggregation over a full context, identical task instances at every length (temp=0), measured at TRUE actual-token lengths (top point 255,800):

| Preset | 1K | 32K | 65K | 131K | 200K | 256K | overall |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemma4` (26B MoE) | 100 | 67§ | 100 | 100 | 100 | **100** | 94 |
| `gemma4-31b` (dense) | 100 | 100 | 100 | 100 | 100 | **100** | **100** |
| `gemma4-21b-reap` (MoE) | 100 | 100 | 100 | 100 | 100 | 67§ | 94 |
| `gemma4-12b` (unified omni, int4 RTN) | 100 | 67◊ | 67◊ | 33◊ | 33◊ | 67◊ | 61 |
| `qwen36` / `qwen36-dense` / `qwen36-ream` | 100 | 100 | 100 | 100 | 100 | **100** | **100** |

§ isolated single-cell misses, non-monotonic (noise, not a depth cliff). ◊ `gemma4-12b` chain-reasons to 256K but its multi-needle retrieval saturates at ~3 needles at depth. `qwen35-moe` is genuinely soft at depth (0.72 overall — `multikey` fails ≥131K); at 178K `qwen3-ream` 100%, `devstral` ~80%.

**Every new AWQ ship must pass `scripts/eval/validate_capabilities.py`** (basic + thinking + image + video + audio + tool, per applicable modality) before entering these tables.

## Quick start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.22, apply patches/, create the conda env

./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE AWQ-Marlin — 256K, thinking+vision   (eval port :23334)
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — 256K, thinking+vision
./scripts/launch.sh qwen38                  # Qwen3.8-27B AWQ — thinking+image+video
./scripts/launch.sh coder-30b               # Qwen3-Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AWQ (thinking+image+video)
./scripts/launch.sh devstral                # Devstral-Small-2-24B AWQ (tool+vision)
# full preset list: grep -E "^        [a-z][a-zA-Z0-9-]*[\|\)]" scripts/launch.sh

./scripts/serve_production.sh gemma4-31b    # persistent PRODUCTION endpoint on :30000 (start/stop/status/restart, detached, health-checked)
EXTRA='--decode-topk-pages 256 --decode-topk-page-size 64' ./scripts/serve_production.sh gemma4-31b   # opt-in sparse-KV topk (patch 059); superseded at depth by the shipped kv-splits 64

python scripts/eval/validate_capabilities.py --port 23334    # auto-skips thinking/vision/video per preset
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
BASELINE=save scripts/bench/bench_regression.sh <preset>     # lock a new perf level into the tripwire
```

Production on :30000 and the eval harness on :23334 don't collide, so a capability/bench sweep can run against :23334 while :30000 serves (the script refuses to start if another sglang server already holds the GPUs). Every preset carries an explicit `--tool-call-parser` matching its chat template (Qwen3-Coder + Qwen3.5/3.6/3.8 → `qwen3_coder`; Qwen3-VL / Qwen3-30B REAM → `qwen25`; Devstral → `mistral`; Gemma 4 → `gemma4`); thinking presets launch with `--sampling-defaults model`. Use `temperature >= 0.3` on the Qwen3 family — greedy decode loops.

## Stack

| Component | Version |
|-----------|---------|
| SGLang | v0.5.22 + 32 local patches (`/data/sglang-rebase-v0522`, env `sglang-v0522`; v0.5.20 tree + env kept for one-revert rollback) |
| PyTorch | 2.14.1 + cu130 |
| CUDA | 13.2 driver (595.71.05) / cu130 wheel |
| transformers | 5.19.0 (ships gemma4_unified natively; routes Mistral ckpts to MistralCommonBackend — countered by patch 057) |
| FlashInfer | 0.7.1 [cu13] (sgl-kernel 0.4.9, xgrammar 0.2.7) |
| compressed-tensors | serving env pin; 0.15.1.dev in the separate `quant` calibration env |

**Patches** — 32 logical units in [`patches/`](patches/), applied idempotently by `setup.sh`: AWQ/CT int4 weight loading, Qwen3.5/3.6/3.8 enablement, Gemma 4 bring-up (26B MoE / 31B dense / 12B unified omni), Nemotron-3-Nano-Omni serving, MoE gelu coverage, kernel precision, sm_86 enablement, serving/agentic robustness. Each rebase is gated by the 3-gate pristine replay (`scripts/test_patch_gates.sh`) and a per-tokenizer-family A/B encode (`scripts/eval/tokenizer_ab_encode.py`), then a detached fleet validation campaign (`scripts/eval/flip_campaign.sh`). Narratives, the upstream-PR ledger, and per-flip receipts: [`patches/README.md`](patches/README.md).

**OCI image** — `Dockerfile` builds the CUDA/v0.5.22 stack without a GPU (pinned wheels, driver injected by the NVIDIA container toolkit); runs unprivileged with `SGLANG_SECURE_LAUNCH=1` (API keys from files, protected server options refused, NCCL on loopback). Build, run, and the security caveats vs the R9700 image: [`docs/oci-image.md`](docs/oci-image.md).

**Quantization** (the `quant` conda env, calibration device):

```bash
conda activate quant
REAP_ENV=quant ./scripts/quantize/run_reap.sh --model <bf16> --save-path <reap_bf16> --keep-experts N  # MoE expert prune
./scripts/quantize/run_ream_qwen3moe.sh <bf16> <ream_bf16>                                    # MoE expert merge (Samsung SAIL)
CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template (GPTQ → CT)
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                          # MoE CT→native AWQ
python scripts/eval/check_awq_scales.py <awq_dst> --base <bf16_base_dir>                      # ship gate: 0 = clean
```

`scripts/quantize/calibration_datasets.py` builds capability-preserving recipes (`thinking_vision_video` / `code_vision_tools` / `balanced_thinking_vision` …) from AM-Thinking-v1, NuminaMath-CoT, LLaVA-Instruct / LLaVA-Video, Hermes-function-calling, UltraChat. `check_awq_scales.py --base` is the dead-channel comparator that separates benign MoE structural-sparsity zero-scales from real defects. REAP vs REAM: [`scripts/quantize/REAM.md`](scripts/quantize/REAM.md).

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 3090 (24 GB each) — NVLink bridge, `nvidia-smi topo -m` reports `NV4` (~56 GB/s aggregate); 300 W power cap + CPU boost off + GPU fan floor 100 % (cooling profile, load-bearing — [2026-10-24 sweep + CPU trial](benchmarks/power-sweep-2026-10-24/); the DDR5 DIMMs sit against the socket on the mini-ITX board and set the ceiling) |
| CPU / RAM | AMD Ryzen 9 7900 (12C/24T) / 64 GB DDR5-6000 |
| Storage | 2× 2 TB NVMe (`/data` = models + caches) |
| OS / Kernel | Arch (EndeavourOS) / `linux-zen-p2p` 6.18 (pinned) · `nvidia-open-dkms` 595.71.05 · `amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction` |

Why each of those is load-bearing (NVLink/P2P boot args, the zen kernel, the cooling units under [`systemd/`](systemd/), Arch toolchain gotchas): [`docs/host-setup.md`](docs/host-setup.md).

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner + RDNA4 serving stack; we own evals + AWQ/INT4 + EAGLE3 draft training. Patches port both ways; model-behavior findings always do. Last relay to them 2026-10-30 (their `CROSS-TEAM.md`): the Flint full-300 lanes with the recall-bucket table, the strict-grammar key-order defect + fix, the three leak-classifier rules from this campaign's lane closes. No R9700 push since 2026-10-08. **Open asks to them:** (0) check whether their sandbox's bridge port / bearer are discoverable the way ours were (`ps` argv + the scaffold's provider file) — our pi-0.83 lane found the model self-dispatching to the served model through the bridge, and their recall audit should count self-oracle calls; (1) the host-side BF16→AWQ trap defuse + double-quant guard port before the next `gemma4-26b` recal (this box is clean — [manifest](benchmarks/models-manifest-letsrtfm-amd-2026-07-19.json)); (2) host the memory-marginal `Qwen3.6-35B-A3B` REAP prune on their 64 GB — the fused-`Qwen3_5Moe` unfuse tooling is ready to port to their `ream-patches/`; (3) on their omp lane, check `--no-title` (the title generator forks the full prefix to the session model on every todo-initialised plan — a hidden second prefill per step).
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal plumbing. No open asks either way.

## Repo layout

```
README.md                 # this file: direction, results, status + next steps, model/quality tables
docs/                     # long-form: decode-levers, swebench-bakeoff, quality-evals, roadmap, host-setup, oci-image
patches/                  # SGLang v0.5.22 patches (32) — narratives + rebase receipts in patches/README.md
benchmarks/               # charts, per-model regression JSON, lever receipts; quality/ = eval + bake-off receipts
evals/swebench/           # SWE-bench Lite v2 Docker harness (cycle driver, scaffolds, scorer, aggregation)
scripts/
  launch.sh / serve_production.sh / common.sh / setup.sh
  bench/ eval/ quantize/ specforge/ upstream-pr/ maint/ host-setup/
docker/                   # OCI image entrypoint + secure launcher
systemd/                  # cooling profile + bake-off auto-resume units
components/sglang/        # legacy vendored SGLang tree (historical; live serving trees are under /data/)
```
