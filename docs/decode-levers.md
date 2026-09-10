# Decode levers — the single-user 256K optimization ledger

The standing mission is single-user decode at 256K (tok/s, TPOT). This file is the ledger: what won, what was measured null, what is blocked and why, plus the serving-side notes (spec decoding, KV pools, thinking defaults) that the README only summarizes. Every entry carries its receipt path under `benchmarks/`.

## Fleet decode picture (server-verified at depth)

**True-256K decoders** (solid right bars): **qwen35-moe 144** and **qwen36-ream 144** lead @255K (210/209 short), then **qwen36 121** (209 short), **nemotron3-omni 93** (101 short — the Mamba2-hybrid's near-flat depth curve is genuine: 23 recurrent layers are O(1), only 6 attention layers scale), the 30B-A3B MoE trio **coder-30b / coder-reap-25b / qwen3-ream ≈ 69** (200 short; the three are M=1 twins — same arch class, same active experts), **qwen36-dense 47** (69 short) and its Qwen3.8 successor **qwen38 48** (71 short, measured @261,916 actual — the two dense DeltaNet 27Bs are M=1 twins, as the shared 48-GDN/16-attn shape predicts). The Gemma hybrids trail hard at depth — **gemma4 / gemma4-21b-reap 24, gemma4-12b 17.5, gemma4-31b 13** — consistent with their whole quantized stack running fallback kernels (group-32 AWQ + TP-hostile shapes fail Marlin; root cause + fix directions in the ledger below). **KV-capped models** (hatched, at their pool/cap): qwen3-vl-32b **35 @127K** (model-card cap); devstral left this club on v0.5.18 — patch 062 recovered its pool to 339K and it now decodes **36.4 @261,916 actual** (88 short). REAM/REAP pruning does NOT speed M=1 decode (gemma4-21b-reap ≡ gemma4, coder-reap ≡ coder-30b): activated-expert count and attention dominate, not total expert weights — the pruned variants win on KV headroom and load footprint instead.

## Ledger

   - **Decode-topk sparse-KV (patch 059, opt-in `_ENV_GEMMA_TOPK`) — WIN 2026-07-19, the deep-context Gemma lever.** R9700's 069 rebased to v0.5.15; first SWA-hybrid + first CUDA datapoint. gemma4-31b **12.9 → 26.2 tok/s @261,916 actual (2.03×)**, depth curve FLAT (34.1→38.1ms TPOT, 2K→262K; the 45.5ms depth term collapses to ~4.6ms of page selection), crossover vs graphs-on ≈ **80-90K**. Recall perfect at every gate (tooluse 1.0/1.0 + quality 100% to ~255.9K actual, caps 5/5, agentic A/B parity-within-noise with 0/161 garbled tool calls). Opt-in only: below the crossover graphs-on wins (agentic median ~41K). `_ENV_GEMMA_TOPK="--decode-topk-pages 256 --decode-topk-page-size 64"` on any gemma preset. Receipts: [`benchmarks/gemma-topk-port/verdict.md`](../benchmarks/gemma-topk-port/verdict.md).
   - **NGRAM spec (opt-in `NGRAM=1`, `coder-30b`/`coder-30b-eval` only).** Draft-model-free (CPU trie → works at 256K where EAGLE3/DFlash OOM) and does not collapse at depth: @172K, no-spec 89 t/s → **235–237 t/s (~2.6×)** on copy-heavy spans (accept 6–7.6), ~42 t/s floor on novel spans. Opt-in because it's gated by copy fidelity; REAP/REAM pruning degrades it and DeltaNet thinkers are excluded (recurrent verify wall). Receipt: [`benchmarks/ngram-copyheavy-at-depth-2026-06-15.md`](../benchmarks/ngram-copyheavy-at-depth-2026-06-15.md).
   - **EAGLE3-at-24GB pool-capping — TESTED 2026-07-19, split verdict** ([receipt](../benchmarks/quality/coder30b-eagle3-poolcap-2026-07-19.json)). **Memory WIN:** `--max-total-tokens` defuses the profiler's over-provision (4.3× at CTX=98K/MEM=0.80; reclaim 7.2 GiB/card) — coder-30b + draft + cuda graphs boot at every CTX **including full 256K** (16.3 GiB/card; the 16K lane cap was a profiler artifact). **Perf NULL at depth:** spec wins only ≤~8-10K (1.53-1.76× at 2.5K); server-verified 0.86× @14K, 0.85× @40K, 0.76× @61K, 0.62× @92K, 0.49× @252K as accept decays 3.8→1.58 (the v0.5.12 "1.65× @16K" receipt was depth-unverified delta-method). `--speculative-draft-window-size` (the counter-lever) is **blocked upstream**: FlashInfer's multi-step draft backend asserts `num_wrappers==1` on the shared `kv_indptr_buf` path while a draft window forces 2. NGRAM remains the only depth spec; EAGLE3 stays a short-prompt lever behind the task-#17 depth-crossover fallback (~8-10K threshold, not a VRAM band).
   - **Agentic-workload profile (qwen38, live bake-off traffic, 2026-09-10)** — [`benchmarks/qwen38-agentic-workload-profile-2026-09-10.md`](../benchmarks/qwen38-agentic-workload-profile-2026-09-10.md). Server time is **86% decode / 11% prefill / 3% queue** (14.6K requests, 94% prefix-hit, median 450 uncached + 276 generated tokens per turn). Decode step at 14.6K ctx = 15.4 ms: **INT4 Marlin 57%** (727 GB/s = 78% of DRAM peak), **fp16 lm_head 10%** (863 GB/s, at roofline), **NCCL allreduce 12%** (129 × 13.6 µs), attention 7%, GDN 5%, launch-floor kernels + gaps 9%; = 55% of the pure-bandwidth roofline. Prefill Marlin runs at ~90% of tensor peak; the sm_86 flashinfer prefill attention is 21% at ~28% of peak. The 260 W cap binds 100% of the time but **350 W = only +5.5% decode — rejected** (DIMMs at ALARM HIGH). Ranked levers that remain: **NGRAM spec (1.3–2×, blocked on the spec-verify conv1d cast, [`roadmap.md`](roadmap.md#tooling))** > lm_head INT8 via Marlin (≤7%) > custom allreduce (6%, blocked sm_86) > everything else <3%. Open: 3–4 GB/s of unexplained PCIe host reads during decode (off the critical path). Tool: `scripts/bench/trace_step_anatomy.py` (per-step exposed-time attribution of a `/start_profile` trace).
   - Null/closed levers (one-line findings, receipts under `benchmarks/`): **decode-attention roofline** — sm_86 already runs ~72% of BW roofline at depth (≤+23% ceiling; [`attn-roofline-sm86-2026-07-15.md`](../benchmarks/attn-roofline-sm86-2026-07-15.md)); **R9700 MoE-campaign ports** — fused-topk already default-on CUDA, qk-norm-rope ±0.2%, MoE config tuning null/blocked (Ampere heuristics at-optimum; [`nemotron-moe-tune-null-2026-07-14.md`](../benchmarks/nemotron-moe-tune-null-2026-07-14.md), [`gemma4-tune-and-flashinfer-close-2026-07-14.md`](../benchmarks/gemma4-tune-and-flashinfer-close-2026-07-14.md)), their BF16 collectives reclaim a HIP-only penalty; **per-layer-type FlashInfer for Gemma** — refuted at the model level (backend assert); **dense-TP allreduce acceleration** — null/blocked on sm_86 ([`allreduce-accel-null-2026-06-15.md`](../benchmarks/allreduce-accel-null-2026-06-15.md), re-test toggle `ENABLE_CUSTOM_AR=1`). Also null: **bf16-operand PV** (R9700 087's +21%-at-depth) — measured **+1% @244K here, recall clean both arms** (their win fixed a 51%-of-roofline occupancy bind; sm_86 already runs ~72% BW, so there's nothing to collect; 011's fp32-PV precision margin costs ~1% and stays; receipts `pv-precision-ab-{bf16,fp32}-2026-07-16.json`). Still-open from the gemma4 fallback-kernel root cause (group-32 AWQ + TP-hostile shapes → dense MLP on unoptimized AWQ GEMM, experts on wna16 Triton): **Marlin-friendly gemma4 requant** (calibration-device item) — now the sole fix direction: R9700's grid-level split-K for narrow-shape GEMM was implemented, benchmarked, and refuted on their side (their #25, 2026-07-16), so the kernel path is dead on both stacks.

## Where the step time goes (qwen38 agentic profile, 2026-09-10)

Full write-up: [`benchmarks/qwen38-agentic-workload-profile-2026-09-10.md`](../benchmarks/qwen38-agentic-workload-profile-2026-09-10.md); tool: [`scripts/bench/trace_step_anatomy.py`](../scripts/bench/trace_step_anatomy.py) (per-step exposed-time attribution of a `POST /start_profile` trace).

| component | ms/step | share |
|---|---|---|
| Marlin INT4 GEMMs (727 GB/s = 78% of DRAM peak) | 8.7 | 57% |
| fp16 lm_head gemv (863 GB/s, at roofline) | 1.5 | 10% |
| NCCL allreduce, 129 × 13.6 µs | 1.8 | 12% |
| flashinfer decode attention (fp8 KV) | 1.0 | 7% |
| GDN recurrent + conv | 0.7 | 5% |
| norms / act / launch gaps | 1.4 | 9% |

Step = 15.4 ms at 14.6K context = 55% of the pure-bandwidth roofline. The ranked experiments derived from it are queued in the README's *Next steps*.

## Speculative decoding

**EAGLE3 drafts (published; SpecForge online training on our 2×24 GB).** **Devstral-24B** → [`mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ-EAGLE3) (`LlamaForCausalLMEagle3`). **Measured decode speedup on our 2×3090** (single-user coding, `num-steps=3`): **short 91.9 → 207.5 tok/s = 2.26×** (accept_len 3.32); **~16K 80.2 → 153.4 tok/s = 1.91×** (accept_len 2.86). Receipt: [`benchmarks/quality/devstral-eagle3-speedup.json`](../benchmarks/quality/devstral-eagle3-speedup.json). `ttt`/num-steps capped at **3** by 24 GB training memory; spec is a **≤~64K win** (use no-spec at true 256K depth). ⚠ **Serving caveat:** EAGLE3 attaches to the Devstral **text decoder** (`Ministral3ForCausalLM`) — sglang's full-VLM wrapper (`LlavaForConditionalGeneration`) lacks `set_eagle3_layers_to_capture`, so serve the draft against the text-decoder, not the VLM wrapper (a sglang delegation patch would close that). **Qwen3-VL-32B** → [`mattbucci/Qwen3-VL-32B-AWQ-EAGLE3`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ-EAGLE3) (2026-07-15). Measured here (steps 3 / topk 4 / draft 8): **short 60.4→112.2 tok/s = 1.86×** (accept 2.47), **~16K 52.1→83.2 = 1.60×** (accept 2.16) — trained at **max-length 6144** (the 19 GB 32B target on 24 GB cards can't fit the Devstral 16K recipe: full-vocab logits + target spill OOM both GPUs; the zero-copy reduction-shift refactor for 16K-class targets is documented in `scripts/specforge/launch_qwen3vl_eagle3_realrun.sh`). Same text-decoder attach caveat as Devstral (serve against the extracted `Qwen3ForCausalLM`, not the VL wrapper — extraction: `scripts/specforge/extract_qwen3vl_text_only.py`). Receipt: [`benchmarks/quality/qwen3vl32b-eagle3-speedup.json`](../benchmarks/quality/qwen3vl32b-eagle3-speedup.json). Recipe + the 2×24 GB memory fixes: [`scripts/specforge/eagle3_training_plan.md`](../scripts/specforge/eagle3_training_plan.md). (Serve on v0.5.13 with `TVM_FFI_GPU_BACKEND=cuda` + `SGLANG_ENABLE_SPEC_V2=0`.)

Below is the **serving** picture (draft stays BF16; target quant is independent). Receipt: `benchmarks/quality/specdec-v0512-2026-05-29.json`.

| Target | Algo / Draft | Baseline | With spec | Speedup |
|---|---|:---:|:---:|:---:|
| `coder-30b` AWQ-native | EAGLE3, `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge` (steps 4 / topk 4 / draft 8) | 185 tok/s | **306 tok/s** | **1.65×** |
| `qwen36` AWQ | DFlash, `z-lab/Qwen3.6-35B-A3B-DFlash` (`--dtype bfloat16` + spec-v2) | 126 tok/s | 126 tok/s | **~1.0× (moot)** |

DFlash buys nothing on `qwen36` — graph-ON no-spec already decodes 126 tok/s @256K (174 @1K), matching DFlash at its 32K cap. A second reason (beyond the 24 GB-fit limits below) no-spec is the only viable path.

**FP8: not a lever on this hardware** — see the [Direction](../README.md#direction) table in the README for the sm_86 compile wall + the int4-wins-decode economics; same-weights control decoded **210 tok/s short / 179 @16K** on AWQ-int4 where FP8 won't run at all. Receipt: [`benchmarks/fp8-vs-awq-coder-reap.json`](../benchmarks/fp8-vs-awq-coder-reap.json).

**Spec collapses at true 256K depth** (draft acceptance craters + the draft re-attends the full deep KV every micro-step) — confirmed for both EAGLE3 and DFlash, pure-attention and DeltaNet alike. So the documented `@256K` spec bars are short-depth-on-a-256K-server; **at depth, no-spec is the path** and spec is a ≤~64K optimization.

**Constraints on 24 GB cards** (R9700 has 32 GB headroom; ours doesn't):
- Drop `--mem-fraction-static 0.70` so the target leaves room for the draft + its cuda graphs (preset `MEM=0.85` OOMs the draft).
- EAGLE3: R9700's wide ladder (topk 16 / draft 32) OOMs the draft graphs here; our wider-but-fits ladder (steps 4 / topk 4 / draft 8) is the sweet spot.
- DFlash on `Qwen3_5MoeForConditionalGeneration`: must export `SGLANG_ENABLE_SPEC_V2=1`, pass `--mamba-scheduler-strategy extra_buffer`, **and force `--dtype bfloat16`** (the BF16 draft mismatches the FP16 target → `Index put dtype mismatch` at boot). Cap context at 32K to fit.
- Universal: `--speculative-draft-model-quantization unquant` (draft stays BF16) and `--speculative-attention-mode decode`.

Not applicable: gemma4 (no DFlash hook); AWQ's bundled MTP head is int4-dead, so NEXTN/MTP stays FP8-only.

**⚠ Spec-decode is not viable for our target workloads on 24 GB cards.** The numbers above are short-prompt decode. Two constraints kill it for the real workloads:

1. **SWE-bench prompts exceed the caps.** Measured against the finished qwen36-opencode-v2 cycle (300 instances): median peak prompt 41K, p90 82K, max 230K. **97.3% exceed EAGLE3's 16K**, **65.3% exceed DFlash's 32K**. Receipt: [`benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json`](../benchmarks/quality/qwen36-opencode-v2-prompt-length-distribution.json).
2. **256K + spec doesn't fit on 24 GB.** Per VRAM accounting (~15 GB weights TP=2 + 9 GB KV @ 256K + 5 GB cuda graphs + 0.4 GB draft) = ~21 GB/card → OOMs at MEM=0.85. R9700's 32 GB cards have headroom we don't.

The `SPEC_DECODE=1` opt-in remains wired for short-prompt uses. For our 256K agentic workloads, no-spec is the only viable path on 24 GB hardware. Full reasoning: [`evals/swebench/spec_decode_plan.md`](../evals/swebench/spec_decode_plan.md).

**MTP-on-int4 rule:** in-ckpt MTP heads do NOT graft onto int4 targets — the BF16 MTP mispredicts on int4-shifted hidden states (Qwen3.5-27B graft probe: accept 0.00, 0.1 tok/s, worse than no-spec). MTP transfer tolerates FP8 but not int4. For int4 spec-decode use a trained EAGLE3/DFlash draft, never a grafted MTP. Vision towers, on the other hand, graft cleanly — they're input-side and quant-decoupled.

## KV pools and context limits

‡‡ **"Max context" is the REAL KV-pool capacity** (`max_total_num_tokens` from the serve log), not the declared `--context-length`. The heavy-**VL** ships are KV-bound (dense weights + FP16 vision tower), but two config levers recover most of it: **SWA-ratio right-sizing** (the default `--swa-full-tokens-ratio 0.8` gives the sliding sub-pool 80% of full although sliding layers attend only `window=1024` — 12B 102K→565K, 26B 118K→652K) and **e5m2 FP8 KV on the triton-forced path** (31B 24K→347K combined with ratio 0.05 + MEM 0.92). devstral 202K (MEM 0.90) is full-attention-bound — no SWA pool to right-size; 262K needs MEM≈1.0. **Why the *smaller* devstral-24B caps lower than the dense gemma-4-31B (347K):** "dense" describes the FFN (no MoE routing), not the attention — and KV, the 256K bottleneck, is set by the attention layout. gemma-31B is hybrid-SWA (50/60 layers sliding at `window=1024`, only **10 full-attention**) with **MQA-4** global KV heads (`num_global_key_value_heads=4`) and `attention_k_eq_v=True` (caches K=V *once*) → **~12 KB/token**; devstral runs **full attention on all 40 layers** with GQA-8 and separate K/V → **~40 KB/token** (~3–4× heavier). At 262K that's ≈3 GB vs ≈10.5 GB of KV/GPU, so devstral exhausts the 48 GB ~3× sooner — its 0.7 GB-lighter weights don't compensate. The A3B-MoE models are genuinely 256K+ by arch: qwen36 996K / qwen36-ream 2.4M / qwen36-dense 657K / qwen3-ream 578K / Coder-A3B ~900K (fleet serve logs). Tool-use verified 1.0 to 258K true on 12B/26B/31B (probe table in the README).



## Thinking serving defaults





Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.
