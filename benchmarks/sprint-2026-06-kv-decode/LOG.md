# Experimentation sprint — KV economics + Gemma decode (2026-06-10, bake-off paused)

Lab notebook. One entry per experiment run: hypothesis → method → result → decision. All runs TP=2 on the live v0.5.12 tree (24 patches, `b212aa2`), single variable changed at a time, receipts as JSON next to this file. Instruments identical to the fleet eval (`run_v0512_fleet_eval.sh` mechanics: port 23334, MAX_RUNNING=6, `bench_long_context.py`, `probe_256k_tooluse.py`, `validate_capabilities.py`) so results compare directly to existing baselines in `benchmarks/<slug>/results.json`.

Patterns mined from the patch set that motivate the tracks:
- 043/047 (hybrid-SWA KV routing): the 12B "serves 256K" but **KV caps at 102K full / 81K swa**; 26B KV-walled at ~118K. SGLang sizes the SWA sub-pool at `--swa-full-tokens-ratio` **0.8** of the full pool by default — but a sliding layer can only attend `window=1024` tokens, and 40-of-48 (12B) / 25-of-30 (26B) layers are sliding. The SWA pool is the dominant KV consumer yet almost all of it is physically unreachable at MAX_RUNNING=1.
- 011/017/021 + the cuda-graph memory ("flat TPOT-vs-ctx = missing-graph tell"): Gemma 4 26B is our only family still decode-slow (~36-53 tok/s vs 126-196 elsewhere) because head_dim=512 forces triton attention → cuda-graph OFF.
- 007 (BV=64 sweep): default kernel configs under-serve sm_86 at M=1 — same logic applies to the MoE runner choice (marlin vs wna16) for the 26B.

## Track A — SWA sub-pool right-sizing (goal: true 256K KV on the Gemma hybrids)

**H-A1:** On `gemma4-12b` (window 1024, 40/48 sliding, int4 AWQ), `--swa-full-tokens-ratio` is the binding constraint on KV capacity. Lowering it from 0.8 toward the physical floor (~window + prefill chunk + slack ≈ 16K tokens ⇒ ratio ≈ 0.0625 at 262144 full) lifts the full pool to ≥ 262144 with **no** quality or decode regression — sliding layers cannot attend beyond 1024 tokens, so the freed slots were dead weight.

**Method:**
1. *Ladder* (boot-only, ~5 min/run): ratios {0.8 control, 0.5, 0.25, 0.1, 0.0625, 0.03}; scrape full/swa pool tokens + mem headroom from server log. Output: capacity-vs-ratio curve, `A1/ladder-*.json`.
2. *Probe* (3 runs: control, candidate, candidate-1-step): `validate_capabilities.py` (all modalities) + `probe_256k_tooluse.py` @ {16K, 65K, 131K, 196K, 256K} + `bench_long_context.py` decode curve. A too-small swa pool should fail loudly here (256K prefill thrash or assert), not silently.

**Decision rule:** ship-ratio = smallest ratio with boot OK ∧ caps all-PASS ∧ tool-use ≥ control at every length ∧ decode within 5% of control at every ctx — then ×2 safety. Wire into preset, note in README VRAM table. If no ratio reaches 262144 full tokens, record the actual ceiling + what became binding (weights? full pool? mem-fraction) and stop Track A1.

**H-A2:** same for `gemma4` (26B MoE AWQ, 25/30 sliding); KV wall ~118K → expect full 256K at ratio ≈ 0.0625–0.1.

## Track B — Gemma 4 26B decode (goal: ≥1.3× at 131K+, 5/5 caps held)

**H-B1:** torch_native attention + cuda-graph ON beats triton + graph OFF at M=1 long-ctx decode — launch-overhead domination (the R9700 dispatch-bound profile: ~48% GPU util) outweighs torch_native's slower kernels once graphs remove the fixed ~27 ms/step overhead.
Method: A/B `_ENV_CUDA_GRAPH` + `EXTRA_ARGS="--attention-backend torch_native"` vs preset; decode @ {1K, 32K, 131K, max}; caps battery both arms. Also record *why* triton+graph fails (capture error) — if it's shallow, that's a patch candidate.

**H-B2:** MoE runner choice (awq_marlin vs moe_wna16) is measurably different at M=1 E=128 gelu; the preset may be on the slower one.
Method: A/B QUANT routing, decode @ {1K, 131K}, same caps battery.

## Track C — fleet TPOT-vs-ctx audit (no GPU; runs in wait gaps)

Mine every `benchmarks/*/results.json` for the flat-TPOT tell (missing graphs / launch-bound) and for decode cliffs; tabulate slope 1K→max per preset. Anomalies become Track-B-style experiments.

---

## Run log

(entries appended chronologically by the sweep runner + analysis)

### 2026-06-10 01:55 — Track C: fleet TPOT-vs-ctx audit (existing results.json, fleet eval 2026-06-07)

| preset | max KV tok | ctx measured | TPOT lo→hi ms | ratio | tok/s lo→hi | cliff |
|---|---:|---|---|---:|---|---|
| devstral | 172,058 | 1K–128K | 10.8→17.4 | 1.61 | 92→58 | 64K→128K −19% |
| **gemma4-12b** | 102,094 | 1K–64K | 23.4→25.1 | **1.07** | 43→40 | – |
| **gemma4 (26B)** | 117,840 | 1K–64K | 29.3→29.6 | **1.01** | 34→34 | – |
| **gemma4-31b** | 24,248 | 1K–16K | 29.9→29.8 | **0.99** | 33→34 | – |
| qwen3-ream | 578,433 | 1K–256K | 5.5→9.5 | 1.75 | 183→105 | **64K→128K −26%** |
| qwen35-moe | 2,108,823 | 1K–256K | 5.9→7.3 | 1.23 | 170→138 | – |
| qwen36-dense | 657,010 | 1K–256K | 14.6→18.3 | 1.26 | 69→55 | – |
| qwen36 | 875,947 | 1K–256K | 5.8→7.9 | 1.38 | 174→126 | – |
| qwen36-ream | 2,166,819 | 1K–256K | 5.8→7.2 | 1.24 | 172→139 | – |

**Findings.** (1) The flat-TPOT missing-graph tell fires on **all three Gemma presets** (1.07 / 1.01 / 0.99 — a decode step that doesn't get more expensive from 1K to 64K KV is launch-bound, not compute-bound). Track B therefore targets the whole Gemma family, expected headroom 2–4× like the qwen36-family graph win. (2) `qwen3-ream` has a real −26% decode cliff at 64K→128K (5.5→9.5 ms TPOT) — healthy at 105 tok/s @256K but the step is anomalous → queued as **B3** (suspect: attention kernel regime change, e.g. split-KV path switch). (3) Coverage gaps corroborate Track A: the Gemma rows stop exactly where their KV walls sit (102K/118K/24K vs 578K–2.2M for the Qwen fleet); `coder-30b-awq/results.json` is an old format with no usable rows — re-bench when convenient. (4) Qwen3.5-27B legacy row (Apr 13, 13 tok/s, 75 ms flat TPOT) is the known replicated-DeltaNet case — superseded by qwen36-dense, not pursuing.

### 2026-06-10 01:56 — A1 ladder result (gemma4-12b, boot-only, warm cache ~27s/boot)

| `--swa-full-tokens-ratio` | full tokens | swa tokens | 262144 full? |
|---:|---:|---:|:---:|
| 0.8 (default = control) | 102,094 | 81,675 | ✗ |
| 0.5 | 153,141 | 76,570 | ✗ |
| 0.25 | 262,528 | 65,632 | ✓ (exactly) |
| 0.1 | 459,425 | 45,942 | ✓ |
| 0.0625 | 565,446 | 35,340 | ✓ |
| 0.03 | 706,807 | 21,204 | ✓ |

**Cost model validated:** control receipt gives per-GPU per-token K+V ≈ 152.6 KB (swa: ~38 sliding-layer-equivalents × 4 kv × 256 × 2B × 2) vs ≈ 15.3 KB (full: 8 global layers × MQA × 512 × 2B × 2); budget 14.03 GB. Predictions {0.25: 262,540, 0.0625: 565,4xx, 0.03: 706,8xx} matched measurements to ≤0.1% on every rung — capacity is purely `budget / (15.3 + r×152.6) KB`. **H-A1 capacity half: CONFIRMED — the SWA sub-pool was the binding constraint; 6.9× full-pool headroom at the smallest rung.** Excess full tokens beyond 262144 = radix-cache retention for multi-turn agentic reuse (not waste). Next: probe phase on {0.03, 0.0625} for the quality/safety floor (256K prefill exercises the small swa pool: working set ≈ chunked-prefill chunk + window 1024 ≪ 21,204).

### 2026-06-10 02:32 — A1 collateral finding: 12B VIDEO crashes the server (pre-existing, ratio-orthogonal) → A1-V

First probe pass died mid-battery: the **video** request kills the scheduler (`RemoteDisconnected`), and every subsequent probe hit a dead server. Control-ratio confirm (0.8, receipt `A1-gemma4-12b-videoctl/probe-ratio-0.8.caps.log`): identical 4/5 PASS + video crash → **pre-existing, not a ratio effect** (video was never validated in the 043 bring-up — only text/tool/thinking/vision). Traceback: `mm_utils._get_chunked_embedding_by_item` → `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 768 ... but got split_sizes=[64]`. Read: the vendored `Gemma4UnifiedProcessor.__call__` (patch 048) expands the video placeholder to **64** tokens while the model's video path emits **768** embeddings (3 frames × 256/frame) — the image branch was aligned at 256, the video branch wasn't. Same defect class 048 fixed for images. **Action: A1-V — fix video expansion in the vendored processor (patch 050 candidate); video excluded from the A1 gate (can't regress what never worked).** Probe-v2 relaunched with `--skip-video`, control 0.8 included for same-day A/B.

### 2026-06-10 03:15 — A1 CONCLUDED: gemma4-12b ships ratio 0.0625 (102K → 565K full KV, true-256K verified, 5/5 omni)

Probe gate (receipts `A1-gemma4-12b/probe-ratio-*.{caps.log,tooluse.json,decode.json}`, control = same-day 0.8):

| metric | 0.8 control | 0.0625 | 0.03 (floor) |
|---|---|---|---|
| full KV tokens | 102,094 | 565,446 | 706,807 |
| caps (video excluded, A1-V) | 4/4 | 4/4 | 4/4 |
| 256K tool-use valid/correct | 0.6/0.6 (walled @75,613 true) | **1.0/1.0** | **1.0/1.0** |
| decode 1K→64K | 41.0→39.6 | 41.3→38.4 (≤3% Δ) | 41.3→38.7 |
| decode 128K/192K/256K | — (over cap) | 30.9/31.1/30.9 | 30.8/30.8/30.9 |

Floor 0.03 passes everything → **ship 0.0625 (floor×2 per decision rule)**, wired into the `gemma4-12b` preset. Final wired-preset validation (`A1-gemma4-12b-final/`): **5/5 capabilities — video now PASSES** ("a red circle moving vertically on a white background", patch 050 live-verified) and tool-use **1.0/1.0 at true prompt lengths {16,655 / 66,179 / 132,211 / 198,243 / 258,085} tokens**. Note for future gemma probes: the tool-use filler renders ~0.576 true-tok/approx on the Gemma tokenizer — approx 448000 ≈ 258K true (lengths used: 28672–448000). The 128K+ decode band (~31 tok/s, TPOT 32 ms) is the new dense-attention regime — Track B's graph work is the lever there. H-A1 fully CONFIRMED.

### 2026-06-10 03:20 — A2 ladder (gemma4 26B MoE): lever generalizes

| ratio | full tokens | swa tokens |
|---:|---:|---:|
| 0.8 (control) | 117,840 | 94,272 |
| 0.5 | 176,760 | 88,380 |
| 0.25 | 303,017 | 75,754 |
| 0.1 | 530,280 | 53,028 |
| 0.0625 | 652,652 | 40,790 |
| 0.03 | 815,815 | 24,474 |

262144 clears at ratio ≤0.25. Probe phase gates `{0.8 control, 0.0625 ship-candidate}` directly (vs re-probing floor 0.03): the swa working set is arch-invariant — `window 1024 + prefill chunk + slack`; A1 verified **21,204 swa tokens** sufficient under a 258K-true prefill, and the 26B at 0.0625 carries **40,790** (~2× that verified absolute floor). Full caps battery incl. video (works on the 26B via patch 026). Extended tooluse lengths (approx 28672–448000 ≈ true 16K–258K on the Gemma tokenizer).

### 2026-06-10 03:25 — Track D pass 1 (logged-fallback fingerprint, fleet boot logs): HIT on gemma4 26B

Method: grep fleet `/tmp/v0512-eval-logs/*/server.log` for `Using <x> kernel` + `Falling back` + quant-method lines. Result: **devstral, gemma4-12b, gemma4-31b, qwen36, qwen36-dense, qwen36-ream, qwen3-ream = clean awq_marlin.** `gemma4` (26B): boot banner says awq_marlin, but per-layer:
- `layers.{0..29}.mlp.down_proj` → **"not supported by AWQMarlin → unoptimized AWQ kernels"** — the dequant+GEMM slow path (R9700's 5× class), on the **dense** MLP that runs EVERY token (parallel dense+MoE block).
- `layers.{0..29}.moe.experts` → "not supported by AWQMoeMarlin → Moe WNA16" — structural (expert `down_proj` in=704 ≠ 0 mod 128/64-shard; WNA16 Triton fused-MoE is the correct fallback, acceptable).

Root cause of the dense reject: down_proj `in=2112`, checkpoint `group_size=64` (33 groups, the patch-015-era shape) → at TP=2 RowParallel each rank holds 1056 → **16.5 groups** → Marlin can't pack fractional groups. At TP=1 it would pack (33 ✓). Candidate fixes, queued as **B2′** for the next GPU slot: (a) replicate the dense `down_proj` (tp_size=1 for that module — ~170 MB total replicated int4, marlin-packs at full 2112); (b) check marlin_utils' pad-groups path; (c) microbench first: marlin vs unoptimized-AWQ at (in 1056, out 5376) M=1 to size the prize (`scripts/test_marlin_repack.py` harness). This compounds with the graph-off finding (Track C) for why the 26B is fleet-slowest (22–33 tok/s).

**Limitation:** pass 1 only sees *logged* fallbacks — R9700's GEMV bug was silent. Pass 2 (queued): runtime kernel verification per preset (one profiled decode step; compare hottest GEMM kernel name against intent).

### 2026-06-10 03:35 — A2 CONCLUDED: gemma4 (26B) ships ratio 0.0625 (118K → 652K full KV, true-256K verified, 5/5 caps)

| metric | 0.8 control | 0.0625 |
|---|---|---|
| full / swa KV tokens | 117,840 / 94,272 | 652,652 / 40,790 |
| caps (full battery incl. video) | 5/5 | 5/5 |
| tool-use valid/correct | 0.4/0.4 (walls at 66,179 true) | **1.0/1.0 to 258,085 true** |
| decode 1K→64K | 33.1→32.5 | 34.1→33.3 (≤1% Δ, if anything faster) |
| decode 128K/192K/256K | — (over cap) | 31.6/31.1/31.2 |

**Track A CONCLUDED.** Both Gemma hybrids are now genuine 256K models. Same-shaped result as A1; the swa floor argument held (40,790 swa tokens ≈ 2× the A1-verified 21,204 working set; no eviction thrash — decode flat ~31 tok/s through 256K). The ~31 tok/s 128K+ band on BOTH models is the launch-bound ceiling Track B attacks next (graph-off TPOT ~32 ms dominates regardless of attention length — consistent with the Track C flat-TPOT tell).

### 2026-06-10 04:57 — B1 arm N + B2′ microbench

**B1 arm N (torch_native, graphs OFF):** decode 34.8@1K → 32.6@64K → 31.1@256K, caps **5/5** — indistinguishable from the triton control (34.1→31.2). The backend swap is free on the 26B; the decisive arm is G (torch_native + cuda-graph), relaunched after an env-assignment bug no-oped it (`${var:+VAR=x}` prefix-assignments don't survive expansion — fixed with explicit `export` in a subshell).

**B2′ — dense down_proj fallback prize: PARKED (1.9% < 5% bar).** Microbench at the production per-rank shape (M=1, K=1056, N=5376, g=32), receipt `B2-downproj-microbench.json`: dequant+mm (current fallback) **43.9 µs**; in-tree `awq_gemm_triton` fused W4A16 **23.8 µs**; pure fp16 mm floor **23.5 µs**. The Triton path is already AT the bandwidth floor, so best-case saving = 20 µs × 30 layers = **0.60 ms/token ≈ 1.9% of the ~32 ms TPOT**. Not worth a routing patch; the 26B's gap is launch-bound, not this. Root-cause note for the ledger: the marlin reject is the `K % 128` thread-tile check (1056%128=32; full-width 2112%128=64 — so replication would not have helped either); checkpoint g=32 is itself marlin-legal. Track D pass-1 finding hereby *sized and closed for the 26B dense path*; the `moe.experts`→WNA16 fallback remains structural-and-fine.

### 2026-06-10 05:27 — B1 CONCLUDED on the 26B: graphs were the whole gap; "triton can't capture" FALSIFIED

| arm | 1K | 32K | 64K | 128K | 256K | caps |
|---|---:|---:|---:|---:|---:|---|
| control (triton, no graph — A2) | 34.1 | 33.9 | 33.3 | 31.6 | 31.2 | 5/5 |
| N (torch_native, no graph) | 34.8 | 33.5 | 32.6 | 31.2 | 31.1 | 5/5 |
| **G (torch_native + graphs, MEM 0.78)** | **83.3** | 65.7 | 55.2 | 40.9 | **40.9** | 5/5 |
| **GT (triton + graphs, MEM 0.78)** | **82.9** | 65.8 | 54.9 | 41.1 | **40.9** | 5/5 |

**Verdict:** graphs deliver **2.44× @1K → 1.31× @256K** (clears the ≥1.3× bar at every point); backend choice is irrelevant (N ≈ control; GT ≈ G). **The 2026-06-06 belief "Gemma's triton SWA attention can't graph-capture" is falsified on v0.5.12 at bs=1 capture** — the 26B sat at flat-33 tok/s for no current reason. Decode is now properly attention-bound (83→41 vs the flat launch-bound 33). Wired: `gemma4` preset graphs ON (`--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph`) + MEM 0.85→0.78 (capture headroom, qwen36 precedent). 12B/31B graph arms running for their own receipts (hooks added, defaults unchanged until receipts land). The flat-TPOT tell from Track C is hereby explained for all three Gemma presets; B2′ already showed the dense-MLP fallback contributes only 1.9%.

### 2026-06-10 06:35 — B1b: graphs flipped fleet-wide for Gemma (12B + 31B receipts)

| preset | control @1K→cap | graphs @1K→cap | gain | caps |
|---|---|---|---|---|
| gemma4 (26B) | 34.1→31.2 @256K | 82.9→40.9 @256K | 2.44×→1.31× | 5/5 |
| gemma4-12b | 41.3→30.9 @256K | **107.3**→34.1 @256K | 2.60×→1.10× | 5/5 |
| gemma4-31b | 33.4→33.6 @16K (flat) | **57.7**→46.9 @16K | 1.73×→1.40× | 5/5 (caps from G arm) |

All three Gemma presets now default `--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph` via the `_ENV_GEMMA_GRAPH` hook (env-overridable for A/Bs); MEM stays 0.85 — capture fits with the full Track-A pools (the 0.78 precaution was unnecessary, receipts confirm). 12B note: 256K gain is small (1.10×) because dense-attention time dominates there; the 1K–64K band (where most agentic decode lives — median prompt 41K) gains 1.3–2.6×. Two shell-expansion traps cost one arm-run each, both now documented in-script: `${var:+VAR=x}` prefix-assignments don't assign; `local a="$1" b="$a"` expands before assigning.

### 2026-06-10 07:12 — B3 CONCLUDED: the qwen3-ream "cliff" does not exist (coarse-sampling artifact)

Fine-step decode on the preset's FlashInfer path (receipt `B3-qwen3-ream/fi.json`): 48K **6.2 ms** → 64K 7.0 → 80K 7.9 → 96K 7.9 → 112K 8.5 → 128K 9.4 → 144K **9.4 ms** TPOT — smooth, monotone, attention-bound. No regime discontinuity anywhere in 48K–144K; the fleet table's "−26% at 64K→128K" was two samples on a doubling interval read as a step. **Parked: nothing to fix.** (Triton comparison arm boot-failed on the known sm_86 `fp8e4nv` KV limitation — the preset serves FP8 KV, which triton can't emit on Ampere, patch-005 domain; irrelevant to the verdict since the production path IS FlashInfer.) Meta-lesson for Track C-style audits: a "cliff" needs ≥3 points inside the step before it's real.

### 2026-06-10 07:14 — D2 demonstrator (qwen36-dense): CLEAN; Track D concluded

Profiled decode (256 tok) via `/start_profile` + `SGLANG_TORCH_PROFILER_DIR` (`scripts/bench/profile_decode_step.sh`; receipt `D2-qwen36-dense/top_kernels.json`): **Marlin is the dominant kernel (4,461 ms aggregate) — `marlin_present: True, dequant_present: False`.** The 1,530 ms of cuBLAS `gemvx` fp16 is the by-design unquantized set (lm_head + DeltaNet `in_proj_a/b`, patch-031 domain; `fused_recurrent_gated_delta_rule` 284 ms confirms the hybrid). No silent fallback. **Track D concluded:** pass 1 (boot-log fingerprint) caught the fleet's only logged fallback (26B dense down_proj — sized at 1.9% by B2′ and parked); pass 2 (this profiled-decode recipe) is the per-preset audit for the silent class — run it on any new preset or after any kernel-adjacent bump.

---

## SPRINT SUMMARY (2026-06-10, ~5.5 h wall, all tracks concluded)

| Track | Verdict | Wired / Receipts |
|---|---|---|
| **A — SWA ratio** | Default 0.8 was the Gemma KV wall. 12B 102K→**565K**, 26B 118K→**652K** full tokens; tool-use **1.0/1.0 to 258,085 true tokens** both; caps 5/5 | `--swa-full-tokens-ratio 0.0625` in both presets; ladder cost-model exact to 0.1%; A1/A2 receipt dirs |
| **A collateral** | 12B video crashed pre-existing (processor declared per-frame tokens) | **Patch 050** (3-gated); 12B now **5/5 full omni** |
| **B1 — graphs** | "Triton SWA can't capture" FALSIFIED; graphs were the whole Gemma decode gap | All 3 Gemma presets graphs-ON: 26B 34→**83**@1K/31→**41**@256K; 12B 41→**107**@1K; 31B 33→**58**@1K; 5/5 caps everywhere |
| **B2′ — down_proj fallback** | Real but tiny: 0.60 ms/tok = **1.9%** of TPOT (triton W4A16 already at fp16-mm floor) | **Parked** with microbench receipt |
| **B3 — qwen3-ream cliff** | **No cliff** — smooth 6.2→9.4 ms TPOT 48K→144K; fleet "−26%" was 2-point sampling on a doubling | Parked; meta-rule: a cliff needs ≥3 in-step points |
| **C — TPOT audit** | Flat-TPOT tell found all 3 Gemma graph-off cases; cliff candidate disproven by B3 | Table in LOG; charts regen at next fleet pass |
| **D — kernel gates** | Pass-1 fingerprint: 7 presets clean, 1 hit (sized, parked). Pass-2 profiled-decode: qwen36-dense clean | `profile_decode_step.sh` = standing per-preset audit |

**Fleet effect:** the Gemma family went from "KV-walled (24K–118K) + flat ~33 tok/s + 12B video-crashing" to **two genuine 256K models at 83–107 tok/s short-ctx / 41+34 @256K with full multimodal batteries**, and the 31B at 58 tok/s within its weight-bound 24K. Tooling kept: swa_ratio_sweep.sh (capacity ladders + probe gates), b1_arms-style A/B harnesses, profile_decode_step.sh, fine-step decode protocol. Shell-trap lessons baked into scripts: `${var:+VAR=x}` doesn't assign; one-line `local a="$1" b="$a"` expands before assigning. Bake-off remains paused; resume picks up the new gemma presets automatically.

### 2026-06-10 07:52 — A3 ladder + B5 resolved-by-measurement (hypothesis dead, better one found)

**A3 ladder (gemma4-31b):** 0.8→24,248 / 0.5→36,373 / 0.25→62,354 / 0.1→**109,120** / 0.0625→**134,302** / 0.03→167,877 full tokens. 5.5× at 0.0625; weights bound it short of 256K, but ≥109K covers the agentic p90 (82K). Floor nuance vs A1/A2: at 0.0625 the swa pool is **8,393 tokens** — close to the working set (chunk 4096 + window 1024 + slack), so 0.1 (swa 10,912) is the safety-margin candidate. Probe gates {0.8, 0.1, 0.0625}; tooluse lengths capped at ~99K true (within both candidate pools).

**B5 (qwen36-dense replication): hypothesis DEAD — it was never replicated.** Empirical: D2 boot log shows **8.80 GB/GPU** (`Load weight end`), and v0.5.12's `qwen3_5.py` DeltaNet uses `ColumnParallelLinear` with attn-tp sharding — the v0.5.10-era full-replication hack is gone. The README VRAM row (13.5 GB/GPU) was stale → fixed. Real MBU = 68.7 / (936/8.8) ≈ **65%**, and the D2 trace names the residual: **NCCL AllReduce 815 ms vs Marlin 4,461 ms over 256 tok ≈ 3.2 ms/token ≈ 22% of the 14.6 ms TPOT**. New cheap experiment **B5′**: `--enable-flashinfer-allreduce-fusion` (in-tree, default off — fuses allreduce+rmsnorm) A/B on qwen36-dense; queued behind B4.

### 2026-06-10 08:22 — A3 CONCLUDED: gemma4-31b ships ratio 0.1 (24K → 109K full KV)

Probe gate (`A3-gemma4-31b/probe-ratio-*`): all three rungs caps **5/5**; tooluse — control 0.25 (walls past 16,655 true; 24K pool), **0.1 and 0.0625 both 1.0/1.0 through 99,194 true tokens**; decode identical at shared ctx (57.4 vs 57.5 @1K; ≤0.3% everywhere), candidates extend to 64K @30.7 (0.1) and 128K @21.1 (0.0625, 134K pool). **Ship 0.1, not 0.0625:** swa margin 10,912 vs the floor evidence (0.0625's 8,393 survived a 128K prefill — proven, but thin for multi-turn radix pressure, the bake-off pattern); 0.1's 109K full already covers the agentic p90 (82K). Tradeoff documented: 0.0625 buys 134K cap at ~77% of the proven swa working margin — revisit only if a workload needs >109K on this model. Track A now fully closed across all three Gemma hybrids: **24K/102K/118K → 109K/565K/652K**.

### 2026-06-10 09:16 — B5′ fusion NULL; B4 NGRAM-on-hybrid blocked (patch-051 candidate); trial retargeted

**B5′ (allreduce-fusion, qwen36-dense): NULL RESULT — parked.** Code probe 62.6 vs ctl 62.4; decode 68.3/63.6/54.8 vs 68.6/63.3/54.8 — identical within noise (target was ≥72 @1K). `--enable-flashinfer-allreduce-fusion` changes nothing measurable on this path; the 3.2 ms/token allreduce stands. Remaining ideas for that 22%: NVLink symm-mem/MSCCLPP paths (`--enable-torch-symm-mem`, `--enable-mscclpp`) — logged, not started.

**B4 (NGRAM on qwen36-dense): blocked twice, root-caused, retargeted.** Boot 1: spec×radix on DeltaNet hybrids requires `--mamba-scheduler-strategy extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1` (the DFlash recipe generalizes to NGRAM). Boot 2 (with recipe): Triton compile assert in the conv1d spec-verify branch — `matrix_x` defined fp16 then redefined bf16 in the `KERNEL_WIDTH` branches: **patch 003's exact bug class in a branch 003 didn't cover** (the spec path loads conv_state cols without the cast). Filed as **patch-051 candidate** (extend 003's `.to(_x_dtype)` to the KERNEL_WIDTH col loads); NGRAM-on-hybrids gated on it. Trial retargeted at **qwen3-ream** (full-attention MoE, no conv path, bake-off model, 105-183 tok/s) — B4c running with same-day ctl.

### 2026-06-10 09:38 — B4c (NGRAM on qwen3-ream): split verdict, one tuning arm before park

| probe | ctl | ngram (draft 12) |
|---|---|---|
| code agg | 186.5 | 188.9 (**1.01×**) |
| code per-row | 182/186/190/188 | **122 / 581 / 236 / 142** |
| filler decode 1K/40K/128K | 184/163/104 | 1042/714/328 (filler is repetitive → n-gram paradise; real but unrepresentative) |
| accept | — | len 1.48–2.10, rate 0.04–0.10 |

The variance is the story: **3.1× on the verbatim-copy prompt** (docstring insertion = mostly re-emission), **+24%** on fresh codegen, **−25-33%** on edit/diff prompts — the trie mispredicts on copy-with-deltas (THE agentic pattern) and 12-token drafts make each mispredict expensive. Hypothesis arm `ngram_d6` (`--speculative-num-draft-tokens 6`) running: shallower drafts cut mispredict cost, should keep most copy-span wins. Verdict after d6; bar stays ≥1.3× aggregate for a preset wire-in (opt-in env at most — bake-off latency variance matters too).

### 2026-06-10 09:55 — B4 CONCLUDED: NGRAM parked at 1.24× aggregate (pre-registered bar 1.3×); full tuning curve on record

| arm | code agg | rows (rename/docstring/fresh/diff) | accept |
|---|---|---|---|
| ctl | 186.5 | 182 / 186 / 190 / 188 | — |
| ngram draft-12 | 188.9 (1.01×) | 122 / 581 / 236 / 142 | len 1.5–2.1, rate 4–10% |
| **ngram draft-6** | **230.6 (1.24×)** | 131 / 394 / 251 / 314 | len 3.0–3.45, rate 40–49% |

Shallow drafts validated the mispredict-cost hypothesis (d12→d6: diff prompt 142→314, fresh 236→251, aggregate 1.01→1.24×) at the cost of capping copy-span wins (581→394). **Parked per the bar**: 1.24× aggregate with a −28% worst-case row isn't bake-off-safe as a default, and opt-in wiring was gated on ≥1.3×. On record for future use: **copy-heavy workloads (doc-gen, boilerplate, refactor bots) get 1.3–2.1× from one flag** (`--speculative-algorithm NGRAM --speculative-num-draft-tokens 6`, zero VRAM); receipts here. Spec-decode state after B4: hybrids gated on **patch-051 candidate** (conv1d spec-verify fp16/bf16, 003-class); the EAGLE3 pool-cap (+`--speculative-draft-window-size`) path remains the untested 24 GB option; filler-bench numbers under NGRAM are invalid as decode floors (repetitive filler = trie paradise — 1042 tok/s artifacts).
