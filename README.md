# NVIDIA Inference: SGLang on 2x RTX 3090

Single-user, **256K-context** LLM inference on 2× NVIDIA RTX 3090 (GA102, Ampere, 48 GB total) — SGLang **v0.5.20** + 31 local patches, CUDA 13.2 / PyTorch cu130, every model an **AWQ-int4 ship calibrated in-house** from the upstream BF16 base. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

Long-form material lives in [`docs/`](docs/) — [decode levers](docs/decode-levers.md) · [SWE-bench harness](docs/swebench-bakeoff.md) · [quality-eval methodology](docs/quality-evals.md) · [roadmap detail](docs/roadmap.md) · [host setup](docs/host-setup.md) · [OCI image](docs/oci-image.md) — and the per-patch history in [`patches/README.md`](patches/README.md). Agent operating rules: [`CLAUDE.md`](CLAUDE.md), [`rules-for-agents.md`](rules-for-agents.md).

## Direction

**We optimize for 256K-context single-user agentic workloads on AWQ-int4 ships.** SWE-bench Lite is the canonical eval — every preset serves at full 256K (or model-card max) so agentic harnesses with multi-turn tool-call context (median ~41K, p90 ~82K, max 230K per instance) actually fit. Decode tok/s / TPOT at depth is the primary metric; multi-user throughput is secondary and never bought with single-user latency.

This rules out three axes other 3090 stacks chase:

| Their focus | Why not ours |
|---|---|
| Short-ctx multi-stream throughput (vLLM-style 80-140 TPS @ <32K) | We need the full prompt of an agentic instance in context; truncating mid-conversation loses correctness. |
| FP8 quantization | **FP8 W8A8 MoE doesn't compile on sm_86** (the Triton fused-MoE kernel needs `fp8e4nv`; Ampere Triton only has e5m2), and even where FP8 is native (R9700 gfx1201), AWQ-int4 wins single-user M=1 decode — weight-byte-bound, FP8 is 2× int4 (Coder-30B 56 vs 38 tok/s). FP8 is the R9700 lane. Receipt: [`benchmarks/fp8-vs-awq-coder-reap.json`](benchmarks/fp8-vs-awq-coder-reap.json). |
| Spec-decode (EAGLE3 / DFlash / MTP) | Draft acceptance collapses at depth and 256K + draft + cuda graphs OOM on 24 GB cards; 97% of SWE-bench prompts exceed the drafts' caps. Short-prompt opt-in only — [`docs/decode-levers.md`](docs/decode-levers.md#speculative-decoding). |

What we **don't** ship: random community quants. Every `mattbucci/*-AWQ` is calibrated end-to-end from the upstream BF16 base via our own GPTQ → CT → AWQ-Marlin pipeline, with thinking + image + video + audio preserved and probed. When a model needs MoE expert compression we run REAP (pruning) or REAM (merging) ourselves on the upstream weights (`scripts/quantize/run_reap.py`, `run_ream_qwen3moe.sh`).

## Results at a glance

![Per-model single-user decode tok/s — peak (short ctx) vs 256K, or the model's real KV cap where it doesn't reach 256K](benchmarks/all_models_decode.png)

![Single-user decode tok/s vs context length — all AWQ presets, unified 256K x-axis](benchmarks/all_models_context.png)

![SWE-bench Lite resolve rate per preset × scaffold, full-300 cells](benchmarks/bakeoff_swebench_lite.png)

![Speculative decoding — AWQ int4 vs +draft single-user decode; Qwen3.8-27B DSpark on the shipped config (fp8 KV, true 256K) at 60-tok / 49K / 130K / 250K](benchmarks/specdec_comparison.png)

Every deep point is **server-verified at its labeled depth** (`actual_input_tokens`; deepest honest point 255K = 262144 − output − margin), single-user (M=1), fresh prefill, at each model's real KV pool. **True-256K decoders:** the A3B DeltaNet MoEs lead (`qwen35-moe` / `qwen36-ream` **144 @255K**, `qwen36` 121, ~210 short), the Mamba2-hybrid `nemotron3-omni` is near-flat (101 → 93), the 30B-A3B coder trio sits at ~69 @255K (200 short), and the dense DeltaNet 27Bs (`qwen36-dense` / `qwen38`) at ~48 (70 short). The Gemma 4 hybrids trail at depth (13–24 tok/s) because their group-32 AWQ + TP-hostile shapes fall off Marlin onto fallback kernels; the opt-in decode-topk lever doubles `gemma4-31b` at depth (12.9 → 26.2). Expert pruning/merging does **not** speed M=1 decode (activated experts + attention dominate) — REAP/REAM win on KV headroom and footprint instead. **Quality:** the Qwen3.6 dense thinker leads agentic coding at **62.3%** on SWE-bench Lite (provisional — every historical cell ran under two since-fixed harness defects, scaffold context and output/thinking budgets, and is being re-rolled; see the bake-off section); the Qwen3.6 family and the Gemma 4 26B/31B reason and tool-call perfectly at true 256K, while the Coder-30B family hits a ~64K agentic ceiling. Receipts and the full ledger: [`docs/decode-levers.md`](docs/decode-levers.md), [Quality evals](#quality-evals). **Speculative decoding (new, 2026-09-27):** DSpark spec-decode on `qwen38` is a validated **1.4–1.5× single-user decode win on our CUDA verify** (60-tok 1.54×, 34K 1.47×, 49K 1.39×; up to 2.79× on predictable output; **1.21× at 250K actual — 58.5 vs 48.4 tok/s — with the true 256K window kept because fp8 KV leaves a 307K-token pool under the draft**) — R9700's 0.24×-at-49K Triton cliff does **not** reproduce on flashinfer verify ([receipt](benchmarks/quality/dspark-cuda-depth-ab-2026-09-27.md), [fp8-KV/250K receipts](benchmarks/quality/dspark-cuda-2026-09-27/)). It serves the bake-off's `qwen38` cycle from 2026-09-27 (`SPEC_DECODE=1`). Drafts exist for `qwen38` (DSpark), `qwen36`-family (DFlash), `coder-30b-eval` + `devstral` (EAGLE3); NGRAM (no draft) covers the rest. **The spec-decode chart above is current (shipped config, 2026-09-27); the decode/context PNGs predate it and the current stack — re-measuring the fleet with spec-decode + current settings is a GPU campaign queued for the next serving-only window (it cannot run while a bake-off lane holds the GPUs).**

## Status & next steps

**RESTARTED 2026-09-20 — the 256K re-roll runs from scratch as isolated `-v3` cells, served from the v0.5.20 OCI image.** Every cell before it is exposure-confounded: the rollout container ran `--network=host`, and on the two finished qwen38 opencode lanes **56 % / 53 % of instances fetched upstream** (27 % their own PR; the exposed group's patches overlap the gold patch at a median 100 % of added lines and resolve at 94.6 % / 91.2 % vs 69.5 % / 76.4 % for the rest — an effect size, not a cell; [receipt](benchmarks/quality/swebench-leak-audit-qwen38-netopen-2026-09-19.md)). Every v3 cell carries: **no network** (an in-container loopback bridge is the only address), refs stripped past HEAD, per-instance mounts staged under a neutral path, `/testbed` re-initialised to one `eval@local` commit, the task on **stdin** from a read-only file, the served **262144** window / template-max thinking / **32768** output budget, and the pinned server image id — gated per lane by [`audit_leakage.py`](evals/swebench/audit_leakage.py) `--require-isolation` (0 exposed + all four isolation proofs on 300/300) and per cycle by the Phase-0 request audit. Contract, proofs, docker-serving parity smoke (qwen38 / qwen36: bench within −1.6 … +3.5 % of bare metal, 2-instance isolated rollouts clean) and the four harness defects the boundary smoke caught before any v3 cell rolled: [`swebench-harness-isolation-2026-09-20.md`](benchmarks/quality/swebench-harness-isolation-2026-09-20.md). Earlier harness-defect receipts that still hold: context budget (2026-09-11, `9c31fff`), thinking + output budget (2026-09-13).

**Speculative decoding for the eval queue — DECIDED 2026-09-27: the campaign restarts as `-v4` with `qwen38` served under DSpark at the same true-256K window.** The 256K-vs-spec trade-off was a measurement artifact: with the preset's fp8_e4m3 KV the DSpark server's pool is **307,659 tokens at `--context-length 262144`** (draft weights cost 2.38 GB/card; a bf16-KV DSpark server only reaches 153K, which is where the "128K compromise" reading came from). Verified at 250,550 actual tokens: **58.5 tok/s vs 48.4 no-spec**, 119–130 at 60 tok, 69–80 at 130K, thinking + reasoning parser intact ([receipts](benchmarks/quality/dspark-cuda-2026-09-27/)). Spec is rejection-sampled, so resolved-rate stays unbiased at the bake-off's temp=1.0; the point of the flip is the **wall rate** — 17.7 % of the no-spec qwen38 opencode cell walled (53/300 rc 124, ≈29 h of its 96 h; 12 more finished past 1795 s of `rollout_seconds` because that figure spans the image build), and R9700 reports its first spec instances finishing in 210–658 s against a 1245 s no-spec mean. Mechanics: `SPEC_DECODE=1` opt-in on the `qwen38` preset (patch 099 + the flattened draft at `drafts/qwen38-dspark`), `SPEC_FOR=qwen38:1` in `run_all_cycles.sh` (the other presets stay no-spec until their opt-ins are validated at 256K/fp8 KV), `RUN_TAG=v4`, every cell's `meta.json` records the served config from `/get_server_info`. The finished v3 qwen38 cells (opencode 300/300, opencode-dcp 201/300) are kept as the **no-spec reference**. **Sample read (2026-09-27, 31 instance-matched opencode pairs, [receipt](benchmarks/quality/dspark-cuda-2026-09-27/v4-vs-v3-paired-sample.md)): spec walls 2 vs 3, empties 3 vs 3, median rollout 739 s vs 992 s (−25 %), mean 854 vs 1015 s, per-instance median speedup 1.16×; 0 errors.** The gain is entirely in the model-bound session phase (735 → 525 s median; image build + boot unchanged at ~200 s), the live lane shows accept length 3.0 / 88.7 tok/s median over 12K decode batches, and the one done→wall flip (`django-11742`) was a session mid-final-summary at 1800 s — a capture-at-wall case, not a spec effect. Spec therefore stays on for the whole v4 cycle; the full-300 cell is the number that lands in the bake-off table. **Full first pass (2026-09-30, 300/300, 292 instance-matched pairs — 8 v4 rows whose rollout image failed to build re-roll at Phase 4; [receipt](benchmarks/quality/dspark-cuda-2026-09-27/v4-vs-v3-paired-first-pass-300.md)): walls 33 vs 52 (11.3 % vs 17.8 %), empties 33 vs 53, median rollout 801 vs 1108 s (−28 %), mean 960 vs 1147 s, per-instance 1.18×, flips wall→done 36 / done→wall 17; session phase 537 vs 735 s with build + boot unchanged (193 vs 198 s); accept length 3.02 / 88 tok/s median over 124K decode batches, steady from first hour to last; 0 server errors; leak gate 0 exposed on 153 attempts.** Non-wall empties are the model's final think hitting the 32,768 output budget (`finish_reason=length`: 2 in v4, 1 in v3). Spec stays on for the remaining five v4 lanes; resolved-rate is compared only after scoring.

**Next steps, in order:**

1. **qwen38 v4 cycle (queue head, restarted 2026-09-27 under DSpark — see the spec paragraph above; the v3 no-spec reference cells: opencode ✓ 300/300 2026-09-24, 0 exposed, 53 walls / 1 `length`; opencode-dcp 201/300). opencode ✓ first pass 300/300 2026-09-30 (0 exposed, 33 walls / 2 `length`, 8 infra rows re-roll at Phase 4 — spec paragraph above) → opencode-dcp ✓ 300/300 2026-10-04 (0 exposed on 152 attempts, 35 walls vs the control's 33 on 290 paired IDs, median rollout 940 vs 802 s — the lane pays ~70 s of offline-npm plugin boot plus the full 120 s cleanup ceiling on 245/256 instances while its model sessions are 87 s *shorter*; DCP engaged on 44 % of instances and on exactly those IDs the lane walls 28 vs 18 and patches 67 vs 77; two finished, verified fixes lost at the wall to the capture path (`django-14999`, `sympy-17022`); 2 infra rows re-roll at Phase 4; [receipt](benchmarks/quality/dcp-lane-close-qwen38-v4-2026-10-04/README.md)) → little-coder ✓ 300/300 2026-10-07 (0 exposed on 224 attempts, **16 walls vs opencode's 30 on 274 paired IDs**, median rollout 669 vs 792 s, 1.11×; 69 % of sessions compacted at a 94K median peak vs 79 % / 112K; one finished fix lost at the wall to the capture path (`django-15252`); 23 infra rows — three more registry blips, item 4 — re-roll at Phase 4; the cell also surfaced a serving-side streaming defect, **patch 065**: an increment spanning two tool calls yields the inter-call `\n` ahead of the first call's closing deltas, and pi-ai 0.68 opens a phantom nameless tool call for them — 8.9 % of multi-call turns, 5 with the real call's arguments displaced; opencode and pi-ai 0.83 resolve by index and are immune, so the rtk A/B reads it as a control-only cost until the image carries 065 at the boundary; [receipt](benchmarks/quality/lc-lane-close-qwen38-v4-2026-10-07/README.md)) → little-coder-rtk rolling → prime → dcode → close it:** six-cell table, leader verdict, DCP and RTK A/B deltas (engagement receipts in `benchmarks/quality/{dcp,rtk}-engagement/`; the RTK lane is read through [`ab_lane_receipt.py`](evals/swebench/ab_lane_receipt.py) because the two little-coder lanes pin different pi versions, 0.68 control / 0.83 RTK); R9700's little-coder-v5 cell is the cross-rig pi-0.83 arm to set beside the RTK lane, never beside the 0.68 control. Verify on each lane's first instances that the `context budget:` lines and the Phase-0 audit receipt (`scaffold-audit.json`) show the served window / 32768 budget / no effort override, and, at lane close, that [`prompt_sawtooth.py`](evals/swebench/prompt_sawtooth.py) shows no compaction loops; the opencode lane's pre-score recall table is relayed ([receipt](benchmarks/quality/swebench-recall-audit-qwen38-opencode-v3-2026-09-24.md): 223/247 finished sessions recall the benchmark, duration tracks recall 4× across buckets, the trigger is the task contract itself — "Do not modify tests" + diff capture — not harness paths; walls blind in that cell, snapshotted from the opencode-dcp lane on); the scored `recall-audit.json` lands at Phase 7b. Then the historical re-roll (queue above), restore production, and the remaining new-lane (prime / dcode / DCP / RTK) cells for the receipted presets; `nemotron3-omni` last.
2. **GPU-profile experiments** — from the live-traffic profile of this cycle ([`benchmarks/qwen38-agentic-workload-profile-2026-09-10.md`](benchmarks/qwen38-agentic-workload-profile-2026-09-10.md)): server time is 86% decode, and a decode step is 57% INT4 weight streaming (78% of DRAM peak), 12% NCCL latency, 10% fp16 `lm_head`, 7% attention — 55% of the bandwidth roofline. Ranked by exposed time, each gated on needle + HumanEval + the tool probe:
   1. **DSpark speculative decoding on qwen38 — SHIPPED as the `SPEC_DECODE=1` opt-in (1.4–1.5× decode at short/mid depth, 1.21× at 250K, true 256K kept via fp8 KV).** CUDA A/B receipt: [`dspark-cuda-depth-ab-2026-09-27.md`](benchmarks/quality/dspark-cuda-depth-ab-2026-09-27.md) (60-tok 1.54×, 34K 1.47×, 49K 1.39×, 2.79× on predictable continuations — R9700's 0.24×-at-49K Triton cliff does not reproduce on flashinfer verify); fp8-KV / 250K receipts in [`dspark-cuda-2026-09-27/`](benchmarks/quality/dspark-cuda-2026-09-27/). In tree: patch 099 (`pp_proxy_tensors=` kwarg port, 3-gate clean), [`flatten_dspark_speculators_config.py`](scripts/specforge/flatten_dspark_speculators_config.py), the v4 harness wiring. **Remaining (boundary, GPU-bound):** `bench_regression.sh` baseline for the spec arm at the true KV cap + needle/HumanEval/tool-probe gate on the served config; ship the flattened draft to `mattbucci/Qwen3.8-27B-DSpark-sgl`; validate the other opt-ins at 256K/fp8 KV before adding them to `SPEC_FOR` — the `qwen36` DFlash opt-in still carries v0.5.1x flags (`SGLANG_ENABLE_SPEC_V2`, `--mamba-scheduler-strategy`) and a 32K window, EAGLE3 (`coder-30b-eval`, `devstral`) is validated at 16K only, NGRAM (patch 064, no draft) covers the REAP/REAM variants.
   2. **`lm_head` INT8 (or INT4) through Marlin** — the only fp16 weight left in the step (1.27 GB/rank, already at roofline); load-time W8 patch or calibration-recipe change. −4.6% / −7% per step.
   3. **PCIe host-read decisive tests** — 3–4 GB/s of GPU-initiated host reads during decode for no known purpose: 60 s each under `dmon` with `--disable-cuda-graph` and `NCCL_P2P_LEVEL` / `NCCL_WORK_FIFO_DEPTH` variants.
   4. Smaller levers: `in_proj_ba` fp16 gemv → real kernel (−2%), P0 memory clock via `-lmc` (~+1.5%), W4A8 Marlin for prefill (≤−4% e2e, activation-quant quality gate), prefill-only NCCL `Simple` protocol via a tuner plugin (−0.5% e2e). Custom allreduce (−6%) stays blocked on sm_86 graph capture (re-test `ENABLE_CUSTOM_AR=1` at each rebase). 350 W power cap = +5.5% only — rejected, the DIMMs are at ALARM HIGH.
3. **Harness decision at the qwen38 cycle boundary — one wall-hit convention for every scaffold.** Today only the dcode lane runs its scaffold under an inner timeout (diff + session captured at the wall); opencode / little-coder / prime are killed at 1800 s, so a wall hit is an empty diff by construction, while R9700's `docker_sandbox.sh` diffs the tree after the kill so a wall carries the partial edit. **R9700 has now sent the numbers (`f4407eb`, 2026-09-26): 42/124 opencode + 18/60 dcp v4 walls carry a non-empty, scorable diff — a third of their walls, none of ours (dcode aside) — and they propose both rigs capture at the wall from the next cycle, tagging each wall `model_timeout` so resolved-at-wall stays separable in the readout.** Recommendation: **adopt capture-at-wall** (dcode-style inner timeout on every scaffold + the `model_timeout` tag) at the qwen36-dense boundary — it recovers a third of walls, matches R9700 for cross-rig comparability, and the tag keeps "resolved by working" separate from "resolved at the wall"; the in-flight qwen38 cell stays internally consistent (empty-at-wall across all six of its scaffolds), so only qwen36-dense onward changes. Pick one rule for both rigs before the qwen36-dense cycle starts — inside a cycle the A/B pairs must keep the same rule, so this never changes mid-cycle. **Two more capture defects ride on the same change** (dcp lane close, [receipt](benchmarks/quality/dcp-lane-close-qwen38-v4-2026-10-04/README.md)): (a) a session that finishes its fix in the last ~3 minutes still scores empty, because the diff is printed only after the 120 s cleanup pass — 2/300 complete, model-verified fixes lost in the DCP cell (sqlite snapshots hold a single finished session), so the wall-hit path must `docker exec` the diff itself; (b) the `=== DIFF ===` stdout marker collides with the model's own `ps aux` (the inner script sits in `bash -lc` argv) — on a wall the extractor's `rfind` lands on the echoed copy and records the JSON event tail as a 20–200 KB "patch" (4 wall rows cycle-wide, all unapplyable, so resolve-rates are untouched; the readers now treat a non-diff `model_patch` as empty) — anchor the marker to a line start or `docker cp` the diff out instead. **Same boundary, sibling decision — exclude image-shipped untracked content from the captured diff.** The official `psf__requests-863` image carries an untracked, un-ignored `build/` (1 MB); the `git add -A && git diff --cached` capture sweeps it into the patch as 68 new-file hunks (874 KB) on top of the real fix, and every SWE-bench 4.1.0 apply method then fails (`git apply` "already exists", `--reject`, `patch --fuzz=5` reverses the real hunk) — the instance is unresolved-by-construction in 26 of 32 historical cells, uniformly across models (the six clean cells are runs where the model happened to delete `build/`), so cell comparisons stand but absolute scores carry a ≤1/300 floor. Fix candidate: write the image's pre-existing untracked paths to `.git/info/exclude` at re-init (`git status` clean for the model, capture skips them); R9700's sandbox is immune by construction (it `git add -A`s the base commit). Until then the isolation gate accepts `dirty=` equal to the image's own untracked count ([`image_untracked_baseline.json`](evals/swebench/image_untracked_baseline.json); `dirty_before=` reported per instance from the next lane on). **Same boundary, serving side — patch 065 into the live tree + serving image** (3-gate replay, then `bench_regression.sh` compare on the Qwen3-Coder-parser presets), and carry 063 + 065 upstream: sgl-project `main`'s detector still scores 2/8 on [`test_qwen3_coder_detector_orphan_tags.py`](scripts/eval/test_qwen3_coder_detector_orphan_tags.py) (R9700 reproduced it byte-for-byte on their tree), so index-blind clients hit it on every release.
4. **Harness lever — stop building per-instance rollout images.** Measured on the v3 lanes ([receipt](benchmarks/quality/gpu-utilization-harness-overhead-2026-09-26.md)): the GPUs are busy **73 % / 65 %** of the opencode / opencode+DCP lane's wall clock, the rest is one 5–10 min gap per instance, and the per-instance image build is **176 s** median (dcode conda env 68 s + layer export 60 s), 14–16 % of `rollout_seconds` — **≈ 289 h over the remaining ~5,900 instances**. R9700 runs exactly this design (hub image + bind-mounted toolchain) and measures **97.3 % / 97.8 % GPU-busy** on the same two lanes with a **5 s** scaffold boot (`f4407eb`) — a direct proof the ~32 % idle is all recoverable. Their opencode 1.18.25 reaches a session in 5 s, so the DCP 70 s offline-npm tax is **1.14.25-specific** — a scaffold bump would also close it (a version change, so boundary-only). Same boundary: vendor the opencode-dcp plugin so its loader resolves offline (the 70 s + 120 s tax above), or bump opencode. The per-instance build is also a **network dependency per instance**: every step (apt, nodejs.org, three npm installs, two `curl | sh` installers, conda) re-runs for each base image, and two independent registry blips cost the qwen38 opencode v4 lane **8 of its first 53 instances** — npm (2026-09-27 21:23–21:26, the DCP plugin step, 7 consecutive; the first build failures in 535 instances) and PyPI (2026-09-28 02:17, `quickjs-rs` momentarily had no distribution so `deepagents-code` hit `ResolutionImpossible`, 1) — and a third, the prime-agent installer at `app.primeintellect.ai` answering HTTP 500 for ~6 min (2026-10-04 17:09–17:15 PDT), failed **9 consecutive little-coder builds** (`django-11742`…`12113`) on a lane that never runs prime-agent — and the same installer plus npm E404s took 14 more on the rest of the little-coder lane — 33 instances lost to upstream blips so far in this cycle, each classed `infra_rollout_nonzero_rc` and re-rolled at lane close, never a model number. A hub image built once removes both the 176 s and the exposure. Bind-mount a host-built scaffold stack (`/opt/node`, npm prefixes, the A/B HOMEs, dcode's env, `rtk`) read-only into the official `swebench/sweb.eval.x86_64.<iid>` image — same versions, same `testbed` env, zero builds. Candidate for the qwen38 → historical boundary of the re-roll queue (a harness-version change recorded per cell), otherwise after the queue.
5. **Patch 058 observational cell** — a fresh full-300 opencode `devstral` cell post-058 vs the completed pre-058 cell (clean A/B on the empty-diff rate; ship receipt [`devstral-058-baseline`](benchmarks/quality/devstral-058-baseline-2026-07-19.txt)).
6. **Tooling for the MoE backlog:** port the Samsung SAIL REAM merge to the Gemma 4 arch (40–60 h; plan [`scripts/quantize/ream_gemma4_port_plan.md`](scripts/quantize/ream_gemma4_port_plan.md)); extend `run_reap.py` to the Gemma 4 parallel dense+MoE and Nemotron-H Mamba2-hybrid layouts (the fused-`Qwen3_5Moe` unfuse is done, 7/7).
7. **Calibration backlog (calibration device, not this box):** `North-Mini-Code-1.0-AWQ` → `Qwen3.6-35B-A3B-REAP-AWQ` → `gemma-4-26B-A4B-REAM-AWQ` → the `Qwen3.6-VL-30B-A3B` native/REAM/REAP trio with vision retained → `Qwen3-30B-Instruct-2507` native + REAP → `Qwen3.5-28B-A3B` native + REAM → Nemotron REAP/REAM; plus a Marlin-friendly `gemma4` requant (the sole fix for the Gemma fallback-kernel decode). Gated further out: a `Qwen3.8-Flash-Next` REAP/REAM shrink (blocked until sglang ships the QSA model class), the `diffusiongemma-26B-A4B` port, the official Gemma QAT-W4A16 compare. Recipes and rationale: [`docs/roadmap.md`](docs/roadmap.md).
8. **User-gated:** green-light the six prepared upstream-PR packages (`scripts/upstream-pr/`, 011 first — its 7-site hand re-port recurs every rebase); schedule the sudo-domain durable fix for the ~9–17 h docker-I/O kernel BUG (every multi-day cycle eats a crash).

### Known issues (open)

- **`nemotron3-omni` decode −12% at depth since v0.5.18** (93.3 → 82.4 tok/s @261,916; v0.5.20 measures 85.9, −7.9 % against the same kept baseline — the largest single move of that campaign, inside the band). Bisected to the flashinfer-side decode dispatch (0.6.15.post1 → 0.6.17); preset unchanged (still the fastest path), tripwire baseline deliberately kept at 93.3 so each flip measures against one reference; upstream-report candidate with receipts `benchmarks/regression/exp-nemotron-*.json`.
- **`qwen36-ream` × claw-code is 122/300 with ~30 instances that hard-fail in the scaffold** (`rc=3` GLIBC rollout landmine, not a model issue). claw-code is retired, so the cell stays as scored — read it with that discount.
- **Host reboots every ~9–17 h under sustained docker rollout I/O (kernel BUG).** Predictions on disk survive; `swebench-bakeoff.service` auto-resumes. Durable fix is user-gated (item 7). Forensic recipe: [`CLAUDE.md`](CLAUDE.md) → Operational Lessons.
- **opencode 1.14.25 `task` subagents hang headless on the `external_directory` permission prompt.** The main session is fine — `--dangerously-skip-permissions` covers it (`cd /tmp && …` finished 529/531 times in the closed v4 opencode cell; the two open parts are a test suite still running at the wall, `sympy-13915`, and the `matplotlib-23562` subagent hang) — but a subagent's ask has no responder under `opencode run`, so a subagent `bash`/`read` touching a path outside `/testbed` idles the instance to the 1800 s wall with the server empty (0/2 completed: `django-16820` `cat /etc/resolv.conf`, `matplotlib-23562` `cd /tmp && python -c …`, both `task` subagents, 1500 s+ open on a 120 s ceiling, server idle from that second). [`audit_wall_causes.py`](evals/swebench/audit_wall_causes.py) classes it from the session snapshot (an open leaf tool part that outlived its own execution ceiling — the permission check runs before the command; it judges the open part, not the newest one, since a step's parallel calls can leave a finished sibling on top) — 2 of the 33 rc-124 walls in the closed v4 opencode cell (first pass); 15 are the model mid-think, 14 the model still running a test suite inside the command's budget when the wall hit (model walls, not charged to the harness), 1 a step boundary, and 1 (`sphinx-8474`) a session that removed its own testbed environment; v3 walls are blind (no snapshots). Same opencode in both arms, so the paired read is unbiased, but every opencode-family wall count carries it. Fix = `permission: {external_directory: allow, doom_loop: allow}` in the rollout `opencode.json` (global config folds into every agent's ruleset), gated in Phase 0 on a subagent probe that touches `/tmp` — lands at the cycle boundary, never mid-cell.
- `check_awq_scales.py` reads native-AWQ format only — CT-format checkpoints crash its tensor reader (use a native-AWQ mirror or HF Range-fetch mode).

## Model Support

**Max ctx** = what the AWQ ship + 2× 24 GB actually serves end-to-end (validator + bake-off receipts). Every preset defaults to the full ctx; single-user tok/s measured at the listed context with **fresh prefill** (radix cache disabled).

| Model | Type | Max ctx | tok/s | Launch | HF + notes |
|-------|------|:-------:|:----:|:------:|:-------|
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE A3B (256 exp, VL) | **262K** | 209 (**121 @255K**) | `qwen36` | [`mattbucci/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ). Bake-off top tier (177/300 = 59.0% × opencode at 256K). |
| **Qwen3.6-REAM-A3B AWQ** | DeltaNet+MoE A3B (192 exp, VL) | **262K** | 209 (**144 @255K**) | `qwen36-ream` | [`mattbucci/Qwen3.6-REAM-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ). Vision tower grafted. Bake-off 177/300 = 59.0% × opencode at 256K. |
| **Qwen3-30B-Instruct-2507 REAM AWQ** | MoE A3B (96 exp) | **262K** | 200 (**69 @255K**) | `qwen3-ream` | [`mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ). REAM 128→96; text-only generalist. |
| **Qwen3.5-28B MoE REAP** | DeltaNet+MoE A3B (205 exp, VL) | **262K** | 210 (**144 @255K**) | `qwen35-moe` | Cerebras REAP of Qwen3.5-28B-A3B; thinking+vision. |
| **Nemotron-3-Nano-Omni-30B-A3B AWQ** | Mamba2-hybrid MoE A3B (128 exp, AVLM) | **262K** ✓ (5.25M pool) | 101 (**93 @255K**) | `nemotron3-omni` | [`mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ). int4; **6/6 caps** (basic+thinking+tool+vision+**video**+**audio** — the only audio ship). Serves via **`QUANT=moe_wna16`** (baked into the preset; the auto-detected awq_marlin path fails on this model's TP=2 shard shapes — mechanism in `patches/v0.5.14-rebase-status.md`), enabled by **patches 052** (non-gated squared-ReLU moe_wna16) + **053** (EVS video routing). Mamba2 O(1) recurrent → decode ~flat (101→93 @255K); beats R9700 FP8 (74.79/49.22). ⚠ **Agentic depth 131K (needs ≥8K token budget; 76K at 2K)**: ≥196K it spirals past even 8K and at 253K stops calling entirely — flat decode does not buy agentic 256K here. |
| **Qwen3-Coder-30B-A3B AWQ** | MoE A3B (128 exp) | **262K** | 200 (**69 @255K**) | `coder-30b` or `coder-30b-eval` | [`mattbucci/Qwen3-Coder-30B-A3B-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-AWQ). Two presets serve the same model; for short-ctx batch-decode benchmarks override `CTX=16384 MAX_RUNNING=32 ./scripts/launch.sh coder-30b`. |
| Coder-REAP-30B AWQ-Marlin | MoE A3B (96 exp) | **262K** | 200 (**69 @255K**) | `coder-reap-25b` | [`mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ) (R9700 in-house). |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **262K** ✓ (347K pool) | 54 (30 @64K, **13 @255K**; 26 @255K with the topk lever) | `gemma4-31b` | [`mattbucci/gemma-4-31B-AWQ`](https://huggingface.co/mattbucci/gemma-4-31B-AWQ). LM INT4, vision tower FP16, **KV fp8_e5m2**. Tool-use 1.0 → 258K true tokens, 5/5 caps incl. video. 347K pool via `--swa-full-tokens-ratio 0.05` + `MEM 0.92` + e5m2 FP8 KV (the only FP8 that compiles on the triton-forced path — sm_86 rejects e4m3). |
| **Gemma 4 26B MoE AWQ** | MoE A4B (103 exp, VL) | **262K** ✓ | 81 (**24 @255K**) | `gemma4` | [`mattbucci/gemma-4-26B-AWQ`](https://huggingface.co/mattbucci/gemma-4-26B-AWQ). **Tool-use 1.0 → 258K true tokens, 5/5 caps** incl. video. 652K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Gemma 4 12B Unified AWQ** | Omni, encoder-free | **262K** ✓ | 100 (**17.5 @255K**) | `gemma4-12b` | [`mattbucci/gemma-4-12B-AWQ`](https://huggingface.co/mattbucci/gemma-4-12B-AWQ). In-house int4 RTN-from-QAT. MMLU 77 / HE 93 / **tool-use 1.0 → 258K true tokens**, **5/5 omni** (vision + video ✓; gemma4_unified is native since transformers 5.12.1). 565K-token full pool via `--swa-full-tokens-ratio 0.0625`. |
| **Qwen3.6-27B Dense AWQ** | Dense + DeltaNet (VL) | **262K** (657K KV) | 69 (**47 @255K**) | `qwen36-dense` | [`mattbucci/Qwen3.6-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (R9700 self-cal). Bake-off leader at 187/300 = 62.3% (× opencode; superseded-harness cell, re-roll queued). |
| **Qwen3.8-27B AWQ** | Dense + DeltaNet (VL **+video**) | **262K** (652K KV @ M=1; 307K under the DSpark draft) | 71 (**48 @262K**) · `SPEC_DECODE=1` **125 / 90 @49K / 75 @130K / 58 @250K** | `qwen38` | [`mattbucci/Qwen3.8-27B-AWQ`](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ) (3090 self-cal 2026-08-18, GPTQ `thinking_vision_video`). MMLU **0.93** · HE **0.96** · needle **1.0** at server-verified 250,077 actual · caps **5/5** incl. video (LAB 0.16 — inside the DeltaNet-family spread 0.11–0.32 on a 56-question probe; watch item). 48 GDN + 16 full-attn layers; ships MTP weights (inert — main loader skips them). ⚠ `--max-running 1` is load-bearing: at 8 the KV pool collapses 652K→32K (18.7 GB ship: untied 248,320 vocab + BF16 lm_head, and DeltaNet state replicates per slot). |
| **Devstral-Small-2-24B AWQ** | Dense (VL) | **262K** ✓ (339K pool, patch 062) | 88 (52 @128K, **36 @262K**) | `devstral` | [`mattbucci/Devstral-Small-2-24B-AWQ`](https://huggingface.co/mattbucci/Devstral-Small-2-24B-AWQ). The canonical Devstral; built from [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512). FP8→BF16→GPTQ+tool-cal→AWQ. KV 202K → **339K on v0.5.18+062** (the loader-garbage fix recovered ~5.3 GB/rank): true 256K with headroom. |
| **Qwen3-VL-32B Instruct AWQ** | Dense (VL) | **131K** (model-card cap) | 63 (**35 @127K**) | `qwen3-vl-32b` | [`mattbucci/Qwen3-VL-32B-AWQ`](https://huggingface.co/mattbucci/Qwen3-VL-32B-AWQ) (R9700). 63→45→35 tok/s @ 1K/64K/127K. |
| Gemma 4 21B REAP AWQ | MoE (VL) | **262K** ✓ (653K pool) | 81 (**24 @255K**) | `gemma4-21b-reap` | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Cerebras-style expert prune of the 26B parent; same Gemma 4 serving flags (graphs ON + `--swa-full-tokens-ratio 0.0625` → 652K pool); tool-use 1.0/1.0 on the standard ladder. ⚠ HumanEval 0% (REAP prune lost coding — Quality Evals below): vision/chat ship, not code. |

Per-preset receipts for the current stack: `benchmarks/quality/*-v0520.json` (flip smoke) + `cap-*-v0520.json` (capability matrix); flip table in [`patches/v0.5.20-rebase-status.md`](patches/v0.5.20-rebase-status.md) (previous stack: [`v0.5.18-rebase-status.md`](patches/v0.5.18-rebase-status.md)).

### HuggingFace model zoo

Every `mattbucci/*-AWQ` row is built end-to-end from the linked upstream BF16 tensor — calibration, CT export, native AWQ conversion, scales audit, ship. **No 3rd-party pre-quantized AWQ used as a base.** ⚠ rows were calibrated on a 3rd-party pre-pruned BF16 (Cerebras / atbender) before the prune-ourselves rule; they're grandfathered live until in-house rebuilds replace them (rebuild paths in the [MoE coverage matrix](#moe-coverage-matrix)).

> **HF naming convention:** `mattbucci/<ModelName>-<format>` only. No descriptive suffixes (`-thinking-vision`, `-4bit`, `-native`, `-v2-fixed`) — the model card carries detail. `<format>` is `AWQ`, `AWQ-CT`, `GPTQ`, or `GPTQ-CT`. REAM/REAP are part of the model name, not a format suffix.

| Ship | HuggingFace | Upstream base |
|------|-------------|---------------|
| Qwen3.6-35B-A3B AWQ | [mattbucci/Qwen3.6-35B-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ) (native AWQ-Marlin) · [mattbucci/Qwen3.6-35B-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-CT) (compressed-tensors) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-REAM-A3B AWQ | [mattbucci/Qwen3.6-REAM-A3B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ) (native) · [mattbucci/Qwen3.6-REAM-A3B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-REAM-A3B-AWQ-CT) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Samsung SAIL `merge.py`, 256→192 experts) |
| Qwen3.6-27B Dense AWQ | [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) (native) · [mattbucci/Qwen3.6-27B-AWQ-CT](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ-CT) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (R9700 self-cal) |
| Qwen3.8-27B AWQ | [mattbucci/Qwen3.8-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (3090 self-cal, `thinking_vision_video` recipe) |
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

Each MoE base should ship in three flavors: **native** (no expert compression), **REAP** (Cerebras-style pruning, in-house via `scripts/quantize/run_reap.py`), **REAM** (Samsung SAIL merging, in-house via `scripts/quantize/run_ream_qwen3moe.sh`). All entries are self-calibrated AWQ-int4 from the upstream BF16 base. The missing cells are the calibration backlog (Next steps item 7; recipes in [`docs/roadmap.md`](docs/roadmap.md)).

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
| Devstral-Small-2-24B AWQ | 7.0 GB | ~40 KB (fp8 KV) | **339K** (MEM 0.90, patch 062) |
| Gemma 4 26B A4B MoE AWQ | 6.5 GB | ~12 KB (SWA) | **652K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 21B REAP AWQ | ~5 GB | ~12 KB (SWA) | **653K full / 41K swa** (262K ✓ @ ratio 0.0625) |
| Gemma 4 31B Dense AWQ | 7.7 GB | ~12 KB (SWA, fp8_e5m2) | **347K full / 17K swa** (`--swa-full-tokens-ratio 0.05` + MEM 0.92 + e5m2 KV) |
| Gemma 4 12B Unified AWQ | 5.4 GB | 15.3 KB full + 152.6 KB swa | **565K full / 35K swa** (262K ✓ @ ratio 0.0625) |
| Qwen3-VL-32B Dense AWQ | 10.0 GB | 24 KB | 131K (model-card cap) |

## Coding-eval bake-off (SWE-bench Lite)

Full 300-instance SWE-bench Lite, v3 Docker harness, one attempt per instance, single-user, `swebench==4.1.0` scorer. **Only full-300 cells are compared, and only cells whose scaffold ran at the served 262K window with the model's full thinking and a 32K output budget, under rollout isolation, count** — since 2026-09-11/13 the harness hands every scaffold the server's `max_model_len` and a uniform output budget per run, sends no reasoning-effort override, audits the first request of every scaffold before the cycle starts, and records both budgets per cell (`scaffold_context_window`, `scaffold_output_budget`); since 2026-09-20 the rollout container has no network (loopback bridge to the server only), refs past HEAD stripped, neutral mount staging, `/testbed` re-initialised to one neutral commit, the task on stdin from a read-only file, and the server runs from the pinned OCI image — every lane closes on `audit_leakage.py --require-isolation` ([contract](benchmarks/quality/swebench-harness-isolation-2026-09-20.md)). Current roster: opencode, opencode+DCP, little-coder, little-coder+RTK, prime, dcode (claw-code retired 2026-08-31; its column stays as a receipt). Harness mechanics, A/B-lane engagement receipts, and exclusions: [`docs/swebench-bakeoff.md`](docs/swebench-bakeoff.md).

| Preset | opencode | claw-code (retired) | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-dense` (Qwen3.6-27B Dense AWQ, thinking) | 187/300 = 62.3% ‡ | **165/300 = 55.0%** | 187/300 = 62.3% ‡ |
| `qwen36` (Qwen3.6-35B-A3B AWQ-Marlin, thinking) | 177/300 = 59.0% ‡ | **161/300 = 53.7%** | 177/300 = 59.0% ‡ |
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | 177/300 = 59.0% ‡ | 122/300 = 40.7% † | 150/300 = 50.0% ‡ |
| `qwen35-moe` (Qwen3.5-28B-A3B-REAP-AWQ, thinking) | 169/300 = 56.3% ‡ | 142/300 = 47.3% | 138/300 = 46.0% ‡ |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% ‡ | 107/300 = 35.7% | 74/300 = 24.7% ‡ |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% ‡ | 122/300 = 40.7% | 107/300 = 35.7% ‡ |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% ‡ | 109/300 = 36.3% | 76/300 = 25.3% ‡ |
| `devstral` (Devstral-Small-2-24B-AWQ) | 40/300 = 13.3% ‡ | — | — |
| `qwen38` (Qwen3.8-27B AWQ, thinking) | *re-rolling* | — | *re-rolling* (+ DCP / RTK / prime / dcode lanes) |

‡ **Superseded — re-rolling under the fixed harness.** Every cell in this table ran with `--network=host` (exposure-confounded — the scaffolds could and did fetch the upstream fix). On top of that, every little-coder cell ran at pi's 32K fallback with `reasoning_effort: medium` and a 16K output cap; every opencode cell ran with an 8192-token output cap (`qwen36-dense` also at a 32K window, `devstral` at 131K). The numbers stay as the last receipt (`benchmarks/quality/bakeoff-*-ctx32k.json`, `-ctx131k.json`, `-opencode-out8k.json`) until the re-rolled cell lands; the chart above already omits them. claw-code is retired and its budgets were never audited; its column is a receipt only.

**Read:** no cell has landed under the fixed harness yet — the table is the last receipts, and every historical read (the `qwen36` / `qwen36-ream` tie, the `qwen36-dense` lead, the ~9 pp REAM gap on little-coder) is withdrawn until its re-roll. All three defects hit every preset the same way, so the *ordering* may survive, but the truncation-caused empties were not uniform (a thinker at `xhigh` loses more to an 8K cap than a non-thinking coder), so no cross-family comparison is made from the superseded cells. `qwen3-ream` is excluded on model grounds (can't sustain agentic sessions — [verdict](benchmarks/quality/bakeoff-qwen3-ream-verdict-2026-07-17.md)). † `qwen36-ream` claw: ~30 instances hard-fail in the retired scaffold (Known issues). Per-cell receipts: [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/); failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling): [`patches/README.md`](patches/README.md).

## Quality evals

Every shipped AWQ model is scale-integrity clean (fleet `check_awq_scales.py --base` audit, [receipt](benchmarks/quality/fleet-integrity-audit-2026-05-31.json)) and passes every applicable capability probe (thinking / image / video / audio / tool) on the current stack (`cap-*-v0518.json`). Methodology, footnotes, and per-probe findings: [`docs/quality-evals.md`](docs/quality-evals.md).

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

† REAP prune lost coding (use `gemma4-31b` for code). ‡ partial 333-question LAB subset. ◊ non-coder text generalist on a code task.

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
./scripts/setup.sh                          # clone SGLang v0.5.20, apply patches/, create the conda env

./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE AWQ-Marlin — 256K, thinking+vision   (eval port :23334)
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — bake-off leader (superseded-harness cell; re-roll queued)
./scripts/launch.sh qwen38                  # Qwen3.8-27B AWQ — thinking+image+video
./scripts/launch.sh coder-30b               # Qwen3-Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AWQ (thinking+image+video)
./scripts/launch.sh devstral                # Devstral-Small-2-24B AWQ (tool+vision)
# full preset list: grep -E "^        [a-z][a-zA-Z0-9-]*[\|\)]" scripts/launch.sh

./scripts/serve_production.sh gemma4-31b    # persistent PRODUCTION endpoint on :30000 (start/stop/status/restart, detached, health-checked)
EXTRA='--decode-topk-pages 256 --decode-topk-page-size 64' ./scripts/serve_production.sh gemma4-31b   # deep-context topk lever (patch 059)

python scripts/eval/validate_capabilities.py --port 23334    # auto-skips thinking/vision/video per preset
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
BASELINE=save scripts/bench/bench_regression.sh <preset>     # lock a new perf level into the tripwire
```

Production on :30000 and the eval harness on :23334 don't collide, so a capability/bench sweep can run against :23334 while :30000 serves (the script refuses to start if another sglang server already holds the GPUs). Every preset carries an explicit `--tool-call-parser` matching its chat template (Qwen3-Coder + Qwen3.5/3.6/3.8 → `qwen3_coder`; Qwen3-VL / Qwen3-30B REAM → `qwen25`; Devstral → `mistral`; Gemma 4 → `gemma4`); thinking presets launch with `--sampling-defaults model`. Use `temperature >= 0.3` on the Qwen3 family — greedy decode loops.

## Stack

| Component | Version |
|-----------|---------|
| SGLang | v0.5.20 + 31 local patches (`/data/sglang-rebase-v0520`, env `sglang-v0520`; v0.5.18 tree + env kept for one-revert rollback) |
| PyTorch | 2.13.0 + cu130 |
| CUDA | 13.2 driver (595.71.05) / cu130 wheel |
| transformers | 5.12.1 (ships gemma4_unified natively; routes Mistral ckpts to MistralCommonBackend — countered by patch 057) |
| FlashInfer | 0.6.17 [cu13] |
| compressed-tensors | serving env pin; 0.15.1.dev in the separate `quant` calibration env |

**Patches** — 31 logical units in [`patches/`](patches/), applied idempotently by `setup.sh`: AWQ/CT int4 weight loading, Qwen3.5/3.6/3.8 enablement, Gemma 4 bring-up (26B MoE / 31B dense / 12B unified omni), Nemotron-3-Nano-Omni serving, MoE gelu coverage, kernel precision, sm_86 enablement, serving/agentic robustness. Each rebase is gated by the 3-gate pristine replay (`scripts/test_patch_gates.sh`) and a per-tokenizer-family A/B encode (`scripts/eval/tokenizer_ab_encode.py`), then a detached fleet validation campaign (`scripts/eval/flip_campaign.sh`). Narratives, the upstream-PR ledger, and per-flip receipts: [`patches/README.md`](patches/README.md).

**OCI image** — `Dockerfile` builds the CUDA/v0.5.20 stack without a GPU (pinned wheels, driver injected by the NVIDIA container toolkit); runs unprivileged with `SGLANG_SECURE_LAUNCH=1` (API keys from files, protected server options refused, NCCL on loopback). Build, run, and the security caveats vs the R9700 image: [`docs/oci-image.md`](docs/oci-image.md).

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
| GPU | 2× NVIDIA RTX 3090 (24 GB each) — NVLink bridge, `nvidia-smi topo -m` reports `NV4` (~56 GB/s aggregate); 260 W power cap (cooling profile, load-bearing) |
| CPU / RAM | AMD Ryzen 9 7900 (12C/24T) / 64 GB DDR5-6000 |
| Storage | 2× 2 TB NVMe (`/data` = models + caches) |
| OS / Kernel | Arch (EndeavourOS) / `linux-zen-p2p` 6.18 (pinned) · `nvidia-open-dkms` 595.71.05 · `amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction` |

Why each of those is load-bearing (NVLink/P2P boot args, the zen kernel, the cooling units under [`systemd/`](systemd/), Arch toolchain gotchas): [`docs/host-setup.md`](docs/host-setup.md).

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner + RDNA4 serving stack; we own evals + AWQ/INT4 + EAGLE3 draft training. Patches port both ways; model-behavior findings always do. **Current (2026-10-07, R9700 `10b4a22`):** they confirmed the Qwen3-Coder streaming-detector defects on their v0.5.20 tree with our unit test (2/8 → 8/8 with 063 + 065; sgl-project `main` is byte-identical, so there is no upstream release to wait for — an upstream fix is a boundary item below) and staged both as their 101/102 for the v5 boundary; their scaffolds all resolve tool deltas by index, so their little-coder-v5 cell is a clean pi-0.83 arm — **cross-rig, it pairs with our rtk lane, not the pi-0.68 control**. Their qwen38 v5 cycle runs beside our v4 (opencode-v5 300/300: 73 walls vs their v4's 126 under DSpark; opencode-dcp-v5 300/300; little-coder-v5 rolling), capture-at-wall is already their convention (41 of 73 walls carried a patch), and DSpark leaves benchmark recall unchanged while the 10–29-hit bucket finishes instead of walling. **Open asks to them:** (1) the host-side BF16→AWQ trap defuse + double-quant guard port before the next `gemma4-26b` recal (this box is clean — [manifest](benchmarks/models-manifest-letsrtfm-amd-2026-07-19.json)); (2) host the memory-marginal `Qwen3.6-35B-A3B` REAP prune on their 64 GB — the fused-`Qwen3_5Moe` unfuse tooling is ready to port to their `ream-patches/`.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal plumbing. No open asks either way.

## Repo layout

```
README.md                 # this file: direction, results, status + next steps, model/quality tables
docs/                     # long-form: decode-levers, swebench-bakeoff, quality-evals, roadmap, host-setup, oci-image
patches/                  # SGLang v0.5.20 patches (31) — narratives + rebase receipts in patches/README.md
benchmarks/               # charts, per-model regression JSON, lever receipts; quality/ = eval + bake-off receipts
evals/swebench/           # SWE-bench Lite v2 Docker harness (cycle driver, scaffolds, scorer, aggregation)
scripts/
  launch.sh / serve_production.sh / common.sh / setup.sh
  bench/ eval/ quantize/ specforge/ upstream-pr/ maint/ host-setup/
docker/                   # OCI image entrypoint + secure launcher
systemd/                  # cooling profile + bake-off auto-resume units
components/sglang/        # legacy vendored SGLang tree (historical; live serving trees are under /data/)
```
