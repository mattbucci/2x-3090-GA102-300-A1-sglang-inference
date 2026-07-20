# 3090-K: Extend the regression tripwire to the production presets, characterize variance vs the 10% gate, and attach a GPU-telemetry sidecar (attested baselines)

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | ~1-1.5 days total: ~2-3h sidecar + script work (no GPU) + one ~2-3h detached arming window + ~1-2h variance repeats in bake-off gaps + ~1h docs. Power rider: ~30-60min ATTENDED, MATT-GATED (root). |
| **GPU time** | ~4-6h serving occupancy: 3 new presets x [load + 3 depths] ≈ 2-3h; variance repeats (N≥5 x {1024, deep} x 3 presets) ≈ 1.5-2.5h; telemetry rides along at zero marginal GPU cost. Rider adds ~1h attended if approved. |
| **Depends on** | 3090-D (**done 2026-07-19**) — extends its armed schema-v2 tripwire; must NOT overwrite its 7 baselines. No other hard deps; scheduling per Rule 1/2. |
| **Provides to** | Every future flip: the pre-FLIP gate widens 7→10 presets incl. the production bring-up preset; the committed flap policy de-risks the first real gate exercise; R9700 + M4: telemetry-attestation pattern as an optional run-JSON extension of the already-pushed schema v2; Matt: a measured power-cost curve (if the rider is approved) for the 260W-cap decision |

## Objective

Close the three gaps doc 07 shipped around: (1) the armed tripwire does not cover gemma4-31b — simultaneously the production bring-up preset and the preset whose decode profile moved most recently — nor the coder-reap-25b / qwen3-ream ships; (2) run-to-run variance vs the 10% gate is named in doc 07's own risk list but never measured, so the first real pre-FLIP gate exercise runs blind; (3) every perf receipt in the repo was recorded with zero GPU telemetry on a rig deliberately power-capped to 260W/card — no receipt can attest whether its number was taken throttled or at full clocks. Arm the three missing presets with thermally-attested baselines, commit a measured flap policy, and ship the telemetry sidecar as standard instrument equipment.

## Background & receipts

- bench_regression.sh:72: `TRIPWIRE_PRESETS=(qwen36 qwen36-dense coder-30b qwen35-moe gemma4 nemotron3-omni devstral)` — 7 armed 2026-07-19 (commit fb6b199; receipts benchmarks/regression/*-2026-07-19.json + tripwire-validation-2026-07-19.md, negative control fired). benchmarks/baselines.json verified: `_meta` + exactly those 7 keys — no gemma4-31b, no coder-reap-25b, no qwen3-ream.
- gemma4-31b is the production bring-up preset: serve_production.sh (89d0b62, 2026-07-20) usage examples + validation ran on it, as does the CLAUDE.md Key Commands line — and its decode profile is the repo's most recently moved (patch 059 a1a63fc: opt-in topk 12.9→26.2 tok/s @261,916, README:74; README Quick Start :169 recommends exactly that EXTRA on production). The preset most exposed to silent regression has no baseline.
- coder-reap-25b (bake-off ship: 41.7% opencode, README:92) and qwen3-ream (perf-table ship: 69 @255K, README:257) also uncovered. The M=1-twins fact (30B-A3B trio ≈69 @255K, README:43) makes their deep rows a free preset-divergence cross-check — the b09882f class the tripwire exists for.
- Variance is named-but-unmeasured: doc 07's risk list (experiments/07-arm-baselines-regression-tripwire.md:87) calls gate-flap the mitigation-if-it-happens case; the gate has never blocked or passed a real flip.
- Zero telemetry in any receipt: the rig runs a load-bearing cooling profile (gpu-cooling.service: 260W from 350W stock + 75% fan floor; README:219-237 — DDR5 SPD ALARM HIGH correlated with stock 350W) yet no benchmark records power/clocks/throttle-reasons. NVML is already on the host: systemd/gpu-cooling.sh:70 verifies `import pynvml` for the fan services; gpu-fan-curve.py drives nvmlDeviceSetFanSpeed_v2 continuously.
- Instrument bounds: bench_long_context.py 1800s subprocess timeout per point (line 46); arm mode self-caps deep from the live server and BASELINE=save merges per-preset (idempotent resume) — bench_regression.sh header.
- bench_serving's deterministic default prompt seed makes repeats prompt-identical within a serve session → repeat 1 (post-flush) is cold, repeats 2+ are radix-warm. That determinism is load-bearing for the cold/warm split below.

## Method

1. Read scripts/bench/bench_regression.sh + scripts/bench/bench_long_context.py end-to-end; read the three preset blocks (launch.sh gemma4-31b :325-355, coder-reap-25b :230, qwen3-ream); read benchmarks/regression/tripwire-validation-2026-07-19.md (the self-test step 6 re-fires).
2. Author scripts/bench/gpu_telemetry.py: pynvml poller, 1Hz per GPU → CSV (ts, gpu, power_w, sm_mhz, mem_mhz, temp_c, pstate, throttle_reasons bitmask via nvmlDeviceGetCurrentClocksThrottleReasons); `--summarize <csv>` emits p50/p95/max per field + %-samples at the power cap (≥255W) + throttle-reason histogram. Env check: pynvml importable in sglang-v0515; if absent, `pip install nvidia-ml-py` (pure NVML client, no CUDA dep).
3. Wire `--telemetry` into bench_long_context.py: spawn the poller as a sidecar for the sweep; on exit fold the summary into the run JSON under a `telemetry` key (additive; verify the bench_regression.sh compare path ignores unknown keys — baselines.json stays schema 2). Zero-perturbation gate: A/B one short cell with/without the sidecar; >1% tok/s shift → drop to 0.2Hz; still >1% → ship default-off with a finding.
4. Extend TRIPWIRE_PRESETS with `gemma4-31b coder-reap-25b qwen3-ream` (7→10) and arm in a bake-off gap, setsid-detached: `BASELINE=save scripts/bench/bench_regression.sh arm gemma4-31b coder-reap-25b qwen3-ream` with --telemetry threaded through — the repo's first thermally-attested baselines. Anchors BEFORE save: gemma4-31b deep ≈12.9-13 tok/s graphs-on (059 verdict baseline + README perf table); coder-reap-25b / qwen3-ream deep within ~10% of the armed coder-30b twin (69.6). A >10% mismatch = investigate, do not save (doc 07 step-8 rule).
5. Variance characterization: 3 kernel classes x 2 depths x N≥5 — coder-30b (Qwen3Moe native-AWQ), qwen36 (fused AWQ-Marlin MoE), gemma4-31b (triton-forced group-32 fallback) at 1024 and deep, one serve session per preset (AS SHIPPED). Repeat 1 cold after `curl -s -X POST localhost:23334/flush_cache`, repeats 2-5 warm; record the two classes separately. Compute per-cell CV% and (max-min)/median for tok_per_sec + tpot_ms; replay each run JSON through `scripts/bench/bench_regression.sh check <preset> <run.json>` to observe real gate behavior against the live baseline.
6. Flap-policy decision (committed either way): if warm-class spread >5% (half the gate) on any cell → implement best-of-3-median in the compare path AND re-fire the offline perturbation self-test (doc 07 step-6 pattern, receipt updated); else keep single-pass and write the threshold math (observed max spread vs the 10% gate) into the bench_regression.sh header. Either way: benchmarks/regression/variance-characterization-<date>.md with the CV% table + decision.
7. Standing gate-transcript rule: one line in the patches/README.md rebase checklist — the first live pre-FLIP gate run (now 10 presets) captures its full transcript into benchmarks/regression/ as a receipt (the gate has never gated a real flip).
8. Severable ATTENDED power rider (MATT-GATED — root `nvidia-smi -pl`): deliver scripts/maint/power_cost_curve.sh ready-to-run — for PL in 260/290/320W: set the limit, run one deep-decode cell + one cold 262K prefill on coder-30b and gemma4-31b with telemetry, watch `sensors` for DDR5 SPD ALARM lines; restore 260W and verify via `nvidia-smi -q -d POWER`. Must NOT test 350W (stock-350W ALARM/crash correlation, README:219). If sudo is agent-gated, hand the script to Matt; steps 1-7 complete regardless.
9. Sister notes (read-only courtesy, README push): the telemetry block as an optional run-JSON extension of the already-pushed schema v2 + a "thermally attest your baselines on your own instrument" ask (R9700: rocm-smi; M4: powermetrics — their choice).

## Baseline & instrument

The armed 7-preset benchmarks/baselines.json (schema 2, saved 2026-07-19) is the fixed reference — this task ADDS three preset objects and never rewrites existing rows. Instrument stays bench_long_context.py via bench_regression.sh arm/check (self-capped deep, server-verified actual_input_tokens, degenerate points never saved). Telemetry is an observe-only sidecar, gated zero-perturbation (step 3) before it touches any armed run.

## Success criteria

- benchmarks/baselines.json holds 10 presets under schema 2; every new point has actual_input_tokens ≥95% of label and no invalid/depth_shortfall flags; the original 7 preset objects unchanged (merge-save adds keys only).
- gemma4-31b deep within 10% of the 12.9-13 tok/s graphs-on receipt class; coder-reap-25b + qwen3-ream deep within ~10% of the 69.6 coder-30b twin — or the divergence explained with receipts before save.
- Every new baseline's backing run JSON in benchmarks/regression/ carries a telemetry block incl. throttle-reason summary + %-at-cap; gpu_telemetry.py committed with the zero-perturbation gate receipt.
- Variance report: N≥5 per cell across 3 presets x 2 depths, cold/warm classes separated, CV% table, and an explicit committed flap policy (median-of-N implemented + self-test re-fired, or single-pass kept with threshold math) in variance-characterization-<date>.md + the bench_regression.sh header.
- patches/README.md rebase checklist names the 10-preset gate-transcript rule.
- Rider, only if run: tok/s + cold-prefill seconds at 260/290/320W with max DDR5 SPD temp per cell; cap verified restored to 260W.

## Kill criteria

- A new preset failing 2 arming attempts (persistent invalid/depth_shortfall or unstable serve) → one-line finding + receipt, baseline at its remaining valid depths (doc 07 rule). qwen3-ream especially: it is agentic-excluded on model grounds — do NOT debug model quality here, only serve/bench stability.
- Zero-perturbation gate fails even at 0.2Hz → sidecar ships default-off + finding; never attach a perturbing instrument to the tripwire.
- Variance GPU budget capped ~2.5h: a cell flapped by a visible external confound (bake-off resume, telemetry-visible thermal event) is discarded and re-run ONCE; persistent confounds → publish with N achieved + note.
- Rider: any DDR5 SPD ALARM at 290/320W → abort the rung, restore 260W immediately, record; never proceed to the next rung after an ALARM.
- Kernel-BUG reboot mid-arm: resume via per-preset merge-save; two reboots during serving-only runs = NEW finding that outranks this task (doc 07 rule).
- Must NOT widen THRESHOLD to solve flap — gate-policy changes only via the step-6 decision.

## Deliverables

- scripts/bench/gpu_telemetry.py + `--telemetry` wiring in bench_long_context.py (+ zero-perturbation gate receipt under benchmarks/regression/)
- scripts/bench/bench_regression.sh: TRIPWIRE_PRESETS 10 + flap-policy header line (+ median-of-3 compare path if triggered)
- benchmarks/baselines.json: +gemma4-31b, +coder-reap-25b, +qwen3-ream (schema 2); backing run JSONs + telemetry CSVs under benchmarks/regression/
- benchmarks/regression/variance-characterization-<date>.md (CV% table, cold/warm split, committed decision)
- Docs: patches/README.md gate-transcript rule; scripts/bench/README.md telemetry-block documentation
- scripts/maint/power_cost_curve.sh (prepared; receipt benchmarks/regression/power-cost-<date>.md only if Matt approves/runs)
- Sister README notes (R9700 + M4): optional telemetry extension + thermal-attestation ask

## Constraints

- Rule 1/2 preflight (arm mode already refuses concurrent workloads); setsid-detach the arming window (>30min rule); bench every preset AS SHIPPED via launch.sh (the b09882f lesson).
- BASELINE=save is deliberate and merge-per-preset — must NOT overwrite the armed 7; M=1 fleet invariants throughout (--random-range-ratio 1, server-verified depth, degenerate points never saved).
- 260W cap + fan services stay active for ALL of steps 1-7; only the MATT-GATED rider touches the power limit, and it restores + verifies 260W before exit.
- Telemetry is observe-only: no serve-flag changes, no preset edits, no patches — bench/doc surfaces only (same surface discipline as doc 07).
- Read-only courtesy to sister repos: README notes only.

## Risks

- pynvml API drift (throttle-reasons unsupported on the open driver) — catch NVMLError per field and record "unsupported" instead of crashing; power/clocks/temp are the load-bearing fields.
- Sidecar perturbation at 1Hz — measured, not assumed (step-3 gate); NVML reads are host-side and should cost ~nothing.
- Deep variance repeats are the GPU-budget hog (a 262K point per repeat) — the warm repeats radix-hit the prefill, which is exactly the cold/warm structure being characterized; keep the classes separate so the cold N=1 never averages into warm spread.
- qwen3-ream's agentic instability (runaway loops) is a rollout property random-prompt benching never exercises — serve-stability risk low; the kill bullet bounds it.
- The first live 10-preset gate may surface honest >10% regressions on presets nobody watches — that is the tool working; triage belongs to the flip that finds it, not this task.

---
*Vetted 2026-07-20: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
