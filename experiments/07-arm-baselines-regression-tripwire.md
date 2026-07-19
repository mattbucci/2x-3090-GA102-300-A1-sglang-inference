# 3090-D: Arm benchmarks/baselines.json as a depth-verified throughput regression tripwire (fleet-standard schema)

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | ~1-2 days total: ~3-4h script rework + one unattended evening (~5-7h) arming run + ~1h validation/docs |
| **GPU time** | 3090-box: ~5-7h serving occupancy (7 presets x [model load + 3 depth points], sequential, single-user) + ~20min negative-control run; no other rig GPUs |
| **Depends on** | None hard. Scheduling only: the arming evening must not overlap the resuming SWE-bench bake-off cells (Rule 1/2 — one GPU/docker workload at a time) or any calibration-device push that flips a preset mid-run. |
| **Provides to** | Every future 3090 rebase/flip (the pre-FLIP gate) and serving-path patch chain — automatic >10% decode tripwire replaces post-hoc forensics like b09882f; R9700 team: schema v2 + re-arm ask (their baselines.json is a 2026-04-12 relic with peak_throughput keys; their bench_regression.sh has the same stale 5-model map); M4 team: schema v2 + re-arm ask at its ~32K ceiling (their relic already matches the old 3090 parser keys, so schema v2 supersedes both); Fleet-audit queue item 4 in the 3090 README (closes it) |

## Objective

The rig flipped its SGLang default twice in 16 days (v0.5.14 flip 2026-06-26 via 98c0231, v0.5.15 flip 2026-07-12 via 15950d2) with full capability probes but zero throughput tripwire — benchmarks/baselines.json is literally `{}` and bench_regression.sh has been untouched since the initial-setup commit. Arm the tripwire with the fleet's depth-verified instrument so every future rebase/patch chain gets an automatic >10% decode-regression gate at short+medium+deep context, and define the versioned baseline schema all three rigs re-arm against (both sister baselines.json are 2026-04-12 relics with mutually incompatible keys).


## Background & receipts

- benchmarks/baselines.json is `{}`; git log shows it untouched since initial-setup commit 1ee94de (2026-04-12). bench_regression.sh (scripts/bench/bench_regression.sh) documents the full >10%-deviation workflow (THRESHOLD=10, BASELINE=save, exit 1 on regression) against it — the workflow exists, the data never did.
- bench_regression.sh's current instrument violates fleet invariants: 128-in/50-out random + multi@16 concurrency (lines 66-76) — no deep context, no --random-range-ratio 1, off-mission multi-user. Its BENCH_MODELS map is also stale ("Qwen/Qwen3.5-27B", "google/gemma-4-26B-A4B-it" — not the served mattbucci AWQ ships; served-model-name defaults to the PRESET name per launch.sh line 812, which is not a valid HF id, so tokenizer resolution via --model would fail anyway).
- The correct instrument already exists in-repo: scripts/bench/bench_long_context.py — pins --random-range-ratio 1, records server-verified actual_input_tokens, auto-caps the sweep at the live server's max_total_num_tokens + context_length, and flags degenerate points (invalid/depth_shortfall). Hardened by the 2026-07-14 depth-bug forensics (benchmarks/bench-depth-bug-2026-07-14.md, commit 594c059: all labeled-deep numbers had been shallow).
- Cost of no-baseline receipt: commit b09882f (2026-06-26) — nemotron3-omni's awq_marlin crash was mis-diagnosed as a "v0.5.14 regression" and users were told to roll back to v0.5.13; root cause was a preset gap (QUANT=moe_wna16 lived only in a shell override). A per-preset tripwire benching launch.sh presets as-shipped would have caught the preset/command divergence immediately.
- Cross-check data exists: README "Performance — single-user decode at 256K" table has post-depth-fix (2026-07-14+) server-verified deep receipts — qwen35-moe 144, qwen36-ream 144, qwen36 121, nemotron3-omni 93, coder-30b ~69, qwen36-dense 47, gemma4 24 @255K; devstral 42 @196K (202K pool). Arming numbers must reconcile with these.
- Sister relics (both last touched 2026-04-12): R9700 benchmarks/baselines.json (commit 1726ac6) uses keys {single_tpot_ms, single_throughput, peak_throughput}; M4's (commit a0b652b) uses the 3090 parser's {single_*, multi8_*} keys. Three rigs, two incompatible schemas, zero current data — hence the versioned fleet-standard schema in this spec.
- scripts/bench/README.md is stale on two rows (describes bench_long_context.py as "/v1/completions endpoint" — it now shells out to sglang.bench_serving; describes bench_regression.sh's old instrument) — fix while documenting the schema.


## Method

1. 1. Read the three source files end-to-end before editing: scripts/bench/bench_regression.sh, scripts/bench/bench_long_context.py, scripts/launch.sh (preset blocks + SERVED_NAME/PORT plumbing at lines ~693-813). Port default is 23334 (scripts/common.sh line 39).
2. 2. Rework scripts/bench/bench_regression.sh: keep the contract (BASELINE=save / compare / THRESHOLD=${THRESHOLD:-10} / exit 1 on regression / benchmarks/baselines.json path) but replace the 128-in+multi@16 instrument with a per-preset call to bench_long_context.py: `python scripts/bench/bench_long_context.py --port $PORT --name <preset> --contexts 1024 32768 262144 --output-tokens 100 --tokenizer <MODEL path from the preset block> --output benchmarks/regression/<preset>-$(date +%F).json` (the script self-caps 262144 to the honest deepest point from live max_total_num_tokens/context_length; devstral will cap ~196K — that IS its baseline depth). Delete the stale BENCH_MODELS map; key everything by launch.sh preset name. --tokenizer is mandatory: served-model-name defaults to the preset name, not an HF id.
3. 3. Tripwire preset set (one per distinct kernel/arch path, 7 total): qwen36 (fused Qwen3.5/3.6 MoE AWQ-Marlin), qwen36-dense (dense Marlin, bake-off leader), coder-30b (Qwen3Moe native-AWQ), qwen35-moe (DeltaNet hybrid), gemma4 (group-32 AWQ fallback-kernel path — most sensitive to kernel re-routing), nemotron3-omni (Mamba2-hybrid moe_wna16 — the b09882f model), devstral (dense Mistral + the tokenizer-backend canary family from patches 054/057). Encode the list in the script.
4. 4. Add an `arm` mode that sequentially serves each preset itself: for each preset — `scripts/launch.sh <preset>` (bench the preset AS SHIPPED, never a hand-rolled serve command — that divergence is the b09882f lesson), poll /health, bench, kill server, wait for VRAM drain, next. Pre-flight: verify no calibration/docker workload running (nvidia-smi + docker ps; Rule 1/2).
5. 5. Baseline file format (fleet-standard schema v2), written by BASELINE=save: top-level `_meta` {schema: 2, instrument: "scripts/bench/bench_long_context.py", stack: "sglang-v0.5.15", hardware, output_tokens: 100, saved: date} + per-preset objects keyed by launch.sh preset name: {"1024": {tok_per_sec, tpot_ms, ttft_ms, actual_input_tokens}, "32768": {...}, "deep": {... , "label": <actual capped depth>}}. Compare logic: FAIL (exit 1) if tok_per_sec at any depth drops >THRESHOLD% vs baseline; refuse to compare/save any point carrying invalid or depth_shortfall flags or actual_input_tokens <95% of label; ttft_ms recorded and reported as warn-only (prefill drift signal, no gate).
6. 6. Compare-path validation (no GPU): copy baselines.json to scratch, inflate one preset's deep tpot/deflate tok_per_sec by 20%, run compare mode against a stored run JSON, assert REGRESSION + exit 1; restore.
7. 7. Arming run (the unattended evening): detach via the setsid pattern from CLAUDE.md (>30min rule), `BASELINE=save scripts/bench/bench_regression.sh arm`. Per-run JSONs land in benchmarks/regression/ as receipts; kernel-BUG reboot mid-run resumes per-preset (completed presets' JSONs persist — make save idempotent/merge-per-preset).
8. 8. Cross-check before committing baselines.json: each preset's deep tok_per_sec within ~10% of the README perf-table receipt (qwen36 121, coder-30b ~69, gemma4 24, nemotron3-omni 93, qwen35-moe 144, qwen36-dense 47, devstral 42@196K). A >10% mismatch means instrument or serve-config divergence — investigate before saving, do not save a wrong baseline.
9. 9. End-to-end negative control (GPU, ~20min): serve coder-30b with --disable-cuda-graph appended (receipted 4.15x graphs win guarantees >10% deficit), run compare mode, confirm REGRESSION exit 1. Write receipt benchmarks/regression/tripwire-validation-<date>.md capturing both validations (steps 6+9).
10. 10. Wire into ops + docs: (a) add "run bench_regression.sh (compare, all 7) — must PASS before the FLIP commit" to the rebase gate checklist in patches/README.md (extends the boot-smoke + per-modality gate sequence); (b) rule in the script header + CLAUDE.md loop: re-run affected presets after any patch touching the serving hot path, and re-run BASELINE=save for a preset after any receipted WIN changes its perf (lock in the new level); (c) fix the two stale rows in scripts/bench/README.md and document schema v2 there; (d) push a short note to both sister READMEs (per cross-team convention): schema v2 definition + "your baselines.json is a 2026-04-12 relic in an incompatible format — re-arm on your own instrument at short/medium/deep" (R9700 has its own bench_regression.sh + stale 5-model map; M4 arms at its ~32K ceiling, e.g. depths 1024/8192/32768, oom_guard.sh mandatory).


## Baseline & instrument

The arming run itself is the first measurement: BASELINE=save via scripts/bench/bench_long_context.py (fixed instrument: --random-range-ratio 1, num-prompts 1, output 100, server-verified actual_input_tokens) at depths 1024 / 32768 / deep-capped per preset, cross-checked against the README 256K perf-table receipts (2026-07-14+ post-depth-fix numbers) before commit.


## Success criteria

- benchmarks/baselines.json contains _meta (schema 2) + all 7 presets x 3 depths; every point has actual_input_tokens >=95% of its label and no invalid/depth_shortfall flags (per-run receipts in benchmarks/regression/).
- Each preset's deep tok_per_sec reconciles within ~10% of its README perf-table receipt (listed in step 8); any larger mismatch resolved and explained before save.
- Tripwire demonstrably fires both ways: perturbed-baseline compare test AND the coder-30b --disable-cuda-graph negative control both produce REGRESSION + exit 1 (receipt: benchmarks/regression/tripwire-validation-<date>.md).
- Rebase gate checklist in patches/README.md names bench_regression.sh compare-PASS as a pre-FLIP requirement; scripts/bench/README.md documents schema v2 and no longer describes the stale instruments.
- Fleet-standard note landed in both sister READMEs (R9700 + M4) with the schema and the re-arm ask.


## Kill criteria

- A preset that cannot produce a valid deep point in 2 attempts (persistent invalid/depth_shortfall) gets a one-line finding + receipt under benchmarks/regression/ and is baselined at its remaining valid depths — one bad preset never blocks arming the other six.
- If the box kernel-BUG-reboots twice during serving-only bench runs (no docker rollout I/O in flight), stop and record as a NEW finding — it widens the known docker-I/O trigger surface and outranks the arming task.
- Hard cap ~10h GPU occupancy on the arming evening: commit whatever preset baselines are complete + a coverage note; do not loop retries unattended.
- If step 8 cross-check reveals the README perf-table numbers themselves irreproducible (>10% off on 3+ presets with a healthy instrument), stop arming and escalate — that is a live undetected regression, the exact event this tripwire exists for; document with receipts before saving anything.


## Deliverables

- Reworked scripts/bench/bench_regression.sh (depth-curve instrument, arm mode, schema-v2 save/compare, stale BENCH_MODELS map deleted)
- Armed benchmarks/baselines.json (schema v2, 7 presets x 3 depths + _meta)
- Per-run receipts: benchmarks/regression/<preset>-<date>.json (one per preset per run)
- Validation receipt: benchmarks/regression/tripwire-validation-<date>.md (perturbation test + --disable-cuda-graph negative control, both exit 1)
- Docs: schema v2 + corrected instrument rows in scripts/bench/README.md; pre-FLIP gate line in patches/README.md rebase checklist; WIN-resave rule in CLAUDE.md loop
- Sister-README notes in /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/README.md and /home/letsrtfm/AI/m4-sglang-inference/README.md (schema v2 + re-arm ask)


## Constraints

- Rule 1 / Rule 2: no concurrent calibration, eval rollout, or scoring during any bench serve — pre-flight nvidia-smi + docker ps; one preset served at a time.
- Bench the launch.sh preset exactly as shipped (env via common.sh + activate_conda + setup_nvidia_env); never a hand-assembled serve command — preset/command divergence is the b09882f failure mode.
- Detach the arming run via the setsid pattern (>30min rule); per-preset receipts must survive a kernel-BUG reboot and resume with completed presets skipped.
- Fleet measurement invariants: --random-range-ratio 1, single-user M=1, server-verified actual_input_tokens at true depth, degenerate points (ttft=0 / shortfall) never saved or compared.
- 260W power cap + gpu-fan-curve.service stay active (load-bearing for sustained runs). Do not touch presets or patches — this task changes only bench/doc surfaces.
- Baseline re-save is a deliberate act: only after a receipted WIN or a verified flip, never automatically on a PASS.


## Risks

- Run-to-run variance near the 10% threshold at short depth could flap the gate — mitigation: warmup is built into bench_long_context.py; if a preset flaps, add one repeat-and-median at the flapping depth rather than widening THRESHOLD.
- bench_regression.sh has plausibly never run end-to-end (stale model map, untouched since 1ee94de) — budget debugging time in the rework, don't trust its untested plumbing.
- devstral tokenizer loading in bench_serving could hit the MistralCommonBackend routing quirk (patch 057 lesson) — random-dataset benching only needs token counting, but verify the actual_input_tokens field looks sane on devstral's first run.
- The arming evening spans ~5-7h on a box with ~9-17h kernel-BUG uptime windows — resume logic (step 7) is load-bearing, not optional.
- Deep-point prefill at ~255K adds minutes per preset; a hung prefill inflates the evening — bench_long_context.py's 1800s subprocess timeout bounds each point.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
