# Regression-tripwire validation — 2026-07-19

Lane 3090-D, spec steps 6 + 9. Instrument: reworked `scripts/bench/bench_regression.sh`
(schema v2; depth points 1024/32768/deep via `bench_long_context.py`).

## Step 6 — compare-path validation (no GPU) — DONE

Synthetic fixtures (README-receipt-flavored qwen36 numbers: 145/135/121 tok/s at
1024/32768/255600), `check` mode against a scratch baselines file:

| case | expected | got |
|---|---|---|
| save from clean run | 3 depths saved, `_meta.schema=2` | ✓ |
| compare identical run | PASS exit 0 | ✓ |
| deep -21.5% run | `deep(255600) … [REGRESSION]` + exit **1** | ✓ |
| deep flagged `invalid+depth_shortfall` | `NOT-COMPARED … rerun this point`, shallow depths still compared, no false pass | ✓ |
| save from flagged run | flagged point refused; **prior deep baseline preserved** (merge-per-depth); 1024/32768 updated | ✓ |

The merge-per-depth save behavior was added during validation: the first
implementation replaced the whole preset entry, so a save from a partially-flagged
rerun silently dropped the previously-armed deep point.

## Step 9 — end-to-end negative control (GPU, ~20 min) — PENDING

Runs on the arming evening: serve coder-30b with `--disable-cuda-graph` appended
(receipted 4.15× graphs win ⇒ guaranteed >10% deficit), compare mode must print
REGRESSION and exit 1. Appended here when run.

## Step 9 — RESULT (2026-07-19): PASS

Arming run: all 7 presets armed in one detached evening pass, zero flagged
points, every deep point at true depth (261,916 actual; devstral self-capped at
201,687 = its 202K pool). Cross-check vs README perf-table receipts: all 7
within 3% (qwen36 +0.8%, dense +2.6%, coder-30b +0.9%, qwen35-moe +0.3%,
gemma4 +0.8%, nemotron3-omni +0.3%, devstral +1.7%).

Negative control: `EXTRA_ARGS="--disable-cuda-graph" launch.sh coder-30b` +
compare mode:

    coder-30b/1024:          200.8 -> 34.1 tok/s (-83.0%) [REGRESSION]
    coder-30b/32768:         155.8 -> 33.9 tok/s (-78.2%) [REGRESSION]
    coder-30b/deep(261916):   69.6 -> 33.4 tok/s (-52.0%) [REGRESSION]
    exit=1 (ttft WARNs fired as warn-only)

Tripwire fires both ways (perturbed-baseline offline + live mis-config) — armed.

Ops scar from the first arming attempt: `common.sh` redefines `SCRIPT_DIR`, so
the instrument path must be pinned BEFORE sourcing (script now asserts the
instrument exists before launching any server — a path bug can never burn 7
model loads again).
