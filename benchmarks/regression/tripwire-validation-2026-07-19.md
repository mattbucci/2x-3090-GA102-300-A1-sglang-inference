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
