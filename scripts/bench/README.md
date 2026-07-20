# Benchmark Scripts

## Primary

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Primary benchmark — context sweep + throughput sweep, JSON output |

```bash
# Start model server, then:
python scripts/bench/bench_all_unified.py \
    --name "Devstral-24B AWQ" --port 23334 \
    --output benchmarks/devstral-24b-awq/results.json
```

Runs two sweeps:
1. **Context sweep** — Single-user, 100 output tokens, context 128 to max
2. **Throughput sweep** — Concurrency 1/2/4/8/16/32, 200 output tokens each

## SGLang Utilities

| Script | Purpose |
|--------|---------|
| `bench_comprehensive.sh` | Shell wrapper using `sglang.bench_serving` (256 in / 256 out) |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing |
| `bench_long_context.py` | Depth-verified context sweep (shells out to `sglang.bench_serving`; pins `--random-range-ratio 1`, records server-verified `actual_input_tokens`, self-caps at the live KV pool, flags degenerate points `invalid`/`depth_shortfall`) |
| `bench_regression.sh` | Throughput regression tripwire (>10% `tok_per_sec` drop at any of 3 depths ⇒ exit 1). Instrument = `bench_long_context.py` at 1024 / 32768 / deep-capped. `arm` mode serves each tripwire preset via `launch.sh` as-shipped; tokenizer read from the live server's `/get_model_info`. See schema v2 below. |

## Baseline schema v2 (`benchmarks/baselines.json`, fleet standard)

```json
{
  "_meta": {"schema": 2, "instrument": "scripts/bench/bench_long_context.py",
             "stack": "sglang-v0.5.15", "hardware": "2x RTX 3090 TP=2",
             "output_tokens": 100, "saved": "YYYY-MM-DD"},
  "<launch.sh preset>": {
    "1024":  {"tok_per_sec": 0, "tpot_ms": 0, "ttft_ms": 0, "actual_input_tokens": 0},
    "32768": {"...": "..."},
    "deep":  {"...": "...", "label": 255600}
  }
}
```

Rules: keyed by **launch.sh preset name** (never HF ids — presets are what we ship);
`deep` is the 262144 request self-capped by the live server (devstral lands ~196K —
that IS its baseline depth, recorded in `label`); points flagged `invalid` /
`depth_shortfall` / `actual_input_tokens` <95% of label are never saved and never
compared (reported NOT-COMPARED); `ttft_ms` drift is warn-only; `BASELINE=save` is a
deliberate act (after a receipted WIN or verified flip only) and merges per-depth so
a flagged rerun can't drop a previously-armed point. Compare-PASS on all 7 tripwire
presets is a pre-FLIP gate (see `patches/README.md` rebase checklist). Sister rigs
re-arm the same schema on their own instruments (M4 at its ~32K ceiling).
