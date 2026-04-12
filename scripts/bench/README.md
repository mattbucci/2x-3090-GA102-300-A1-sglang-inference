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
| `bench_long_context.py` | Context-length sweep via `/v1/completions` endpoint |
| `bench_regression.sh` | Regression test against stored baselines (>10% threshold) |
