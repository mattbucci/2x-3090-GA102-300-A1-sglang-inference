# Benchmarks — 2x RTX 3090

## Models

| Model | Single tok/s | Peak tok/s | Status |
|-------|-------------|------------|--------|
| Devstral-24B AWQ | — | — | Not yet tested |
| Coder-30B MoE AWQ | — | — | Not yet tested |
| Gemma 4 26B MoE AWQ | — | — | Not yet tested |
| Qwen3.5-27B AWQ | — | — | Not yet tested |

## Methodology

**Always measure TPOT** (time per output token) with `sglang.bench_serving`, never wall-clock time (which mixes prefill and decode).

### SGLang (primary)
1. **Context sweep**: Single-user, context 128 to max, 100 output tokens
2. **Throughput sweep**: Concurrency 1/2/4/8/16/32, 256 in / 256 out

### Regression Detection

After any configuration change:
```bash
./scripts/bench/bench_regression.sh devstral
```

Compares against `benchmarks/baselines.json` — flags >10% deviation. Save new baselines after verified changes:
```bash
BASELINE=save ./scripts/bench/bench_regression.sh devstral
```

### Quality Evaluation

```bash
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4
```

39-test suite (math, code, reasoning, vision, parallel) designed to catch TP=2 precision errors.

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `scripts/bench/bench_all_unified.py` | Primary: context + throughput sweep, JSON output |
| `scripts/bench/bench_comprehensive.sh` | SGLang wrapper (256 in / 256 out) |
| `scripts/bench/bench_quick.sh` | Fast 3-point A/B test |
| `scripts/bench/bench_long_context.py` | Context-length sweep |
| `scripts/bench/bench_regression.sh` | Regression testing |
| `scripts/bench/generate_charts.py` | PNG chart generation |
