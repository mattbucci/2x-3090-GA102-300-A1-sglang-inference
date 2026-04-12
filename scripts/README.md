# Scripts

## Quick Reference

```bash
# Launch a model
./scripts/launch.sh devstral             # Devstral-24B AWQ
./scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
./scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
./scripts/launch.sh qwen35               # Qwen3.5-27B AWQ

# Override defaults
./scripts/launch.sh devstral --context-length 16384 --port 8000
MODEL=/path/to/weights ./scripts/launch.sh coder-30b

# Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334

# Evaluate quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4
```

## Layout

| Path | Purpose |
|------|---------|
| `launch.sh` | Unified model launcher with presets and CLI overrides |
| `common.sh` | Shared NVIDIA environment (conda, NCCL) |
| `setup.sh` | Full setup: clone SGLang, create env, install |
| [`bench/`](bench/) | Benchmark scripts |
| [`eval/`](eval/) | Quality evaluation |
