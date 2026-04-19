# 2x RTX 3090 Inference Project

SGLang for 2x NVIDIA RTX 3090 (GA102-300-A1, 48GB total VRAM).

**All inference MUST use SGLang.** Uses AWQ_Marlin kernels for maximum INT4 performance. Patches may be needed for performance tuning.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support |
| [rules-for-agents.md](rules-for-agents.md) | VRAM budget, launch rules, quantization |

## Key Commands
```bash
scripts/setup.sh                       # Full setup (no patches needed)
scripts/launch.sh devstral             # Devstral 24B AWQ
scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
scripts/launch.sh qwen35               # Qwen3.5-27B DeltaNet AWQ
scripts/launch.sh qwen35-moe           # Qwen3.5-28B MoE REAP CT (205 experts)
scripts/launch.sh qwen3-ream           # Qwen3-30B REAM AWQ (96 experts, 197 tok/s)
```

## Critical Rules
- **SGLang only** — uses AWQ_Marlin kernels (sm_80+), patches may be needed for tuning
- **48GB VRAM limit** — 80B+ models do NOT fit
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- Always source `scripts/common.sh` + `activate_conda` + `setup_nvidia_env` before launching

## Optimization Target
- **Primary:** single-user **256K context** performance (decode tok/s, TPOT). Measure at long context first.
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win batch benchmarks.

## Calibration Rules
- **Preserve vision and thinking.** Past calibrations on this rig broke both (Qwen3.5-28B REAP lost structured `<think>` tags; community VL AWQs broke image handling). Do not repeat.
- **Calibration data requirements:**
  - Thinking-mode models (Qwen3.5, Qwen3-30B, Gemma4): include `glaiveai/reasoning-v1-20m` or `a-m-team/AM-Thinking-v1-Distilled` in the mix. Plain Open-Platypus silently strips reasoning.
  - Math/code models: add `AI-MO/NuminaMath-CoT` (~9.81% GPTQ accuracy gain over WikiText2).
  - Vision models: include multimodal image+text examples. Never calibrate from text-only loaders.
- **Post-calibration verification:** run a thinking-format health check and a vision sanity probe before publishing. A model that passes MMLU/HumanEval can still be silently broken on thinking/vision.
- **Multi-hour calibration runs are allowed** without user check-in — kick them off and keep working on other fronts.
- **Detach long-running jobs from the shell session.** `run_in_background: true` alone does NOT survive a session interrupt — we lost 7h 45min of Qwen3.5-28B calibration (layer 13/41) when the harness restarted. Launch via `setsid` + redirect all std streams + write PID to a file so the process gets PPID=1 and its own session ID. Verify: `ps -p $PID -o ppid=` must print `1`. Pattern:
  ```bash
  mkdir -p /tmp/<job>-logs
  setsid bash -c '<CMD> > /tmp/<job>-logs/run.log 2>&1 & echo $! > /tmp/<job>-logs/pid; disown' </dev/null >/dev/null 2>&1 &
  disown
  ```
  Any job expected to run > 30 minutes (calibrations, long benches, downloads of 50 GB+) must use this pattern.

## Workflow
- **Work autonomously.** User checks in periodically; README.md is the status document they read first.
- **Commit + push as progress is made** — small, self-contained commits, not one giant batch.
- **R9700 team collaboration:** `git fetch origin` on `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` to pull their commits; we can also edit their README to share 3090 findings. Patches are often portable.
- **Never stop to ask for confirmation.** If the user wants a redirect they'll interrupt with new signal.
