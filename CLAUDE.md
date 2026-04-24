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
- **Preserve thinking + image + video + audio.** Past calibrations on this rig broke thinking (Qwen3.5-28B REAP lost `<think>` tags) and image (community VL AWQs broke alignment). Gemma 4 and Qwen3.5/3.6 also support **video and audio** natively — both easy to miss if your calibration recipe is image+text only. See [Gemma video docs](https://ai.google.dev/gemma/docs/capabilities/vision/video); Qwen3.5/3.6 handle video via `<|vision_start|><|video_pad|><|vision_end|>` in the chat template.
- **Calibration data requirements:**
  - Thinking-mode models (Qwen3.5, Qwen3-30B, Gemma4): `glaiveai/reasoning-v1-20m` or `a-m-team/AM-Thinking-v1-Distilled`. Plain Open-Platypus silently strips reasoning.
  - Math/code models: `AI-MO/NuminaMath-CoT` (~9.81% GPTQ accuracy gain over WikiText2).
  - Image models: `liuhaotian/LLaVA-Instruct-150K` or equivalent image+text pairs.
  - **Video models (Gemma4 all variants, Qwen3.5/3.6):** include video-text pairs (e.g. `lmms-lab/LLaVA-Video-178K`, `ShareGPT4Video`, or frame-sampled subsets). Never calibrate video-capable models without video samples — the temporal-attention weights drift otherwise.
  - **Audio models (Gemma 4 all variants):** include audio-text pairs (e.g. `mozilla-foundation/common_voice`, `google/covost2`). Audio preprocessor_config.json must ship with the checkpoint — M4 team has a known bug where community checkpoints omit it.
- **Post-calibration verification:** run ALL applicable probes before publishing — thinking-terminates, image-caption, video-summary (frame or clip), audio-transcription. A model that passes MMLU/HumanEval can still be silently broken on any single modality.
- **Multi-hour calibration runs are allowed** without user check-in — kick them off and keep working on other fronts.
- **Detach long-running jobs from the shell session.** `run_in_background: true` alone does NOT survive a session interrupt — we lost 7h 45min of Qwen3.5-28B calibration (layer 13/41) when the harness restarted. Launch via `setsid` + redirect all std streams + write PID to a file so the process gets PPID=1 and its own session ID. Verify: `ps -p $PID -o ppid=` must print `1`. Pattern:
  ```bash
  mkdir -p /tmp/<job>-logs
  setsid bash -c '<CMD> > /tmp/<job>-logs/run.log 2>&1 & echo $! > /tmp/<job>-logs/pid; disown' </dev/null >/dev/null 2>&1 &
  disown
  ```
  Any job expected to run > 30 minutes (calibrations, long benches, downloads of 50 GB+) must use this pattern.

## Workflow (RECONFIRMED 2026-04-24)
- **Work autonomously. Never stop to ask for confirmation.** User checks in periodically by reading the README and will interrupt with new ideas or redirects. Max effort is the default.
- **Multi-hour calibrations are pre-authorized.** Downloading 50-70 GB BF16 bases + running 10-13h GPTQ calibrations does NOT need user check-in. Detach via `setsid` pattern and keep working on other fronts.
- **Note the next step in the README before starting it** — user can interject if they see a better path. Commit + push as progress is made (small self-contained commits, not one giant batch). Every commit should stand on its own.
- **Keep README.md clean.** It is the single source of truth. Once a ship supersedes a debugging narrative, trim the narrative. Reader should see current status + known issues + next step without scrolling.
- **Carry forward these design principles** — user has re-emphasized them and they should not drift:
  - **REAP / REAM pruning is preferred** for long-context MoE at 3090/R9700 scale. Dropped rare experts fit 256K in 48 GB VRAM where full experts can't.
  - **Chat templates are load-bearing.** Wrong BOS/EOS, missing `<think>` handling, or reasoning stripped from calibration data silently destroys quality. Inspect `chat_template.jinja` and validate thinking tags on every new model before claiming ship.
  - **Calibration data must cover all live modalities** (thinking + image + video + audio as applicable). Text-only Open-Platypus breaks both reasoning and vision alignment.
- **Sister-team collaboration:**
  - **R9700 (AMD RDNA4, ROCm 7.2):** `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` — calibration pipeline owner, FP32-softmax patch 011 originator, CT→native AWQ converter (saved us 13h on Qwen3.6-35B).
  - **M4 (Apple Silicon, MLX bridge):** `~/AI/m4-sglang-inference` — patch 013 owner (DeltaNet cache-wiring fix). Identified that Qwen3.5/3.6 support video and Gemma 4 supports audio; preprocessor_config.json often missing on community checkpoints.
  - `git fetch origin` each, read their commits, push findings to their READMEs. Patches are often portable; findings about model behavior (stop tokens, template quirks) always are.
