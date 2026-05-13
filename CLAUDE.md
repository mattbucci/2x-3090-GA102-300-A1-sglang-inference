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
scripts/setup.sh                       # Full setup (clones SGLang v0.5.11, applies 17 patches)
scripts/launch.sh devstral             # Devstral 24B AWQ (Dense, Mistral)
scripts/launch.sh coder-30b-eval       # Qwen3-Coder-30B-A3B AWQ CT (256K, bakeoff lead 40.3% opencode)
scripts/launch.sh coder-reap-25b       # Qwen3-Coder-REAP-25B-A3B AWQ (256K, 33% claw)
scripts/launch.sh qwen36               # Qwen3.6-35B-A3B-AWQ-CT (thinking+vision, 256K)
scripts/launch.sh qwen36-dense         # Qwen3.6-27B Dense AWQ
scripts/launch.sh qwen35-moe           # Qwen3.5-28B-A3B-REAP-AWQ (DeltaNet+MoE, thinking+vision)
scripts/launch.sh qwen3-ream           # Qwen3-30B Instruct REAM AWQ (96 experts, 107 tok/s @ 256K)
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ (thinking+image+video+audio)
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
```
Full preset list (20 total — `grep -E "^        [a-z][a-zA-Z0-9-]*[\|\)]" scripts/launch.sh`); every preset carries an explicit `--tool-call-parser` matching its chat-template's tool format (qwen3_coder / qwen25 / mistral / gemma4 — see Critical Rules below).

## Critical Rules
- **SGLang only** — uses AWQ_Marlin kernels (sm_80+), patches may be needed for tuning
- **48GB VRAM limit** — 80B+ models do NOT fit
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- Always source `scripts/common.sh` + `activate_conda` + `setup_nvidia_env` before launching
- **Calibration recipe `ignore` lists must use regex for descendants.** llmcompressor matches at module-name granularity — bare strings like `"model.embed_vision"` do NOT exclude `model.embed_vision.embedding_projection` (the actual Linear underneath). Always use `r"re:.*embed_vision.*"` / `r"re:.*vision_tower.*"` / `r"re:.*multi_modal_projector.*"` patterns. Cost of forgetting: 16h calibration silently produces zero scales for the descendant Linear, model dequantizes image embeddings to zero, NaN cascade in LM forward, sampler crashes (HSAIL on RDNA4, "Detected errors during sampling! NaN in the logits" warmup-time on Ampere). Both stacks lost ~16h on 2026-05-06 Gemma 4 26B v3 to this. Cross-stack root cause via R9700 forensic v2-vs-v3 safetensors diff (their commit `176b917`). 3090 ports: `3960477` + `6ca8c12` (audited all `quantize_*.py`).
- **Run `scripts/eval/check_awq_scales.py` after every CT→native AWQ conversion.** Scans every `*.scales` / `*.weight_scale` tensor for all-zero / NaN / Inf / extreme-magnitude values. `validate_capabilities.py` cannot catch silent zero-scales — the model loads, the server boots, generation produces NaN logits that get masked or returned as empty content. The forensic-safetensors method took 30 seconds to find the v3 disaster the validator missed. Make it part of every pipeline step before ship; non-zero exit means do NOT ship. (Note: the script currently reads native-AWQ format; CT-format checkpoints crash it — fix or use Range-fetch HF mode for CT mirrors.)

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

## Current Hardware State (2026-05-13)
- **Both 3090s online.** TP=2 / 256K is the matrix-standard configuration. No TP=1 fallback presets exist — every preset in launch.sh is tuned for TP=2 / 256K. Default `--tp 2 --context-length 262144 --max-running 1`. Cooling profile (260W power cap + gpu-fan-curve.service) is load-bearing for sustained bake-off runs.

## Workflow (RECONFIRMED 2026-05-09)
- **Work autonomously. Never stop to ask for confirmation.** User checks in periodically by reading the README and will interrupt with new ideas or redirects. Max effort is the default.
- **Multi-hour calibrations are pre-authorized.** Downloading 50-70 GB BF16 bases + running 10-13h GPTQ calibrations does NOT need user check-in. Detach via `setsid` pattern and keep working on other fronts.
- **Note the next step in the README before starting it** — user can interject if they see a better path. Commit + push as progress is made (small self-contained commits, not one giant batch). Every commit should stand on its own.
- **Keep README.md clean.** It is the single source of truth. Once a ship supersedes a debugging narrative, trim the narrative. Reader should see current status + known issues + next step without scrolling.
- **Carry forward these design principles** — user has re-emphasized them and they should not drift:
  - **REAP and REAM are different MoE compression strategies** — don't conflate. **REAP** ([Cerebras](https://github.com/CerebrasResearch/reap)) = expert **pruning** (drops low-impact experts, tends better on generative tasks). **REAM** ([Samsung SAIL](https://github.com/SamsungSAILMontreal/ream)) = expert **merging** (groups similar experts, ~94%+ quality). Both shrink MoE to fit 256K in 48 GB VRAM but via different algorithms with different tradeoffs. Full details in `scripts/quantize/REAM.md`.
  - **Chat templates are load-bearing.** Wrong BOS/EOS, missing `<think>` handling, or reasoning stripped from calibration data silently destroys quality. Inspect `chat_template.jinja` and validate thinking tags on every new model before claiming ship.
  - **Calibration data must cover all live modalities** (thinking + image + video + audio as applicable). Text-only Open-Platypus breaks both reasoning and vision alignment.
- **Sister-team collaboration:**
  - **R9700 (AMD RDNA4, ROCm 7.2):** `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` — calibration pipeline owner, FP32-softmax patch 011 originator, CT→native AWQ converter (saved us 13h on Qwen3.6-35B).
  - **M4 (Apple Silicon, MLX bridge):** `~/AI/m4-sglang-inference` — patch 013 owner (DeltaNet cache-wiring fix). Identified that Qwen3.5/3.6 support video and Gemma 4 supports audio; preprocessor_config.json often missing on community checkpoints.
  - `git fetch origin` each, read their commits, push findings to their READMEs. Patches are often portable; findings about model behavior (stop tokens, template quirks) always are.
