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
scripts/setup.sh                       # Full setup (clones SGLang v0.5.23, applies all patches/*.patch — 44 logical units)
scripts/serve_production.sh gemma4-31b # Persistent PRODUCTION endpoint on :30000 (start/stop/status/restart; eval harness stays :23334)
scripts/launch.sh devstral             # Devstral 24B AWQ (Dense, Mistral)  [ad-hoc/eval serve on :23334]
scripts/launch.sh coder-30b-eval       # Qwen3-Coder-30B-A3B AWQ CT (256K, bake-off 43.0% opencode)
scripts/launch.sh coder-reap-25b       # Qwen3-Coder-REAP-25B-A3B AWQ (256K, 40.7% claw)
scripts/launch.sh qwen36               # Qwen3.6-35B-A3B-AWQ (thinking+vision, 256K)
scripts/launch.sh qwen36-dense         # Qwen3.6-27B Dense AWQ
scripts/launch.sh qwen35-moe           # Qwen3.5-28B-A3B-REAP-AWQ (DeltaNet+MoE, thinking+vision)
scripts/launch.sh qwen3-ream           # Qwen3-30B Instruct REAM AWQ (96 experts, 69 tok/s @ 255K)
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ (thinking+image+video+audio)
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
```
Full preset list: `grep -E "^        [a-z][a-zA-Z0-9-]*[\|\)]" scripts/launch.sh` (21 currently). Every preset carries an explicit `--tool-call-parser` matching its chat-template's tool format (qwen3_coder / qwen25 / mistral / gemma4 — see Critical Rules below).

## Critical Rules
- **SGLang only** — uses AWQ_Marlin kernels (sm_80+), patches may be needed for tuning
- **48GB VRAM limit** — 80B+ models do NOT fit
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- Always source `scripts/common.sh` + `activate_conda` + `setup_nvidia_env` before launching
- **Calibration recipe `ignore` lists must use regex for descendants** (`r"re:.*embed_vision.*"`, never a bare module name) — detail in [docs/agent-operations.md](docs/agent-operations.md#critical-calibration-rules).
- **Run `scripts/eval/check_awq_scales.py --base <bf16>` after every CT→native AWQ conversion**; non-zero exit = do not ship — detail in [docs/agent-operations.md](docs/agent-operations.md#critical-calibration-rules).

## Calibration rules and operational lessons

In force, read before touching a checkpoint, the harness or the host: [docs/agent-operations.md](docs/agent-operations.md) — capability preservation (thinking + image + video + audio), calibration datasets, the detached-job (`setsid`) pattern, Rule 1 / Rule 2, the conda env split, README discipline, recipe specifics, SWE-bench rollout landmines, patch hygiene, known model issues.

## Optimization Target
- **Primary:** single-user **256K context** performance (decode tok/s, TPOT). Measure at long context first.
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win batch benchmarks.
- **Eval harness:** SWE-bench Lite on **one lane per preset, `omp` (oh-my-pi) by default** — the scaffolds tied within noise (user decision); do not spend GPU time on scaffold A/Bs.
- **Current focus (user): Flint** — after the v0.5.23 flip campaign lands, Flint levers first (issues #15 / #12: omp re-roll under the fixed tool grammar, prefill kernel, deep-retrieval band, calibrated g32 A/B); other presets only for fleet-wide fixes. One-off cycles may use the GPUs; the main bake-off queue stays PAUSED until the user resumes it.

## Optimization Iteration Loop (the standing process)
This is a **continuous loop**, not a one-shot. Each iteration (a wake-up, a finished background job, or a fresh session) runs these steps. Never stop to ask for confirmation — the user reads the README + everything we report and interjects with course-corrections.

1. **Sync first.** `git fetch` BOTH repos (this one + R9700 `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference`); read new commits (`git log HEAD..origin/main`); `git pull --rebase` ours (the calibration device pushes here). New R9700 findings can re-prioritize the loop.
2. **Read live state.** `nvidia-smi`, `sudo docker ps`, check for running serving/scoring/calibration. Honor **Rule 1** (no concurrent calibration + eval/serving) and **Rule 2** (no concurrent rollout + score) — only one GPU/docker workload at a time.
3. **Pick the next lever.** Highest-value untried *serving/optimization or eval* item for a model in the perf table — README → Status & next steps (item 2) / `docs/decode-levers.md`, a bake-off cycle, or a 256K quality/throughput probe. **This box owns evals + serving/256K-optimization + analysis + docs + cross-team sharing, and (user, after the Qwopus ship) may also run calibrations** — GPTQ→CT→AWQ and Flint AWQ-calibrator builds are allowed here under Rule 1 (never concurrent with serving/eval; GPU work is serial), the separate same-repo calibration device still pushes its own commits that we `git pull --rebase`. Note the chosen lever in its GitHub issue (open one if none exists) *before* starting (user can redirect); the README carries state only.
4. **Baseline → apply → measure.** Single-user (M=1), at the model's real KV cap, fresh prefill. Harness: `scripts/eval/run_v0512_fleet_eval.sh` (quality + tok/s + 256K probe) or `scripts/bench/bench_long_context.py`. Record tok/s + TPOT.
5. **Decide & document.** WIN → update the preset in `launch.sh` + the [Performance] / [Model Support] tables (these tables ARE the receipts), and `BASELINE=save scripts/bench/bench_regression.sh <preset>` to lock the new level into the tripwire. NULL/parked → one ledger line in `docs/decode-levers.md` with the receipt path under `benchmarks/`; negative results are findings, keep them. Either way, push portable findings to R9700's README. Any patch touching the serving hot path → re-run `bench_regression.sh` compare on affected presets.
6. **Commit + push** at the clean boundary (small self-contained commits). Then **schedule the next iteration** (background-job completion re-invokes us; otherwise `ScheduleWakeup`).

Capability guardrail on every iteration that touches a checkpoint: **preserve thinking + image + video + audio** (we have silently broken thinking and image during calibration — see Calibration Rules). Run the applicable probes before claiming a win.

## Current Hardware State
- **Both 3090s online.** TP=2 / 256K is the matrix-standard configuration. No TP=1 fallback presets exist — every preset in launch.sh is tuned for TP=2 / 256K. Default `--tp 2 --context-length 262144 --max-running 1`. Cooling profile (GPU cap 300 W + CPU boost off + GPU fan floor 100 %, `systemd/gpu-cooling.sh` + `gpu-fan-curve.service`) is load-bearing for sustained bake-off runs.

## Workflow
- **Work autonomously. Never stop to ask for confirmation.** User checks in periodically by reading the README + everything we report, and will interrupt our flow with better ways to frame a problem or new ideas to try — fold those in immediately and keep iterating. Max effort is the default.
- **Multi-hour calibrations are pre-authorized.** Downloading 50-70 GB BF16 bases + running 10-13h GPTQ calibrations does NOT need user check-in. Detach via `setsid` pattern and keep working on other fronts.
- **Note the next step in its GitHub issue before starting it** — user can interject if they see a better path. Commit + push as progress is made (small self-contained commits, not one giant batch). Every commit should stand on its own.
- **Keep README.md clean.** It is the single source of truth. Once a ship supersedes a debugging narrative, trim the narrative. It carries current state only; open work and known issues are GitHub issues (the shared docs-check CI enforces the skeleton, budgets and banned patterns).
- **Carry forward these design principles** — user has re-emphasized them and they should not drift:
  - **REAP and REAM are different MoE compression strategies** — don't conflate. **REAP** ([Cerebras](https://github.com/CerebrasResearch/reap)) = expert **pruning** (drops low-impact experts, tends better on generative tasks). **REAM** ([Samsung SAIL](https://github.com/SamsungSAILMontreal/ream)) = expert **merging** (groups similar experts, ~94%+ quality). Both shrink MoE to fit 256K in 48 GB VRAM but via different algorithms with different tradeoffs. Full details in `scripts/quantize/REAM.md`.
  - **Chat templates are load-bearing.** Wrong BOS/EOS, missing `<think>` handling, or reasoning stripped from calibration data silently destroys quality. Inspect `chat_template.jinja` and validate thinking tags on every new model before claiming ship.
  - **Calibration data must cover all live modalities** (thinking + image + video + audio as applicable). Text-only Open-Platypus breaks both reasoning and vision alignment.
- **Sister-team collaboration:**
  - **R9700 (AMD RDNA4, ROCm):** `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` — FP8 calibration owner (native gfx1201 FP8); RDNA4/ROCm serving stack. Many of our patches are ports of theirs (and vice versa) — patches are usually portable, findings about model behavior always are.
  - **M4 (Apple Silicon, MLX bridge):** `~/AI/m4-sglang-inference` — cross-checks chat-template + multimodal plumbing. Standing reminders from them: Qwen3.5/3.6 support video and Gemma 4 supports audio; `preprocessor_config.json` is often missing on community checkpoints.
  - **Monitor sister-team pushes EVERY loop iteration (user directive):** `git fetch origin` on BOTH this repo and theirs at each wake-up; read any new commits (`git log HEAD..origin/main`) before continuing — the calibration device pushes to OUR repo and R9700 pushes findings/asks that change priorities. Push findings to their READMEs. Patches are often portable; findings about model behavior (stop tokens, template quirks) always are.
  - **Division of labor (current):** 3090 owns **all evals + AWQ/INT4 calibrations** end-to-end — Ampere has native INT4 / AWQ_Marlin acceleration so GPTQ → CT → AWQ recipe runs, REAM/REAP rebuilds, SWE-bench bake-offs, and capability sweeps all live here. **R9700 owns FP8 calibrations** — gfx1201 has native FP8 weight acceleration that doesn't pay off on Ampere. When a 3090 bake-off surfaces an AWQ model regression, the recal lands in *this* repo, not R9700's; an FP8 variant request goes to them. Both stacks keep full pipeline capability so either can step in for parallel/backup runs.

