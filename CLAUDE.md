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
scripts/setup.sh                       # Full setup (clones SGLang v0.5.15, applies all patches/*.patch — 24 logical units)
scripts/launch.sh devstral             # Devstral 24B AWQ (Dense, Mistral)
scripts/launch.sh coder-30b-eval       # Qwen3-Coder-30B-A3B AWQ CT (256K, bake-off 43.0% opencode)
scripts/launch.sh coder-reap-25b       # Qwen3-Coder-REAP-25B-A3B AWQ (256K, 40.7% claw)
scripts/launch.sh qwen36               # Qwen3.6-35B-A3B-AWQ (thinking+vision, 256K)
scripts/launch.sh qwen36-dense         # Qwen3.6-27B Dense AWQ
scripts/launch.sh qwen35-moe           # Qwen3.5-28B-A3B-REAP-AWQ (DeltaNet+MoE, thinking+vision)
scripts/launch.sh qwen3-ream           # Qwen3-30B Instruct REAM AWQ (96 experts, 69 tok/s @ 255K)
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ (thinking+image+video+audio)
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
```
Full preset list: `grep -E "^        [a-z][a-zA-Z0-9-]*[\|\)]" scripts/launch.sh` (24 currently). Every preset carries an explicit `--tool-call-parser` matching its chat-template's tool format (qwen3_coder / qwen25 / mistral / gemma4 — see Critical Rules below).

## Critical Rules
- **SGLang only** — uses AWQ_Marlin kernels (sm_80+), patches may be needed for tuning
- **48GB VRAM limit** — 80B+ models do NOT fit
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- Always source `scripts/common.sh` + `activate_conda` + `setup_nvidia_env` before launching
- **Calibration recipe `ignore` lists must use regex for descendants.** llmcompressor matches at module-name granularity — bare strings like `"model.embed_vision"` do NOT exclude `model.embed_vision.embedding_projection` (the actual Linear underneath). Always use `r"re:.*embed_vision.*"` / `r"re:.*vision_tower.*"` / `r"re:.*multi_modal_projector.*"` patterns. Cost of forgetting: 16h calibration silently produces zero scales for the descendant Linear, model dequantizes image embeddings to zero, NaN cascade in LM forward, sampler crashes (HSAIL on RDNA4, "Detected errors during sampling! NaN in the logits" warmup-time on Ampere). Both stacks lost ~16h on 2026-05-06 Gemma 4 26B v3 to this. Cross-stack root cause via R9700 forensic v2-vs-v3 safetensors diff (their commit `176b917`). 3090 ports: `3960477` + `6ca8c12` (audited all `quantize_*.py`).
- **Run `scripts/eval/check_awq_scales.py` after every CT→native AWQ conversion.** Scans every `*.scales` / `*.weight_scale` tensor for all-zero / NaN / Inf / extreme-magnitude values. `validate_capabilities.py` cannot catch silent zero-scales — the model loads, the server boots, generation produces NaN logits that get masked or returned as empty content. The forensic-safetensors method took 30 seconds to find the v3 disaster the validator missed. Make it part of every pipeline step before ship; non-zero exit means do NOT ship. **Pass `--base <bf16_base_dir>` for MoE ships** — the dead-channel comparator downgrades the benign structural-sparsity zero-scales (MoE bases ship 50-72% of layer-0 expert gate/up channels at ~7.8e-38; AWQ flushes them to fp16-0) while still flagging zero-scales over *live* base blocks as `DEFECT`. Validated on qwen36 (144 flags → 0 residual; injected live-block zero still caught). Without `--base` the audit stays conservative and over-flags. (Note: the script reads native-AWQ format; CT-format checkpoints crash its tensor reader — use a native-AWQ mirror or Range-fetch HF mode for CT audits.)

## Optimization Target
- **Primary:** single-user **256K context** performance (decode tok/s, TPOT). Measure at long context first.
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win batch benchmarks.

## Optimization Iteration Loop (the standing process)
This is a **continuous loop**, not a one-shot. Each iteration (a wake-up, a finished background job, or a fresh session) runs these steps. Never stop to ask for confirmation — the user reads the README + everything we report and interjects with course-corrections.

1. **Sync first.** `git fetch` BOTH repos (this one + R9700 `~/AI/rdna4-inference-triton36`); read new commits (`git log HEAD..origin/main`); `git pull --rebase` ours (the calibration device pushes here). New R9700 findings can re-prioritize the loop.
2. **Read live state.** `nvidia-smi`, `sudo docker ps`, check for running serving/scoring/calibration. Honor **Rule 1** (no concurrent calibration + eval/serving) and **Rule 2** (no concurrent rollout + score) — only one GPU/docker workload at a time.
3. **Pick the next lever.** Highest-value untried *serving/optimization or eval* item for a model in the perf table — README → [Tooling → Decode ideas], a bake-off cycle, or a 256K quality/throughput probe. **Stay in lane: this box owns evals + serving/256K-optimization + analysis + docs + cross-team sharing. Do NOT start calibrations here** — GPTQ→CT→AWQ, REAM/REAP rebuilds, and the MoE coverage backlog run on a separate same-repo device; we only `git pull --rebase` to pick up their commits. Note the chosen lever in the README *before* starting (user can redirect).
4. **Baseline → apply → measure.** Single-user (M=1), at the model's real KV cap, fresh prefill. Harness: `scripts/eval/run_v0512_fleet_eval.sh` (quality + tok/s + 256K probe) or `scripts/bench/bench_long_context.py`. Record tok/s + TPOT.
5. **Decide & document.** WIN → update the preset in `launch.sh` + the [Performance] / [Model Support] tables (these tables ARE the receipts). NULL/parked → one line in README "Decode ideas" with the receipt path under `benchmarks/`; negative results are findings, keep them. Either way, push portable findings to R9700's README.
6. **Commit + push** at the clean boundary (small self-contained commits). Then **schedule the next iteration** (background-job completion re-invokes us; otherwise `ScheduleWakeup`).

Capability guardrail on every iteration that touches a checkpoint: **preserve thinking + image + video + audio** (we have silently broken thinking and image during calibration — see Calibration Rules). Run the applicable probes before claiming a win.

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

## Current Hardware State (2026-07-12)
- **Both 3090s online.** TP=2 / 256K is the matrix-standard configuration. No TP=1 fallback presets exist — every preset in launch.sh is tuned for TP=2 / 256K. Default `--tp 2 --context-length 262144 --max-running 1`. Cooling profile (260W power cap + gpu-fan-curve.service) is load-bearing for sustained bake-off runs.

## Workflow (RECONFIRMED 2026-05-31)
- **Work autonomously. Never stop to ask for confirmation.** User checks in periodically by reading the README + everything we report, and will interrupt our flow with better ways to frame a problem or new ideas to try — fold those in immediately and keep iterating. Max effort is the default.
- **Multi-hour calibrations are pre-authorized.** Downloading 50-70 GB BF16 bases + running 10-13h GPTQ calibrations does NOT need user check-in. Detach via `setsid` pattern and keep working on other fronts.
- **Note the next step in the README before starting it** — user can interject if they see a better path. Commit + push as progress is made (small self-contained commits, not one giant batch). Every commit should stand on its own.
- **Keep README.md clean.** It is the single source of truth. Once a ship supersedes a debugging narrative, trim the narrative. Reader should see current status + known issues + next step without scrolling.
- **Carry forward these design principles** — user has re-emphasized them and they should not drift:
  - **REAP and REAM are different MoE compression strategies** — don't conflate. **REAP** ([Cerebras](https://github.com/CerebrasResearch/reap)) = expert **pruning** (drops low-impact experts, tends better on generative tasks). **REAM** ([Samsung SAIL](https://github.com/SamsungSAILMontreal/ream)) = expert **merging** (groups similar experts, ~94%+ quality). Both shrink MoE to fit 256K in 48 GB VRAM but via different algorithms with different tradeoffs. Full details in `scripts/quantize/REAM.md`.
  - **Chat templates are load-bearing.** Wrong BOS/EOS, missing `<think>` handling, or reasoning stripped from calibration data silently destroys quality. Inspect `chat_template.jinja` and validate thinking tags on every new model before claiming ship.
  - **Calibration data must cover all live modalities** (thinking + image + video + audio as applicable). Text-only Open-Platypus breaks both reasoning and vision alignment.
- **Sister-team collaboration:**
  - **R9700 (AMD RDNA4, ROCm 7.2):** `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` — FP8 calibration owner (RDNA4 gfx1201 has native FP8 weight acceleration); RDNA4/ROCm serving stack. Recipe originator for FP32-softmax patch 011 and the CT→native AWQ converter (saved us 13h on Qwen3.6-35B).
  - **M4 (Apple Silicon, MLX bridge):** `~/AI/m4-sglang-inference` — patch 013 owner (DeltaNet cache-wiring fix). Identified that Qwen3.5/3.6 support video and Gemma 4 supports audio; preprocessor_config.json often missing on community checkpoints.
  - **Monitor sister-team pushes EVERY loop iteration (user directive 2026-06-10):** `git fetch origin` on BOTH this repo and theirs at each wake-up; read any new commits (`git log HEAD..origin/main`) before continuing — the calibration device pushes to OUR repo and R9700 pushes findings/asks that change priorities. Push findings to their READMEs. Patches are often portable; findings about model behavior (stop tokens, template quirks) always are.
  - **Division of labor (current, 2026-05-19):** 3090 owns **all evals + AWQ/INT4 calibrations** end-to-end — Ampere has native INT4 / AWQ_Marlin acceleration so GPTQ → CT → AWQ recipe runs, REAM/REAP rebuilds, SWE-bench bake-offs, and capability sweeps all live here. **R9700 owns FP8 calibrations** — gfx1201 has native FP8 weight acceleration that doesn't pay off on Ampere. When a 3090 bake-off surfaces an AWQ model regression, the recal lands in *this* repo, not R9700's; an FP8 variant request goes to them. Both stacks keep full pipeline capability so either can step in for parallel/backup runs.

## Operational Lessons (consolidated from working memory)

### Hardware quirks
- **Kernel BUG reboots every ~9-17h.** Sustained docker rollout I/O hangs the box (kernel-level lock; journal stops writing mid-`docker run swebench-rollout-...`). User hard-resets when noticed. Predictions on disk survive; last 5-15 unflushed pagecache writes are lost. To resume: relaunch wrapper via setsid + relaunch `run_all_cycles.sh WAIT_FOR_PID=<new-pid>`. systemd unit `swebench-bakeoff.service` (tracked at `systemd/`) auto-resumes on boot — **but it starts the *default* full-cycle queue, NOT an opencode-only baseline sweep**; if a baseline run is the intended one, `sudo systemctl stop swebench-bakeoff` before relaunching `evals/swebench/run_opencode_baseline.sh` (predictions persist; `--skip-existing` resumes mid-queue). Forensic recipe: `journalctl --boot=-1 --no-pager | tail -1`. Durable fix needs sudo (kernel upgrade or storage-driver swap); don't try to "prevent" from inside the agent.
- **Rule 1 — no concurrent calibration + eval.** GPTQ calibration (15+ cores, ~60 GB RAM hessians) + SGLang server (90% VRAM) + opencode-driven traffic exceeds host headroom. Crashed the box 2026-04-25 on Qwen3.5-28B.
- **Rule 2 — no concurrent rollout + score.** Both spin per-instance docker containers. Concurrent VFS pressure triggers the kernel BUG (above). `run_model_cycle.sh` sequences them; don't bypass.
- **Cooling profile is load-bearing.** 260 W power cap + `gpu-fan-curve.service` keeps DDR5 below ALARM HIGH and prevents thermal-Python-heap corruption (separate failure from Rule 1/2). Don't disable.

### Conda env split — `quant` vs the serving env
- **`sglang-v0515`** (current serving env; version-suffixed per stack, resolved by `common.sh` ENV_NAME) — SGLang + CUDA + `compressed_tensors==0.15.0.1` (lacks `.distributed`). Use for `launch.sh`, `validate_capabilities.py`, `probe_*.py`.
- **`quant`** — calibration. Has `compressed_tensors==0.15.1.dev6+g077e752` with `.distributed`, llmcompressor importable. Use for `quantize_*.py`, GPTQ, CT→AWQ conversion.
- Wrong env on quantize gives `ModuleNotFoundError: compressed_tensors.distributed`. Don't try to upgrade the serving env — use `quant`. Pattern: `conda activate quant` (NOT the serving env) before any llmcompressor work.

### README discipline (public-facing surface)
- **README is live state + planning ONLY (user directive 2026-06-10).** ALWAYS remove completed/solved tasks from the main README — don't mark them DONE/CONCLUDED and leave them ("CONCLUDED" in the README is the violation tell). `patches/README.md` is the historical log: fold durable outcomes into the live sections they changed (model tables, perf narrative, presets), move the narrative to patches/README.md (or the lab LOG it links), DELETE the roadmap/sprint entry.
- **Cross-team / sister-teams section is a heartbeat.** Show only open asks + current relationships. Remove past-tense bullets ("we adopted X" — the preset list IS the receipt) and informational-only sister-stack findings with no 3090 action. Same applies to narrative residue like "(from 30-instance smoke matrix, since wiped)".
- **No internal/agent notes in the public README.** No `/tmp/...` paths, no `setsid` mechanics, no "active session YYYY-MM-DD" banners, no "in flight this session" framing. Those belong in CLAUDE.md, task lists, or memory — not the project face. Quick test: would a stranger visiting the repo find this line useful?

### Calibration recipe specifics
- **DeltaNet narrow ignore — only `in_proj_a$/b$`.** Broad `re:.*linear_attn\..*` fails because loader expects INT4 qkvz. v1 (everything INT4) and v2 (whole linear_attn BF16) both gave garbage; v3 working pattern: `["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$"]` + model-specific MoE router/shared-gate exclusions. Probe with "What is the capital of France?"; `!!!!!` output means wrong ignore.
- **Per-expert AWQ dense-MLP detection.** Don't hardcode `quant_config=None` for "dense MLP on MoE block" — recipes vary. Detect at construction from `quantization_config.ignore`: regex-match the module path against ignore entries, switch to None only when matched. Symptom of getting this wrong: `<pad>` output for every prompt; `check_awq_scales.py` won't catch it (safetensors are correct; the model just doesn't bind them).
- **Drop file caches before calibration; don't reduce quality settings.** `echo 3 > /proc/sys/vm/drop_caches` first. Investigate actual memory before lowering samples / sequence length.
- **Recalibrate over source edits.** When a detached calibration dies, find the external cause (RAM pressure, competing process) and restart. Don't add try/except, checkpointing, or "harden" the script — compute is cheap, silent correctness bugs are expensive.

### SWE-bench rollout details
- **Three scaffold landmines that silently produce empty diffs.** (a) little-coder ignores `OPENAI_BASE_URL` — use `--model llamacpp/<served-name>` and sed-patch the packaged `models.json`. (b) claw binary GLIBC mismatch — build inside `rust:bullseye` (Debian 11, GLIBC 2.31) in `scripts/build_claw.sh`, not host Arch. (c) per-instance rollout images cached by tag — Dockerfile/opencode.json edits do NOT propagate; nuke `swebench-rollout/*` after every scaffold-config edit.
- **`--tool-call-parser` is per-preset and load-bearing.** Without the right parser, the model's `<tool_call><function=NAME>...` XML is served as `content` plain text, harness drops it, diff is empty. Family→parser mapping audited 2026-05-13: Qwen3-Coder + every Qwen3.5/3.6 (incl. dense, MoE, VL-REAP, REAM) → `qwen3_coder`; Qwen3-VL non-coder + Qwen3-30B-Instruct REAM → `qwen25`; Devstral → `mistral`; Gemma 4 → `gemma4`. Reasoning + tool-call parsers compose correctly when thinking is enabled.
- **Bake-off numbers are only trustworthy at the FULL 300 — partial prediction sets lie, in BOTH directions.** Rule: only compare cells at 300/300 predictions; show `resolved/n_pred` and discount any cell under 300 (two plausible-looking over-claims from sub-300 data both died on full cycles).
- **When a scaffold exits suspiciously fast with no model output, suspect a server-side 4xx (template/role/param reject) BEFORE theorizing about model capability — capture the actual HTTP request/response.** Canonical case: pi-ai/little-coder sends its system prompt as role `developer`; Qwen3.5/3.6 templates raised on it → SGLang 400 → ~3 s empty exits that looked like an architecture limit. Fix: `developer`→`system` in the templates (`scripts/eval/patch_chat_templates_developer_role.py`, wired into setup.sh); thinking ships then run full-300 on all three scaffolds. Coder templates don't validate roles, which is why coders never showed it.
- **REAM ties native on A3B-MoE agentic coding** (qwen36 = qwen36-ream on opencode, full-300 both) — REAM-merge does not degrade it; qwen3-ream (Instruct, non-thinking) failing opencode+claw is a genuine model gap, not infra.
- **Common failure mode for 30B-class on SWE-bench Lite: incomplete fix, not silence.** Thinking IS engaged in production (opencode under-reports because SGLang returns `usage.reasoning_tokens` FLAT, but opencode parses OpenAI's nested `completion_tokens_details.reasoning_tokens` path). 40% on Lite is competitive for 30B-class; failures are "model patched the symptom but missed the root cause", not infra. Don't treat 40% as broken.

### Project hygiene
- **No hardcoded `/home/letsrtfm/...` paths.** Use `$REPO_DIR` / `$MODELS_DIR` (set in `scripts/common.sh`) or derive from `$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`. Hardcoded paths break for any other operator or host.
- **HuggingFace push target: `mattbucci/<NAME>`** (not `letsrtfm/`). Token at `~/.secrets/hf_token`. For uploads ≤25 GB use `hf upload <repo> <dir>`; for 50 GB+ that stall in commit phase, use R9700's `scripts/quantize/upload_repo_per_file.py` (one `HfApi.upload_file()` per file, idempotent on retry). GitHub PAT at `~/.secrets/gh_token` (already embedded in remote URLs).
- **Patch hygiene — 3-gate test for any new `patches/NNN-*.patch` (learned 2026-06-10).** Generate the diff against the *predecessor-patched* tree, NEVER pristine (old 045 was diffed vs pristine, conflicted with 040's rewrite of the same block, and fresh setups silently skipped it — the idempotent loop hides broken chains on the live tree). Hand-written hunks need unique anchoring context, ≥U10 from the real tree (old 026's anonymous hunk re-targeted the *image* path's identical block on a setup.sh rerun). Gate before commit: (a) full glob-order sequence applies on pristine v0.5.13.post1; (b) result is byte-identical to the live serving tree; (c) `git apply --check` FAILS for every patch on the already-patched tree (rerun safety). Recipe + narratives in `patches/README.md`. (2026-06-16: the v0.5.12→v0.5.13.post1 flip exercised exactly this — 023 and 026 touch disjoint regions of `gemma4_mm.py` with 023 < 026; gate (a) caught two traps: (i) `git apply` doesn't stage, so regenerating a patch via `git diff` against an un-staged worktree silently captured the *other* patch's hunks — use a committed baseline per step; (ii) a shared-file earlier patch must be generated against pristine+later so its hunk is isolated, and kept small enough that the later patch still applies via offset. Net: 023 generated against pristine+026, 026 left unchanged.)

### Known model issues
- **CT-format MoE at TP=2 `_load_w2` crash — FIXED by patch 030** (presharded-w2 detect). CT pre-shards w2 to per-rank size; the loader's `narrow(shard_dim, shard_size*tp_rank, shard_size)` overflowed on rank>0. Native AWQ was never affected (full-global w2). **The "drops at the next rebase" prediction was WRONG (verified at the v0.5.15 flip):** upstream's `use_presharded_weights` plumbing exists at exactly our narrow sites but nothing wires it for compressed-tensors loads (only Quark sets it) — 030 stays until a CT-wiring or shape-guard PR lands upstream. Residual caution: 030 is verified at the load path — smoke any TP=2 preset before repointing it at a CT checkpoint.
- **transformers bumps can silently re-route tokenizer backends (learned at the v0.5.15 flip, patch 057).** tx 5.12 sends Mistral-family checkpoints shipping tekken.json to `MistralCommonBackend`, which never parses special tokens from text — sglang's render-then-encode chat path then feeds `[INST]`/`[TOOL_CALLS]` as plain text. Symptom set: quality collapse (needle 0.0, HE halved, dead tool-calls) while boot + basic probes stay green. On any future tx bump, A/B-encode a chat-rendered prompt per tokenizer family (both envs, compare token ids) BEFORE trusting fleet results; a stack flip must fleet-probe QUALITY, not just boot+basic — the two Mistral-family breaks in consecutive rebases (054, 057) were both smoke-invisible.
