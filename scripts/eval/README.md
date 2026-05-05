# Evaluation Scripts

A layered set of tools for verifying that served models are actually working
correctly. The validator (`validate_capabilities.py`) is the fast gate; the
probe trio is for catching silent quality regressions that a pass/fail gate
misses.

## Validators (gate)

| Script | Purpose |
|--------|---------|
| [`validate_capabilities.py`](validate_capabilities.py) | basic / thinking / vision / video gate (PASS/FAIL per check). Exit 0 if all enabled checks pass. |
| [`test_capabilities_all.sh`](test_capabilities_all.sh) | Sweep `validate_capabilities.py` across all default-MODELS-set presets. Persists results to `benchmarks/quality/capability_check.json`. |
| [`validate_chat_template.py`](validate_chat_template.py) | Verifies the model's chat template renders correctly with thinking + multimodal markers. |

The validator's checks are intentionally loose to keep the gate fast and
robust — `check_vision` requires only one color word AND one shape word in
the response, `check_basic` looks for "paris", etc. **A `4/4 PASS` from this
validator does not mean the model is producing high-quality content** —
it means the model didn't silently break. For content-quality verification
use the probe trio below.

```bash
# Single model (assumes server already running):
python scripts/eval/validate_capabilities.py --port 23334

# Sweep all default presets (each preset boots → validates → kills):
./scripts/eval/test_capabilities_all.sh
```

## Probe trio (deeper than gate)

Each probe sends a structured prompt, inspects content + reasoning_content,
classifies the response into STRONG / DEGRADED / FAIL, and returns exit
code 0 / 1 / 2 for CI integration. Use these to catch **silent quality
regressions** that the validator's loose keyword grep misses.

| Script | Probes | Classifier |
|--------|--------|------------|
| [`probe_thinking.py`](probe_thinking.py) | Multi-step word problem with `skip_special_tokens=False` so the raw `<\|channel>thought` markers are visible. | THINKING VERIFIED if structured channel + correct answer + clean termination; THINKING DEGRADED otherwise. |
| [`probe_vision.py`](probe_vision.py) | Synthetic 256x256 red circle (with optional `--no-outline`). | STRONG if color + shape word found AND no degradation keywords; DEGRADED if keyword grep passes but content reads pixel/scatter/gradient (Gemma 4 task #66 pattern); FAIL otherwise. |
| [`probe_codegen.py`](probe_codegen.py) | Two algorithmic prompts (paren-balance + interval-merge). Extracts code blocks, exec's them, runs 8 hand-rolled unit tests. | STRONG if 8/8 pass; PARTIAL if some pass; FAIL otherwise. |

```bash
# Run against any served model:
python scripts/eval/probe_thinking.py  --port 23334 --model gemma4
python scripts/eval/probe_vision.py    --port 23334 --model qwen35-tp1
python scripts/eval/probe_codegen.py   --port 23334 --model coder-reap-25b
```

The probe trio is what surfaced the Gemma 4 vision degradation (task #66)
that the validator's keyword-grep was passing on. See `README.md` Known
Issues section for the full investigation arc.

## Quality benchmarks (fuller suite)

| Script | Purpose |
|--------|---------|
| [`eval_comprehensive.py`](eval_comprehensive.py) | 39-test quality suite (math, code, reasoning, vision, parallel). Designed to catch TP=2 precision errors. |
| [`eval_quality.py`](eval_quality.py) | Smaller quality probe focused on math/code correctness. |
| [`eval_and_chart.py`](eval_and_chart.py) | Wraps `eval_*` with chart generation for tracking quality across versions. |

```bash
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4 --thinking-budget 512
```

Run after configuration changes to verify model quality. Catches:
- Off-by-one arithmetic (389 vs 391)
- Garbled code (`s[::-]` instead of `s[::-1]`)
- Wrong imports
- Vision/multimodal regressions

## Other utilities

| Script | Purpose |
|--------|---------|
| [`audit_calib_quality.py`](audit_calib_quality.py) | Range-fetches HF safetensors index of any AWQ model and audits the recipe ignore-list (vision tower preserved BF16, MoE router preserved BF16, etc). RAM-safe — no model load. |
| [`warmup.py`](warmup.py) | Server warmup utility (cold-cache mitigation). |

```bash
# Audit our shipped checkpoints' calibration completeness:
python scripts/eval/audit_calib_quality.py
```
