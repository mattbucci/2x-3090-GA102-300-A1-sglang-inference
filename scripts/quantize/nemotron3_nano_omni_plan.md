# Nemotron-3-Nano-Omni-30B-A3B AWQ — SHIPPED

**Ship:** [`mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`](https://huggingface.co/mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ) — native AWQ-int4 from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`.

**Status:** live. Serves on v0.5.13.post1 via **patches 052** (non-gated squared-ReLU `moe_wna16`) + **053** (EVS video routing); preset `nemotron3-omni`; `--reasoning-parser nano_v3 --tool-call-parser qwen3_coder --trust-remote-code`. **6/6 caps** (basic + thinking + tool + vision + video + audio — our only audio ship). Decode ~flat (102.7 short → 97.8 @255K, 5.25M-token pool; 23 Mamba2 layers are O(1) recurrent). Perf + caps in the [README](../../README.md) model table.

## Build essentials (reusable for the REAP/REAM variants)

- **Mamba2-hybrid: only the 23 MLP/MoE layers are int4** (per `hybrid_override_pattern`); the 23 Mamba2 + 6 attention layers stay BF16 (SSM recurrent state cannot INT4).
- **Sub-encoders stay BF16** via `modules_to_not_convert`: CRADIO v4-H (vision/video) + Parakeet tdt-0.6b-v2 (audio). Audio encoder needs `librosa` in the serving env.
- **Calibration preserves all live modalities** (thinking + image + video + audio) — `thinking_vision_video` + audio-text pairs; ship the repo `chat_template.jinja`; gate with `check_awq_scales.py --base`.
- No in-checkpoint MTP head and no published EAGLE3/DFlash draft; cross-target draft transfer is ruled out (Mamba2-hybrid + vocab/hidden mismatch). For spec-decode, the recurrent verify wall makes model-draft spec net-negative — no-spec is the path.

## Open next step

REAP + REAM variants (MoE coverage matrix): `run_reap.py` + the REAM merge must be extended to the Nemotron-H Mamba2-hybrid layout (prune only the 23 MLP/MoE layers). Tracked in the README MoE coverage matrix.
