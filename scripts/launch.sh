#!/bin/bash
# Unified model launcher for SGLang on 2x RTX 3090 (48GB total)
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh gemma4 --port 8000
#   MODEL=/path/to/weights ./scripts/launch.sh devstral
#
# Models (all TP=2 / 48 GB; matrix work runs --context-length 262144 --max-running 1):
#
# Dense:
#   devstral / devstral-32k / devstral-long  Devstral-24B AWQ (131K / 32K / 217K)
#   qwen35-dense                              Qwen3.5-27B Dense AWQ
#   qwen36-dense / qwen36-27b                 Qwen3.6-27B Dense AWQ (native)
#   qwen36-dense-ct                           Qwen3.6-27B Dense compressed-tensors
#   gemma4-31b                                Gemma 4 31B Dense AutoRound-AWQ
#   qwen3-vl-32b                              Qwen3-VL-32B Dense (VL)
#
# MoE (R9700 team is actively recalibrating MoE checkpoints):
#   qwen3-ream                                Qwen3-30B-Instruct-2507-REAM (96 exp)
#   coder-30b / coder-30b-eval                Qwen3-Coder-30B-A3B (128 exp)
#   coder-reap / coder-reap-25b               Qwen3-Coder-REAP-25B-A3B (103 exp)
#   coder-30b-ream                            Qwen3-Coder-30B-A3B-REAM (96 exp)
#   qwen35-moe                                Qwen3.5-28B-A3B-REAP (205 exp)
#   qwen36                                    Qwen3.6-35B-A3B (256 exp, thinking+vision)
#   qwen36-ream                               Qwen3.6-REAM-A3B (192 exp)
#   qwen36-vl-reap                            Qwen3.6-VL-REAP-26B-A3B (192 exp, VL)
#   gemma4                                    Gemma 4 26B MoE (103 exp)
#   qwen3-vl-moe                              Qwen3-VL-30B-A3B (VL MoE)
#
# Note: 80B+ models (coder-next, glm45-air) do NOT fit in 48GB VRAM.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
# QUANT default is set by the per-preset block below so presets that use
# `${QUANT:-X}` can honor an env override. Presets that hardcode `QUANT="X"`
# ignore env by design (most do — their weight format is fixed).
QUANT="${QUANT:-}"
# Capture env-provided values BEFORE applying global defaults, so presets
# can do `CTX="${_ENV_CTX:-4096}"` and have the env override take precedence.
# Without this, the global default below (e.g. CTX=32768) would shadow the
# `:-` fallback inside the preset case-block. KV_DTYPE was already on this
# pattern; CTX/MEM/MAX_RUNNING/CHUNKED added 2026-05-01 after the
# qwen3-vl-32b preset edit revealed the gotcha. DTYPE added 2026-05-03 for
# gemma4 (BF16 required to avoid SigLIP vision tower NaN on FP16).
_ENV_KV_DTYPE="${KV_DTYPE:-}"
_ENV_CTX="${CTX:-}"
_ENV_MEM="${MEM:-}"
_ENV_MAX_RUNNING="${MAX_RUNNING:-}"
_ENV_CHUNKED="${CHUNKED:-}"
_ENV_DTYPE="${DTYPE:-}"
DTYPE="${_ENV_DTYPE:-float16}"
CTX=32768
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
MEM=0.85
MAX_RUNNING=32
CHUNKED=4096
DECODE_STEPS=4
CUDA_GRAPH=""
MAMBA_CACHE=""
CHAT_TEMPLATE=""
REASONING=""
OVERLAP=""
WARMUP=""
WATCHDOG=600
TP=2
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Model presets (tuned for 48GB total VRAM) ---
apply_preset() {
    case "$1" in
        devstral)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="${QUANT:-awq_marlin}"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            WARMUP="--skip-server-warmup"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser mistral"
            ;;
        devstral-32k)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=32768; MEM=0.90; MAX_RUNNING=64; CHUNKED=8192
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser mistral"
            ;;
        devstral-long)
            # Single-user long-context preset: pushes KV ceiling from 131K (default)
            # to ~217K tokens at MEM=0.97 + no CUDA graph/overlap/radix cache.
            # Decode plateaus ~56 tok/s past 131K. Not for multi-user.
            #
            # 2026-05-09: --skip-server-warmup baked in. Devstral 24B is registered
            # as Mistral3ForConditionalGeneration (Pixtral path) so SGLang warmup
            # sends an image-bearing test request through the pixtral image
            # processor; at MEM=0.97 there's <1 GB free per GPU and the image
            # preprocess `torch.stack(images_list, dim=0)` OOMs on warmup,
            # killing the server before /health=200. Skipping warmup defers the
            # first image alloc to actual user requests where it will either
            # succeed (text-only requests don't trip pixtral) or surface an
            # actionable error to the caller.
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.97; MAX_RUNNING=1; CHUNKED=2048
            EXTRA_ARGS="${EXTRA_ARGS} --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache --tool-call-parser mistral"
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            WARMUP="--skip-server-warmup"
            ;;
        coder-reap)
            # Coder-REAP-25B-A3B-W4A16. On TP=1 / 24 GB the piecewise CUDA
            # graph capture (58 num-token shapes at CTX=131072 + CHUNKED=8192)
            # finishes cleanly but the detokenizer worker then hangs on the
            # very first prefill — /health stays 503 with "couldn't get a
            # response from detokenizer for last 20 seconds" forever. Adding
            # --skip-server-warmup alone doesn't fix it (the next real request
            # hits the same hang). --disable-piecewise-cuda-graph is the
            # working workaround (~5-10% TPOT cost at long context vs the
            # graph-on path). For TP=2 once the second 3090 returns this can
            # be removed and the headline 46 tok/s @ 131K should restore.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-REAP-25B-A3B-W4A16}"
            QUANT="auto-round"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        coder-30b)
            # Repointed 2026-05-01 from local Apr-17 self-built AWQ-Marlin to the
            # CT→AWQ-Marlin conversion of `mattbucci/Qwen3-Coder-30B-A3B-AWQ`
            # (R9700-team's Apr-29 calibration). Same Marlin format, newer recipe.
            # Source: hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ (CT format).
            # Conversion: scripts/quantize/convert_moe_ct_to_awq.py --group-size 128.
            #
            # 2026-05-07: bake --disable-piecewise-cuda-graph. Capability sweep
            # found the awq_marlin MoE inference path on TP=1 with piecewise
            # CUDA graph capture takes >>120s for a 20-token completion (GPU
            # at 100% util but throughput collapses to ~1 tok/s from a normal
            # ~80 tok/s). Disabling piecewise CUDA graph drops a 20-token
            # /v1/chat/completions to <1s; full validator basic check at 3.9s
            # (was 0/1 / 120s timeout). Same kernel path is fine on TP=2 with
            # piecewise enabled (where the preset was originally tuned), but
            # TP=1 either needs piecewise off or further investigation of the
            # awq_marlin MoE replay path. Tracked in README Known Issues.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT}"
            QUANT="awq_marlin"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        coder-30b-eval)
            # SWE-bench eval preset: 256K + single-batch CUDA graph, mirrors
            # `coder-reap-25b` so the two models can be benched apples-to-apples.
            # mattbucci/Qwen3-Coder-30B-A3B-AWQ ships compressed-tensors format,
            # not the native AWQ-Marlin layout the local AWQ-Marlin dir uses.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ}"
            QUANT="compressed-tensors"
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        coder-reap-25b)
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        coder-30b-ream)
            # Qwen3-Coder-30B-A3B-REAM-AWQ — Samsung SAIL REAM merge of the
            # Coder-30B-A3B MoE down to 96 experts (~23B params). Native AWQ
            # format. Distinct from `coder-reap-25b` (Cerebras prune of the
            # same coder base) and from `coder-30b` (full 128-expert base
            # in awq_marlin) and `coder-30b-eval` (CT format of base).
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        gemma4)
            # Gemma 4 26B MoE AWQ — repointed 2026-05-09 to canonical HF mirror
            # post patch 023 (gemma4-moe-mlp-no-quant-config) detection upgrade,
            # which routes the dense MLP to AWQ when the checkpoint has
            # quantized mlp.*.qweight (HF mirror + 21B-REAP-v3b case) and to
            # BF16 only when the calibration recipe explicitly ignores mlp.*.
            # Validator 4/4 PASS (basic + thinking + content-aware vision +
            # video) 2026-05-09 with v0.5.11 + patch 023 detection.
            # Same head_dim=256 / Ampere FP8 incompat as gemma4-31b (FlashInfer
            # rejects head_dim=256, triton rejects FP8 E4M3 KV on sm_86) — fix
            # combo: triton attn + KV_DTYPE=auto + disable-cuda-graph + bf16.
            # Patches 024 (vision/audio towers no quant_config) + 025/026 also
            # apply.
            #
            # IMPORTANT: requires checkpoint config arch=Gemma4ForConditionalGeneration
            # (multimodal route). With Gemma4ForCausalLM the language model loads
            # text-only and image_url payloads silently degrade.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/gemma-4-26B-AWQ}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            DTYPE="${_ENV_DTYPE:-bfloat16}"
            # Bumped CTX 4096 → 16384: validate_capabilities.check_thinking sends
            # max_tokens=4096 which overflows 4096 CTX with any prompt.
            CTX=16384; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --tool-call-parser gemma4"
            ;;
        gemma4-31b)
            # Gemma 4 31B Dense AWQ AutoRound (head_dim=256). On 3090 sm_86 the
            # default FlashInfer + FP8 E4M3 KV combo fails twice:
            #   1. FlashInfer rejects head_dim=256 (NUM_MMA_D_QK=32 invalid in
            #      both BatchPrefillWithPaged- and BatchPrefillWithRagged-
            #      KVCacheDispatched), so even --disable-cuda-graph crashes on
            #      first prefill with "Unsupported max_mma_kv: 0".
            #   2. Triton attention works around head_dim but rejects FP8 E4M3
            #      KV on sm_86 (only fp8e4b15 / fp8e5 supported).
            # Combo --attention-backend triton + KV_DTYPE=auto (FP16) is the
            # working pair, validated 2026-05-01 via gemma4-31b-revalidate-Apr29
            # (basic+thinking PASS; vision hallucinates because checkpoint
            # registers as Gemma4ForCausalLM — separate metadata bug, not flags).
            #
            # 2026-05-03: repointed to R9700's HF mirror at
            # mattbucci/gemma-4-31B-it-AutoRound-AWQ. Their HF config carries
            # architectures=Gemma4ForConditionalGeneration (task #63 metadata
            # flip shipped 2026-04-29) AND it's native AWQ format (bits=4,
            # group_size=128) instead of compressed-tensors — SGLang loads it
            # via awq_marlin on Ampere sm_80+ for ~5x faster cold-load (5.2s
            # vs 30s for the local CT) and likely faster decode too. Validated
            # 2026-05-03 at port 23350 / TP=1 / 4K via validate_capabilities.py:
            # 4/4 PASS (basic + thinking + vision-validator-passes-but-degraded
            # + video skipped). Same Gemma 4 vision degradation as the local
            # checkpoint — the format swap doesn't fix the calibration-side
            # vision quality issue tracked in Known Issues. Override with
            # MODEL=$MODELS_DIR/gemma-4-31B-it-AWQ-4bit if the older local CT
            # build is needed for A/B.
            #
            # CACHE GOTCHA: if you previously pulled this HF mirror before the
            # 2026-04-29 metadata flip, your local config.json may still say
            # Gemma4ForCausalLM. Refresh with `huggingface-cli download
            # mattbucci/gemma-4-31B-it-AutoRound-AWQ config.json` or curl -L.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/gemma-4-31B-it-AutoRound-AWQ}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            # Default BF16 — Gemma 4 SigLIP vision tower NaNs in FP16 (same
            # vision base as 26B; FP16 attention softmax overflows past 65504).
            # If this checkpoint's config has architectures=Gemma4ForCausalLM
            # (text-only) and the user doesn't run image_url through it, FP16
            # is technically fine — but BF16 is safe for the multimodal path
            # that opens up after R9700's task #63 (metadata flip on the HF
            # mirror) lands. Override DTYPE=float16 if memory-constrained.
            DTYPE="${_ENV_DTYPE:-bfloat16}"
            CTX=16384; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --tool-call-parser gemma4"
            ;;
        qwen3-vl-moe)
            # Repointed 2026-05-07 from missing $MODELS_DIR/Qwen3-VL-30B-A3B-
            # Instruct-AWQ-4bit (path didn't exist locally — silent broken
            # preset) to the self-cal $MODELS_DIR/Qwen3-VL-30B-A3B-AWQ-native-
            # thinking-vision (single-file native AWQ, full metadata + chat
            # template, build-from-scratch per CLAUDE.md). Override via env
            # `MODEL=$MODELS_DIR/Qwen3-VL-30B-A3B-Instruct-AWQ ./scripts/launch.sh
            # qwen3-vl-moe` for the community 6-shard variant if needed.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-30B-A3B-AWQ-native-thinking-vision}"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen25"
            ;;
        qwen3-vl-32b)
            # Qwen3-VL-32B-Instruct AWQ — 20 GB weights (11 shards). The prior
            # MEM=0.85 / MAX_RUNNING=16 / CTX=16384 defaults OOM cold at TP=1
            # on the KV-pool sizing step (post-weight-load: ~3 GB free, KV
            # pool wants 16 × 16384 × ~24 KB ≈ 6 GB). TP=1-viable defaults
            # below. For TP=2 once the second 3090 returns, override via env:
            # `CTX=150000 MEM=0.85 MAX_RUNNING=16 ./scripts/launch.sh qwen3-vl-32b`.
            #
            # 2026-05-06: repointed from community QuantTrio/Qwen3-VL-32B-
            # Instruct-AWQ-4bit to R9700's self-cal `mattbucci/Qwen3-VL-32B-
            # AWQ` (their task #58 ship, commit 62fa459). Self-calibrated
            # from Qwen/Qwen3-VL-32B-Instruct BF16 base via balanced_thinking_
            # vision recipe (27h GPTQ on CPU). Validated cleanly on Ampere
            # at TP=1 / 4K port 23370: 3/3 PASS validator (basic + vision
            # 'a solid red circle with a black outline centered on a white
            # background' + video 'a red circle moves from the left side of
            # the screen to the right side'); probe_vision STRONG; probe_
            # codegen STRONG 8/8. Per the build-from-scratch rule, R9700's
            # self-cal supersedes the community QuantTrio reference. Override
            # back to QuantTrio with MODEL=$MODELS_DIR/Qwen3-VL-32B-Instruct-
            # AWQ-4bit if you want to A/B against the community reference.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-VL-32B-AWQ}"
            CTX="${_ENV_CTX:-4096}"
            MEM="${_ENV_MEM:-0.93}"
            MAX_RUNNING="${_ENV_MAX_RUNNING:-1}"
            CHUNKED="${_ENV_CHUNKED:-4096}"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen25"
            ;;
        qwen36-vl-reap)
            # Qwen3.6-VL-REAP-26B-A3B-AWQ — Qwen3.6 VL (vision-language) with
            # Cerebras REAP prune to ~26B params (192 experts). Native AWQ.
            # Multimodal serving path same as qwen3-vl-moe; --enable-multimodal
            # required for image inputs.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.85; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --tool-call-parser qwen3_coder"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            ;;
        qwen36-dense|qwen36-27b)
            # Qwen3.6-27B Dense AWQ. Canonical preset name is qwen36-dense
            # (alias: qwen36-27b). The earlier `qwen35` alias was removed
            # because it served a Qwen3.6-family model — misleading.
            # Qwen3.5 and Qwen3.6 are distinct generations: each has its own
            # dense and MoE variants. The Qwen3.5-27B Dense AWQ is available
            # at $MODELS_DIR/hf-mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated;
            # add a separate `qwen35-dense` preset if it ever ships.
            #
            # Repointed 2026-05-01 (second time): now defaults to
            # mattbucci/Qwen3.6-27B-AWQ (R9700's `balanced_thinking_text` recal
            # 2026-05-01, 3/3 PASS basic+thinking+vision on Ampere TP=1 / 4K).
            # Prior default was mattbucci/Qwen3.5-27B-AWQ (Apr-29) which only
            # passed basic+thinking — vision regressed on the validator-patch.
            # Override with `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`
            # if you want the older Qwen3.5 family for A/B testing.
            #
            # Cross-team breadcrumb (R9700 commit 6de2ff9, 2026-05-08): R9700
            # found DECODE_STEPS=32 + per-Linear AWQ + DeltaNet + thinking-mode
            # crashes the scheduler on RDNA4. They reduced their qwen36-27b
            # to =8 (matching qwen36-moe). Ampere may tolerate =32 (different
            # kernel selection); leaving =32 here for now since all 3090 TP=2
            # validation has been at MAX_RUNNING=8 multi-user where higher
            # decode-steps amortize launch overhead. If thinking-mode regresses
            # on Ampere TP=2 once 2nd 3090 returns, try DECODE_STEPS=8.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-27B-AWQ}"
            CTX=32768; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        qwen36-dense-ct)
            # CT-format variant of qwen36-dense (Qwen3.6-27B Dense, compressed-
            # tensors quant instead of native AWQ). Same Qwen3.6 base, different
            # layout — useful for A/B against the AWQ variant since some
            # patches behave differently across quant formats.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-27B-AWQ-CT}"
            QUANT="compressed-tensors"
            CTX=32768; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        qwen35-dense)
            # Qwen3.5-27B Dense AWQ (R9700 self-cal at hf-mattbucci/Qwen3.5-
            # 27B-AWQ-4bit-calibrated). Older sibling of qwen36-dense —
            # Qwen3.5 family, not Qwen3.6. Vision regressed on the
            # validator-patch (basic+thinking only); fine for SWE-bench since
            # codegen is text-only.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated}"
            CTX=32768; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        qwen35-moe)
            # Repointed 2026-05-02 from local Apr-14 REAP-AWQ (broken thinking
            # — Open-Platypus calibration stripped <think> tags) to the recal
            # built with `balanced_thinking_vision` recipe + llava_instruct
            # loader fix (18.22h CPU GPTQ then CT→AWQ). 3/3 PASS Ampere TP=1
            # / 8K: basic finish=stop, thinking 3145 tok finish=stop, vision
            # saw red+circle+round. Cerebras's REAP variant (unlike the
            # atbender Qwen3.6-VL-REAP-26B that strips its tower) retained
            # 333 visual tensors so vision is functional after this recal.
            # 2026-05-03: Repointed to canonical hf-mattbucci/Qwen3.5-28B-A3B
            # -REAP-AWQ name (symlinked to the local descriptive build dir);
            # matches our shipped HF mirror (mattbucci/Qwen3.5-28B-A3B-REAP-
            # AWQ commit 2cf434c8) so launch.sh paths align with HF naming
            # convention.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.5-28B-A3B-REAP-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.80; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=4
            MAMBA_CACHE="--max-mamba-cache-size 4"
            REASONING="--reasoning-parser qwen3"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        qwen3-ream)
            # 2026-05-07: bake --disable-piecewise-cuda-graph for TP=1 cold-fit.
            # Same awq_marlin MoE + piecewise CUDA graph regression as coder-30b
            # (TP=1 inference collapses to ~1 tok/s with piecewise enabled).
            # When TP=2 returns this can be revisited — the original preset
            # (which omits this flag) was tuned for TP=2/256K where piecewise
            # captures cleanly and gives ~74 tok/s headline throughput.
            # Tracked in README Known Issues.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-30B-Instruct-2507-REAM-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            REASONING="--reasoning-parser qwen3"
            # Qwen3-30B-Instruct uses qwen25 JSON-in-<tool_call> format (vs qwen3-coder XML).
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen25"
            ;;
        qwen36)
            # Qwen3.6-35B-A3B (thinking + vision): 256-expert hybrid DeltaNet
            # + gated attn, 3B active, 262K native context. Loads as
            # Qwen3_5MoeForConditionalGeneration with patch 019 + patch 030
            # applied (patch 030 detects pre-sharded w2 at TP>=2 so CT MoE
            # serves cleanly; without it the loader crashes at narrow).
            #
            # 2026-05-09: default switched to mattbucci/Qwen3.6-35B-A3B-AWQ-CT
            # (compressed-tensors). The CT mirror is the calibration-clean
            # variant (0/31010 audit-flagged scales) while the native AWQ
            # has 144 flagged rare-expert findings (evals/awq-audit-2026-05-07.md).
            # TP=2 / 256K verified 4/4 PASS + decode 30-32 tok/s flat 1K-250K
            # (benchmarks/qwen3.6-35b-a3b/v0511-tp2-ct-patch030.json), within
            # 1-3% of native AWQ on the same context grid. Override to native:
            #   MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ \
            #   QUANT=awq_marlin ./scripts/launch.sh qwen36
            #
            # On TP=1 / 24 GB the default CTX=262K OOMs ("Not enough memory")
            # — use the qwen36-tp1 preset variant below for cold-launch on a
            # single card.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT}"
            QUANT="${QUANT:-compressed-tensors}"
            # Force bf16 KV: fp8_e4m3 KV produces garbage on this model via
            # Qwen3_5MoeForConditionalGeneration on Ampere. Env override
            # (KV_DTYPE=X on the command line) still wins.
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            # Cross-team breadcrumb (R9700 commit 6de2ff9): DECODE_STEPS=32 +
            # DeltaNet + thinking crashes scheduler on RDNA4; their fix was
            # =8. Ampere TP=2 thinking-mode 4/4 PASS at =32 (2026-05-09).
            CTX=262144; MEM=0.85; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            # 2026-05-13: bakeoff p13 ran qwen36 x claw at 1/300 = 0.3%. Forensics
            # (286/300 .claw-only diffs) showed the model emits valid
            # <function=NAME><parameter=...>VAL</parameter></function> tool tags
            # but without --tool-call-parser SGLang serves them as raw text;
            # claw treats them as commentary and never executes any edit.
            # Coder-30B/REAP-25B work in claw because their presets already
            # carry this flag. Add it here so qwen36 routes correctly.
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        qwen36-ream)
            # Qwen3.6-REAM-A3B-AWQ — Qwen3.6 base with Samsung SAIL REAM
            # merge to 192 experts. Distinct from `qwen3-ream` (which is
            # Qwen3-30B-Instruct-2507-REAM, an older Qwen3 base with REAM
            # applied) and from `qwen36` (full 256-expert Qwen3.6 MoE).
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-REAM-A3B-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.85; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments ---
# CLI flags are collected into *_OVERRIDE vars first, then applied AFTER the
# preset so CLI always wins. Preset-only fields (MODEL, QUANT, CHAT_TEMPLATE,
# REASONING, CUDA_GRAPH, MAMBA_CACHE, WARMUP) are left to the preset.
PRESET=""
CTX_OVERRIDE=""
PORT_OVERRIDE=""
MEM_OVERRIDE=""
MAX_RUNNING_OVERRIDE=""
DECODE_STEPS_OVERRIDE=""
CHUNKED_OVERRIDE=""
WATCHDOG_OVERRIDE=""
TP_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -17 "$0" | tail -16
            exit 0
            ;;
        --context-length) CTX_OVERRIDE="$2"; shift 2 ;;
        --port) PORT_OVERRIDE="$2"; shift 2 ;;
        --mem-fraction) MEM_OVERRIDE="$2"; shift 2 ;;
        --max-running) MAX_RUNNING_OVERRIDE="$2"; shift 2 ;;
        --decode-steps) DECODE_STEPS_OVERRIDE="$2"; shift 2 ;;
        --chunked-prefill) CHUNKED_OVERRIDE="$2"; shift 2 ;;
        --watchdog) WATCHDOG_OVERRIDE="$2"; shift 2 ;;
        --tp) TP_OVERRIDE="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$PRESET" ]]; then
                PRESET="$1"; shift
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            ;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 <model> [options]"
    echo "Run with -h for available models."
    exit 1
fi

apply_preset "$PRESET"

# Apply CLI overrides after preset so CLI always wins.
[[ -n "$CTX_OVERRIDE" ]] && CTX="$CTX_OVERRIDE"
[[ -n "$PORT_OVERRIDE" ]] && PORT="$PORT_OVERRIDE"
[[ -n "$MEM_OVERRIDE" ]] && MEM="$MEM_OVERRIDE"
[[ -n "$MAX_RUNNING_OVERRIDE" ]] && MAX_RUNNING="$MAX_RUNNING_OVERRIDE"
[[ -n "$DECODE_STEPS_OVERRIDE" ]] && DECODE_STEPS="$DECODE_STEPS_OVERRIDE"
[[ -n "$CHUNKED_OVERRIDE" ]] && CHUNKED="$CHUNKED_OVERRIDE"
[[ -n "$WATCHDOG_OVERRIDE" ]] && WATCHDOG="$WATCHDOG_OVERRIDE"
[[ -n "$TP_OVERRIDE" ]] && TP="$TP_OVERRIDE"

# Resolve chat template (deferred $MODEL expansion)
CHAT_TEMPLATE=$(eval echo "$CHAT_TEMPLATE")

# --- Setup environment ---
activate_conda
setup_nvidia_env

echo "=============================================="
echo "$PRESET — SGLang TP=$TP on RTX 3090"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Model:  $MODEL"
echo "Quant:  ${QUANT:-none}  Context: $CTX  Port: $PORT"
echo "=============================================="

# --- Build command ---
CMD=(python -m sglang.launch_server
    --model-path "$MODEL"
    --tensor-parallel-size "$TP"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_DTYPE"
    --context-length "$CTX"
    --mem-fraction-static "$MEM"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size "$CHUNKED"
    --num-continuous-decode-steps "$DECODE_STEPS"
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
    --disable-custom-all-reduce
)

# Friendly model name for OpenAI-compatible API consumers (eg. opencode).
# Defaults to the preset name; override with SERVED_NAME=foo.
SERVED_NAME="${SERVED_NAME:-$PRESET}"
[[ -n "$SERVED_NAME" ]] && CMD+=(--served-model-name "$SERVED_NAME")

[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")
[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$MAMBA_CACHE" ]] && CMD+=($MAMBA_CACHE)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
[[ -n "$CUDA_GRAPH" ]] && CMD+=($CUDA_GRAPH)
# EXTRA_ARGS lets callers append/override flags (e.g. --disable-cuda-graph,
# --enable-multimodal) without editing the script. Honor it from env.
[[ -n "${EXTRA_ARGS:-}" ]] && CMD+=(${EXTRA_ARGS})

exec "${CMD[@]}"
