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
#   gemma4-31b                                Gemma 4 31B Dense AWQ (thinking+image+video)
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
            # Devstral-Small-2-24B-Instruct-2512 — in-house BF16->GPTQ->AWQ rebuild
            # (mattbucci/Devstral-Small-2-24B-AWQ). The community AWQ degenerated on
            # tool prompts (under-calibrated [TOOL_CALLS] pathway); this rebuild adds
            # function-calling calibration (code_vision_tools recipe). 3/3 PASS
            # basic+tool_call+vision on v0.5.12 (2026-05-28).
            # Uses the canonical Mistral template (with BOS) from devstral2_chat_template.jinja —
            # serving must match the template the model was calibrated with, else the
            # tool pathway degenerates. 2026-05-31: switched the DEFAULT from the EMBEDDED
            # template to scripts/devstral2_chat_template.jinja (R9700 fix) because the
            # upstream embedded template carries an alternation guard that 400s opencode-
            # style agentic flows ([user, assistant(tool_call), tool, user] is mis-counted
            # as user→user and raises "roles must alternate"). The R9700 fix drops the
            # guard ONLY; the formatting loop is identical. Override with
            # DEVSTRAL_CHAT_TEMPLATE="--chat-template <file>".
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Devstral-Small-2-24B-AWQ}"
            QUANT="${QUANT:-awq_marlin}"
            # 2026-05-31: CTX bumped 131K -> 262144 (full native max). Prior 131K
            # was a conservative cap citing "BF16 vision tower eats KV"; VRAM
            # math doesn't actually require that at FP8 KV defaults:
            #   AWQ weights      ~7 GB/card
            #   FP8 KV @ 256K    ~5 GB/card  (40 KB/tok at fp8_e4m3, halved vs BF16)
            #   vision tower     ~2.5 GB total (Pixtral, kept FP16)
            #   cuda graphs      ~1 GB/card
            #   = ~15.5 GB/card, MEM=0.85 budget is 20.4 GB → ~5 GB headroom.
            # The old devstral-long preset (MEM=0.97 + everything-disabled,
            # 217K text-only) is now redundant and was removed in the same
            # commit; if you need to override down, use CTX=131072 env or
            # use coder-30b/etc. presets that already have shorter caps.
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=8192  # MEM 0.85->0.90 (A4): 172K->202K KV, gate 3/3 + tooluse 1.0@132K true
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            CHAT_TEMPLATE="${DEVSTRAL_CHAT_TEMPLATE:---chat-template $SCRIPT_DIR/devstral2_chat_template.jinja}"
            WARMUP="--skip-server-warmup"
            # 2026-05-31: --sampling-defaults model. The model's recommended
            # temperature 0.15 locks Devstral-2 into in-context repetition
            # loops on agentic tool-calling (django-10914: 412 identical glob
            # calls -> timeout-empty; locks in by ~4 repeats, no penalty
            # escapes it). generation_config.json now ships temperature=0.5
            # + repetition_penalty=1.1 (R9700 commit ca22ed8 + HF c0451ce).
            # --sampling-defaults model makes SGLang fill these in when the
            # caller (opencode etc.) omits them — opencode does omit, so this
            # is load-bearing for SWE-bench cycles. Verified empirically:
            # devstral baseline-opencode 2/4 EMPTY at 1300-1400s near-timeout
            # before this fix landed.
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser mistral --sampling-defaults model"
            ;;
        devstral-32k)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=32768; MEM=0.90; MAX_RUNNING=64; CHUNKED=8192
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser mistral"
            ;;
        devstral-long)
            # Deprecated 2026-05-31: the default `devstral` preset now serves at
            # full 262144 ctx (FP8 KV math fits 256K + vision tower + graphs at
            # MEM=0.85). This alias forwards to the same config + flags for
            # backward compat; please use `devstral` going forward.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Devstral-Small-2-24B-AWQ}"
            QUANT="${QUANT:-awq_marlin}"
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            CHAT_TEMPLATE="${DEVSTRAL_CHAT_TEMPLATE:---chat-template $SCRIPT_DIR/devstral2_chat_template.jinja}"
            WARMUP="--skip-server-warmup"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser mistral --sampling-defaults model"
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
            # 2026-05-26: the intended native AWQ-Marlin-from-CT checkpoint was
            # never built; point at the on-disk CT dir and serve as
            # compressed-tensors (Qwen3MoeForCausalLM CT loads cleanly on
            # v0.5.12 — the w2_weight_packed bug is qwen3_5-only). coder-30b-eval
            # is the 256K eval-tuned sibling on the same model.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ-CT}"
            QUANT="${QUANT:-compressed-tensors}"
            # 2026-05-31: CTX bumped 16K -> 262144 (full model max). The prior
            # 16K + MAX_RUNNING=32 was throughput-tuned for batch decode at
            # short ctx; AWQ-int4 + 2x24GB easily fits 256K (R9700 runs this
            # same model class at FP8 / 256K on their 32GB cards — INT4 has
            # half the weight bytes so headroom is comfortable). Override for
            # short-ctx batch benchmarks via `CTX=16384 MAX_RUNNING=32 ./launch.sh coder-30b`.
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            # SPEC_DECODE opt-in: EAGLE3 spec-decode (validated 2026-05-29:
            # 1.65x decode, 185.5 -> 306.0 tok/s, accept_len 4.12 on the wider
            # ladder). MEM 0.85 -> 0.70 to fit draft + cuda graphs at TP=2 on
            # 24GB cards (R9700's full ladder topk16/draft32 OOMs ours; our
            # steps=4/topk=4/draft=8 is the sweet spot that fits). EAGLE3 needs
            # native AWQ (awq_marlin), so callers must also set QUANT=awq_marlin
            # (the preset default is compressed-tensors). Same gate as qwen36.
            # CTX overrides down to 16K under SPEC_DECODE since the draft +
            # cuda graphs don't fit 256K on 24GB cards.
            if [[ -n "${SPEC_DECODE:-}" ]]; then
                MEM=0.70
                CTX=16384
                EXTRA_ARGS="$EXTRA_ARGS \
                    --speculative-algorithm EAGLE3 \
                    --speculative-draft-model-path $MODELS_DIR/drafts/eagle3-coder30b \
                    --speculative-draft-model-quantization unquant \
                    --speculative-num-steps 4 \
                    --speculative-eagle-topk 4 \
                    --speculative-num-draft-tokens 8 \
                    --speculative-attention-mode decode"
            fi
            # NGRAM=1 spec-decode is wired once, post-preset, for the whole
            # pure-attention coder fleet (see after the case) — not per-preset.
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
        coder-reap-25b|coder-reap-30b)
            # 2026-05-14: SWAPPED from Cerebras pre-pruned 25B-AWQ to R9700's
            # in-house Qwen3-Coder-30B-A3B-REAP-AWQ (HF commit d09a18c). Built
            # end-to-end from upstream Qwen/Qwen3-Coder-30B-A3B-Instruct BF16
            # via R9700's homegrown pure-pytorch REAP (no vLLM dep): saliency
            # = Σ_t(gate_t × ‖down_proj_E(x)‖₂) on 1024 evol-codealpaca samples;
            # 128→96 experts/layer (per-layer survivor lists, NOT uniform);
            # then GPTQ W4A16 group_size=128 on code+thinking+math+chat mix
            # with moe_calibrate_all_experts=True. Replaces the broken
            # 2026-04-29 Cerebras-pre-pruned ship per "build-from-scratch"
            # rule. The "25b" alias is kept for backward compat but the model
            # is now 30B-base reduced to 96 experts.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ}"
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
            # combo: triton attn + KV_DTYPE=auto + bf16. CUDA graphs ENABLED 2026-06-10
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
            # 2026-05-31: CTX bumped 16K -> 262144 (model card native max).
            # Prior 16K was set 2026-05-09 to clear validator thinking probe
            # (`check_thinking` sends max_tokens=4096); the 16K default became
            # the de-facto cap but the model has no architectural reason to
            # stay there. R9700 runs gemma-4-26B at FP8 / 32-64K (torch_native
            # is their SWA limit) — we have triton + AWQ-int4 (half weight
            # bytes) so 256K fits comfortably on 2x24GB.
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096  # graphs capture fine at 0.85 with the full 652K pool (B1 G receipt)
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton ${_ENV_GEMMA_GRAPH:---cuda-graph-max-bs 1 --disable-piecewise-cuda-graph} --tool-call-parser gemma4 --swa-full-tokens-ratio 0.0625"
            ;;
        gemma4-21b-reap)
            # Gemma 4 21B REAP AWQ — Cerebras-style expert prune of the 26B
            # parent (smaller A4B-class MoE). Same Gemma 4 architecture as the
            # 26B preset above — patch 023's MLP quant detection was designed
            # to route both the 26B HF mirror AND the 21B-REAP-v3b checkpoint
            # through the same AWQ path. Reuses every gemma4 serving flag:
            # bf16 dtype (SigLIP vision tower NaNs in fp16), triton attention
            # (head_dim=256 FlashInfer-unsupported on sm_86), gemma4 reasoning
            # + tool-call parsers, CUDA graphs ON (bs=1 capture — the 2026-06-10
            # sprint falsified "triton SWA can't capture"; graphs were the whole
            # Gemma decode gap), --swa-full-tokens-ratio 0.0625 (the 26B's
            # right-sizing; default 0.8 starves the full pool). The 21B-REAP
            # weights are ~10 GB (vs 26B's 13 GB) so KV headroom is wider at
            # TP=2. Model card: mattbucci/gemma-4-21B-REAP-AWQ (commit a31a584
            # v3b, regex-ignore recal post the May-7 v2 audit disaster).
            # If not yet downloaded locally:
            #   hf download mattbucci/gemma-4-21B-REAP-AWQ \
            #     --local-dir /data/models/hf-mattbucci/gemma-4-21B-REAP-AWQ
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/gemma-4-21B-REAP-AWQ}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            DTYPE="${_ENV_DTYPE:-bfloat16}"
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton ${_ENV_GEMMA_GRAPH:---cuda-graph-max-bs 1 --disable-piecewise-cuda-graph} --tool-call-parser gemma4 --swa-full-tokens-ratio 0.0625"
            ;;
        gemma4-31b)
            # Gemma 4 31B Dense AWQ — in-house BF16->GPTQ->AWQ rebuild
            # (mattbucci/gemma-4-31B-AWQ). Multimodal: language model quantized
            # to INT4, vision_tower + embed_vision kept FP16 via
            # modules_to_not_convert (same layout as the working 26B AWQ). 5/5
            # capability PASS at 256K — basic + tool + thinking + content-aware
            # vision + video (2026-05-27). Replaces the AutoRound mirror, whose
            # vision hallucinated (text-only calibration).
            #
            # head_dim=256 forces the serving flags below: on 3090 sm_86,
            # FlashInfer rejects head_dim=256 (NUM_MMA_D_QK=32 invalid → first
            # prefill crashes "Unsupported max_mma_kv: 0"), so use
            # --attention-backend triton; triton in turn rejects FP8 E4M3 KV on
            # sm_86, so KV_DTYPE=auto (FP16); graphs ENABLED 2026-06-10 (sprint B1:
            # triton captures fine at bs=1 — the old head_dim concern was wrong).
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/gemma-4-31B-AWQ}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-fp8_e5m2}"  # A5: e5m2 FP8 KV compiles on triton sm_86 (e4m3/fp8e4nv does not) -> 130K->260K pool, retrieval 1.0 to 258K true
            # BF16 — Gemma 4 SigLIP vision tower NaNs in FP16 (attention softmax
            # overflows past 65504). Override DTYPE=float16 only for text-only use.
            DTYPE="${_ENV_DTYPE:-bfloat16}"
            CTX=262144; MEM=0.92; MAX_RUNNING=1; CHUNKED=4096  # MEM 0.92 + e5m2 FP8 KV (A4+A5): 24K->260K pool, 5/5 caps, tooluse 1.0@258K true
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            # 2026-05-31: ship R9700's gemma4_chat_template.jinja override (two fixes
            # vs the upstream embedded template: always close the model turn after
            # tool_call, and open a fresh model turn after tool_response). Without
            # these, opencode's title-gen pattern [assistant(tool_call), tool, user]
            # leaves the prior turn unclosed → runaway to max_tokens=8192 → empty diff.
            # Override with GEMMA4_31B_CHAT_TEMPLATE="--chat-template <file>".
            CHAT_TEMPLATE="${GEMMA4_31B_CHAT_TEMPLATE:---chat-template $SCRIPT_DIR/gemma4_chat_template.jinja}"
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton ${_ENV_GEMMA_GRAPH:---cuda-graph-max-bs 1 --disable-piecewise-cuda-graph} --tool-call-parser gemma4 --swa-full-tokens-ratio 0.05"
            ;;
        gemma4-12b)
            # Gemma 4 12B unified omni (Gemma4UnifiedForConditionalGeneration) —
            # serving-side bringup (patch 043 grafts the upstream gemma4_unified
            # loader onto our v0.5.12 tree). Most KV-efficient Gemma: MQA on the 8
            # global layers (num_global_key_value_heads=1) + 5:1 sliding:full (48
            # layers, window 1024) + attention_k_eq_v → ~8 KB/token, so 256K fits
            # trivially. UNLIKE 26B/31B it has NO heavy SigLIP vision tower — raw
            # pixel/audio patches go straight into LM space via lightweight
            # embedders, so BF16 weights are only ~24 GB and fit 2x24GB at TP=2
            # WITHOUT quantization (this preset validates the loader at BF16; an
            # AWQ int4 build comes later on the separate calib device).
            #
            # head_dim=256 / global_head_dim=512 → same Ampere attn constraints as
            # 26B/31B: FlashInfer rejects head_dim>256, so --attention-backend
            # triton; triton rejects FP8 E4M3 KV on sm_86, so KV_DTYPE=auto (FP16);
            # graphs ENABLED 2026-06-10 (sprint B1: triton captures at bs=1).
            #
            # Serves the -it (instruction-tuned) checkpoint — the deployable one;
            # the base is a raw next-token model. gemma4_unified is a tx-5.10 arch:
            # patches 042/044/045/046 back-port the loader + config + full processor
            # stack (Processor/Image/Audio/Video) so our tx-5.6 env serves it without
            # a fleet-wide transformers bump. reasoning/tool parsers reuse gemma4
            # (chat_template ships in tokenizer_config.json). Override MODEL= for an
            # Defaults to the in-house int4 AWQ (RTN-from-QAT, /data/models/gemma-4-12B-it-AWQ):
            # weights 5.4 GB/rank vs 24 GB BF16, KV cap 102K vs 47K @TP=2, text 4096 quality
            # == BF16. Override MODEL=$MODELS_DIR/hf-google/gemma-4-12B-it for the BF16 -it.
            # Full omni: text + reasoning + tool-call + vision (patch 048 added the unified
            # processor's __call__ image-token expansion — answers "Red" to a red image).
            MODEL="${MODEL:-/data/models/gemma-4-12B-it-AWQ}"
            [ -d "$MODEL" ] || MODEL="$MODELS_DIR/hf-google/gemma-4-12B-it"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            # AWQ weights are int4; BF16 only for the vision-tower numerics (n/a here, no tower).
            DTYPE="${_ENV_DTYPE:-bfloat16}"
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton ${_ENV_GEMMA_GRAPH:---cuda-graph-max-bs 1 --disable-piecewise-cuda-graph} --tool-call-parser gemma4 --swa-full-tokens-ratio 0.0625"
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
            # 2026-05-31: CTX bumped 4K -> 131072 (model card max — 131K is
            # the architectural limit, not a memory wall), MEM 0.93 -> 0.85
            # (fleet default). Prior 4K + 0.93 defaults were TP=1-era cold-
            # launch safety; we're long-since back on TP=2. AWQ-int4 weights
            # are 10 GB/card at TP=2; KV @ 131K × ~24 KB/tok = ~3 GB → comfortable.
            CTX="${_ENV_CTX:-131072}"
            MEM="${_ENV_MEM:-0.85}"
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
            # 2026-05-31: CTX bumped 32K -> 262144 (model card native max),
            # MEM 0.80 -> 0.85 (matches fleet default; DeltaNet replication
            # adds ~3 GB but AWQ-int4 weights are 13.5 GB so headroom is fine
            # at TP=2). Prior 32K was stale TP=1-era conservative.
            CTX=262144; MEM=0.85; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
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
            # Default bumped 4 -> 8 at the v0.5.14 flip (2026-06-26): v0.5.14's
            # extra_buffer mamba-radix cache reserves `mamba_ratio` (=5) slots/req,
            # so max_num_reqs = max_mamba_cache_size // 5; size 4 -> 0 servable (boot
            # RuntimeError). 8 -> 1 req (single-user 256K target). Harmless on the
            # v0.5.13 rollback (just a slightly larger cache). Override QWEN35_MAMBA_CACHE.
            MAMBA_CACHE="--max-mamba-cache-size ${QWEN35_MAMBA_CACHE:-8}"
            REASONING="--reasoning-parser qwen3"
            # 2026-06-07: CUDA graph ENABLED — same stale-disable fix as the
            # qwen36 family (DeltaNet+MoE hybrid; ~4x single-user 256K decode,
            # 5/5 capabilities under graph replay). bs=1 capture, piecewise off.
            CUDA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
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
            # Qwen3_5MoeForConditionalGeneration.
            #
            # 2026-05-26: default switched to native AWQ-Marlin (the fleet's
            # standard path). The prior CT (compressed-tensors) default hit a
            # v0.5.12 MoE-loader bug — KeyError: experts.w2_weight_packed, since
            # patch 028's per-expert key map handles AWQ suffixes (qweight/qzeros/
            # scales) but not CT's weight_packed. Rebuilt from the BF16 base:
            # fresh GPTQ W4A16 (1024-tok x 256-sample, moe_calibrate_all_experts,
            # thinking_vision recipe) -> convert_moe_ct_to_awq -> merge BF16
            # vision tower. 5/5 PASS (basic+tool_call+thinking+vision+video) at
            # TP=2/256K (benchmarks/quality/qwen36-awq-marlin-rebuild-v0512.json).
            # NOTE: the 144 "zero-scale" check_awq_scales flags are inherent
            # base-model expert sparsity (gate/up channels are 7.8e-38 in the
            # BF16 base, faithfully flushed to 0 in fp16 AWQ) — NOT under-cal.
            #
            # On TP=1 / 24 GB the default CTX=262K OOMs ("Not enough memory")
            # — use the qwen36-tp1 preset variant below for cold-launch on a
            # single card.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ}"
            QUANT="${QUANT:-awq_marlin}"
            # Force bf16 KV: fp8_e4m3 KV produces garbage on this model via
            # Qwen3_5MoeForConditionalGeneration on Ampere. Env override
            # (KV_DTYPE=X on the command line) still wins.
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            # Cross-team breadcrumb (R9700 commit 6de2ff9): DECODE_STEPS=32 +
            # DeltaNet + thinking crashes scheduler on RDNA4; their fix was
            # =8. Ampere TP=2 thinking-mode 4/4 PASS at =32 (2026-05-09).
            CTX=262144; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            # 2026-06-07: CUDA graph ENABLED (was --disable-cuda-graph — stale,
            # defensive baggage). The DeltaNet+MoE hybrid captures cleanly on
            # v0.5.12/Ampere TP=2 and validates 5/5 (basic+tool+thinking+vision+
            # video) under graph replay. Single-user decode 31 -> 129 tok/s
            # @262K (4.15x; TPOT 32.1 -> 7.8 ms), flat curve becomes the correct
            # attention-bound one (5.7 ms @1K -> 7.8 ms @262K). bs=1 capture
            # (single-user 256K is the target); piecewise stays OFF (the
            # MoE-marlin TP regression qwen3-ream documents). MEM 0.85 -> 0.80
            # for graph+warmup headroom — the 0.85 config died at the final init
            # step with avail 2.9 GB (KV pool still ~900K >> 262K at 0.80).
            CUDA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
            # 2026-05-13: bakeoff p13 ran qwen36 x claw at 1/300 = 0.3%. Forensics
            # (286/300 .claw-only diffs) showed the model emits valid
            # <function=NAME><parameter=...>VAL</parameter></function> tool tags
            # but without --tool-call-parser SGLang serves them as raw text;
            # claw treats them as commentary and never executes any edit.
            # Coder-30B/REAP-25B work in claw because their presets already
            # carry this flag. Add it here so qwen36 routes correctly.
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            # SPEC_DECODE opt-in: DFlash spec-decode (validated 2026-05-29:
            # 4.10x decode, 30.6 -> 126.3 tok/s, accept_len 5.62). Context cap
            # drops 256K -> 32K and MEM 0.85 -> 0.70 to fit draft + cuda graphs.
            # Requires --dtype bfloat16 (BF16 draft mismatches FP16 target),
            # SGLANG_ENABLE_SPEC_V2=1 + --mamba-scheduler-strategy extra_buffer
            # (DFlash on Qwen3_5MoeForConditionalGeneration boot-rejects without
            # these). Gate via the swebench bench (evals/swebench/bench_swebench_instance_time.py)
            # before enabling for a multi-day re-sweep; gate is median wall
            # speedup >= 1.5x AND no resolved-count regression.
            if [[ -n "${SPEC_DECODE:-}" ]]; then
                MEM=0.70
                CTX=32768
                DTYPE="bfloat16"
                export SGLANG_ENABLE_SPEC_V2=1
                EXTRA_ARGS="$EXTRA_ARGS \
                    --speculative-algorithm DFLASH \
                    --speculative-draft-model-path $MODELS_DIR/drafts/qwen36-dflash \
                    --speculative-draft-model-quantization unquant \
                    --speculative-attention-mode decode \
                    --disable-overlap-schedule \
                    --mamba-scheduler-strategy extra_buffer"
            fi
            ;;
        qwen36-ream)
            # Qwen3.6-REAM-A3B-AWQ — Qwen3.6 base with Samsung SAIL REAM
            # merge to 192 experts. Distinct from `qwen3-ream` (which is
            # Qwen3-30B-Instruct-2507-REAM, an older Qwen3 base with REAM
            # applied) and from `qwen36` (full 256-expert Qwen3.6 MoE).
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-REAM-A3B-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.80; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser qwen3"
            # 2026-06-07: CUDA graph ENABLED — same stale-disable fix as qwen36
            # (Qwen3.6 DeltaNet+MoE captures cleanly @ v0.5.12/Ampere TP=2, 5/5
            # capabilities under graph replay, ~4x single-user 256K decode).
            CUDA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
            EXTRA_ARGS="${EXTRA_ARGS:-} --tool-call-parser qwen3_coder"
            ;;
        nemotron3-omni)
            # Nemotron-3-Nano-Omni-30B-A3B-Reasoning AWQ (mattbucci, calib-device
            # build 2026-06-16). arch NemotronH_Nano_Omni_Reasoning_V3 = Mamba2-
            # Transformer hybrid MoE (23 Mamba + 23 MoE + 6 attn, 128 routed +1
            # shared expert, top-6) + CRADIO vision/video + Parakeet audio.
            # Native AWQ gemm group_size=64; ONLY MoE/MLP quantized — Mamba2,
            # attention, and the vision+audio towers stay BF16 (verified via the
            # safetensors index: layers 0/2/5 qweight=0; vision/radio/encoder
            # qweight=0; check_awq_scales 5934 scales 0-flagged). Thinking ON by
            # default (<think>, reasoning parser nemotron_3). Richest modality set
            # in the catalog — adds AUDIO. Audio needs librosa (installed).
            # Mamba2-hybrid → keep a mamba cache; multimodal → --enable-multimodal
            # + --trust-remote-code (custom processing.py/audio_model.py).
            # ⚠ 256K-on-triton needs R9700 patch 047 (hybrid v_head_dim); trying
            # FlashInfer (default) first — graft 047 if 256K OOMs/errs on triton.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ}"
            CTX=262144; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            MAMBA_CACHE="--max-mamba-cache-size 8"
            REASONING="--reasoning-parser nemotron_3"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --tool-call-parser qwen3_coder --trust-remote-code"
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

# NGRAM=1 opt-in: draft-FREE n-gram spec-decode (CPU trie, no draft weights →
# stays at 256K; lossless — target verifies every token). Does NOT collapse at
# depth like model-draft spec; on the FULL coder-30b it's ~2.6x no-spec @172K
# copy-heavy (receipt benchmarks/ngram-copyheavy-at-depth-2026-06-15.md). Disables
# the overlap scheduler → opt-in (neutral-to-negative on novel-text decode).
# FULL Qwen3-Coder-30B ONLY (coder-30b / coder-30b-eval = the 128-expert base).
# The benefit is gated by the model's COPY FIDELITY (n-gram acceptance):
#  - full coder-30b copies verbatim → accept len 6-7.6 → big win.
#  - REAP/REAM-pruned coders (coder-reap-25b / coder-30b-ream, 96e) copy less
#    faithfully → accept ~2 → NET-NEGATIVE (measured: slower than no-spec at BOTH
#    draft=8 and draft=4), so they're excluded from the allowlist.
#  - DeltaNet thinkers (qwen36*/qwen35-moe): recurrent verify is sequential in
#    num-draft-tokens → net-negative regardless of acceptance (R9700 verify-wall).
# Tune draft tokens via NGRAM_DRAFT (default 8; lower toward the accept len).
if [[ -n "${NGRAM:-}" ]]; then
    case "$PRESET" in
        coder-30b|coder-30b-eval)
            EXTRA_ARGS="$EXTRA_ARGS --speculative-algorithm NGRAM --speculative-num-draft-tokens ${NGRAM_DRAFT:-8}"
            echo "NGRAM spec-decode ENABLED for $PRESET (draft tokens ${NGRAM_DRAFT:-8})" ;;
        *)
            echo "WARN: NGRAM=1 ignored for '$PRESET' — only the FULL Qwen3-Coder-30B presets (coder-30b/coder-30b-eval) benefit. Pruned coders (reap/ream) measure net-negative; DeltaNet (qwen36*/qwen35-moe) is incompatible." ;;
    esac
fi

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
)
# Custom all-reduce stays OFF: it breaks cuda-graph capture on 3090 TP=2
# (sm_86) and graphs are worth far more than the decode allreduce it would speed
# up. Disabled 2026-04-12 (commit 45c4810, v0.5.11); RE-CONFIRMED still broken on
# v0.5.12 2026-06-15 — `Capture cuda graph failed: invalid argument` at
# custom_all_reduce.cuh:508. The ENABLE_CUSTOM_AR=1 toggle is kept so a future
# sglang/driver bump can be re-tested in one flag (receipt:
# benchmarks/allreduce-accel-null-2026-06-15.md). Don't ship it on until capture
# survives. (symm-mem variants are a null result for decode — they touch only the
# embedding allreduce, not the per-layer decode allreduce; same receipt.)
[[ -z "${ENABLE_CUSTOM_AR:-}" ]] && CMD+=(--disable-custom-all-reduce)

# Friendly model name for OpenAI-compatible API consumers (eg. opencode).
# Defaults to the preset name; override with SERVED_NAME=foo.
SERVED_NAME="${SERVED_NAME:-$PRESET}"
[[ -n "$SERVED_NAME" ]] && CMD+=(--served-model-name "$SERVED_NAME")

[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")
[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$MAMBA_CACHE" ]] && CMD+=($MAMBA_CACHE)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
# Thinking serving defaults (adopted 2026-06-07, cross-team with R9700). A
# non-empty REASONING (--reasoning-parser) marks a thinking model; for those:
#  * --sampling-defaults model — use the checkpoint's own recommended sampling
#    (Qwen3.6 ships temp 1.0 / top_p 0.95 / top_k 20) instead of SGLang's
#    generic temp 1.0 / top_p 1.0 / top_k -1 (untruncated tail). The model's
#    top_p/top_k truncation curbs the int4 overthinking / degenerate-repeat
#    failure mode (arXiv:2606.00206) and avoids the temp=0 "</think> Paris
#    </think> Paris…" greedy-decode loop R9700 documented. No token cap, so
#    deep single-user 256K reasoning (qwen36 family, 100% to 255K) is untouched.
#    Only added if the preset/caller didn't already set --sampling-defaults.
#  * STRICT_THINK=1 (opt-in) — --enable-strict-thinking, letting a per-request
#    custom_params.thinking_budget bound the think-loop. ONLY for agentic
#    multi-turn tool-use where int4 thinking spirals without committing the
#    edit (R9700: budget≈300 turned 0→1 applied edits). Deliberately NOT a
#    default: a ~300-token cap would gut our 256K deep-reasoning win. Leave
#    OFF for reasoning / 256K decode workloads.
if [[ -n "$REASONING" ]]; then
    [[ "${EXTRA_ARGS:-}" != *--sampling-defaults* ]] && CMD+=(--sampling-defaults model)
    [[ -n "${STRICT_THINK:-}" ]] && CMD+=(--enable-strict-thinking)
fi
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
# _ENV_CUDA_GRAPH lets a caller A/B the graph-capture config from the env
# without editing the preset (e.g. test cuda-graph-ON on the DeltaNet+MoE
# hybrids whose presets default to --disable-cuda-graph). Empty/unset = preset.
[[ -n "${_ENV_CUDA_GRAPH:-}" ]] && CUDA_GRAPH="$_ENV_CUDA_GRAPH"
[[ -n "$CUDA_GRAPH" ]] && CMD+=($CUDA_GRAPH)
# EXTRA_ARGS lets callers append/override flags (e.g. --disable-cuda-graph,
# --enable-multimodal) without editing the script. Honor it from env.
[[ -n "${EXTRA_ARGS:-}" ]] && CMD+=(${EXTRA_ARGS})

exec "${CMD[@]}"
