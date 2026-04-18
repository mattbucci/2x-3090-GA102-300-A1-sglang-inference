# Patches, Fixes, and Historical Findings — 3090

This file collects the details of **what was fixed and why** — per-patch narratives, root-cause notes, and cross-team learnings. The top-level `README.md` keeps only current state; once an issue is closed with a patch, the narrative lives here.

Patches apply in numeric order against SGLang v0.5.10. `scripts/setup.sh` applies every `*.patch` in this directory idempotently.

---

## Per-patch notes

### 001 — upstream-sync (~3,000 LOC)
Cherry-picks from SGLang main that we need for supported architectures: Gemma 4, Qwen3.5 / Qwen3-Next, Triton attention updates, pool_configurator.

### 002 — nvidia-model-fixes (923 LOC)
Marlin shape fallback, DeltaNet TP replication, Gemma4 config fixes.

### 003 — deltanet-triton-dtype-fix (51 LOC)
DeltaNet `conv_state` bf16/fp16 cast fix.

### 004 — gemma4-causal-lm-fix (19 LOC)
CausalLM architectures are text-only even if the config class exposes `vision_config` / `audio_config` as class defaults (Gemma4ForCausalLM uses Gemma4Config). Fix: gate `is_multimodal` on `is_causal_lm_only`.

### 005 — ampere-fp8-triton-fallback (59 LOC)
FP8 KV cache on sm_86. sm_86 Triton emits `fp8e4nv` which is not supported on Ampere; route through a PyTorch fallback. FlashInfer handles FP8 KV for `head_dim ≤ 256`.

### 006 — awq-bf16-activation-support (15 LOC)
BF16 activations with AWQ dequant. Needed for Gemma 4 (hidden_size=5376 → FP16 overflow at layer 2). Marlin kernels accumulate in FP32 so BF16 activations are safe.

### 007 — ampere-deltanet-kernel-tuning (48 LOC)
DeltaNet BV=64 tuning for sm_86. Default `BV=32, num_warps=1` under-utilizes the RTX 3090. Sweep found BV=64 gives **1.57x**:

| Config      | BV  | ms/layer | Speedup |
|-------------|:---:|:--------:|:-------:|
| baseline    | 32  | 0.018    | 1.00x   |
| **BV64-w1** | 64  | 0.011    | **1.57x** |

### 008 — awq-moe-wna16-fallback (64 LOC)
`awq_marlin_moe_repack` doubles peak memory during repacking (old + new tensors coexist). For 128-expert MoE models this adds ~7 GB peak/GPU. Env var `SGLANG_FORCE_MOE_WNA16=1` bypasses Marlin repack and routes to [MoeWNA16 Triton kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/moe_wna16.py). Only works with compressed-tensors format; AWQ packing ≠ WNA16 packing.

### 009 — qwen35-moe-causalLM
Qwen3.5 MoE text-only CausalLM wrapper. Adds logits processor + mrope handling; `setup.sh` auto-patches on apply. Required for the REAP-28B path.

### 011 — triton-attention-fp32 (R9700 backport)
**Root cause:** Triton attention kernels accumulate `e_max`/`e_sum`/`re_scale` in BF16 and `tl.dot()` is called without `out_dtype=tl.float32`. Causes 15% mean error vs FP32 after 128 KV tokens; compounds catastrophically over 60 layers (Gemma 31B). Both RDNA4 and Blackwell SM12.x hit this; Ampere/Hopper tolerate it. **Fix:** FP32 casts throughout the decode + extend kernels. We use FlashInfer (FP32 internally) for most models, so this only bites on models forced onto Triton attention (Gemma 4 if `head_dim=512` workaround ever lands; Qwen3-VL-32B Dense with 64 layers is also a candidate).

### 012 — sliding-window-decode-fix (R9700 backport)
`window_kv_offsets` was captured then discarded at `triton_backend.py:278` — SWA decode computed on full pool indices instead of the window slice.

### 014 — gemma4-reasoning-parser (40 LOC, R9700 backport)
`Gemma4Detector` cherry-picked from unreleased upstream SGLang PR [#21952](https://github.com/sgl-project/sglang/pull/21952). Enables `--reasoning-parser gemma4` for `<|channel>thought` / `<channel|>` markers. Wired into the `gemma4` and `gemma4-31b` launch presets.

### 015 — ct-wna16-dequant-layout-fix
The `CompressedTensorsWNA16` dequant fallback (the path Marlin rejects) assumed `[in//pack, out]` layout and `[num_groups, out]` scales. Real CT layout is `[out, in//pack]` and `[out, num_groups]`. On TP-sharded RowParallel layers this silently produced garbage shapes (Gemma 4 down_proj: in=2112 → 1056/GPU → 33 groups, so scales came out `[out, 33]` but were multiplied as if `[1056, out]`). Rewrote the fallback to keep the native `[out, in]` orientation and transpose once at the final matmul.

### 016 — ct-moe-gelu-triton-route (47 LOC)
Gemma 4 MoE uses gelu activation. `CompressedTensorsWNA16MoE` (Marlin MoE path) asserts silu-only. Sibling `CompressedTensorsWNA16TritonMoE` uses the Triton fused-MoE runner which handles both activations (`moe_runner/triton.py` lines 215/231). Two changes:
- `compressed_tensors.py`: env var `SGLANG_FORCE_CT_MOE_TRITON=1` routes CT MoE to the Triton scheme on CUDA (default path on HIP).
- `compressed_tensors_wNa16_moe.py`: relax SiLU-only assertion in the Triton scheme to `silu | gelu`.

Launches Gemma 4 26B MoE on 3090. Generation quality is still degraded (see Known Issues / top-level README).

---

## Cross-team findings (3090 ⟷ R9700)

The sister RDNA4 project runs the same SGLang v0.5.10 stack. Findings that produced patches or changed how we ship are here; day-to-day sync happens in the two READMEs.

- **BF16 attention precision** affects every new architecture (RDNA4, Blackwell SM12.x). Fix: FP32 accumulation in the online softmax (patch 011).
- **AWQ calibration silently breaks thinking and vision.** Quants calibrated on plain text (Open-Platypus, WikiText2, c4) lose `<think>` stop-token behavior and vision-language alignment. Rule: every new quantized model must validate (a) an image+text roundtrip and (b) a thinking-tagged generation that cleanly terminates, before launch. `scripts/eval/validate_chat_template.py` (static) + R9700's `validate_capabilities.py` (live) are the pre-flight gates.
- **Recommended calibration datasets** (reasoning + vision preserving): `a-m-team/AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`, `LLaVA-Instruct-150K`, `AI-MO/NuminaMath-CoT` (+9.81% GPTQ accuracy vs WikiText2 in R9700 measurements). Recipe builder: `scripts/quantize/calibration_datasets.py` (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`).
- **Chat template is a silent bug magnet.** Devstral community AWQ's template emits BOS → `<unk>`; Gemma 4 community weights ship with no template at all; Qwen3 family needs `temperature ≥ 0.3` to avoid greedy-decode repetition loops. Always render the template with and without `enable_thinking`, and verify `chat_template is not None`.
- **AutoRound > GPTQ > AWQ for INT4 quality** — Intel AutoRound (arXiv 2309.05516) uses SignSGD for 200 iterations to jointly optimize rounding offsets and clipping ranges. Can export to both GPTQ and AWQ formats. [RedHatAI reports 99.4%+ quality](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) on CUDA.
- **DeltaNet layers must stay BF16.** INT4 noise accumulates through the recurrent state `S(t) = g*S(t-1) + delta` and destroys quality. Architectural limit.
- **DeltaNet replication mandatory for TP=2.** Qwen3.5-27B: TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16; the recurrent state compounds it across layers. Fix: replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`. Costs 19 GB/GPU weights replicated, which is why Qwen3.5-27B is context-limited to 32K.
- **Community AWQ fails for DeltaNet** on both rigs. Self-calibrate with GPTQ + CT→AWQ.

---

## MoE quantization lessons

Standard GPTQ/AWQ **fails** for MoE (MoEQuant, ICML 2025):
1. **Inter-expert imbalance.** The router unevenly distributes calibration data; rare experts get zero / garbage calibration. Our Gemma 4 26B baseline: 1/128 experts received calibration data, the rest got inf scales.
2. **DeltaNet/SSM sensitivity.** Recurrent state accumulates INT4 error — must stay BF16.

**Fixes used:**
- Expert-balanced sampling (MoEQuant EBSS, GPTQModel FailSafe).
- Skip recurrent layers from INT4 targets (see `scripts/quantize/quantize_qwen35_28b_moe_reap*.py` `ignore=[...]`).
- **In-memory expert fusion for Qwen3.5 MoE REAP:** BF16 source stores `experts.{id}.{gate,up,down}_proj.weight` per expert, but the HF model class expects fused `experts.gate_up_proj` / `experts.down_proj`. Calibrating without fusing silently produces garbage. Pipeline does the fusion before `oneshot()` runs.

---

## Qwen3.5-27B Technical details

Hybrid DeltaNet (linear attention) + full attention. TP=2 requires replicating all layers to avoid FP16 precision errors accumulating through the DeltaNet recurrence.

**Root cause:** TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16. DeltaNet's `S(t) = g*S(t-1) + delta` compounds this across 48 layers × N tokens.

**Fix:** Replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`.

VRAM per GPU: ~19 GB model (replicated) + 1.27 GB DeltaNet state + 0.92 GB KV cache (FP8) = ~21 GB. Only 32K context fits on a 3090.

**Pipeline bottleneck analysis (pre-patch 007):**

| Operation | ms/model | %  |
|-----------|:--------:|:--:|
| MLP forward | 19.9 | 45% |
| Recurrent update | 8.3 | 19% |
| QKV projection | 7.9 | 18% |
| Output projection | 2.9 | 7% |
| RMSNorm + gating | 2.1 | 5% |
| Conv1d | 1.3 | 3% |
| **Theoretical** | 44.1 | 22.7 tok/s |
| **Actual** | 74 | 13.5 tok/s |

MoE kernel configs generated for RTX 3090 (Triton 3.5.1): `E=128,N=768` (Coder-30B, Qwen3-VL-30B), `E=128,N=704` (Gemma 4), `E=103,N=768` (Coder-REAP). Auto-loaded by `device_name=NVIDIA_GeForce_RTX_3090`.

---

## Gemma 4 notes (Ampere sm_86)

Blocked by FlashInfer on `head_dim=512`. Partial unblock (text path boots) via patches 015 + 016 + `SGLANG_FORCE_CT_MOE_TRITON=1` + `--attention-backend torch_native`. Historical fix list:

1. **FP16 overflow at layer 2** (hidden_size=5376). Fix: `--dtype bfloat16` + patch 006.
2. **CT→AWQ conversion quality poor** — cosine similarity 0.845 on q_proj. Community CT quants generate garbage; self-calibrate.
3. **Missing chat template** — embed jinja into `tokenizer_config.json`.
4. **`num_experts` is None for Dense** — fix: `getattr(config, "num_experts", 0) or 0`.
5. **MLP down_proj shape mismatch** — patch 015 fixes the CT WNA16 dequant layout.
6. **MoE gelu activation** — patch 016 routes to Triton runner which supports gelu.

**Still open:** generation emits `<pad>` tokens — weight-layout or kernel mismatch in the CUDA Triton MoE path, or AWQ calibration that dropped the gelu activation profile. See top-level README Known Issues.

## FlashInfer head_dim support

| head_dim | FlashInfer (sm_86) | Models |
|:--------:|:------------------:|--------|
| 64-256 | Supported | Qwen, Devstral |
| **512** | **Not supported** | **Gemma 4** |

Possible unblock paths: SDPA fallback (partial — patches 015/016 got the text path booting), [FFPA kernels](https://github.com/DefTruth/ffpa-attn-mma), TRTLLM FMHA, or llama.cpp for Gemma 4 serving (reported 80-110 tok/s).
