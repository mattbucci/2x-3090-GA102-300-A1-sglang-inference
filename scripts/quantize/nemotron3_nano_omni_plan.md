# Nemotron-3-Nano-Omni-30B-A3B-Reasoning AWQ build plan

**Source:** [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
**Target ship:** `mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`
**Status:** PROPOSED — blocked on current bake-off (#11) per CLAUDE.md Rule 1 (no concurrent calibration + serving).

## Why this is interesting

| Property | Value | Why we care |
|---|---|---|
| Active / total | 3B / 31B MoE | A3B-class — same regime as our qwen36 + coder-30b winners; AWQ-friendly active params |
| Modalities | text + image + video (2 min) + audio (1 hr) | **Richest modality set in our catalog.** Currently we serve image+video on gemma4 + qwen36, but no audio. This adds audio. |
| Reasoning | default ON, `<think>...</think>`, `reasoning_budget=16384` | Same thinking-mode handling as qwen36/qwen3.5 — known calibration territory |
| Max ctx | 256K | Matches our standard |
| Backbone | Mamba2-Transformer hybrid + 128/192-expert MoE | Same "DeltaNet/SSM cannot INT4" rule applies — Mamba2 layers stay BF16 |
| Sub-encoders | CRADIO v4-H (vision/video) + Parakeet tdt-0.6b-v2 (audio) | Stay BF16 via `modules_to_not_convert`; same recipe as our gemma4-31b vision_tower handling |
| Distillation lineage | Distilled from Qwen3-VL-30B, Qwen3.5-122B/397B, Qwen2.5-VL-72B, gpt-oss-120b | A strong base; calibration just has to preserve, not improve |
| SGLang v0.5.12 | ✅ supported day-0 — `nemotron_h.py`, `nano_nemotron_vl.py`, `nemotron_h_mtp.py`; dispatch at `server_args.py:2280` | Serve side is solved |
| Launch flags | `--reasoning-parser nano_v3 --tool-call-parser qwen3_coder --trust-remote-code` | New preset in launch.sh |

## What does NOT exist yet (and our spec-decode strategy)

| Asset | Status | Implication |
|---|---|---|
| AWQ INT4 of the Omni-Reasoning variant | ❌ none on HF (stelterlab's AWQ is the non-Omni text-only Nano) | **We'd be first to ship.** |
| In-checkpoint MTP head | ❌ Nemotron-3 Nano family doesn't ship MTP — confirmed by R9700 finding empty mtp/eagle shards in BOTH FP8 and BF16 file lists. NVIDIA whitepaper restricts MTP to Super-120B / Ultra | Native MTP path unusable; #21138 NEXTN path is also broken |
| Published EAGLE3 draft | ❌ none | Requires training |
| Published DFlash draft | ❌ none (conceptually possible — Mamba2-hybrid like our DFlash'd qwen36) | Requires training |
| Cross-target draft transfer (e.g. reuse qwen36 DFlash) | ❌ ruled out by R9700 — Mamba2-hybrid + vocab 131072 + hidden 2688 mismatch with Qwen3-30B-A3B | Not viable |
| **NGRAM (`--speculative-algorithm NGRAM`)** | ✅ **in our SGLang v0.5.12 + CUDA-only** (server_args.py:600-608; R9700 explicitly noted this works our side) | **Free win on code workloads — try this BEFORE training a neural draft** |

**Spec-decode strategy (refined per R9700 cross-team banner 2026-05-31):**

1. **Phase 0 — Ship AWQ no-spec.** Baseline expected ~30 tok/s based on similar A3B-MoE models. The receipt becomes the "before" number for any spec improvement.
2. **Phase 1 — Try NGRAM (zero training cost).** N-gram speculative decoding works by predicting next-token candidates from the prompt's own vocabulary; wins are largest on coding/agentic workloads where prompt + output share vocabulary. CLI:
   ```
   --speculative-algorithm NGRAM \
   --speculative-num-draft-tokens 12 \
   --speculative-ngram-max-bfs-breadth 10
   ```
   Optional: `--speculative-ngram-external-corpus-path <coding-corpus>` to seed the n-gram trie from a representative corpus (separate `--sam-budget` required). Typical reported speedup on code: 1.3-1.8×. If NGRAM clears 1.3×, we may not need to train a neural draft at all.
3. **Phase 2 — Train our own EAGLE3 via SpecForge** (only if NGRAM is insufficient OR if multi-domain coverage needed). User-authorized 2026-05-31 ("when we get there we can just make our own model"); realistic cost ~1-3 days wall on 2× 3090. Recipe scaffolding tracked in task #27. Open research question: SpecForge's EAGLE3 training is well-trodden on Llama / Qwen Transformer-only architectures; on a Mamba2-dominant hybrid like Nemotron-H the hidden-feature hook points may need adaptation (the EAGLE3 head reads target hidden states at specific layer positions — these line up cleanly on a 100% Transformer stack but Mamba2 layers have a different state shape).

## Pre-flight findings (task #18, 2026-05-31)

`AutoConfig.from_pretrained(..., trust_remote_code=True)` in our `quant` env
(transformers 5.5.4, llmcompressor 0.10.1.dev92) succeeded. Architecture
class: `NemotronH_Nano_Omni_Reasoning_V3`. Concrete numbers:

**LLM backbone (`NemotronHConfig`):**
- 52 hidden layers, hidden_size=2688, head_dim=128, GQA (32 q-heads, 2 kv-heads)
- vocab_size=131072, max_position=262144 (256K)
- `hybrid_override_pattern = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME'`
  → **M (Mamba2): 23 layers**, **E (MLP/MoE): 23 layers**, **\* (Attention): 6 layers**
- MoE: **128 routed experts + 1 shared expert**, top-6 routing, moe_intermediate_size=1856
- Mamba2: mamba_num_heads=64, mamba_head_dim=64, ssm_state_size=128, conv_kernel=4, expand=2
- `mlp_hidden_act='relu2'` (Nemotron-H squared-ReLU)

**Layer indices:**
- **Mamba (23):** `[0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]` → keep BF16
- **Attention (6):** `[5, 12, 19, 26, 33, 42]` → keep BF16 (per stelterlab rule)
- **MLP/MoE (23) — INT4 candidates:** `[1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]`

**56% of layers stay BF16** by count (Mamba + Attn). MoE experts inside the
23 quantized MLP/MoE layers are where the parameter mass lives, so this still
gives a substantial size reduction (the 31B → ~14-15 GB AWQ target is realistic).

**Top-level wrapper:**
- `max_sequence_length=131072` (Omni recommends 128K production despite LLM 256K) — use **128K in our launch.sh preset** at first, sweep to 256K if quality holds
- Vision: `RADIOConfig` (CRADIO v4-H), vit_hidden_size=1280, separate_video_embedder=True
- Sound: `SoundConfig` (Parakeet tdt-0.6b-v2), 24 layers, hidden=1024, projection_hidden=4096

**Note:** Nemotron-H uses a single `mixer` module per layer (Mamba mixer OR
attention mixer depending on layer type). Generic `re:.*mixer.*` would catch
BOTH and over-quantize attention. Use the per-layer-index pattern below.

## Calibration recipe sketch

Hybrid + multimodal — the recipe needs all of these layered exclusions, all using
the `re:.*pattern.*` regex form (M4 team's discovery — llmcompressor matches at
module-name granularity, bare strings miss descendants).

```python
RECIPE_NEMOTRON3_NANO_OMNI = {
    "targets": ["Linear"],
    "scheme": "W4A16",  # GPTQ INT4, AWQ-compatible
    "group_size": 128,
    "ignore": [
        "lm_head",
        # Per-layer-index ignore for ALL Mamba2 + Attention layers (29 of 52
        # layers, derived from hybrid_override_pattern; see "Pre-flight" above).
        # Nemotron-H uses a single `mixer` module name for BOTH Mamba and
        # Attention layers, distinguished only by layer type — generic
        # `re:.*mixer.*` would over-quantize. Per-layer-index is the safe pattern.
        # Mamba (23 layers):
        "re:^.*\\.layers\\.(0|2|4|7|9|11|14|16|18|21|23|25|28|30|32|35|37|39|41|44|46|48|50)\\..*$",
        # Attention (6 layers, stelterlab rule: keep BF16):
        "re:^.*\\.layers\\.(5|12|19|26|33|42)\\..*$",
        # MoE routing — gates stay BF16
        "re:.*router.*",
        "re:.*\\.gate$",
        "re:.*_gate$",
        # Vision encoder + image projector — CRADIO v4-H stays BF16
        "re:.*vision_tower.*",
        "re:.*radio.*",
        "re:.*image_embed.*",
        "re:.*image_projector.*",
        "re:.*multi_modal_projector.*",
        # Audio encoder + audio projector — Parakeet stays BF16
        "re:.*audio_tower.*",
        "re:.*sound_tower.*",
        "re:.*parakeet.*",
        "re:.*audio_embed.*",
        "re:.*sound_embed.*",
        "re:.*audio_projector.*",
        "re:.*sound_projector.*",
        # Embedding tables stay BF16 by convention
        "re:.*embed_tokens.*",
    ],
    "moe_calibrate_all_experts": True,
}

# Data mix — must include every live modality
DATA_MIX = {
    "thinking":  ("a-m-team/AM-Thinking-v1-Distilled", 0.30),
    "code":      ("iamtarun/python_code_instructions_18k_alpaca", 0.15),
    "math":      ("AI-MO/NuminaMath-CoT", 0.10),
    "chat":      ("HuggingFaceH4/ultrachat_200k", 0.10),
    "image":     ("liuhaotian/LLaVA-Instruct-150K", 0.15),
    "video":     ("lmms-lab/LLaVA-Video-178K", 0.10),
    "audio":     ("mozilla-foundation/common_voice", 0.05),  # may need google/covost2 too
    "tools":     ("NousResearch/hermes-function-calling-v1", 0.05),
}
N_SAMPLES = 1024  # likely needs 2048 for 256-expert MoE calibration mass
SEQ_LEN = 2048
```

Specific calibration env: `quant` (has llmcompressor 0.10+ with `.distributed`),
NOT `sglang-v0512`. Pattern: `conda activate quant + python -u` per the
[detached calibration logging memory].

## Build sequence (after bake-off finishes)

1. **Pre-flight (disk + arch):**
   - `df /data` shows ≥130 GB free (need 62 GB BF16 + 15 GB CT + 15 GB AWQ + 20 GB
     hessians transient = ~110 GB peak; current 158 GB is OK but tight).
   - `df /data` may need pruning — delete old `/data/models/community/<unused>` and
     stale draft caches before download.
   - `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('nvidia/...-BF16', trust_remote_code=True)"`
     to confirm config loads in our `quant` env. Pinpoint Mamba2 module
     prefixes from the loaded config (the recipe's TODO ignore-list completion).

2. **Download** the BF16 base (~62 GB) to `/data/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/`
   via `hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --local-dir ...`
   (detached if it takes >30 min).

3. **Validate base** serves cleanly first on SGLang at TP=2 ctx=32K:
   `MEM=0.80 CTX=32768 ./scripts/launch.sh nemotron3-omni` (new preset). Smoke
   with a single text + image + audio probe before quantizing. (Catching arch
   issues at BF16 saves a 12-20h calibration mistake.)

4. **Calibrate** via a new `scripts/quantize/quantize_nemotron3_nano_omni.py`,
   modeled on `quantize_devstral2_code_vision_tools.py` (Mistral3 with image
   modality) but with the recipe above. Loader is `AutoModelForCausalLM` with
   `trust_remote_code=True`; if vision/audio heads need a different loader,
   route through `AutoModelForVision2Seq` or similar — discover at step 3.

   - Detach via `setsid + conda activate + python -u + tee` (no `conda run`).
   - Expected runtime: 12-20 h based on devstral (3 h) × ~5 for the larger
     MoE + multimodal data mix. Single GPU; serving must be down.

5. **Convert** CT → AWQ-Marlin via `scripts/quantize/convert_moe_ct_to_awq.py`
   (the per-expert MoE variant — qwen36/qwen36-ream pattern, not the dense
   gemma4-31b pattern).

6. **Verify scales** with `scripts/eval/check_awq_scales.py`. Non-zero exit
   means do not ship (CLAUDE.md hard rule).

7. **Validate capabilities** with `scripts/eval/validate_capabilities.py` +
   add an audio probe (new). Need 6/6: basic + thinking + image + video + audio
   + tool-call. This is the most modalities we've validated; the audio
   harness will need writing/adapting.

8. **Ship** to `mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ`.
   Use `hf upload` (≤25 GB, this should fit). Include `chat_template.jinja`
   (pre-checked for the same alternation-guard / unclosed-turn class of bugs
   we just shipped fixes for on devstral + gemma4).

9. **Wire preset** in `scripts/launch.sh`: `nemotron3-omni` preset with
   `--reasoning-parser nano_v3 --tool-call-parser qwen3_coder --trust-remote-code`
   + the standard TP=2 / MEM=0.85 / CTX=262144 numbers, KV dtype matching
   our existing hybrid presets.

10. **Spec-decode investigation** (separate receipt under
    `benchmarks/quality/nemotron3-omni-specdec-investigation.json`):
    - Confirm no published draft (likely still true at calibration time).
    - Evaluate the cost of training one (EAGLE3 + SpecForge + a few hundred
      GPU-hours of activation data). Decide: train, defer, or document
      as "baseline only".
    - If we defer, leave the preset baseline-only; the hybrid arch should
      get reasonable single-user decode tok/s on its own (similar regime
      to our qwen36 baseline, ~30 tok/s).

## Disk / serialization risk

Calibration intermediates are large. Discipline:
- Keep BF16 base on `/data` (62 GB).
- CT calibration output on `/data` (~15 GB).
- AWQ conversion output on `/data` (~15 GB).
- Delete BF16 base only after AWQ + spec investigation are both shipped (we
  may need to re-calibrate; redownload is 62 GB).
- Calibration hessian cache (~20 GB) goes to `/tmp` (tmpfs, 31 GB free) or
  `/data/calibration-tmp/` and is deleted at end.
- Concurrent bake-off prediction files also write to `/data` — measure
  available space before starting.
