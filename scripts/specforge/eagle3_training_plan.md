# EAGLE3 training plan (SpecForge) — for "no public draft" models

**Purpose.** When we ship an AWQ-INT4 target on `mattbucci/*` and no public
EAGLE3/DFlash/Medusa draft exists for it, this is the path to train our own
draft so we can serve with `--speculative-draft-model-path` on the same 2× 3090
box. User-authorized 2026-05-31 ("we can just make our own model"). Currently
applies to task #27 (Nemotron-3-Nano-Omni-30B-A3B-Reasoning) but pattern is
reusable for any future shipped AWQ where the spec-decode lane is empty.

**Status.** PROPOSED scaffold — concrete commands fleshed out when we actually
run this, gated by:
- #11 (current bake-off — needs the GPUs)
- #26 (Nemotron-3-Nano-Omni AWQ shipped — so we have a target to train against)
- #28 (NGRAM trial — if NGRAM clears 1.3× on its own, we may not need to train)

---

## Repository + concept

[`sgl-project/SpecForge`](https://github.com/sgl-project/SpecForge) is the
canonical EAGLE3 / Medusa / Multi-Token Prediction trainer for SGLang targets.
Two modes:

- **Online training.** Target model is loaded alongside the draft on the
  training rig; draft trains against fresh forward passes from the target as
  the prompts stream in. Highest fidelity (acceptance rate closest to ideal),
  but needs more VRAM headroom (target + draft + activation cache).
- **Offline training.** Target is run once to capture hidden states + logits
  to disk; draft trains against the cache. **Documented to work on as few as
  1 GPU.** Cheaper, looser fidelity bound, easier to fit on our 2× 24 GB.
  This is the path we'll use unless online-fidelity becomes a quality blocker.

Reference example scripts in the SpecForge repo:
- `examples/run_llama3.1_8b_eagle3_online.sh`
- `examples/run_llama3.1_8b_eagle3_offline.sh`

Data prep utility: `scripts/prepare_data.py --dataset sharegpt`.

---

## Concrete plan when we run this

### Step 1 — Clone + env

```bash
mkdir -p /data/specforge && cd /data/specforge
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
# Create a SIDECAR conda env so this doesn't poison sglang-v0512 or quant
conda create -n specforge python=3.12 -y
conda activate specforge
pip install -e .[train]
# Inspect what train deps the editable install pulled; expect torch + accelerate
# + datasets + deepspeed (online mode) or just torch + datasets (offline).
```

Sidecar env rationale: SpecForge's training deps will conflict with our pinned
sglang-v0512 (different transformers / accelerate versions). The trained
checkpoint is just safetensors + a small config — no env coupling at serve time.

### Step 2 — Pick the target + capture activations (offline mode)

For Nemotron-3-Nano-Omni-AWQ:

```bash
TARGET=/data/models/hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ
# Use the same env that serves it: sglang-v0512 (so the AWQ-Marlin path runs)
# Offline data capture: feed ShareGPT (or a domain-matched corpus) through the
# target once and snapshot hidden states + logits per token.
python scripts/prepare_data.py --dataset sharegpt --output /data/specforge/data/sharegpt
python scripts/capture_offline.py \
    --target $TARGET \
    --tokenizer $TARGET \
    --dataset /data/specforge/data/sharegpt \
    --output /data/specforge/cache/nemotron3-omni-offline \
    --max-samples 100000 \
    --tp 2  # the target itself runs at TP=2 on our box
```

(Script names above are placeholders — read SpecForge's `examples/run_llama3.1_8b_eagle3_offline.sh` for the exact entrypoints when we run this.)

For Nemotron-3-Nano-Omni in particular: the model has thinking mode (default
ON). Capture should run with thinking ON so the draft trains against the
distribution it'll actually need to predict at serve time. Verify by spot-
checking a captured chunk that `<think>` tokens are present in the prompts.

### Step 3 — Train the draft

```bash
# Single-node, 2× 3090. Offline mode keeps the draft + cache on one device,
# target weights aren't in the loop here.
bash examples/run_llama3.1_8b_eagle3_offline.sh   # template
# ...with overrides for our target name + cache path + draft config size.
```

Typical EAGLE3 draft size: ~360 MB for Llama-3.3-70B target (small head),
~900 MB for Qwen3.5-MoE-DFlash. For 30B-A3B-MoE the EAGLE3 head is likely
~300-500 MB.

**Wall-clock estimate.** Public EAGLE3 receipts for ~30B targets run ~1-3 days
on a single A100-80GB. Our 2× 3090 (24 GB each) is slower per-GPU; expect the
training-loop hot path to be memory-bound on one card while the other holds
the captured cache. Realistic: **2-4 days wall.** Detach via the standard
setsid pattern — calibration is GPU-locked too so we cannot run this
concurrently with another bake-off or model build.

### Step 4 — Validate the draft

Once the draft is on disk:

```bash
# Smoke: load target + draft into SGLang, send 5 prompts, measure accept_len.
MODEL=$TARGET \
DRAFT=/data/specforge/cache/nemotron3-omni-eagle3 \
EXTRA_ARGS="--speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $DRAFT \
            --speculative-draft-model-quantization unquant \
            --speculative-num-steps 3 --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4 \
            --speculative-attention-mode decode \
            --mem-fraction-static 0.70" \
./scripts/launch.sh nemotron3-omni --port 30001
# Acceptance gate: accept_len >= 3.5 on coding+thinking probes, coherent output.
# Anything below = misaligned head + restart with different layer hook points.
```

### Step 5 — Ship

If validation gates clear:

```bash
hf upload mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-EAGLE3 \
    /data/specforge/cache/nemotron3-omni-eagle3 \
    --commit-message "EAGLE3 draft trained against mattbucci/...-AWQ"
```

Update the launch.sh preset to default `SPEC_DECODE=1` to the new draft path,
and add a `swebench-spec-bench` receipt comparing OFF vs ON wall-time per the
gating pattern in `evals/swebench/bench_swebench_instance_time.py`.

---

## Open research caveats

1. **EAGLE3 on Mamba2-hybrid is unproven.** SpecForge's documented support is
   Transformer-only target architectures (Llama, Qwen). The EAGLE3 head reads
   target hidden states at specific layer positions to predict the next token.
   On a 100% Transformer stack those positions line up cleanly. On Nemotron-H
   (52 layers: 23 Mamba2 / 6 Attention / 23 MLP/MoE per `hybrid_override_pattern`)
   the hidden states from a Mamba2 layer have a *different* shape (SSM state +
   per-time-step output) than from an attention layer. Two possible outcomes
   when we run:
   - (a) Hook only the MLP/MoE layer outputs — these are dense vectors matching
     EAGLE3's expected shape. Acceptance rate may be lower (skipping half the
     layers) but the head trains.
   - (b) Hybrid hook scheme that aggregates Mamba2 and attention outputs
     separately, requires a small SpecForge fork.
   - (c) Train it as-is and measure. Honest answer is we don't know until
     we try. R9700 explicitly flagged "unproven on a Mamba-dominant stack" in
     their cross-team banner — they're not solving this either.

2. **Multimodal target activation capture.** Nemotron-3-Nano-Omni has image +
   video + audio encoders feeding into the LLM. EAGLE3 typically trains on
   text-only ShareGPT data. The text-only draft should still help text decode,
   but may regress when a real prompt includes vision/audio tokens (the head
   never saw those activation patterns at training time). Mitigation: blend
   multimodal samples into the capture dataset — same recipe as our AWQ
   calibration mix (`omni_thinking_tools`). Cost: more compute, larger cache.

3. **Reproducibility cost vs ship cost.** A single EAGLE3 training run that
   fails the acceptance gate wastes 2-4 days of both 3090s. Before committing,
   we should already have a NGRAM receipt (#28) — if NGRAM clears 1.3×, we
   don't need to train, and the GPUs are free for the next AWQ build.

---

## Decision rule (for future ships)

When ship pair (AWQ, EAGLE3-draft) for a new model is being planned:

1. Ship the AWQ (the easier deliverable). Document `no spec available` in the README.
2. Run NGRAM trial. If `accept_len >= 3` and wall-time speedup >= 1.3× on a
   5-prompt coding probe, **ship as-is + skip the draft training**.
3. If NGRAM is below the gate AND the model is high-value enough to justify
   2-4 days of GPU time, then run the SpecForge offline-training path here.
4. If NGRAM is below the gate AND the model is low-priority, document the
   spec-decode gap and move on. Future SpecForge / community drafts may fill
   the gap; revisit when one is published.

That rule keeps us from committing irreversible GPU-days on a roll of the
dice. Same risk register applies to any "first-of-its-class" arch (Mamba2,
Jamba, etc.) where the hidden-state hook point is uncertain.

---

## Path A — EXECUTED env setup (2026-06-20, Devstral online)

The SpecForge-pinned env (tx4.57.1/sglang0.5.9) CANNOT load Devstral (`ministral3`
needs tx≥5). Fix: run SpecForge on OUR serving stack instead.

1. **Env:** `conda create --clone sglang-v0513 -n specforge-cuda` (gets sglang 0.5.13
   + tx 5.8.1 + our patches + `ministral3`). Then:
   `pip install --no-deps -e /data/specforge/SpecForge` + `pip install --no-deps yunchang wandb tensorboard`.
2. **One-line SpecForge port to sglang 0.5.13** (only break of ~20 sglang-internal
   imports — the rest resolve): `specforge/modeling/target/eagle3_target_model.py`
   `from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw`
   → `from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw`.
   Verified: `import specforge` OK + `ministral3` registered on this env.
3. **Draft config:** `configs/devstral-24b-eagle3.json` (copy in `scripts/specforge/`):
   hidden 5120, inter 32768, 32 heads / 8 kv / head_dim 128, 1 draft layer, vocab
   131072, draft_vocab 32000, rope_theta 1e8, max_position 32768 (for long-ctx),
   bf16. (Devstral `sliding_window: null` → plain full-attention decoder.)
4. **Online command (target-as-our-sglang, no disk dump):**
   ```
   torchrun --standalone --nproc_per_node 2 scripts/train_eagle3.py \
     --target-model-path $MODELS_DIR/hf-mattbucci/Devstral-Small-2-24B-AWQ \
     --target-model-backend sglang --draft-model-config configs/devstral-24b-eagle3.json \
     --train-data-path <code-weighted jsonl> --output-dir outputs/devstral-24b-eagle3 \
     --num-epochs 10 --batch-size 1 --tp-size 2 --learning-rate 1e-4 \
     --max-length 16384 --cache-dir cache --sglang-mem-fraction-static 0.25
   ```
5. **Data (code-weighted, amendment wants long-ctx):** SpecForge `prepare_data.py`
   has code sets — `opencodeinstruct` / `magicoder-evol-instruct` / `opc` /
   `codealpaca-20k` + general `sharegpt`/`ultrachat`. Curate a code-heavy mix.
6. **Remaining gates:** (a) tiny smoke (1 epoch, max-length 2048, ~64 samples) to
   confirm target loads + a step runs on our sglang 0.5.13; (b) correctness gate —
   harvested hidden states match the serving stack; THEN the multi-day run at
   max-length 16K→32K. Validate accept_len≥3.5 + coherent at ≤64K before ship.

---

## Path A — ONLINE pipeline WORKING on 2×24GB (2026-06-20)

Online EAGLE3 training of the AWQ Devstral target on 2×3090 — the INFRA is solved
(smoke trained real steps end-to-end). SpecForge 0.2.0 is pinned to sglang 0.5.9 +
assumes target/draft co-located; running it on our sglang-0.5.13/tx5.8.1 stack with a
2×24GB split took a series of fixes (full diff: `scripts/specforge/devstral-online-patches/`).

**Env:** `conda create --clone sglang-v0513 -n specforge-cuda`; `pip install --no-deps -e
/data/specforge/SpecForge` + `pip install --no-deps yunchang wandb tensorboard gptqmodel`.

**Target:** text-only extraction (drops the VLM wrapper so it's a plain
`Ministral3ForCausalLM` AWQ — `scripts/specforge/extract_devstral_text_only.py` →
`/data/models/Devstral-Small-2-24B-AWQ-textonly`). Avoids the `Mistral3` VLM AutoModel
rejection + the no-`outputs.logits` base-model issue.

**SpecForge fixes (devstral-online-patches/specforge-0.2.0-devstral-online.diff):**
1. `eagle3_target_model.py` import: `scheduler_dp_attn_mixin` → `scheduler_components.dp_attn` (0.5.13 move).
2. `args.py`: ServerArgs `enable_piecewise_cuda_graph` → `disable_piecewise_cuda_graph` (inverted, 0.5.13).
3. `eagle3_target_model.py` hf loader: VLM fallback (AutoModelForImageTextToText → `.language_model`) — now moot with the text-only target, kept harmless.
4. `eagle3_target_model.py` device placement: `device_map` with `max_memory` cap on cuda:0 (`SPECFORGE_TARGET_GPU0_GIB`, default 5; use 1 to push the whole ~14GB AWQ target to cuda:1) — transformers `tp_plan="auto"` TENSOR-shards AWQ → Marlin-repack break; pipeline/whole-device placement keeps AWQ weights intact.
5. `eagle3_target_model.py` cross-device: move forward INPUTS to the target's embedding device, and the target OUTPUTS (3 aux hidden + logits) back to the training device (`input_ids.device`) — SpecForge assumes co-location; the split needs explicit handoffs.
6. `template.py`: registered `devstral` chat template (`assistant_header="[/INST]"`, `user_header="[INST]"`, `end_of_turn_token=""`) so the loss-mask regex captures Mistral `[/INST]…</s>` assistant spans (llama3 default → all-zero mask → loss=0/grad=0).

**Working layout:** target wholly on cuda:1 (~14GB), draft+optimizer on cuda:0 (~23GB).
Launch: `CUDA_VISIBLE_DEVICES=0,1 SPECFORGE_TARGET_GPU0_GIB=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node 1 scripts/train_eagle3.py --target-model-path <textonly> --target-model-backend hf
--draft-model-config configs/devstral-24b-eagle3.json --chat-template devstral --tp-size 1 ...`

**Remaining (data, not infra):** sharegpt convs break Devstral's strict chat template
(`apply_chat_template` IndexError on non-alternating convs); use clean code-instruct data
(magicoder / opencodeinstruct) — which is also the code-weighted long-ctx mix the real run wants.

---

## Path A — ONLINE pipeline FULLY CONFIRMED (learning) 2026-06-20

Devstral EAGLE3 online training trains end-to-end on 2×3090 with a real signal:
loss 9.54→~5 (≈ln(vocab) start, falling), grad_norm healthy, acceptance_rate
rising; loss_mask.sum non-zero (319/352 etc.). Layout: target cuda:1 (~14GB),
draft+optim cuda:0 (~22GB).

Two final fixes beyond the infra set:
7. `template.py` devstral: `end_of_turn_token="</s>"` (NOT "") — the ACTIVE parser
   (`GeneralParser`, parse.py) builds the assistant-span regex as
   `\[/INST\]([\s\S]*?(?:{end_of_turn}|$))`; end_of_turn is the span TERMINATOR, so
   "" makes it match empty → zero-length span → loss_mask=0 → loss=0/grad=0.
8. ⚠ **CACHE-BUST is load-bearing:** SpecForge caches the preprocessed dataset under
   `cache/processed_dataset` (+ `cache/vocab_mapping`) with a key that does NOT include
   the chat template — so changing `--chat-template` reuses a stale (mask=0) dataset.
   `rm -rf cache/processed_dataset cache/vocab_mapping` after ANY template/data change.

Confirmed working command (smoke; real run = larger code-weighted long-ctx data + max-length 16K→32K):
```
CUDA_VISIBLE_DEVICES=0,1 SPECFORGE_TARGET_GPU0_GIB=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path /data/models/Devstral-Small-2-24B-AWQ-textonly --target-model-backend hf \
  --draft-model-config configs/devstral-24b-eagle3.json --train-data-path <code.jsonl> \
  --output-dir outputs/devstral-eagle3 --num-epochs 10 --batch-size 1 --tp-size 1 \
  --learning-rate 1e-4 --max-length 16384 --cache-dir cache --chat-template devstral
```

---

## Path A — LONG-CONTEXT (16K) online training on 2×24GB — memory engineering (2026-06-20)

The "FULLY CONFIRMED" smoke above was at max-length 2048. Scaling to the
R9700-amendment target (≥16K) hit the 24GB wall on the draft GPU (cuda:0). The
trainer TRUNCATES to --max-length (it does NOT pack), so long training needs
genuinely-long samples (built by `build_longctx_code_data.py`, streaming +
packing OpenCodeInstruct into ~15.2K-token multi-turn code conversations) AND a
series of memory fixes. Four contained changes, each found by reading the actual
OOM traceback (not guessing), make full ttt-length 16K fit:

1. **Teacher reduction chunked + bf16** (`core/eagle3.py::_compute_target_p_padded`).
   `_compute_target_p` casts the FULL-vocab logits to fp32 (seq×131072×4 = 8.6GB at
   16K) and builds draft-vocab softmaxes (seq×32000 ×2) all at once. Fix: keep the
   logits on the TARGET gpu (cuda:1, integration change in `eagle3_target_model.py`
   — stop `.to(_dev)`-ing them to cuda:0), chunk the reduction over the sequence
   (TEACHER_SEQ_CHUNK=1024, pulling only small slices to cuda:0), and store the
   persistent teacher (target_p / target_p_on_draft) in **bf16** (halves the 4.2GB
   result; negligible KL precision loss). Clears the pre-TTT teacher OOM.
2. **MLP gradient checkpoint** (`modeling/draft/llama3_eagle.py::LlamaDecoderLayer`).
   The draft MLP has a 32768 intermediate; across the TTT unroll its gate/up
   activations are the largest training term (~2GB/step at 16K). The MLP is a pure
   function (no cache_hidden mutation, no dropout) → checkpoint it during training.
3. **Per-step logits gradient checkpoint** (`core/eagle3.py` TTT loop).
   compute_logits (norm + lm_head over draft_vocab=32000) is pure; its (seq×32000)
   output is held for backward each step. Checkpoint → recompute in backward.
4. **Acceptance-rate softmax chunked** (`core/lk_loss.py`). THE decisive one. The
   acceptance-rate METRIC did `F.softmax(logits.to(float32))` over the full
   (1,seq,32000) every step — a ~4GB fp32 transient that OOM'd at step ~4
   regardless of ttt-length (a per-step forward metric, not in the unroll). It's
   detached (grad disabled when lk_loss_type=None), so chunk it over the sequence
   (exact). This dropped the failing allocation 1.70GiB → 892MiB.

⚠ Do NOT gradient-checkpoint the draft ATTENTION: it mutates `cache_hidden`
(`cache_hidden[0] = cache_hidden[0] + [k]`) across TTT steps; a backward recompute
would re-append and corrupt the KV. MLP + logits are the safe (pure) checkpoint
targets.

**Layout unchanged:** target wholly on cuda:1 (~14GB model + ~4.3GB logits), draft
+ optimizer + TTT activations on cuda:0. Optimizer = `BF16Optimizer` over only the
TRAINABLE params (embed_tokens is target-copied + frozen via `freeze_embedding()`,
so ~0.83B of the 1.5B draft is optimized). Env vars + extraction (text-only
ministral3 target) all as in the smoke recipe above.
