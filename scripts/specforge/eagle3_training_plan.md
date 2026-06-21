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
