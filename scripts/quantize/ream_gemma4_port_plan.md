# Samsung SAIL REAM → Gemma 4 port plan (task #35)

**Status:** PROPOSED — well-scoped 1-2 week project. Unblocks task #31
(`mattbucci/gemma-4-26B-A4B-REAM-AWQ`), which is the #2 MoE coverage gap.

## Background — current state

The Samsung SAIL REAM repo at `/data/ream-repo/` is **Qwen3-family-only**.
Hard-coded assumptions live across the codebase:

| File | Line(s) | Assumption |
|---|---|---|
| `merge.py` | 137-141 | `for sfx in ['qwen3', 'glm']:` — model-name suffix matcher; raises if neither matches |
| `ream/merger.py` | 38 | `tokenizer_name: str = 'qwen3'` default |
| `ream/merger.py` | 59 | docstring: `tokenizer_name: qwen3 or glm` |
| `ream/merger.py` | 154-160 | MTP-layer builder selects `qwen3_5` or `qwen3` paths only |
| `ream/saliency.py` | 50 | "copied from the qwen3 moe forward pass" — qwen3 expert forward shape |
| `ream/weight_utils.py` | 69 | `has_gate_up = hasattr(ffn, 'gate_up_proj')  # for qwen3.5` — Qwen3-fused FFN check |

Our 3090 wrapper `scripts/quantize/run_ream_qwen3moe.sh` adds two more
Qwen3-specific pieces:

- `patches/qwen3moe_unfused_experts.py` — monkey-patches `Qwen3MoeExperts`
  to use unfused `ModuleList[Qwen3MoeMLP]` so the per-expert checkpoint
  keys load instead of silently dropping into random-init fused params.
- `ream-patches/001-merger-skip-hid-act-and-checkpointing.patch` — adds
  resume-from-checkpoint + per-layer snapshot support (Qwen3 layer count
  defaults).

## Gemma 4 26B A4B MoE — what's different

`google/gemma-4-26b-a4b-it` (the upstream BF16 base) has:

1. **Parallel dense+MoE decoder layer** (patch 023's case). The decoder
   layer runs `Gemma4MLP` (dense) AND `Gemma4MoeBlock` (sparse) **in
   parallel** and sums their outputs. REAM merging applies only to the
   MoE half.
2. **Per-expert unfused** `{gate_proj, up_proj, down_proj}` linears (not
   fused like Qwen3's `gate_up_proj` + `down_proj`). Our
   `qwen3moe_unfused_experts.py` monkey-patch trick **doesn't apply** —
   Gemma 4 is already unfused.
3. **Router uses `.proj`** (not `.gate`) — caught by patch 023's detection.
4. **gelu activation** (Qwen3 uses silu).
5. **Vision tower coupling** via `multi_modal_projector`. REAM doesn't
   touch vision but the merged checkpoint must preserve vision tensors.
   Existing `merge_vision_weights.py` (used for qwen36-ream) is reusable
   post-merge.
6. **No MTP head.** Gemma 4 has no MTP — skip qwen3_5_mtp builder logic.

## Concrete porting steps

### Step 1 — Branch / patch infrastructure

Either fork the REAM repo with a `gemma4-port` branch, or maintain the
diff as additional patches in `ream-patches/`.

### Step 2 — Add Gemma 4 arch detection

`merge.py` line 137: extend the suffix matcher.

```python
# WAS:
for sfx in ['qwen3', 'glm']:
# NEW:
for sfx in ['qwen3', 'glm', 'gemma-4', 'gemma4']:
```

`ream/merger.py` line 38: accept `'gemma4'` as a tokenizer_name choice.
Add a branch in MTP-builder logic (line 154-160):

```python
elif 'gemma4' in self.model.__class__.__name__.lower():
    build_mtp_layer = None  # Gemma 4 has no MTP head; skip
```

### Step 3 — Saliency adapter for Gemma 4 router

`ream/saliency.py:50`: the "qwen3 moe forward pass" copy. Need a Gemma 4
equivalent that reads `router.proj` output (not `gate`) and iterates the
Gemma 4 expert list (`block.experts[i]` with separate `gate_proj`/
`up_proj`/`down_proj` rather than Qwen3's fused `gate_up_proj`).

```python
def gemma4_expert_forward(block, hidden_states, expert_indices, routing_weights):
    router_logits = block.router.proj(hidden_states)
    # ... top-k routing identical to Qwen3 structurally
    for expert_idx in selected_experts:
        e = block.experts[expert_idx]
        gate = e.gate_proj(hidden_states_for_expert)
        up = e.up_proj(hidden_states_for_expert)
        hidden = nn.functional.gelu(gate) * up  # gelu, not silu
        out = e.down_proj(hidden)
        # saliency: routing_weight × ||out||₂
        ...
```

### Step 4 — Merging adapter

`ream/merger.py` logit+weights merging: average across experts in a
group. For Gemma 4, the `gate_proj`/`up_proj`/`down_proj` weights average
independently (no fused 3D tensor to reslice). Simpler than Qwen3's case.

`ream/weight_utils.py:69` (`has_gate_up = hasattr(ffn, 'gate_up_proj')`):
add a Gemma 4 branch that returns False (no fused gate_up). The
downstream code paths need branching on this flag throughout.

### Step 5 — Wrapper script

New file `scripts/quantize/run_ream_gemma4.sh` modeled on
`run_ream_qwen3moe.sh` but without the `Qwen3MoeExperts` monkey-patch
(Gemma 4 is already unfused). Usage:

```bash
./scripts/quantize/run_ream_gemma4.sh \
    --model ~/AI/models/google/gemma-4-26b-a4b-it \
    --merge_size 80 \
    --save_path /data/models/gemma-4-26B-A4B-REAM-BF16
```

### Step 6 — Validate REAM output before AWQ recal

After the merge produces the REAM BF16, smoke-test:

1. `transformers AutoModelForCausalLM.from_pretrained` loads cleanly
   (no UNEXPECTED keys, no random init in the experts).
2. Short generation test produces coherent text (not the "sweat sweat
   aster aster" gibberish the qwen3 unfused-experts bug produced
   pre-monkey-patch).
3. Validation on a few image+text examples confirms vision tower still
   wired correctly.

### Step 7 — End-to-end ship

Per the pattern of `project_gemma4_31b_shipped.md`:

1. REAM merge: Gemma-4-26B-A4B (103e) → Gemma-4-26B-REAM-A4B (~80e)
2. AWQ calibration: GPTQ W4A16 + `code_vision_video_audio` recipe + regex
   `ignore` (vision_tower, multi_modal_projector, audio_tower, router,
   embeddings stay BF16)
3. CT → AWQ-Marlin conversion via `convert_moe_ct_to_awq.py`
4. `check_awq_scales.py` audit (non-zero exit = do not ship)
5. 5-modality validation
6. Ship to `mattbucci/gemma-4-26B-A4B-REAM-AWQ`
7. Wire `gemma4-ream` preset in `launch.sh`

## Estimated effort

| Step | Hours |
|---|---|
| 1. Branch / patch infrastructure | 1 |
| 2. Arch detection | 1 |
| 3. Saliency adapter | 8-16 (core work) |
| 4. Merging adapter | 4-8 |
| 5. Wrapper script | 1 |
| 6. Validate REAM BF16 output | 4 |
| 7. End-to-end ship | 20-30 |
| **Total dev** | **40-60 h** |

Plus the AWQ calibration GPU time (12-20 h CPU calibration + 2 h
convert + 1 h validation = ~16 h GPU-locked).

## Test plan

Before committing to the full 26B run, test on a SMALLER scope:

1. Start from a downsampled-experts test model (set num_local_experts
   to 10 via config override on a copy of the BF16 base).
2. Run REAM merge with `--merge_size 5` — verify the merged model loads
   + generates.
3. Compare known good prompt output: REAM-merged vs original.

Only then run the full 103 → 80 merge on Gemma-4-26B-A4B.

## Open questions

- Does Gemma 4's parallel dense+MoE structure need special handling?
  Likely no — REAM only touches MoE experts; the dense MLP weights pass
  through untouched. Verify before committing.
- What's the right `merge_size`? Qwen3-Coder-30B-A3B went 128 → 96;
  Qwen3.6-35B went 256 → 192. Gemma 4 26B has 103 experts; reducing to
  ~80 keeps the same ~25% pruning ratio. Start there, sweep if needed.
