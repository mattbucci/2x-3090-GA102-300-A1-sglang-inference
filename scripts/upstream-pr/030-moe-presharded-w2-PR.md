# Upstream PR draft — fused-MoE loader: detect already-presharded expert weights (our patch 030)

**Status: PREPARED, NOT OPENED** — user green-light required.
Acceptance retires local patch 030.

## Bug status on main (verified 2026-07-16, `27ad9d1`)
Live. `layers/moe/fused_moe_triton/layer.py` narrows every incoming expert
weight by `shard_size * tp_rank` (L525/528, L596/599 guard on
`self.use_presharded_weights`), but the ONLY thing that ever sets that flag is
`quark_int4fp8_moe.py:168`. Any non-Quark checkpoint whose fused-MoE w2 is
serialized at per-rank size crashes on rank>0 at TP>=2:

    RuntimeError: start (S*r) + length (S) exceeds dimension size (S)

## PR title
```
[Bugfix] Fused MoE loader: don't narrow expert weights that are already shard-sized
```

## PR body
```
### Motivation

FusedMoE's weight loader assumes every incoming expert weight is global-sized
and narrows it by `shard_size * tp_rank`. The `use_presharded_weights` escape
hatch exists in the layer, but only the Quark int4fp8 scheme ever sets it —
for every other quant format a checkpoint that serializes expert w2 at
per-rank size crashes on rank>0 at TP>=2 with

    RuntimeError: start (S*r) + length (S) exceeds dimension size (S)

We hit this in production with compressed-tensors MoE checkpoints at TP=2
(rank 0 loads fine, rank 1 overflows the narrow).

### Modification

Shape-guard at the narrow sites: if the incoming weight already equals the
per-rank shard size on the shard dim, copy it through unchanged; otherwise
narrow as before.

    if loaded_weight.shape[shard_dim] != shard_size:
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )

This is the format-agnostic version of what quark_int4fp8_moe already does
via `layer.use_presharded_weights = True` (its wrapper even documents the
same situation: "Some models as `lmzheng/grok-1` are already be sharded").
At TP=1 the guard is a no-op (global == shard). A checkpoint presharded for
a *different* TP degree remains an error in both the old and new code.

Alternative if maintainers prefer explicit plumbing: wire
`use_presharded_weights` through the compressed-tensors MoE scheme the way
Quark does. The shape-guard is smaller and covers every format at once.

### Validation

Production on our 2x RTX 3090 (TP=2) since v0.5.13: compressed-tensors AWQ
MoE checkpoints (Qwen3-Coder-30B-A3B class) crash on first load without this
and serve cleanly with it; native-AWQ (global-sized w2) checkpoints take the
narrow path unchanged. Carried across three rebases; re-verified at the
v0.5.15 flip.
```

## Reference diff
`patches/030-fused-moe-w2-presharded-detect.patch` — one hunk in
`fused_moe_triton/layer.py`; main has the same narrow sites (regenerate with
line drift at open time).

## Checklist before opening
- [ ] Fork + branch `fix/fused-moe-presharded-detect`
- [ ] Regenerate hunk against main (narrow sites at L525/L596 as of 27ad9d1)
- [ ] Decide guard placement: both w13 and w2 sites (our patch guards w2 —
      extend to w13 for symmetry if reviewers want)
- [ ] Run their pre-commit
