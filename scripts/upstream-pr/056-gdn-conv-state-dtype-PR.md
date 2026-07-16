# Upstream PR draft — gdn_backend conv-state dtype cast (our patch 056)

**Status: PREPARED, NOT OPENED** — opening a public PR on sgl-project/sglang is an
outward-facing action; user green-light required. Everything below is copy-paste-ready.

## Why this one first
Re-audited 2026-07-12: the bug is still live on sglang main. Any fp16-serving
GatedDeltaNet model (Qwen3.5/3.6 family launched with `--dtype float16`) crashes on the
FIRST decode forward when the mamba-track radix path is active. Smallest, most obvious,
crash-fixing PR in our queue.

## PR title
```
[Bugfix] GDN backend: cast conv-state track write to cache dtype (fp16 models crash on bf16 conv cache)
```

## PR body
```
### Motivation

The mamba-track radix-cache path in the GatedDeltaNet backend `index_put`s the
conv source into the conv-state cache without a dtype cast:

    conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track

When the model is served with `--dtype float16` (common for AWQ int4 checkpoints
of Qwen3.5/3.6-class GDN hybrids), `mixed_qkv` is fp16 while the conv-state cache
is allocated bf16, and the first decode forward crashes:

    RuntimeError: Index put requires the source and destination dtypes match,
    got BFloat16 (dest) ... Half (source)

Reproduces on any GDN model at `--dtype float16` with the mamba radix cache
active (default `--mamba-radix-cache-strategy auto` resolves to the track path),
e.g. Qwen3.5/3.6 AWQ checkpoints; first observed on v0.5.14 (where the
mamba-track write path was introduced) and still present on main.

### Modification

Cast the tracked conv source to the cache dtype at the write site — a no-op when
dtypes already match:

    conv_states[forward_metadata.conv_states_mask_indices] = (
        mixed_qkv_to_track.to(conv_states.dtype)
    )

This is the torch-level sibling of the kernel-side conv_state cast that
causal_conv1d already performs ("conv_state may be bf16 while activations are
fp16 under AWQ").

### Validation

Serving Qwen3.6-35B-A3B (AWQ int4, --dtype float16, TP=2) crashes on first
decode without this change and serves cleanly with it; fleet-validated across
Qwen3.5/3.6 GDN presets at 256K context (needle/tool-use/reasoning probes
unchanged). Carried in production since 2026-06-26.
```

## Diff (against main; the hunk context is stable — verify path
`python/sglang/srt/layers/attention/linear/gdn_backend.py`, the
`has_mamba_track_mask` branch in the decode forward)

Apply our `patches/056-gdn-conv-state-dtype-cast.patch`; on main the same
one-line cast applies at the `conv_states[...mask_indices] = mixed_qkv_to_track`
write (confirmed present at main sweep 2026-07-12, ~line 525).

## Checklist before opening
- [ ] Fork + branch `fix/gdn-conv-state-dtype`
- [ ] Re-verify hunk applies on current main (kernel tree moved for attention
      triton_ops; gdn_backend.py path unchanged as of 2026-07-12 — recheck)
- [ ] Run their lint/format (pre-commit) on the one-line change
- [ ] Link nothing internal; receipts are self-contained in the PR body
