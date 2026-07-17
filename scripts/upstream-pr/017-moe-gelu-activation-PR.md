# Upstream PR draft — gated GELU through the Marlin / WNA16 MoE path (our patch 017)

**Status: PREPARED, NOT OPENED** — user green-light required.

## Bug status on main (verified 2026-07-16, `27ad9d1`)
Live, all three files:
- `layers/moe/moe_runner/marlin.py:96` — `assert runner_config.activation == "silu", "Only gated SiLU is supported."`
- `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py:456,560` — same assert, both schemes
- `layers/moe/fused_moe_triton/fused_marlin_moe.py` — dispatches `silu_and_mul` only (no gelu branch)

Precedent: main's marlin.py:97 already relaxed the non-gated branch to
`{"silu", "relu2"}` — adding gated `"gelu"` follows the same pattern.

## PR title
```
[Feature] MoE: support gated GELU through the Marlin / compressed-tensors WNA16 path
```

## PR body
```
### Motivation

Gemma-family MoE checkpoints use gated gelu_pytorch_tanh, but the Marlin MoE
runner and both compressed-tensors WNA16 MoE schemes hard-assert gated SiLU:

    assert runner_config.activation == "silu", "Only gated SiLU is supported."

so any INT4/WNA16-quantized gelu-gated MoE (e.g. Gemma MoE AWQ/GPTQ) cannot
serve through the optimized path at all. The non-gated branch already accepts
{"silu", "relu2"} — this extends the gated branch the same way.

### Modification

- fused_marlin_moe.py: dispatch `gelu_and_mul` when `activation == "gelu"`
  and the MoE is gated (sgl_kernel ships gelu_and_mul already — tanh
  approximation, matching gelu_pytorch_tanh)
- moe_runner/marlin.py + compressed_tensors_wNa16_moe.py (both schemes):
  relax the gated assert to {"silu", "gelu"}

### Validation

In production on our 2x RTX 3090 stack since v0.5.12 across three rebases:
Gemma-4-class MoE AWQ (26B MoE, 21B expert-pruned variant) serving TP=2 at
256K context through exactly this path — long-context retrieval (needle 3/3
at ~250K actual tokens), reasoning probe 94-100%, tool-use 1.0/1.0, MMLU/
HumanEval consistent with the BF16 parent. gelu_and_mul output verified
against the eager gated-gelu reference at model bring-up.
```

## Reference diff
`patches/017-moe-gelu-activation.patch` — hunks apply to main's files with
path/line drift only (targets re-verified present, see Bug status).

## Checklist before opening
- [ ] Fork + branch `feat/marlin-moe-gated-gelu`
- [ ] Regenerate hunks against main (all 3 files verified present 2026-07-16)
- [ ] Check sgl_kernel `gelu_and_mul` import path on main (ours imports from
      `sglang.jit_kernel.activation`; main may have moved it)
- [ ] Run their pre-commit
