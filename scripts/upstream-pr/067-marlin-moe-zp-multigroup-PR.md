# Upstream PR draft — Marlin MoE JIT template: zero-point fetch for multi-group k-tiles (our patch 067)

**Status: PREPARED, NOT OPENED** — opening a public PR on sgl-project/sglang is an
outward-facing action; user green-light required. Everything below is copy-paste-ready.

## Why this one next
Re-audited 2026-10-25 against main (`8b9b992e`, 2026-10-23): the bug is live. Any
AWQ / GPTQ-with-zero-points MoE checkpoint whose group size is smaller than the Marlin
tile's `thread_k` (group 32 at every config, group 64 at `thread_k = 128`) dequantizes
every group but the first of each k-tile with the wrong zero points and serves token
salad. main's own end-of-slice re-init (the `else` of `group_blocks >= thread_k_blocks`
in the slice-advance block, ~line 1980) already computes the multi-group `zp_gl_rd`; the
initial computation and the shared-memory write predicate never caught up with it, so the
two halves of the same kernel disagree. Correctness fix, no perf change, one header.

## PR title
```
[Bugfix] marlin_moe: fetch every group's zero points when a k-tile spans several groups (group 32/64 AWQ MoE)
```

## PR body
```
### Motivation

In `python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h` the zero-point
stage layout mirrors the scales (`zp_tb_groups = s_tb_groups`, `zp_sh_stage =
zp_tb_groups * zp_sh_stride`) and `fetch_zp_to_registers()` reads
`sh_zp_stage + cur_group_id * zp_sh_stride`, i.e. one row per group of the current
k-tile. The producer side does not fill those rows:

* the initial `zp_gl_rd` has only the single-group form
  (`zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + ...`), with no
  `threadIdx.x / zp_sh_stride` row term;
* `zp_sh_wr_pred = threadIdx.x < zp_sh_stride` lets only row 0's threads issue
  the `cp_async4`;
* the global pointer still advances by `zp_gl_rd_delta * zp_tb_groups` per stage.

So whenever `group_blocks < thread_k_blocks` (group size 32 with thread_k 64/128,
group size 64 with thread_k 128) groups 1..n of every k-tile dequantize with stale
or never-written zero points. Scales are correct (the `s_*` side has the row term
and `s_sh_wr_pred = threadIdx.x < s_sh_stage`), which is why symmetric checkpoints
and group-128 AWQ never show it. The slice-advance block at the end of the kernel
already re-initialises `zp_gl_rd` with the multi-group form, so the kernel is
internally inconsistent rather than deliberately single-group.

Observed on a group-32 AWQ MoE checkpoint (RecursiveSI/flint, 256 experts, top-6,
sm_86, TP=1 and TP=2): every response was token salad through the fused Marlin MoE
path while the same checkpoint through the Triton `moe_wna16` path and through
dense Marlin (group-32 dense layers) was fine.

### Modification

Make the zero-point producer follow the scale producer:

* `zp_gl_rd` gets the `group_blocks < thread_k_blocks` branch
  (`+ threadIdx.x / zp_sh_stride` row term, `threadIdx.x % zp_sh_stride` column),
  identical to the existing slice-advance form;
* `zp_sh_wr_pred = threadIdx.x < zp_sh_stage` (was `< zp_sh_stride`).

The `group_blocks == -1` and `group_blocks >= thread_k_blocks` paths are byte-for-byte
unchanged, so group-128 / channel-wise behaviour is untouched.

### Validation

Unit check against a reference (dequantize → bf16 matmul) on real AWQ MoE tensors
(2048 × 768 experts, top-6, 8 active experts, 8 / 16 tokens):

| config | max rel. err before | after |
|---|---|---|
| group 32, TP=1 shapes | 124 | 0.0011 |
| group 32, TP=2 shapes | 31 | 0.0014 |
| group 64 (thread_k 64 config on sm_86) | 0.0012 | 0.0012 |
| dense Marlin, group 32 | 0.0003 | 0.0003 |

End-to-end: the group-32 checkpoint serves coherent text at TP=2 / 262144 context with
no decode cost (164 / 148 / 93 tok/s at 1K / 32K / 256K, identical to the group-64
build of the same model). Carried in production since 2026-10-25.
```

## Diff (against main; one header + no behaviour change for g128)

Apply the `marlin_template.h` hunk of our
`patches/067-marlin-moe-zp-multigroup-fetch.patch` (the second hunk, a comment in
`moe_wna16_marlin.cuh`, exists only to bump our JIT build key and should NOT go
upstream — see the checklist). On main the anchor is the `// Zero-points` block right
after `bool s_sh_wr_pred = threadIdx.x < s_sh_stage;` (~line 804–814 at `8b9b992e`).

## JIT cache note for the PR (worth a sentence in the body or a follow-up)
The JIT build key digests the `.cuh` translation unit, not its includes, so a header-only
fix is invisible to an existing `~/.cache/sglang/jit/` entry — users who upgrade keep the
old binary until the cache is cleared. Either hash the include set or mention
`rm -rf ~/.cache/sglang/jit` in the release note.

## Checklist before opening
- [ ] Fork + branch `fix/marlin-moe-zp-multigroup`
- [ ] Re-verify the hunk applies on current main (file last touched `8b9b992e`, 2026-10-23;
      the `// Zero-points` block was not part of that change)
- [ ] Replace our Flint-tensor unit test with a self-contained one for `test/srt/`:
      random `awq` MoE weights at group 32 and 64 through `fused_marlin_moe` vs a
      dequantized reference (our `scripts/eval/test_marlin_moe_zp_groups.py` is the
      template; swap the on-disk tensors for `torch.randint` packed weights)
- [ ] Drop the `moe_wna16_marlin.cuh` comment hunk (ours only)
- [ ] Run their lint/format (pre-commit, clang-format) on the header
- [ ] Link nothing internal; the table above is self-contained
- [ ] Mention issue #44999 (Marlin MoE AWQ g128 intermittently wrong on SM86) as
      *not* covered by this fix — different symptom (g128 is never multi-group)
