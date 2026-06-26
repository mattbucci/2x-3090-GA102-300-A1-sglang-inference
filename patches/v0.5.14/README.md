# Staged SGLang v0.5.14 patch set (25 patches)

Self-contained patch set for the **v0.5.14** stack (env `sglang-v0514`, tree
`/data/sglang-rebase-v0514`). The default stack is still **v0.5.13.post1** (top-level
`../*.patch`, 24 patches); this dir is staged for a future flip and is kept **separate** because
3 of the patches are v0.5.14-specific and do **not** apply to v0.5.13.post1 (verified) — putting
them in the shared top-level dir would silently break a fresh v0.5.13 setup.

## Build / reproduce
```bash
SGLANG_TAG=v0.5.14 \
PATCH_DIR="$REPO_DIR/patches/v0.5.14" \
ENV_NAME=sglang-v0514 \
SGLANG_DIR=/data/sglang-rebase-v0514 \
./scripts/setup.sh
```
Serve / eval the stack by exporting `ENV_NAME=sglang-v0514 SGLANG_DIR=/data/sglang-rebase-v0514`
(common.sh + launch.sh honor both). v0.5.14 pins the same dep stack as v0.5.13.post1
(torch 2.11.0 / transformers 5.8.1 / flashinfer 0.6.12 [cu13] / xgrammar 0.2.1).

## Composition (vs the v0.5.13.post1 top-level set)
- **21 shared** (byte-identical copies of the top-level patches; apply clean to both stacks):
  002 003 004 005 007 011 018 023 025 026 029 030 031 035 037 039 041 049 052 054 055.
- **3 regenerated** for v0.5.14 (still needed; conflict was context drift / partial upstreaming —
  these versions do NOT apply to v0.5.13.post1): **017** (gated-GELU MoE — wNa16_moe refactored to
  the MoeRunner arch; the old `activation=` kwarg sub-hunk is now upstream), **051** (cohere2moe
  hybrid-SWA — archs set grew upstream), **053** (EVS video routing — import-block drift).
- **1 new** (caught by validation): **056** gdn-conv-state-dtype-cast — v0.5.14's mamba-track
  radix path `index_put`s an fp16 conv source into the bf16 conv cache (qwen36/qwen35-moe DeltaNet
  crash on first decode). Sibling of 003. (Applies to v0.5.13.post1 too, but isn't needed there.)

**0 patches were fully upstreamed/archived** this bump (only the one 017 sub-hunk).

## 3-gate pristine replay — ✅ PASS (against pristine v0.5.14)
(a) all 25 apply clean in glob order; (b) result byte-identical to the live `/data/sglang-rebase-v0514`
tree; (c) every patch refuses re-apply on the patched tree (rerun-safe).

## Launch note — qwen35-moe needs a bigger mamba cache on v0.5.14
v0.5.14's `extra_buffer` mamba-radix cache reserves `mamba_ratio` (=5) slots/request, so
`max_num_reqs = max_mamba_cache_size // 5`. The preset's v0.5.13 default `--max-mamba-cache-size 4`
→ 0 servable → boot `RuntimeError`. Launch with `QWEN35_MAMBA_CACHE>=5` (validated at 8 → 1 req);
the launch.sh preset default stays 4 for v0.5.13. Bump the preset default at flip.

Full per-patch verdicts + validation receipts: [`../v0.5.14-rebase-status.md`](../v0.5.14-rebase-status.md).
