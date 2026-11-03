# Gemma 4 31B prefill: V-split of the 512-wide heads in the triton extend kernel — NULL (2026-11-02/03)

Idea: each program computes one 256-wide half of the value/output head dim (grid axis = head × 2), so the
fp32 accumulator and the staged V tile halve and the global-attention layers (head dim 512, forced to a 32×32
tile by the 99 KB shared memory on sm_86) can run a wider tile. QK and softmax are recomputed per half.
Env-gated experiment `SGLANG_EXTEND_ATTN_DV_SPLIT=2` + the patch-075 tile override; the exactness test
(`scripts/eval/test_extend_dv_split.py`, kept) shows split-2 bit-identical to the unsplit kernel on full,
sliding-window and GQA cases.

Fresh-prefill TTFT (natural prompt, cache flushed), greedy 131K needle 3/3 on every arm that booted:

| arm | 32K (34.0K actual) | 131K (132.7K actual) |
|---|--:|--:|
| shipped kernel (32×32 / 8 warps, no split) | 17.63 s | 145.27 s |
| split 2, same 32×32 tile (pure recompute cost) | 22.40 s (+27 %) | 219.58 s (+51 %) |
| split 2, 32×64 / 8 | 18.39 s (+4 %) | 159.46 s (+10 %) |
| split 2, 64×64 / 8 | does not compile: shared memory 163,840 B > 101,376 B | — |

The recompute penalty grows with the prefix (the prefix loop is QK-heavy), i.e. the kernel is compute-bound
at depth, not occupancy-bound; the widest tile that fits recovers most of the penalty but never beats the
shipped kernel. Combined with the earlier head-split sweep (QK over two 256-halves, −3..−7 %) and the D=256
sweep (64×64 is the shared-memory ceiling), the tile/split family is exhausted for Gemma 4 prefill: the lever
has to change the per-tile work (e.g. a different attention formulation for 512-wide heads), see issue #12.
The experiment code was removed from the tree; receipts `base.json`, `s2-32-32-8.json`, `s2-32-64-8.json`.
