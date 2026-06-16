# Allreduce acceleration for qwen36-dense — NULL/BLOCKED on 3090 TP=2 (2026-06-15)

**Question.** `qwen36-dense` (Qwen3.6-27B Dense+DeltaNet AWQ) is the slowest decoder; the README
Decode-ideas listed "symm-mem/MSCCLPP for the dense-TP allreduce (~3.2 ms/token; flashinfer fusion was
a null result)" as a lever. Can any allreduce-accel path speed up single-user decode?

**Why it looked promising.** qwen36-dense decode is nearly flat in context (DeltaNet layers don't grow
with KV) — baseline single-user (M=1, server-log gen throughput): **@2K 67.6 t/s, @64K 61.6 t/s**. The
per-step TP allreduce is a meaningful constant fraction (~3.2 ms of a ~15 ms step @2K).

## Result — every allreduce-accel path is null or blocked

| Path | Flag | Outcome |
|---|---|---|
| Custom all-reduce (one/two-shot) | (remove `--disable-custom-all-reduce`) | **BLOCKED** — breaks cuda-graph capture: `Capture cuda graph failed: invalid argument` @ `custom_all_reduce.cuh:508`. Same failure as 2026-04-12 (v0.5.11), **re-confirmed on v0.5.12**. cuda graphs are worth +25% (commit 45c4810) ≫ the allreduce, so it stays disabled. |
| Torch symmetric memory | `--enable-torch-symm-mem` | **NULL** — @2K 67.4 (vs 67.6), @64K 61.6 (vs 61.6), identical within noise. symm-mem is applied to the **embedding** all-reduce (`vocab_parallel_embedding.py: use_symmetric_memory`), not the per-layer decode allreduce → decode untouched. |
| NCCL symmetric memory | `--enable-symm-mem` | **BLOCKED (build)** — the JIT `nccl_allocator` cpp extension fails to link: `/usr/bin/ld: cannot find -lnccl` (libnccl ships only as `nvidia/nccl/lib/libnccl.so.2`, no unversioned `.so`). Even if linked, it's the same embedding-only path → would be null for decode. |
| FlashInfer allreduce fusion | `--enable-flashinfer-allreduce-fusion` | NULL (prior README finding). |

## Conclusion
**No allreduce-accel lever helps single-user dense decode on 3090 TP=2 (sm_86).** The one path that
targets the per-layer decode allreduce (custom all-reduce) is mutually exclusive with cuda-graph capture
on this hardware, and cuda graphs win more. The symm-mem variants only touch the embedding allreduce, so
they're null for decode. Lever **closed** — removed from the README roadmap. The `ENABLE_CUSTOM_AR=1`
toggle in `launch.sh` is kept (one-flag re-test) for a future sglang/driver bump that might fix the
capture incompatibility. Baselines (67.6 @2K / 61.6 @64K, M=1) recorded for future reference.
