# Upstream PR draft — triton decode stage1 fp32 softmax (narrowed from our patch 011)

**Status: PREPARED, NOT OPENED** — opening a public PR/issue on sgl-project/sglang is an
outward-facing action; user green-light required. Everything below is copy-paste-ready.

## Scope decision (2026-07-16 site audit vs main @ `27ad9d1`)

Our local 011 touches 11 hunks. Most do NOT survive an honest upstream review:

| Local 011 piece | Upstream verdict |
|---|---|
| decode `_fwd_kernel_stage1` fp32 QK/PV sums | **PR-worthy — real behavior fix** (`tl.sum` does not promote; the whole online softmax runs fp16 on the true-MHA path) |
| extend/decode QK `out_dtype=tl.float32` sweep | **Cosmetic** — `tl.dot` fp16×fp16 already accumulates fp32 by default; upstream's own kernel-B site (extend L546) is documentation, not behavior. Not PR-worthy on its own. |
| extend `q.to(k.dtype)` → `k.to(q.dtype)` | **Real but perf-contested**: with fp8 KV cache the current code quantizes live queries to e5m2/e4m3 (unscaled, 2-3 mantissa bits); but upcasting k forfeits fp8 tensor-core dots on sm_89+/Hopper. File as an **issue** (draft below), let upstream weigh. |
| PV `p.to(v.dtype)` → fp32 dot (extend + decode grouped) | **Keep local.** Our own A/B (bf16-PV vs fp32-PV, coder-30b @244K: `pv-precision-ab-*-2026-07-16.json`) showed recall 3/3 BOTH arms and ~1% perf — we have no receipt that survives review. Local precaution only. |
| decode `_fwd_kernel_stage2` `.to(tl.float32)` loads | **No-op upstream** — `attn_logits`/`attn_lse` are allocated `torch.float32` (triton_backend.py). Drop. |

## PR title
```
[Bugfix] Triton decode attention (MHA path): run online softmax reductions in fp32
```

## PR body
```
### Motivation

`_fwd_kernel_stage1` in `python/sglang/kernels/ops/attention/decode_attention.py`
(the non-grouped path, used when `kv_group_num == 1`, i.e. true-MHA models) computes
its online softmax with `tl.sum`, which — unlike `tl.dot` — does NOT promote to an
fp32 accumulator. With fp16 q/k/v the whole chain runs in fp16:

    qk = tl.sum(q[None, :] * k, 1)        # fp16 product, fp16 reduction over Dh
    ...
    p = tl.exp(qk - n_e_max)              # fp16
    acc += tl.sum(p[:, None] * v, 0)      # fp16 product + reduction per block
    e_sum = e_sum * re_scale + tl.sum(p, 0)

Every other attention kernel in the tree accumulates softmax in fp32 (the grouped
decode kernel via `tl.dot`'s default fp32 accumulator; the extend kernels
likewise) — stage1 is the odd one out, and fp16 exp/rescale error compounds with
KV length. fp32 online softmax is the flash-attention reference contract.

### Modification

Cast the reduction operands to fp32 at the two sites (no-op where inputs are
already fp32):

    qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), 1)
    ...
    acc += tl.sum(p[:, None] * v.to(tl.float32), 0)

`p` is fp32 as a consequence of `qk` being fp32, which also makes `e_sum`/`e_max`
handling fp32 throughout, matching the grouped kernel.

### Benchmarks / validation

The kernel is bandwidth-bound; the cast adds no measurable latency on Ampere
(the loads dominate). Carried in production on our 2x RTX 3090 stack since
v0.5.12 across three rebases (v0.5.13.post1, v0.5.14, v0.5.15) with fleet-wide
long-context validation at 250K+ actual tokens (needle + tool-use probes,
`usage.prompt_tokens`-verified).
```

## Diff (against main paths — `python/sglang/kernels/ops/attention/decode_attention.py`,
main L200 + L234; hunks are our 011 hunks 1-2 re-pathed)

```diff
--- a/python/sglang/kernels/ops/attention/decode_attention.py
+++ b/python/sglang/kernels/ops/attention/decode_attention.py
@@ (in _fwd_kernel_stage1, first KV loop)
-            qk = tl.sum(q[None, :] * k, 1)
+            # fp32 reduction: tl.sum does not promote, and fp16 softmax error
+            # compounds with KV length (all other attention kernels are fp32 here)
+            qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), 1)
@@ (same kernel, V accumulation)
-            acc += tl.sum(p[:, None] * v, 0)
+            acc += tl.sum(p[:, None] * v.to(tl.float32), 0)
```

## Companion ISSUE draft (separate, not part of the PR)

Title:
```
Triton extend attention downcasts live queries to fp8 when the KV cache is fp8
```
Body:
```
In `python/sglang/kernels/ops/attention/extend_attention.py`, the prefix-KV QK
dots cast q INTO the cache dtype (L424 `qk = tl.dot(q.to(k.dtype), k)`, same at
L999 in the unified kernel, pe variants at L444/L1019). With `--kv-cache-dtype
fp8_e5m2` / `fp8_e4m3` this quantizes the live query tensor to 2-3 mantissa
bits, unscaled — k at least went through a scaled quantizer at cache-write time,
q gets a raw dtype cast. The newer kernel at L546 (`tl.dot(q, k,
out_dtype=tl.float32)`) keeps q intact, so the tree is currently inconsistent.

We measure retrieval surviving this on Ampere (e5m2 over 128-dim heads averages
out), so this is not a crash report — but it is silent precision loss on the
prefill path for every fp8-KV deployment, and it is invisible to short-context
evals. Upcasting k to q.dtype preserves precision but forfeits fp8 tensor-core
dots on sm_89+; scaling-aware dequant at load would keep both. Raising for a
maintainer decision on the intended contract; happy to PR whichever direction
you pick.
```

## Checklist before opening
- [ ] Fork + branch `fix/decode-stage1-fp32-softmax`
- [ ] Re-verify both hunks apply on current main (kernels moved to
      `python/sglang/kernels/ops/attention/` — confirmed present at main
      sweep 2026-07-16, `27ad9d1`, L200/L234)
- [ ] Run their pre-commit on the change
- [ ] Open the fp8-q-downcast issue separately (do not bundle)
- [ ] After merge: local 011 shrinks to the kept-local pieces (PV fp32 + stage2
      casts + extend QK forms) — regenerate at next rebase
