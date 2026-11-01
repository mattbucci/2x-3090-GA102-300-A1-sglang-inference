# Cache-hit extend latency — the agentic turn shape — 2026-11-01

`scripts/bench/bench_cache_hit_extend.py`: a natural-text prefix is prefilled once (fresh, cache flushed), then requests that reuse it and append N new tokens with `max_tokens=1` are timed (median of 3; the suffix text differs per repeat so only the prefix is a radix hit). Wall time includes HTTP + tokenizing the whole prompt (~50–80 ms at 233K). Shipped presets, TP=2, 300 W. R9700's lead (their patch 107): a 1-token turn on their qwen38 cost 10.1 s (hybrid cache hits only at 8192-token boundaries) and routing ≤ 16-token extends through the split-KV kernel halved a dense preset's turn at 158K.

| preset (backend) | prefix tokens | fresh prefill | +1 token | +16 tokens | +128 tokens | decode step |
|---|---:|---:|---:|---:|---:|---:|
| qwen38 (flashinfer, GDN hybrid) | 35,087 | 14.1 s | 85 ms | 126 ms | 128 ms | 11.6 |
|  | 142,336 | 78.8 s | 88 ms | 285 ms | 304 ms | 13.2 |
|  | 232,797 | 159.9 s | 96 ms | 385 ms | 490 ms | 17.0 ms |
| flint (triton, sinks + SWA) | 32,064 | 3.4 s | 123 ms | 133 ms | 114 ms | 5.4 |
|  | 129,770 | 31.6 s | 318 ms | 338 ms | 344 ms | 5.9 |
|  | 253,261 | 107.4 s | 614 ms | 619 ms | 638 ms | 8.7 ms |
| qwen36 (flashinfer, GDN MoE) | 35,087 | 3.0 s | 76 ms | 126 ms | 118 ms | 3.5 |
|  | 142,336 | 18.9 s | 78 ms | 261 ms | 259 ms | 4.0 |
|  | 232,797 | 44.2 s | 84 ms | 399 ms | 402 ms | 6.7 ms |

## Reading

- **No tail recompute on this stack.** A 1-token turn costs 76–96 ms at every depth on the GDN presets: the hybrid cache checkpoints the recurrent state every 64 tokens here (`mamba_cache_chunk_size` = max(FLA chunk 64, page size 1)), so R9700's 8192-boundary re-prefill does not exist on v0.5.22 + our patches.
- **The cost is the small-query extend kernel, on both backends.** With 16 (or 128) new tokens the step grows ~1.4 ms per 1K tokens of prefix on flashinfer (qwen38 126 → 385 ms, qwen36 126 → 399 ms from 35K to 233K) while a decode step over the same KV is 17 / 7 ms: the prefill kernel with a handful of query rows runs one CTA per (query tile, head), so it walks the whole KV serially per head. On triton (Flint) even the 1-token case pays it — 123 / 318 / 614 ms at 32K / 130K / 253K against an 8.7 ms decode step — because that backend has no 1-query shortcut.
- **What it is worth:** at 233K every follow-up turn carries ~0.3 s (flashinfer) to ~0.6 s (triton) of attention that a split-KV kernel does in ~10 ms-class time. That is the first-token latency of every agentic step at depth, i.e. part of the 256K single-user target; on the bake-off's median 41K prompts it is ~50 ms per turn.

## Next

1. Triton presets (Flint, Gemma 4): port R9700's short-extend route onto our `TritonAttnBackend.forward_extend` — our tree already has the split-KV verify kernel (`verify_splitkv.py`, upstream) and gates it on speculative verify; widen the gate to plain EXTEND with ≤ 16 new tokens on a prefix ≥ 4K at batch 1, pad to a 16-row tile. Flint needs the kernel to take **learned sinks** (one more term in the final LSE merge: `exp(sink − m)` in the denominator, mirroring `extend_attention`'s `deno += exp(sink − e_max)`) and the SWA layers routed with the window KV ranges the backend already builds. Gate: an fp32-reference test at the real shapes (port of R9700's `test_extend_splitkv_route.py`), then the extend bench on flint.
2. flashinfer presets: find why a 1-query extend is fast and a 16-query one is not (plan/scheduler path for tiny `qo` lengths), then either split KV for small extends or batch the new tokens through the decode wrapper.
3. A 16-token extend profile on flint and qwen38 at 131K is queued to confirm the attention share before either change lands.

## 16-token extend profiles at 131K (2026-11-01)

- **flint (triton), 129,770-token cached prefix + 16 new:** GPU span **164 ms** of a 318 ms wall — the triton extend attention `_fwd_kernel` is 108 ms (49 launches; the full-attention layers carry it, the 513-token SWA layers are cheap), 44 ms is idle gap (the extend step is CPU-launch-bound: CPU wall 159 ms ≈ GPU span), Marlin 7.5 ms. The other ~150 ms of the wall is outside the forward (tokenizing a 130K-token prompt, HTTP, scheduling). So the split-KV route can take ~100 of the 164 GPU ms at 131K and ~200 of ~330 at 253K; the remaining per-turn floor is the re-tokenization of the whole conversation, which the chat API re-sends every turn.
- **qwen38 (flashinfer), 142,336-token cached prefix + 16 new, shape warmed first:** GPU span **188 ms** with **120 ms of idle gap** (CPU wall 187 ms = GPU span: the eager extend forward is launch-bound), flashinfer attention only 31 ms (16 layers, 1.9 ms each), Marlin 26 ms. The first 16-token extend after a fresh prefill was a 2.48 s one-off (plan / JIT at the new shape, 768 Marlin launches) and is not the steady state. So on the flashinfer GDN presets the attention kernel is *not* the problem — the per-turn cost is kernel-launch overhead of an un-graphed 64-layer extend, plus ~90 ms of tokenizing the whole prompt. The lever there is a CUDA graph for small extends (`PIECEWISE_GRAPH=1` → `--cuda-graph-backend-prefill`), not the split-KV route.
- **Two different levers, by backend:** triton presets (Flint, Gemma 4) → split-KV route for the attention (108 of 164 GPU ms at 131K); flashinfer GDN presets → graph the small extend (120 of 188 ms).

Per-layer kernel numbers from the fp32-reference test (`test_extend_splitkv_route.py`, qwen38 shape, bf16 cache): split-KV vs extend kernel **2.9× at 8K, 6.5× at 58K, 7× at 197K** (28.2 → 4.0 ms per layer for one new token at 197K), cosine 0.999997 against fp32 in every cell, errors at the bf16 rounding floor on both kernels.

