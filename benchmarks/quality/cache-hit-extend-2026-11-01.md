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


## Flint served with the split-KV route (patch 070, 2026-11-01)

`SGLANG_EXTEND_SPLITKV_MAX_TOKENS=16` on the shipped `flint` preset (triton, e5m2 KV, TP=2, 300 W); same bench as the table above, same server for the gate. Receipts: `cache-hit-extend-flint-splitkv16-2026-11-01.json`, `flint-splitkv16-2026-11-01.json`, `flint-splitkv16-needle-2026-11-01.json`.

| flint, cached prefix | +1 token before → with route | +16 tokens | +128 tokens (not routed) |
|---|---:|---:|---:|
| 32,064 | 123 → **124 ms** | 133 → 133 ms | 114 → 127 ms |
| 129,770 | 318 → **269 ms** | 338 → 268 ms | 344 → 338 ms |
| 253,261 | 614 → **466 ms** | 619 → 464 ms | 638 → 650 ms |

- **−15 % at 130K, −24 % at 253K** on the routed turns; the remaining ~250 ms at 253K is outside attention (re-tokenizing the whole prompt, HTTP, the SWA layers, Marlin) — the 164 → ~60 ms GPU share predicted by the profile.
- **Quality on the routed server:** MMLU 29/30, HumanEval 27/30 (stock server 10-31: 47/57, 25/30); natural needle 131K 3/3 found (depth 0.9 answered inside the reasoning and hit the 2048-token cap — Flint's thinking shape, not the route), 250K 1/3 — the checkpoint's known ≈240K band (2/6 on both allreduce arms 10-31). Parity.
- **Numerics:** fp32-reference test 53/54 cells at the bf16 floor; the one flagged cell (e5m2 + sinks, prefix 58K, K=16) re-run under five seeds reads max-err 3.5–6.5e-3 vs the extend kernel's 2.8–3.5e-3 and rms 6–7e-4 vs 4–5e-4 — one bf16 ulp class, not systematic.
- Shipped as the `flint` preset default (`launch.sh`, 0 disables); the Gemma 4 triton presets get the same extend bench + gate before the env is widened to them.

## qwen38 (flashinfer): 1-token vs 16-token extend at 142K — launch-bound either way (2026-11-01)

`profile_prefill.sh EXTEND_NEW=1|16` + `trace_cpu_ops.py` on the warmed shape: CPU wall **178 ms / 185 ms**, GPU span 179 / 186 ms, kernel sum **84 / 70 ms**. The step costs the same with one new token as with sixteen and more than half of it is idle gap, so on the flashinfer GDN presets the per-turn tax is the un-graphed 64-layer eager extend, not attention work. The prefill-CUDA-graph A/B (`--cuda-graph-backend-prefill breakable|full --cuda-graph-max-bs-prefill 1 --cuda-graph-max-seq-len-prefill 16`) failed to boot on both backends: CUDA OOM (2.37 GiB) during capture at the preset's 0.85 memory fraction → re-queued with a smaller fraction / smaller KV pool for the A/B only.

## Gemma 4 gate → two kernel defects, fixed (2026-11-01, kernel v2)

Baselines without the route (same bench): **gemma4-31b** 16-token turn 144 / 471 / 782 ms at 35K / 140K / 233K (1 token 97 / 274 / 437); **gemma4** (26B-A4B) 137 / 408 / 678 ms (1 token 73 / 197 / 310). Both routed arms died on their first routed request with `Triton Error [CUDA]: out of memory` at the combine-kernel launch, with 1.9 GB (31B) and 3.4 GB (26B) free after graph capture.

`test_extend_splitkv_route.py --mem` (first-launch free-memory delta + per-kernel registers/spills) on the free GPU:

| variant (before) | regs | spills / thread | first-launch reserve |
|---|---:|---:|---:|
| combine, head_dim 256 | 168 | 3034 B | **1336 MiB** |
| combine, head_dim 128 (Flint) | 128 | 1546 B | fits inside the 256 reserve |
| prefix stage 1 | 63–96 | 0 | 0 |

Cause: `_verify_combine_stage2` was launched with `num_warps=1` and materialised the whole `[N_SPLITS, 16, Dv]` prefix-partial tile and two `[16, 16, D]` broadcast products in registers; the driver sizes its per-thread local-memory pool for the worst kernel at first launch. Fix: stream the splits (one `[16, Dv]` tile per split), fp32-exact `tl.dot` for the draft-draft block at the 16-row tile (the spec-verify L_EXT = 4 tile keeps the broadcast form), 4 warps. After: 126 regs, **0 spills, 0 MiB** on every shape, and the kernel is faster (Flint per layer 121 → 52 µs at 8K, 334 → 265 µs at 58K; gemma4-31b 366 → 146, 994 → 784 µs).

Second defect, found by the same test on a sink cell that scaled with the sink value: `VerifySplitKV.__call__` accepted `sinks` but did not forward them to the combine launch, so `HAS_SINKS` was always False and the route ignored Flint's learned sinks (output too large by the sink's softmax share: ~1 % at a +4.3 sink over an 8K prefix — the "ulp-class outlier" of the first test was this). Fixed; the sink cells now sit at the bf16 floor like the rest.

Receipt `benchmarks/profiling/extend-splitkv-route-v2-2026-11-01.json` (qwen38, flint ± sinks, gemma4-31b, gemma4 × bf16/e5m2 × 8K/58K/197K × 1/8/16 new tokens — all cells pass; 7.0–7.3× (qwen38), 12–16× (flint), 9.6–12× (gemma4-31b), 20–22× (gemma4) per layer at 197K). The served Flint numbers above were measured with the sink-less kernel and are being re-taken; the Gemma 4 gates and the DSpark spec arm (same kernel, L_EXT = 4) run behind it.

## Gemma 4 31B on the fixed route (kernel v2 + boot warm-up, 2026-11-01)

Warm-up compiled both kernels at boot in 2.1 s (`[070] split-KV short-turn route warmed (h_q=16 h_kv=8 d=256 kv=e5m2)`); no first-request JIT.

| gemma4-31b, cached prefix | +1 token before → with route | +16 tokens | +128 tokens |
|---|---:|---:|---:|
| 34,649 | 97 → **64 ms** | 144 → 140 ms | 183 → 178 ms |
| 140,261 | 274 → **71 ms** | 471 → 473 ms | 528 → 526 ms |
| 232,581 | 437 → **92 ms** | 782 → 791 ms | 848 → 860 ms |

- The 1-token turn loses the whole serial-prefix walk (**−74 % at 140K, −79 % at 233K**) and what remains (71–92 ms) is the tokenizer + launch floor.
- The bench's "16 new tokens" did **not** route on Gemma: the server counts the chat template's turn framing with the new text (`prompt_tokens` grows by 16 over the 1-token row, which itself carries ~6 template tokens), so the routed class `K ≤ 16` excludes it. Flint's shorter framing kept its 16-token row inside the cap. Short tool results are 20–60 tokens — the 16-row tile is the limit, not the lever: a 32/64-row tile test (`--suffixes 17,24,32,48,64`) and a served `MAX_TOKENS=64` arm on flint + gemma4-31b are queued.
