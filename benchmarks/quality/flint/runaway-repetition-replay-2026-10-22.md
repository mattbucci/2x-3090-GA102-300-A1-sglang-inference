# Flint runaway repetition — replay matrix, root cause, calibration (2026-10-22)

**Question.** The `flint:opencode` parity smoke on the rebuilt serving image produced 0/2 patches: 4 of 51 assistant turns ran to the 32,768-token output cap in a degenerate loop (two inside `bash` tool-call arguments, two in reasoning). Is that the serving stack (kernels, grammar, SWA pool, sampling) or the checkpoint?

**Answer.** The checkpoint. The data-free INT4 of the *always-active* path (attention projections + the ungated shared expert) is the damage; every serving arm is null, and a checkpoint with those two families in BF16 replays 0/72. The fix is a calibrated re-quantization (GPTQ error feedback on activation Grams, AWQ layout) — builds and results in §6.

## 1. Method

* `scripts/eval/build_opencode_replay.py` rebuilt the four runaway turns byte-faithfully from the opencode session stores (`benchmarks/quality/flint/replay/replay-<session>-k<turn>.json.gz` = `{messages, tools, input_tokens}`; the server's prompt-token counts match the originals).
* `scripts/eval/replay_turns.py <label> <out> <n> 8192 <extra-json> <replay…>` draws the turn N times at `max_tokens` 8192 under the server's sampling defaults (the vendor's T 1.0 / top-p 0.95 / top-k 20) or an override. A draw counts as a runaway when `finish_reason == length` or a 12–80-char n-gram repeats 5× back to back ("exact loop").
* Prefixes: `12907-k4` (15,072 tokens, clean), `12907-k25` (68,737 — "poisoned": an earlier runaway sits in the context), `12907-k28` (109,315), `14182-k20` (104,651, the other session).
* Server: `scripts/launch.sh flint` (TP=2 / 262144 / triton attention / AWQ-Marlin MoE / bf16 KV / kv-splits 64 / strict tools), one arm changed per row. Raw rows: `/var/tmp/flint-loop-logs/replay-*.jsonl` (table via `summ_replays.py`, 2 g32 rows excluded — see §4).

## 2. Replay matrix

| config | sampling override | prefix (ctx) | n | hit 8192 cap | exact loop | median out tok (non-cap) |
|---|---|---|---:|---:|---:|---:|
| **ship** (kv-splits 64, strict tools) | model default | 12907-k4 (15,072) | 24 | 3 (12 %) | 3 | 490 |
| ship | model default | 14182-k20 (104,651) | 24 | 0 | 0 | 60 |
| ship | model default | **12907-k25 (68,737)** | 24 | **11 (46 %)** | 11 | 506 |
| ship | model default | 12907-k28 (109,315) | 24 | 6 (25 %) | 3 | 772 |
| ship, `/flush_cache` before every draw | model default | 12907-k4 | 48 | 1 (2 %) | 1 | 661 |
| ship | greedy (T 0, top-k 1) | 12907-k4 / 14182-k20 / 12907-k28 | 2 each | 0 / 0 / 0 | 0 | 648 / 59 / 165 |
| ship | greedy | **12907-k25** | 2 | **2 (100 %)** | 2 | – |
| ship | T 0.6 | 12907-k25 | 24 | **24 (100 %)** | 22 | – |
| ship | T 0.6 | 12907-k4 | 48 | 1 (2 %) | 1 | 491 |
| ship | presence_penalty 0.5 | 12907-k25 | 24 | 11 (46 %) | 12 | 627 |
| ship | presence_penalty 0.5 | 12907-k4 | 48 | 0 | 0 | 655 |
| ship | repetition_penalty 1.05 | 12907-k25 | 24 | 14 (58 %) | 14 | 602 |
| ship | repetition_penalty 1.05 | 12907-k4 | 48 | 1 (2 %) | 1 | 611 |
| kv-splits 8 (CUDA default) | model default | 12907-k25 | 24 | 15 (62 %) | 13 | 820 |
| kv-splits 8 | model default | 12907-k4 | 48 | 2 (4 %) | 2 | 551 |
| strict-tools grammar off | model default | 12907-k25 | 24 | 14 (58 %) | 14 | 489 |
| strict-tools grammar off | model default | 12907-k4 | 48 | 5 (10 %) | 5 | 644 |
| `--disable-hybrid-swa-memory` (CTX 131072) | model default | 12907-k25 | 24 | 13 (54 %) | 12 | 642 |
| `--disable-hybrid-swa-memory` | model default | 12907-k4 | 48 | 0 (43 errors: the 131K window rejected the long draws) | 0 | 160 |
| **checkpoint: attention + shared expert BF16** (`--keep-bf16 'self_attn\|shared_expert'`, +1.6 GiB) | model default | **12907-k25** | 24 | **0** | 0 | 440 |
| attention + shared expert BF16 | model default | 12907-k4 | 48 | **0** | 0 | 147 |
| checkpoint: attention BF16 only | model default | 12907-k25 | 24 | 9 (38 %) | 8 | 749 |
| attention BF16 only | model default | 12907-k4 | 48 | 1 (2 %) | 1 | 856 |

Readings:

1. **Every serving arm is null.** kv-splits, the tool grammar, the hybrid SWA pool and the radix cache all leave the k25 rate at 46–62 % (N=24 each; the differences are within the binomial noise of that N). Sampling penalties do nothing on either prefix (presence 0.5 → 11/24, repetition 1.05 → 14/24 on k25).
2. **Lower temperature makes it worse** (T 0.6 24/24, greedy 2/2 on k25). The loop is the argmax path of the damaged model on the poisoned prefix; T 1.0 sampling escapes it about half the time. This is not a sampling-tail artifact and no sampling setting fixes it.
3. **The checkpoint arm is the only lever that moves it:** attention + shared expert in BF16 → 0/72. Attention alone is not enough (9/24).

## 3. Where the damage is — reference-forward KL

`scripts/eval/flint_cpu_reference.py` is a serving-independent fp32 forward of the Flint graph (SWA + NoPE + sinks + QK-norm + sigmoid top-6 routing + shared expert) that runs the BF16 base or any AWQ checkpoint dequantized, over the first 2,048 tokens of the k4 prefix; `served_prompt_logprobs.py` pulls the same per-position distributions from the live server (`return_logprob`); `compare_prompt_logprobs.py` scores them (top-1 agreement, mean |Δ log p| of the next token, truncated top-20 KL). All numbers vs the BF16 base reference unless noted.

| forward | top-1 agree | mean \|Δlp\| | mean KL (nats) |
|---|---:|---:|---:|
| ship checkpoint (g64 RTN), dequantized | 0.644 | 1.273 | 0.727 |
| ship checkpoint, **served** (TP=2 AWQ-Marlin) | 0.640 | 1.260 | 0.717 |
| served ship vs its own dequantized reference (**stack-only error**) | 0.897 | 0.327 | **0.107** |
| attention + shared expert BF16 | 0.781 | 0.643 | **0.278** |
| attention BF16 only | 0.690 | 1.051 | 0.517 |
| attention BF16 only, served | 0.691 | 1.071 | 0.527 |
| g32 RTN, dequantized | 0.686 | 1.012 | 0.493 |
| g32 RTN, **served** | 0.000 | 49.4 | 10.27 |

Attribution — BF16 base with exactly one family overlaid from the ship checkpoint (`--overlay … --overlay-re`):

| INT4 family (rest BF16) | parameters | top-1 agree | mean KL |
|---|---:|---:|---:|
| all 256 routed experts | 37.8 B | 0.782 | 0.261 |
| attention (q,k,v,o) | 0.8 B | 0.766 | 0.369 |
| — qkv only | | 0.815 | 0.253 |
| — o only | | 0.796 | 0.278 |
| **shared expert** | **0.3 B** | 0.710 | **0.432** |
| dense MLPs (layers 0–1) | 0.05 B | 0.871 | 0.159 |

The ship checkpoint's next-token perplexity on this prefix is 47 vs the base's 32, with *lower* entropy (top-20 mass 0.981 vs 0.965) — the INT4 model is more confident and more wrong, which is the shape of a repetition attractor. The runaway rate tracks the KL monotonically (0.73 → 46 %, 0.52 → 38 %, 0.28 → 0 %). Per-tensor weight error is a uniform 8.8 % relative RMS across families, so the loss is in how the rounding error projects onto the activations of the always-active path, not in any one tensor.

## 4. g32 serves as garbage (open serving bug)

The g32 checkpoint is sane (dequantized KL 0.49 < g64's 0.73, `check_awq_scales` clean) but the TP=2 server emits `market's market's …` with constant top-3 ids from the first position (top-1 agreement 0.000, top-20 mass 0.9999). Routed-expert `down_proj` at g32 / TP=2 has per-rank K = 256 → 8 groups per row; suspect the Marlin MoE path. Not needed for the ship (g32 is not a lever here); a TP=1 boot of the same checkpoint splits kernel-vs-shard when someone needs g32.

## 5. Why GPTQ, not just AWQ scaling

`scripts/quantize/quantize_flint_awq_calib.py` runs the same reference forward layer by layer on a calibration set and quantizes each layer from its activation Grams: an AWQ per-channel scale search (folded into `input_layernorm` / `post_attention_layernorm` / the BF16 router / `up_proj` rows, so the served graph is unchanged), a Gram-weighted per-group clip search, and with `--gptq` batched GPTQ error feedback on the same Grams (act-order with static groups, so the output is plain AWQ layout with no `g_idx`). Calibration set (`--calib-only`): 96 × 2,048 = 138,680 tokens — AM-Thinking code with reasoning 32, the-stack-smol 24, the *other* opencode session (14182, never the 12907 test prefixes) chunked 24, SWE-bench Lite problem statements 8, Open-Platypus 8.

CPU smoke, first 3 layers, 12 × 512-token mini set, hidden-state relative error after layer 3 on three calibration samples + the held-out k4 prefix slice:

| quantizer | sample 1 | sample 2 | sample 3 | **held-out** |
|---|---:|---:|---:|---:|
| ship (data-free MSE-RTN) | 0.269 | 0.284 | 0.248 | 0.237 |
| AWQ scale + Gram clip | 0.257 | 0.261 | 0.219 | 0.228 |
| **+ GPTQ** | **0.074** | **0.079** | **0.068** | **0.173** |

The scale search's own loss ratios are 0.86–0.997 per layer (qkv 0.97–0.997, dense gate/up 0.95–0.97, dense down 0.86–0.88, MoE gate/up 0.996) — per-channel scaling recovers 3–14 % of a layer's loss on this model, which is why the attention-BF16 arm and "AWQ it properly" were never going to be enough. Error feedback is the lever; the full 139K-token set is expected to close most of the in-sample/held-out gap that the 5.6K-token smoke shows.

## 6. First GPTQ build: experts only barely move (0.278 → 0.270) — and why

The first chain built `Flint-AWQ-INT4-g64-gptq-densebf16` (routed experts GPTQ at the GPTQ-default 1 % damping, always-active path BF16): `check_awq_scales` clean, every layer 256/256 experts hit with ≥ `--min-expert-tokens`, folds verified (router effective error 2.4e-3 after undoing the `post_attention_layernorm` scale) — and the end-to-end KL moved from 0.278 (data-free RTN, same BF16 set) to only **0.2701** (top-1 0.789, p95 1.14, max 9.4). The all-INT4 build of that chain died at layer 49: the `o_proj` Gram is not positive-definite at 1 % damping in fp32 (leading minor of order 49 — the massive-activation channels after act-order); the calibrator now factorizes the failing batch members alone in fp64 with an escalating-damping ladder (`_hinv_upper`, counts in `calibration-log.json["factor_fallbacks"]`).

Why so little: GPTQ only generalises where the Gram is well estimated. Held-out Linear-output error at layer 10 (Grams from the full 138,680-token calibration set, error measured on the 2,048-token k4 prefix; `scripts/quantize/probe_flint_expert_gptq.py`):

| Linear (layer 10) | rows in the Gram | rows / input dim | Gram clip | GPTQ 1 % damp | GPTQ 10 % | GPTQ 30 % |
|---|---:|---:|---:|---:|---:|---:|
| attn `q_proj` | 138,680 | 68 | 0.0470 | **0.0379** | 0.0377 | 0.0379 |
| attn `k_proj` | 138,680 | 68 | 0.0344 | **0.0274** | 0.0272 | 0.0273 |
| attn `v_proj` | 138,680 | 68 | 0.0711 | **0.0584** | 0.0582 | 0.0583 |
| attn `o_proj` | 138,680 | 34 | 0.0846 | **0.0542** | 0.0541 | 0.0557 |
| shared `gate_proj` | 138,680 | 68 | 0.0405 | **0.0325** | 0.0323 | 0.0325 |
| expert 192 `gate_proj` | 2,062 | 1.0 | 0.0704 | 0.0654 (in-sample 0.036) | 0.0562 | 0.0547 |
| expert 111 `gate_proj` | 5,201 | 2.5 | 0.0815 | 0.0744 (in-sample 0.046) | 0.0677 | 0.0660 |
| expert 129 `gate_proj` | 977 | 0.5 | 0.0810 | **0.0847** (in-sample 0.031) | 0.0712 | 0.0675 |
| expert 192 `down_proj` | 2,062 | 4.0 | 0.0826 | **0.0895** | 0.0805 | 0.0770 |
| expert 111 `down_proj` | 5,201 | 10 | 0.0912 | **0.1027** | 0.0944 | 0.0902 |
| expert 129 `down_proj` | 977 | 1.9 | 0.0912 | **0.1135** | 0.0975 | 0.0931 |

The always-active path (every token in every Gram, 34–68 rows per input dimension) gains 20–36 % over the clip and does not care about damping. A routed expert sees 0.5–12.6K rows for 2,048 input dims (median 2.7K at layer 10) and GPTQ at 1 % damping fits the noise in that Gram — in-sample error halves while the held-out error barely moves, and for `down_proj` (heavy-tailed `silu(g)·u` intermediates, 2–10 rows per dim) it comes out **worse than the clip** on every expert tried. That is the mechanism behind "standard GPTQ under-calibrates rare experts": not too few experts hit, but too few rows per expert for a 2,048-dim Hessian, and the default damping is tuned for dense layers that see every token.

Fix, measured on the whole layer-10 routed block (all 256 experts, held-out error of the routed sum vs BF16):

| expert policy | routed-block error |
|---|---:|
| data-free clip (the RTN ship) | 0.1206 |
| Gram clip, no feedback | 0.1164 |
| GPTQ 1 % damping on all three (the first build) | 0.1155 |
| gate/up GPTQ on the expert Gram + 1× trace-matched layer-wide Gram, 10 % damp; down Gram clip | 0.1016 |
| **… + down GPTQ at 100 % damping** | **0.0980** |
| … + down GPTQ at 30 % | 0.0990 |
| gate/up on the layer-wide Gram only (no expert data) | 0.1064 |

Shrinking each expert's Gram toward the layer-wide input Gram (the 138K-token one, which the expert's tokens are drawn from) and damping the thin `down_proj` Hessians hard gives the routed path the same ~20 % error reduction the dense path gets. These are now the calibrator defaults (`--expert-gram-shrink 1 --expert-percdamp 0.1 --expert-down-percdamp 1`, always-active path `--percdamp 0.01`); the 1 %-damping expert build is kept as `…-gptq-densebf16-d001` for the record.

## 7. Calibrated builds (second chain)

Builds (`calib-chain2.sh`: `quantize_flint_awq_calib.py --src <bf16> --calib-cache calib-96x2048.json --gptq` with the defaults above, one GPU each, then `check_awq_scales.py --base`, the CPU reference KL, and the k25 N=24 / k4 N=48 replays per candidate):

| build | kept BF16 | size | `check_awq_scales` | top-1 agree | mean KL | k25 runaway | k4 runaway |
|---|---|---:|---|---:|---:|---:|---:|
| `Flint-AWQ-INT4-g64-gptq-densebf16-d001` (first chain, 1 % expert damping) | attention + shared expert + dense MLPs | 23 GiB | clean | 0.789 | 0.270 | — | — |
| `Flint-AWQ-INT4-g64-gptq` | router / embed / lm_head / norms / sinks | | | | | | |
| `Flint-AWQ-INT4-g64-gptq-sharedbf16` | + shared expert | | | | | | |
| `Flint-AWQ-INT4-g64-gptq-densebf16` | + attention + shared expert + dense MLPs | | | | | | |

Decision rule, written before the results: ship the all-INT4 GPTQ build if its KL approaches the 0.28 of the BF16-path variant and it replays clean on both prefixes; otherwise the experts-only build (+1.6 GiB, always-active path 2.4× the bytes — decode measured before shipping). Results are filled in below as the chain reports.

_PENDING_
