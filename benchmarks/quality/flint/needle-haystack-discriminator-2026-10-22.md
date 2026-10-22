# Flint needle: the stock filler is a NoPE repetition attractor, not a serving defect (2026-10-22)

**Server:** `launch.sh flint` (Flint-AWQ-INT4-g64, TP=2, `--context-length 262144`, triton attention, bf16 KV,
AWQ-Marlin fused MoE, `--sampling-defaults model`), v0.5.22 tree, bare metal, :23334.
**Probe:** `scripts/eval/needle_band_probe.py` — needle `The secret password is: BANANA42.` at a depth fraction of
the haystack, question `What is the secret password? Answer with just the password.`, `max_tokens 2048`,
thinking on (per-turn `<think>`). `--flush` = `POST /flush_cache` before every cell (0 cached prefix).

## Result table

| haystack | sampling | cache | 1K | 32K | 131K @0.1 | 131K @0.5 | 131K @0.9 | ~244–250K @0.1 | @0.5 | @0.9 | receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| filler | model (T 1.0 / p .95 / k 20) | warm | 3/3 | 3/3 | found (167 tok) | **miss** (length) | **miss** (length) | miss (3 tok, `stop`) | **miss** (length) | **miss** (length) | [`flint-tp2-bringup.json`](../flint-tp2-bringup.json) |
| filler | greedy (T 0) | warm | | | | **miss** (length) | **miss** (length) | | | | [`…-131k-greedy.json`](../flint-tp2-bringup-131k-greedy.json) |
| filler | greedy | **flushed** | | | | **miss** (length) | | | | | [`…-131k-greedy-flushed.json`](../flint-tp2-bringup-131k-greedy-flushed.json) |
| **natural** (Open-Platypus) | greedy | flushed | | | | **found** (58 tok) | **found** (57 tok) | | **found** (50 tok, 243,522 actual) | **found** (73 tok, 243,521 actual) | [`…-natural-greedy.json`](../flint-tp2-bringup-natural-greedy.json), [`…-natural-250k-greedy.json`](../flint-tp2-bringup-natural-250k-greedy.json) |
| **natural** | model | flushed | | | | **found** (142 tok) | | | **miss** (1 sample: content `B`, `stop` after 549 tok — reasoning ends *"The system/developer context maybe no secret. I'll final: 'The secret password is: [REDACTED]'"*; re-probed ×3: **2/3** on this checkpoint and 2/3 on the attn-BF16 build — [`…-natural-250k-default-x3.json`](../flint-tp2-bringup-natural-250k-default-x3.json)) | | [`…-natural-default.json`](../flint-tp2-bringup-natural-default.json), [`…-natural-250k-default.json`](../flint-tp2-bringup-natural-250k-default.json) |

Filler = `"The quick brown fox jumps over the lazy dog. " × N` (4.5 chars/token under the Flint tokenizer → ~29K
copies at 131K). Natural = `scripts/eval/build_natural_haystack.py` (Open-Platypus instruction+output docs tiled,
docs mentioning the needle vocabulary dropped; 28.5 MB), `--chars-per-token 3.43` for the 131K cells and `3.22`
for the 250K cells (the first attempt at 3.43 tokenized to 264,485 > 262,144 → HTTP 400; at 3.22 the cell lands at
243,522 actual tokens — 3.31 would hit 250K).

## What the misses look like

Every filler miss at ≥131K@0.5 reaches `finish_reason: length` at 2048 tokens: the model either loops a short
phrase (`We need answer. We need answer. …`, `The final should be just the password.`) or quotes the filler back.
The 250K@0.1 filler cell stops after 3 tokens with the wrong answer. The natural-haystack hits answer in 50–142
tokens with `finish_reason: stop`. The one natural miss (243.5K@0.5 under the model's own sampling, T 1.0 / top-p
0.95 / top-k 20) is a different shape from the filler misses: no loop, the reasoning is unsure whether a secret is
present and the answer degenerates to a single token — a marginal-retrieval sample at the edge of the window, which
is why the sampled 243.5K cell was repeated ×3 on both checkpoints rather than read from one draw: **2/3 found on each build**
(the misses share a shape — `finish stop`, an unsure trace, a one-token or invented answer). Under the model's own T 1.0 /
top-k 20 sampling the 243.5K@0.5 cell is a marginal retrieval; greedy retrieves it on the ship checkpoint. The last ~10 %
of the 262K window is where sampled retrieval starts to soften, not a cliff.

## Discriminators run, in order

1. **Sampling** — greedy (`temperature 0`) reproduces the 131K@0.5/@0.9 filler misses → not a sampling artifact.
2. **Radix / SWA cache** — `POST /flush_cache` before the cell (prefix hit 0) reproduces the miss → not the hybrid
   SWA radix cache (`RustUnifiedTreeCore`, `hybrid_swa`) or SWA-pool eviction.
3. **Kernel review** (no change made): the Lean decode kernel only engages at batch ≥ 16 (bs = 1 here); the triton
   sink folding is `row_sum += exp2(sink·log2e − row_max·scale·log2e)`, the same convention as the vendor's
   `fa4-learned-sink-fix.patch`; the per-head sinks shard with the q heads at TP=2; the rope / NoPE layer map
   (`no_rope_layer_indices` = 0,1,5,…,45,48 — 14 layers) matches the vendor `model.py`; the dense-MLP and
   shared-expert `reduce_results=False` matches upstream `qwen3_moe.py`.
4. **Haystack** — the same server, same cells, same greedy decoding, with a natural-text haystack: found at
   131K@0.5, 131K@0.9, 243.5K@0.5, 243.5K@0.9. → The input is the variable.

## Mechanism

Flint's 14 full-attention layers are **NoPE** (no positional encoding — long-range retrieval rides on content
similarity alone, positions are only implicit via the causal mask). With the stock filler every key in the haystack
is one of ~10 token identities repeated ~29K times, so on those layers the needle competes with tens of thousands of
near-identical keys and the softmax (sinks included) spreads over the repetition; the model's own output then falls
into the same attractor (phrase loops). Natural text has distinct keys and the needle wins. This is the Qwen3.6
filler attractor (the v0.5.22 campaign's `qwen36` 131,072@0.5 filler-loop miss, reproduced on both stacks — [patches/README.md](../../../patches/README.md), v0.5.20→v0.5.22 entry) in a stronger form; RoPE models tolerate the filler because
position still separates the copies.

## Rule

Flint quality receipts use the natural haystack (`needle_band_probe.py --haystack benchmarks/quality/haystack-platypus.txt
--chars-per-token 3.43` at ≤131K, `3.31` for a 250K cell, `--flush` for a cold-cache verdict). The filler cells
above are a **known-bad input**, not a regression signal — do not compare them against other presets' filler needle
scores. The fleet harness (`run_v0512_fleet_eval.sh` → `eval_and_chart.needle_eval`) keeps the filler for every
other preset so their historical bands stay comparable.

## Follow-ups covered in the same session

- **TP=1 vs TP=2:** final answers byte-identical on 5/6 greedy prompts, identical `read_file` call on the sixth (preamble wording differs); reasoning traces diverge the way greedy does across GEMM shapes and re-converge — [`tp-check-2026-10-22.md`](tp-check-2026-10-22.md). TP=2 is validated for the ship.

- **Attention-in-BF16 A/B — INT4 ship stands.** `Flint-AWQ-INT4-g64-attnbf16` (`quantize_flint_rtn_awq.py --keep-bf16 self_attn`: every
  `q/k/v/o_proj` left BF16 and listed in `modules_to_not_convert`, experts/shared-expert/dense MLP INT4 as before; 22.1 GiB vs 20.9,
  `check_awq_scales.py --base` 0 / 36,243 — [log](check_awq_scales-Flint-AWQ-INT4-g64-attnbf16-2026-10-22.log)), same server flags,
  same cells, same probe:

  | cell (attnbf16) | result | INT4 ship (same cell) | receipt |
  |---|---|---|---|
  | filler 131K @0.5, greedy, flushed | **miss** (`length`, loop *"The final should just be the password"*) | miss (`length`) | [`…-attnbf16-131k-greedy-flushed.json`](../flint-attnbf16-131k-greedy-flushed.json) |
  | filler, model sampling: 32K ×3 / 131K @0.1 0.5 0.9 / 250K @0.1 0.5 0.9 | 3/3 · **0/3** · **0/3** (loops / quotes the filler; 250K@0.5 stops after 21 tokens with no answer) | 3/3 · 1/3 (@0.1) · 0/3 | [`…-attnbf16-filler-default.json`](../flint-attnbf16-filler-default.json) |
  | natural 131K @0.5 / @0.9, greedy, flushed | **found** (200 tok) / **found** (75 tok) | found / found | [`…-attnbf16-natural-greedy.json`](../flint-attnbf16-natural-greedy.json) |
  | natural 243.5K @0.5 / @0.9, greedy, flushed | **miss** @0.5 (`stop`, 467 tok, content `S3cr3t!` — reasoning: *"There is no explicit password in conversation… I'll output a plausible secret password"*) / **found** @0.9 (200 tok) | found / found | [`…-attnbf16-natural-250k-greedy.json`](../flint-attnbf16-natural-250k-greedy.json) |
  | natural 243.5K @0.5, model sampling, flushed, ×3 | **2/3** (miss: `stop`, 278 tok, content *"The secret password is: openai"*; hits 169 / 486 tok) | **2/3** (miss: `stop`, 485 tok, content `C`; hits 197 / 243 tok) | [`…-attnbf16-natural-250k-default-x3.json`](../flint-attnbf16-natural-250k-default-x3.json), [`…-tp2-bringup-natural-250k-default-x3.json`](../flint-tp2-bringup-natural-250k-default-x3.json) |

  Keeping attention in BF16 does not move the filler cliff (131K/250K filler cells are identical misses with the same loop
  shapes) and does not improve the natural 243.5K cell — it loses the greedy @0.5 retrieval the INT4 ship has. So the
  attractor is not INT4 attention-projection noise; it is the input (NoPE + repeated keys), as the mechanism section says.
  The 1.2 GiB is not worth carrying: the INT4 g64 checkpoint ships (`mattbucci/Flint-AWQ`), the attnbf16 directory was
  deleted after the ×3 draw, its audit log and the receipts above are kept.
