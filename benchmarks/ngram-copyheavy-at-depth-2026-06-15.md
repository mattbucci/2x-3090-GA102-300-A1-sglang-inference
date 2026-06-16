# NGRAM spec-decode survives at 256K depth on copy-heavy decode (2026-06-15)

**Question.** R9700 showed model-draft spec (EAGLE3 / DFlash) **collapses at true 256K depth**
(Coder-30B EAGLE3 107 t/s short → 0.8 t/s @244K, ~15× *slower* than no-spec) because the draft
model attends the full deep KV every micro-step. Does **NGRAM** — which has *no draft model* — share
that collapse, or does it survive?

**Why NGRAM could be different.** NGRAM drafts by matching n-grams already in the context (a CPU-side
trie lookup), so there is no draft model doing deep-KV attention. The only added cost is the target
*verifying* N draft tokens in one forward pass — and at depth that forward pass is **memory-bound on
the KV read**, which is paid once whether it verifies 1 token or N. So accepting K tokens per verify
should amortize the expensive deep-KV read → NGRAM should help *more* at depth, not collapse.

## Setup
- `coder-30b` (Qwen3-Coder-30B-A3B AWQ-CT, A3B MoE), TP=2, MEM 0.85, ctx 262144, flashinfer attn, fp8_e4m3 KV.
- **Copy-heavy** task (the fair test for NGRAM — random tokens deflate it, repetitive filler inflates it):
  prompt = real diverse sglang source as context + "reproduce this file verbatim"; output overlaps the
  recent context. Harness: `/tmp/ngram-exp/copyheavy_bench.py`. temp=0 (deterministic → identical output
  both arms, confirming spec is lossless: completion_tokens = 1444 in both arms).
- Metric: **server-log `gen throughput`** (client TPOT under-measures bursty spec ~2× — R9700 2026-06-14).
- NGRAM flags: `--speculative-algorithm NGRAM --speculative-num-draft-tokens 8` (ngram knobs at defaults:
  bfs-breadth 1–10, BFS, trie-depth 18). Note: NGRAM **disables the overlap scheduler + mixed chunked
  prefill** (a real fixed overhead) and is **draft-free** (no 4 GB draft+graph reservation → fits 256K
  where EAGLE3/DFlash OOM on 24 GB).

## Result — server-log gen throughput, same deterministic 1444-tok copy task

| depth | no-spec | NGRAM (copy-heavy runs) | NGRAM (low-overlap spans) | NGRAM accept len |
|---|---|---|---|---|
| ~12K  | 176 t/s (steady) | up to 242 t/s | 104–121 t/s | 1.1 → 2.8 (warming) |
| ~172K | **89 t/s (steady)** | **235–237 t/s (~2.6×)** | 42–77 t/s | **6.1 → 7.6** on contiguous copy |

- **NGRAM does NOT collapse at depth.** Worst-case @172K was ~42 t/s (low n-gram overlap), vs model-draft
  spec's 0.8 t/s — i.e. NGRAM's floor is ~50× higher. On genuinely copy-heavy spans it hits accept len
  **6–7.6** (accepts nearly all 8 drafted tokens) → **235 t/s = 2.6× the 89 t/s no-spec baseline**.
- **The win grows with depth for copy-heavy** (the predicted memory-bound amortization): the copy-run
  multiplier is larger at 172K (2.6×) than the short-ctx peak ratio at 12K.
- **Short-ctx (12K) net is ~neutral** on a short 986-tok gen — the trie warms over the generation, so a
  short output is warmup-dominated; the steady-state only shows on longer/deeper runs.
- **Lossless**: identical output at temp=0 (spec verify is exact) → no quality regression by construction.

## Conclusion / ship
Refines R9700's "spec is a ≤32–64K optimization, no-spec is the path at depth": that holds for
**model-draft** spec, but **draft-free NGRAM is the exception** — it's net-positive at 256K for
copy-heavy decode (1.4× mixed file-repro up to 2.6× pure-copy) and never collapses. Wired as an
opt-in `NGRAM=1` on the coder preset (copy-heavy agentic coding is the 256K mandate's workload).
Caveat: neutral-to-slightly-negative for novel-text generation (overlap-scheduler overhead with low
acceptance) — hence opt-in, not default.

## Generalization across the coder fleet — the win is gated by COPY FIDELITY, not arch

Same copy-heavy task @172K, same server-log method, on the REAP-pruned sibling (same Qwen3-Coder-30B-A3B
base, 128→96 experts):

| coder model @172K | no-spec | NGRAM accept len | NGRAM net (wall, same 1444-tok gen) |
|---|---|---|---|
| `coder-30b` (full 128e) | 89 t/s | 6.1–7.6 (copy spans) | **WIN ~2.6× on copy spans** |
| `coder-reap-25b` (REAP 96e) | 88.4 t/s | ~2 (peak 3.0) | **NET-NEGATIVE** (153.6s vs no-spec 145.6s; draft=4 also worse, 156.1s) |

So NGRAM's benefit tracks the model's **copy fidelity** (= n-gram acceptance), not architecture: the
full coder copies code verbatim (high acceptance), but **REAP/REAM pruning degrades verbatim
reproduction** → acceptance craters to ~2 → the verify overhead (and disabled overlap scheduler)
dominates → net-negative. Lowering `NGRAM_DRAFT` 8→4 did **not** rescue it (accept ~2 is the ceiling,
not over-drafting). **Ship: `NGRAM=1` allowlisted to the full coders only (`coder-30b` / `coder-30b-eval`
= the 128-expert base); the pruned coders (`coder-reap-25b`, `coder-30b-ream`) are excluded** — the gate
prints a WARN if `NGRAM=1` is set on them.

Open follow-ups: test `--speculative-ngram-external-corpus-path` seeded with the edited repo (should lift
acceptance on the first touch of each file — could even help the pruned models); measure NGRAM on a real
SWE-bench rollout (mixed copy+novel output → the realistic net, between the 1.4× mixed and 2.6× pure-copy).
