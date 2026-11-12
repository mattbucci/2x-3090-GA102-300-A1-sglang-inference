# flint × omp v5 — lane close (2026-11-11)

Cell `evals/swebench/runs/flint-omp-v5`, started 2026-11-09 08:55, closed 2026-11-11 17:35 (56 h 41 m), serving
image `3a6019266963` (SGLang v0.5.23, fp8_e5m2 KV, pool 1,097,420 tokens), omp 18.12.2, isolated harness
(network none, refs stripped, git re-init, neutral mounts).

| | v4 (2026-10-27, image ee9c00c20e95, v0.5.22, bf16 KV, strict grammar in schema order) | v5 (this cell, key-order fix) |
|---|---:|---:|
| resolved | **186** | **176** |
| unresolved / empty / error | 109 / 5 / 0 | 107 / 17 / 0 |
| walls (rc 124) | 3 | 10 |
| median rollout | 452 s | 526 s |

Paired: 162 both, 24 v4-only, 14 v5-only, exact McNemar **p = 0.14** — a tie
([paired-flint-omp-v5-vs-v4.json](paired-flint-omp-v5-vs-v4.json)). Leak audit PASS, EXPOSED 0/193 attempted
([leak-audit-final.log](leak-audit-final.log)) — the first clean close on the first run of the audit.

## The key-order fix moved the defect from `bash` to `edit`

The strict tool grammar enforces the schema's property order. v4 forced omp's declared order (`i` first);
the 2026-10-28 fix (`TRAILING_KEYS` in `plugins/flint_sglang/flint_sglang/tool_grammar.py`) moves `i` last for
every tool. Counting tool calls whose main argument is a filler (`x`, `y`, `true`, `false`, empty) in the
session stores ([tool-arg-degeneracy-v4-vs-v5.json](tool-arg-degeneracy-v4-vs-v5.json)):

| tool | v4 (`i` first) | v5 (`i` last) |
|---|---:|---:|
| `bash` | 1,457 / 18,385 (7.9 %), 106 instances | 46 / 15,224 (0.3 %), 12 instances |
| `edit` | 49 / 2,532 (1.9 %), 31 instances | **4,301 / 9,418 (45.7 %)**, 92 instances |

Flint writes `bash` command-first but `edit` intent-first: forcing `input` before `i` makes it write a filler
into `input` (then `i`), the hashline edit fails ("input must begin with [PATH#HASH]"), and the model retries —
edit calls ×3.7, two sessions hit omp's thinking-loop retry cap, and 11 of the 17 empties end with no edit
landed. The score cost is not concentrated there (the 92 affected instances: 46 vs 51 resolved; the other 208:
130 vs 135), so the cell reads as a tie with a mechanical defect, not a regression attributable to it alone;
v5 also differs from v4 by the v0.5.23 stack and fp8_e5m2 KV.

Next: per-tool order (keep `i` first for `edit`, last for `bash`), CPU grammar test, then a served omp sample
before any re-roll.
