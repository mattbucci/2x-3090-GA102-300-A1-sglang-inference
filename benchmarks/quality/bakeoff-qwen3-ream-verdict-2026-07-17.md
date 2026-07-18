# qwen3-ream bake-off verdict — agentic-incapable, mechanism-receipted (2026-07-17)

**Verdict: the June exclusion stands on model grounds.** Qwen3-30B-Instruct-2507
(REAM) cannot sustain agentic coding sessions. No score cell is published — the
20/300 partial is not leaderboard-comparable (full-300 rule) and finishing the
cycle would spend 1-2 GPU-days raising the precision of a foregone conclusion.

## What was ruled out first (all fixed/verified before the verdict)
1. **v0.5.12 tool-call parse breakage** — the June probe 0.0; on v0.5.15 the
   probe is 1.0/1.0 to 148K (`tooluse256k-qwen3-ream-v0515.json`).
2. **Template blanked OpenAI structured content** — real latent bug (list
   content rendered as ''), render-test-proven, fixed
   (`patch_chat_templates_list_content.py`, HF repo updated). Not the live
   mechanism: failure identical after the fix, and early calls in failing
   sessions adapt to tool results (the model was seeing them).
3. **SGLang generic sampling** — preset lacked `--sampling-defaults model`
   (checkpoint ships temp 0.7 / top_k 20 / top_p 0.8); added, verified in
   server_args. Failure distribution changed shape but not rate.
4. **Transport** — opencode session logs show tool calls flowing and completing
   with outputs both directions.

## The model's agentic profile (20 instances, all infra fixed)
- **18/20 empty diffs (90%).**
- **Prose-quit** (~8 instances): 3-4 steps — one tool call, then a final prose
  answer with no edits; opencode ends the session when no tool call is emitted.
- **Runaway loop** (2 instances): 1,246 and 1,400 steps (~1,426 tool calls);
  one filled the full 262,144-token window and died on the server 400
  ("requested 264,247 tokens"). Pre-sampling-fix variant: 116x the identical
  glob after one empty result.
- **Working minority** (~2 instances): 16-94 step sessions producing real
  diffs — the capability exists but does not sustain.

Evidence: `evals/swebench/runs/qwen3-ream-opencode-v2/` (predictions + per-
instance jsonl logs, preserved), server log `/tmp/run-model-cycle-logs/qwen3-ream/`.

## Durable takeaways
- Tool-call validity (single-turn probe 1.0) ≠ agentic competence. The probe
  needs a multi-turn tool-response rung (queued).
- The int4 degenerate-repeat trap is real but sampling-mitigation only reshapes
  it here (prose-quit replaces some loops); on agentic-trained ships
  (qwen36 family) the same serving stack sustains 60%+ resolve rates.
- REAM merging is not implicated: qwen36-ream (agentic-trained parent) scores
  59/45*/50; this model's failure is the base model's training, not the merge.
  (*claw partial.)
