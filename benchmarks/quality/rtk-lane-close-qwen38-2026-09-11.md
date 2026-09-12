# qwen38 little-coder+RTK lane close (pre-score) — 2026-09-11

Lane: `evals/swebench/runs/qwen38-little-coder-rtk-v2` (little-coder 1.19.0 / pi-coding-agent 0.83.0,
`rtk` 0.46.0 loaded via `-e`, HOME `/opt/rtk-home`). Control: `qwen38-little-coder-v2` (little-coder 1.1.0 /
pi 0.68.1). Both lanes: `--max-running-requests 1`, 1800 s per instance, SWE-bench Lite 300.
Machine-generated tables: [`rtk-lane-receipt-qwen38-2026-09-11.md`](rtk-lane-receipt-qwen38-2026-09-11.md)
(+ `.json`), compaction ledger [`rtk-lane-compaction-qwen38-2026-09-11.json`](rtk-lane-compaction-qwen38-2026-09-11.json).
Quality verdict waits for the scored cell; this receipt is about *what the two lanes actually ran*.

## Rollout outcomes (299 same-ID instances)

| | RTK lane | control |
|---|---|---|
| patched (non-empty diff) | 232 | 240 |
| empty diff | 29 | 59 |
| **timeout (rc=124, 0-byte prediction)** | **38** | **0** |
| median / p90 wall | 375.9 s / 1966.8 s | 411.9 s / 619.7 s |

Agreement: both patched 198, control-only 42, lane-only 34, neither 25. Of the 38 lane timeouts the
control patched 22 and returned empty on 16. The 300th instance (`pytest-dev__pytest-7490`) never
ran on this lane — its rollout image build failed on an `archive.ubuntu.com` connection timeout
(`rollout_returncode -1`); the audit step classifies that as `infra_rollout_nonzero_rc` and rerolls it.

## RTK engagement

290/299 sessions (97%) executed ≥1 rtk rewrite; median 6 rewrites per session (p90 14); rtk-reported
savings median 490 tokens / session, 761K total; families `grep` 1292, `ls` 506, `git` 103, `cat` 96,
`find` 58, `wc` 22, `pytest` 13; one parse failure. The 9 unengaged sessions (7 patched / 2 empty /
0 timeout) are too few to split on. Engaged vs the control on the same 290 IDs: 225 vs 233 patched,
27 vs 57 empty, 38 vs 0 timeouts.

## The 38-vs-0 timeout asymmetry is a context-budget bug, not rtk

`evals/swebench/prompt_sawtooth.py` reads the server log's per-prefill prompt size (`#new-token +
#cached-token`) inside each instance's window (log mtime − elapsed). A "compaction" is a drop to
<50% of the running peak after the peak passed 20K tokens (one drop per non-timeout instance is
discounted for the fresh cleanup-prompt session).

| subset | n | patched | empty | timeout | median wall | median peak prompt |
|---|---|---|---|---|---|---|
| compacted ≥1× | 119 (40%) | 69 (58%) | 12 | **38** | 1248 s | 31,542 |
| never compacted | 180 | 163 (91%) | 17 | 0 | 299 s | 18,058 |

994 compaction events; every timeout session compacted (median 13.5 times), while the server was
decoding flat-out the whole 30 min (~380 `Decode batch` lines per 5 min) — a model-side loop, not a
hung shell command. Max prompt seen on the lane: 62,473 tokens; 37 sessions ever exceeded 32K.

**Root cause (verified in both packaged pi builds).** `buildFallbackModel()` in
`pi-coding-agent/dist/core/model-resolver.js` resolves an id that is not in `models.json` as
`{...providerModels[0], id, name}` — i.e. `llamacpp/qwen38` inherits the *first* llamacpp entry's
`contextWindow: 32768, maxTokens: 4096` (1.1.0 ships `qwen3.6-27b`, 1.19.0 `qwen3.8-27b`, both
32768/4096). The Dockerfile only rewrites `baseUrl` in `models.json`; no served-model entry is
added, so **every little-coder cell in the bake-off table ran pi with a 32,768-token context
budget** regardless of the preset's 256K KV. Compaction settings are identical in both versions
(`reserveTokens 16384`, `keepRecentTokens 20000`, `shouldCompact = contextTokens > contextWindow −
reserveTokens`, with `contextTokens = usage.totalTokens`), and the overflow path (`isContextOverflow`
case 2: `stop` + `input + cacheRead > contextWindow` → `_runAutoCompaction("overflow", true)` + retry)
fires whenever the model's *real* prompt exceeds 32K. pi never sends `max_tokens` (default 0), so
the 4096 cap does not bind.

**Why the two lanes fail differently.** pi 0.83 (RTK lane) compacts, retries, re-explores the repo,
crosses 16K again, compacts again — until the 1800 s cap. pi 0.68 (control) never times out; instead
171/300 control sessions end with `Request was aborted` printed by print-mode.js and a max session
length of 1,167 s. The abort message originates from the aborted-signal branch in
`openai-completions.js`; **the exact 0.68.1 trigger site was not traced** — the observed behaviour
(a hard session end well under the rollout cap, with no server-side error) is what is recorded here.
Net: the control lane gets a de-facto session cap that *keeps* its partial edits, the RTK lane gets a
loop that hits the harness timeout.

**Harness amplifier.** On timeout the rollout kills the container before the trailing `git diff
--cached` runs, so a timed-out instance is a 0-byte prediction even if the tree already held a fix.
Same rule for all lanes; it matters here because only the RTK lane times out.

## What the scored cell will and won't say

- The RTK cell ≠ "RTK effect". It is RTK + (pi 0.83 vs 0.68) × (32K fallback budget), and the second
  factor alone costs up to 38 instances. Read the cell as a lower bound on RTK-at-real-context.
- Engagement is clean (97%), so the earlier "env confound" (HOME override dropping the testbed
  env) is not in play this cycle — the comment in `docker_rollout.py` attributing early timeouts to
  it predates this finding.
- `evals/swebench/docker/opencode.json` carries the same class of limit for `qwen36-dense`,
  `qwen36-dense-ct`, `qwen35-dense` (`context: 32768`) and `coder-30b` (16384): the qwen36-dense
  62.3% opencode cell ran on a 32K budget while qwen38's opencode lanes run at 262144.

## Fix (cycle boundary, not mid-series)

1. Write a `<served-name>` entry with the preset's real KV cap into both packaged `models.json`
   files at run time (prime already writes its provider file per run), and align the four
   `opencode.json` entries.
2. Re-roll `qwen38` little-coder + little-coder-rtk (the A/B at real context) and the
   `qwen36-dense` opencode leader cell. Nuke `swebench-rollout/*` images first (Dockerfile edit).
3. Run `prompt_sawtooth.py` on the prime and dcode lanes as they close — their context defaults
   are unverified.
