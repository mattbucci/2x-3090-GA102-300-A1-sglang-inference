# qwen38 little-coder-rtk-v4 vs little-coder-v4 (pi 0.83 control) — pre-score rollout receipt

Same-ID instances: **250** (lane 279 logs, control 271 logs).

| | lane | control |
|---|---|---|
| patched (non-empty diff) | 225 | 229 |
| empty diff | 1 | 0 |
| timeout (rc=124) | 24 | 21 |
| other non-zero rc | 0 | 0 |
| median wall (s) | 781.4 | 772.0 |
| p90 wall (s) | 1920.5 | 1891.0 |

Agreement (patched or not): both_patched 214, control_only 15, lane_only 11, neither 10

## rtk engagement

- Observed containers: 250; snapshots with a ledger: 250; **engaged (≥1 executed rewrite): 249 = 100%**
- Executed rewrites per engaged session: median 17, p90 34
- Tokens rtk reports saved: median 2588 / session, total 3,167,950; parse failures 9
- Rewritten command families: `grep` 2530, `ls` 1237, `cat` 563, `find` 178, `git` 165, `wc` 17, `du` 4, `pytest` 4, `ps` 3

## Lane split by engagement

| subset | n | patched | empty | timeout | median wall (s) |
|---|---|---|---|---|---|
| lane, engaged | 249 | 224 | 1 | 24 | 786.4 |
| lane, not engaged | 1 | 1 | 0 | 0 | 485.1 |
| control, same engaged IDs | 249 | 228 | 0 | 21 | 767.4 |

Verdict on quality is the scored cell (`scores-docker-summary.json`), not this receipt.
