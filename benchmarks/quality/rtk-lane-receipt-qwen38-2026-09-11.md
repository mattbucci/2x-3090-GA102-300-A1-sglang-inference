# qwen38 little-coder+RTK vs little-coder control — pre-score rollout receipt

Same-ID instances: **299** (lane 299 logs, control 300 logs).

| | lane | control |
|---|---|---|
| patched (non-empty diff) | 232 | 240 |
| empty diff | 29 | 59 |
| timeout (rc=124) | 38 | 0 |
| other non-zero rc | 0 | 0 |
| median wall (s) | 375.9 | 411.9 |
| p90 wall (s) | 1966.8 | 619.7 |

Agreement (patched or not): both_patched 198, control_only 42, lane_only 34, neither 25

## rtk engagement

- Observed containers: 299; snapshots with a ledger: 299; **engaged (≥1 executed rewrite): 290 = 97%**
- Executed rewrites per engaged session: median 6.0, p90 14
- Tokens rtk reports saved: median 490.0 / session, total 761,269; parse failures 1
- Rewritten command families: `grep` 1292, `ls` 506, `git` 103, `cat` 96, `find` 58, `wc` 22, `pytest` 13

## Lane split by engagement

| subset | n | patched | empty | timeout | median wall (s) |
|---|---|---|---|---|---|
| lane, engaged | 290 | 225 | 27 | 38 | 379.9 |
| lane, not engaged | 9 | 7 | 2 | 0 | 354.3 |
| control, same engaged IDs | 290 | 233 | 57 | 0 | 414.0 |

Verdict on quality is the scored cell (`scores-docker-summary.json`), not this receipt.
