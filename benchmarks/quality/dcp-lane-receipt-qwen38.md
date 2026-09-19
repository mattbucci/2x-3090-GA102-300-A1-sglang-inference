# qwen38 opencode-dcp vs opencode (256K, 32K output budget) — pre-score rollout receipt

Same-ID instances: **298** (lane 299 logs, control 299 logs).

| | lane | control |
|---|---|---|
| patched (non-empty diff) | 278 | 272 |
| empty diff | 0 | 2 |
| timeout (rc=124) | 20 | 24 |
| other non-zero rc | 0 | 0 |
| median wall (s) | 736.5 | 672.6 |
| p90 wall (s) | 1529.5 | 1640.6 |

Agreement (patched or not): both_patched 258, control_only 14, lane_only 20, neither 6

## DCP engagement

- Observed containers: 298; snapshots: 222; **engaged (pruned >0 tokens): 65 = 22%**
- Pruned tokens when engaged: median 43,961, total 3,054,293

## Lane split by engagement

| subset | n | patched | empty | timeout | median wall (s) |
|---|---|---|---|---|---|
| lane, engaged | 65 | 54 | 0 | 11 | 1279.6 |
| lane, not engaged | 233 | 224 | 0 | 9 | 664.0 |
| control, same engaged IDs | 65 | 58 | 0 | 7 | 817.9 |

Verdict on quality is the scored cell (`scores-docker-summary.json`), not this receipt.
