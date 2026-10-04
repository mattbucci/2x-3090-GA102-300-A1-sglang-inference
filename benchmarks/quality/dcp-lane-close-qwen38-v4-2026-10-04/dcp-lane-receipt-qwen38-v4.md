# qwen38 opencode-dcp-v4 vs opencode-v4 (DSpark spec, 256K, 32K output) — pre-score rollout receipt

Same-ID instances: **290** (lane 298 logs, control 292 logs).

| | lane | control |
|---|---|---|
| patched (non-empty diff) | 252 | 255 |
| empty diff | 3 | 2 |
| timeout (rc=124) | 35 | 33 |
| other non-zero rc | 0 | 0 |
| median wall (s) | 939.8 | 802.2 |
| p90 wall (s) | 1972.1 | 1972.5 |

Agreement (patched or not): both_patched 235, control_only 20, lane_only 17, neither 18

## DCP engagement

- Observed containers: 217; snapshots: 217; **engaged (pruned >0 tokens): 95 = 44%**
- Pruned tokens when engaged: median 36,273, total 4,033,961

## Lane split by engagement

| subset | n | patched | empty | timeout | median wall (s) |
|---|---|---|---|---|---|
| lane, engaged | 95 | 67 | 0 | 28 | 1514.8 |
| lane, not engaged | 195 | 185 | 3 | 7 | 746.7 |
| control, same engaged IDs | 95 | 77 | 0 | 18 | 1269.2 |

Verdict on quality is the scored cell (`scores-docker-summary.json`), not this receipt.
