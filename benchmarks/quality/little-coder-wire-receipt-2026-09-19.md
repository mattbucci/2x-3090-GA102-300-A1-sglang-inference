# little-coder lane — on-the-wire request receipt (qwen38, 2026-09-19)

First lane to run with the per-run little-coder model-profile pin (`docker_rollout.LC_MODEL_PROFILE`,
commit `1e9c8dc`; the cycle's Phase-0 `scaffold-audit.json` predates it). Verified two ways at lane
start (instances 1–4 of `qwen38-little-coder-v2`):

**Per-instance stderr** (`logs/astropy__astropy-12907.log`), all three budget lines present:

```
context budget: little-coder models.json qwen38 contextWindow=262144
context budget: little-coder model profile llamacpp/qwen38 {"max_tokens":32768,"thinking_budget":1000000,"skill_token_budget":300,"knowledge_token_budget":200,"system_prompt_budget":0,"max_retries":1,"context_limit":262144}
context budget: /root/.pi/agent/settings.json compaction={"reserveTokens":32768}
```

**Loopback sniff of the live lane** (raw AF_PACKET on `lo`, dst port 23334, 100 s window during
instance 4): 12 `POST /v1/chat/completions` bodies, every one of them

| key | value | count |
|---|---|---|
| `model` | `"qwen38"` | 12 |
| `stream` | `true` | 12 |
| `max_completion_tokens` | `32000` (pi 0.68 wire clamp; ≥ `OUTPUT_CAP_FLOOR`) | 12 |
| `temperature` | — | **0** |
| `reasoning_effort` | — | **0** |
| `top_p` / `top_k` / `chat_template_kwargs` | — | 0 |

Sampling on this lane is therefore the preset's `--sampling-defaults model`, thinking effort is the
template default (Qwen3.8 `xhigh`), and the packaged profile's 2048-token thinking abort is disarmed
(budget 1e6). First three instances: real patches at 417 / 698 / 482 s (rc 0). No
`CONTEXT-BUDGET TRIPWIRE`.
