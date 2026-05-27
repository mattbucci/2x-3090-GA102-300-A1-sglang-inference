# Fleet capability validation — sglang v0.5.12, 2026-05-26

Each preset launched TP=2 on the v0.5.12 stack (`ENV_NAME=sglang-v0512
SGLANG_DIR=/data/sglang-rebase-v0512`) and run through
`validate_capabilities.py` (now including a **tool_call** probe). 256K context
where the model supports it, else the preset's native max. "n/a" = capability
not applicable to that model (auto-skipped). Receipts in
`benchmarks/quality/capability_check.json` + `qwen36-awq-marlin-rebuild-v0512.json`.

| Preset | ctx | basic | tool_call | thinking | image | video | verdict |
|--------|:---:|:-----:|:---------:|:--------:|:-----:|:-----:|:--------|
| **qwen36** (AWQ-Marlin rebuild) | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** — shipped to HF |
| **qwen35-moe** | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **qwen36-dense** | 131K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **gemma4** (26B MoE) | 16K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **qwen3-vl-32b** | 131K | ✅ | ✅ | n/a | ✅ | ✅ | **4/4** (non-thinking VL) |
| **coder-30b** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| **coder-30b-ream** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| **coder-reap-25b** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| qwen36-ream | 256K | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | thinking+tool OK; **vision degraded/unstable** |
| devstral | 131K | ✅ | ❌ | n/a | ✅ | n/a | basic+vision OK; **tool-call not emitted** (prompt echo) |
| gemma4-31b | 16K | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | loader FIXED (patches 039+040); basic+tool+thinking pass, vision degraded |
| qwen3-ream | — | — | — | — | — | — | **model not on disk** |

## What got fixed this pass
- **qwen36** rebuilt from BF16 → native AWQ-Marlin (was CT, broke on v0.5.12). 5/5 incl. tool-calling + vision (after BF16 vision-tower merge). Shipped to `mattbucci/Qwen3.6-35B-A3B-AWQ`.
- **qwen35-moe** stale-symlink fixed → 5/5.
- **coder-30b** dead MODEL path repointed to the on-disk CT dir + QUANT corrected (was `awq_marlin` over a CT config).
- **qwen3-vl-32b** missing `hf-mattbucci/Qwen3-VL-32B-AWQ` symlink created → 4/4.
- **validate_capabilities.py** gained a `tool_call` probe + coder-family auto-skip classification.
- **patch 039**: `gemma4_causal` `num_experts` getattr fallback (dense Gemma4 crash on load).

## Remaining gaps (pre-existing, need deeper work)
- **gemma4-31b** loader RESOLVED (patches 039 + 040) — boots TP=2, basic+tool+thinking pass. Was a config-remap gap (dense `Gemma4ForCausalLM` reads the top-level `Gemma4Config`, which never got the global/swa head-dim remap → full-attention layers built at head_dim 256 not 512). Vision remains degraded (separate AutoRound vision-tower issue).
- **devstral tool-calling**: the model echoes the prompt (degenerate, finish=length) instead of emitting a `tool_call`. NOT a template gap — the custom `scripts/devstral_chat_template.jinja` *does* render `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`, and basic + vision pass. Likely a model-behavior / sampling issue on the non-coding weather prompt, or an assistant-turn-open edge in the template under `tools=`. Needs a deeper trace of the rendered prompt vs Mistral's expected `[AVAILABLE_TOOLS]` placement.
- **qwen36-ream vision**: unstable — sometimes describes the image, sometimes "I can't see it" (degraded VLM alignment from the REAM merge / calibration; keyword-grep masked it). Coding-critical thinking + tool-calling are solid.
- **qwen3-ream**: `Qwen3-30B-Instruct-2507-REAM-AWQ` is not on disk; preset references a missing checkpoint. (Text-only, non-thinking; documented not viable for codegen.)
