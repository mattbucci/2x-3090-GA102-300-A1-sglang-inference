# flint-sglang (v0.5.22 port)

SGLang support for [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) — the vendor's
`flint_sglang` 0.1.4 plugin (pinned to SGLang 0.5.16 / transformers 5.12.1, shipped as a wheel under
`sglang/` in the model repo) ported to the SGLang v0.5.22 tree this stack serves from. Installed by
`scripts/setup.sh`; `scripts/launch.sh flint` exports the required env and flags.

Install (serving env only):

```bash
PYTHONNOUSERSITE=1 pip install --no-deps plugins/flint_sglang   # (-e while iterating)
```

What the plugin registers (unchanged from the vendor): the `flint` model type with `AutoConfig` (so no
`--trust-remote-code`), `SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang` (model class
`FlintForCausalLM`), `FLASHINFER_USE_CUDA_NORM=1`, and the `flint` tool-call parser / reasoning parser /
grammar backend. `check_flags` refuses the combinations the vendor refuses (grammar flint ⇒ reasoning
flint; tool parser flint ⇒ grammar flint; no `--enable-strict-thinking`, no
`--constrained-json-disable-any-whitespace`).

## Port delta (0.5.16 → v0.5.22)

| Vendor dependency | v0.5.22 | Change |
|---|---|---|
| `sglang.srt.layers.communicator.{LayerCommunicator,LayerScatterModes}` | removed; `layers.layer_boundary.factories.{declare_attn,declare_ffn,append_stages}` | `FlintDecoderLayer` rebuilt on the boundary protocol; `stage_facts(config, layer_id)` staticmethod reads `mlp_only_layers` (`Qwen2MoeModel.make_pp_layers` calls it per layer) |
| `Qwen2MoeMLP(reduce_results=True)` default for the dense MLP + shared expert | FFN modules return TP-partial sums, `ffn_boundary.finish` completes them | both built with `reduce_results=False` (upstream convention; the old default would double-reduce at TP=2) |
| `get_global_server_args()` | raises | `get_exec().kernel.{attention,prefill_attention,decode_attention,grammar}_backend`, `get_exec().moe.{moe_runner_backend,moe_a2a_backend}`, `get_exec().features.enable_fp32_lm_head`, `get_parallel().{enable_dp_attention,enable_dp_lm_head}` |
| `sglang.srt.distributed.get_pp_group()` | `get_parallel().pp_group` | as `Qwen3MoeForCausalLM` does |
| `FlintDetector.get_{auto_tool_call_,}structural_tag(tools, tool_choice, thinking_mode)` | base hooks carry `parallel_tool_calls` | keyword added (vendor signature → `TypeError`); `pass_parallel_tool_calls` hook kept as fallback |
| `Glm47MoeDetector` dispatch in `get_structure_constraint` | new fast path: `parses_required_natively()` ⇒ GLM-4.7 EBNF with GLM special tokens | `FlintDetector.parses_required_natively()` pinned to `False` |
| sinks detection: arch allowlist | generic `hf_text_config.learnable_sink` | `FlintConfig.__post_init__` sets `learnable_sink = (softmax_type == "learnable")` so auto backend selection never picks FlashInfer (no sinks path) |

Unchanged: `grammar_backend.py` (`XGrammarGrammarBackend` API identical), `reasoning_parser.py`, `plugin.py`.
`tool_grammar.py` (xgrammar 0.2.7 `get_model_structural_tag("glm_4_7", …)`) gains one 3090 knob:

* **`FLINT_STRICT_TOOLS=1`** (env, read when a request's tool grammar is built; `launch.sh flint` exports it)
  treats every tool as `strict: true`, so argument values follow each tool's JSON schema — required keys,
  property names, types. The vendor semantics (schema-free unless the client sends `strict: true`, as
  OpenAI's API does) stay the default without the env. Why: Flint ignores a schema's `required` keys on
  its own. Against opencode's tools (`bash` requires `description`, `read` takes `filePath`) the INT4
  checkpoint wrote neither in 23/24 calls with the grammar off and 23/24 with it on schema-free — so the
  grammar is not the cause — and 20/20 with the schema enforced
  (`benchmarks/quality/flint/tool-schema-probe-2026-10-22.json`, three arms × three tasks × 5 draws). The
  first docker-serving smoke rollout spent 86 turns re-issuing `bash` without `description` (opencode
  returns the `SchemaError` each time) before producing nothing. The vendor's SWE-bench path is
  mini-swe-agent (bash in text, no JSON-schema tools), so this contract was never exercised there. All
  ten opencode 1.14.25 tool schemas compile under strict (`todowrite`'s nested arrays included).

## Serving on this stack

* `--attention-backend triton` (sinks + SWA via `SWAKVPool`; FlashInfer has no sinks path, FA3/FA4 need
  Hopper/Blackwell), `--grammar-backend flint --reasoning-parser flint --tool-call-parser flint`,
  `--enable-fp32-lm-head`, `FLASHINFER_USE_CUDA_NORM=1` exported before launch. The MoE runner stays
  `auto`: on the AWQ checkpoint `AWQMoEScheme.create_moe_runner` asserts `is_auto()` and dispatches
  `fused_marlin_moe` itself — the vendor's `--moe-runner-backend triton` is the BF16/FP8-weight setting
  and would fail the assert here. KV stays `auto` (bf16): triton rejects FP8 E4M3 KV on sm_86 and Flint's
  full-attention KV is small (14 layers × 4 KV heads); `fp8_e5m2` is a later lever.
* TP=2 is **not vendor-validated** (the plugin only warns). The 3090 bring-up validates it against TP=1
  short-context outputs before any quality number is published.
* Checkpoint: the BF16 base is 79.6 GB (does not fit 48 GB). The INT4 route is
  `scripts/quantize/quantize_flint_rtn_awq.py` — a data-free per-group MSE-clipped round-to-nearest from
  the BF16 base into native AWQ GEMM packing (group 64, integer zero-points; router / embeddings /
  lm_head / norms / sinks stay BF16). RecursiveSI's own MLX int4 export uses a free per-group bias
  (`w = s·q + b`, `−b/s` fractional) that the AWQ format cannot carry exactly, and transformers has no
  Flint modeling file, so neither a lossless MLX→AWQ repack nor an llmcompressor/GPTQ path exists.
