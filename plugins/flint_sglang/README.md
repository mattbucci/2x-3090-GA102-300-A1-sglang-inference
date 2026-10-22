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

Unchanged: `grammar_backend.py` (`XGrammarGrammarBackend` API identical), `reasoning_parser.py`,
`tool_grammar.py` (xgrammar 0.2.7 `get_model_structural_tag("glm_4_7", …)`), `plugin.py`.

## Serving on this stack

* `--attention-backend triton` (sinks + SWA via `SWAKVPool`; FlashInfer has no sinks path, FA3/FA4 need
  Hopper/Blackwell), `--moe-runner-backend triton`, `--grammar-backend flint --reasoning-parser flint
  --tool-call-parser flint`, `--enable-fp32-lm-head`, `FLASHINFER_USE_CUDA_NORM=1` exported before launch.
* TP=2 is **not vendor-validated** (the plugin only warns). The 3090 bring-up validates it against TP=1
  short-context outputs before any quality number is published.
* Checkpoint: the BF16 base is 79.6 GB (does not fit 48 GB). The INT4 route is
  `scripts/quantize/convert_mlx_int4_to_awq.py` over RecursiveSI's MLX int4 export (group 64, symmetric
  affine `w = s·q + b` → AWQ zero-point form) — no llmcompressor path exists because transformers has no
  Flint modeling file.
