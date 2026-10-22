"""SGLang implementation of Flint — ported from flint_sglang 0.1.4 (SGLang 0.5.16) to SGLang v0.5.22.

Portions adapted from SGLang (Copyright 2023-2024 SGLang Team, Apache License 2.0); modified.
Vendor source: https://huggingface.co/RecursiveSI/flint/tree/main/sglang (flint_sglang-0.1.4-py3-none-any.whl).

Flint is a decoder-only mixture-of-experts transformer built from SGLang's stock MoE decoder
blocks plus:

1. Learned attention sinks: a per-head logit (``softmax_offset``) added to the softmax
   denominator, with no value contribution.
2. Hybrid attention: per-layer sliding-window or full attention from ``config.layer_types``.
3. No positional encoding on the layers listed in ``config.no_rope_layer_indices``.
4. Dense SwiGLU MLPs on ``config.mlp_only_layers``; sparse MoE blocks elsewhere.
5. Sigmoid routing: top-k on sigmoid scores (an optional expert bias affects selection
   only), routing weights renormalized to sum to one.
6. An ungated shared expert added to every MoE block.

Serve with ``--moe-runner-backend triton`` (Ampere) or ``flashinfer_trtllm_routed`` (Blackwell).
``flashinfer_trtllm`` and ``triton_kernel`` bypass the routing function and are not supported.
FlashInfer must use its CUDA RMSNorm kernels (``FLASHINFER_USE_CUDA_NORM=1``); the CuTe-DSL kernels
produce materially different logits for this model.

v0.5.22 port notes (what changed vs the 0.5.16 vendor file, and why):

* ``sglang.srt.layers.communicator`` (``LayerCommunicator`` / ``LayerScatterModes``) is gone; decoder
  layers now declare their stages via ``layer_boundary.factories`` (``declare_attn`` / ``declare_ffn`` /
  ``append_stages``) and ``Qwen2MoeModel.make_pp_layers`` calls ``decoder_layer_type.stage_facts(config,
  idx)`` for every layer — so ``FlintDecoderLayer`` carries a ``stage_facts`` staticmethod that reads
  ``config.mlp_only_layers``.
* Under the boundary protocol the FFN *module* returns a TP-partial sum and ``ffn_boundary.finish``
  completes it (``FusedMoE`` never all-reduces; upstream builds every dense MLP / shared expert with
  ``reduce_results=False``). The vendor file left ``Qwen2MoeMLP``'s default ``reduce_results=True`` in
  place (correct under the old ``LayerCommunicator``, double-reduced here) — both the dense ``FlintMLP``
  and the shared expert now pass ``reduce_results=False``.
* ``get_global_server_args()`` raises on v0.5.22; serving/exec/parallel state comes from
  ``sglang.srt.runtime_context`` (``get_exec().kernel.*``, ``get_exec().moe.*``, ``get_parallel()``).
* ``get_pp_group()`` → ``get_parallel().pp_group`` (what ``Qwen3MoeForCausalLM`` itself does now).
"""

import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from flint_sglang.grammar_backend import GRAMMAR_BACKEND_NAME
from sglang.srt.layers.layer_boundary.factories import (
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import LazyValue, add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from torch import nn

logger = logging.getLogger(__name__)

SLIDING_ATTENTION = "sliding_attention"
SUPPORTED_MOE_RUNNERS = frozenset({"auto", "triton", "flashinfer_trtllm_routed"})
FA4_SINK_FIX_MARKERS = ("sink_rescale", "tmem_alloc_warp_ids")
EXPERT_BIAS_SUFFIX = ".mlp.gate.expert_bias"
SINK_SUFFIX = ".self_attn.softmax_offset"


class FlintMLP(Qwen2MoeMLP):
    """Dense or shared-expert SwiGLU MLP that accepts the decoder's optional forward batch.

    Always built with ``reduce_results=False``: the enclosing layer boundary completes the TP sum.
    """

    def __init__(self, *args, reduce_results: bool = False, **kwargs):
        super().__init__(*args, reduce_results=reduce_results, **kwargs)

    def forward(self, hidden_states: torch.Tensor, forward_batch: object = None) -> torch.Tensor:
        del forward_batch
        return super().forward(hidden_states)


class FlintAttention(Qwen3MoeAttention):
    """Grouped-query attention with QK RMSNorm, learned sinks, optional sliding window and RoPE."""

    def __init__(
        self, *, layer_type: str, skip_rope: bool, sliding_window: int, layer_id: int, prefix: str = "", **kwargs
    ):
        super().__init__(layer_id=layer_id, prefix=prefix, **kwargs)
        self.skip_rope = skip_rope
        # The fused qk-norm/RoPE and fused KV-buffer paths assume RoPE on every layer and no sinks.
        self.use_fused_qk_norm_rope = False
        self.compatible_with_fused_kv_buffer = False

        sinks_dtype = torch.float32 if get_exec().kernel.attention_backend == "trtllm_mha" else torch.bfloat16
        self.softmax_offset = nn.Parameter(torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False)

        # SGLang's sliding_window_size excludes the current token; the config value includes it.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(sliding_window - 1) if layer_type == SLIDING_ATTENTION else -1,
            prefix=add_prefix("attn", prefix),
        )

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        if not self.skip_rope:
            q, k = self.rotary_emb(positions, q, k)
        self._used_fused_qk_norm_rope_last_call = False
        return q, k, v

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        q, k, v, fb = inner_state
        # compatible_with_fused_kv_buffer is False, so the KV cache is always written here.
        attn_output = self.attn(q, k, v, fb, sinks=self.softmax_offset)
        output, _ = self.o_proj(attn_output)
        return output


class FlintSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Sparse MoE block with sigmoid top-k routing and an ungated shared expert."""

    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        super().__init__(layer_id, config, quant_config=quant_config, prefix=prefix)

        # Checkpoints without a router bias leave this at zero.
        self.gate.expert_bias = nn.Parameter(torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False)

        def sigmoid_routing(hidden_states, gating_output, topk, renormalize):
            scores = torch.sigmoid(gating_output.float())
            scores_for_choice = scores + self.gate.expert_bias.unsqueeze(0)
            topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
            topk_weights = scores.gather(1, topk_ids)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=False,  # the routing function renormalizes
            use_grouped_topk=False,
            layer_id=layer_id,
            custom_routing_function=sigmoid_routing,
        )

        shared_size = getattr(config, "shared_expert_intermediate_size", 0)
        self.shared_expert = (
            FlintMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_expert", prefix),
                reduce_results=False,
            )
            if shared_size > 0
            else None
        )

    def forward(self, hidden_states, forward_batch=None):
        # Both terms are TP-partial sums; the layer's ffn_boundary completes them together.
        shared_output = self.shared_expert(hidden_states) if self.shared_expert is not None else None
        output = super().forward(hidden_states, forward_batch)
        if shared_output is not None:
            output = output + shared_output
        return output


class FlintDecoderLayer(Qwen3MoeDecoderLayer):
    """Pre-norm decoder layer: Flint attention followed by a dense MLP or a sparse MoE block."""

    def __init__(self, config, layer_id, start_layer=0, quant_config=None, prefix="", alt_stream=None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        rope_theta, _ = get_rope_config(config)
        self.rope_theta = rope_theta

        self.self_attn = FlintAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=self.rope_theta,
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            layer_type=config.layer_types[layer_id],
            skip_rope=layer_id in set(config.no_rope_layer_indices),
            sliding_window=config.sliding_window,
        )

        attn, ffn = self.stage_facts(config, layer_id)
        self.is_layer_sparse = ffn.sparse
        if self.is_layer_sparse:
            self.mlp = FlintSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = FlintMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config, layer_id: int):
        """The stages a Flint layer declares, from the config alone (``make_pp_layers`` calls this per layer)."""
        dense_layers = frozenset(config.mlp_only_layers)
        return (
            declare_attn(),
            declare_ffn(
                sparse=layer_id not in dense_layers,
                next_layer_sparse=layer_id + 1 < config.num_hidden_layers and layer_id + 1 not in dense_layers,
            ),
        )


class FlintModel(Qwen3MoeModel):
    """Embedding, Flint decoder stack and final norm."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix, decoder_layer_type=FlintDecoderLayer)


class FlintForCausalLM(Qwen3MoeForCausalLM):
    """Flint causal language model for SGLang."""

    def __init__(self, config, quant_config=None, prefix=""):
        # Build directly instead of calling the parent constructor, which would allocate a stock model first.
        nn.Module.__init__(self)
        parallel = get_parallel()
        _check_runtime(parallel)
        self.pp_group = parallel.pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = FlintModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=parallel.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        # IPC loading bypasses load_weights(), so initialize the EPLB descriptor here (as the parent does).
        self.routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
            }
        )
        self.attn_cp_size = parallel.attn_cp_size
        self.attn_cp_rank = parallel.attn_cp_rank
        self.moe_dp_size = parallel.moe_dp_size
        self.stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        self.expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.num_experts,
        )

    def get_attention_sliding_window_size(self) -> int:
        return self.config.sliding_window - 1

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False) -> None:
        """Load checkpoint tensors and fail if any tensor or parameter is left unmatched."""
        params = dict(self.named_parameters())
        loaded_targets: set[str] = set()
        unmatched: list[str] = []
        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        # Modules owned by other pipeline stages are PPMissingLayer placeholders without parameters.
        off_stage_prefixes = tuple(f"{n}." for n, m in self.named_modules() if isinstance(m, PPMissingLayer))

        def canonical_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.endswith(SINK_SUFFIX):
                    tensor = _shard_heads(tensor, attn_tp_size=attn_tp_size, attn_tp_rank=attn_tp_rank)
                target = _target_param_name(name, self.stacked_params_mapping, self.expert_params_mapping)
                if target in params:
                    loaded_targets.add(target)
                elif not self._skipped_by_parent_loader(
                    name, target=target, is_mtp=is_mtp, off_stage_prefixes=off_stage_prefixes
                ):
                    unmatched.append(name)
                yield name, tensor

        Qwen3MoeForCausalLM.load_weights(self, canonical_weights(), is_mtp=is_mtp)
        _check_load_coverage(
            params=params,
            loaded_targets=loaded_targets,
            unmatched=unmatched,
            check_missing=not is_mtp and self.quant_config is None,
        )

    def _skipped_by_parent_loader(
        self, name: str, *, target: str, is_mtp: bool, off_stage_prefixes: tuple[str, ...]
    ) -> bool:
        """Checkpoint tensors the stock loader drops on purpose: other stages/ranks, MTP, RoPE buffers, extra biases."""
        layer_id = get_layer_id(name)
        return (
            (layer_id is not None and not self.model.start_layer <= layer_id < self.model.end_layer)
            or name.startswith(off_stage_prefixes)
            or _is_off_rank_expert(name, self.expert_params_mapping)
            or ("mtp" in name) != is_mtp
            or "rotary_emb.inv_freq" in name
            or target.endswith(".bias")
        )


def _check_runtime(parallel) -> None:
    """Fail fast on serving configurations that silently change Flint's outputs; warn on unvalidated ones."""
    exec_ = get_exec()
    kernel, moe = exec_.kernel, exec_.moe
    if kernel.grammar_backend != GRAMMAR_BACKEND_NAME:
        raise RuntimeError(
            f"Flint requires --grammar-backend flint (got {kernel.grammar_backend}): Flint decides on each turn "
            "whether to reason, and only that backend lets it reason before a constraint or a tool call."
        )
    _require_cuda_norm()
    if moe.moe_runner_backend not in SUPPORTED_MOE_RUNNERS:
        raise RuntimeError(
            f"--moe-runner-backend {moe.moe_runner_backend} bypasses or is not validated with Flint's sigmoid "
            f"routing; use one of {sorted(SUPPORTED_MOE_RUNNERS)}."
        )
    if moe.moe_a2a_backend != "none":
        raise RuntimeError(f"--moe-a2a-backend {moe.moe_a2a_backend} is not supported for Flint.")
    if "fa4" in (kernel.attention_backend, kernel.prefill_attention_backend, kernel.decode_attention_backend):
        _require_fa4_sink_fix()
    if not exec_.features.enable_fp32_lm_head:
        logger.warning("Flint was validated with --enable-fp32-lm-head; outputs without it differ slightly.")
    sizes = {
        "tp": parallel.tp_size,
        "moe_ep": parallel.moe_ep_size,
        "moe_tp": parallel.moe_tp_size,
        "pp": parallel.pp_size,
        "attn_cp": parallel.attn_cp_size,
    }
    if any(size > 1 for size in sizes.values()) or parallel.enable_dp_attention:
        logger.warning("Flint is validated upstream with one GPU per replica; %s is not vendor-validated.", sizes)


def _require_fa4_sink_fix() -> None:
    """The learned-sink fix for SGLang's vendored FA4 kernel is not in any SGLang release (install.sh applies it)."""
    kernel = Path(sys.modules["sglang"].__file__).parent / "jit_kernel" / "flash_attn" / "cute" / "flash_fwd_sm100.py"
    source = kernel.read_text() if kernel.is_file() else ""
    if not all(marker in source for marker in FA4_SINK_FIX_MARKERS):
        raise RuntimeError(f"--attention-backend fa4 requires fa4-learned-sink-fix.patch (run install.sh): {kernel}")


def _require_cuda_norm() -> None:
    """Fail fast unless FlashInfer RMSNorm runs on its CUDA kernels (read once, at import)."""
    if os.environ.get("FLASHINFER_USE_CUDA_NORM") != "1":
        raise RuntimeError("Flint requires FLASHINFER_USE_CUDA_NORM=1 in the environment that starts SGLang.")
    norm = sys.modules.get("flashinfer.norm")
    if norm is not None and getattr(norm, "_USE_CUDA_NORM", True) is not True:
        raise RuntimeError(
            "flashinfer.norm was imported before FLASHINFER_USE_CUDA_NORM=1 was set; "
            "export it before starting SGLang."
        )


def _shard_heads(tensor: torch.Tensor, *, attn_tp_size: int, attn_tp_rank: int) -> torch.Tensor:
    if tensor.shape[0] % attn_tp_size != 0:
        raise ValueError(f"{tensor.shape[0]} attention heads are not divisible by attention TP size {attn_tp_size}")
    return tensor.chunk(attn_tp_size, dim=0)[attn_tp_rank]


def _target_param_name(
    name: str,
    stacked_params_mapping: list[tuple[str, str, str | int]],
    expert_params_mapping: list[tuple[str, str, int, str]],
) -> str:
    """Mirror the parent loader's name mapping to find the parameter a checkpoint tensor fills."""
    if "mlp.experts" not in name:
        for param_name, weight_name, _ in stacked_params_mapping:
            if weight_name in name:
                return name.replace(weight_name, param_name)
    for param_name, weight_name, _, _ in expert_params_mapping:
        if weight_name in name:
            return name.replace(weight_name, param_name)
    return name


def _is_off_rank_expert(name: str, expert_params_mapping: list[tuple[str, str, int, str]]) -> bool:
    return "mlp.experts" in name and any(weight_name in name for _, weight_name, _, _ in expert_params_mapping)


def _check_load_coverage(
    *, params: dict[str, nn.Parameter], loaded_targets: set[str], unmatched: list[str], check_missing: bool
) -> None:
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} checkpoint tensors match no Flint parameter, e.g. {unmatched[:5]}")
    if not check_missing:
        return
    missing = sorted(name for name in params if name not in loaded_targets and not name.endswith(EXPERT_BIAS_SUFFIX))
    if missing:
        raise RuntimeError(f"{len(missing)} Flint parameters were not loaded from the checkpoint, e.g. {missing[:5]}")
    logger.info("Flint weights loaded: %d parameters matched the checkpoint.", len(loaded_targets))


EntryClass = FlintForCausalLM
