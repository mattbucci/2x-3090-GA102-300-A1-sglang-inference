"""Flint model configuration.

The SGLang plugin registers ``FlintConfig`` with ``AutoConfig``, so SGLang loads Flint checkpoints without
``--trust-remote-code``. Transformers does not support Flint yet: outside SGLang, load ``config.json`` with
``FlintConfig.from_pretrained``. Requires a transformers release with dataclass configs (vendor: 5.12.1; this port: 5.19.0).
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

SLIDING_ATTENTION = "sliding_attention"


@strict
class FlintConfig(PreTrainedConfig):
    r"""Configuration for Flint, a decoder-only mixture-of-experts transformer.

    Attention layers alternate between sliding-window layers, which use RoPE, and full-attention
    layers listed in ``no_rope_layer_indices``, which use no positional encoding. Every attention
    head has a learned sink logit (``softmax_type="learnable"``). Layers in ``mlp_only_layers`` use
    a dense SwiGLU MLP. All other layers use a sparse MoE block: sigmoid router scores, top-k
    selection, renormalized weights, plus an always-on shared expert.

    ``sliding_window`` counts the current token: a sliding layer attends to the current token and
    the ``sliding_window - 1`` tokens before it.
    """

    model_type = "flint"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 150144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_sliding_window: bool = True
    sliding_window: int | None = 513
    layer_types: list[str] | None = None
    no_rope_layer_indices: list[int] | None = None
    softmax_type: str = "learnable"
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    num_experts: int = 256
    norm_topk_prob: bool = True
    router_score_function: str = "sigmoid"
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        self.no_rope_layer_indices = [] if self.no_rope_layer_indices is None else self.no_rope_layer_indices
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers is {self.num_hidden_layers}"
            )
        super().__post_init__(**kwargs)
        # SGLang reads these to size its hybrid sliding-window KV cache (1 = sliding layer).
        self.hybrid_layer_pattern = [int(layer_type == SLIDING_ATTENTION) for layer_type in self.layer_types]
        self.is_hybrid_swa = self.sliding_window is not None and any(self.hybrid_layer_pattern)
        # SGLang v0.5.22 detects learned sinks on non-allowlisted architectures via this flag
        # (ModelConfig._detect_attention_sinks); without it the auto backend picks FlashInfer, which
        # has no sinks path. Flint's own marker is softmax_type="learnable".
        self.learnable_sink = self.softmax_type == "learnable"


__all__ = ["FlintConfig"]
