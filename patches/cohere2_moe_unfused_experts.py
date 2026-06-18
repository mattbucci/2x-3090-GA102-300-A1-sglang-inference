"""Post-load unfuse of `Cohere2MoeExperts` → per-expert ModuleList for GPTQ.

Companion to `qwen3moe_unfused_experts.py` and `qwen3_5moe_unfused_experts.py`,
this version targets the **Cohere2 MoE family** (`Cohere2MoeForCausalLM`, e.g.
`CohereLabs/North-Mini-Code-1.0`).

The defect this fixes
---------------------
`Cohere2MoeExperts` stores expert weights as fused 3-D `nn.Parameter`:

    gate_up_proj : [num_experts, 2*intermediate_size, hidden_size]
    down_proj    : [num_experts, hidden_size, intermediate_size]

These are RAW PARAMETERS, not `nn.Linear` modules — so
`llmcompressor.GPTQModifier(targets="Linear")` walks past them and quantizes
ZERO experts. Caught 2026-06-17 on the North-Mini-Code-1.0 run: only the
3 L0-dense + 49×4 attention Linears got INT4-quantized (≈ 199 of 12491
intended Linears, i.e. 1.6% of the model bulk). The 30B MoE bulk would have
shipped at BF16.

Why post-load (not the qwen3.5-style load-state-dict pre-hook)
-------------------------------------------------------------
Transformers 5.10's new `convert_and_load_state_dict_in_model` path in
`core_model_loading.py` does `getattr(module, "down_proj")` directly before
populating a param, bypassing the `_register_load_state_dict_pre_hook` the
qwen3.5 patch relied on. If we delete the fused attributes pre-load, the new
loader AttributeErrors. So instead we:

  1. let `from_pretrained` populate the standard fused 3-D tensors;
  2. immediately after, walk the model and convert every `Cohere2MoeExperts`
     instance in-place into an `nn.ModuleList` of per-expert
     `Cohere2MoeMLPUnfused` — each backed by real `nn.Linear` modules that
     `GPTQModifier(targets="Linear")` discovers and quantizes.

On `save_pretrained` the per-expert Linears get re-stacked back into the fused
3-D shape (via the state_dict post-hook), so the saved CT/AWQ checkpoint is a
standard `Cohere2Moe` shape that reloads WITHOUT this patch — both for
SGLang's loader and for round-trip identity with upstream-style ships.

Usage
-----
    from cohere2_moe_unfused_experts import unfuse_cohere2_moe_experts
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(...)
    unfuse_cohere2_moe_experts(m)   # walk + replace; mutates m in place
    # m's MoE experts are now per-expert nn.Linear instances, ready for GPTQ
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.models.cohere2_moe import modeling_cohere2_moe as _M


class Cohere2MoeMLPUnfused(nn.Module):
    """Single expert. `gate_up_proj` stays fused as one `nn.Linear` (a clean 1:1
    slice of the upstream 3-D param); `down_proj` is a separate `nn.Linear`.
    Both are visible to `targets="Linear"` for GPTQ quantize."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        inter = config.intermediate_size  # Cohere2Moe uses intermediate_size for MoE experts
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Cohere2MoeExpertsUnfused(nn.ModuleList):
    """`nn.ModuleList` of per-expert `Cohere2MoeMLPUnfused`, with a
    state_dict post-hook that re-stacks the per-expert Linears back into
    the fused 3-D names on save."""

    def __init__(self, config):
        super().__init__([Cohere2MoeMLPUnfused(config) for _ in range(config.num_experts)])
        self.num_experts = config.num_experts
        self._register_state_dict_hook(self._fuse_state_dict_hook)

    def _fuse_state_dict_hook(self, module, state_dict, prefix, local_metadata):
        """save: per-expert Linears -> fused 3-D checkpoint."""
        n = len(self)
        gate_ups, downs = [], []
        for e in range(n):
            gate_ups.append(state_dict.pop(f"{prefix}{e}.gate_up_proj.weight"))
            downs.append(state_dict.pop(f"{prefix}{e}.down_proj.weight"))
        if gate_ups:
            state_dict[f"{prefix}gate_up_proj"] = torch.stack(gate_ups, dim=0)
            state_dict[f"{prefix}down_proj"] = torch.stack(downs, dim=0)
        return state_dict

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror of fused `Cohere2MoeExperts.forward`, dispatching per expert."""
        final_hidden_states = torch.zeros_like(hidden_states)
        n_experts = len(self)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=n_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0]) if expert_idx.dim() else int(expert_idx)
            if expert_idx == n_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


def unfuse_cohere2_moe_experts(model: nn.Module) -> int:
    """Walk `model`, replace every `Cohere2MoeExperts` instance with a
    `Cohere2MoeExpertsUnfused` populated from the fused tensors.

    Returns the number of MoE layers converted (helpful for sanity-print).
    """
    count = 0
    # Each Cohere2MoeSparseMoeBlock has `.experts` as its child name.
    # Walk parent modules so we can replace .experts with the new module.
    for parent_name, parent in list(model.named_modules()):
        experts = getattr(parent, "experts", None)
        if experts is None or not isinstance(experts, _M.Cohere2MoeExperts):
            continue
        config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        unfused = Cohere2MoeExpertsUnfused(config)
        # Copy fused -> per-expert
        gate_up_fused = experts.gate_up_proj.detach()  # [E, 2I, H]
        down_fused = experts.down_proj.detach()        # [E, H, I]
        for e in range(unfused.num_experts):
            with torch.no_grad():
                unfused[e].gate_up_proj.weight.copy_(gate_up_fused[e])
                unfused[e].down_proj.weight.copy_(down_fused[e])
        # Preserve dtype/device
        unfused = unfused.to(dtype=gate_up_fused.dtype, device=gate_up_fused.device)
        # Replace the child in place
        setattr(parent, "experts", unfused)
        count += 1
    return count
