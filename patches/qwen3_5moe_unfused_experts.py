"""Monkey-patch `Qwen3_5MoeExperts` → unfused per-expert ModuleList for REAP.

Companion to `qwen3moe_unfused_experts.py`, but for the Qwen3.5 / Qwen3.6
family (`Qwen3_5MoeForConditionalGeneration`, incl. Qwen3.6-35B-A3B). The
difference that forces a separate patch:

  * Plain Qwen3Moe (Coder-30B) ships **unfused** per-expert weights on disk
    (`experts.0.gate_proj.weight` …), so the unfused ModuleList's child names
    match the checkpoint directly.
  * Qwen3.5/3.6 ship **fused 3-D** expert params:
        experts.gate_up_proj : [num_experts, 2*moe_intermediate, hidden]
        experts.down_proj    : [num_experts, hidden, moe_intermediate]
    so loading them into per-expert Linears needs a slice on the way in, and
    saving a pruned model needs a re-stack on the way out.

This module installs:
  1. a **load_state_dict pre-hook** that slices the fused 3-D checkpoint tensors
     into per-expert `experts.{e}.gate_up_proj.weight` / `.down_proj.weight`
     so `from_pretrained` populates the per-expert Linears; and
  2. a **state_dict post-hook** that re-stacks the (possibly pruned) per-expert
     Linears back into the fused 3-D names, so `save_pretrained` writes a
     STANDARD fused Qwen3_5Moe checkpoint (just fewer experts) that reloads
     WITHOUT this patch.

Each expert keeps `gate_up` fused as a single Linear (a clean 1:1 slice of the
3-D param) plus a `down_proj` Linear — the latter is what run_reap.py's
saliency hook reads. The per-expert forward reproduces the fused
`Qwen3_5MoeExperts` inner loop exactly (verified numerically by
`scripts/quantize/test_qwen3_5moe_unfuse.py`).

Apply BEFORE any `from_pretrained` of a Qwen3_5Moe checkpoint:
    import qwen3_5moe_unfused_experts  # patches transformers in place
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)

NOTE (run_reap.py): the Qwen3_5Moe router (`mlp.gate`) is a custom module that
returns `(router_logits, routing_weights, selected_experts)`, NOT a plain
`nn.Linear` emitting logits. run_reap.py's saliency router-hook must be taught
to read that tuple before REAP works end-to-end on this family — tracked in the
README MoE backlog.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _M


class Qwen3_5MoeMLPUnfused(nn.Module):
    """Single expert. `gate_up_proj` stays fused (matches the 3-D slice);
    `down_proj` is separate so run_reap.py can hook its output."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3_5MoeExpertsUnfused(nn.ModuleList):
    """`nn.ModuleList` of per-expert `Qwen3_5MoeMLPUnfused`, with fused-3-D
    checkpoint <-> per-expert load/save bridging hooks."""

    def __init__(self, config):
        super().__init__([Qwen3_5MoeMLPUnfused(config) for _ in range(config.num_experts)])
        # plain attribute (not a submodule) — see qwen3moe patch note about
        # ACT2FN nn.Module children inflating len(self).
        self.num_experts = config.num_experts
        self._register_load_state_dict_pre_hook(self._split_fused_pre_hook)
        self._register_state_dict_hook(self._fuse_state_dict_hook)

    # ---- load: fused 3-D checkpoint -> per-expert Linears -----------------
    def _split_fused_pre_hook(self, state_dict, prefix, *args):
        gu, dn = prefix + "gate_up_proj", prefix + "down_proj"
        if gu in state_dict:
            w = state_dict.pop(gu)  # [E, 2I, H]
            for e in range(w.shape[0]):
                state_dict[f"{prefix}{e}.gate_up_proj.weight"] = w[e]
        if dn in state_dict:
            w = state_dict.pop(dn)  # [E, H, I]
            for e in range(w.shape[0]):
                state_dict[f"{prefix}{e}.down_proj.weight"] = w[e]

    # ---- save: per-expert Linears -> fused 3-D checkpoint -----------------
    # Uses len(self) (the CURRENT, possibly-pruned expert count), so a model
    # pruned by run_reap.py saves as a standard fused Qwen3_5Moe with K experts.
    def _fuse_state_dict_hook(self, module, state_dict, prefix, local_metadata):
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
        # Mirror of the fused Qwen3_5MoeExperts.forward, dispatching per expert.
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


# ---- Monkey-patch the symbol so AutoModelForCausalLM picks it up ----
_M.Qwen3_5MoeExperts = Qwen3_5MoeExpertsUnfused


# Skip the fused-3-D manual init for our unfused collection (nn.Linear children
# self-initialize); mirror the qwen3moe patch.
if hasattr(_M, "Qwen3_5MoePreTrainedModel"):
    _orig_init = _M.Qwen3_5MoePreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        if isinstance(module, Qwen3_5MoeExpertsUnfused):
            return
        return _orig_init(self, module)

    _M.Qwen3_5MoePreTrainedModel._init_weights = _patched_init_weights
