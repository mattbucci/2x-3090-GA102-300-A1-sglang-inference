#!/usr/bin/env python3
"""Miniature validation for patches/qwen3_5moe_unfused_experts.py — no 62 GB
model load, no transformers MoE-integration wrapper. Checks the unfused patch:

  1. load-split hook: fused 3-D checkpoint -> per-expert Linears loads cleanly;
  2. forward equivalence: unfused output == a hand-rolled reference of the
     documented Qwen3_5MoeExperts inner-loop math (same weights);
  3. save-fuse hook: state_dict() re-stacks to the fused 3-D names/values;
  4. prune round-trip: a shortened ModuleList saves as a fused K-expert ckpt.

Run in the `quant` env (transformers + torch). Exit 0 = all pass.
"""
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

torch.manual_seed(0)

cfg = SimpleNamespace(num_experts=6, hidden_size=16, moe_intermediate_size=8, hidden_act="silu")
E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size

# --- synthetic fused 3-D checkpoint (the on-disk Qwen3.5/3.6 format) ---
fused_sd = {
    "gate_up_proj": torch.randn(E, 2 * I, H) * 0.05,   # [E, 2I, H]
    "down_proj":    torch.randn(E, H, I) * 0.05,        # [E, H, I]
}

T, top_k = 5, 2
hidden = torch.randn(T, H)
top_k_index = torch.randint(0, E, (T, top_k))
top_k_weights = torch.rand(T, top_k)


def reference_forward(gate_up, down, hidden, idx, w):
    """The documented Qwen3_5MoeExperts math, computed token-by-token."""
    out = torch.zeros(T, H)
    for t in range(T):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            g, u = F.linear(hidden[t], gate_up[e]).chunk(2, dim=-1)
            h = F.linear(F.silu(g) * u, down[e])
            out[t] += w[t, k] * h
    return out


out_ref = reference_forward(fused_sd["gate_up_proj"], fused_sd["down_proj"],
                            hidden, top_k_index, top_k_weights)

# --- apply the patch, build unfused, load the fused checkpoint via split hook ---
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "patches"))
import qwen3_5moe_unfused_experts as P  # noqa: E402  (patches transformers in place)

unfused = P.Qwen3_5MoeExpertsUnfused(cfg)
missing, unexpected = unfused.load_state_dict(fused_sd, strict=True)
print(f"[1] load-split: OK (missing={list(missing)}, unexpected={list(unexpected)})")
assert isinstance(unfused, torch.nn.ModuleList) and len(unfused) == E
assert hasattr(unfused[0], "down_proj"), "per-expert .down_proj missing (run_reap hook target)"

# --- forward equivalence vs the reference math ---
unfused.eval()
with torch.no_grad():
    out_unfused = unfused(hidden, top_k_index, top_k_weights)
maxerr = (out_ref - out_unfused).abs().max().item()
print(f"[2] forward equivalence vs reference: max|Δ| = {maxerr:.2e}")
assert torch.allclose(out_ref, out_unfused, atol=1e-6), f"forward mismatch ({maxerr})"

# --- save-fuse hook: state_dict() must re-stack to the fused names + values ---
refused_sd = unfused.state_dict()
print(f"[3] save-fuse keys: {sorted(refused_sd)}")
assert set(refused_sd) == {"gate_up_proj", "down_proj"}, sorted(refused_sd)
assert torch.allclose(refused_sd["gate_up_proj"], fused_sd["gate_up_proj"], atol=1e-6)
assert torch.allclose(refused_sd["down_proj"], fused_sd["down_proj"], atol=1e-6)
print("[3] save-fuse round-trip: values match original fused checkpoint")

# --- prune round-trip: drop experts 5 and 2, save as fused 4-expert ckpt ---
for idx in (5, 2):  # delete high-to-low so indices stay valid
    del unfused[idx]
assert len(unfused) == 4
pruned_sd = unfused.state_dict()
assert pruned_sd["gate_up_proj"].shape == (4, 2 * I, H), pruned_sd["gate_up_proj"].shape
assert pruned_sd["down_proj"].shape == (4, H, I), pruned_sd["down_proj"].shape
kept = [0, 1, 3, 4]  # surviving original indices
for new_i, old_i in enumerate(kept):
    assert torch.allclose(pruned_sd["gate_up_proj"][new_i], fused_sd["gate_up_proj"][old_i], atol=1e-6)
    assert torch.allclose(pruned_sd["down_proj"][new_i], fused_sd["down_proj"][old_i], atol=1e-6)
print(f"[4] prune round-trip: 6->4 experts, fused stack = kept rows {kept}")

# ===========================================================================
# run_reap.py integration — router hook (tuple gate) + prune_model end-to-end
# ===========================================================================
import torch.nn as nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
import run_reap as R  # same dir; imports cleanly (applies the qwen3moe patch too)

cfg2 = SimpleNamespace(num_experts=6, hidden_size=16, moe_intermediate_size=8,
                       hidden_act="silu", num_experts_per_tok=2)

# [5] _gate_num_experts recognises the custom router (not an nn.Linear)
router = Qwen3_5MoeTopKRouter(cfg2)
nn.init.normal_(router.weight, std=0.05)
assert R._gate_num_experts(router) == 6, R._gate_num_experts(router)
assert R._gate_num_experts(nn.Linear(16, 6)) == 6
print("[5] _gate_num_experts: router=6, Linear=6 (both recognised)")

# Build a tiny model whose names contain ".layers.{N}.mlp" so the tracker +
# _discover_mlp_modules match (needs a wrapper above .layers).
class TinyMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(c)
        nn.init.normal_(self.gate.weight, std=0.05)
        self.experts = P.Qwen3_5MoeExpertsUnfused(c)
        for p in self.experts.parameters():
            nn.init.normal_(p, std=0.05)
    def forward(self, x):
        _, scores, idx = self.gate(x)
        return self.experts(x, idx, scores)

class TinyDecoder(nn.Module):
    def __init__(self, c): super().__init__(); self.mlp = TinyMLP(c)
class TinyInner(nn.Module):
    def __init__(self, c): super().__init__(); self.layers = nn.ModuleList([TinyDecoder(c)])
class TinyModel(nn.Module):
    def __init__(self, c): super().__init__(); self.model = TinyInner(c)
    def forward(self, x): return self.model.layers[0].mlp(x)

model = TinyModel(cfg2).eval()

# [6] saliency tracker: hooks fire, routing + expert norms accumulate
tracker = R.REAPSaliencyTracker(model, top_k=2, verbose=False)
x = torch.randn(12, cfg2.hidden_size)
with torch.no_grad():
    model(x)
sal = tracker.saliency[0]
assert sal.numel() == 6 and (sal > 0).sum() >= 2, sal
tracker.remove()
survivors = tracker.survivors_per_layer(4)
print(f"[6] saliency tracker: layer0 saliency={[round(v,3) for v in sal.tolist()]}, survivors={survivors[0]}")
assert len(survivors[0]) == 4

# [7] prune_model on the custom-router path: preserves the unfused instance
keep = survivors[0]
R.prune_model(model, survivors)
g = model.model.layers[0].mlp.gate
ex = model.model.layers[0].mlp.experts
assert g.weight.shape == (4, 16) and g.num_experts == 4, (g.weight.shape, g.num_experts)
assert isinstance(ex, P.Qwen3_5MoeExpertsUnfused) and len(ex) == 4, type(ex)
psd = ex.state_dict()  # fuse hook must still fire on the pruned instance
assert set(psd) == {"gate_up_proj", "down_proj"}, sorted(psd)
assert psd["gate_up_proj"].shape == (4, 2 * I, H) and psd["down_proj"].shape == (4, H, I), \
    (psd["gate_up_proj"].shape, psd["down_proj"].shape)
with torch.no_grad():
    model(x)  # post-prune forward must run (4 experts)
print(f"[7] prune_model (router path): gate->[4,16], experts kept {keep}, fused save OK, forward runs")

print("\nALL PASS — Qwen3_5Moe unfuse patch + run_reap router/prune validated in miniature.")
