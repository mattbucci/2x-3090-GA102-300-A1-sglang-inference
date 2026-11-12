#!/usr/bin/env python3
"""Miniature validation for qwen3_5moe_unfused_experts.py + run_reap.py — no
62 GB model load. Byte-identical across the 3090 and R9700 repos (the helper
dir resolves by file presence: ream-patches/ first, then patches/).
Checks the unfused patch:

  1. load-split hook: fused 3-D checkpoint -> per-expert Linears loads cleanly;
  2. forward equivalence: unfused output == a hand-rolled reference of the
     documented Qwen3_5MoeExperts inner-loop math (same weights);
  3. save-fuse hook: state_dict() re-stacks to the fused 3-D names/values;
  4. prune round-trip: a shortened ModuleList saves as a fused K-expert ckpt.

and run_reap.py: router hook / prune_model (5-7), helper-dir resolution (8),
composite-config loader + expert-count helpers (9-11), the crash-safe report
(12), and the stream (shard-slicing) save stage against the reload stage on a
real tiny Qwen3_5Moe checkpoint, text-only and composite (13-14).

Run in the `quant` env (transformers + torch). Exit 0 = all pass.
"""
import os
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
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HELPER_DIR = next((os.path.join(_REPO, d) for d in ("ream-patches", "patches")
                    if os.path.isfile(os.path.join(_REPO, d, "qwen3_5moe_unfused_experts.py"))), None)
assert _HELPER_DIR, "qwen3_5moe_unfused_experts.py not found under ream-patches/ or patches/"
sys.path.insert(0, _HELPER_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

# ===========================================================================
# run_reap.py — helper-dir resolution, composite-config helpers, report
# ===========================================================================
import json
import shutil
import tempfile

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

_TMP = tempfile.mkdtemp(prefix="reap-test-")

# [8] dual-dir resolution keys on the helper FILE, not the directory
assert R._PATCH_DIR == _HELPER_DIR, (R._PATCH_DIR, _HELPER_DIR)
fake = os.path.join(_TMP, "repo"); os.makedirs(os.path.join(fake, "ream-patches")); os.makedirs(os.path.join(fake, "patches"))
assert R.resolve_patch_dir(fake) is None                       # both dirs, no helper
open(os.path.join(fake, "patches", "qwen3_5moe_unfused_experts.py"), "w").close()
assert R.resolve_patch_dir(fake) == os.path.join(fake, "patches")       # 3090 layout: empty ream-patches falls through
open(os.path.join(fake, "ream-patches", "qwen3_5moe_unfused_experts.py"), "w").close()
assert R.resolve_patch_dir(fake) == os.path.join(fake, "ream-patches")  # R9700 layout after the copy
print(f"[8] helper-dir resolution: file-keyed (live = {os.path.basename(_HELPER_DIR)}/)")

# [9] composite config (vision_config + text_config, no top-level num_experts)
TEXT = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts=256,
            num_experts_per_tok=8, moe_intermediate_size=8, max_position_embeddings=64)
composite = Qwen3_5MoeConfig(text_config=dict(TEXT))
assert getattr(composite, "vision_config", None) is not None
assert getattr(composite, "num_experts", None) is None, "composite unexpectedly mirrors num_experts at the top"
assert R.pick_model_class(composite) is AutoModelForImageTextToText
assert R.read_expert_counts(composite) == (256, 8), R.read_expert_counts(composite)
print("[9] composite config: pick_model_class -> AutoModelForImageTextToText, read_expert_counts -> (256, 8)")

# [10] text-only config keeps the CausalLM loader
text_only = composite.get_text_config()
assert R.pick_model_class(text_only) is AutoModelForCausalLM
assert R.read_expert_counts(text_only) == (256, 8)
try:
    R.read_expert_counts(SimpleNamespace(hidden_size=16))
    raise AssertionError("read_expert_counts accepted a non-MoE config")
except ValueError:
    pass
print("[10] text-only config: AutoModelForCausalLM, (256, 8); non-MoE config rejected")

# [11] nested-config write: text level always, top level when it mirrors
written = R.set_expert_counts(composite, 192)
assert ("text", "num_experts") in written and composite.get_text_config().num_experts == 192, written
assert json.loads(composite.to_json_string())["text_config"]["num_experts"] == 192
mirror = SimpleNamespace(num_experts=256, num_local_experts=256, text_config=SimpleNamespace(num_experts=256))
written = R.set_expert_counts(mirror, 192)
assert mirror.num_experts == mirror.num_local_experts == mirror.text_config.num_experts == 192, written
print(f"[11] set_expert_counts: composite -> text_config=192 (serialised), mirrored levels {written}")

# [12] crash-safe report: atomic write, survivors round-trip
rep_dir = os.path.join(_TMP, "report")
rp = R.write_report(rep_dir, {"stage": "survivors", "survivors_per_layer": {"0": [3, 1, 0], "1": [2, 4, 5]}})
surv, rep = R.load_survivors(rp)
assert surv == {0: [0, 1, 3], 1: [2, 4, 5]} and rep["stage"] == "survivors" and not os.path.exists(rp + ".tmp")
print("[12] reap_report.json: atomic write + survivors round-trip")


# ===========================================================================
# stream save stage vs reload save stage on a real tiny Qwen3_5Moe checkpoint
# ===========================================================================
def _stream_vs_reload(tag, cfg_obj, n_exp, keep_n, shard_bytes):
    cls = R.pick_model_class(cfg_obj)
    torch.manual_seed(1)
    base = cls.from_config(cfg_obj).eval()
    src = os.path.join(_TMP, f"{tag}-src")
    base.save_pretrained(src)
    for extra in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        with open(os.path.join(src, extra), "w") as f:
            f.write("{}" if extra.endswith(".json") else "{{ messages }}")
    del base
    n_layers = R._text_config(cfg_obj).num_hidden_layers
    g = torch.Generator().manual_seed(2)
    survivors = {L: sorted(torch.randperm(n_exp, generator=g)[:keep_n].tolist()) for L in range(n_layers)}

    # reload stage (the proven path): load unfused, prune in place, save fused
    m = cls.from_pretrained(src, dtype=torch.float32).eval()
    assert type(m).__name__.endswith("ForConditionalGeneration") == (cls is AutoModelForImageTextToText), type(m)
    R.prune_model(m, survivors)
    R.set_expert_counts(m.config, keep_n)
    ref = os.path.join(_TMP, f"{tag}-reload"); m.save_pretrained(ref); del m

    # stream stage: slice the shards, never build the model
    out = os.path.join(_TMP, f"{tag}-stream")
    info = R.stream_prune_checkpoint(src, out, survivors, n_exp, max_shard_bytes=shard_bytes)
    assert info["shards"] > 1, info                       # exercises the multi-shard index
    for extra in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        assert extra in info["files_copied"], info["files_copied"]

    a = cls.from_pretrained(ref, dtype=torch.float32).eval()
    b = cls.from_pretrained(out, dtype=torch.float32).eval()
    assert R.read_expert_counts(b.config) == (keep_n, R.read_expert_counts(cfg_obj)[1])
    sa, sb = a.state_dict(), b.state_dict()
    # the loader really loaded the sliced shards (no silent re-init of experts)
    from safetensors.torch import load_file
    raw = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".safetensors"):
            raw.update(load_file(os.path.join(out, f)))
    probe = next(k for k in raw if k.endswith("layers.0.mlp.experts.down_proj"))
    loaded = next(v for k, v in sb.items() if k.endswith("layers.0.mlp.experts.down_proj"))
    assert raw[probe].shape[0] == keep_n and torch.equal(raw[probe].float(), loaded), probe
    assert sa.keys() == sb.keys(), sorted(set(sa) ^ set(sb))[:5]
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    ids = torch.randint(0, 64, (1, 12), generator=g)
    with torch.no_grad():
        d = (a(input_ids=ids, use_cache=False).logits - b(input_ids=ids, use_cache=False).logits).abs().max().item()
    assert d == 0.0, d
    has_vision = any(k.startswith("model.visual.") for k in sb)
    return info, len(sb), has_vision


small = dict(TEXT, num_experts=6, num_experts_per_tok=2)
info, n_t, vis = _stream_vs_reload("text", Qwen3_5MoeConfig(text_config=dict(small)).get_text_config(), 6, 4, 4096)
assert not vis
print(f"[13] stream vs reload (text-only, 6->4 experts, {info['shards']} shards): {n_t} tensors bit-equal, logits Δ = 0")

vcfg = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16)
comp_small = Qwen3_5MoeConfig(text_config=dict(small), vision_config=vcfg)
info, n_t, vis = _stream_vs_reload("composite", comp_small, 6, 4, 4096)
assert vis, "vision tower missing from the stream-pruned composite checkpoint"
assert info["config_levels_written"] == [("text", "num_experts")], info["config_levels_written"]
print(f"[14] stream vs reload (composite, vision tower kept, {info['shards']} shards): "
      f"{n_t} tensors bit-equal, logits Δ = 0, config written at {info['config_levels_written']}")

# stream stage refuses an unrecognised layout instead of writing a pruned config
try:
    R.stream_prune_checkpoint(os.path.join(_TMP, "text-src"), os.path.join(_TMP, "bad"), {99: [0, 1, 2, 3]}, 6)
    raise AssertionError("stream prune accepted survivors for a layer that does not exist")
except RuntimeError as e:
    assert "sliced no router" in str(e), e
print("[15] stream stage: unknown layer -> RuntimeError (no silent unpruned save)")

shutil.rmtree(_TMP, ignore_errors=True)
print(f"\nALL PASS — Qwen3_5Moe unfuse patch + run_reap helpers / prune / stream save validated in miniature "
      f"(transformers {transformers.__version__}, torch {torch.__version__}).")
