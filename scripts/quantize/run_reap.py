#!/usr/bin/env python3
"""Homegrown REAP (Router-aware Expert Pruning, arxiv:2510.13999) — pure
pytorch + transformers, no vLLM dependency.

Algorithm (per-layer):
  1. Forward `--num-samples` calibration prompts through the BF16 model.
  2. On each MoE layer, hook the router (`mlp.gate`) to capture per-token
     routing probabilities, and hook each expert's `down_proj` to capture
     per-token output L2 norms.
  3. For each expert E in layer L: accumulate its saliency as
        S_{L,E} = sum_t (gate_t_E * ||down_proj_E(intermediate_t_E)||_2)
     summing only over tokens that routed to expert E (top-k routing).
  4. Per layer, keep the top-`--keep-experts` highest-saliency experts;
     drop the rest.
  5. Re-pack the model: ModuleList[Qwen3MoeMLP] is filtered to surviving
     experts; router gate weight rows are sliced to the same surviving set.
     `config.num_experts` and `num_local_experts` are updated to the new size.
  6. Write `reap_report.json` (survivors + saliency) to the save dir BEFORE any
     reload or save, so a failed save never re-runs the observation pass
     (`--survivors-from <report>` resumes from it).
  7. Save the pruned BF16 + tokenizer + processor. Two save stages:
       * `reload` — reload the model on the CPU, prune in place, save_pretrained
         (needs host RAM > the BF16 checkpoint);
       * `stream` — slice the safetensors shards directly (router rows + expert
         rows per layer), never instantiating the model: peak RAM = one shard.
     `--save-mode auto` picks `stream` when the checkpoint would not fit in RAM.

Compared to Cerebras's REAP tool (github.com/CerebrasResearch/reap), this
implementation:
  * pure pytorch — no vLLM dependency (their tool requires vLLM 0.10 pinned)
  * controls memory via explicit `max_memory` per device (their tool uses
    accelerate's auto split which OOMs on R9700 with Coder-30B)
  * per-layer pruning (matches the Cerebras paper)
  * router renormalization is implicit (softmax over surviving rows)

Arch coverage:
  * Qwen3Moe (Coder-30B-A3B etc.) — shipped Coder-30B-A3B-REAP path. ✅
  * Qwen3_5Moe (Qwen3.5/3.6, incl. Qwen3.6-35B-A3B) — ✅ via the companion
    `qwen3_5moe_unfused_experts.py` (fused-3-D experts) + the custom
    `Qwen3_5MoeTopKRouter` tuple-output handling in the saliency hook below.
    Composite (vision) checkpoints load through `pick_model_class` so the
    vision tower survives, and expert counts are read/written on the text
    config (`read_expert_counts` / `set_expert_counts`).
    Miniature-validated by `test_qwen3_5moe_unfuse.py`; on-box run pending.
  * Gemma4Moe / Nemotron-3 — NOT yet: their experts/router layouts still need
    an unfuse patch + (Nemotron-H) Mamba2-hybrid layer skipping.
The saliency hook selectors require `.mlp.gate` (any router type) and per-expert
`.mlp.experts.<i>.down_proj` Linears — the unfuse patches create the latter.

This file, `test_qwen3_5moe_unfuse.py` and `qwen3_5moe_unfused_experts.py` are
kept byte-identical across the 3090 and R9700 repos: the helper directory is
resolved by file presence (`ream-patches/` first, then `patches/`).

Usage (text-only):
    conda activate quant  # (any env with transformers + accelerate + datasets)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/quantize/run_reap.py \\
        --model ~/AI/models/Qwen3-Coder-30B-A3B-BF16 \\
        --save-path ~/AI/models/Qwen3-Coder-30B-A3B-REAP-BF16 \\
        --keep-experts 96 \\
        --dataset theblackcat102/evol-codealpaca-v1 \\
        --num-samples 1024
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


# Apply the unfused-experts monkey-patches BEFORE from_pretrained.
# Required so per-expert nn.Modules survive — the fused 3D Parameter form
# can't be hook-instrumented per-expert.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HELPER_DIRS = ("ream-patches", "patches")
_HELPER_MARKERS = ("qwen3_5moe_unfused_experts.py", "qwen3moe_unfused_experts.py")


def resolve_patch_dir(repo_dir=_REPO_DIR):
    """First helper dir that CONTAINS an unfuse helper file, or None.

    Keyed on the file, not the directory: both repos have a `ream-patches/`
    dir, but the 3090 repo keeps the helpers in `patches/` and the R9700 repo
    in `ream-patches/`. The Qwen3_5Moe helper wins over the Qwen3Moe one so a
    repo carrying them in different dirs still resolves the newer layout."""
    for marker in _HELPER_MARKERS:
        for d in _HELPER_DIRS:
            if os.path.isfile(os.path.join(repo_dir, d, marker)):
                return os.path.join(repo_dir, d)
    return None


_PATCH_DIR = resolve_patch_dir()
if _PATCH_DIR is not None:
    sys.path.insert(0, _PATCH_DIR)
    # Qwen3Moe (Coder-30B etc.) — ships unfused per-expert weights on disk.
    try:
        import qwen3moe_unfused_experts  # noqa: F401
        print("[run_reap] Qwen3MoeExperts → unfused per-expert ModuleList")
    except ImportError as e:
        print(f"[run_reap] note: qwen3moe unfuse patch not applied: {e}")
    # Qwen3_5Moe (Qwen3.5/3.6, incl. Qwen3.6-35B-A3B) — ships FUSED 3-D experts;
    # this patch adds the load-split + save-fuse bridging hooks.
    try:
        import qwen3_5moe_unfused_experts  # noqa: F401
        print("[run_reap] Qwen3_5MoeExperts → unfused per-expert ModuleList (fused-ckpt bridged)")
    except ImportError as e:
        print(f"[run_reap] note: qwen3_5moe unfuse patch not applied: {e}")
else:
    print(f"[run_reap] note: no unfuse helper found under {_HELPER_DIRS} — fused-expert archs will hook 0 experts")


def _text_config(config):
    """The config level that owns the MoE fields: `text_config` on composite
    (vision) configs, the config itself otherwise."""
    get = getattr(config, "get_text_config", None)
    tc = get() if callable(get) else getattr(config, "text_config", None)
    return tc if tc is not None else config


def pick_model_class(config):
    """Loader for `config`: composite configs (a `vision_config` sub-config) go
    through AutoModelForImageTextToText so the vision tower is loaded and
    saved; AutoModelForCausalLM would extract `text_config` and silently drop
    it from the pruned checkpoint."""
    if getattr(config, "vision_config", None) is not None:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def read_expert_counts(config):
    """(num_experts, num_experts_per_tok) from the level that owns them. On the
    Qwen3.6-35B composite the top level carries neither (text_config: 256 / 8)."""
    tc = _text_config(config)
    num_experts = None
    for level in (tc, config):
        num_experts = getattr(level, "num_experts", None) or getattr(level, "num_local_experts", None)
        if num_experts:
            break
    top_k = getattr(tc, "num_experts_per_tok", None) or getattr(config, "num_experts_per_tok", None)
    if not num_experts or not top_k:
        raise ValueError(
            f"{type(config).__name__}: no num_experts / num_experts_per_tok at the text or top "
            f"config level (got {num_experts} / {top_k}) — not an MoE config this tool knows")
    return int(num_experts), int(top_k)


_COUNT_KEYS = ("num_experts", "num_local_experts")


def set_expert_counts(config, keep):
    """Write the pruned expert count at every config level that carries it
    (text_config always when it owns the field; the top level too when it
    mirrors it). Returns the (level, key) pairs written; raises if none."""
    written = []
    tc = _text_config(config)
    levels = [("text", tc)] if tc is not config else []
    levels.append(("top", config))
    for label, level in levels:
        for key in _COUNT_KEYS:
            if getattr(level, key, None) is not None:
                setattr(level, key, int(keep))
                written.append((label, key))
    if not written:
        raise ValueError(f"{type(config).__name__}: no expert-count field to update")
    return written


def write_report(save_path, report):
    """Atomic reap_report.json write (tmp + rename)."""
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, "reap_report.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    return path


def load_survivors(report_path):
    """survivors_per_layer ({int layer: [expert idx]}) from a reap_report.json."""
    with open(report_path) as f:
        report = json.load(f)
    return {int(L): sorted(int(e) for e in keep) for L, keep in report["survivors_per_layer"].items()}, report


_MLP_KEY = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp)\."
    r"(?P<kind>gate\.weight|gate\.bias|experts\.gate_up_proj|experts\.down_proj|experts\.(?P<expert>\d+)\.(?P<rest>.+))$")


def stream_prune_checkpoint(src, dst, survivors, num_experts, max_shard_bytes=2 * 1024**3):
    """Prune a safetensors checkpoint shard by shard without building the model.

    Per MoE layer in `survivors`: router rows (`mlp.gate.weight` / `.bias`) and
    the fused expert stacks (`mlp.experts.gate_up_proj` / `.down_proj`,
    first dim = experts) are `index_select`ed to the survivors; per-expert keys
    (`mlp.experts.<i>.*`, the unfused on-disk layout) are dropped or renumbered.
    Every other tensor is copied. config.json gets the new count at each level
    that carries one; all non-weight files (tokenizer, chat template,
    preprocessor configs) are copied verbatim — the tokenizer is not re-saved,
    so the chat template ships byte-identical. Peak RAM ≈ one shard.

    Raises if any surviving layer ends up without a sliced router AND sliced
    experts — the silent failure mode is an unpruned save under a pruned config.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = os.path.abspath(src), os.path.abspath(dst)
    if src == dst:
        raise ValueError("stream prune needs a save path different from the source")
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards under {src}")
    os.makedirs(dst, exist_ok=True)
    keep_t = {L: torch.tensor(keep, dtype=torch.long) for L, keep in survivors.items()}
    renumber = {L: {old: new for new, old in enumerate(keep)} for L, keep in survivors.items()}
    sliced = {L: {"gate": 0, "experts": 0} for L in survivors}

    out, out_bytes, out_files, weight_map, total = {}, 0, [], {}, 0

    def flush():
        nonlocal out, out_bytes
        if not out:
            return
        name = f"model-{len(out_files) + 1:05d}.safetensors.part"
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
        out_files.append((name, list(out)))
        out, out_bytes = {}, 0

    for shard in shards:
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                m = _MLP_KEY.match(key)
                L = int(m.group("layer")) if m else None
                if m and L in survivors:
                    kind = m.group("kind")
                    if m.group("expert") is not None:
                        old = int(m.group("expert"))
                        if old not in renumber[L]:
                            continue
                        key = f"{m.group('prefix')}.experts.{renumber[L][old]}.{m.group('rest')}"
                        sliced[L]["experts"] += 1
                    else:
                        if t.shape[0] != num_experts:
                            raise RuntimeError(f"{key}: first dim {t.shape[0]} != num_experts {num_experts}")
                        t = t.index_select(0, keep_t[L]).contiguous()
                        sliced[L]["gate" if kind.startswith("gate.") else "experts"] += 1
                nbytes = t.numel() * t.element_size()
                if out and out_bytes + nbytes > max_shard_bytes:
                    flush()
                out[key] = t
                out_bytes += nbytes
                total += nbytes
    flush()

    bad = sorted(L for L, c in sliced.items() if not (c["gate"] and c["experts"]))
    if bad:
        raise RuntimeError(f"stream prune sliced no router and/or experts for layers {bad[:8]} — "
                           f"key layout not recognised; refusing to write a pruned config")

    n = len(out_files)
    for i, (part, keys) in enumerate(out_files, 1):
        final = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
        os.replace(os.path.join(dst, part), os.path.join(dst, final))
        for k in keys:
            weight_map[k] = final
    if n > 1:
        with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    keep_n = len(next(iter(survivors.values())))
    written = []
    for label, level in (("text", cfg.get("text_config")), ("top", cfg)):
        if not isinstance(level, dict):
            continue
        for key in _COUNT_KEYS:
            if level.get(key) is not None:
                level[key] = keep_n
                written.append((label, key))
    if not written:
        raise RuntimeError("config.json carries no num_experts / num_local_experts at either level")
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    skip = {"config.json", "model.safetensors.index.json", "reap_report.json"}
    copied = []
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if name in skip or name.endswith(".safetensors") or name.startswith(".") or not os.path.isfile(path):
            continue
        shutil.copy2(path, os.path.join(dst, name))
        copied.append(name)
    return {"shards": n, "tensors": len(weight_map), "total_bytes": total,
            "config_levels_written": written, "files_copied": copied}


def scan_checkpoint_layout(src, num_experts):
    """Header-only scan (no tensor data read) of the MoE keys the stream stage
    will slice. Returns {layer: {"gate": n, "experts": n}}; raises on a fused
    stack / router whose first dim is not `num_experts`."""
    from safetensors import safe_open

    layers = {}
    for shard in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for key in f.keys():
                m = _MLP_KEY.match(key)
                if not m:
                    continue
                c = layers.setdefault(int(m.group("layer")), {"gate": 0, "experts": 0})
                if m.group("expert") is None:
                    shape = f.get_slice(key).get_shape()
                    if shape[0] != num_experts:
                        raise RuntimeError(f"{key}: first dim {shape[0]} != num_experts {num_experts}")
                c["gate" if m.group("kind").startswith("gate.") else "experts"] += 1
    return layers


def _checkpoint_bytes(path):
    if not os.path.isdir(path):
        return None
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))


def _gate_num_experts(mod):
    """Return the expert count of an MoE router/gate module, or None if `mod`
    isn't one. Handles both the plain `nn.Linear` gate (Qwen3Moe) and custom
    routers like `Qwen3_5MoeTopKRouter` (a raw `weight [num_experts, hidden]`
    Parameter that returns a (logits, scores, indices) tuple)."""
    if isinstance(mod, torch.nn.Linear):
        return mod.out_features
    ne = getattr(mod, "num_experts", None)
    if ne is not None:
        return int(ne)
    w = getattr(mod, "weight", None)
    if w is not None and w.dim() == 2:
        return int(w.shape[0])
    return None


class REAPSaliencyTracker:
    """Hook every MoE router + expert.down_proj; accumulate per-expert saliency.

    The saliency formula is a simplification of the Cerebras paper:
        S_{L,E} = sum_t (gate_t_E * ||down_proj_E(intermediate_t_E)||_2)
    where the sum is over calibration tokens that the router sent to expert E
    (i.e., E was in the top-k for token t).

    We accumulate per-expert across the full calibration corpus, then
    `top_k_survivors` returns the layer-wise pruned set.
    """

    def __init__(self, model: torch.nn.Module, top_k: int, verbose: bool = True):
        self.top_k = top_k
        self.verbose = verbose
        # layer_idx -> tensor[num_experts] (float64 for accumulation precision)
        self.saliency: dict[int, torch.Tensor] = {}
        # transient: layer_idx -> (topk_probs[B*T,top_k], topk_idx[B*T,top_k]) cpu
        self._routing_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks: list = []
        self._install(model)

    def _install(self, model: torch.nn.Module):
        n_routers = 0
        n_experts_hooked = 0
        for name, mod in model.named_modules():
            n_exp_gate = _gate_num_experts(mod) if name.endswith(".mlp.gate") else None
            if n_exp_gate is not None:
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                self.saliency[layer_idx] = torch.zeros(n_exp_gate, dtype=torch.float64)
                h = mod.register_forward_hook(self._make_router_hook(layer_idx))
                self.hooks.append(h)
                n_routers += 1
            elif ".mlp.experts." in name and name.endswith(".down_proj") and isinstance(mod, torch.nn.Linear):
                # name = model.layers.{L}.mlp.experts.{E}.down_proj
                try:
                    parts = name.split(".")
                    L = int(parts[parts.index("layers") + 1])
                    E = int(parts[parts.index("experts") + 1])
                except (IndexError, ValueError):
                    continue
                h = mod.register_forward_hook(self._make_expert_hook(L, E))
                self.hooks.append(h)
                n_experts_hooked += 1
        if self.verbose:
            print(f"[REAPSaliencyTracker] hooked {n_routers} routers + {n_experts_hooked} expert.down_proj modules")

    def _make_router_hook(self, layer_idx: int):
        def hook(module, inp, out):
            with torch.no_grad():
                if isinstance(out, (tuple, list)):
                    # Qwen3_5MoeTopKRouter: (router_logits, router_scores, router_indices)
                    # — already top-k'd; use the scores/indices directly.
                    topk_probs = out[1].detach().float()
                    topk_idx = out[2].detach()
                    if topk_probs.dim() > 2:
                        topk_probs = topk_probs.reshape(-1, topk_probs.shape[-1])
                        topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
                else:
                    # Plain nn.Linear gate (Qwen3Moe): raw logits over all experts.
                    logits = out.detach().float()
                    if logits.dim() > 2:
                        logits = logits.reshape(-1, logits.shape[-1])  # flatten batch + seq
                    probs = logits.softmax(dim=-1)
                    topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
                # Cache on CPU so expert hooks can read it without GPU sync overhead per call
                self._routing_cache[layer_idx] = (topk_probs.cpu(), topk_idx.cpu())
        return hook

    def _make_expert_hook(self, layer_idx: int, expert_idx: int):
        def hook(module, inp, out):
            with torch.no_grad():
                # out: [num_tokens_routed_to_this_expert, hidden_size]
                if out.numel() == 0:
                    return
                # Per-routed-token L2 norm of the expert's output
                output_norms = out.detach().float().norm(dim=-1)  # [num_routed]
                num_routed = output_norms.numel()
                if num_routed == 0:
                    return
                # Lookup average routing prob this expert received in the current batch
                # (we don't have token-by-token alignment between the gate hook's flat
                # output and the expert's filtered token batch, so we use the mean
                # routing prob for this expert as an approximation; sum_t gate_t * norm
                # collapses to mean(gate) * sum(norm) under that approximation, which
                # matches Cerebras's gating-weighted formula in expectation).
                cache = self._routing_cache.get(layer_idx)
                if cache is None:
                    weight = 1.0
                else:
                    topk_probs, topk_idx = cache
                    # mask of (token, k) positions where this expert was selected
                    mask = (topk_idx == expert_idx)
                    if mask.any():
                        weight = topk_probs[mask].mean().item()
                    else:
                        weight = 0.0
                contribution = float(weight) * float(output_norms.sum().item())
                self.saliency[layer_idx][expert_idx] += contribution
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self._routing_cache.clear()

    def survivors_per_layer(self, k: int) -> dict[int, list[int]]:
        """For each MoE layer return the top-k expert indices (sorted ascending)."""
        out = {}
        for L in sorted(self.saliency):
            sal = self.saliency[L]
            top = sal.topk(min(k, sal.numel())).indices.tolist()
            out[L] = sorted(top)
        return out


def _discover_mlp_modules(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Walk the model and find {layer_idx: mlp_module} for every MoE layer.

    Path-agnostic: works for Qwen3MoeForCausalLM (model.model.layers[i].mlp) AND
    Qwen3.5/3.6-VL classes where the language model is nested deeper
    (model.language_model.layers[i].mlp etc.). We match on the trailing
    `.layers.{N}.mlp` suffix in named_modules and parse N from the path.
    """
    layer_to_mlp: dict[int, torch.nn.Module] = {}
    for name, mod in model.named_modules():
        # mlp module itself, not children — has both .gate and .experts attrs
        if not (hasattr(mod, "gate") and hasattr(mod, "experts")):
            continue
        # name format: ...layers.{L}.mlp (or similar — strip trailing .mlp and look for layers)
        if ".layers." not in name:
            continue
        try:
            tail = name.split(".layers.", 1)[1]
            layer_idx = int(tail.split(".")[0])
        except (IndexError, ValueError):
            continue
        layer_to_mlp[layer_idx] = mod
    return layer_to_mlp


def prune_model(model: torch.nn.Module, survivors_per_layer: dict[int, list[int]]) -> None:
    """In-place: drop pruned experts and slice router gate weight rows to surviving set.

    Two gate flavors:
      * Qwen3Moe — `mlp.gate` is an `nn.Linear [num_experts, hidden]`; replace it
        with a sliced Linear and rebuild experts as a plain ModuleList (the
        original, proven Coder-30B-A3B-REAP path, unchanged).
      * Qwen3_5Moe — `mlp.gate` is a `Qwen3_5MoeTopKRouter` (raw `weight`
        Parameter + `num_experts`); slice the Parameter in place and prune the
        experts by in-place deletion so the `Qwen3_5MoeExpertsUnfused` instance
        (and its fuse-on-save hook) survives → the pruned model saves as a
        standard fused Qwen3_5Moe checkpoint.
    """
    mlp_by_layer = _discover_mlp_modules(model)
    missing = sorted(set(survivors_per_layer) - set(mlp_by_layer))
    if missing:
        raise RuntimeError(
            f"_discover_mlp_modules failed to find mlp for layers {missing[:5]}... "
            f"(found {len(mlp_by_layer)} MoE layers; survivors expects {len(survivors_per_layer)})"
        )
    for layer_idx, keep in survivors_per_layer.items():
        layer_mlp = mlp_by_layer[layer_idx]
        gate = layer_mlp.gate
        keep_t = torch.tensor(keep, dtype=torch.long, device=gate.weight.device)
        if isinstance(gate, torch.nn.Linear):
            new_gate = torch.nn.Linear(gate.in_features, len(keep), bias=(gate.bias is not None),
                                       device=gate.weight.device, dtype=gate.weight.dtype)
            with torch.no_grad():
                new_gate.weight.copy_(gate.weight.index_select(0, keep_t))
                if gate.bias is not None:
                    new_gate.bias.copy_(gate.bias.index_select(0, keep_t))
            layer_mlp.gate = new_gate
            old_experts = layer_mlp.experts
            layer_mlp.experts = torch.nn.ModuleList([old_experts[i] for i in keep])
        else:
            # Custom router (Qwen3_5MoeTopKRouter): slice the raw weight Parameter.
            with torch.no_grad():
                new_w = gate.weight.data.index_select(0, keep_t).clone()
            gate.weight = torch.nn.Parameter(new_w, requires_grad=gate.weight.requires_grad)
            if hasattr(gate, "num_experts"):
                gate.num_experts = len(keep)
            # Prune experts IN PLACE (preserve the unfused instance + its hooks).
            experts = layer_mlp.experts
            keep_set = set(keep)
            for i in reversed(range(len(experts))):
                if i not in keep_set:
                    del experts[i]
            if hasattr(experts, "num_experts"):
                experts.num_experts = len(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or HF id of BF16 base model")
    p.add_argument("--save-path", required=True, help="Output dir for pruned BF16")
    p.add_argument("--keep-experts", type=int, default=96, help="Experts to KEEP per layer")
    p.add_argument("--dataset", default="theblackcat102/evol-codealpaca-v1")
    p.add_argument("--text-field", default=None,
                   help="Field name in the dataset for the prompt text. Default = auto-detect.")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-memory-per-gpu", default="20GiB",
                   help="Per-GPU max memory for accelerate device_map (24GB 3090s: ~20GiB; "
                        "32GB R9700s: pass 28GiB)")
    p.add_argument("--max-memory-cpu", default="auto",
                   help="CPU offload cap for accelerate device_map. 'auto' = host RAM minus a "
                        "14 GiB headroom (this 62 GB box → ~48GiB). A fixed >RAM value OOMs the box.")
    p.add_argument("--save-mode", choices=("auto", "reload", "stream"), default="auto",
                   help="reload = CPU reload + save_pretrained (needs RAM > checkpoint); "
                        "stream = slice the safetensors shards directly (RAM ≈ one shard); "
                        "auto = stream when the checkpoint exceeds host RAM minus 14 GiB.")
    p.add_argument("--survivors-from", default=None,
                   help="reap_report.json from an earlier run: skip the observation pass and "
                        "go straight to the save stage with its survivors.")
    p.add_argument("--preflight", action="store_true",
                   help="CPU-only check, no weights loaded: config → loader class + expert counts, "
                        "helper import, shard key layout for the stream stage. Exit 0 = ready.")
    args = p.parse_args()

    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    print(f"Model:    {args.model}")
    print(f"Save:     {args.save_path}")
    print(f"Keep:     {args.keep_experts} experts per layer")
    print(f"Dataset:  {args.dataset} ({args.num_samples} samples × {args.max_length} tokens)")
    print(f"RAM:      {ram_gb:.1f} GB")

    # Config-only preflight: fail before the multi-ten-GB load, not after it.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cls = pick_model_class(config)
    num_experts, top_k = read_expert_counts(config)
    print(f"Config:   {type(config).__name__} → {model_cls.__name__}; num_experts={num_experts}, top_k={top_k}")
    if args.keep_experts >= num_experts:
        raise ValueError(f"--keep-experts {args.keep_experts} ≥ num_experts {num_experts}; nothing to prune")

    save_mode = args.save_mode
    ckpt_bytes = _checkpoint_bytes(args.model)
    if save_mode == "auto":
        fits = ckpt_bytes is None or ckpt_bytes / 1024**3 < ram_gb - 14
        save_mode = "reload" if fits else "stream"
    if save_mode == "stream" and ckpt_bytes is None:
        raise ValueError("--save-mode stream needs a local checkpoint directory")
    ckpt_gb = "n/a" if ckpt_bytes is None else f"{ckpt_bytes / 1024**3:.1f} GiB"
    print(f"Save mode: {save_mode} (checkpoint {ckpt_gb})")

    if args.preflight:
        if ckpt_bytes is None:
            raise ValueError("--preflight needs a local checkpoint directory")
        layers = scan_checkpoint_layout(args.model, num_experts)
        bad = sorted(L for L, c in layers.items() if not (c["gate"] and c["experts"]))
        print(f"Layout:   {len(layers)} MoE layers with router + expert keys"
              + (f"; INCOMPLETE layers {bad[:8]}" if bad else ""))
        n_cfg = getattr(_text_config(config), "num_hidden_layers", None)
        ok = bool(layers) and not bad and (n_cfg is None or len(layers) == n_cfg)
        print(f"Preflight: {'OK' if ok else 'FAIL'} (config num_hidden_layers={n_cfg})")
        sys.exit(0 if ok else 1)

    report = {
        "args": vars(args),
        "original_num_experts": num_experts,
        "kept_experts_per_layer": args.keep_experts,
        "save_mode": save_mode,
    }

    if args.survivors_from:
        survivors, prior = load_survivors(args.survivors_from)
        report["saliency_per_layer"] = prior.get("saliency_per_layer", {})
        report["survivors_from"] = os.path.abspath(args.survivors_from)
        bad = sorted(L for L, keep in survivors.items() if len(keep) != args.keep_experts)
        if bad:
            raise ValueError(f"{args.survivors_from}: layers {bad[:5]} keep ≠ --keep-experts {args.keep_experts}")
        if save_mode == "reload":
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"\n[1-4/4] Survivors for {len(survivors)} layers read from {args.survivors_from}; observation skipped")
    else:
        # Build max_memory dict for accelerate. CPU cap must stay under host RAM —
        # the original R9700 hardcode of 100GiB OOMs this 62 GB box (and any host
        # where the BF16 base + activations exceed RAM during offload).
        cpu_cap = args.max_memory_cpu
        if cpu_cap == "auto":
            cpu_cap = f"{max(8, int(ram_gb) - 14)}GiB"
        n_visible = torch.cuda.device_count()
        max_memory = {i: args.max_memory_per_gpu for i in range(n_visible)}
        max_memory["cpu"] = cpu_cap
        print(f"max_memory: {max_memory} (n_visible_gpus={n_visible})")

        print("\n[1/4] Loading tokenizer + model on multi-GPU...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = model_cls.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        model.eval()
        print(f"  loaded in {time.time()-t0:.0f}s ({type(model).__name__})")

        print("\n[2/4] Installing REAPSaliencyTracker hooks...")
        tracker = REAPSaliencyTracker(model, top_k=top_k)
        if not tracker.saliency:
            raise RuntimeError("no MoE router hooked (.mlp.gate) — nothing to observe; "
                               "an unfuse helper is probably missing for this arch")

        print(f"\n[3/4] Loading {args.dataset} + running observation pass...")
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
        print(f"  loaded {len(ds)} samples; columns: {ds.column_names[:6]}")

        # Auto-detect text field
        text_field = args.text_field
        if text_field is None:
            for cand in ("text", "instruction", "prompt", "content", "input", "messages"):
                if cand in ds.column_names:
                    text_field = cand
                    break
        if text_field is None:
            text_field = ds.column_names[0]
        print(f"  using text-field: {text_field!r}")

        t0 = time.time()
        n_processed = 0
        for i, sample in enumerate(ds):
            text = sample.get(text_field) or ""
            # Some datasets pair instruction+output; concatenate if available
            if "output" in sample and sample.get("output"):
                text = f"{text}\n{sample['output']}"
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer(text, return_tensors="pt", max_length=args.max_length,
                            truncation=True).input_ids.to(next(model.parameters()).device)
            try:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                n_processed += 1
            except Exception as e:
                print(f"  WARN: sample {i} failed ({type(e).__name__}: {e})")
                continue
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(ds) - i - 1) / rate
                print(f"  {i+1}/{len(ds)}  ({rate:.1f} sample/s, ETA {eta/60:.1f}min)")
        elapsed = time.time() - t0
        print(f"  observation done: {n_processed}/{len(ds)} samples in {elapsed/60:.1f}min")
        if n_processed == 0:
            raise RuntimeError("every calibration sample failed — refusing to prune on zero saliency")

        tracker.remove()

        print(f"\n[4/4] Computing per-layer survivors (keep top-{args.keep_experts})...")
        survivors = tracker.survivors_per_layer(args.keep_experts)
        report["saliency_per_layer"] = {str(L): tracker.saliency[L].tolist() for L in tracker.saliency}
        report["samples_observed"] = n_processed
        del model, tracker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_layers = len(survivors)
    first = min(survivors)
    print(f"  {n_layers} MoE layers; example survivors layer {first}: {survivors[first][:10]}...")
    report["num_layers_pruned"] = n_layers
    report["survivors_per_layer"] = {str(L): survivors[L] for L in survivors}

    # Crash-safe: the survivors are on disk before any reload / slicing, so a
    # failed save resumes with --survivors-from instead of re-observing.
    report["stage"] = "survivors"
    path = write_report(args.save_path, report)
    print(f"  {path} written (stage=survivors; resume a failed save with --survivors-from)")

    if save_mode == "stream":
        print(f"\n[save/stream] Slicing safetensors shards {args.model} → {args.save_path} ...")
        report["stream"] = stream_prune_checkpoint(args.model, args.save_path, survivors, num_experts)
        print(f"  {report['stream']['shards']} shards, {report['stream']['tensors']} tensors, "
              f"config levels {report['stream']['config_levels_written']}")
    else:
        # Reload the model to CPU only, then prune and save.
        # device_map="auto" with CPU offload leaves params as meta tensors after
        # dispatch, which crashes save_pretrained at safetensors._tobytes
        # (NotImplementedError: Cannot copy out of meta tensor). Loading to CPU
        # avoids the meta path entirely so save_pretrained gets real tensors.
        print("\n[save/reload] Reloading on CPU for prune + save...")
        model = model_cls.from_pretrained(
            args.model,
            device_map=None,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        model.eval()

        print("\nPruning experts + slicing router weights in-place...")
        prune_model(model, survivors)
        report["config_levels_written"] = set_expert_counts(model.config, args.keep_experts)

        print(f"\nSaving pruned BF16 to {args.save_path}...")
        model.save_pretrained(args.save_path, max_shard_size="2GB")
        # Processor (preprocessor_config.json / video_preprocessor_config.json):
        # best effort — text-only checkpoints have none.
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.save_path)
            report["processor_saved"] = True
        except Exception as e:
            report["processor_saved"] = False
            print(f"  note: no processor saved ({type(e).__name__})")

        tokenizer.save_pretrained(args.save_path)

    report["stage"] = "saved"
    write_report(args.save_path, report)
    print("  reap_report.json updated (stage=saved)")
    print("\nDone.")


if __name__ == "__main__":
    main()
