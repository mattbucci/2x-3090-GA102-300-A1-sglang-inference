#!/usr/bin/env python3
"""Does GPTQ on the calibration Grams generalise to held-out tokens, per Flint Linear family?

Forwards the cached calibration set (96 x 2048) plus a held-out token file through the BF16 model to
layer L on one GPU, builds the Grams the calibrator would see, quantizes with
`quantize_flint_awq_calib`'s own routines and reports the held-out output error (|| X (W_q - W)^T || /
|| X W^T || on the held-out rows) for:
  1. attention q/k/v/o and the shared-expert gate_proj (every token is in the Gram);
  2. the gate_proj / down_proj of three routed experts (most / second-most / median held-out rows) —
     Gram clip vs GPTQ at several dampings, in-sample vs held-out;
  3. the whole routed block of the layer (all experts, routed sum vs BF16) under several expert
     quantization policies — the table that set the calibrator's expert defaults
     (receipt: benchmarks/quality/flint/runaway-repetition-replay-2026-10-22.md section 6).

Usage (serving env, ~1 min at layer 10, ~3 min at layer 45; needs ~22 GB on the GPU):
  PYTHONNOUSERSITE=1 python scripts/quantize/probe_flint_expert_gptq.py --layer 10 \
      --held-ids <ref-ids.json> [--calib-cache /var/tmp/flint-calib/calib-96x2048.json] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
sys.path.insert(0, HERE)
from flint_cpu_reference import Weights, rmsnorm  # noqa: E402
import quantize_flint_awq_calib as Q  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/data/models/hf-3rdparty/flint-bf16")
    ap.add_argument("--calib-cache", default="/var/tmp/flint-calib/calib-96x2048.json")
    ap.add_argument("--held-ids", required=True, help='json with {"input_ids": [...]} (the reference-forward prefix)')
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--shrink-min", type=float, default=0.5)
    ap.add_argument("--skip-linears", action="store_true", help="only the whole-layer policy table")
    args = ap.parse_args()
    L, dev, g = args.layer, args.device, args.group_size
    torch.set_grad_enabled(False)
    cfg = json.load(open(os.path.join(args.base, "config.json")))
    calib = json.load(open(args.calib_cache))
    held = json.load(open(args.held_ids))["input_ids"]
    seqs = [c["ids"] for c in calib] + [held]
    base = Weights(args.base, dev)
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    nope, dense = set(cfg["no_rope_layer_indices"]), set(cfg["mlp_only_layers"])
    win = cfg["sliding_window"] - 1
    E, topk = cfg["num_experts"], cfg["num_experts_per_tok"]
    assert L not in dense, "pick an MoE layer"
    Tmax = max(len(s) for s in seqs)
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    fr = torch.arange(Tmax, device=dev, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    pos = torch.arange(Tmax, device=dev)
    causal = pos[None, :] <= pos[:, None]
    swa = causal & ((pos[:, None] - pos[None, :]) <= win)

    def rope(v, T):
        v1, v2 = v[..., : hd // 2], v[..., hd // 2:]
        return v * cos[:T] + torch.cat([-v2, v1], -1) * sin[:T]

    bounds, o = [], 0
    for s in seqs:
        bounds.append((o, o + len(s)))
        o += len(s)
    X = base.get("model.embed_tokens.weight")[torch.cat([torch.tensor(s, device=dev) for s in seqs])]
    t0 = time.time()

    def attn(l, X):
        p = f"model.layers.{l}."
        sliding = cfg["layer_types"][l] == "sliding_attention"
        h = rmsnorm(X, base.get(p + "input_layernorm.weight"), eps)
        Wq, Wk, Wv, Wo = (base.get(p + f"self_attn.{m}_proj.weight") for m in "qkvo")
        q = rmsnorm((h @ Wq.T).view(-1, nh, hd), base.get(p + "self_attn.q_norm.weight"), eps)
        k = rmsnorm((h @ Wk.T).view(-1, nkv, hd), base.get(p + "self_attn.k_norm.weight"), eps)
        v = (h @ Wv.T).view(-1, nkv, hd)
        sink = base.get(p + "self_attn.softmax_offset")
        O = torch.empty(X.shape[0], nh * hd, device=dev)
        for a, b in bounds:
            T = b - a
            qq, kk, vv = q[a:b], k[a:b], v[a:b]
            if l not in nope:
                qq, kk = rope(qq, T), rope(kk, T)
            kk = kk.repeat_interleave(nh // nkv, 1)
            vv = vv.repeat_interleave(nh // nkv, 1)
            sc = torch.einsum("thd,shd->hts", qq, kk) * hd ** -0.5
            sc = sc.masked_fill(~(swa if sliding else causal)[:T, :T][None], float("-inf"))
            sc = torch.cat([sc, sink[:, None, None].expand(nh, T, 1)], -1)
            O[a:b] = torch.einsum("hts,shd->thd", torch.softmax(sc, -1)[..., :T], vv).reshape(T, nh * hd)
        return X + O @ Wo.T, h, O, (Wq, Wk, Wv, Wo)

    def moe(l, X):
        p = f"model.layers.{l}."
        h = rmsnorm(X, base.get(p + "post_attention_layernorm.weight"), eps)
        if l in dense:
            m = (torch.nn.functional.silu(h @ base.get(p + "mlp.gate_proj.weight").T)
                 * (h @ base.get(p + "mlp.up_proj.weight").T)) @ base.get(p + "mlp.down_proj.weight").T
            return X + m, h, None, None
        sc = torch.sigmoid(h @ base.get(p + "mlp.gate.weight").T)
        top_i = torch.topk(sc, topk, dim=-1, sorted=False).indices
        top_w = sc.gather(1, top_i)
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
        m = (torch.nn.functional.silu(h @ base.get(p + "mlp.shared_expert.gate_proj.weight").T)
             * (h @ base.get(p + "mlp.shared_expert.up_proj.weight").T)) @ base.get(p + "mlp.shared_expert.down_proj.weight").T
        for e in range(E):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            he, pe = h[rows], p + f"mlp.experts.{e}."
            ye = (torch.nn.functional.silu(he @ base.get(pe + "gate_proj.weight").T)
                  * (he @ base.get(pe + "up_proj.weight").T)) @ base.get(pe + "down_proj.weight").T
            m.index_add_(0, rows, ye * top_w[rows, slot][:, None])
        return X + m, h, top_i, top_w

    for l in range(L):
        X, _, _, _ = attn(l, X)
        X, _, _, _ = moe(l, X)
    X, h1, O, (Wq, Wk, Wv, Wo) = attn(L, X)
    _, hh, top_i, top_w = moe(L, X)
    del X
    print(f"forward to layer {L}: {time.time() - t0:.0f}s", flush=True)
    a, b = bounds[-1]  # held-out rows
    p = f"model.layers.{L}."
    qargs = types.SimpleNamespace(grid=args.grid, shrink_min=args.shrink_min)

    def deq(W, s_, z_, q_):
        return ((q_.float().view(W.shape[0], -1, g) - z_.float()[..., None]) * s_.float()[..., None]).view(W.shape)

    def err(W, Wd, XA, XB):
        eA = ((XA @ (Wd - W).T).norm() / (XA @ W.T).norm()).item()
        eB = ((XB @ (Wd - W).T).norm() / (XB @ W.T).norm()).item()
        return f"{eA:.4f}/{eB:.4f}"

    def linear_arms(W, GA, XA, XB, prior=None):
        out = []
        s_, z_, q_ = Q.search_groups_gram(W[None], GA[None], g, qargs.grid, qargs.shrink_min)
        out.append("clipG " + err(W, deq(W, s_[0], z_[0], q_[0]), XA, XB))
        for pd in (0.01, 0.1, 0.3, 1.0):
            s_, z_, q_ = Q.gptq_quantize(W[None], GA[None], g, qargs.grid, qargs.shrink_min, percdamp=pd)
            out.append(f"gptq-d{pd:g} " + err(W, deq(W, s_[0], z_[0], q_[0]), XA, XB))
        if prior is not None:
            tr = GA.diagonal().sum() / prior.diagonal().sum()
            for lam in (0.3, 1.0):
                s_, z_, q_ = Q.gptq_quantize(W[None], (GA + lam * tr * prior)[None], g, qargs.grid, qargs.shrink_min, percdamp=0.1)
                out.append(f"gptq-shrink{lam:g}-d0.1 " + err(W, deq(W, s_[0], z_[0], q_[0]), XA, XB))
        return " | ".join(out)

    G2 = hh[:a].T @ hh[:a]  # layer-wide MoE-input Gram (what the calibrator shrinks toward)
    if not args.skip_linears:
        print("in-sample/held-out output error")
        hA, hB = h1[:a], h1[a:b]
        GA = hA.T @ hA
        for nm, W in zip("qkv", (Wq, Wk, Wv)):
            print(f"  L{L} attn {nm}_proj n={len(hA)}: " + linear_arms(W, GA, hA, hB), flush=True)
        GO = O[:a].T @ O[:a]
        print(f"  L{L} attn o_proj: " + linear_arms(Wo, GO, O[:a], O[a:b]), flush=True)
        Wsg = base.get(p + "mlp.shared_expert.gate_proj.weight")
        print(f"  L{L} shared gate_proj: " + linear_arms(Wsg, G2, hh[:a], hh[a:b]), flush=True)
        del GA, GO
    del h1, O
    cnt_held = torch.bincount(top_i[a:b].flatten(), minlength=E)
    order = torch.argsort(cnt_held, descending=True)
    picks = [order[0].item(), order[1].item(), order[len(order) // 2].item()]
    if not args.skip_linears:
        for e in picks:
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            inA = rows < a
            rA, rB = rows[inA], rows[~inA]
            pe = p + f"mlp.experts.{e}."
            Wg_, Wu_, Wd_ = (base.get(pe + f"{m}_proj.weight") for m in ("gate", "up", "down"))
            wA = top_w[rA, slot[inA]]
            xw = hh[rA] * wA[:, None]
            print(f"  L{L} expert {e} gate_proj n_cal={len(rA)} n_held={len(rB)}: "
                  + linear_arms(Wg_, xw.T @ xw, hh[rA], hh[rB], prior=G2), flush=True)
            mid = torch.nn.functional.silu(hh[rows] @ Wg_.T) * (hh[rows] @ Wu_.T)
            MA = mid[inA] * wA[:, None]
            print(f"  L{L} expert {e} down_proj [{Wd_.shape[0]}x{Wd_.shape[1]}]: "
                  + linear_arms(Wd_, MA.T @ MA, MA, mid[~inA]), flush=True)
    # ---- whole-layer routed block under expert policies
    I_e = cfg["moe_intermediate_size"]
    Wg_all = torch.stack([base.get(p + f"mlp.experts.{e}.gate_proj.weight") for e in range(E)])
    Wu_all = torch.stack([base.get(p + f"mlp.experts.{e}.up_proj.weight") for e in range(E)])
    Wd_all = torch.stack([base.get(p + f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
    H = Wg_all.shape[-1]
    Ge = torch.zeros(E, H, H, device=dev)
    Gd = torch.zeros(E, I_e, I_e, device=dev)
    rowsB, n_cal = {}, []
    for e in range(E):
        rows, slot = (top_i == e).nonzero(as_tuple=True)
        inA = rows < a
        rA, sA = rows[inA], slot[inA]
        rowsB[e] = (rows[~inA], slot[~inA])
        n_cal.append(len(rA))
        if len(rA) == 0:
            continue
        xw = hh[rA] * top_w[rA, sA][:, None]
        Ge[e] = xw.T @ xw
        mw = (torch.nn.functional.silu(hh[rA] @ Wg_all[e].T) * (hh[rA] @ Wu_all[e].T)) * top_w[rA, sA][:, None]
        Gd[e] = mw.T @ mw
    n_cal.sort()
    print(f"grams built {time.time() - t0:.0f}s; calibration rows/expert min/median/max {n_cal[0]}/{n_cal[E // 2]}/{n_cal[-1]}", flush=True)

    def routed(Wg, Wu, Wd):
        out = torch.zeros(b - a, H, device=dev)
        for e in range(E):
            rows, slot = rowsB[e]
            if len(rows) == 0:
                continue
            he = hh[rows]
            ye = (torch.nn.functional.silu(he @ Wg[e].T) * (he @ Wu[e].T)) @ Wd[e].T
            out.index_add_(0, rows - a, ye * top_w[rows, slot][:, None])
        return out

    y0 = routed(Wg_all, Wu_all, Wd_all)

    def quant(Wall, Gall, mode, pd=0.01, shrink=0.0):
        res = []
        for c0 in range(0, E, 32):
            W = Wall[c0:c0 + 32]
            G = None if Gall is None else Gall[c0:c0 + 32].clone()
            if G is not None and shrink > 0:
                tr = G.diagonal(dim1=1, dim2=2).sum(1) / G2.diagonal().sum()
                G += shrink * tr[:, None, None] * G2[None]
            if mode == "clip":
                s_, z_, q_ = Q.search_groups_gram(W, G, g, qargs.grid, qargs.shrink_min)
            else:
                s_, z_, q_ = Q.gptq_quantize(W, G, g, qargs.grid, qargs.shrink_min, percdamp=pd)
            B, out_, inn = W.shape
            res.append(((q_.float().view(B, out_, -1, g) - z_.float()[..., None]) * s_.float()[..., None]).view(B, out_, inn))
        return torch.cat(res)

    policies = {
        "data-free clip (RTN ship)": lambda: (quant(Wg_all, None, "clip"), quant(Wu_all, None, "clip"), quant(Wd_all, None, "clip")),
        "Gram clip": lambda: (quant(Wg_all, Ge, "clip"), quant(Wu_all, Ge, "clip"), quant(Wd_all, Gd, "clip")),
        "GPTQ d0.01 all": lambda: (quant(Wg_all, Ge, "gptq"), quant(Wu_all, Ge, "gptq"), quant(Wd_all, Gd, "gptq")),
        "gu GPTQ shrink1 d0.1 + down clip": lambda: (quant(Wg_all, Ge, "gptq", 0.1, 1.0), quant(Wu_all, Ge, "gptq", 0.1, 1.0), quant(Wd_all, Gd, "clip")),
        "gu GPTQ shrink1 d0.1 + down GPTQ d0.3": lambda: (quant(Wg_all, Ge, "gptq", 0.1, 1.0), quant(Wu_all, Ge, "gptq", 0.1, 1.0), quant(Wd_all, Gd, "gptq", 0.3)),
        "gu GPTQ shrink1 d0.1 + down GPTQ d1 (calibrator default)": lambda: (quant(Wg_all, Ge, "gptq", 0.1, 1.0), quant(Wu_all, Ge, "gptq", 0.1, 1.0), quant(Wd_all, Gd, "gptq", 1.0)),
        "gu GPTQ shrink3 d0.1 + down GPTQ d1": lambda: (quant(Wg_all, Ge, "gptq", 0.1, 3.0), quant(Wu_all, Ge, "gptq", 0.1, 3.0), quant(Wd_all, Gd, "gptq", 1.0)),
        "gu GPTQ on the layer Gram only + down clip": lambda: (quant(Wg_all, G2[None].expand(E, -1, -1), "gptq"), quant(Wu_all, G2[None].expand(E, -1, -1), "gptq"), quant(Wd_all, Gd, "clip")),
    }
    print(f"layer {L} routed-block held-out error vs BF16 (|routed| {y0.norm():.1f})")
    for nm, fn in policies.items():
        Wg, Wu, Wd = fn()
        y = routed(Wg, Wu, Wd)
        print(f"  {nm:58s} {((y - y0).norm() / y0.norm()).item():.4f}", flush=True)
        del Wg, Wu, Wd, y


if __name__ == "__main__":
    main()
