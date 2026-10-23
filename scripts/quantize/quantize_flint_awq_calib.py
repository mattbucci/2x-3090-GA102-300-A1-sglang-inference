#!/usr/bin/env python3
"""Flint (RecursiveSI/flint) BF16 -> native AWQ-Marlin W4A16 g64, ACTIVATION-AWARE.

Same output format / coverage / packer as `quantize_flint_rtn_awq.py` (data-free MSE-RTN,
the 2026-10-22 ship), but the rounding is calibrated the AWQ way on the model's own
activations — the serving-independent reference forward (`scripts/eval/flint_cpu_reference.py`)
showed that the data-free ship loses 36 % top-1 agreement with the BF16 base over a real
agentic prompt (mean KL 0.73 nats) although every tensor's weight error is a uniform 8.8 %
RMS: the damage is how the rounding error projects onto the activations, which weight-only
rounding cannot see. Attribution (one family INT4 at a time, KL to BF16): shared expert 0.43,
attention 0.37 (qkv 0.25 / o_proj 0.28), the two dense MLPs 0.16, all 256 x 47 routed experts 0.26.

What actually moves it (3-layer CPU smoke, hidden-state error after layer 3, held-out prompt):
data-free RTN 0.237 -> AWQ scale search + Gram clip 0.228 -> + GPTQ error feedback 0.173
(in-sample 0.269 -> 0.257 -> 0.074). The per-channel scale search recovers only 3-14 % of a
layer's loss on this model; --gptq is the lever, the scale search / clip are kept because they
are free and set the group ranges GPTQ rounds into.

GPTQ generalises only where the Gram is well estimated (layer-10 held-out Linear output error,
96 x 2048 calibration tokens): attention q/k/v/o and the shared expert see every token
(68 rows per input dim) and gain 20-36 % over the clip (o_proj 0.085 -> 0.054, insensitive to
damping 1-30 %). A routed expert sees 0.5-12K rows for 2048 dims (median 2.7K) and GPTQ at 1 %
damping OVERFITS that Gram: in-sample 0.036 vs held-out 0.065 against clip 0.070 for gate_proj,
and expert down_proj comes out WORSE than the clip (0.090-0.114 vs 0.083-0.091). That is why the
first GPTQ-densebf16 build only moved the end-to-end KL 0.278 -> 0.270. Fix (whole-layer routed
block, held-out, layer 10): RTN 0.121 / as-built GPTQ 0.116 / expert gate+up GPTQ on the expert
Gram shrunk toward the layer-wide input Gram (--expert-gram-shrink 1, trace-matched) at 10 %
damping + expert down GPTQ at 100 % damping (--expert-down-percdamp 1) 0.098. Those are the
defaults; the always-active path keeps --percdamp 0.01.

Algorithm (AutoAWQ / llm-awq, layer by layer, fp activations propagated):
  1. run the calibration tokens through the BF16 layer, capturing the input of every
     quantized Linear and its Gram matrix  G = sum_t x_t x_t^T  (routing-weighted for experts);
  2. per input-sharing group, search the per-input-channel scale  s = x_mean^a / w_mean^(1-a)
     (a in 0..1, plus s = 1) minimising  sum_consumers tr(D G D^T),  D = W - Q(W s) s^-1  — the
     output error of the pseudo-quantized consumers over the captured tokens; fold s into the
     producer (input_layernorm for q/k/v; post_attention_layernorm + the BF16 router + every
     gate/up for the MoE block; the matching up_proj rows for each down_proj);
  3. quantize each tensor with the same per-(row, group) clipping-range search as the RTN
     script, but minimising the Gram-weighted output error instead of the weight MSE;
  3b. --gptq: instead of rounding W independently per group, run GPTQ error feedback over the
     input dimension on the same Gram (damped Cholesky inverse, block 128, act-order by diag H
     with STATIC groups — scale/zero come from the clip-searched unpermuted weights, so the
     output stays plain AWQ layout with no g_idx); batched over the experts of a layer;
  4. pack to AutoAWQ GEMM (SGLang AWQ-Marlin reads it), write BF16 for everything else.
o_proj is clipped but not scaled (GQA: its input channels of the q heads sharing a kv head would
need one scale — AutoAWQ skips it too). Experts with fewer than --min-expert-tokens routed
calibration tokens fall back to the data-free search.

Calibration mix (--n-samples x --seq-len tokens, cached to --calib-cache): AM-Thinking code
reasoning rendered through Flint's chat template (<think> history kept), raw code from
the-stack-smol (5 languages), opencode agentic transcript chunks (a held-out session:
system prompt + tool schemas + tool calls/results + reasoning), SWE-bench Lite problem
statements as user turns, Open-Platypus instruction pairs.

Usage (serving env, one GPU):
  PYTHONNOUSERSITE=1 python scripts/quantize/quantize_flint_awq_calib.py \
      --src /data/models/hf-3rdparty/flint-bf16 --dst /data/models/hf-mattbucci/Flint-AWQ-INT4-g64-gptq \
      --device cuda:0 --calib-cache /var/tmp/flint-calib/calib-96x2048.json --gptq
  (--calib-only builds the cache first; the gated HF sources need HF_TOKEN in the environment.)
  --keep-bf16 'self_attn|shared_expert|layers\\.[01]\\.mlp' keeps the always-active path BF16
  (experts still calibrated). Then: scripts/eval/check_awq_scales.py <dst> --base <src>;
  scripts/eval/flint_cpu_reference.py + compare_prompt_logprobs.py for the KL receipt.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_flint_rtn_awq import (  # noqa: E402
    LAYER_RE, MAX_Q, PACK, QUANT_RE, W_BIT, pack_awq, read_headers, search_groups,
)

SOURCES = ("am-code", "stack", "agentic", "swebench", "platypus")


# ----------------------------------------------------------------------------- calibration data
def _render(tok, msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, **kw)


def build_calib_ids(tok, src_dir: str, n_samples: int, seq_len: int, seed: int, repo_dir: str):
    """Token-id lists (each <= seq_len) from the five sources, roughly 1/3 reasoning, 1/4 code,
    1/4 agentic transcript, the rest task text."""
    rng = random.Random(seed)
    quota = {"am-code": n_samples // 3, "stack": n_samples // 4, "agentic": n_samples // 4}
    rest = n_samples - sum(quota.values())
    quota["swebench"] = rest // 2
    quota["platypus"] = rest - rest // 2
    out: list[dict] = []
    from datasets import load_dataset

    def take(source, text_iter):
        n = 0
        for text in text_iter:
            ids = text if isinstance(text, list) else tok.encode(text)
            if len(ids) < 256:
                continue
            if len(ids) > seq_len:  # random window so long samples are not all prefixes
                off = rng.randint(0, len(ids) - seq_len)
                ids = ids[off: off + seq_len]
            out.append({"source": source, "ids": ids})
            n += 1
            if n >= quota[source]:
                break
        print(f"  {source}: {n} samples", flush=True)

    # 1. reasoning + code (AM-Thinking code split), rendered with reasoning_content history
    def am_iter():
        ds = load_dataset("a-m-team/AM-Thinking-v1-Distilled", split="train", streaming=True,
                          data_files="code.jsonl")
        for ex in ds.shuffle(seed=seed, buffer_size=2000):
            conv = ex["conversations"]
            if len(conv) < 2:
                continue
            info = conv[1].get("info") or {}
            think, ans = info.get("think_content"), info.get("answer_content")
            if not think or not ans:
                continue
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": ex["system"]})
            msgs += [{"role": "user", "content": conv[0]["value"]},
                     {"role": "assistant", "reasoning_content": think, "content": ans}]
            yield _render(tok, msgs)
    take("am-code", am_iter())

    # 2. raw code
    def stack_iter():
        langs = ["python", "python", "javascript", "typescript", "go", "rust"]
        its = []
        for lang in langs:
            ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}", split="train", streaming=True)
            its.append(iter(ds.shuffle(seed=seed, buffer_size=500)))
        while True:
            for it in its:
                ex = next(it)
                yield ex["content"]
    take("stack", stack_iter())

    # 3. agentic transcript chunks — the held-out session (14182 k20), never 12907 (the replay test set)
    def agentic_iter():
        sys.path.insert(0, os.path.join(repo_dir, "scripts", "eval"))
        from flint_cpu_reference import render_ids
        ids = render_ids(src_dir, os.path.join(repo_dir, "benchmarks/quality/flint/replay/replay-14182-k20.json.gz"), 10**9)
        starts = list(range(0, max(1, len(ids) - seq_len), seq_len // 2))
        rng.shuffle(starts)
        for st in starts:
            yield ids[st: st + seq_len]  # ids straight through (special tokens would not round-trip a decode)
    take("agentic", agentic_iter())

    # 4. SWE-bench Lite problem statements as user turns (task text the eval sees)
    def swe_iter():
        import pyarrow.parquet as pq
        fs = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--princeton-nlp--SWE-bench_Lite/snapshots/*/data/test-*.parquet"))
        rows = pq.read_table(fs[0], columns=["problem_statement"]).column("problem_statement").to_pylist() if fs else \
            [r["problem_statement"] for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")]
        rng.shuffle(rows)
        for ps in rows:
            yield _render(tok, [{"role": "user", "content": ps}], add_generation_prompt=True)
    take("swebench", swe_iter())

    # 5. general instruction pairs
    def platypus_iter():
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx:
            ex = ds[i]
            yield _render(tok, [{"role": "user", "content": ex["instruction"]}, {"role": "assistant", "content": ex["output"]}])
    take("platypus", platypus_iter())
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------- quantizer pieces
def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def pseudo_quant(W: torch.Tensor, group: int) -> torch.Tensor:
    """Plain min-max asymmetric 4-bit round trip per group (the AWQ scale-search surrogate)."""
    out, inn = W.shape
    Wg = W.view(out, inn // group, group)
    mx, mn = Wg.amax(-1, keepdim=True), Wg.amin(-1, keepdim=True)
    s = ((mx - mn) / MAX_Q).clamp(min=1e-5)
    z = (-mn / s).round().clamp(0, MAX_Q)
    q = (Wg / s + z).round().clamp(0, MAX_Q)
    return ((q - z) * s).view(out, inn)


def gram_loss(D: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """tr(D G D^T): output error energy of weight error D [out,in] over tokens with Gram G [in,in]."""
    return ((D @ G) * D).sum()


def search_scale(x_mean: torch.Tensor, consumers: list[tuple[torch.Tensor, torch.Tensor]], group: int,
                 n_grid: int = 11):
    """AWQ per-input-channel scale shared by `consumers` [(W [out,in] fp32, Gram [in,in])].

    Returns (s [in], best_alpha or None for s=1, loss_ratio vs unscaled). n_grid 0 -> s = 1."""
    inn = x_mean.numel()
    if n_grid <= 0 or not consumers:
        return torch.ones(inn, device=x_mean.device), None, 1.0
    Wcat = torch.cat([W for W, _ in consumers], 0).abs()
    wg = Wcat.view(-1, group)
    w_mean = (wg / (wg.amax(1, keepdim=True) + 1e-6)).view(-1, inn).mean(0)
    best = (float("inf"), None, None)
    base = None
    cands = [(None, torch.ones(inn, device=x_mean.device))]
    for a in torch.linspace(0, 1, n_grid).tolist():
        s = (x_mean.clamp(min=1e-5).pow(a) / (w_mean.pow(1 - a) + 1e-4)).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        cands.append((round(a, 2), s))
    for a, s in cands:
        loss = 0.0
        for W, G in consumers:
            D = pseudo_quant(W * s, group) / s - W
            loss += gram_loss(D, G).item()
        if a is None:
            base = loss
        if loss < best[0]:
            best = (loss, a, s)
    return best[2], best[1], (best[0] / base if base else float("nan"))


def search_groups_gram(W: torch.Tensor, G: torch.Tensor | None, group: int, grid: int, shrink_min: float,
                       ridge: float = 1e-2):
    """Per-(row, group) clipping search minimising the Gram-weighted output error (batched).

    W [B,out,in] fp32, G [B,in,in] or None (-> plain weight MSE). Same contract as
    quantize_flint_rtn_awq.search_groups per batch element: scales fp16 [B,out,in/g] (rounded as
    stored), zeros int32 [B,out,in/g], q uint8 [B,out,in]."""
    B, out, inn = W.shape
    ng = inn // group
    Wg = W.view(B, out, ng, group)
    if G is None:
        Gb = None
    else:
        Gb = G.view(B, ng, group, ng, group).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)  # [B,ng,g,g]
        d = Gb.diagonal(dim1=2, dim2=3).mean(2).clamp(min=1e-12)  # [B,ng]
        Gb = Gb + ridge * d[..., None, None] * torch.eye(group, device=W.device)
    wmax = Wg.amax(-1, keepdim=True)
    wmin = Wg.amin(-1, keepdim=True)
    best_err = torch.full_like(wmax, float("inf"))
    best_s = torch.zeros_like(wmax)
    best_z = torch.zeros_like(wmax)
    for p in torch.linspace(1.0, shrink_min, grid, device=W.device):
        mx, mn = wmax * p, wmin * p
        sc = ((mx - mn) / MAX_Q).clamp(min=1e-5).half().float()
        z = (-mn / sc).round().clamp(0, MAX_Q)
        q = (Wg / sc + z).round().clamp(0, MAX_Q)
        D = (q - z) * sc - Wg
        if Gb is None:
            err = D.pow(2).sum(-1, keepdim=True)
        else:
            err = torch.einsum("bogk,bgkj,bogj->bog", D, Gb, D)[..., None]
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, sc, best_s)
        best_z = torch.where(better, z, best_z)
    q = (Wg / best_s + best_z).round().clamp(0, MAX_Q)
    return best_s.squeeze(-1).half(), best_z.squeeze(-1).to(torch.int32), q.view(B, out, inn).to(torch.uint8)


FACTOR_FALLBACKS = {"fp64": 0, "damp": 0}


def _factor(Hd: torch.Tensor):
    """(U, ok) with U the upper Cholesky factor of Hd^-1 per batch element; ok marks the elements whose
    factorizations succeeded (cholesky_ex: no raise, so the good members of a batch are kept)."""
    Lc, info = torch.linalg.cholesky_ex(Hd)
    ok = info == 0
    if not ok.any():
        return None, ok
    Lc = Lc.masked_fill(~ok[:, None, None], 0.0)
    Lc.diagonal(dim1=1, dim2=2).masked_fill_(~ok[:, None], 1.0)
    U, info2 = torch.linalg.cholesky_ex(torch.cholesky_inverse(Lc), upper=True)
    ok = ok & (info2 == 0)
    return U, ok


def _hinv_upper(H: torch.Tensor, percdamp: float) -> torch.Tensor:
    """Upper Cholesky factor of (H + damp I)^-1, batched. fp32 batched first; a Gram accumulated in fp32
    over 1e5 tokens with massive-activation channels (Flint's last layers: o_proj input, leading minor
    of order 49 indefinite at 1 % damping; expert gate/up Grams past layer ~30) can fail, so the failing
    elements alone are redone one at a time in fp64 (32 MB each, not a 1 GB batch copy next to a
    24 GB build), then with escalating damping, rather than dying 13 minutes into a build."""
    B = H.shape[0]
    dmean = H.diagonal(dim1=1, dim2=2).mean(1)
    Hd = H.clone()
    Hd.diagonal(dim1=1, dim2=2).add_((percdamp * dmean)[:, None])
    U, ok = _factor(Hd)
    del Hd
    if ok.all():
        return U
    out = U if U is not None else torch.empty_like(H)
    ladder = [percdamp, percdamp * 10, percdamp * 100, 1.0]
    for b in (~ok).nonzero().flatten().tolist():
        FACTOR_FALLBACKS["fp64"] += 1
        for frac in ladder:
            Hb = H[b].to(torch.float64)
            Hb.diagonal().add_(float(frac * dmean[b]))
            Ub, okb = _factor(Hb[None])
            del Hb
            if okb.all():
                out[b] = Ub[0].float()
                if frac != percdamp:
                    FACTOR_FALLBACKS["damp"] += 1
                    print(f"    [gptq] Gram {tuple(H.shape[1:])} #{b} needed damping {frac:g} in fp64", flush=True)
                break
        else:
            raise RuntimeError(f"Hessian #{b} not factorizable even at damp 1.0 in fp64")
    return out


def gptq_quantize(W: torch.Tensor, G: torch.Tensor | None, group: int, grid: int, shrink_min: float,
                  act_order: bool = True, block: int = 128, percdamp: float = 0.01):
    """GPTQ (Frantar et al.) with error feedback on the Gram, batched over B independent matrices.

    W [B,out,in] fp32, G [B,in,in] (None -> identity, i.e. no feedback). Group scale/zero are the
    static per-(row, group) clip-searched params of the unpermuted weights (search_groups_gram), so
    act-order needs no g_idx — the AWQ layout stays contiguous. Returns (scales fp16, zeros int32,
    q uint8) like search_groups_gram."""
    B, out, inn = W.shape
    dev = W.device
    W = W.clone()
    if G is None:
        G = torch.eye(inn, device=dev).expand(B, inn, inn)
    H = G.clone()
    diag = H.diagonal(dim1=1, dim2=2)  # view
    dead = diag == 0
    diag[dead] = 1.0
    W = W.masked_fill(dead[:, None, :], 0.0)
    scales, zeros, _ = search_groups_gram(W, None if G is None else G, group, grid, shrink_min)
    scales_f, zeros_f = scales.float(), zeros.float()  # [B,out,ng]
    if act_order:
        perm = torch.argsort(diag, dim=1, descending=True)  # [B,in]
        W = W.gather(2, perm[:, None, :].expand(B, out, inn))
        H = H.gather(1, perm[:, :, None].expand(B, inn, inn)).gather(2, perm[:, None, :].expand(B, inn, inn))
        invperm = torch.argsort(perm, dim=1)
    else:
        perm = torch.arange(inn, device=dev)[None].expand(B, inn)
        invperm = None
    gidx = perm // group  # [B,in] static group of each processed column
    Hinv = _hinv_upper(H, percdamp)
    del H
    Qint = torch.zeros(B, out, inn, dtype=torch.uint8, device=dev)
    for i1 in range(0, inn, block):
        i2 = min(i1 + block, inn)
        n = i2 - i1
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros(B, out, n, device=dev)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        for i in range(n):
            w = W1[:, :, i]  # [B,out]
            d = Hinv1[:, i, i]  # [B]
            g = gidx[:, i1 + i]  # [B]
            sc = scales_f.gather(2, g[:, None, None].expand(B, out, 1)).squeeze(2)
            z = zeros_f.gather(2, g[:, None, None].expand(B, out, 1)).squeeze(2)
            qi = (w / sc + z).round().clamp(0, MAX_Q)
            q = (qi - z) * sc
            Qint[:, :, i1 + i] = qi.to(torch.uint8)
            err = (w - q) / d[:, None]
            W1[:, :, i:] -= err[:, :, None] * Hinv1[:, i, i:][:, None, :]
            Err1[:, :, i] = err
        if i2 < inn:
            W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])
    if invperm is not None:
        Qint = Qint.gather(2, invperm[:, None, :].expand(B, out, inn))
    return scales, zeros, Qint


def quantize_batch(W: torch.Tensor, G: torch.Tensor | None, args, chunk: int = 32, percdamp: float | None = None):
    """W [B,out,in] (+ G [B,in,in] or None) -> list of B packed AWQ dicts (CPU tensors)."""
    pd = args.percdamp if percdamp is None else percdamp
    B, out, inn = W.shape
    assert inn % args.group_size == 0 and out % PACK == 0, f"bad shape {tuple(W.shape)} @g{args.group_size}"
    outs = []
    for c0 in range(0, B, chunk):
        Wc = W[c0:c0 + chunk]
        Gc = None if G is None else G[c0:c0 + chunk]
        if args.gptq:
            s, z, q = gptq_quantize(Wc, Gc, args.group_size, args.grid, args.shrink_min, act_order=not args.no_act_order,
                                    percdamp=pd)
        else:
            s, z, q = search_groups_gram(Wc, Gc, args.group_size, args.grid, args.shrink_min)
        for b in range(Wc.shape[0]):
            outs.append({k: v.cpu().contiguous() for k, v in pack_awq(s[b], z[b], q[b]).items()})
    return outs


def quantize(W: torch.Tensor, G: torch.Tensor | None, args):
    return quantize_batch(W[None], None if G is None else G[None], args)[0]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    md = os.environ.get("MODELS_DIR", "/data/models")
    ap.add_argument("--src", default=os.path.join(md, "hf-3rdparty/flint-bf16"))
    ap.add_argument("--dst", default=os.path.join(md, "hf-mattbucci/Flint-AWQ-INT4-g64-awq"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--shrink-min", type=float, default=0.5)
    ap.add_argument("--scale-grid", type=int, default=11, help="alpha candidates in [0,1] for the AWQ scale")
    ap.add_argument("--keep-bf16", default="", help="regex of quantizable tensor names to leave BF16 (still folded)")
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-cache", default="", help="json of the tokenised calibration set (built if missing)")
    ap.add_argument("--min-expert-tokens", type=int, default=32)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log", default="", help="per-layer calibration log json (default <dst>/calibration-log.json)")
    ap.add_argument("--gptq", action="store_true", help="GPTQ error feedback on the Grams instead of per-group clip only")
    ap.add_argument("--no-act-order", action="store_true", help="GPTQ: process columns in natural order")
    ap.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damping (fraction of mean diag H) for the always-active path")
    ap.add_argument("--expert-percdamp", type=float, default=0.1, help="GPTQ damping for routed-expert gate/up (thin Grams overfit at 0.01)")
    ap.add_argument("--expert-down-percdamp", type=float, default=1.0, help="GPTQ damping for routed-expert down_proj (worse than clip below ~0.3)")
    ap.add_argument("--expert-gram-shrink", type=float, default=1.0,
                    help="add this multiple of the (trace-matched) layer-wide input Gram to each expert's gate/up Gram")
    ap.add_argument("--calib-only", action="store_true", help="build + cache the calibration set, then exit")
    ap.add_argument("--layers", type=int, default=0, help="DEBUG: stop after this many layers (partial checkpoint)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    group = args.group_size
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---- calibration tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    if args.calib_cache and os.path.exists(args.calib_cache):
        calib = json.load(open(args.calib_cache))
        print(f"calibration set from {args.calib_cache}: {len(calib)} samples", flush=True)
    else:
        print("building calibration set", flush=True)
        calib = build_calib_ids(tok, args.src, args.n_samples, args.seq_len, args.seed, repo_dir)
        if args.calib_cache:
            os.makedirs(os.path.dirname(args.calib_cache) or ".", exist_ok=True)
            json.dump(calib, open(args.calib_cache, "w"))
    lens = [len(c["ids"]) for c in calib]
    n_tok = sum(lens)
    by_src = {s: sum(1 for c in calib if c["source"] == s) for s in SOURCES}
    print(f"{len(calib)} samples / {n_tok} tokens  {by_src}", flush=True)
    if args.calib_only:
        return

    # ---- plan
    keep_re = re.compile(args.keep_bf16) if args.keep_bf16 else None
    names = read_headers(args.src)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    H, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, L = cfg["rms_norm_eps"], cfg["num_hidden_layers"]
    nope, dense = set(cfg["no_rope_layer_indices"]), set(cfg["mlp_only_layers"])
    win, E, topk = cfg["sliding_window"] - 1, cfg["num_experts"] if "num_experts" in cfg else None, cfg["num_experts_per_tok"]
    if E is None:
        E = 1 + max(int(m.group(1)) for n in names for m in [re.search(r"mlp\.experts\.(\d+)\.", n)] if m)
    quant_names = [n for n in names if QUANT_RE.match(n)]
    is_quant = {n: not (keep_re and keep_re.search(n)) for n in quant_names}
    # modules_to_not_convert (same rule as the RTN script: name the parent when every child is kept)
    by_parent: dict[str, set[str]] = {}
    for n in quant_names:
        mod = LAYER_RE.sub("", n[: -len(".weight")])
        by_parent.setdefault(mod.rsplit(".", 1)[0], set()).add(mod)
    kept_mods = {LAYER_RE.sub("", n[: -len(".weight")]) for n in quant_names if not is_quant[n]}
    not_convert: set[str] = set()
    for parent, children in by_parent.items():
        if children <= kept_mods:
            not_convert.add(parent)
        else:
            not_convert |= children & kept_mods
    print(f"quantize {sum(is_quant.values())} / keep BF16 {len(quant_names) - sum(is_quant.values())} quantizable tensors; "
          f"modules_to_not_convert += {sorted(not_convert)}", flush=True)

    # ---- output dir
    os.makedirs(args.dst, exist_ok=True)
    stale = glob.glob(os.path.join(args.dst, "model*.safetensors*"))
    if stale and not args.overwrite:
        raise SystemExit(f"{args.dst} already holds {len(stale)} model files; pass --overwrite")
    for fn in stale:
        os.remove(fn)
    for fn in glob.glob(os.path.join(args.src, "*")):
        b = os.path.basename(fn)
        if os.path.isfile(fn) and not b.endswith((".safetensors", ".safetensors.index.json")) and b != "config.json":
            shutil.copy2(fn, os.path.join(args.dst, b))
    ocfg = dict(cfg)
    ocfg.pop("quantization", None)
    ocfg.pop("mlx_export", None)
    ocfg["quantization_config"] = {
        "quant_method": "awq", "bits": W_BIT, "group_size": group, "zero_point": True, "version": "gemm",
        "modules_to_not_convert": ["lm_head", "embed_tokens", *sorted(not_convert)],
        "producer": {
            "script": "scripts/quantize/quantize_flint_awq_calib.py",
            "method": "activation-aware AWQ: per-input-channel scale search (x_mean^a / w_mean^(1-a), folded into "
                      "norms / router / up_proj rows) + " + ("GPTQ error feedback on the Grams"
                      + (" (natural order)" if args.no_act_order else " (act-order, static groups)") if args.gptq
                      else "Gram-weighted per-group clipping search") + ", fp activations propagated",
            "grid": args.grid, "shrink_min": args.shrink_min, "scale_grid": args.scale_grid, "gptq": args.gptq,
            "percdamp": args.percdamp, "expert_percdamp": args.expert_percdamp,
            "expert_down_percdamp": args.expert_down_percdamp, "expert_gram_shrink": args.expert_gram_shrink,
            "calibration": {"n_samples": len(calib), "n_tokens": n_tok, "seq_len": args.seq_len, "seed": args.seed,
                            "sources": by_src},
            "kept_bf16": sorted({re.sub(r"\.\d+\.", ".N.", n) for n in names if n not in is_quant or not is_quant[n]}),
        },
    }
    json.dump(ocfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

    handles: dict[str, object] = {}

    def raw(name):
        shard = names[name][2]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.src, shard), framework="pt")
        return handles[shard].get_tensor(name)

    def getf(name):
        return raw(name).to(dev, torch.float32)

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    out_size = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}
    shard_bytes = int(args.shard_gb * 2**30)

    def emit(name, t):
        nonlocal out_size
        t = t.cpu().contiguous()
        out[name] = t
        out_size += t.numel() * t.element_size()
        if out_size >= shard_bytes:
            flush()

    def flush():
        nonlocal out, out_size, shard_idx
        if not out:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}.safetensors"
        save_file(out, os.path.join(args.dst, fn), metadata={"format": "pt"})
        os.chmod(os.path.join(args.dst, fn), 0o644)  # docker-served stack reads /models as UID 10001
        for k in out:
            weight_map[k] = fn
        print(f"  wrote {fn} ({out_size/2**30:.2f} GiB, {len(out)} tensors)", flush=True)
        out, out_size = OrderedDict(), 0

    def emit_packed(name, packed):
        for suffix, t in packed.items():
            emit(f"{name[:-len('.weight')]}.{suffix}", t)

    def emit_linear(name, W, G):
        """W already folded. Quantize+pack or keep BF16 per plan."""
        if is_quant[name]:
            emit_packed(name, quantize(W, G, args))
        else:
            emit(name, W.to(torch.bfloat16))

    # ---- activations
    X = torch.cat([torch.tensor(c["ids"], device=dev) for c in calib])
    bounds = []
    o = 0
    for n in lens:
        bounds.append((o, o + n))
        o += n
    emb = getf("model.embed_tokens.weight")
    X = emb[X]  # [Ntok, H] fp32
    emit("model.embed_tokens.weight", raw("model.embed_tokens.weight"))
    del emb
    Tmax = max(lens)
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    fr = torch.arange(Tmax, device=dev, dtype=torch.float32)[:, None] * inv[None]
    embt = torch.cat([fr, fr], -1)
    cos_t, sin_t = embt.cos()[:, None, :], embt.sin()[:, None, :]
    pos = torch.arange(Tmax, device=dev)
    causal = pos[None, :] <= pos[:, None]
    swa_mask = causal & ((pos[:, None] - pos[None, :]) <= win)
    scale = hd ** -0.5
    rep = nh // nkv

    def rope(v, T):
        v1, v2 = v[..., : hd // 2], v[..., hd // 2:]
        return v * cos_t[:T] + torch.cat([-v2, v1], -1) * sin_t[:T]

    def gram(x):
        return x.T @ x

    log = {"layers": [], "calibration": ocfg["quantization_config"]["producer"]["calibration"]}
    t0 = time.time()
    for l in range(args.layers or L):
        tl = time.time()
        p = f"model.layers.{l}."
        rec = {"layer": l}
        sliding = cfg["layer_types"][l] == "sliding_attention"
        ln1 = getf(p + "input_layernorm.weight")
        ln2 = getf(p + "post_attention_layernorm.weight")
        Wq, Wk, Wv, Wo = (getf(p + f"self_attn.{m}_proj.weight") for m in "qkvo")
        # ---------------- attention: qkv scale on the normed input
        h = rmsnorm(X, ln1, eps)
        G1 = gram(h)
        cons = [(W, G1) for m, W in zip("qkv", (Wq, Wk, Wv)) if is_quant[p + f"self_attn.{m}_proj.weight"]]
        s_qkv, a_qkv, r_qkv = search_scale(h.abs().mean(0), cons, group, args.scale_grid) if cons \
            else (torch.ones(H, device=dev), None, 1.0)
        rec["qkv"] = {"alpha": a_qkv, "loss_ratio": r_qkv}
        # folded producer / consumers
        emit(p + "input_layernorm.weight", (ln1 / s_qkv).to(torch.bfloat16))
        Gs = G1 / s_qkv[:, None] / s_qkv[None, :]  # Gram of h/s
        for m, W in zip("qkv", (Wq, Wk, Wv)):
            emit_linear(p + f"self_attn.{m}_proj.weight", W * s_qkv, Gs)
        # fp propagation of attention (original weights)
        q = (h @ Wq.T).view(-1, nh, hd)
        k = (h @ Wk.T).view(-1, nkv, hd)
        v = (h @ Wv.T).view(-1, nkv, hd)
        qn, kn = getf(p + "self_attn.q_norm.weight"), getf(p + "self_attn.k_norm.weight")
        q, k = rmsnorm(q, qn, eps), rmsnorm(k, kn, eps)
        sink = getf(p + "self_attn.softmax_offset")
        O = torch.empty(X.shape[0], nh * hd, device=dev)
        for (a, b) in bounds:
            T = b - a
            qq, kk, vv = q[a:b], k[a:b], v[a:b]
            if l not in nope:
                qq, kk = rope(qq, T), rope(kk, T)
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
            sc = torch.einsum("thd,shd->hts", qq, kk) * scale
            sc = sc.masked_fill(~(swa_mask if sliding else causal)[:T, :T][None], float("-inf"))
            sc = torch.cat([sc, sink[:, None, None].expand(nh, T, 1)], -1)
            pr = torch.softmax(sc, -1)[..., :T]
            O[a:b] = torch.einsum("hts,shd->thd", pr, vv).reshape(T, nh * hd)
        del q, k, v, sc, pr
        for nm in ("q_norm", "k_norm"):
            emit(p + f"self_attn.{nm}.weight", raw(p + f"self_attn.{nm}.weight"))
        emit(p + "self_attn.softmax_offset", raw(p + "self_attn.softmax_offset"))
        emit_linear(p + "self_attn.o_proj.weight", Wo, gram(O))  # clip only (GQA)
        X = X + O @ Wo.T
        del O, G1, Gs, Wq, Wk, Wv, Wo, h
        # ---------------- MLP
        h = rmsnorm(X, ln2, eps)
        G2 = gram(h)
        x2_mean = h.abs().mean(0)
        if l in dense:
            Wg, Wu, Wd = (getf(p + f"mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
            cons = [(W, G2) for m, W in (("gate", Wg), ("up", Wu)) if is_quant[p + f"mlp.{m}_proj.weight"]]
            s_gu, a_gu, r_gu = search_scale(x2_mean, cons, group, args.scale_grid) if cons \
                else (torch.ones(H, device=dev), None, 1.0)
            rec["mlp_gate_up"] = {"alpha": a_gu, "loss_ratio": r_gu}
            emit(p + "post_attention_layernorm.weight", (ln2 / s_gu).to(torch.bfloat16))
            # mid in chunks (6144-wide), Gram + mean
            I = Wd.shape[1]
            Gm = torch.zeros(I, I, device=dev)
            m_mean = torch.zeros(I, device=dev)
            M = torch.empty(X.shape[0], I, device=dev, dtype=torch.float16)
            for c0 in range(0, X.shape[0], 16384):
                mid = torch.nn.functional.silu(h[c0:c0 + 16384] @ Wg.T) * (h[c0:c0 + 16384] @ Wu.T)
                Gm += gram(mid)
                m_mean += mid.abs().sum(0)
                M[c0:c0 + 16384] = mid.half()
            m_mean /= X.shape[0]
            s_d, a_d, r_d = search_scale(m_mean, [(Wd, Gm)], group, args.scale_grid) if is_quant[p + "mlp.down_proj.weight"] \
                else (torch.ones(I, device=dev), None, 1.0)
            rec["mlp_down"] = {"alpha": a_d, "loss_ratio": r_d}
            Gs2 = G2 / s_gu[:, None] / s_gu[None, :]
            emit_linear(p + "mlp.gate_proj.weight", Wg * s_gu, Gs2)
            emit_linear(p + "mlp.up_proj.weight", (Wu * s_gu) / s_d[:, None], Gs2)
            emit_linear(p + "mlp.down_proj.weight", Wd * s_d, Gm / s_d[:, None] / s_d[None, :])
            for c0 in range(0, X.shape[0], 16384):
                X[c0:c0 + 16384] += M[c0:c0 + 16384].float() @ Wd.T
            del M, Gm, Wg, Wu, Wd, Gs2
        else:
            Wr = getf(p + "mlp.gate.weight")  # router [E, H], BF16 in the output
            logits = h @ Wr.T
            scr = torch.sigmoid(logits)
            top_i = torch.topk(scr, topk, dim=-1, sorted=False).indices
            top_w = scr.gather(1, top_i)
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
            del logits, scr
            Wsg, Wsu, Wsd = (getf(p + f"mlp.shared_expert.{m}_proj.weight") for m in ("gate", "up", "down"))
            # expert inputs: rows + routing weights; routing-weighted Grams (Ge_all[e] = sum w^2 x x^T)
            I_e = cfg["moe_intermediate_size"]
            Wg_all = torch.empty(E, I_e, H, device=dev)
            Wu_all = torch.empty(E, I_e, H, device=dev)
            Wd_all = torch.empty(E, H, I_e, device=dev)
            Gd_all = torch.zeros(E, I_e, I_e, device=dev)
            Ge_all = torch.zeros(E, H, H, device=dev)
            has_d = torch.zeros(E, dtype=torch.bool)
            ex_rows, ex_w = {}, {}
            consumers = [(W, G2) for m, W in (("gate", Wsg), ("up", Wsu)) if is_quant[p + f"mlp.shared_expert.{m}_proj.weight"]]
            n_small = 0
            for e in range(E):
                pe = p + f"mlp.experts.{e}."
                Wg_all[e], Wu_all[e], Wd_all[e] = (getf(pe + f"{m}_proj.weight") for m in ("gate", "up", "down"))
                rows, slot = (top_i == e).nonzero(as_tuple=True)
                if rows.numel() == 0:
                    continue
                w = top_w[rows, slot]
                ex_rows[e], ex_w[e] = rows, w
                xe = h[rows] * w[:, None]
                Ge_all[e] = gram(xe)
                if rows.numel() >= args.min_expert_tokens:
                    consumers += [(Wall[e], Ge_all[e]) for m, Wall in (("gate", Wg_all), ("up", Wu_all))
                                  if is_quant[pe + f"{m}_proj.weight"]]
                else:
                    n_small += 1
            # joint scale for everything fed by post_attention_layernorm
            s_gu, a_gu, r_gu = search_scale(x2_mean, consumers, group, args.scale_grid) if consumers \
                else (torch.ones(H, device=dev), None, 1.0)
            rec["moe_gate_up"] = {"alpha": a_gu, "loss_ratio": r_gu, "experts_hit": len(ex_rows), "experts_small": n_small}
            emit(p + "post_attention_layernorm.weight", (ln2 / s_gu).to(torch.bfloat16))
            emit(p + "mlp.gate.weight", (Wr * s_gu).to(torch.bfloat16))
            Gs2 = G2 / s_gu[:, None] / s_gu[None, :]
            # shared expert
            smid = torch.nn.functional.silu(h @ Wsg.T) * (h @ Wsu.T)
            Gsm = gram(smid)
            s_sd, a_sd, r_sd = search_scale(smid.abs().mean(0), [(Wsd, Gsm)], group, args.scale_grid) \
                if is_quant[p + "mlp.shared_expert.down_proj.weight"] else (torch.ones(Wsd.shape[1], device=dev), None, 1.0)
            rec["shared_down"] = {"alpha": a_sd, "loss_ratio": r_sd}
            emit_linear(p + "mlp.shared_expert.gate_proj.weight", Wsg * s_gu, Gs2)
            emit_linear(p + "mlp.shared_expert.up_proj.weight", (Wsu * s_gu) / s_sd[:, None], Gs2)
            emit_linear(p + "mlp.shared_expert.down_proj.weight", Wsd * s_sd, Gsm / s_sd[:, None] / s_sd[None, :])
            Mo = smid @ Wsd.T
            del smid, Gsm
            # routed experts: pass 1 = forward contribution (original weights) + down-proj Grams/scales + folds,
            # pass 2 = batched quantization per family
            alphas_d = []
            inv_s2 = 1.0 / (s_gu[:, None] * s_gu[None, :])
            for e in range(E):
                pe = p + f"mlp.experts.{e}."
                s_d = torch.ones(I_e, device=dev)
                if e in ex_rows:
                    rows, w = ex_rows[e], ex_w[e]
                    he = h[rows]
                    mid = torch.nn.functional.silu(he @ Wg_all[e].T) * (he @ Wu_all[e].T)
                    Mo.index_add_(0, rows, (mid @ Wd_all[e].T) * w[:, None])
                    if rows.numel() >= args.min_expert_tokens:
                        midw = mid * w[:, None]
                        Gd_all[e] = gram(midw)
                        has_d[e] = True
                        Ge_all[e] *= inv_s2
                        if args.expert_gram_shrink > 0:  # thin per-expert Gram -> shrink toward the layer-wide one
                            tr = Ge_all[e].diagonal().sum() / Gs2.diagonal().sum()
                            Ge_all[e] += (args.expert_gram_shrink * tr) * Gs2
                        if is_quant[pe + "down_proj.weight"]:
                            s_d, a_d, _ = search_scale(midw.abs().mean(0), [(Wd_all[e], Gd_all[e])], group, args.scale_grid)
                            alphas_d.append(a_d if a_d is not None else -1)
                            Gd_all[e] *= 1.0 / (s_d[:, None] * s_d[None, :])
                        del midw
                    del mid, he
                if not has_d[e]:  # too few / no routed tokens: the layer-wide input Gram stands in for gate/up
                    Ge_all[e] = Gs2
                Wg_all[e] *= s_gu
                Wu_all[e] *= s_gu / s_d[:, None]
                Wd_all[e] *= s_d
            for fam, Wall, Gall, mask in (("gate", Wg_all, Ge_all, None), ("up", Wu_all, Ge_all, None),
                                          ("down", Wd_all, Gd_all, has_d)):
                qidx = [e for e in range(E) if is_quant[p + f"mlp.experts.{e}.{fam}_proj.weight"]]
                for e in range(E):
                    if e not in qidx:
                        emit(p + f"mlp.experts.{e}.{fam}_proj.weight", Wall[e].to(torch.bfloat16))
                if not qidx:
                    continue
                if mask is None:
                    packed = quantize_batch(Wall[qidx], Gall[qidx], args, percdamp=args.expert_percdamp)
                    for e, pk in zip(qidx, packed):
                        emit_packed(p + f"mlp.experts.{e}.{fam}_proj.weight", pk)
                else:  # down: experts with a mid Gram batched together, the rest data-free
                    with_g = [e for e in qidx if mask[e]]
                    without = [e for e in qidx if not mask[e]]
                    if with_g:
                        for e, pk in zip(with_g, quantize_batch(Wall[with_g], Gall[with_g], args, percdamp=args.expert_down_percdamp)):
                            emit_packed(p + f"mlp.experts.{e}.{fam}_proj.weight", pk)
                    if without:
                        for e, pk in zip(without, quantize_batch(Wall[without], None, args)):
                            emit_packed(p + f"mlp.experts.{e}.{fam}_proj.weight", pk)
            del Wg_all, Wu_all, Wd_all, Gd_all, Ge_all
            rec["expert_down_alpha_hist"] = {str(a): alphas_d.count(a) for a in sorted(set(alphas_d))}
            X = X + Mo
            del Mo, ex_rows, ex_w, Wsg, Wsu, Wsd, Wr, top_i, top_w, Gs2
        del h, G2
        rec["secs"] = round(time.time() - tl, 1)
        log["layers"].append(rec)
        print(f"  layer {l + 1}/{L}  qkv a={rec['qkv']['alpha']} r={rec['qkv']['loss_ratio']:.3f}  "
              f"mlp a={(rec.get('moe_gate_up') or rec.get('mlp_gate_up'))['alpha']} "
              f"r={(rec.get('moe_gate_up') or rec.get('mlp_gate_up'))['loss_ratio']:.3f}  "
              f"{rec['secs']}s  total {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    emit("model.norm.weight", raw("model.norm.weight"))
    emit("lm_head.weight", raw("lm_head.weight"))
    flush()
    if args.layers:
        print(f"DEBUG partial checkpoint ({args.layers} layers) written; skipping completeness check", flush=True)
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map},
                  open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=2)
        json.dump(log, open(args.log or os.path.join(args.dst, "calibration-log.json"), "w"), indent=1)
        return
    total = sum(os.path.getsize(os.path.join(args.dst, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=2)
    log["secs"] = round(time.time() - t0, 1)
    log["factor_fallbacks"] = dict(FACTOR_FALLBACKS)
    json.dump(log, open(args.log or os.path.join(args.dst, "calibration-log.json"), "w"), indent=1)
    # completeness check: every source tensor is represented (as itself or as its 3 AWQ tensors)
    missing = [n for n in names if n not in weight_map and f"{n[:-len('.weight')]}.qweight" not in weight_map]
    print(f"done: {len(weight_map)} tensors in {log['secs']}s -> {args.dst} ({total/2**30:.1f} GiB); missing={missing[:5]}",
          flush=True)
    if missing:
        raise SystemExit(f"{len(missing)} source tensors not written")


if __name__ == "__main__":
    main()
