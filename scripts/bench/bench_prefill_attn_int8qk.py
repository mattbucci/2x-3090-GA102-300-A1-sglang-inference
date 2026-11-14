#!/usr/bin/env python3
"""Cold-prefill attention with INT8 QK^T (SageAttention v1 style) against exact FP16: kernel time and output error.

Flint's deep cold prefill is ~73 % attention and the flashinfer kernel already runs at the FP16 tensor peak
(docs/roofline.md), so the remaining arithmetic lever is the INT8 tensor rate for the QK^T half; P.V stays bf16.
One chunk of query rows on top of a cached prefix, one layer, one rank's heads. Three kernel arms:

  flashinfer   the served kernel over the fp8_e5m2 pool (absolute reference time)
  triton-fp16  this file's one-pass kernel with bf16 QK^T (the structural twin: isolates the INT8 effect)
  triton-int8  the same kernel with int8 Q / K tiles, per-block scales, optional K mean-smoothing

The twin reads contiguous K/V (no slot table) and precomputed int8 K, so its time is an upper bound for a
served INT8 route. Error is measured on sampled rows against an fp32 torch reference of the pool's values.

  python scripts/bench/bench_prefill_attn_int8qk.py --prefixes 131072,245760 --rows 8192
  python scripts/bench/bench_prefill_attn_int8qk.py --dump /var/tmp/flint-int8qk/dump-l12.pt --out r.json
"""
import argparse, json, os, sys
import torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from test_extend_splitkv_route import time_us  # noqa: E402


@triton.jit
def _prefill_kernel(Q, K, V, QS, KS, O, LSE, L, N, P, sm_scale,
                    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, INT8: tl.constexpr):
    pid_m = tl.program_id(0); h = tl.program_id(1); hk = h // (H_Q // H_KV)
    offs_m = pid_m * BM + tl.arange(0, BM); offs_d = tl.arange(0, D); offs_n = tl.arange(0, BN)
    mask_m = offs_m < L
    q = tl.load(Q + offs_m[:, None] * (H_Q * D) + h * D + offs_d[None, :], mask=mask_m[:, None], other=0)
    if INT8:
        qs = tl.load(QS + offs_m * H_Q + h, mask=mask_m, other=0.0)  # per-row scale, sm_scale folded in
    m_i = tl.full([BM], -1.0e30, dtype=tl.float32); l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    hi = tl.minimum(N, P + (pid_m + 1) * BM)
    for start_n in range(0, hi, BN):
        cols = start_n + offs_n
        k = tl.load(K + cols[None, :] * (H_KV * D) + hk * D + offs_d[:, None], mask=cols[None, :] < N, other=0)
        if INT8:
            ks = tl.load(KS + cols * H_KV + hk, mask=cols < N, other=0.0)  # per-key scale
            qk = tl.dot(q, k).to(tl.float32) * qs[:, None] * ks[None, :]
        else:
            qk = tl.dot(q, k).to(tl.float32) * sm_scale
        valid = (cols[None, :] <= (P + offs_m)[:, None]) & (cols[None, :] < N)
        qk = tl.where(valid, qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + cols[:, None] * (H_KV * D) + hk * D + offs_d[None, :], mask=cols[:, None] < N, other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * (H_Q * D) + h * D + offs_d[None, :], acc.to(O.dtype.element_ty), mask=mask_m[:, None])
    tl.store(LSE + offs_m * H_Q + h, m_i + tl.log(l_i), mask=mask_m)


def quant_tiles(x, block, extra_scale=1.0):
    """x [n, H, D] float -> int8 [n, H, D] and a per-(token, head) scale [n, H] (x ~= int8 * scale).
    block > 1 shares one scale per (tile of `block` tokens, head); block = 1 is per-token."""
    n, H, D = x.shape; nt = (n + block - 1) // block
    pad = torch.zeros(nt * block, H, D, device=x.device, dtype=torch.float32); pad[:n] = x
    s = pad.view(nt, block, H, D).abs().amax(dim=(1, 3)).clamp_min(1e-8) / 127.0
    xi = torch.round(pad.view(nt, block, H, D) / s[:, None, :, None]).clamp_(-127, 127).to(torch.int8).view(nt * block, H, D)[:n]
    return xi.contiguous(), (s * extra_scale).repeat_interleave(block, 0)[:n].contiguous()


def run_kernel(q, k, v, qs, ks, P, sm_scale, BM, BN, warps, int8, stages=1):
    L, Hq, D = q.shape; N, Hkv, _ = k.shape
    o = torch.empty(L, Hq, D, device=q.device, dtype=torch.bfloat16); lse = torch.empty(L, Hq, device=q.device, dtype=torch.float32)
    grid = ((L + BM - 1) // BM, Hq)
    def f():
        _prefill_kernel[grid](q, k, v, qs, ks, o, lse, L, N, P, sm_scale, H_Q=Hq, H_KV=Hkv, D=D, BM=BM, BN=BN, INT8=int8,
                              num_warps=warps, num_stages=stages)
        return o
    return f, o


def reference(qf, kf, vf, P, sm_scale, rows, logits_fn=None):
    """fp32 output (and attention rows) for sampled chunk rows. qf [L,Hq,D], kf/vf [N,Hkv,D] float32."""
    Hq, Hkv = qf.shape[1], kf.shape[1]; G = Hq // Hkv; N = kf.shape[0]; outs, ps = [], []
    pos = torch.arange(N, device=qf.device)
    for r in rows.tolist():
        o_h, p_h = [], []
        for hk in range(Hkv):
            if logits_fn is None:
                lg = qf[r, hk * G:(hk + 1) * G] @ kf[:, hk].T * sm_scale
            else:
                lg = logits_fn(r, hk)
            lg = lg.masked_fill(pos[None, :] > P + r, float("-inf"))
            p = torch.softmax(lg, dim=-1); o_h.append(p @ vf[:, hk]); p_h.append(p)
        outs.append(torch.cat(o_h)); ps.append(torch.cat(p_h))
    return torch.stack(outs), torch.stack(ps)


def err(o, ref):
    o, ref = o.float(), ref.float()
    return dict(rel_l2=round(((o - ref).norm() / ref.norm()).item(), 5),
                cosine=round(torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item(), 6),
                worst_row_rel_l2=round(((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item(), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="torch file: q [L,Hq,D] bf16, k/v [N,Hkv,D] pool dtype, k_scale, v_scale, sm_scale (N = prefix + L)")
    ap.add_argument("--prefixes", default="131072,245760"); ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--tiles", default="128x128x8,128x64x8,64x128x4", help="BMxBNxwarps[xstages] list for the triton arms")
    ap.add_argument("--scales", default="block,token", help="int8 scale granularity: block (one per tile) and/or token")
    ap.add_argument("--sample-rows", type=int, default=48); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import flashinfer
    dev = torch.device("cuda"); torch.manual_seed(0); out_rows = []
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    cases = []
    if a.dump:
        d = torch.load(a.dump, map_location=dev)
        cases.append((d["q"], d["k"], d["v"], float(d["k_scale"]), float(d["v_scale"]), float(d["sm_scale"]), d["k"].shape[0] - d["q"].shape[0], "dump"))
    else:
        for P in [int(x) for x in a.prefixes.split(",")]:
            n = P + a.rows
            cases.append((torch.randn(a.rows, 16, 128, device=dev, dtype=torch.bfloat16), (torch.randn(n, 2, 128, device=dev) * 6.0).to(torch.float8_e5m2),
                          (torch.randn(n, 2, 128, device=dev) * 6.0).to(torch.float8_e5m2), 0.07, 0.05, 128 ** -0.5, P, "randn"))
    for q, k_pool, v_pool, k_scale, v_scale, sm_scale, P, src in cases:
        L, Hq, D = q.shape; N, Hkv, _ = k_pool.shape
        kf = k_pool.float() * k_scale; vf = v_pool.float() * v_scale; qf = q.float()
        rows = torch.linspace(0, L - 1, a.sample_rows, device=dev).long()
        ref, p_ref = reference(qf, kf, vf, P, sm_scale, rows)
        base = dict(src=src, prefix=P, rows=L, h_q=Hq, h_kv=Hkv, d=D)

        # flashinfer over the pool (page size 1, identity slot table)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        qo = torch.tensor([0, L], device=dev, dtype=torch.int32)
        w.plan(qo, torch.tensor([0, N], device=dev, dtype=torch.int32), torch.arange(N, device=dev, dtype=torch.int32),
               torch.ones(1, device=dev, dtype=torch.int32), Hq, Hkv, D, 1, causal=True, sm_scale=sm_scale,
               q_data_type=torch.bfloat16, kv_data_type=k_pool.dtype)
        kw = dict(k_scale=k_scale, v_scale=v_scale) if k_pool.dtype == torch.float8_e5m2 else {}
        f_fi = lambda: w.run(q, (k_pool.unsqueeze(1), v_pool.unsqueeze(1)), **kw)
        o_fi = f_fi(); torch.cuda.synchronize(); ms_fi = time_us(f_fi, iters=2, repeats=3) / 1e3
        row = dict(base, arm="flashinfer", ms=round(ms_fi, 2), **err(o_fi[rows], ref)); out_rows.append(row); print(json.dumps(row), flush=True)

        k16 = kf.to(torch.bfloat16).contiguous(); v16 = vf.to(torch.bfloat16).contiguous(); dummy = torch.zeros(1, device=dev)
        for t in a.tiles.split(","):
            BM, BN, warps, stages = (tuple(int(x) for x in t.split("x")) + (1,))[:4]
            try:
                f, o = run_kernel(q.contiguous(), k16, v16, dummy, dummy, P, sm_scale, BM, BN, warps, False, stages)
                f(); torch.cuda.synchronize(); ms = time_us(f, iters=2, repeats=3) / 1e3
                row = dict(base, arm="triton-fp16", tile=t, ms=round(ms, 2), vs_flashinfer=round(ms / ms_fi, 3), **err(o[rows], ref))
            except Exception as e:  # noqa: BLE001
                row = dict(base, arm="triton-fp16", tile=t, error=str(e)[:200])
            out_rows.append(row); print(json.dumps(row), flush=True)
            for smooth, gran in [(s, g) for s in (True, False) for g in a.scales.split(",")]:
                try:
                    kq = kf - kf.mean(dim=0, keepdim=True) if smooth else kf  # softmax is invariant to a constant shift of K
                    qi, qs = quant_tiles(qf, BM if gran == "block" else 1, extra_scale=sm_scale); ki, ks = quant_tiles(kq, BN if gran == "block" else 1)
                    f, o = run_kernel(qi, ki, v16, qs, ks, P, sm_scale, BM, BN, warps, True, stages)
                    f(); torch.cuda.synchronize(); ms = time_us(f, iters=2, repeats=3) / 1e3
                    G = Hq // Hkv; qd = qi.float() * qs[:, :, None]; kd = ki.float() * ks[:, :, None]
                    _, p_q = reference(qf, kf, vf, P, sm_scale, rows, logits_fn=lambda r, hk: qd[r, hk * G:(hk + 1) * G] @ kd[:, hk].T)
                    kl = (p_ref * (torch.log(p_ref.clamp_min(1e-30)) - torch.log(p_q.clamp_min(1e-30)))).sum(-1)
                    row = dict(base, arm="triton-int8", tile=t, k_smooth=smooth, scale=gran, ms=round(ms, 2), vs_flashinfer=round(ms / ms_fi, 3), **err(o[rows], ref),
                               attn_kl_mean=round(kl.mean().item(), 5), attn_kl_max=round(kl.max().item(), 4),
                               attn_top1_agree=round((p_ref.argmax(-1) == p_q.argmax(-1)).float().mean().item(), 4))
                except Exception as e:  # noqa: BLE001
                    row = dict(base, arm="triton-int8", tile=t, k_smooth=smooth, scale=gran, error=str(e)[:200])
                out_rows.append(row); print(json.dumps(row), flush=True)
        del w; torch.cuda.empty_cache()
    if a.out:
        json.dump(dict(flashinfer=flashinfer.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(0), rows=out_rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
