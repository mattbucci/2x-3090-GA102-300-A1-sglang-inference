# Parity: Ampere rel-bias attention vs a torch fp32 reference of the transformers eager math (bias, causal, window).
import torch, sys
from sglang.kernels.ops.attention.inkling2_rel_bias_ampere import rel_bias_attention
torch.manual_seed(0); dev = "cuda"
H, HKV, D, E = 12, 4, 128, 1024
def ref(q, k, v, bias, pos, window, scale):
    # q [n,H,D] at absolute positions pos[n]; k/v [L,HKV,D] positions 0..L-1
    L = k.shape[0]; G = H // HKV
    kk = k.float().repeat_interleave(G, 1); vv = v.float().repeat_interleave(G, 1)
    s = torch.einsum("nhd,lhd->hnl", q.float(), kk) * scale
    dist = pos[:, None] - torch.arange(L, device=dev)[None, :]
    vis = dist >= 0
    if window: vis &= dist < window
    b = torch.zeros_like(s)
    idx = dist.clamp(0, E - 1)
    b = bias.float().permute(1, 0, 2).gather(-1, idx[None].expand(H, -1, -1))
    b = torch.where((dist >= 0) & (dist < E), b, torch.zeros_like(b))
    s = torch.where(vis[None], s + b, torch.full_like(s, float("-inf")))
    lse = torch.logsumexp(s, -1)
    p = torch.softmax(s, -1)
    o = torch.einsum("hnl,lhd->nhd", p, vv)
    return o, lse.T
worst = 0.0
for window in (0, 1024):
    for (prefix, new) in ((0, 300), (2000, 1), (2000, 7), (5000, 1), (1500, 200), (300, 1)):
        seqs = [(prefix, new), (prefix // 2 + 17, max(1, new // 2))]
        slots = sum(p + n for p, n in seqs) + 64
        kb = torch.randn(slots, HKV, D, device=dev, dtype=torch.bfloat16); vb = torch.randn_like(kb)
        perm = torch.randperm(slots, device=dev).to(torch.int32)
        qs, biases, qst, ql, kvs, kvl, idx, off_k, off_q, qlc = [], [], [], [], [], [], [], 0, 0, []
        for p, n in seqs:
            L = p + n
            qs.append(torch.randn(n, H, D, device=dev, dtype=torch.bfloat16))
            biases.append(torch.randn(n, H, E, device=dev, dtype=torch.bfloat16))
            qst.append(off_q); ql.append(n); kvs.append(off_k); kvl.append(L); qlc.append(n)
            idx.append(perm[off_k:off_k + L]); off_k += L; off_q += n
        q = torch.cat(qs); bias = torch.cat(biases); kvi = torch.cat(idx)
        t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
        o, lse = rel_bias_attention(q, kb, vb, bias, kvi, t(qst), t(ql), t(kvs), t(kvl), qlc, scale=D ** -0.5, window=window)
        for i, (p, n) in enumerate(seqs):
            L = p + n; sl = slice(qst[i], qst[i] + n)
            kk = kb[idx[i].long()]; vv = vb[idx[i].long()]
            ro, rl = ref(q[sl], kk, vv, bias[sl], torch.arange(p, L, device=dev), window, D ** -0.5)
            e_o = ((o[sl].float() - ro).abs().max() / ro.abs().max()).item()
            e_l = (lse[sl] - rl).abs().max().item()
            worst = max(worst, e_o)
            print(f"window={window:4} prefix={p:5} new={n:3}  o err/max {e_o:.2e}  lse abs err {e_l:.2e}")
print("WORST", worst, "PASS" if worst < 2e-2 else "FAIL")
