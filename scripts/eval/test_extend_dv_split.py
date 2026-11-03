#!/usr/bin/env python3
"""V-split (SGLANG_EXTEND_ATTN_DV_SPLIT) exactness test for the triton extend kernel at head dim 512:
the split-2 output must equal the unsplit output bit-for-bit on a prefix + extend batch (GQA, causal,
sliding-window and full variants). Run on one GPU; exit 1 on mismatch.
"""
import os, sys, importlib, torch

def run(split, H=8, KVH=4, D=512, prefix=1500, extend=700, sliding=-1, seed=0):
    os.environ["SGLANG_EXTEND_ATTN_DV_SPLIT"] = str(split)
    import sglang.kernels.ops.attention.extend_attention as m
    importlib.reload(m)
    torch.manual_seed(seed); dev = "cuda"; dt = torch.bfloat16
    B = 2
    q = torch.randn(B * extend, H, D, device=dev, dtype=dt)
    ke = torch.randn(B * extend, KVH, D, device=dev, dtype=dt)
    ve = torch.randn(B * extend, KVH, D, device=dev, dtype=dt)
    o = torch.empty_like(q)
    total = B * (prefix + extend)
    kb = torch.randn(total, KVH, D, device=dev, dtype=dt)
    vb = torch.randn(total, KVH, D, device=dev, dtype=dt)
    qo_indptr = torch.tensor([0, extend, 2 * extend], device=dev, dtype=torch.int64)
    kv_indptr = torch.tensor([0, prefix, 2 * prefix], device=dev, dtype=torch.int64)
    kv_indices = torch.cat([torch.arange(0, prefix), torch.arange(prefix + extend, 2 * prefix + extend)]).to(dev, torch.int64)
    m.extend_attention_fwd(q, ke, ve, o, kb, vb, qo_indptr, kv_indptr, kv_indices, None, True, None, extend,
                           1.0, 1.0, sliding_window_size=sliding)
    torch.cuda.synchronize()
    return o.float().clone()

def main():
    ok = True
    for name, kw in (("full", {}), ("swa1024", {"sliding": 1024}), ("gqa1", {"H": 4, "KVH": 4})):
        a = run(1, **kw); b = run(2, **kw)
        d = (a - b).abs().max().item()
        print(f"{name}: max|split2 - split1| = {d:.3e}  (|o| max {a.abs().max().item():.3e})")
        ok &= d == 0.0
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
