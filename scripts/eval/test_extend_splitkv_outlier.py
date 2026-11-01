#!/usr/bin/env python
"""Re-run one split-KV route cell under several seeds and report the error distribution (max, rms, worst element,
and the extend kernel's own numbers), to tell a rounding-sensitive single element from a systematic defect.
    python scripts/eval/test_extend_splitkv_outlier.py --preset flint --kv e5m2 --prefix 58000 --K 16 --sinks 1 --seeds 0,1,2,3,4
"""
import argparse, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_extend_splitkv_route import PRESETS, fp32_reference

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preset", default="flint"); ap.add_argument("--kv", default="e5m2"); ap.add_argument("--prefix", type=int, default=58000)
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--sinks", type=int, default=1); ap.add_argument("--seeds", default="0,1,2,3,4"); a = ap.parse_args()
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
    p = PRESETS[a.preset]; dev = torch.device("cuda"); sm_scale = p["head_dim"] ** -0.5
    kv_dtype = {"bf16": torch.bfloat16, "e5m2": torch.float8_e5m2}[a.kv]; fp8 = kv_dtype != torch.bfloat16
    k_scale = 0.07 if fp8 else 1.0; v_scale = 0.05 if fp8 else 1.0; P, K = a.prefix, a.K
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); pool_n = P + 1024
        k_buf = (torch.randn(pool_n, p["h_kv"], p["head_dim"], device=dev) * (6.0 if fp8 else 0.5)).to(kv_dtype)
        v_buf = (torch.randn(pool_n, p["h_kv"], p["v_head_dim"], device=dev) * (6.0 if fp8 else 0.5)).to(kv_dtype)
        kv_indices = torch.randperm(pool_n, device=dev, dtype=torch.int32)[:P].contiguous(); kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
        q = torch.randn(K, p["h_q"], p["head_dim"], device=dev, dtype=torch.bfloat16)
        k_ext = torch.randn(K, p["h_kv"], p["head_dim"], device=dev, dtype=torch.bfloat16) * 0.5; v_ext = torch.randn(K, p["h_kv"], p["v_head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
        sinks = (torch.randn(p["h_q"], device=dev) * 2).float() if a.sinks else None
        qo = torch.tensor([0, K], device=dev, dtype=torch.int32); o_ref = torch.empty(K, p["h_q"], p["v_head_dim"], device=dev, dtype=torch.bfloat16)
        extend_attention_fwd(q, k_ext, v_ext, o_ref, k_buf, v_buf, qo, kv_indptr, kv_indices, None, True, None, K, k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks)
        n_pad = max(16, 1 << (K - 1).bit_length()); pad = n_pad - K
        q3 = torch.cat([q, q.new_zeros((pad,) + q.shape[1:])]); k3 = torch.cat([k_ext, k_ext.new_zeros((pad,) + k_ext.shape[1:])]); v3 = torch.cat([v_ext, v_ext.new_zeros((pad,) + v_ext.shape[1:])])
        o3 = q3.new_empty((n_pad, p["h_q"], p["v_head_dim"])); qo_pad = torch.tensor([0, n_pad], device=dev, dtype=torch.int32)
        verify_splitkv_fwd(q3, k3, v3, o3, k_buf, v_buf, qo_pad, kv_indptr, kv_indices, None, True, None, n_pad, k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks, max_bs=1)
        torch.cuda.synchronize()
        ref = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks); refmax = ref.abs().max().item()
        for name, o in (("extend", o_ref), ("splitkv", o3[:K])):
            d = (o.float() - ref).abs(); i = d.argmax().item(); idx = (i // (p["h_q"] * p["v_head_dim"]), (i // p["v_head_dim"]) % p["h_q"], i % p["v_head_dim"])
            print(f"seed {seed} {name:7} max {d.max().item()/refmax:.2e} rms {d.pow(2).mean().sqrt().item()/refmax:.2e} at (row {idx[0]}, head {idx[1]}, d {idx[2]}) ref {ref.flatten()[i].item():+.4f} got {o.float().flatten()[i].item():+.4f}" + (f" sink {sinks[idx[1]].item():+.2f}" if sinks is not None else ""), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
