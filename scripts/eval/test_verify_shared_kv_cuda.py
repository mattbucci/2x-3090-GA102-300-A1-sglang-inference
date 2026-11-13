#!/usr/bin/env python
"""Grouped-head target verify on CUDA (3090 patch 079, after R9700's 125): does `verify_shared_kv_fwd` match the per-head
`verify_splitkv_fwd` (patch 078's CUDA route) at Flint's per-rank shape, with learned sinks and an fp8_e5m2 cache, and how
much faster per layer? Topk=1 verify: K draft tokens over a P-token prefix, qo_indptr [0, K], causal draft block.
Gate against the fp32 reference: grouped max error <= 1.6x the per-head kernel's, cosine >= 0.9999, finite.
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_verify_shared_kv_cuda.py --out benchmarks/profiling/verify-shared-kv-cuda-<date>.json
"""
import argparse, json, os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_extend_splitkv_route import PRESETS, err_vs, fp32_reference, time_us  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--presets", default="flint,qwen38"); ap.add_argument("--prefixes", default="8192,131000,254000")
    ap.add_argument("--drafts", default="8,16"); ap.add_argument("--kv-dtypes", default="bf16,e5m2"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
    from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
    DT = {"bf16": torch.bfloat16, "e5m2": torch.float8_e5m2}
    dev = "cuda"; torch.manual_seed(0); results = []; fails = 0
    for pname in a.presets.split(","):
        p = PRESETS[pname]
        for kvn in a.kv_dtypes.split(","):
            for P in map(int, a.prefixes.split(",")):
                for K in map(int, a.drafts.split(",")):
                    for use_sinks in ([False, True] if p["sinks"] else [False]):
                        h_q, h_kv, d, dv = p["h_q"], p["h_kv"], p["head_dim"], p["v_head_dim"]
                        q = torch.randn(K, h_q, d, device=dev, dtype=torch.bfloat16)
                        k_ext = torch.randn(K, h_kv, d, device=dev, dtype=torch.bfloat16)
                        v_ext = torch.randn(K, h_kv, dv, device=dev, dtype=torch.bfloat16)
                        k_buf = torch.randn(P + 64, h_kv, d, device=dev, dtype=torch.bfloat16).to(DT[kvn])
                        v_buf = torch.randn(P + 64, h_kv, dv, device=dev, dtype=torch.bfloat16).to(DT[kvn])
                        kv_indices = torch.randperm(P + 64, device=dev)[:P].to(torch.int64)
                        kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
                        qo_indptr = torch.tensor([0, K], device=dev, dtype=torch.int32)
                        sinks = (torch.randn(h_q, device=dev) * 2).float() if use_sinks else None
                        sm_scale = d ** -0.5; k_scale = v_scale = 1.0
                        o_ref = torch.empty(K, h_q, dv, device=dev, dtype=torch.bfloat16); o_new = torch.empty_like(o_ref)
                        def run(fn, o):
                            return bool(fn(q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K,
                                           k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks, max_bs=1))
                        row = {"preset": pname, "kv": kvn, "prefix": P, "K": K, "sinks": use_sinks}
                        try:
                            ok_ref = run(verify_splitkv_fwd, o_ref); ok_new = run(verify_shared_kv_fwd, o_new)
                        except Exception as e:
                            row["status"] = f"error: {str(e).splitlines()[0][:160]}"; results.append(row); fails += 1
                            print(pname, kvn, P, K, use_sinks, row["status"], flush=True); continue
                        if not (ok_ref and ok_new):
                            row["status"] = f"can_handle per-head={ok_ref} grouped={ok_new}"; results.append(row); fails += 1
                            print(pname, kvn, P, K, use_sinks, row["status"], flush=True); continue
                        ref32 = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks)
                        e_ref, e_new = err_vs(ref32, o_ref), err_vs(ref32, o_new)
                        ok = e_new["finite"] and e_new["cosine"] >= 0.9999 and e_new["max_err_over_refmax"] <= max(1.6 * e_ref["max_err_over_refmax"], 1e-3)
                        t_ref = time_us(lambda: run(verify_splitkv_fwd, o_ref)); t_new = time_us(lambda: run(verify_shared_kv_fwd, o_new))
                        row.update(status="ok" if ok else "FAIL", per_head=e_ref, grouped=e_new, per_head_us=round(t_ref, 1), grouped_us=round(t_new, 1))
                        results.append(row); fails += 0 if ok else 1
                        print(f"{pname:6} {kvn:4} P={P:>6} K={K:>2} sinks={use_sinks!s:5} {'ok  ' if ok else 'FAIL'} cos {e_new['cosine']:.6f} "
                              f"err {e_new['max_err_over_refmax']:.2e} (per-head {e_ref['max_err_over_refmax']:.2e})  {t_ref:7.0f} -> {t_new:7.0f} us  x{t_ref/t_new:.2f}", flush=True)
    if a.out:
        json.dump({"results": results, "fails": fails}, open(a.out, "w"), indent=1)
    print(f"fails={fails}"); sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
