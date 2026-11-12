#!/usr/bin/env python
"""Extend kernel prefix loop on an fp8 pool with the bf16 P.V dot (flint preset: SGLANG_EXTEND_ATTN_FP32_PV=0): does
UPCAST_KV (q and softmax weights stay bf16, K/V upcast) sit closer to an fp32 reference than upstream's cast of q and p
to the pool dtype, and at what cost per layer? The cast arm is the shipped preset path before 2026-11-11; R9700 R97-S
confirmed the same cast on their e4m3 pool. Gate per cell: upcast max error <= 0.5x the cast arm's (or both under 1e-3),
cosine >= the cast arm's, finite; time reported, not gated (the served A/B gates speed).
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_extend_fp8_upcast.py --out benchmarks/profiling/extend-fp8-upcast-2026-11-11.json
"""
import argparse, json, os, sys, torch

os.environ.setdefault("SGLANG_EXTEND_ATTN_FP32_PV", "0")  # the flint preset arm; read at import
sys.path.insert(0, os.path.dirname(__file__))
from test_extend_splitkv_route import PRESETS, err_vs, fp32_reference, time_us  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--presets", default="flint"); ap.add_argument("--prefixes", default="8192,58000,131072")
    ap.add_argument("--suffixes", default="256,1024"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import sglang.kernels.ops.attention.extend_attention as ea
    assert not ea._FP32_SOFTMAX_PV, "run with SGLANG_EXTEND_ATTN_FP32_PV=0 (the bf16 P.V path under test)"
    assert hasattr(ea, "_FP8_KV_UPCAST"), "tree lacks the UPCAST_KV change"
    dev = torch.device("cuda"); torch.manual_seed(0); rows = []; fails = 0
    for pname in a.presets.split(","):
        p = PRESETS[pname]; sm_scale = p["head_dim"] ** -0.5; k_scale, v_scale = 0.07, 0.05
        for P in [int(x) for x in a.prefixes.split(",")]:
            pool_n = P + 64
            k_buf = (torch.randn(pool_n, p["h_kv"], p["head_dim"], device=dev) * 6.0).to(torch.float8_e5m2)
            v_buf = (torch.randn(pool_n, p["h_kv"], p["v_head_dim"], device=dev) * 6.0).to(torch.float8_e5m2)
            kv_indices = torch.randperm(pool_n, device=dev, dtype=torch.int32)[:P].contiguous()
            kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
            for K in [int(x) for x in a.suffixes.split(",")]:
                for use_sinks in ([False, True] if p["sinks"] else [False]):
                    q = torch.randn(K, p["h_q"], p["head_dim"], device=dev, dtype=torch.bfloat16)
                    k_ext = torch.randn(K, p["h_kv"], p["head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
                    v_ext = torch.randn(K, p["h_kv"], p["v_head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
                    sinks = (torch.randn(p["h_q"], device=dev) * 2).float() if use_sinks else None
                    qo = torch.tensor([0, K], device=dev, dtype=torch.int32)
                    ref32 = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks)
                    arms = {}
                    for arm, flag in (("cast", False), ("upcast", True)):
                        ea._FP8_KV_UPCAST = flag
                        o = torch.empty(K, p["h_q"], p["v_head_dim"], device=dev, dtype=torch.bfloat16)
                        fn = lambda: ea.extend_attention_fwd(q, k_ext, v_ext, o, k_buf, v_buf, qo, kv_indptr, kv_indices, None, True, None, K,
                                                             k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks)
                        fn(); torch.cuda.synchronize()
                        arms[arm] = {**err_vs(ref32, o), "us": round(time_us(fn, iters=5, repeats=3), 1)}
                    c, u = arms["cast"], arms["upcast"]
                    ok = (u["finite"] and u["cosine"] >= c["cosine"]
                          and (u["max_err_over_refmax"] <= 0.5 * c["max_err_over_refmax"] or max(u["max_err_over_refmax"], c["max_err_over_refmax"]) < 1e-3))
                    fails += not ok
                    row = dict(preset=pname, prefix=P, new_rows=K, sinks=use_sinks, cast=c, upcast=u, ok=ok)
                    rows.append(row); print(json.dumps(row), flush=True)
                    del ref32
            del k_buf, v_buf; torch.cuda.empty_cache()
    print(f"{len(rows) - fails}/{len(rows)} pass")
    if a.out:
        json.dump({"gpu": torch.cuda.get_device_name(), "rows": rows, "fails": fails}, open(a.out, "w"), indent=1)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
