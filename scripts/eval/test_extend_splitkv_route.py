#!/usr/bin/env python
"""Short cache-hit extend through the split-KV kernel (3090 patch 070, port of R9700's 107): is it numerically the same
as `extend_attention_fwd` at the real shapes, with learned sinks and an fp8 cache, and how much faster per layer?
Mirrors the route: q/k/v of K new tokens zero-padded to a 16-row tile, qo_indptr [0, 16], max_len_extend 16, first K
output rows compared. Gate against an fp32 reference (both kernels quantize softmax weights to the V dtype on an fp8
cache, so neither is exact): split-KV max error <= 1.25x the extend kernel's, cosine >= 0.9999, finite.
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_extend_splitkv_route.py --out benchmarks/profiling/extend-splitkv-route-2026-11-01.json
"""
import argparse, json, statistics, sys, time, torch

PRESETS = {  # per-rank (TP=2) full-attention shapes
    "qwen38": dict(h_q=32, h_kv=2, head_dim=256, v_head_dim=256, sinks=False),
    "flint": dict(h_q=16, h_kv=2, head_dim=128, v_head_dim=128, sinks=True),
}

def fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks):
    K, h_q, d = q.shape; h_kv = k_ext.shape[1]; grp = h_q // h_kv
    Kp = k_buf[kv_indices].float() * k_scale; Vp = v_buf[kv_indices].float() * v_scale
    out = torch.empty(K, h_q, v_ext.shape[2], device=q.device, dtype=torch.float32)
    causal = torch.ones(K, K, device=q.device, dtype=torch.bool).tril()
    for h in range(h_q):
        kh = h // grp; qh = q[:, h, :].float()
        lp = qh @ Kp[:, kh, :].T * sm_scale; ls = (qh @ k_ext[:, kh, :].float().T * sm_scale).masked_fill(~causal, float("-inf"))
        logits = torch.cat([lp, ls], dim=1)
        if sinks is not None:
            logits = torch.cat([logits, sinks[h].float().expand(K, 1)], dim=1)
        pr = torch.softmax(logits, dim=1)
        out[:, h, :] = pr[:, :lp.shape[1]] @ Vp[:, kh, :] + pr[:, lp.shape[1]:lp.shape[1] + K] @ v_ext[:, kh, :].float()
    return out

def err_vs(ref32, out):
    o = out.float(); finite = bool(torch.isfinite(o).all())
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref32.flatten(), dim=0).item() if finite else float("nan")
    refmax = ref32.abs().max().item(); mx = (o - ref32).abs().max().item() if finite else float("inf")
    return {"finite": finite, "cosine": round(cos, 7), "max_err_over_refmax": round(mx / max(refmax, 1e-30), 6)}

def time_us(fn, iters=10, repeats=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--presets", default="qwen38,flint"); ap.add_argument("--prefixes", default="8192,58000,197000")
    ap.add_argument("--suffixes", default="1,8,16"); ap.add_argument("--kv-dtypes", default="bf16,e4m3,e5m2"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
    dev = torch.device("cuda"); torch.manual_seed(0); results = []; fails = 0
    DT = {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    for pname in a.presets.split(","):
        p = PRESETS[pname]; sm_scale = p["head_dim"] ** -0.5
        for kvn in a.kv_dtypes.split(","):
            kv_dtype = DT[kvn]; fp8 = kv_dtype != torch.bfloat16
            k_scale = 0.07 if fp8 else 1.0; v_scale = 0.05 if fp8 else 1.0
            for P in [int(x) for x in a.prefixes.split(",")]:
                pool_n = P + 1024
                k_buf = (torch.randn(pool_n, p["h_kv"], p["head_dim"], device=dev) * (6.0 if fp8 else 0.5)).to(kv_dtype)
                v_buf = (torch.randn(pool_n, p["h_kv"], p["v_head_dim"], device=dev) * (6.0 if fp8 else 0.5)).to(kv_dtype)
                kv_indices = torch.randperm(pool_n, device=dev, dtype=torch.int32)[:P].contiguous()
                kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
                for K in [int(x) for x in a.suffixes.split(",")]:
                    for use_sinks in ([False, True] if p["sinks"] else [False]):
                        q = torch.randn(K, p["h_q"], p["head_dim"], device=dev, dtype=torch.bfloat16)
                        k_ext = torch.randn(K, p["h_kv"], p["head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
                        v_ext = torch.randn(K, p["h_kv"], p["v_head_dim"], device=dev, dtype=torch.bfloat16) * 0.5
                        sinks = (torch.randn(p["h_q"], device=dev) * 2).float() if use_sinks else None
                        qo = torch.tensor([0, K], device=dev, dtype=torch.int32)
                        o_ref = torch.empty(K, p["h_q"], p["v_head_dim"], device=dev, dtype=torch.bfloat16)
                        def ref():
                            extend_attention_fwd(q, k_ext, v_ext, o_ref, k_buf, v_buf, qo, kv_indptr, kv_indices, None, True, None, K,
                                                 k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks)
                        # the route's call: pad to a 16-row tile
                        n_pad = max(16, 1 << (K - 1).bit_length()); pad = n_pad - K
                        q3 = torch.cat([q, q.new_zeros((pad,) + q.shape[1:])]); k3 = torch.cat([k_ext, k_ext.new_zeros((pad,) + k_ext.shape[1:])])
                        v3 = torch.cat([v_ext, v_ext.new_zeros((pad,) + v_ext.shape[1:])]); o3 = q3.new_empty((n_pad, p["h_q"], p["v_head_dim"]))
                        qo_pad = torch.tensor([0, n_pad], device=dev, dtype=torch.int32)
                        def cand():
                            return verify_splitkv_fwd(q3, k3, v3, o3, k_buf, v_buf, qo_pad, kv_indptr, kv_indices, None, True, None, n_pad,
                                                      k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks, max_bs=1)
                        ref(); torch.cuda.synchronize()
                        try:
                            ran = cand(); torch.cuda.synchronize()
                        except Exception as e:
                            results.append({"preset": pname, "kv": kvn, "prefix": P, "K": K, "sinks": use_sinks, "status": f"error: {str(e).splitlines()[0][:160]}"}); fails += 1
                            print(f"{pname:6} {kvn:4} P={P:>6} K={K:>2} sinks={use_sinks!s:5} ERROR {str(e).splitlines()[0][:90]}", flush=True); continue
                        if not ran:
                            results.append({"preset": pname, "kv": kvn, "prefix": P, "K": K, "sinks": use_sinks, "status": "can_handle=False"}); fails += 1
                            print(f"{pname:6} {kvn:4} P={P:>6} K={K:>2} sinks={use_sinks!s:5} can_handle=False", flush=True); continue
                        ref32 = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks)
                        e_ref = err_vs(ref32, o_ref); e_new = err_vs(ref32, o3[:K])
                        ok = e_new["finite"] and e_new["cosine"] >= 0.9999 and e_new["max_err_over_refmax"] <= max(1.25 * e_ref["max_err_over_refmax"], 1e-3)
                        t_ref = time_us(ref); t_new = time_us(cand)
                        r = {"preset": pname, "kv": kvn, "prefix": P, "K": K, "sinks": use_sinks, "status": "ok" if ok else "FAIL",
                             "extend_err": e_ref, "splitkv_err": e_new, "extend_us": round(t_ref, 1), "splitkv_us": round(t_new, 1), "speedup": round(t_ref / t_new, 2)}
                        results.append(r); fails += 0 if ok else 1
                        print(f"{pname:6} {kvn:4} P={P:>6} K={K:>2} sinks={use_sinks!s:5} {'ok  ' if ok else 'FAIL'} cos {e_new['cosine']:.6f} err {e_new['max_err_over_refmax']:.2e} (extend {e_ref['max_err_over_refmax']:.2e})  {t_ref:8.0f} -> {t_new:7.0f} us  x{t_ref/t_new:.1f}", flush=True)
    if a.out:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0), "results": results}, open(a.out, "w"), indent=1); print("wrote", a.out)
    print("RESULT:", "FAIL" if fails else "PASS"); return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
