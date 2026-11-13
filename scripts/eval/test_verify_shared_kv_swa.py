#!/usr/bin/env python
"""Grouped-head target verify on a SLIDING-WINDOW layer (3090 patch 082): does `verify_shared_kv_fwd` with
`sliding_window_size` match `extend_attention_fwd` (the path those layers took before) at Flint's per-rank shape, with
learned sinks and an fp8_e5m2 cache, and how much faster per layer? The rail holds the last min(seq_len, window) prefix
keys; draft row l must drop the l oldest keys of a full rail and none of a short one.
The first K+1 rail values are 64x sentinels, so a window edge off by one key moves the output by far more than rounding.
Gate against an fp32 reference built with the same mask: grouped max error <= 1.6x the extend kernel's or within one bf16
ulp of the reference max (8e-3), cosine >= 0.9999, finite; on a full rail the window-1 / window+1 references must both
sit >= 10x further away (the mask edge is exact).
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_verify_shared_kv_swa.py --out benchmarks/profiling/verify-shared-kv-swa-<date>.json
"""
import argparse, json, os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_extend_splitkv_route import PRESETS, err_vs, time_us  # noqa: E402


def fp32_window_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks, window):
    K, h_q, _ = q.shape; h_kv = k_ext.shape[1]; g = h_q // h_kv; P = kv_indices.numel()
    kp = k_buf[kv_indices].float() * k_scale; vp = v_buf[kv_indices].float() * v_scale
    k_all = torch.cat([kp, k_ext.float()], 0).repeat_interleave(g, 1); v_all = torch.cat([vp, v_ext.float()], 0).repeat_interleave(g, 1)
    s = torch.einsum("lhd,nhd->hln", q.float(), k_all) * sm_scale
    l = torch.arange(K, device=q.device)[:, None]; n = torch.arange(P + K, device=q.device)[None, :]
    keep = (n <= P + l) & (P + l <= n + window)
    s = s.masked_fill(~keep[None], float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.float()[:, None, None].expand(h_q, K, 1)], -1)
    p = torch.softmax(s, -1)[..., : P + K]
    return torch.einsum("hln,nhd->lhd", p, v_all)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--presets", default="flint"); ap.add_argument("--rails", default="512,511,300,40,3")
    ap.add_argument("--window", type=int, default=512); ap.add_argument("--drafts", default="2,4,8,16"); ap.add_argument("--kv-dtypes", default="bf16,e5m2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
    DT = {"bf16": torch.bfloat16, "e5m2": torch.float8_e5m2}
    dev = "cuda"; torch.manual_seed(0); results = []; fails = 0; W = a.window
    for pname in a.presets.split(","):
        p = PRESETS[pname]
        for kvn in a.kv_dtypes.split(","):
            for P in map(int, a.rails.split(",")):
                for K in map(int, a.drafts.split(",")):
                    for use_sinks in ([False, True] if p["sinks"] else [False]):
                        h_q, h_kv, d, dv = p["h_q"], p["h_kv"], p["head_dim"], p["v_head_dim"]
                        q = torch.randn(K, h_q, d, device=dev, dtype=torch.bfloat16)
                        k_ext = torch.randn(K, h_kv, d, device=dev, dtype=torch.bfloat16)
                        v_ext = torch.randn(K, h_kv, dv, device=dev, dtype=torch.bfloat16)
                        k_buf = torch.randn(P + 64, h_kv, d, device=dev, dtype=torch.bfloat16).to(DT[kvn])
                        kv_indices = torch.randperm(P + 64, device=dev)[:P].to(torch.int64)
                        v_bf = torch.randn(P + 64, h_kv, dv, device=dev, dtype=torch.bfloat16)
                        v_bf[kv_indices[: K + 1]] *= 64
                        v_buf = v_bf.to(DT[kvn])
                        kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
                        qo_indptr = torch.tensor([0, K], device=dev, dtype=torch.int32)
                        offs = torch.zeros(1, device=dev, dtype=torch.int32)
                        sinks = (torch.randn(h_q, device=dev) * 2).float() if use_sinks else None
                        sm_scale = d ** -0.5; k_scale = v_scale = 1.0
                        o_ref = torch.empty(K, h_q, dv, device=dev, dtype=torch.bfloat16); o_new = torch.empty_like(o_ref)
                        def run_ext(o):
                            extend_attention_fwd(q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K,
                                                 k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sliding_window_size=W, sinks=sinks, window_kv_offsets=offs)
                            return True
                        def run_new(o):
                            return bool(verify_shared_kv_fwd(q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K,
                                                             k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sliding_window_size=W, sinks=sinks,
                                                             window_kv_offsets=offs, max_bs=1))
                        row = {"preset": pname, "kv": kvn, "rail": P, "K": K, "sinks": use_sinks, "window": W}
                        try:
                            run_ext(o_ref); ok_new = run_new(o_new)
                        except Exception as e:
                            row["status"] = f"error: {str(e).splitlines()[0][:160]}"; results.append(row); fails += 1
                            print(pname, kvn, P, K, use_sinks, row["status"], flush=True); continue
                        if not ok_new:
                            row["status"] = "can_handle grouped=False"; results.append(row); fails += 1
                            print(pname, kvn, P, K, use_sinks, row["status"], flush=True); continue
                        ref32 = fp32_window_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks, W)
                        e_ref, e_new = err_vs(ref32, o_ref), err_vs(ref32, o_new)
                        ok = e_new["finite"] and e_new["cosine"] >= 0.9999 and e_new["max_err_over_refmax"] <= max(1.6 * e_ref["max_err_over_refmax"], 8e-3)
                        if P >= W and K >= 2:
                            edge = min(err_vs(fp32_window_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, k_scale, v_scale, sm_scale, sinks, w), o_new)["max_err_over_refmax"] for w in (W - 1, W + 1))
                            row["edge_err"] = edge; ok = ok and edge >= 10 * e_new["max_err_over_refmax"]
                        t_ref = time_us(lambda: run_ext(o_ref)); t_new = time_us(lambda: run_new(o_new))
                        row.update(status="ok" if ok else "FAIL", extend=e_ref, grouped=e_new, extend_us=round(t_ref, 1), grouped_us=round(t_new, 1))
                        results.append(row); fails += 0 if ok else 1
                        print(f"{pname:6} {kvn:4} rail={P:>4} K={K:>2} sinks={use_sinks!s:5} {'ok  ' if ok else 'FAIL'} cos {e_new['cosine']:.6f} "
                              f"err {e_new['max_err_over_refmax']:.2e} (extend {e_ref['max_err_over_refmax']:.2e})  {t_ref:6.0f} -> {t_new:6.0f} us  x{t_ref/t_new:.2f}", flush=True)
    if a.out:
        json.dump({"results": results, "fails": fails}, open(a.out, "w"), indent=1)
    print(f"fails={fails}"); sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
