#!/usr/bin/env python3
"""Kernel-level cold-prefill attention bench: the Triton extend kernel (as the triton presets serve it) against
flashinfer's paged prefill over the same fp8 pool, one chunk of query rows on top of a cached prefix.

Sized by R9700's RAD-10c (a one-pass prefill kernel runs at x0.38-0.72 of their Triton extend kernel from 257 rows):
this asks whether the one-pass kernel we already ship for the flashinfer presets beats the two-stage Triton extend
at the triton presets' shapes (Flint sinks via the returned LSE, Gemma 4 global layers).

  SGLANG_EXTEND_ATTN_FP32_PV=0 SGLANG_EXTEND_ATTN_TILE=128:128,128,8 python scripts/bench/bench_prefill_attn_kernel.py --presets flint
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from test_extend_splitkv_route import PRESETS, time_us  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="flint")
    ap.add_argument("--prefixes", default="32768,131072,245760")
    ap.add_argument("--rows", default="4096")
    ap.add_argument("--kv-dtype", default="e5m2", choices=["e5m2", "bf16"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import flashinfer
    import sglang.kernels.ops.attention.extend_attention as ea
    dev = torch.device("cuda"); torch.manual_seed(0); out_rows = []
    pool_dt = torch.float8_e5m2 if a.kv_dtype == "e5m2" else torch.bfloat16
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    for pname in a.presets.split(","):
        p = PRESETS[pname]; d = p["head_dim"]; sm_scale = d ** -0.5
        k_scale, v_scale = (0.07, 0.05) if a.kv_dtype == "e5m2" else (1.0, 1.0)
        for P in [int(x) for x in a.prefixes.split(",")]:
            for L in [int(x) for x in a.rows.split(",")]:
                n = P + L
                k_buf = (torch.randn(n, p["h_kv"], d, device=dev) * (6.0 if a.kv_dtype == "e5m2" else 0.5)).to(pool_dt)
                v_buf = (torch.randn(n, p["h_kv"], d, device=dev) * (6.0 if a.kv_dtype == "e5m2" else 0.5)).to(pool_dt)
                perm = torch.randperm(n, device=dev, dtype=torch.int32)
                pre_idx, ext_idx = perm[:P].contiguous(), perm[P:].contiguous()
                q = torch.randn(L, p["h_q"], d, device=dev, dtype=torch.bfloat16)
                # the chunk's own K/V: bf16 operands for Triton, the pool's (rounded) values for both arms
                k_ext = (k_buf[ext_idx.long()].float() * k_scale).to(torch.bfloat16)
                v_ext = (v_buf[ext_idx.long()].float() * v_scale).to(torch.bfloat16)
                sinks = (torch.randn(p["h_q"], device=dev) * 2).float() if p["sinks"] else None
                qo = torch.tensor([0, L], device=dev, dtype=torch.int32)
                kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
                o_t = torch.empty(L, p["h_q"], d, device=dev, dtype=torch.bfloat16)
                f_tri = lambda: ea.extend_attention_fwd(q, k_ext, v_ext, o_t, k_buf, v_buf, qo, kv_indptr, pre_idx, None, True, None, L,
                                                        k_scale, v_scale, sm_scale=sm_scale, logit_cap=0.0, sinks=sinks)
                f_tri(); torch.cuda.synchronize(); us_tri = time_us(f_tri, iters=2, repeats=3)

                w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
                all_idx = torch.cat([pre_idx, ext_idx]).contiguous()
                w.plan(qo, torch.tensor([0, n], device=dev, dtype=torch.int32), all_idx,
                       torch.ones(1, device=dev, dtype=torch.int32), p["h_q"], p["h_kv"], d, 1,
                       causal=True, sm_scale=sm_scale, q_data_type=torch.bfloat16, kv_data_type=pool_dt)
                kv = (k_buf.unsqueeze(1), v_buf.unsqueeze(1))
                kw = dict(k_scale=k_scale, v_scale=v_scale) if a.kv_dtype == "e5m2" else {}
                def f_fi():
                    o, lse = w.run(q, kv, return_lse=True, **kw)
                    if sinks is not None:  # sink term through the LSE: o *= exp(lse) / (exp(lse) + exp(sink))
                        # flashinfer's LSE is log2-based
                        o = o * torch.sigmoid(lse * 0.6931471805599453 - sinks[None, :]).unsqueeze(-1).to(o.dtype)
                    return o
                o_f = f_fi(); torch.cuda.synchronize(); us_fi = time_us(f_fi, iters=2, repeats=3)
                cos = torch.nn.functional.cosine_similarity(o_t.float().flatten(), o_f.float().flatten(), dim=0).item()
                rel = ((o_t.float() - o_f.float()).abs().max() / o_t.float().abs().max()).item()
                row = dict(preset=pname, prefix=P, rows=L, kv=a.kv_dtype, triton_ms=round(us_tri / 1e3, 2), flashinfer_ms=round(us_fi / 1e3, 2),
                           ratio=round(us_fi / us_tri, 3), cosine=round(cos, 6), max_abs_over_max=round(rel, 4))
                out_rows.append(row); print(json.dumps(row), flush=True)
                del k_buf, v_buf, w; torch.cuda.empty_cache()
    if a.out:
        json.dump(dict(env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_EXTEND")}, flashinfer=flashinfer.__version__,
                       gpu=torch.cuda.get_device_name(0), rows=out_rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
