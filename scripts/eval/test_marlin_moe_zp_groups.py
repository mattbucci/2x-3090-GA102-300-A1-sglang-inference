#!/usr/bin/env python
"""Marlin MoE zero-point regression test (patch 067).

Runs Flint's layer-10 tensors through SGLang's AWQ-Marlin dense GEMM and the fused
Marlin MoE kernel, at group 32 and group 64, against a dequantized reference. Before
067 the MoE path returned rel err ~124 / ~31 at group 32 (TP=1 / TP=2 expert shapes);
after it ~0.001, same as group 64. Needs the sglang env, ~2 GB on the chosen GPU, and
the two checkpoints under /data/models/hf-mattbucci.

  python scripts/eval/test_marlin_moe_zp_groups.py [--device 0]
"""
import json, sys, importlib.util, torch
from safetensors import safe_open
import glob
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--device", type=int, default=0); _args = _ap.parse_args()
torch.cuda.set_device(_args.device); dev = torch.device(f"cuda:{_args.device}")
REPO="/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference"
spec = importlib.util.spec_from_file_location("q", f"{REPO}/scripts/quantize/quantize_flint_rtn_awq.py"); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
from sglang.srt.layers.quantization.awq.awq import AWQMarlinConfig
from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import AWQMarlinLinearKernel, AWQMoEKernel
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

def load(d, names):
    out = {}
    idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
    byfile = {}
    for n in names: byfile.setdefault(idx[n], []).append(n)
    for f, ns in byfile.items():
        with safe_open(f"{d}/{f}", "pt") as fh:
            for n in ns: out[n] = fh.get_tensor(n)
    return out

class L(torch.nn.Module): pass

def dense_test(d, g, qc):
    p = "model.layers.10.self_attn.q_proj"
    t = load(d, [f"{p}.qweight", f"{p}.qzeros", f"{p}.scales"])
    W = q.unpack_awq(t[f"{p}.qweight"], t[f"{p}.qzeros"], t[f"{p}.scales"]).to(dev)  # [out, in]
    layer = L()
    for k in ("qweight", "qzeros", "scales"):
        setattr(layer, k, torch.nn.Parameter(t[f"{p}.{k}"].to(dev), requires_grad=False))
    K, N = t[f"{p}.qweight"].shape[0], t[f"{p}.scales"].shape[1]
    layer.input_size_per_partition = K; layer.output_size_per_partition = N; layer.num_groups = K // g
    kern = AWQMarlinLinearKernel(qc); kern.process_weights_after_loading(layer)
    torch.manual_seed(0); x = (torch.randn(4, K, device=dev) * 0.5).half()
    y = kern.apply(layer, x); ref = x.float() @ W.t()
    return (y.float() - ref).norm() / ref.norm()

def moe_test(d, g, qc, inter, E=8):
    names = []
    for e in range(E):
        for w in ("gate_proj", "up_proj", "down_proj"):
            for k in ("qweight", "qzeros", "scales"): names.append(f"model.layers.10.mlp.experts.{e}.{w}.{k}")
    t = load(d, names)
    H = 2048; pf = 8
    w13q = torch.empty(E, H, 2 * inter // pf, dtype=torch.int32); w13z = torch.empty(E, H // g, 2 * inter // pf, dtype=torch.int32); w13s = torch.empty(E, H // g, 2 * inter, dtype=torch.float16)
    w2q = torch.empty(E, inter, H // pf, dtype=torch.int32); w2z = torch.empty(E, inter // g, H // pf, dtype=torch.int32); w2s = torch.empty(E, inter // g, H, dtype=torch.float16)
    refs = []
    for e in range(E):
        b = f"model.layers.10.mlp.experts.{e}."
        gq, gz, gs = t[b+"gate_proj.qweight"][:, :inter//pf], t[b+"gate_proj.qzeros"][:, :inter//pf], t[b+"gate_proj.scales"][:, :inter]
        uq, uz, us = t[b+"up_proj.qweight"][:, :inter//pf], t[b+"up_proj.qzeros"][:, :inter//pf], t[b+"up_proj.scales"][:, :inter]
        dq, dz, ds = t[b+"down_proj.qweight"][:inter], t[b+"down_proj.qzeros"][:inter//g], t[b+"down_proj.scales"][:inter//g]
        w13q[e] = torch.cat([gq, uq], 1); w13z[e] = torch.cat([gz, uz], 1); w13s[e] = torch.cat([gs, us], 1)
        w2q[e], w2z[e], w2s[e] = dq, dz, ds
        refs.append((q.unpack_awq(gq, gz, gs), q.unpack_awq(uq, uz, us), q.unpack_awq(dq, dz, ds)))  # [out,in] each
    layer = L()
    for k, v in (("w13_qweight", w13q), ("w13_qzeros", w13z), ("w13_scales", w13s), ("w2_qweight", w2q), ("w2_qzeros", w2z), ("w2_scales", w2s)):
        setattr(layer, k, torch.nn.Parameter(v.to(dev), requires_grad=False))
    layer.intermediate_size_per_partition = inter
    kern = AWQMoEKernel(qc); kern.process_weights_after_loading(layer)
    torch.manual_seed(0); M = 8; x = (torch.randn(M, H, device=dev) * 0.5).half()
    topk_ids = (torch.arange(M, device=dev) % E).view(M, 1).to(torch.int32); topk_w = torch.ones(M, 1, device=dev, dtype=torch.float32)
    gating = torch.zeros(M, E, device=dev, dtype=torch.float32)
    ws = marlin_make_workspace(dev, max_blocks_per_sm=4)
    y = fused_marlin_moe(x, layer.w13_qweight, layer.w2_qweight, layer.w13_scales, layer.w2_scales, gating, topk_w, topk_ids,
                         global_num_experts=E, expert_map=None, g_idx1=None, g_idx2=None,
                         sort_indices1=layer.w13_g_idx_sort_indices, sort_indices2=layer.w2_g_idx_sort_indices,
                         w1_zeros=layer.w13_qzeros, w2_zeros=layer.w2_qzeros, workspace=ws, num_bits=4, is_k_full=True)
    ref = torch.zeros(M, H, device=dev)
    for i in range(M):
        Wg, Wu, Wd = [r.to(dev) for r in refs[i % E]]
        h = x[i].float(); a = torch.nn.functional.silu(h @ Wg.t()) * (h @ Wu.t()); ref[i] = a @ Wd.t()
    return (y.float() - ref).norm() / ref.norm()

for tag, g in (("g32", 32), ("g64-gptq", 64)):
    d = f"/data/models/hf-mattbucci/Flint-AWQ-INT4-{tag}"
    cfg = json.load(open(d + "/config.json"))["quantization_config"]
    qc = AWQMarlinConfig.from_config(cfg)
    print(f"{tag}: dense q_proj (K=2048,N=4096) rel err {dense_test(d, g, qc):.4f}")
    for inter in (512, 256):
        try:
            print(f"{tag}: MoE marlin inter={inter} (TP={'1' if inter==512 else '2'}-shape) rel err {moe_test(d, g, qc, inter):.4f}")
        except Exception as ex:
            print(f"{tag}: MoE inter={inter} EXC {type(ex).__name__}: {str(ex)[:160]}")
