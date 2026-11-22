# Marlin MoE GEMM (Inkling 2 repack) vs dense dequant reference, NVFP4 (e2m1 + e4m3 per-16 scale) and MXFP8.
import torch, sys
sys.path.insert(0, "/data/sglang-main-inkling/test/registered/kernels/ops/gemm")
from test_marlin_moe_gemv import _metadata, EXPERTS, N, TOP_K
from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as m
torch.manual_seed(0)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
def run(kind, k, tokens):
    if kind == "nvfp4":
        w = torch.randint(0, 256, (EXPERTS, N, k // 2), dtype=torch.uint8, device="cuda")
        sc = (torch.rand(EXPERTS, N, k // 16, device="cuda") * 4 + 0.25).to(torch.float8_e4m3fn)
        marlin = m.repack(w, sc)
        lo, hi = E2M1[(w & 15).long()], E2M1[(w >> 4).long()]
        wd = torch.stack([lo, hi], -1).reshape(EXPERTS, N, k) * sc.float().repeat_interleave(16, -1)
    else:
        w = torch.randn(EXPERTS, N, k, device="cuda").to(torch.float8_e4m3fn)
        sc = torch.randint(120, 131, (EXPERTS, N, k // 32), dtype=torch.uint8, device="cuda")
        marlin = m.repack_mxfp8(w, sc)
        wd = w.float() * torch.exp2(sc.float() - 127).repeat_interleave(32, -1)
    counts, offsets, dst2src = _metadata(tokens, torch.Generator().manual_seed(tokens))
    x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    routing = m.route(counts, offsets, tokens * TOP_K, tokens, dst2src)
    out = m.gemm(x, marlin, N, routing, TOP_K).float()
    # output rows are in expert-sorted (dst) order: row r = source token-slot dst2src[r]
    sorted_expert = torch.cat([torch.full((int(c),), e) for e, c in enumerate(counts.tolist())])
    expert_of_src = torch.empty_like(sorted_expert)
    expert_of_src[dst2src.long().cpu()] = sorted_expert  # output row r = source slot r
    wd = wd.cpu()
    ref = torch.einsum("rk,rnk->rn", x.float().cpu().repeat_interleave(TOP_K, 0), wd[expert_of_src])
    out = out.cpu()
    if out.shape != ref.shape:
        print(kind, "shape", out.shape, ref.shape); return 1.0
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    err_alt = None
    print(f"{kind:5} k={k:5} tokens={tokens:3}  rel err {err:.2e}")
    return err
worst = max(run(kd, k, t) for kd in ("nvfp4", "mxfp8") for k in (1152, 2304) for t in (1, 3, 64))
print("WORST", worst, "PASS" if worst < 2e-2 else "FAIL")
