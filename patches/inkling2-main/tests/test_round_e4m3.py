# _round_e4m3 (Triton, fp32) must equal torch's float8_e4m3fn round-trip for every |x| <= 448.
import torch, triton, triton.language as tl
from sglang.kernels.ops.moe.inkling2_moe import _round_e4m3
@triton.jit
def k(X, Y, N, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + o, _round_e4m3(tl.load(X + o, o < N)), o < N)
x = torch.cat([torch.randn(2_000_000, device="cuda") * s for s in (1e-3, 0.05, 1, 30, 150)]).clamp(-448, 448)
grid = torch.arange(256, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).float()
grid = grid[grid.isfinite()]; mid = (grid.sort().values[1:] + grid.sort().values[:-1]) / 2
x = torch.cat([x, grid, mid, -mid, torch.tensor([448., -448., 0., 2**-10, 2**-9 * 1.5], device="cuda")])
y = torch.empty_like(x); k[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), 1024)
ref = x.to(torch.float8_e4m3fn).float()
bad = (y != ref) & ~((y == 0) & (ref == 0))
print("mismatches", int(bad.sum()), "of", x.numel()); print("examples", x[bad][:5].tolist(), y[bad][:5].tolist(), ref[bad][:5].tolist())
print("PASS" if int(bad.sum()) == 0 else "FAIL")
