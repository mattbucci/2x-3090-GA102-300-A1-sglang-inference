# M=1-oriented int4 (AWQ moe_wna16 layout) MoE GEMV: one program per (routed row, N block), no padded M rows.
# B: (E, N, K//2) uint8, low nibble = even k.  scale: (E, N, K//G).  zp: (E, N//2, K//G) uint8, low nibble = even n.
import torch, triton, triton.language as tl

@triton.jit
def moe_gemv_int4_kernel(a_ptr, b_ptr, c_ptr, s_ptr, z_ptr, w_ptr, ids_ptr,
                         N, K, stride_am, stride_be, stride_bn, stride_cm,
                         stride_se, stride_sn, stride_ze, stride_zn,
                         top_k: tl.constexpr, G: tl.constexpr, NG: tl.constexpr,
                         BLOCK_N: tl.constexpr, MUL_W: tl.constexpr, HAS_ZP: tl.constexpr):
    r = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(ids_ptr + r).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    if e < 0:
        tl.store(c_ptr + r * stride_cm + offs_n, tl.zeros((BLOCK_N,), dtype=c_ptr.dtype.element_ty), mask=nmask)
        return
    row = r // top_k
    BK: tl.constexpr = NG * G           # k per iteration
    HB: tl.constexpr = BK // 2          # packed bytes per iteration
    offs_g = tl.arange(0, NG)
    offs_h = tl.arange(0, G // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        g0 = k0 // G
        # packed weights [BLOCK_N, NG, G//2]
        boff = offs_n[:, None, None] * stride_bn + (k0 // 2 + offs_g[None, :, None] * (G // 2) + offs_h[None, None, :])
        kvalid = (k0 + offs_g[None, :, None] * G) < K
        byt = tl.load(b_ptr + e * stride_be + boff, mask=nmask[:, None, None] & kvalid, other=0).to(tl.int32)
        lo = (byt & 0xF).to(tl.float32)
        hi = (byt >> 4).to(tl.float32)
        gmask = (g0 + offs_g) < (K // G)
        sc = tl.load(s_ptr + e * stride_se + offs_n[:, None] * stride_sn + (g0 + offs_g)[None, :],
                     mask=nmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
        if HAS_ZP:
            zb = tl.load(z_ptr + e * stride_ze + (offs_n[:, None] // 2) * stride_zn + (g0 + offs_g)[None, :],
                         mask=nmask[:, None] & gmask[None, :], other=0).to(tl.int32)
            zp = ((zb >> ((offs_n[:, None] % 2) * 4)) & 0xF).to(tl.float32)
        else:
            zp = tl.full((BLOCK_N, NG), 8.0, tl.float32)
        # activations: even / odd k within each group
        ka = k0 + offs_g[:, None] * G + offs_h[None, :] * 2
        amask = ka < K
        ae = tl.load(a_ptr + row * stride_am + ka, mask=amask, other=0.0).to(tl.float32)
        ao = tl.load(a_ptr + row * stride_am + ka + 1, mask=amask, other=0.0).to(tl.float32)
        part = tl.sum(lo * ae[None, :, :] + hi * ao[None, :, :], axis=2)        # [BLOCK_N, NG] sum_k q*a
        asum = tl.sum(ae + ao, axis=1)                                             # [NG]  sum_k a
        acc += tl.sum(sc * (part - zp * asum[None, :]), axis=1)
    if MUL_W:
        acc = acc * tl.load(w_ptr + r).to(tl.float32)
    tl.store(c_ptr + r * stride_cm + offs_n, acc.to(c_ptr.dtype.element_ty), mask=nmask)

def moe_gemv_int4(A, B, C2d, S, Z, topk_weights, topk_ids, top_k, mul_w, G, BLOCK_N=32, NG=4, num_warps=4, num_stages=2):
    R = topk_ids.numel(); N = B.shape[1]; K = A.shape[1]
    assert K % G == 0
    NG = max(1, min(NG, K // G))
    while (K // G) % NG: NG //= 2
    grid = (R, triton.cdiv(N, BLOCK_N))
    moe_gemv_int4_kernel[grid](A, B, C2d, S, Z if Z is not None else S, topk_weights.reshape(-1), topk_ids.reshape(-1),
        N, K, A.stride(0), B.stride(0), B.stride(1), C2d.stride(0), S.stride(0), S.stride(1),
        Z.stride(0) if Z is not None else 0, Z.stride(1) if Z is not None else 0,
        top_k=top_k, G=G, NG=NG, BLOCK_N=BLOCK_N, MUL_W=mul_w, HAS_ZP=Z is not None,
        num_warps=num_warps, num_stages=num_stages)
