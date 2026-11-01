#!/usr/bin/env python
"""Recover the Marlin AWQ weight layout as a gather map, empirically and exactly (W4A8 prefill prerequisite).

`sgl_kernel.awq_marlin_repack` is a value-independent permutation of nibble positions from the AWQ layout
([K, N/8] int32, logical column 8j + {0,2,4,6,1,3,5,7}[i] in nibble i of word j) to Marlin's [K/16, N*2] tiles.
Encoding the logical index of every nibble in 4-bit pieces across 7 repacks (28 bits >= log2(K*N)) recovers, for
every output nibble, the logical (k, n) it came from. The test then checks the map is tile-periodic (16 k x 64 n
tiles with a fixed intra-tile order), extracts that 1024-entry table, and validates the whole thing against the
Marlin GEMM itself: dequantising a random AWQ weight through the recovered map must equal Marlin(I_K) within fp16.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_marlin_tile_perm.py --k 5120 --n 1024
"""
import argparse, json, sys, time, torch
from sglang.srt.layers.quantization.marlin_utils import (apply_awq_marlin_linear, awq_to_marlin_zero_points, marlin_make_empty_g_idx,
                                                          marlin_make_workspace, marlin_permute_scales, scalar_types)
from sglang.srt.layers.quantization.utils import pack_cols

AWQ_INTERLEAVE = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])


def _awq_marlin_repack(packed, k, n, bits):
    from sglang.kernels.ops.quantization.awq_marlin_repack import awq_marlin_repack

    return awq_marlin_repack(packed, size_k=k, size_n=n, num_bits=bits)


def awq_pack(q_kn):  # logical uint4 [K, N] -> AWQ int32 [K, N/8]
    k, n = q_kn.shape
    q_il = q_kn.reshape(k, n // 8, 8)[:, :, AWQ_INTERLEAVE].reshape(k, n)  # nibble i of word j = logical col 8j + interleave[i]
    return pack_cols(q_il.to(torch.int32), 4, k, n)

def marlin_repack(packed, k, n):
    return _awq_marlin_repack(packed.contiguous(), k, n, 4)

def nibbles(words):  # int32 [R, C] -> uint8 [R, C*8] little-endian 4-bit fields
    w = words.to(torch.int64) & 0xFFFFFFFF
    return torch.stack([((w >> (4 * i)) & 0xF) for i in range(8)], dim=-1).reshape(words.shape[0], -1).to(torch.uint8)

def recover_map(k, n, dev):
    idx = torch.arange(k * n, device=dev, dtype=torch.int64).reshape(k, n)
    src = torch.zeros(k * 16 // 16 * 0 + (k // 16) * (n * 2) * 8, device=dev, dtype=torch.int64)
    out_shape = None
    for p in range(7):
        q = ((idx >> (4 * p)) & 0xF).to(torch.int32)
        m = marlin_repack(awq_pack(q), k, n)
        out_shape = m.shape
        nb = nibbles(m).reshape(-1).to(torch.int64)
        src |= nb << (4 * p)
    assert src.numel() == k * n, (src.numel(), k * n)
    return src, out_shape  # src[out_nibble] = logical k*n + ... index

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5120); ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--group", type=int, default=128); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = torch.device("cuda"); k, n, g = a.k, a.n, a.group
    t0 = time.time(); src, out_shape = recover_map(k, n, dev); t_map = time.time() - t0
    src_k = src // n; src_n = src % n
    # permutation sanity: every logical index appears exactly once
    assert torch.equal(torch.sort(src).values, torch.arange(k * n, device=dev)), "not a permutation"
    # tile periodicity: Marlin output row r (of K/16) holds k-tile r; within a row, nibble index c in [0, n*16)
    R = out_shape[0]; C = out_shape[1] * 8
    src_k2 = src_k.reshape(R, C); src_n2 = src_n.reshape(R, C)
    k_in_tile = src_k2 - 16 * torch.arange(R, device=dev)[:, None]
    assert bool(((k_in_tile >= 0) & (k_in_tile < 16)).all()), "k-tile rows are not 16-aligned per output row"
    # column groups of 1024 nibbles = one 16x64 tile; find the n-tile each group covers and the intra-tile pattern
    groups = C // 1024; assert C % 1024 == 0
    n_base = (src_n2.reshape(R, groups, 1024) // 64)
    assert bool((n_base == n_base[..., :1]).all()), "a 1024-nibble group spans more than one 64-column tile"
    n_in_tile = src_n2.reshape(R, groups, 1024) % 64; k_t = k_in_tile.reshape(R, groups, 1024)
    table = (k_t[0, 0] * 64 + n_in_tile[0, 0])  # [1024] intra-tile source index for output nibble i
    same = bool(((k_t * 64 + n_in_tile) == table).all())
    tile_order = n_base[:, :, 0]  # [R, groups] -> which n-tile sits in group gi
    seq = bool(torch.equal(tile_order[0], torch.arange(groups, device=dev)))
    print(f"map recovered in {t_map:.2f}s: out {tuple(out_shape)}, intra-tile table periodic={same}, n-tiles in order={seq}")
    # validate against the Marlin GEMM: dequant via the map vs Marlin(I_K)
    torch.manual_seed(0)
    q = torch.randint(0, 16, (k, n), device=dev, dtype=torch.int32)
    z = torch.randint(0, 16, (k // g, n), device=dev, dtype=torch.int32)
    s = (torch.rand(k // g, n, device=dev) * 0.02 + 0.005).half()
    packed = awq_pack(q); zp_packed = awq_pack(z)  # AWQ qzeros share the layout
    mq = marlin_repack(packed, k, n); ms = marlin_permute_scales(s, k, n, g); mz = awq_to_marlin_zero_points(zp_packed, k // g, n, 4)  # size_k here = number of groups (as the kernel object passes it)
    e = marlin_make_empty_g_idx(dev); ws = marlin_make_workspace(dev)
    I = torch.eye(k, device=dev, dtype=torch.float16)
    # AWQ (has_zp=True): the Marlin kernel wants the unbiased uint4 type, not uint4b8
    w_marlin = apply_awq_marlin_linear(input=I, weight=mq, weight_scale=ms, weight_zp=mz, g_idx=e, g_idx_sort_indices=e, workspace=ws,
                                       quant_type=scalar_types.uint4, output_size_per_partition=n, input_size_per_partition=k)  # [k, n]
    # dequant through the recovered map: read nibbles of mq, scatter to logical (k, n)
    nb = nibbles(mq).reshape(-1).to(torch.int32)
    q_rec = torch.empty(k * n, device=dev, dtype=torch.int32); q_rec[src] = nb; q_rec = q_rec.reshape(k, n)
    assert torch.equal(q_rec, q), "recovered q != original q"
    w_map = ((q.float() - z.repeat_interleave(g, dim=0).float()) * s.repeat_interleave(g, dim=0).float()).half()
    err = (w_map.float() - w_marlin.float()).abs().max().item(); ref = w_marlin.float().abs().max().item()
    ok = same and seq and err <= 2e-3 * max(ref, 1e-6)
    print(f"Marlin(I) vs map-dequant: max|err| {err:.3e} (ref max {ref:.3e}) -> {'PASS' if ok else 'FAIL'}")
    if a.out:
        json.dump({"k": k, "n": n, "group": g, "out_shape": list(out_shape), "intra_tile_table": table.tolist(), "periodic": same, "tiles_in_order": seq,
                   "max_err": err, "ref_max": ref, "pass": ok}, open(a.out, "w"))
        print("wrote", a.out)
    print("RESULT:", "PASS" if ok else "FAIL"); return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
