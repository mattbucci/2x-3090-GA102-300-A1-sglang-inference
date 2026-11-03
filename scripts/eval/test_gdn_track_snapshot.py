#!/usr/bin/env python3
"""Patch 076 unit test: the Triton GDN chunk kernel's in-kernel fp32 state snapshot at a
prefix-cache track point must equal the fp32 final state of a prefill truncated at that
boundary, and the bf16 per-chunk `h` the radix cache copied before the patch must NOT.

Runs on one GPU in a few seconds (small shapes). Exit 1 on mismatch.
"""
import sys, torch
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule, CHUNK_SIZE

def run(lens, seed=0, H=4, K=128, V=128, track=True):
    torch.manual_seed(seed)
    T = sum(lens); dev = "cuda"
    q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()), device=dev, dtype=torch.long)
    N = len(lens)
    state = torch.zeros(N, H, V, K, device=dev, dtype=torch.float32)
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    ts = torch.empty(N, H, V, K, device=dev, dtype=torch.float32) if track else None
    tci = torch.tensor([l // CHUNK_SIZE for l in lens], device=dev, dtype=torch.int32) if track else None
    o, _, h = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state, initial_state_indices=idx,
                                     cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
                                     track_state=ts, track_chunk_idx=tci)
    return (q, k, v, g, beta), state, h, ts

def main():
    lens = [300, 190, 64 * 5 + 17]            # all unaligned to 64 except none; boundaries 256, 128, 320
    inputs, final_state, h, ts = run(lens)
    q, k, v, g, beta = inputs
    ok = True
    off = 0; hoff = 0
    for i, L in enumerate(lens):
        b = (L // CHUNK_SIZE) * CHUNK_SIZE
        # reference: fresh prefill of exactly the boundary prefix of this sequence (fp32 final state)
        sl = slice(off, off + b)
        st = torch.zeros(1, *final_state.shape[1:], device="cuda", dtype=torch.float32)
        chunk_gated_delta_rule(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=st,
                               initial_state_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
                               cu_seqlens=torch.tensor([0, b], device="cuda", dtype=torch.long),
                               use_qk_l2norm_in_kernel=True)
        ref = st[0]
        snap_err = (ts[i] - ref).abs().max().item()
        h_bf16 = h[0, hoff + L // CHUNK_SIZE].to(torch.float32)   # what the pre-076 radix cache copied
        h_err = (h_bf16 - ref).abs().max().item()
        rel = ref.abs().max().item()
        print(f"seq {i} len {L} boundary {b}: fp32 snapshot max|err| {snap_err:.3e}  bf16 h max|err| {h_err:.3e}  (state max {rel:.3e})")
        ok &= snap_err <= 1e-5 * max(1.0, rel) and h_err > snap_err
        off += L; hoff += (L - 1) // CHUNK_SIZE + 1
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
