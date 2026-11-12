import time, torch, flashinfer
dev = torch.device("cuda"); torch.manual_seed(0)
L, hq, hkv, d = 8192, 16, 2, 128; N = 140000
k = (torch.randn(N, hkv, d, device=dev) * 6).to(torch.float8_e5m2); v = (torch.randn(N, hkv, d, device=dev) * 6).to(torch.float8_e5m2)
q = torch.randn(L, hq, d, device=dev).to(torch.bfloat16)
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev); ones = torch.ones(1, dtype=torch.int32, device=dev)
def plan(w, P, idt):
    idx = torch.arange(1, P + L + 1, device=dev, dtype=idt)
    w.plan(torch.tensor([0, L], dtype=torch.int32, device=dev), torch.tensor([0, P + L], dtype=torch.int32, device=dev), idx, ones, hq, hkv, d, 1,
           causal=True, sm_scale=d ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e5m2)
    return idx
def t(w):
    f = lambda: w.run(q, (k.unsqueeze(1), v.unsqueeze(1)), return_lse=True)
    f(); torch.cuda.synchronize(); t0 = time.perf_counter(); f(); f(); torch.cuda.synchronize(); return round((time.perf_counter() - t0) * 500, 1)
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD"); keep = plan(w, 122880, torch.int32); print("fresh i32", t(w))
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD"); keep = plan(w, 122880, torch.int64); print("fresh i64", t(w))
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
for P in range(0, 122881, 8192): keep = plan(w, P, torch.int32)
print("replanned-growing i32", t(w))
w2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD"); keep2 = plan(w2, 0, torch.int32); print("w2 P=0", t(w2)); keep2 = plan(w2, 122880, torch.int32); print("w2 replan 122880", t(w2))
