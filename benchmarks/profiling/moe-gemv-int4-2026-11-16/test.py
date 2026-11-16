import os, sys, json, itertools, torch
sys.path.insert(0, "/var/tmp/moe-gemv")
from gemv import moe_gemv_int4
torch.manual_seed(0); dev = "cuda"
E,K,N,G,topk = [int(v) for v in os.environ.get("MOE_SHAPE","128,2816,352,32,8").split(",")]
GATED = os.environ.get("MOE_GATED","1") == "1"
NW = 2*N if GATED else N
def mk(n_out, k_in):
    q = torch.randint(0,16,(E,n_out,k_in),dtype=torch.uint8,device=dev)
    B = (q[:,:,0::2] | (q[:,:,1::2] << 4)).contiguous()
    zq = torch.randint(0,16,(E,n_out,k_in//G),dtype=torch.uint8,device=dev)
    Z = (zq[:,0::2,:] | (zq[:,1::2,:] << 4)).contiguous()
    S = (torch.rand(E,n_out,k_in//G,device=dev)*0.02+0.001).half()
    W = (q.float() - zq.float().repeat_interleave(G,dim=2)) * S.float().repeat_interleave(G,dim=2)
    return B,Z,S,W
B1,Z1,S1,W1 = mk(NW,K); B2,Z2,S2,W2 = mk(K,N)
x = torch.randn(1,K,dtype=torch.float16,device=dev)
ids = torch.randperm(E,device=dev)[:topk].view(1,topk).to(torch.int32)
w = torch.softmax(torch.randn(1,topk,device=dev),-1).float()
# reference
ref1 = torch.stack([W1[ids[0,j]] @ x[0].float() for j in range(topk)])          # (topk, NW)
C1 = torch.empty(topk, NW, dtype=torch.float16, device=dev)
moe_gemv_int4(x, B1, C1, S1, Z1, w, ids, topk, False, G)
print("up   max rel err", ((C1.float()-ref1).abs().max()/ref1.abs().max()).item())
h = torch.randn(topk, N, dtype=torch.float16, device=dev)
ref2 = torch.stack([W2[ids[0,j]] @ h[j].float() * w[0,j] for j in range(topk)])
C2 = torch.empty(topk, K, dtype=torch.float16, device=dev)
moe_gemv_int4(h, B2, C2, S2, Z2, w, ids, 1, True, G)
print("down max rel err", ((C2.float()-ref2).abs().max()/ref2.abs().max()).item())
def t(fn, it=200):
    for _ in range(5): fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): fn()
    s,e=torch.cuda.Event(True),torch.cuda.Event(True); g.replay(); torch.cuda.synchronize()
    s.record()
    for _ in range(it//20): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000
res={}
for bn,ng,nw,st in itertools.product([8,16,32,64],[2,4,8,16],[1,2,4],[2,3,4]):
    try:
        u=t(lambda: moe_gemv_int4(x,B1,C1,S1,Z1,w,ids,topk,False,G,bn,ng,nw,st))
        d=t(lambda: moe_gemv_int4(h,B2,C2,S2,Z2,w,ids,1,True,G,bn,ng,nw,st))
        res[(bn,ng,nw,st)]=(u,d)
    except Exception as ex:
        pass
bu=min(res,key=lambda k:res[k][0]); bd=min(res,key=lambda k:res[k][1])
print("best up", bu, f"{res[bu][0]:.1f} us", " best down", bd, f"{res[bd][1]:.1f} us")
bytes_up = topk*NW*K/2 + topk*NW*(K//G)*2; bytes_dn = topk*K*N/2 + topk*K*(N//G)*2
print(f"roofline (936 GB/s): up {bytes_up/936e3:.1f} us, down {bytes_dn/936e3:.1f} us")
json.dump({"shape":[E,K,N,G,topk,GATED],"best_up":[bu,res[bu][0]],"best_down":[bd,res[bd][1]]},open(f"/var/tmp/moe-gemv/res-{E}-{K}-{N}.json","w"))
