# Sweep Triton MoE (int4 w4a16, zp, non-gated relu2) configs at Nemotron-3-Nano per-rank shapes (TP=2).
# Times up (w13) and down (w2) separately via override_config on the whole fused_moe call is not enough,
# so we time fused_moe with (up_cfg, down_cfg) via try_get_optimal_moe_config monkeypatch.
import itertools, json, sys, torch
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="/home/letsrtfm/AI/models/hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ", tp_size=1, trust_remote_code=True))
torch.manual_seed(0); dev="cuda"
E,K,N,G,topk=128,2688,928,32,6
M=int(sys.argv[1]) if len(sys.argv)>1 else 1
x=torch.randn(M,K,dtype=torch.float16,device=dev)
w1=torch.randint(0,255,(E,N,K//2),dtype=torch.uint8,device=dev)
w2=torch.randint(0,255,(E,K,N//2),dtype=torch.uint8,device=dev)
s1=(torch.rand(E,N,K//G,device=dev)*0.01).half(); s2=(torch.rand(E,K,N//G,device=dev)*0.01).half()
z1=torch.randint(0,255,(E,N//2,K//G),dtype=torch.uint8,device=dev); z2=torch.randint(0,255,(E,K//2,N//G),dtype=torch.uint8,device=dev)
cfg=MoeRunnerConfig(activation="relu2",is_gated=False,inplace=True)
gates=[torch.randn(M,E,device=dev) for _ in range(20)]
tk=[select_experts(x,g,TopKConfig(top_k=topk,renormalize=True)) for g in gates]
orig=fm.try_get_optimal_moe_config
CUR={}
def patched(*a,**kw):
    if kw.get("return_down_config"):
        return dict(CUR["up"]),(dict(CUR["down"]),None)
    return dict(CUR["up"])
def run(t):
    return fm.fused_moe(x,w1,w2,t,moe_runner_config=cfg,use_int4_w4a16=True,w1_scale=s1,w2_scale=s2,w1_zp=z1,w2_zp=z2,block_shape=[0,G])
def timeit(up,down):
    CUR["up"],CUR["down"]=up,down
    try:
        for t in tk[:3]: run(t)
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for t in tk: run(t)
        torch.cuda.synchronize()
        st,en=torch.cuda.Event(True),torch.cuda.Event(True)
        g.replay(); torch.cuda.synchronize()
        st.record()
        for _ in range(10): g.replay()
        en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en)/(10*len(tk))*1000  # us per fused_moe call
    except Exception as e:
        return float("inf")

fm.try_get_optimal_moe_config=patched
D={"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1}
base=timeit(D,D); print(f"M={M} default {base:.1f} us/call", flush=True)
def space():
    for bn,bk,w,st in itertools.product([16,32,64,128],[32,64,128,256],[1,2,4,8],[2,3,4]):
        yield {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":bn,"BLOCK_SIZE_K":bk,"GROUP_SIZE_M":1,"num_warps":w,"num_stages":st}
res=[]
for c in space():
    t=timeit(c,D); res.append((t,c))
res.sort(key=lambda r:r[0]); bu=res[0][1]; print("best up (down=default)", f"{res[0][0]:.1f}", bu, flush=True)
res2=[]
for c in space():
    t=timeit(bu,c); res2.append((t,c))
res2.sort(key=lambda r:r[0]); bd=res2[0][1]; print("best up+down", f"{res2[0][0]:.1f}", bd, flush=True)
# confirm
print(f"confirm default {timeit(D,D):.1f}  tuned {timeit(bu,bd):.1f} us/call")
json.dump({"M":M,"default_us":base,"up":bu,"down":bd,"tuned_us":res2[0][0],"top_up":[(t,c) for t,c in res[:5]],"top_down":[(t,c) for t,c in res2[:5]]},open(f"tune-M{M}.json","w"),indent=1)
