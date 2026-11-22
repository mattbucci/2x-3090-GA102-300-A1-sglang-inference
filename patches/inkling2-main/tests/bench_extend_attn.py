import torch, triton
from sglang.kernels.ops.attention.inkling2_rel_bias_ampere import rel_bias_attention
dev="cuda"; H,HKV,D,E=12,4,128,1024
for prefix, new, win in ((28672, 4096, 0), (126976, 4096, 0), (245904, 4096, 0), (245904, 4096, 1024)):
    L=prefix+new
    kb=torch.randn(L+8,HKV,D,device=dev,dtype=torch.bfloat16); vb=torch.randn_like(kb)
    q=torch.randn(new,H,D,device=dev,dtype=torch.bfloat16); bias=torch.randn(new,H,E,device=dev,dtype=torch.bfloat16)
    idx=torch.arange(L,device=dev,dtype=torch.int32)
    t=lambda x: torch.tensor(x,dtype=torch.int32,device=dev)
    args=(q,kb,vb,bias,idx,t([0]),t([new]),t([0]),t([L]),[new])
    old=lambda: rel_bias_attention(*args,scale=D**-0.5,window=win,allow_grouped=False)
    grp=lambda: rel_bias_attention(*args,scale=D**-0.5,window=win)
    o1,_=old(); o2,_=grp(); err=((o1.float()-o2.float()).abs().max()/o1.float().abs().max()).item()
    a=triton.testing.do_bench(old,rep=200); b=triton.testing.do_bench(grp,rep=200)
    print(f"prefix={prefix:6} chunk={new} window={win:4}  per-head {a:7.1f} ms   grouped {b:7.1f} ms  ({a/b:.2f}x)  diff {err:.1e}")
    del kb,vb
