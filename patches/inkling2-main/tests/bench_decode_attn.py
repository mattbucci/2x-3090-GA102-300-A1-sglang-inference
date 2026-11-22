import torch, triton
from sglang.kernels.ops.attention.inkling2_rel_bias_ampere import rel_bias_attention
dev="cuda"; H,HKV,D,E=12,4,128,1024
for L in (32768, 131072, 250000):
    kb=torch.randn(L+8,HKV,D,device=dev,dtype=torch.bfloat16); vb=torch.randn_like(kb)
    q=torch.randn(1,H,D,device=dev,dtype=torch.bfloat16); bias=torch.randn(1,H,E,device=dev,dtype=torch.bfloat16)
    idx=torch.randperm(L+8,device=dev)[:L].to(torch.int32)
    t=lambda x: torch.tensor(x,dtype=torch.int32,device=dev)
    args=(q,kb,vb,bias,idx,t([0]),t([1]),t([0]),t([L]),[1])
    old=lambda: rel_bias_attention(*args,scale=D**-0.5,num_splits=16)
    new=lambda: rel_bias_attention(*args,scale=D**-0.5,decode=True)
    o1,_=old(); o2,_=new(); err=((o1.float()-o2.float()).abs().max()/o1.float().abs().max()).item()
    ms_old=triton.testing.do_bench(old); ms_new=triton.testing.do_bench(new)
    gb=2*L*HKV*D*2/1e9
    print(f"L={L:6}  per-head {ms_old*1e3:7.0f} us   grouped {ms_new*1e3:6.0f} us ({gb/ms_new*1e3:4.0f} GB/s)   diff {err:.1e}")
