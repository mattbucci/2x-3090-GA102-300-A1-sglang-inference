# Marlin MoE (awq_marlin, uint4 + zp) at M=1 on Flint per-rank shape vs the 095 GEMV, CUDA-graph timed.
import sys, torch
sys.path.insert(0,"/var/tmp/moe-gemv")
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="/home/letsrtfm/AI/models/hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ", tp_size=1, trust_remote_code=True))
from sglang.test.test_marlin_utils import awq_marlin_quantize
from sgl_kernel.scalar_type import scalar_types
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.topk import fused_topk_torch_native
torch.manual_seed(0)
E,K,N,G,topk=256,2048,256,64,6; dt=torch.bfloat16
a=torch.randn((1,K),device="cuda",dtype=dt)/10
def q(w):
    Q,S,Z=[],[],[]
    for i in range(w.shape[0]):
        _,qw,s,z=awq_marlin_quantize(w[i].transpose(1,0),scalar_types.uint4,G); Q.append(qw); S.append(s); Z.append(z)
    return torch.stack(Q).contiguous(),torch.stack(S),torch.stack(Z)
w1=torch.randn((E,2*N,K),device="cuda",dtype=dt)/20; w2=torch.randn((E,K,N),device="cuda",dtype=dt)/20
q1,s1,z1=q(w1); q2,s2,z2=q(w2); del w1,w2
score=torch.randn((1,E),device="cuda",dtype=dt)
tw,ti=fused_topk_torch_native(a,score,topk,False)
run=lambda: fused_marlin_moe(a,q1,q2,s1,s2,score,tw,ti,global_num_experts=E,w1_zeros=z1,w2_zeros=z2,num_bits=4,is_k_full=True)
for _ in range(5): run()
torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(20): run()
s,e=torch.cuda.Event(True),torch.cuda.Event(True); g.replay(); torch.cuda.synchronize(); s.record()
for _ in range(10): g.replay()
e.record(); torch.cuda.synchronize()
print(f"Marlin MoE fused call at M=1 (Flint per-rank shape): {s.elapsed_time(e)/200*1000:.1f} us")
