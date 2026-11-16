# fused_moe end-to-end at M=1: GEMV path (module flag) vs the tuned Triton path; output diff + timing.
import os, sys, json, torch
sys.argv=["x","1"]
src=open("/var/tmp/nem-moe/bench2.py").read().split("fm.try_get_optimal_moe_config=patched")[0]
exec(src)
import sglang.kernels.ops.moe.fused_moe_triton_kernels as km
fm.try_get_optimal_moe_config=patched
cfgs=json.load(open(os.environ["MOE_TUNED"]))
up,down=cfgs["up"],cfgs["down"]
cfg2=MoeRunnerConfig(activation=ACT,is_gated=GATED,inplace=False)
# realistic small weights: random nibbles already; use moderate scales
CUR["up"],CUR["down"]=up,down
def run_out(t):
    return fm.fused_moe(x.clone(),w1,w2,t,moe_runner_config=cfg2,use_int4_w4a16=True,w1_scale=s1,w2_scale=s2,w1_zp=z1,w2_zp=z2,block_shape=[0,G])
def run_out(t):
    xx=x.clone()
    fm.fused_moe(xx,w1,w2,t,moe_runner_config=cfg,use_int4_w4a16=True,w1_scale=s1,w2_scale=s2,w1_zp=z1,w2_zp=z2,block_shape=[0,G])
    return xx
outs={}
for flag in (0,1):
    km._MOE_GEMV_INT4_MAX_M=flag
    outs[flag]=torch.stack([run_out(t) for t in tk[:8]]).float()
d=(outs[0]-outs[1]).abs().max().item(); mag=outs[0].abs().max().item()
print(f"e2e max |diff| {d:.4g} vs max |out| {mag:.4g} (rel {d/mag:.2e}); nan={torch.isnan(outs[1]).any().item()}")
for flag in (0,1):
    km._MOE_GEMV_INT4_MAX_M=flag
    print(f"flag {flag}: {timeit(up,down):.1f} us per fused_moe call")
