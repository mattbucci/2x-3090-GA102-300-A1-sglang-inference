import sys, time, json, importlib, torch, os
sys.path.insert(0, os.path.dirname(__file__))
N=int(os.environ.get("N",222684)); H=int(os.environ.get("H",16)); KH=int(os.environ.get("KH",2)); D=int(os.environ.get("D",512))
torch.manual_seed(0); dev="cuda"
k=(torch.randn(N+64,KH,D,device=dev)*0.5).to(torch.float8_e5m2); v=(torch.randn(N+64,KH,D,device=dev)*0.5).to(torch.float8_e5m2)
q=torch.randn(1,H,D,device=dev,dtype=torch.bfloat16)
indptr=torch.tensor([0,N],dtype=torch.int32,device=dev); idx=torch.arange(N,dtype=torch.int32,device=dev)
out={}
ref=None
for mod,block,warps,splits in json.loads(os.environ["GRID"]):
    m=importlib.import_module(mod)
    src=open(m.__file__).read()
    # per-variant launcher geometry: patch the launcher's locals through a copy
    tag=f"{mod}-N{block}-w{warps}-s{splits}"
    name=f"v_{mod}_{block}_{warps}"
    p=os.path.join(os.path.dirname(__file__),name+".py")
    i=src.index("def _decode_grouped_att_m_fwd("); 
    body=src[i:].replace("    BLOCK = 32\n",f"    BLOCK = {block}\n",1).replace("    num_warps = 4\n",f"    num_warps = {warps}\n",1)
    open(p,"w").write(src[:i]+body); mm=importlib.import_module(name)
    o=torch.zeros(1,H,D,device=dev,dtype=torch.bfloat16)
    logits=torch.zeros(1,H,splits,D,device=dev,dtype=torch.float32); lse=torch.zeros(1,H,splits,device=dev,dtype=torch.float32)
    ns=torch.full((1,),splits,dtype=torch.int32,device=dev)
    def run(): mm.decode_attention_fwd_grouped(q,k,v,o,indptr,idx,logits,lse,ns,splits,D**-0.5,1.0)
    try:
        run(); torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(30): run()
        torch.cuda.synchronize(); ms=(time.perf_counter()-t)/30*1e3
    except Exception as e:
        print(tag,"ERR",str(e)[:200]); continue
    of=o.float().clone()
    if ref is None: ref=of
    err=(of-ref).abs().max().item(); rel=((of-ref).norm()/ref.norm()).item()
    gbs=N*KH*D*2/1e9/(ms/1e3)
    out[tag]={"ms":ms,"GBps":gbs,"max_abs_vs_base":err,"rel_l2_vs_base":rel}
    print(f"{tag:28s} {ms:7.3f} ms  {gbs:6.0f} GB/s  maxabs {err:.4g} rel {rel:.4g}",flush=True)
json.dump(out,open(os.path.join(os.path.dirname(__file__),f"bench-{os.environ.get('TAG','x')}-N{N}.json"),"w"),indent=1)
