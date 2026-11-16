# Grouped verify kernel (verify_shared_kv_fwd) at the gemma4-31b GLOBAL-layer per-rank shape:
# 16 Q / 2 KV heads x 512, e5m2 KV, 6 verify rows. Sweep tile / splits / stages; baseline = served default.
import itertools, json, os, sys, torch
R = "/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference"
sys.path.insert(0, R + "/scripts/eval")
from test_extend_splitkv_route import PRESETS, err_vs, fp32_reference, time_us
import sglang.kernels.ops.attention.verify_mla as vm
p = PRESETS["gemma4-31b"]; dev = "cuda"; torch.manual_seed(0)
h_q, h_kv, d, dv = p["h_q"], p["h_kv"], p["head_dim"], p["v_head_dim"]
K = int(os.environ.get('K_ROWS', '6'))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "tile"
out = []
def setup(P):
    q = torch.randn(K, h_q, d, device=dev, dtype=torch.bfloat16)
    k_ext = torch.randn(K, h_kv, d, device=dev, dtype=torch.bfloat16); v_ext = torch.randn(K, h_kv, dv, device=dev, dtype=torch.bfloat16)
    k_buf = torch.randn(P + 64, h_kv, d, device=dev, dtype=torch.bfloat16).to(torch.float8_e5m2)
    v_buf = torch.randn(P + 64, h_kv, dv, device=dev, dtype=torch.bfloat16).to(torch.float8_e5m2)
    kv_indices = torch.arange(P, device=dev, dtype=torch.int64)   # served pool is allocated near-sequentially
    kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32); qo_indptr = torch.tensor([0, K], device=dev, dtype=torch.int32)
    sm = d ** -0.5
    o = torch.empty(K, h_q, dv, device=dev, dtype=torch.bfloat16)
    ref32 = fp32_reference(q, k_ext, v_ext, k_buf, v_buf, kv_indices, 1.0, 1.0, sm, None)
    def run():
        return bool(vm.verify_shared_kv_fwd(q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices, None, True, None, K, 1.0, 1.0, sm_scale=sm))
    return run, o, ref32
def one(P, run, o, ref32, tile, ms, st, tag=""):
    vm._ENV_TILE = tile; vm._ENV_TILE_OVERFLOW.clear(); vm.MAX_N_SPLITS = ms; vm._STAGE1_NUM_STAGES = st; vm._VMLA_CACHE.clear()
    row = dict(P=P, tile=tile, max_splits=ms, stages=st, tag=tag)
    try:
        assert run(); 
        if tile is not None and d in vm._ENV_TILE_OVERFLOW: raise RuntimeError("SMEM overflow -> table tile")
        e = err_vs(ref32, o); t = time_us(run)
        row.update(us=round(t, 1), cos=e["cosine"], err=e["max_err_over_refmax"])
        print(f"P={P} tile={tile} splits<={ms:3} stages={st} {tag:8} {t:8.0f} us  cos {e['cosine']:.6f} err {e['max_err_over_refmax']:.2e}", flush=True)
    except Exception as ex:
        row["error"] = str(ex).splitlines()[0][:160]; print(row, flush=True)
    out.append(row); return row.get("us")
for P in (244000, 32000):
    run, o, ref32 = setup(P)
    one(P, run, o, ref32, None, 32, 1, "served")
    if STAGE == "tile":
        for tile in itertools.product((4, 8, 16), (16, 32, 64, 128), (4, 8)):
            one(P, run, o, ref32, tile, 40, 3)
    if STAGE == "stages":
        for tile, st, ms in itertools.product(((4, 32, 4), (4, 64, 4), (4, 64, 8), (8, 16, 8), (8, 32, 8), (8, 64, 8), (4, 128, 8)), (1, 2), (32, 40, 64, 80)):
            one(P, run, o, ref32, tile, ms, st)
        for ms in (32, 64, 80):
            one(P, run, o, ref32, (4, 32, 4), ms, 3)
    del run, o, ref32; torch.cuda.empty_cache()
json.dump(out, open(f"/var/tmp/g31-verify/sweep1-{STAGE}-K{K}.json", "w"), indent=1)
