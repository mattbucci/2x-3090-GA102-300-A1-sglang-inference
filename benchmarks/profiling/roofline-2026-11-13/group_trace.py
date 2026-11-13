import gzip, json, glob, re, collections, sys
G = [("dense INT4 GEMM (attn proj + shared expert)", r"device::marlin::Marlin"), ("MoE expert INT4 GEMM", r"marlin_moe"),
     ("attention over KV", r"_fwd_grouped_kernel_stage1|_fwd_kernel_stage2|_verify_mla|^_fwd_kernel|create_flashinfer_kv_indices|store_kvcache|qknorm|rope"),
     ("TP all-reduce / NCCL", r"all_reduce|ncclDev"), ("MoE routing glue (topk/align/sum/scatter)", r"gatherTopK|align|moe_sum|scatter_gather|_moe_|sort|enable_if"),
     ("norms + activations", r"RMSNorm|act_and_mul"), ("lm_head / sampling / other GEMM", r"gemm|cutlass|splitKreduce|gemv|argmax|multinomial|cublas"),
     ("elementwise glue", r"elementwise|reduce_kernel|index_|Memcpy|Memset|copy")]
out = {}
for tag in sys.argv[1:]:
    f = sorted(glob.glob(f"{tag}/*TP-0.trace.json.gz"))[0]
    ev = json.load(gzip.open(f, "rt", errors="replace"))["traceEvents"]
    k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"], key=lambda e: e["ts"])
    span = k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]
    busy = 0; end = k[0]["ts"]
    for e in k:
        s, t = e["ts"], e["ts"] + e["dur"]
        if t > end: busy += t - max(s, end); end = t
    agg = collections.defaultdict(lambda: [0, 0.0]); other = collections.Counter()
    for e in k:
        for name, pat in G:
            if re.search(pat, e["name"]): break
        else:
            name = "other"; other[e["name"][:50]] += e["dur"]
        agg[name][0] += 1; agg[name][1] += e["dur"]
    N = 60
    out[tag] = {"step_ms": span / N / 1000, "busy_ms": busy / N / 1000, "idle_ms": (span - busy) / N / 1000, "kernels": len(k) / N,
                "groups": {n: {"n": a[0] / N, "ms": a[1] / N / 1000} for n, a in agg.items()}}
    print(f"== {tag}: step {span/N/1000:.2f} ms, busy {busy/N/1000:.2f}, idle {(span-busy)/N/1000:.2f}, kernels/step {len(k)/N:.0f}")
    for n, a in sorted(agg.items(), key=lambda x: -x[1][1]): print(f"   {a[1]/N/1000:6.2f} ms  x{a[0]/N:6.0f}  {n}")
    print("   other top:", [(n, round(d / N / 1000, 3)) for n, d in other.most_common(4)])
json.dump(out, open("groups.json", "w"), indent=1)
