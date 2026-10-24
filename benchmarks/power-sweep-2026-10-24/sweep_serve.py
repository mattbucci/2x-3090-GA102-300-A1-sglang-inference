import importlib.util, json, os, subprocess, sys, time, requests
D = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference"
spec = importlib.util.spec_from_file_location("blc", f"{REPO}/scripts/bench/bench_long_context.py")
blc = importlib.util.module_from_spec(spec); spec.loader.exec_module(blc)
BASE = "http://localhost:23334"
LEVELS = list(range(200, 401, 10))
CTXS = [1024, 32768, 131072]
OUT = os.path.join(D, "serve-results.json")
def phase(p): open(os.path.join(D, "phase"), "w").write(p)
def set_pl(p):
    for i in (0, 1):
        subprocess.run(["sudo", "nvidia-smi", "-i", str(i), "-pl", str(p)], capture_output=True, text=True, check=True)
    got = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    assert all(abs(float(g) - p) < 0.5 for g in got), got
def flush():
    for ep in ("/flush_cache",):
        try: requests.post(BASE + ep, timeout=60)
        except Exception as e: print("flush err", e)
    time.sleep(1)
model = blc.server_model(BASE)
tok = requests.get(BASE + "/get_model_info", timeout=10).json()["model_path"]
print("model", model, "tokenizer", tok, flush=True)
results = json.load(open(OUT)) if os.path.exists(OUT) else {}
phase("warmup"); 
for _ in range(2): blc.run_bench(BASE, model, 128, 10, tokenizer=tok)
for P in LEVELS:
    if str(P) in results and all(str(c) in results[str(P)] for c in CTXS):
        continue
    set_pl(P); phase(f"pl{P}-settle"); time.sleep(8)
    results.setdefault(str(P), {})
    for c in CTXS:
        flush(); phase(f"pl{P}-ctx{c}")
        t0 = time.time()
        m = blc.run_bench(BASE, model, c, 100, tokenizer=tok)
        m["wall_s"] = round(time.time() - t0, 1)
        results[str(P)][str(c)] = m
        print(f"[{time.strftime('%H:%M:%S')}] pl={P} ctx={c} ttft={m.get('ttft_ms')} tpot={m.get('tpot_ms')} tok/s={m.get('tok_per_sec')} in={m.get('actual_input_tokens')}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
phase("serve-done")
print("SERVE_SWEEP_DONE", flush=True)
