import gzip, json, glob, statistics as st
for v in ('0522', '0523'):
    for r in (0, 1):
        f = sorted(glob.glob(f'prof-{v}/*TP-{r}.trace.json.gz'))[0]
        ev = [e for e in json.load(gzip.open(f, 'rt', errors='replace'))['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'kernel']
        for g, nm in (('[174, 1, 1]', 'gate/up'), ('[504, 1, 1]', 'down')):
            ks = sorted([e for e in ev if 'gptq_awq' in e['name'] and str(e['args'].get('grid')) == g], key=lambda e: e['ts'])[23 * 40:]
            n = len(ks) // 23
            per = [st.median(ks[s * 23 + l]['dur'] for s in range(n)) for l in range(23)]
            print(f'v{v} rank{r} {nm:8s} sum/step {sum(per)/1000:.2f} ms :', ' '.join(f'{x:.0f}' for x in per))
        ar = [e['dur'] for e in ev if 'all_reduce' in e['name']]; ar = ar[len(ar)//4:]
        print(f'v{v} rank{r} allreduce mean {st.mean(ar):.1f} us x53 = {st.mean(ar)*53/1000:.2f} ms/step')
