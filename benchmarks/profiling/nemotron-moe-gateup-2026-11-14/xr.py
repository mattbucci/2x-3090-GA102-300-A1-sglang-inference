import gzip, json, glob, collections
def lv(pat):
    f = sorted(glob.glob(pat))[0]
    ev = [e for e in json.load(gzip.open(f, 'rt', errors='replace'))['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    ks = sorted([e for e in ev if 'gptq_awq' in e['name'] and str(e['args'].get('grid')) == '[174, 1, 1]'], key=lambda e: e['ts'])
    return [round((k['dur'] - 124) / 37) for k in ks]
for d in ('prof-0522', 'prof-0523', 'prof-ptr'):
    a, b = lv(d + '/*TP-0.trace.json.gz'), lv(d + '/*TP-1.trace.json.gz')
    n = min(len(a), len(b)); a, b = a[23*40:n], b[23*40:n]
    c = collections.Counter(zip(a, b))
    same = sum(v for (x, y), v in c.items() if x == y) / len(a)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b)); va = sum((x-ma)**2 for x in a); vb = sum((y-mb)**2 for y in b)
    print(d, 'n', len(a), 'mean level r0 %.2f r1 %.2f' % (ma, mb), 'same-level frac %.2f' % same, 'pearson %.3f' % (cov / (va*vb)**.5))
    # consecutive-step autocorrelation within a layer on rank0
    ac = [(a[i], a[i+23]) for i in range(len(a)-23)]
    print('   lag-1-step same-layer equal frac %.2f' % (sum(x == y for x, y in ac)/len(ac)), ' level hist r0', sorted(collections.Counter(a).items()))
