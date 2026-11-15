import gzip, json, glob, collections, statistics as st
for tag, pat in (('v0522 r0', 'prof-0522/*TP-0*'), ('v0523 r0', 'prof-0523/*TP-0*'), ('ptr r0', 'prof-ptr/*TP-0*')):
    f = sorted(glob.glob(pat + '.trace.json.gz'))[0]
    ev = sorted([e for e in json.load(gzip.open(f, 'rt', errors='replace'))['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'kernel'], key=lambda e: e['ts'])
    ks = [e for e in ev if 'gptq_awq' in e['name'] and str(e['args'].get('grid')) == '[174, 1, 1]'][23 * 40:]
    import bisect
    ts = [e['ts'] for e in ev]
    by = collections.defaultdict(list); names = collections.defaultdict(collections.Counter)
    for k in ks:
        lvl = round((k['dur'] - 124) / 37)
        i = bisect.bisect_left(ts, k['ts'] - 2000); ov = 0.0
        while i < len(ev) and ev[i]['ts'] < k['ts'] + k['dur']:
            o = ev[i]; i += 1
            if o is k: continue
            a = max(o['ts'], k['ts']); b = min(o['ts'] + o['dur'], k['ts'] + k['dur'])
            if b > a: ov += b - a; names[lvl][o['name'][:45]] += b - a
        by[lvl].append(ov)
    print(tag)
    for l in sorted(by):
        top = ', '.join(f'{n}={v/len(by[l]):.0f}' for n, v in names[l].most_common(3))
        print(f'  level {l}: n={len(by[l]):5d} mean overlap {st.mean(by[l]):6.1f} us | {top}')
