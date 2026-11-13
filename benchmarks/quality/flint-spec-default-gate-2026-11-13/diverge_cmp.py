import json
P=json.load(open('div-plain.json')); S=json.load(open('div-spec.json'))
def cmp(a,b):
    ta,tb=a['toks'],b['toks']
    for i in range(min(len(ta),len(tb))):
        if ta[i][0]!=tb[i][0]:
            top=dict((x[0],x[1]) for x in ta[i][2]); 
            gap = (ta[i][1]-top[tb[i][0]]) if tb[i][0] in top else None
            # margin in a's own distribution between top1 and top2
            srt=sorted(top.values(),reverse=True); m12=srt[0]-srt[1] if len(srt)>1 else None
            return f"div@{i} a={ta[i][0]!r}({ta[i][1]}) b={tb[i][0]!r}({tb[i][1]}) gap_in_a={None if gap is None else round(gap,2)} a_top12={None if m12 is None else round(m12,2)}"
    return f"identical({min(len(ta),len(tb))})"
for k in sorted(P):
    if k.endswith('/1'):
        b=k[:-2]
        print(b)
        print('  plain 1v2', cmp(P[b+'/1'],P[b+'/2'])); print('  plain 2v3', cmp(P[b+'/2'],P[b+'/3']))
        print('  spec  1v2', cmp(S[b+'/1'],S[b+'/2'])); print('  spec  2v3', cmp(S[b+'/2'],S[b+'/3']))
        print('  P1 v S1 ', cmp(P[b+'/1'],S[b+'/1'])); print('  P2 v S2 ', cmp(P[b+'/2'],S[b+'/2']))
