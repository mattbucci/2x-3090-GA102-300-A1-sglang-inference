import json,glob,sys
from collections import Counter,defaultdict
MAIN={'bash':'command','edit':'input'}
FILL={'true','false','x','y','none','null','','test','replace','yes','no'}
ids=set(open(__file__.rsplit('/',1)[0]+'/instance-ids.txt').read().split())
def stats(cell,only):
    c=defaultdict(Counter); bad=defaultdict(set)
    for f in glob.glob(f'{cell}/sessions/*/.omp/agent/sessions/*/*.jsonl'):
        inst=f.split('/sessions/')[1].split('/')[0]
        if only and inst not in ids: continue
        for l in open(f):
            try: d=json.loads(l)
            except: continue
            m=d.get('message') or {}
            if m.get('role')=='assistant':
                for p in m.get('content',[]):
                    if p.get('type')=='toolCall' and p.get('name') in MAIN:
                        n=p['name']; a=p.get('arguments') or {}; v=a.get(MAIN[n])
                        c[n]['calls']+=1
                        if not isinstance(v,str) or v.strip().lower() in FILL: c[n]['degen']+=1; bad[n].add(inst)
            elif m.get('role') in ('toolResult','tool') and m.get('toolName')=='edit' and m.get('isError'): c['edit']['errors']+=1
    return {n:{**c[n],'rate_pct':round(100*c[n]['degen']/max(c[n]['calls'],1),1),'instances':len(bad[n])} for n in MAIN}
out={}
for cell in ['flint-omp-v4','flint-omp-v5','flint-omp-probe-editorder']:
    out[cell]={'same24':stats('evals/swebench/runs/'+cell,True)}
    if cell!='flint-omp-probe-editorder': out[cell]['full300']=stats('evals/swebench/runs/'+cell,False)
print(json.dumps(out,indent=1))
json.dump(out,open(sys.argv[1],'w'),indent=1)
