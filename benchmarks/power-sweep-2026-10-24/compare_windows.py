import csv, re, subprocess, statistics, sys, time, datetime
# usage: compare_windows.py <t0_epoch> <t1_epoch> <t2_epoch> <t3_epoch> [labelA] [labelB]
a0,a1,b0,b1 = map(float, sys.argv[1:5]); la = sys.argv[5] if len(sys.argv)>5 else "A"; lb = sys.argv[6] if len(sys.argv)>6 else "B"
import glob
rows=[r for f in sorted(glob.glob('/var/tmp/power-sweep-2026-10-24/sustained-*.csv')) for r in csv.DictReader(open(f)) if r.get('cpu_tctl_c')]
def win(t0,t1): return [r for r in rows if t0<=float(r['ts'])<=t1]
def summ(R, lab):
    g1=[int(r['temp_c']) for r in R if r['gpu']=='1']; g0=[int(r['temp_c']) for r in R if r['gpu']=='0']
    p=[float(r['power_w']) for r in R]; d=[float(r['dimm50_c']) for r in R if r['dimm50_c']]; c=[float(r['cpu_tctl_c']) for r in R if r.get('cpu_tctl_c')]
    print(f"{lab:14s} n={len(R)//2:3d}  CPU Tctl mean {statistics.mean(c):.1f} max {max(c):.1f} | DIMM mean {statistics.mean(d):.2f} max {max(d):.2f} last {d[-1]:.2f} | GPU0 max {max(g0)} GPU1 max {max(g1)} | GPU mean draw {statistics.mean(p):.0f} W, at-cap {100*sum(x>=float(R[0]['pl_w'])-10 for x in p)/len(p):.0f} %")
summ(win(a0,a1), la); summ(win(b0,b1), lb)
# decode throughput from the lane's server log (UTC timestamps)
log=subprocess.run(["docker","logs","bakeoff-sglang","--since","120m"],capture_output=True,text=True); txt=log.stdout+log.stderr
pat=re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Decode batch, #running-req: (\d+), #full token: (\d+),.*gen throughput \(token/s\): ([\d.]+)")
A,B={},{}
for m in pat.finditer(txt):
    ts=datetime.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    if m.group(2)!="1": continue
    tok=int(m.group(3)); tp=float(m.group(4))
    if tp<5: continue
    bucket = "<8K" if tok<8192 else "8-32K" if tok<32768 else ">32K"
    (A if a0<=ts<=a1 else B if b0<=ts<=b1 else {}).setdefault(bucket,[]).append(tp)
for bk in ("<8K","8-32K",">32K"):
    xa,xb=A.get(bk,[]),B.get(bk,[])
    if xa and xb: print(f"decode {bk:6s}: {la} median {statistics.median(xa):6.1f} tok/s (n={len(xa)})  {lb} median {statistics.median(xb):6.1f} (n={len(xb)})  delta {100*(statistics.median(xb)/statistics.median(xa)-1):+.1f} %")
    else: print(f"decode {bk:6s}: insufficient ({len(xa)} / {len(xb)})")
