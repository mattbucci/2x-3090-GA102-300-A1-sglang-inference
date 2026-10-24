import json, os, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__))
out = open(os.path.join(D, "samples.csv"), "a")
if out.tell() == 0:
    out.write("ts,phase,gpu,pl_w,temp_c,power_w,sm_mhz,mem_mhz,fan_pct,throttle,dimm50_c,dimm52_c\n")
while True:
    try:
        phase = open(os.path.join(D, "phase")).read().strip()
    except Exception:
        phase = "?"
    if phase == "STOP":
        break
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit,temperature.gpu,power.draw,clocks.sm,clocks.mem,fan.speed,clocks_throttle_reasons.active",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        s = json.loads(subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=10).stdout)
        d50 = s.get("spd5118-i2c-12-50", {}).get("temp1", {}).get("temp1_input", "")
        d52 = s.get("spd5118-i2c-12-52", {}).get("temp1", {}).get("temp1_input", "")
        ts = time.time()
        for line in q:
            f = [x.strip() for x in line.split(",")]
            out.write(f"{ts:.0f},{phase},{f[0]},{f[1]},{f[2]},{f[3]},{f[4]},{f[5]},{f[6]},{f[7]},{d50},{d52}\n")
        out.flush()
    except Exception as e:
        print("sampler err", e, file=sys.stderr)
    time.sleep(2)
