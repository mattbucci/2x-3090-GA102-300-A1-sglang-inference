# cold-prefill TTFT: flush the prefix cache, send a fresh ~N-token random-word prompt, max_tokens 1. argv: port N out.json
import json, sys, time, random, urllib.request
port, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = f"http://127.0.0.1:{port}"
urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", data=b"", method="POST"), timeout=60).read()
random.seed(n)
words = [random.choice(["alpha","river","stone","cloud","seven","maple","quartz","lantern","orbit","velvet","harbor","cinder"]) for _ in range(int(n * 0.75))]
body = {"model": "default", "messages": [{"role": "user", "content": " ".join(words) + "\nReply with one word."}], "max_tokens": 1, "temperature": 0}
t = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
el = time.time() - t
r = {"target_tokens": n, "prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "ttft_s": round(el, 2)}
json.dump(r, open(out, "w")); print(r)
