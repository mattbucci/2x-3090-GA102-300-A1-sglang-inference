# greedy decode speed on code-like and prose prompts (short context); argv: port out.json
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
P = {
 "code1": "Write a complete Python module implementing an LRU cache class with get/put, a TTL option, type hints, docstrings and a small unittest suite.",
 "code2": "Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with error handling, then show five example calls with their outputs.",
 "code3": "Here is a function:\n\ndef merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out\n\nIt drops the tail of the longer list. Return the corrected function as a unified diff, then the full fixed file.",
 "tool": "Produce a JSON array of 12 objects describing fictional servers, each with keys hostname, ip, cpu_cores, ram_gb, role, tags (array) and last_deploy (ISO date). Output only JSON.",
 "prose1": "Explain how a B-tree differs from a B+-tree and when a database would choose each. Write about six paragraphs.",
 "prose2": "Write a short story of about 350 words about a lighthouse keeper who finds a message in a bottle.",
}
res = {}
for name, prompt in P.items():
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t1 = None; usage = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("usage"): usage = d["usage"]
        if t1 is None and d.get("choices"): t1 = time.time()
    t2 = time.time()
    res[name] = {"completion_tokens": usage["completion_tokens"], "tok_s": round((usage["completion_tokens"] - 1) / (t2 - t1), 1)}
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
