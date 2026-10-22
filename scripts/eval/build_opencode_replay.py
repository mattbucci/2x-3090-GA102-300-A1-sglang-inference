#!/usr/bin/env python3
"""Rebuild the OpenAI-format conversation opencode sent to the server, from an
opencode session store + a captured first request (system prompt + tool schemas).

Usage: build_opencode_replay.py <opencode.db> <full-request.jsonl> <api-key-file|-> <out-prefix>

  opencode.db         the scaffold's sqlite store (~/.local/share/opencode/opencode.db,
                      copy db + -wal + -shm together); the main session is the first
                      by time_created (opencode's post-run cleanup session is the second)
  full-request.jsonl  rows {"path":..., "body": <chat-completions body>} — a full-body
                      capture of the scaffold's first request (the fake server in
                      evals/swebench/scaffold_request_audit.py slims bodies; capture the
                      whole body once to get the system prompt + tools verbatim)
  api-key-file        the per-run bearer; every occurrence in session data is replaced
                      by <KEY> before writing (pass - for none)
  out-prefix          writes <out-prefix>-k<N>.json for every assistant turn N whose
                      finish was 'length' (the runaway turns): {turn, input_tokens,
                      messages (history up to, not incl., that turn), tools}

Encoding matches what opencode 1.14.x sends: assistant turns carry
reasoning_content + tool_calls; an unparsable call was rewritten by
experimental_repairToolCall to name "invalid" with input {tool, error:<full raw
text>} — the history carries that object as the call's arguments. Prompt-token
counts of the rebuilt requests match the session's recorded input tokens exactly.
Replay with replay_turns.py.
"""
import sqlite3, sys, json, pathlib
db, cap, keyf, outp = sys.argv[1:5]
key = open(keyf).read().strip() if keyf != '-' else None
def scrub(s):
    return s.replace(key, '<KEY>') if key else s
first = [json.loads(l) for l in open(cap)]
first = [r for r in first if r['body'].get('tools')][0]['body']
system = first['messages'][0]['content']; tools = first['tools']
c = sqlite3.connect(db)
sessions = [r[0] for r in c.execute('select id from session order by time_created')]
sid = sessions[0]
msgs = []
for mid, data in c.execute('select id,data from message where session_id=? order by time_created', (sid,)):
    d = json.loads(scrub(data))
    parts = [json.loads(scrub(p[0])) for p in c.execute('select data from part where message_id=? order by time_created', (mid,))]
    msgs.append((d, parts))
conv = [{'role': 'system', 'content': system}]
out_turns = []
k = 0
for d, parts in msgs:
    if d['role'] == 'user':
        text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'text')
        conv.append({'role': 'user', 'content': text})
        continue
    k += 1
    if d.get('finish') == 'length':
        out_turns.append((k, json.loads(json.dumps(conv)), d))
    reasoning = ''.join(p.get('text', '') for p in parts if p.get('type') == 'reasoning')
    text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'text')
    tcs = []; results = []
    for p in parts:
        if p.get('type') != 'tool': continue
        st = p['state']; name = p['tool']
        tcs.append({'id': p['callID'], 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(st.get('input', {}))}})
        results.append({'role': 'tool', 'tool_call_id': p['callID'], 'content': str(st.get('output', ''))})
    am = {'role': 'assistant', 'content': text or None}
    if reasoning: am['reasoning_content'] = reasoning
    if tcs: am['tool_calls'] = tcs
    conv.append(am); conv.extend(results)
for k, cv, d in out_turns:
    p = pathlib.Path(f'{outp}-k{k}.json')
    p.write_text(json.dumps({'turn': k, 'input_tokens': d['tokens']['input'], 'messages': cv, 'tools': tools}))
    print(p, 'turn', k, 'ctx', d['tokens']['input'], 'messages', len(cv))
