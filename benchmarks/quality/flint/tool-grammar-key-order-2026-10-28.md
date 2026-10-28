# Flint strict tool grammar forces the schema's key order — omp's `i` swallowed the command (2026-10-28)

Found on the daily Flint check while attributing the flint:omp lane's second empty patch
(django-15814): the session's bash calls read `{"i": "sed -n '2360,2400p' …", "command": "true"}`,
and one read `{"i": "read<arg_key>path</arg_key><arg_value>/testbed/…", "command": "x"}`.

## Mechanism

`launch.sh flint` exports `FLINT_STRICT_TOOLS=1`: the plugin compiles every tool's JSON schema
into XGrammar's structural tag (`get_model_structural_tag("glm_4_7", strict)`), and XGrammar's
object grammar is **ordered** — properties must appear in the schema's declared order
(`any_order=False`; `any_order=True` exists in xgrammar 0.2.7 but then "required keys may be
missing", which removes the reason strict mode is on: Flint omits required keys on its own,
`tool-schema-probe-2026-10-22.json`).

omp 18.12.2 declares `i` (a short intent string) as the FIRST required property of `bash`,
`read`, `edit`, `write`, `grep`, `glob`, `web_search`. Flint's native habit is `command` first.
So after `<tool_call>bash` the grammar forces `<arg_key>i</arg_key><arg_value>` (jump-forward
fills the tokens), and the model continues with what it was about to write — the command — as
the value of `i`, then fills `command` with `x` / `true` / `echo test`; sometimes it re-emits
the whole intended call (`bash<arg_key>command</arg_key><arg_value>grep …`) inside the value.
Then the tool runs the filler, returns nothing useful, and the model retries — omp's
identical-call loop interrupt fires.

CPU reproduction (served checkpoint's tokenizer, real plugin grammar): with properties
`(i, command, timeout)` the grammar accepts `i`-then-`command` and rejects `command`-then-`i`;
with `(command, timeout, i)` the reverse. Nested `<arg_key>` text inside a string value is
accepted either way (it is just string content).

## Measured on the live cells

| | flint-omp-v4 (177 of 300 at 09:33) | flint-opencode-v4 (300, closed) |
|---|---|---|
| bash tool schema order | `i`, `command`, `timeout` | `command`, `timeout`, `workdir`, `description` |
| bash calls | 11,548 | 14,758 |
| `i` first in the parsed arguments | 11,547 (100 %) | n/a |
| command in the metadata key, filler in `command` | 856 (7.4 %) in 52 sessions | 10 (0.07 %) in 10 sessions |
| sessions with `<arg_key>` re-emitted inside a value | 83 / 177 (47 %) | 0 / 300 |
| sessions with omp's loop interrupt | 36 | n/a |
| empties | 3: django-15814 (12 swapped calls, 5 interrupts), astropy-7746 (0 swapped, 10 interrupts — a genuine identical-call loop), django-11019 (wall, 1 leak) | 4 |

Worst sessions: django-15695 (114 swapped calls, 116 interrupts, still produced a patch),
django-13757 (103), django-16046 (73), django-12125 (59). Most swapped sessions still end with a
patch; the cost is turns, walls and the two empties, not silence.

## Fix (branch `harness/flint-grammar-key-order`, `0ff8976`, cycle-boundary — plugin is baked into the serving image)

`_as_strict` moves `TRAILING_KEYS = (i, description, intent, explanation, reason, summary)` to the
end of `properties`; required-ness and `any_order=False` unchanged. opencode's order is already
compliant and is untouched. Unit test `scripts/eval/test_flint_tool_grammar_key_order.py` (6/6):
omp `command`→`i` accepted, `command` alone rejected (`i` still required), old forced order
rejected; opencode `command`→`description` accepted, `command` alone rejected.

The running flint:omp cell stays under the old grammar (mixing conditions inside a cell is the
quarantine rule's trigger); it is labelled **strict-ordered** in the bake-off table. A re-roll
under the fix is a GPU-bundle candidate at the boundary, user's call.

Not affected: every other preset (the strict grammar is the `flint` parser trio only); R9700
does not serve Flint under a grammar. Portable lesson for any ordered-schema grammar
(XGrammar strict / structural tags): put the model's natural first argument first in the schema,
or audit the parsed-argument key order of the first cell — 100 % `i`-first was the tell.

## Repro

```
FLINT_STRICT_TOOLS=1 CUDA_VISIBLE_DEVICES= python scripts/eval/test_flint_tool_grammar_key_order.py   # serving env
python3 - <<'PY'   # census: key order of every omp bash call
import json,glob,collections
o=collections.Counter()
for f in glob.glob("evals/swebench/runs/flint-omp-v4/sessions/*/.omp/agent/sessions/*/*.jsonl"):
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        for p in (r.get("message") or {}).get("content") or []:
            if isinstance(p,dict) and p.get("type")=="toolCall" and p.get("name")=="bash": o[tuple(p["arguments"])]+=1
print(o.most_common(5))
PY
```
