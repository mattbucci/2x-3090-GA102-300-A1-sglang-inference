# Leak-audit classifier: three false EXPOSED on the flint opencode lane close (2026-10-27)

`audit_leakage.py --require-isolation` on `flint-opencode-v4` (300/300, isolation proven on
every instance: network=none, refs stripped, mounts clean, git reinit) read **EXPOSED=3** and
FAIL — the Phase-4.5 gate would have exited the flint cycle unscored at the omp lane close
(the 2026-10-21 incident class). Every one of the three was a transcript-reading defect:

| instance | what the model ran | why the classifier said ok=True | fix |
|---|---|---|---|
| django-10914 | `curl -s …api.github.com… \| python -c "json.load(sys.stdin)…" 2>&1 \| head` | `head` cut the py3.6 traceback at ten lines, before the final `JSONDecodeError: Expecting value` line the empty-stdin rule keys on | `JSON_EMPTY_RE` also accepts the `raw_decode` frame / `raise JSONDecodeError("Expecting value"` source line |
| django-13710 | `pip show django…; python -c "print(django.__version__…)"; curl -s … \| sed -n '/class InlineModelAdmin/,/def media/p' \| head -80` | the shared output (pip show + version lines) is "content" for the curl segment | `filtered_output_empty`: a grep / `sed -n /x/p` stage behind the net tool whose pattern never occurs in the output means that pipeline printed nothing (grep `-c/-v/-l/-L/-q` excluded) |
| sympy-23191 | `python - <<'PYEOF' … PYEOF` then a 15-line `python -c "…print('--- curl')…"` importing `sympy.vector.Curl` | heredoc / `-c` bodies were split into shell list segments; `\bcurl\b` under `re.I` matched `Curl` and `'--- curl'` | `strip_heredocs` blanks heredoc and `python -c` bodies before segmenting (`PY_FETCH_RE` still reads the whole text; new `PY_SHELLOUT_RE` covers `subprocess`/`os.system` curl inside such bodies for bash tools); `curl|wget` case-sensitive |

Unit tests: `scripts/eval/test_audit_leakage_segments.py` 68/68 (+8 cases incl. controls that keep
a real fetch exposed), `test_audit_leakage_omp.py` 12/12. First cut of `HEREDOC_RE` used
`(?:.*?\n)*?` under `re.S` — catastrophic backtracking on an unterminated 5,000-line body
(the sweep sat on one cell for 10 min); the line-wise form runs in 1 ms.

## Full sweep, HEAD classifier vs fixed, per instance (14 cells)

| cell | attempted | exposed | note |
|---|---|---|---|
| flint-opencode-v4 | 208 → 208 | **3 → 0** | the three above; gate PASS |
| qwen38-opencode-v2-netopen | 187 → 187 | 186 → 186 | netopen regression reference unchanged |
| qwen38-opencode-dcp-v2-netopen | 183 → 183 | 182 → 182 | unchanged |
| qwen38-opencode-v3 | 159 → 159 | 0 → 0 | django-16139 loses a spurious `web OTHER` (`curl = reverse(…)` python variable in a heredoc) |
| qwen38-opencode-dcp-v4 | 158 → 158 | 0 → 0 | sympy-14308 loses a spurious `web OTHER` |
| qwen38-opencode-v4, -dcp-v3, little-coder(-rtk)-v4, prime-v4, dcode-v4-respsplit, qwen36(-dense)-opencode-v4 | = | 0 → 0 | unchanged |

Observation outside this change: `qwen38-opencode-v3`'s verdict flipped False→True between the
two runs because `iso_git_reinit` read 299 then 300 on identical logs — one instance's
re-init proof depends on `image_untracked()` seeing the rollout image, which the janitor
removes; a flaky external dependency for historical cells, to be made log-only.

Flint on the isolated harness: **69 % of instances attempted an upstream fetch** (qwen38
opencode v4: 52 %) and none succeeded; gold-overlap ≥80 % in 56/290 isolated patches
(median 0.0) — attempts are a model trait, exposure is a harness property.
