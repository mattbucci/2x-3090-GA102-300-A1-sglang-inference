# qwopus38 × omp lane close (SWE-bench Lite, isolated harness, run `qwopus38-omp-v4`)

Qwopus3.8-27B-Coder-Flash-AWQ (in-house GPTQ→CT→AWQ, served with the qwen38 preset block, thinking on, 262K window,
32K output budget, oh-my-pi 18.12.2 scaffold, `--network none` + bridge). Rolled 2026-11-03 10:39 → 2026-11-06 11:25
(300/300, 1 wall seen early; 9 infra-failure instances re-rolled 11:26–13:56). The Phase-4.5 leak audit FAILED on one
false EXPOSED (sympy-20154: `pip download … 2>/dev/null && ls; pip show sympy` — the download failed silently under
network=none and the sibling `pip show` metadata read as fetched content) and the cell sat unscored until the loop session
came back 2026-11-08; classifier fixed (commit 2490d39, historical cells unchanged at EXPOSED 0), audit re-run PASS
(`leak-audit-final.log`: attempted 148/300, EXPOSED 0), scored 2026-11-08 11:06–13:33 (`swebench==4.1.0`, one worker).

## Result

| cell | resolved / 300 | rate |
|---|---|---|
| **qwopus38 × omp** | **190** | **63.3 %** (2 empty patches: sphinx-8474, sympy-19254; 0 errors) |
| qwen38 × opencode (reference) | 196 | 65.3 % |
| flint × omp | 186 | 62.0 % |

Paired McNemar (same 300 instances): vs qwen38 opencode **p = 0.45** (qwopus-only 19, qwen38-only 25) — a tie; vs flint
omp p = 0.66 (25 / 21) — a tie. Scaffold differs from the qwen38 cell (omp vs opencode); on Flint the two scaffolds tied
(182 vs 186, p = 0.57), so the comparison is read as model-level within noise.

## Per-repo (qwopus38 omp / qwen38 opencode / instances)

| repo | qwopus38 omp | qwen38 opencode |
|---|---|---|
| astropy | 4/6 | 4/6 |
| django | 78/114 | 84/114 |
| matplotlib | 15/23 | 15/23 |
| mwaskom | 2/4 | 2/4 |
| pallets | 1/3 | 1/3 |
| psf | 1/6 | 6/6 |
| pydata | 2/5 | 2/5 |
| pylint-dev | 3/6 | 4/6 |
| pytest-dev | 11/17 | 11/17 |
| scikit-learn | 16/23 | 16/23 |
| sphinx-doc | 9/16 | 10/16 |
| sympy | 48/77 | 41/77 |

The one systematic gap is psf/requests (1/6 vs 6/6); qwopus38 leads on sympy (48 vs 41). Both within the paired noise.

## Files
`scores-docker-summary.json`, `leak-audit.json` (scanned: no per-cycle key), `leak-audit-final.log`,
`paired-qwopus38-omp-vs-qwen38-opencode.json`, `paired-qwopus38-omp-vs-flint-omp.json`. Cell JSON:
`benchmarks/quality/bakeoff-qwopus38-omp.json`. Issue #21 (qwopus38 opencode cell still queued).
