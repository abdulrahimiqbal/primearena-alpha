# Science Report: science_002

Stage-1 result: the full-DSL generator was rebuilt, but the Phase-3 pair rediscovery gate is NOT green. Once the q=10-only eligible-set defect is removed, simple residue histogram programs separate `WheelNull(30)` from primes with train_auc=1.0 and val_auc=1.0 and outrank the q=10 pair family. Therefore no Stage-2 B'/A'/C' science claim is made in this report.

T1 flags: none.

## Stage-1 Instrument

Implemented instrumentation changes:

- `leadengine/dsl.py`: random typed-tree growth and typed-subtree mutation over the DSL primitives, max depth 8 and max complexity 20, plus constant-stripped shape descriptions.
- `leadengine/search.py`: removed prime/zero/FF hard-coded eligible sets, evaluates semantically distinct `describe()` programs, and returns per-invocation search logs: distinct programs evaluated, distinct constant-stripped shapes, sampled fraction (`open`), and generation attempts.
- `tests/test_dsl_search.py::test_search_rediscovers_pair_bias`: strengthened to require budget=20000, seeds `{0,1}`, and at least 2000 distinct programs.

cmd: `.venv/bin/python - <<'PY' [stage1 gate diagnostic, budget=400, seed=0]`

Artifact: `runs/science_002/stage1_gate_diagnostic.json`

| metric | value |
|---|---:|
| distinct programs evaluated | 400 |
| distinct program shapes | 163 |
| sampled fraction | open |
| generation attempts | 1016 |
| best program | `hist(positions(w),6)` |
| best OOD AUC | 1.000000 |
| best permutation p | 0.000999 |

Key rows from the diagnostic:

| program | train AUC | val AUC | complexity |
|---|---:|---:|---:|
| `hist(positions(w),6)` | 1.000000 | 1.000000 | 3 |
| `hist(mod(positions(w),30),30)` | 1.000000 | 1.000000 | 4 |
| `pair_hist(pairs(mod(positions(w),10),1),10)` | 0.696866 | 0.640793 | 5 |
| `normalize(pair_hist(pairs(mod(positions(w),10),1),10))` | 0.717283 | 0.640793 | 6 |

Gate outcome: FAIL/BLOCKED. The q=10 pair structure is reachable and separating, but it is not the selected best structure in the full DSL. Because seeded residue histogram programs already have val_auc=1.0, increasing the budget cannot make the original best-program pair gate pass without changing the null or narrowing the DSL space. I did not run the exact 20000 x 2 slow gate to completion; a budget-2000 run was attempted and stopped after about four minutes without output.

Claim tier: REDISCOVERED(modular residue leakage under the current `WheelNull(30)` implementation), BLOCKED for the intended full-DSL pair rediscovery gate.

## Science_001 Claims Revised

| science_001 claim | revision in science_002 |
|---|---|
| B fixed-point negative after q=10 pair absorption | VOID outside the q=10 pair-family scope. The rebuilt search first finds stronger residue histogram separation, so B' cannot be claimed. |
| A decade-11 pass/fail | Not re-scored in this run because Stage 1 was not green. Code now supports PI preregistration and k=3 target scoring for a future A'. |
| C tilted GUE rung blocked by simultaneous constraints | Not re-run. Code now builds caller-side sequential support precheck/ESS logging with 8-bin histograms for a future C'. |

## Workstreams

### Workstream B'

Skipped. Stage 1 did not pass, and the attempted B-prime-lite exact complexity<=8 enumeration did not complete. See `runs/science_002/BLOCKERS.md`.

### Workstream A'

Not executed. The preregistration machinery was amended: new PI entries use `z * sqrt(fit_se^2 + sampling_se^2)`, and PI scoring averages k=3 target samples with seeds derived from the preregistration hash. Existing CI-style preregistrations remain readable.

### Workstream C'

Not executed. The zero runner was amended so the empirical tilted rung is built sequentially: mean+var, then `fhist(w,8)`, then `fhist(ratios(w),8)`, with support precheck and ESS logging per step. The caller requests GUE matrix size 1024 and at least 4000 candidate windows. Absorption math was not modified.

## Audit

S1 command: `.venv/bin/python -m pytest -m 'not slow'`

```text
19 passed, 11 deselected in 37.70s
```

S1 CLI import checks:

```text
.venv/bin/python -m leadengine.run_zeros --help
.venv/bin/python -m leadengine.run_absorption --help
.venv/bin/python -m leadengine.run_los_scaling --help
```

All three help commands returned successfully.

S2 command: `git diff --stat science_001 -- leadengine/scoring.py leadengine/search.py leadengine/absorb.py leadengine/prereg.py leadengine/cards.py leadengine/dsl.py leadengine/run_absorption.py leadengine/run_los_scaling.py leadengine/run_zeros.py`

```text
 leadengine/dsl.py             | 176 +++++++++++++++++++++++++++----------
 leadengine/prereg.py          |  45 +++++++++-
 leadengine/run_absorption.py  |  14 ++-
 leadengine/run_los_scaling.py |  21 ++++-
 leadengine/run_zeros.py       |  84 +++++++++++++++---
 leadengine/search.py          | 197 +++++++++++++++++++++++++++++++++---------
 6 files changed, 427 insertions(+), 110 deletions(-)
```

S2 is informative against the `science_001` tag. Report-level red flag: source insertions excluding tests are 427, above the approximate 400-line cap.

S3 invariant audit:

- I1 held for thresholds, promotion criteria, permutation testing, scoring thresholds, absorption math, and prereg hash-seeding intent. Search generation and caller-side absorption schedule were changed under W1/W3.
- I2 held for reported numeric artifacts: the Stage-1 diagnostic includes a command and JSON artifact.
- I3 target-scoring prereg discipline was not exercised because A' was not run.
- I4 held for completed search diagnostics: distinct evaluated counts are reported.
- I5 held: decisions are logged in `runs/science_002/DECISIONS.md`.
- I6 held for run artifacts created in `runs/science_002/`; no science_001 artifacts were overwritten.

Final status: science_002 is a Stage-1 negative/instrument run, not a completed Stage-2 science run.
