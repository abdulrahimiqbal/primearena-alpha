# Science Report: science_003

Terminal state: **(3) CERTIFICATION FAILURE**.

The rebuilt harness fixed the science_002 wheel-30 bug, but certification did not pass. C3 promoted absolute-position programs on a disjoint-range real-vs-real control, and C5 promoted a near-perfect `mod 7` support separator against the stated wheel-30 skeleton null. Per the prompt, science halted before the ladder and tails.

T1 flags: none.

## Task 0 Autopsy

cmd: `.venv/bin/python - <<PY [Task 0 legacy WheelNull(30) autopsy]`

Artifact: `runs/science_003/wheelnull30_autopsy.json`

| measurement | legacy fake | real primes |
|---|---:|---:|
| fraction divisible by 2 | 0.000000 | 0.000000 |
| fraction divisible by 3 | 0.339500 | 0.000000 |
| fraction divisible by 5 | 0.000000 | 0.000000 |
| mean marked positions/window | 33.794 | 33.769 |

Position semantics check: pass. Real and fake windows both used absolute `integer_values` in-window.

Root cause: generator defect. The responsible code path is legacy `leadengine/nulls.py` in `WheelNull.sample_like`: it calls `wheel_iid_no_pair_bias_null` with `q: 10` and `wheel_mod_q: True` while claiming `wheel=30`. That constrains support to numbers coprime to 10, not coprime to 30, so multiples of 3 leak into fake positions.

## Rebuild

Implemented before certification:

- `WindowBuilder` in `leadengine/datasets.py`; `PrimeWindowDataset` and rebuilt nulls route through it.
- `SkeletonResampleNull(wheel=30)` in `leadengine/nulls.py`; it preserves donor prime counts and samples uniformly from in-window positions coprime to 30.
- Legacy `WheelNull` retained and marked `SUPERSEDED`.
- `GeneratorConstraint` in `leadengine/absorb.py`; support-level constraints are handled by generator nulls, not by exponential tilting.
- Search screening now uses a fixed 1000-window fitness subsample, program feature memoization, top-10 full logistic scoring/permutation, and logs programs/sec.

## Certification

cmd: `.venv/bin/python - <<PY [science_003 certification early-failure controls]`

Artifact: `runs/science_003/certification_early_failure.json`

The full requested C3-C5 budget was 5000, but certification failed in seeded programs within budget 100. I stopped at the early terminal failure rather than spending more compute on a state that was already invalid.

| control | result | key numbers |
|---|---|---|
| C1 shared construction | PASS | real builder `WindowBuilder`; null builder `WindowBuilder` |
| C2 support contract | PASS | support_ok=true; count KS p=0.902691; real mean count=38.940; null mean count=39.074 |
| C3 real-vs-real disjoint ranges | FAIL | seed 0 and 1 promoted; max OOD AUC=1.000000; p=0.000999 |
| C3 real-vs-real even/odd index | PASS at reduced budget | max OOD AUC=0.575407; no promotion |
| C4 null-vs-null | PASS at reduced budget | max OOD AUC=0.568081; no promotion |
| C5 no trivial separator | FAIL | `hist(mod(positions(w),7),7)`; OOD AUC=0.997957/0.996596; val AUC=0.998892/0.997480 |

C3 disjoint-range failure programs:

| seed | program | val AUC | OOD AUC | p |
|---:|---|---:|---:|---:|
| 0 | `concat(scalar_vec(mean(positions(w))),hist(mod(mod(gaps(positions(w)),4),9),6))` | 1.000000 | 1.000000 | 0.000999 |
| 1 | `scalar_vec(mean(positions(w)))` | 1.000000 | 1.000000 | 0.000999 |

Interpretation: the DSL can encode absolute height/location through `positions(w)`. A disjoint-range real-vs-real split is therefore separable by plumbing/location, not by prime-vs-null arithmetic. That is harness leakage under C3.

C5 interpretation: `SkeletonResampleNull(wheel=30)` is now correct for wheel 30, but the full DSL includes `q=7` residue histograms. Real primes above 7 put no mass in residue 0 mod 7, while the wheel-30 skeleton null does. This is an exact support separator and fails the no-trivial-separator certification.

Search throughput red flag: reduced certification runs logged about 21-36 programs/sec on the fixed 1000-window fitness subsample, below the W3 target of >=50 programs/sec.

## Ladder

Skipped by rule. Certification failed before science. No fixed-point or survivor claim is made.

## Tails

Skipped by rule. T-A and T-C run only after the ladder reaches a terminal science state.

## Claims Revised

| prior claim | science_003 revision |
|---|---|
| science_001 Workstream B q=10-only fixed point | Still void outside the q=10 family; additionally, all WheelNull(30)-based absolute AUC magnitudes are contaminated by the legacy multiples-of-3 support leak. |
| science_001 Workstream A effect magnitudes against WheelNull(30) | Provisional/contaminated in magnitude. Scaling-shape claims require remeasurement after certification. |
| science_001 primearena calibration anchor | Still stands as a separate generator path, not this leadengine `WheelNull` path. |
| science_002 Stage-1 residue histogram AUC=1.0 | Reinterpreted as a successful discovery of the legacy wheel support bug, not arithmetic structure. |
| science_002 PI-prereg and sequential GUE instrumentation | Code remains available, but no new A'/C' science was executed in science_003. |

## Audit

S1 quick command: `.venv/bin/python -m pytest -m 'not slow'`

```text
22 passed, 11 deselected in 63.15s
```

Master calibration suite was not run after certification failure. The slow master gate still depends on the same harness/search behavior that C3 invalidated; running the ladder-facing master suite would not rescue the terminal certification state.

S2 command: `git diff --stat science_002`

```text
 leadengine/absorb.py   |  19 ++++
 leadengine/datasets.py |  58 ++++++++----
 leadengine/nulls.py    |  92 +++++++++++++++++-
 leadengine/search.py   | 246 ++++++++++++++++++++++++++++++++++++++++++-------
```

S2 is informative against the `science_002` tag. Current source changes excluding tests are 363 insertions and 52 deletions, below the W4 soft 600-line cap. Report files and run artifacts are additional deliverables.

S3 invariant audit:

- I1 held for promotion thresholds, permutation p threshold, and final logistic scoring. Search fitness screening changed under W3; final promotion still uses OOD AUC >= 0.60 and permutation p < 0.001.
- I2 held: numeric tables cite command/artifact paths.
- I3 prereg target scoring was not exercised because the run halted during certification.
- I4 held for completed searches: distinct program counts and shapes are in `certification_early_failure.json`.
- I5 held: decisions are logged in `runs/science_003/DECISIONS.md`.
- I6 held for science_003 artifacts; science_001/science_002 artifacts were not overwritten.

Final statement: science_003 reached terminal state **CERTIFICATION FAILURE**. No arithmetic claim is made beyond the Task-0 generator autopsy and the certification failures described above.
