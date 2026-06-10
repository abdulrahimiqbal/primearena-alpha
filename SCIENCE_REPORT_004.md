# Science Report: science_004

Terminal state: **(3) CERTIFICATION FAILURE**.

Certification v2 fixed the paired-construction and mod-7 triage issues from science_003, but the revised controls still failed. C3v2 promoted programs on even/odd and random real-vs-real splits, and C4 promoted a program on null-vs-null at budget 500. Because these promotions occur inside the requested budget-5000 search, running the longer budget cannot erase the failure. Master gate v2, ladder, and tails were correctly skipped.

T1 flags: none.

## Certification v2

cmd: `.venv/bin/python - <<PY [science_004 certification_v2 budget=500]`

Artifact: `runs/science_004/certification_v2.json`

The disjoint-range real-vs-real control is retired: `positions(w)` intentionally carries absolute location, so `mean(positions(w))` separates disjoint ranges by construction. The v2 controls use even/odd window index and random 50/50 splits.

| control | result | key numbers |
|---|---|---|
| C1 shared construction | PASS | real builder `WindowBuilder`; null builder `WindowBuilder` |
| C2 wheel 30 support/count | PASS | KS p=1.000000; real mean count=39.315; null mean count=39.315 |
| C5v2 location baseline, wheel 30 | PASS | AUC=0.483602 |
| C5v2 triage, wheel 30 | PRE-LADDER RUNG | mod-7 support; wheel 30 -> 210 |
| C2 wheel 210 support/count | PASS | KS p=1.000000; real mean count=39.171; null mean count=39.171 |
| C5v2 location baseline, wheel 210 | PASS | AUC=0.498788 |
| C3v2 even/odd real-vs-real | FAIL | 2/2 seeds promoted; max OOD AUC=0.608376 |
| C3v2 random split 0 | FAIL | 1/2 seeds promoted; max OOD AUC=0.604297 |
| C3v2 random split 1 | FAIL | 1/2 seeds promoted; max OOD AUC=0.619470 |
| C4 null-vs-null wheel 210 | FAIL | 1/2 seeds promoted; max OOD AUC=0.616555 |
| C5v2 post-wheel-210 smoke | PASS at budget 100 | no promotion; max OOD AUC=0.540345 |

C5v2 triage detail:

| seed | program | val AUC | OOD AUC | p | triage |
|---:|---|---:|---:|---:|---|
| 0 | `pair_hist(pairs(positions(w),3),7)` | 1.000000 | 0.999988 | 0.000999 | PRE-LADDER RUNG |
| 1 | `hist(mod(positions(w),7),7)` | 0.999428 | 0.996723 | 0.000999 | PRE-LADDER RUNG |

Post-triage support table showed no mod-q support gaps for q in `{2,3,4,5,6,7,8,9,10,12,30}` under wheel 210. The count contract also held exactly in the sampled pairs because null twins inherit each real window's count.

C3v2/C4 failure details:

| control | seed | program | val AUC | OOD AUC | p |
|---|---:|---|---:|---:|---:|
| even/odd | 0 | `pair_hist(pairs(gaps(gaps(mod(mod(positions(w),10),12))),2),30)` | 0.725616 | 0.604362 | 0.000999 |
| even/odd | 1 | `concat(pair_hist(pairs(positions(w),2),30),pair_hist(pairs(gaps(gaps(mod(positions(w),30))),3),9))` | 0.716606 | 0.608376 | 0.000999 |
| random 0 | 0 | `pair_hist(pairs(gaps(gaps(mod(gaps(positions(w)),9))),3),30)` | 0.847804 | 0.604297 | 0.000999 |
| random 1 | 0 | `pair_hist(pairs(gaps(gaps(gaps(mod(positions(w),9)))),3),30)` | 0.876246 | 0.619470 | 0.000999 |
| null-vs-null | 0 | `pair_hist(pairs(gaps(gaps(gaps(mod(positions(w),7)))),1),30)` | 0.859584 | 0.616555 | 0.000999 |

Interpretation: the harness still produces control promotions after paired construction and after the mod-7 support rung is constrained away. The failure is not the retired disjoint-range location artifact, and it is not the wheel-30/mod-7 support issue. Under the run rules, this is terminal certification failure and no arithmetic ladder claim is made.

Throughput red flag: C3v2/C4 certification searches logged 11.6-16.3 programs/sec, below the revised acceptable target of 20 programs/sec.

## Master Gate v2

Not created or run. Certification v2 was not green, so section 3 was not reached.

## Ladder

Skipped by rule. No fixed-point, survivor, or ladder reconstruction claim is made.

## Tails

Skipped by rule. T-A and T-C run only after a certified ladder terminal state.

## Claims Revised

| prior claim | science_004 revision |
|---|---|
| science_003 C3 disjoint-range failure | Retired as a control-spec error; absolute location is intentionally in the DSL. |
| science_003 C5 mod-7 separator | Reclassified as PRE-LADDER RUNG, not a harness defect. Wheel 30 -> 210 removes the support gap for the DSL q-set. |
| science_003 certification failure | Still unresolved: v2 controls fail via even/odd, random real-vs-real, and null-vs-null promotions. |
| science_001/002 WheelNull(30) AUC magnitudes | Still contaminated/stale; no decontaminated replacement was produced because certification failed. |
| science_002 PI-prereg and GUE instrumentation | Still available but not exercised in science_004. |

## Audit

S1 quick command: `.venv/bin/python -m pytest -m 'not slow'`

```text
24 passed, 11 deselected in 25.78s
```

Master gate v2 was not run because it was not created after certification failure.

S2 command: `git diff --stat science_003`

```text
 leadengine/nulls.py        | 55 ++++++++++++++++++++++++++--------------------
 leadengine/search.py       | 33 ++++++++++++++++++----------
 tests/test_null_harness.py | 20 +++++++++++++++++
```

S2 is informative against the `science_003` tag. Source changes excluding tests are under the W4 soft cap of 500 changed lines.

S3 invariant audit:

- I1 held: promotion thresholds and permutation threshold were not weakened. Search sampling was paired under W1; permutation p-value calculation was optimized but remains an explicit label-permutation test.
- I2 held: numeric certification tables cite the command and JSON artifact.
- I3 prereg target scoring was not exercised because certification failed.
- I4 held: certification search artifacts include distinct program counts and shapes.
- I5 held: decisions are logged in `runs/science_004/DECISIONS.md`.
- I6 held: science_004 artifacts were written under `runs/science_004/`; prior run artifacts were not overwritten.

Final statement: science_004 reached terminal state **CERTIFICATION FAILURE**. The main result is that paired construction and wheel-210 support triage are insufficient to prevent control promotions in C3v2 and C4.
