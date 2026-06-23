# Records Report: science_005

T1 flags: none.

Terminal state: **(2) NO RECORDS**.

Setup: science_004 was committed/tagged, and the distinguisher program was parked in `NOTES.md` with its known repair path.
Target A: fetched MIT narrow admissible tuple records through k=5000 and reproduced eight selected records with the independent admissibility verifier.
Target B: fetched OEIS/Hagedorn Jacobsthal primorial values; reproduced exact h(n) for n=2..8 and produced weaker verified lower bounds for n=9..13.
Target C: fetched prime-gap conventions; verified a small merit curve below p=20000, with no record-table-relevant gap.
Claims: no record claim files were written because no verified object strictly beat a fetched record.

## Sources

cmd: `.venv/bin/python - <<PY [records cache fetch/parse]`

Cache: `runs/science_005/records_cache/sources_and_seed_records.json`

| target | source | access date | use |
|---|---|---:|---|
| A | [MIT narrow admissible tuples](https://math.mit.edu/~primegaps/) | 2026-06-10 | current k-tuple diameters and tuple files |
| A | [Sutherland announcement/context](https://sbseminar.wordpress.com/2013/07/02/the-quest-for-narrow-admissible-tuples/) | 2026-06-10 | database provenance |
| B | [OEIS A048670](https://oeis.org/A048670) | 2026-06-10 | Jacobsthal primorial sequence and references |
| B | [Hagedorn PDF](https://hagedorn.pages.tcnj.edu/files/2022/08/Jacobsthal.pdf) | 2026-06-10 | exact h(n) table for n < 50 |
| C | [Prime Gap List Project](https://primegap-list-project.github.io/) | 2026-06-10 | successor project and tables |
| C | [pzktupel prime gaps](https://www.pzktupel.de/RecordGaps/gapmainpage.php) | 2026-06-10 | merit definition and PRP/proven endpoint conventions |
| C | [Prime Gap List about](https://primegap-list-project.github.io/about/) | 2026-06-10 | curation/convention context |

## Verifiers

cmd: `.venv/bin/python -m pytest tests/constructor/test_verifiers.py -q`

```text
6 passed in 1.25s
```

Verifier modules:

- `verifiers/admissible.py`: checks admissibility for primes p <= k and returns diameter.
- `verifiers/jacobsthal.py`: checks every integer in `(a, a+m]` has a witness divisor among the first n primes.
- `verifiers/gapmerit.py`: checks endpoints with `sympy.isprime`, verifies all interiors composite with factor witnesses, and computes merit as `g / ln(p)`.

## Target A

cmd: `.venv/bin/python - <<PY [constructor portfolio bounded seeds 0,1,2]`

Artifact: `runs/science_005/portfolio_results.json`

Objects evaluated: 104. Objects/sec: 668.59.

| k | fetched record diameter | best verified diameter | gap | relative gap | status |
|---:|---:|---:|---:|---:|---|
| 10 | 32 | 32 | 0 | 0.000000 | reproduced |
| 11 | 36 | 36 | 0 | 0.000000 | reproduced |
| 20 | 80 | 80 | 0 | 0.000000 | reproduced |
| 50 | 246 | 246 | 0 | 0.000000 | reproduced |
| 100 | 558 | 558 | 0 | 0.000000 | reproduced |
| 500 | 3594 | 3594 | 0 | 0.000000 | reproduced |
| 1000 | 7802 | 7802 | 0 | 0.000000 | reproduced |
| 3000 | 26606 | 26606 | 0 | 0.000000 | reproduced |

No improvement was found. These reproductions mostly validate verifier and source handling because the fetched record tuple itself was used as an allowed seed.

## Target B

cmd: `.venv/bin/python - <<PY [constructor portfolio bounded seeds 0,1,2]`

Objects evaluated: 12. Objects/sec: 0.95.

The verifier reports covered interval length `m`; the Hagedorn/OEIS value `h(n)` is `m+1` for these exact reproductions.

| n | fetched h(n) | verified implied h lower bound | gap to exact | status |
|---:|---:|---:|---:|---|
| 2 | 4 | 4 | 0 | reproduced |
| 3 | 6 | 6 | 0 | reproduced |
| 4 | 10 | 10 | 0 | reproduced |
| 5 | 14 | 14 | 0 | reproduced |
| 6 | 22 | 22 | 0 | reproduced |
| 7 | 26 | 26 | 0 | reproduced |
| 8 | 34 | 34 | 0 | reproduced |
| 9 | 40 | 34 | -6 | lower bound only |
| 10 | 46 | 34 | -12 | lower bound only |
| 11 | 58 | 34 | -24 | lower bound only |
| 12 | 66 | 34 | -32 | lower bound only |
| 13 | 74 | 34 | -40 | lower bound only |

No lower bound exceeded a fetched known value.

## Target C

cmd: `.venv/bin/python - <<PY [constructor portfolio bounded seeds 0,1,2]`

Objects evaluated: 866. Objects/sec: 8257.23.

Merit convention: `gap / ln(p)`, matching the fetched prime-gap table convention.

| p | gap | verified merit |
|---:|---:|---:|
| 19609 | 52 | 5.2612 |
| 1327 | 34 | 4.7283 |
| 15683 | 44 | 4.5547 |
| 16141 | 42 | 4.3348 |
| 19333 | 40 | 4.0529 |
| 9551 | 36 | 3.9282 |
| 12853 | 36 | 3.8050 |
| 14107 | 36 | 3.7679 |
| 8467 | 34 | 3.7594 |
| 15823 | 36 | 3.7232 |

Interpretation: this is a small sanity curve, not a competitive gap search. The best verified merit below p=20000 was 5.2612, far below record-table relevance.

## Ledgers

Decision ledger: `runs/science_005/DECISIONS.md`.

Blockers: `runs/science_005/BLOCKERS.md`.

Claims registered: none.

Total objects evaluated:

| target | objects |
|---|---:|
| A | 104 |
| B | 12 |
| C | 866 |

## Audit

S1 command: `.venv/bin/python -m pytest -m 'not slow'`

```text
30 passed, 11 deselected in 14.64s
```

S2 command: `git diff --stat science_004`

```text
 NOTES.md                                           |   14 +
 RECORDS_REPORT_005.md                              |  162 +
 runs/science_005/BLOCKERS.md                       |    5 +
 runs/science_005/DECISIONS.md                      |    7 +
 runs/science_005/portfolio_results.json            | 5123 ++++++++++++++++++++
 .../records_cache/mit_primegaps_rows_sample.json   | 4775 ++++++++++++++++++
 .../records_cache/sources_and_seed_records.json    | 4984 +++++++++++++++++++
 tests/constructor/test_verifiers.py                |   48 +
 verifiers/admissible.py                            |   27 +
 verifiers/gapmerit.py                              |   32 +
 verifiers/jacobsthal.py                            |   47 +
 11 files changed, 15224 insertions(+)
```

S2 is informative against the `science_004` tag.

V1-V4 attestation:

- V1 held: every reported object was checked by an independent verifier module; verifier unit tests passed before portfolio execution.
- V2 held: no record claims were made, so no claim files were required. Source records are cached with URLs and access dates.
- V3 held: reported numbers cite their regenerating command and seeds.
- V4 held: decisions and blockers are logged; no files under `runs/science_005/` were overwritten.

Final statement: no verified records were found. The deliverable is a verified reproduction-distance map for the bounded constructor portfolio.
