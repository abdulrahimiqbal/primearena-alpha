import math

from verifiers.admissible import is_admissible
from verifiers.gapmerit import verify_gap
from verifiers.jacobsthal import exhaustive_best, verify_interval


def test_admissible_known_tuples():
    for k, diam, tup in [
        (3, 6, [0, 2, 6]),
        (4, 8, [0, 2, 6, 8]),
        (5, 12, [0, 4, 6, 10, 12]),
        (10, 32, [0, 2, 6, 12, 14, 20, 24, 26, 30, 32]),
    ]:
        ok, got = is_admissible(tup, k)
        assert ok and got == diam


def test_admissible_corruptions_fail():
    for tup in ([0, 1], [0, 2, 4], [0, 2, 6, 8, 12, 14]):
        ok, _ = is_admissible(tup, len(tup))
        assert not ok


def test_jacobsthal_tiny_exact_intervals():
    known_h = {2: 4, 3: 6, 4: 10}
    for n, h in known_h.items():
        a, m = exhaustive_best(n)
        assert m + 1 == h
        ok, got, witnesses = verify_interval(a, m, n)
        assert ok and got == m and len(witnesses) == m


def test_jacobsthal_corruption_fails():
    assert verify_interval(0, 1, 2)[0] is False


def test_gapmerit_known_gaps():
    for p, g in [(7, 4), (113, 14), (523, 18)]:
        ok, merit, witnesses = verify_gap(p, g)
        assert ok
        assert merit == g / math.log(p)
        assert len(witnesses) == g - 1


def test_gapmerit_fake_gap_fails():
    ok, _, _ = verify_gap(113, 18)
    assert not ok
