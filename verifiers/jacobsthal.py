from __future__ import annotations

import math

import sympy as sp


def primes_upto_n(n: int) -> list[int]:
    return [int(sp.prime(i)) for i in range(1, int(n) + 1)]


def verify_interval(a: int, m: int, n: int) -> tuple[bool, int, dict[int, int]]:
    primes = primes_upto_n(int(n))
    witnesses: dict[int, int] = {}
    for x in range(int(a) + 1, int(a) + int(m) + 1):
        witness = 0
        for p in primes:
            if x % p == 0:
                witness = p
                break
        if witness == 0:
            return False, int(m), witnesses
        witnesses[int(x)] = int(witness)
    return True, int(m), witnesses


def exhaustive_best(n: int, period_cap: int | None = None) -> tuple[int, int]:
    primes = primes_upto_n(int(n))
    period = math.prod(primes)
    if period_cap is not None:
        period = min(period, int(period_cap))
    best_a = 0
    best_m = 0
    run_start = None
    run_len = 0
    for x in range(1, period + 1):
        covered = any(x % p == 0 for p in primes)
        if covered:
            if run_start is None:
                run_start = x - 1
            run_len += 1
            if run_len > best_m:
                best_a, best_m = int(run_start), int(run_len)
        else:
            run_start = None
            run_len = 0
    return best_a, best_m
