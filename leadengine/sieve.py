from __future__ import annotations

import math

import numpy as np


def _small_primes(limit: int) -> np.ndarray:
    limit = int(limit)
    if limit < 2:
        return np.empty(0, dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = False
    return np.flatnonzero(sieve).astype(np.int64)


def primes_in(lo: int, hi: int) -> np.ndarray:
    """Return primes p with lo <= p < hi using a segmented Eratosthenes sieve."""

    lo = int(lo)
    hi = int(hi)
    if hi <= lo or hi <= 2:
        return np.empty(0, dtype=np.int64)
    lo_eff = max(lo, 2)
    span = hi - lo_eff
    if span > 10_000_000:
        raise ValueError("primes_in spans above 1e7 should be chunked by the caller")

    is_prime = np.ones(span, dtype=bool)
    base_primes = _small_primes(int(math.isqrt(hi - 1)) + 1)
    for p_obj in base_primes:
        p = int(p_obj)
        first = max(p * p, ((lo_eff + p - 1) // p) * p)
        if first < hi:
            is_prime[first - lo_eff :: p] = False
    return (np.flatnonzero(is_prime).astype(np.int64) + lo_eff).astype(np.int64)
