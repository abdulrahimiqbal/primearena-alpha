from __future__ import annotations

from math import isqrt
from typing import Iterable, List


def small_prime_sieve(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [i for i, v in enumerate(sieve) if v]


_SMALL_PRIMES_FOR_TRIAL = small_prime_sieve(97)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for unsigned 64-bit integers.

    This is sufficient for the default PrimeArena ranges and remains exact up to 2**64.
    """
    if n < 2:
        return False
    for p in _SMALL_PRIMES_FOR_TRIAL:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n - 1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for testing 64-bit integers.
    # See Jim Sinclair / deterministic variants widely used for n < 2**64.
    bases = [2, 3, 5, 7, 11, 13, 17]
    if n >= 341_550_071_728_321:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    """Return the first prime strictly larger than n."""
    if n < 2:
        return 2
    c = n + 1
    if c <= 2:
        return 2
    if c % 2 == 0:
        c += 1
    while not is_prime(c):
        c += 2
    return c


def passes_wheel(n: int, primes: Iterable[int]) -> bool:
    if n < 2:
        return False
    for p in primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    return True
