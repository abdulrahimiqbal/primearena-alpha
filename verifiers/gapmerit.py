from __future__ import annotations

import math

import sympy as sp


def _factor_witness(n: int) -> int:
    if n < 2:
        return 1
    f = sp.factorint(int(n), limit=100000)
    if f:
        return int(min(f))
    return 0


def verify_gap(p: int, g: int) -> tuple[bool, float, dict[int, int]]:
    p = int(p)
    g = int(g)
    if g <= 0 or p < 2:
        return False, 0.0, {}
    if not sp.isprime(p) or not sp.isprime(p + g):
        return False, float(g / math.log(max(p, 3))), {}
    witnesses: dict[int, int] = {}
    for x in range(p + 1, p + g):
        if sp.isprime(x):
            return False, float(g / math.log(p)), witnesses
        w = _factor_witness(x)
        if w <= 1 or x % w != 0:
            return False, float(g / math.log(p)), witnesses
        witnesses[int(x)] = int(w)
    return True, float(g / math.log(p)), witnesses
