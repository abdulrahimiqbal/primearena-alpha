from __future__ import annotations

from typing import Iterable

import sympy as sp


def diameter(tup: Iterable[int]) -> int:
    vals = sorted(int(x) for x in tup)
    if not vals:
        return 0
    return int(vals[-1] - vals[0])


def is_admissible(tup: Iterable[int], k: int | None = None) -> tuple[bool, int]:
    vals = sorted(int(x) for x in tup)
    if k is None:
        k = len(vals)
    if len(vals) != int(k) or len(set(vals)) != len(vals):
        return False, diameter(vals)
    if vals and vals[0] != 0:
        vals = [x - vals[0] for x in vals]
    for p in list(sp.primerange(2, int(k) + 1)):
        residues = {int(x % p) for x in vals}
        if len(residues) == p:
            return False, diameter(vals)
    return True, diameter(vals)
