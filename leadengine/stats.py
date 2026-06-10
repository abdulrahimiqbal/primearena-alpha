from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import Window


def _reduced_residues(q: int) -> list[int]:
    return [r for r in range(int(q)) if math.gcd(r, int(q)) == 1]


@dataclass(frozen=True)
class ResiduePairCount:
    q: int
    reduced: bool = True

    @property
    def name(self) -> str:
        return f"residue_pair_count_mod{int(self.q)}"

    def __call__(self, w: Window) -> np.ndarray:
        q = int(self.q)
        marked = np.flatnonzero(np.asarray(w.values) > 0)
        if "integer_values" in w.meta:
            ints = np.asarray(w.meta["integer_values"], dtype=np.int64)[marked]
        else:
            ints = int(w.start) + marked

        if self.reduced:
            residues = _reduced_residues(q)
            residue_to_idx = {r: i for i, r in enumerate(residues)}
            encoded = [residue_to_idx[int(x % q)] for x in ints if int(x % q) in residue_to_idx]
            dim = max(1, len(residues))
        else:
            encoded = [int(x % q) for x in ints]
            dim = q

        out = np.zeros(dim * dim, dtype=np.float32)
        if len(encoded) <= 1:
            return out
        arr = np.asarray(encoded, dtype=np.int64)
        pair_ids = arr[:-1] * dim + arr[1:]
        counts = np.bincount(pair_ids, minlength=dim * dim).astype(np.float32)
        total = float(counts.sum())
        return counts / max(total, 1.0)

    def complexity(self) -> float:
        dim = len(_reduced_residues(int(self.q))) if self.reduced else int(self.q)
        return float(2.0 + dim / 10.0)

    def describe(self) -> str:
        space = "reduced residue" if self.reduced else "full residue"
        return f"normalized histogram of consecutive prime {space} pairs modulo {int(self.q)}"
