from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import Window
from .sieve import primes_in


@dataclass(frozen=True)
class PrimeWindowDataset:
    n_min: int
    n_max: int
    window_len: int

    name: str = "prime_windows"
    domain: str = "primes"

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        windows: list[Window] = []
        for _ in range(max(0, int(n_windows))):
            start = int(rng.integers(int(self.n_min), int(self.n_max) + 1))
            values = np.arange(start, start + int(self.window_len), dtype=np.int64)
            indicator = np.zeros(int(self.window_len), dtype=np.int8)
            primes = primes_in(start, start + int(self.window_len))
            if primes.size:
                indicator[(primes - start).astype(np.int64)] = 1
            windows.append(
                Window(
                    values=indicator,
                    start=start,
                    meta={
                        "domain": self.domain,
                        "integer_values": values,
                        "source": "leadengine.sieve.primes_in",
                    },
                )
            )
        return windows

    def scale_of(self, w: Window) -> int:
        return int(math.log10(max(int(w.start), 1)))
