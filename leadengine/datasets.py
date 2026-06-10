from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import Window
from .sieve import primes_in


PositionMarker = Callable[[np.ndarray, np.random.Generator | None], np.ndarray]


@dataclass(frozen=True)
class WindowBuilder:
    domain: str = "primes"

    def build(
        self,
        start: int,
        length: int,
        marker: PositionMarker,
        rng: np.random.Generator | None = None,
        meta: dict | None = None,
    ) -> Window:
        values = np.arange(int(start), int(start) + int(length), dtype=np.int64)
        positions = np.asarray(marker(values, rng), dtype=np.int64)
        indicator = np.zeros(int(length), dtype=np.int8)
        if positions.size:
            offsets = positions - int(start)
            if np.any(offsets < 0) or np.any(offsets >= int(length)):
                raise ValueError("position marker returned out-of-window positions")
            indicator[offsets.astype(np.int64)] = 1
        payload = {"domain": self.domain, "integer_values": values, "window_builder": self.__class__.__name__}
        if meta:
            payload.update(meta)
        return Window(indicator, int(start), payload)


def prime_position_marker(values: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    del rng
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    ps = primes_in(int(values[0]), int(values[-1]) + 1)
    return ps.astype(np.int64)


@dataclass(frozen=True)
class PrimeWindowDataset:
    n_min: int
    n_max: int
    window_len: int

    name: str = "prime_windows"
    domain: str = "primes"
    builder: WindowBuilder = field(default_factory=WindowBuilder)

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        windows: list[Window] = []
        for _ in range(max(0, int(n_windows))):
            start = int(rng.integers(int(self.n_min), int(self.n_max) + 1))
            windows.append(self.builder.build(start, int(self.window_len), prime_position_marker, rng, {"source": "leadengine.sieve.primes_in"}))
        return windows

    def scale_of(self, w: Window) -> int:
        return int(math.log10(max(int(w.start), 1)))
