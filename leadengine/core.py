from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Mapping, Any
import numpy as np


@dataclass(frozen=True)
class Window:
    values: np.ndarray              # primes: 0/1 indicator over consecutive integers
    start: int                      # primes: first integer n of the window
    meta: Mapping[str, Any] = field(default_factory=dict)


class SequenceDataset(Protocol):
    name: str
    domain: str                     # "primes" | "zeros" | "ff_angles"
    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]: ...
    def scale_of(self, w: Window) -> int: ...      # primes: int(log10(w.start))


class Statistic(Protocol):
    name: str
    def __call__(self, w: Window) -> np.ndarray: ...   # 1-D feature vector
    def complexity(self) -> float: ...
    def describe(self) -> str: ...


class NullModel(Protocol):
    name: str
    def sample_like(self, real: SequenceDataset, n_windows: int,
                    rng: np.random.Generator) -> list[Window]: ...
    def absorb(self, stat: Statistic, real: SequenceDataset) -> "NullModel": ...
