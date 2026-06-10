from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .core import NullModel, SequenceDataset, Statistic, Window
from .sieve import primes_in


@lru_cache(maxsize=None)
def _legendre_table(p: int) -> np.ndarray:
    p = int(p)
    table = np.full(p, -1, dtype=np.int8)
    table[0] = 0
    xs = np.arange(1, p, dtype=np.int64)
    table[((xs * xs) % p).astype(np.int64)] = 1
    return table


def ap_elliptic(a, b, p) -> int:
    p = int(p)
    a = int(a) % p
    b = int(b) % p
    if (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p == 0:
        raise ValueError("singular elliptic curve")
    x = np.arange(p, dtype=np.int64)
    rhs = (x * x % p * x + a * x + b) % p
    return int(-np.sum(_legendre_table(p)[rhs.astype(np.int64)]))


@dataclass(frozen=True)
class EllipticAngleDataset:
    p_min: int = 101
    p_max: int = 499
    window_len: int = 64
    name: str = "elliptic_frobenius_angles"
    domain: str = "ff_angles"

    @property
    def primes(self) -> np.ndarray:
        return primes_in(int(self.p_min), int(self.p_max) + 1)

    def _sample_angle(self, rng: np.random.Generator) -> tuple[float, int]:
        primes = self.primes
        while True:
            p = int(rng.choice(primes))
            a = int(rng.integers(0, p))
            b = int(rng.integers(0, p))
            if (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p == 0:
                continue
            ap = ap_elliptic(a, b, p)
            t = np.clip(ap / (2.0 * math.sqrt(p)), -1.0, 1.0)
            return float(math.acos(float(t))), p

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        out: list[Window] = []
        for i in range(max(0, int(n_windows))):
            angles = np.empty(int(self.window_len), dtype=np.float32)
            ps = np.empty(int(self.window_len), dtype=np.int64)
            for j in range(int(self.window_len)):
                theta, p = self._sample_angle(rng)
                angles[j] = theta
                ps[j] = p
            out.append(Window(values=angles, start=i, meta={"domain": self.domain, "p_values": ps}))
        return out

    def scale_of(self, w: Window) -> int:
        ps = np.asarray(w.meta.get("p_values", [self.p_min]), dtype=np.int64)
        return int(math.log10(max(float(np.mean(ps)), 1.0)))


@dataclass(frozen=True)
class UniformAngleNull:
    name: str = "uniform_angle"

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        probe = real.sample(1, rng)
        window_len = int(len(probe[0].values)) if probe else 64
        return [
            Window(
                values=rng.uniform(0.0, np.pi, size=window_len).astype(np.float32),
                start=i,
                meta={"domain": "ff_angles", "null_model": self.name},
            )
            for i in range(max(0, int(n_windows)))
        ]

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        raise NotImplementedError("UniformAngleNull absorption is not implemented in Phase 5b.")


@dataclass(frozen=True)
class SatoTateNull:
    name: str = "sato_tate"

    def _draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        out: list[float] = []
        while len(out) < int(size):
            theta = rng.uniform(0.0, np.pi, size=max(128, int(size) - len(out)))
            accept = rng.random(theta.size) <= np.sin(theta) ** 2
            out.extend(theta[accept].tolist())
        return np.asarray(out[: int(size)], dtype=np.float32)

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        probe = real.sample(1, rng)
        window_len = int(len(probe[0].values)) if probe else 64
        return [
            Window(
                values=self._draw(window_len, rng),
                start=i,
                meta={"domain": "ff_angles", "null_model": self.name},
            )
            for i in range(max(0, int(n_windows)))
        ]

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        raise NotImplementedError("SatoTateNull is already the matched iid family in Phase 5b.")
