from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import NullModel, SequenceDataset, Statistic, Window
from .nulls import CachedWindowDataset


def load_zeros(path) -> np.ndarray:
    return np.loadtxt(Path(path), dtype=np.float64)


def unfold(gammas) -> np.ndarray:
    g = np.asarray(gammas, dtype=np.float64)
    if g.size <= 1:
        return np.empty(0, dtype=np.float64)
    return np.diff(g) * np.log(g[:-1] / (2.0 * np.pi)) / (2.0 * np.pi)


@dataclass(frozen=True)
class ZeroSpacingDataset:
    path: str
    window_len: int = 64
    name: str = "zeta_zero_spacings"
    domain: str = "zeros"
    _zeros: np.ndarray = field(default=None, init=False, repr=False, compare=False)
    _spacings: np.ndarray = field(default=None, init=False, repr=False, compare=False)

    @property
    def zeros(self) -> np.ndarray:
        z = object.__getattribute__(self, "_zeros")
        if z is None:
            z = load_zeros(self.path)
            object.__setattr__(self, "_zeros", z)
        return z

    @property
    def spacings(self) -> np.ndarray:
        s = object.__getattribute__(self, "_spacings")
        if s is None:
            s = unfold(self.zeros)
            object.__setattr__(self, "_spacings", s)
        return s

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        s = self.spacings
        max_start = max(1, s.size - int(self.window_len))
        out: list[Window] = []
        for _ in range(max(0, int(n_windows))):
            idx = int(rng.integers(0, max_start))
            gamma = float(self.zeros[idx])
            out.append(
                Window(
                    values=s[idx : idx + int(self.window_len)].astype(np.float32),
                    start=idx,
                    meta={"domain": self.domain, "gamma_start": gamma, "offset": idx},
                )
            )
        return out

    def scale_of(self, w: Window) -> int:
        return int(math.log10(max(float(w.meta.get("gamma_start", 1.0)), 1.0)))


@dataclass(frozen=True)
class PoissonSpacingNull:
    name: str = "poisson_spacing"

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        probe = real.sample(1, rng)
        window_len = int(len(probe[0].values)) if probe else 64
        out = []
        for i in range(max(0, int(n_windows))):
            out.append(
                Window(
                    values=rng.exponential(scale=1.0, size=window_len).astype(np.float32),
                    start=i,
                    meta={"domain": "zeros", "null_model": self.name},
                )
            )
        return out

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        raise NotImplementedError("PoissonSpacingNull absorption is not implemented in Phase 5a.")

    def as_dataset(self, real: SequenceDataset, n_cache: int, rng: np.random.Generator) -> SequenceDataset:
        return CachedWindowDataset(name=f"{self.name}_cached", domain="zeros", windows=self.sample_like(real, n_cache, rng))


@dataclass
class GUESpacingNull:
    seed: int = 0
    matrix_size: int = 128
    name: str = "gue_spacing"
    _pool: np.ndarray | None = field(default=None, init=False, repr=False)

    def _matrix_spacings(self, rng: np.random.Generator) -> np.ndarray:
        n = int(self.matrix_size)
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        h = (a + a.conj().T) / np.sqrt(2.0 * n)
        eigs = np.linalg.eigvalsh(h).astype(np.float64)
        lo = n // 4
        hi = n - lo
        central = eigs[lo:hi]
        x = np.clip(central / 2.0, -0.999999, 0.999999)
        density = np.sqrt(np.maximum(4.0 - central[:-1] ** 2, 1e-9)) / (2.0 * np.pi)
        return np.diff(central) * n * density

    def _spacing_pool(self, min_size: int) -> np.ndarray:
        if self._pool is not None and self._pool.size >= min_size:
            return self._pool
        rng = np.random.default_rng(int(self.seed))
        chunks: list[np.ndarray] = []
        total = 0
        while total < min_size:
            s = self._matrix_spacings(rng)
            chunks.append(s.astype(np.float32))
            total += s.size
        self._pool = np.concatenate(chunks).astype(np.float32)
        return self._pool

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        probe = real.sample(1, rng)
        window_len = int(len(probe[0].values)) if probe else 64
        pool = self._spacing_pool(max(10_000, (int(n_windows) + 2) * window_len))
        max_start = max(1, pool.size - window_len)
        out = []
        for _ in range(max(0, int(n_windows))):
            idx = int(rng.integers(0, max_start))
            out.append(Window(values=pool[idx : idx + window_len].astype(np.float32), start=idx, meta={"domain": "zeros", "null_model": self.name}))
        return out

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        from .absorb import TiltedNull

        return TiltedNull(base=self, stats=[stat])

    def as_dataset(self, window_len: int = 64, n_cache: int = 4000) -> SequenceDataset:
        class _Probe:
            name = "gue_probe"
            domain = "zeros"
            def sample(self, n_windows, rng):
                return [Window(values=np.ones(window_len, dtype=np.float32), start=0, meta={"domain": "zeros"}) for _ in range(n_windows)]
            def scale_of(self, w):
                return 0

        return CachedWindowDataset(
            name=f"{self.name}_cached",
            domain="zeros",
            windows=self.sample_like(_Probe(), n_cache, np.random.default_rng(int(self.seed) + 17)),
        )
