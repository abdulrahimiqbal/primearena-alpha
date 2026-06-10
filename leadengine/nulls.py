from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from primearena.null_worlds import make_real_window, wheel_iid_no_pair_bias_null

from .core import NullModel, SequenceDataset, Statistic, Window
from .datasets import WindowBuilder


@dataclass(frozen=True)
class CachedWindowDataset:
    name: str
    domain: str
    windows: list[Window]

    def sample(self, n_windows: int, rng: np.random.Generator) -> list[Window]:
        if not self.windows or int(n_windows) <= 0:
            return []
        idx = rng.integers(0, len(self.windows), size=int(n_windows))
        return [self.windows[int(i)] for i in idx]

    def scale_of(self, w: Window) -> int:
        return int(math.log10(max(int(w.start), 1)))


@dataclass(frozen=True)
class WheelNull:
    """SUPERSEDED: retained only for science_003 Task-0 autopsy."""

    wheel: int = 30
    superseded: bool = True

    @property
    def name(self) -> str:
        return f"wheel_{int(self.wheel)}"

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        real_windows = real.sample(n_windows, rng)
        out: list[Window] = []
        for w in real_windows:
            pw = make_real_window(int(w.start) - 1, int(len(w.values)), q_values=(6, 10, 30, 210))
            fake = wheel_iid_no_pair_bias_null(
                pw,
                rng,
                {
                    "sieve_bound": int(self.wheel),
                    "q_values": [6, 10, 30, 210],
                    "q": 10,
                    "density_blocks": max(1, int(len(w.values)) // 64),
                    "wheel_mod_q": True,
                },
            )
            out.append(
                Window(
                    values=fake.indicator.astype(np.int8),
                    start=int(w.start),
                    meta={
                        "domain": real.domain,
                        "integer_values": fake.values.astype(np.int64),
                        "null_model": self.name,
                    },
                )
            )
        return out

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        from .absorb import TiltedNull

        return TiltedNull(base=self, stats=[stat])

    def as_dataset(self, real: SequenceDataset, n_cache: int, rng: np.random.Generator) -> SequenceDataset:
        return CachedWindowDataset(
            name=f"{self.name}_cached",
            domain=real.domain,
            windows=self.sample_like(real, n_cache, rng),
        )


def coprime_residues(wheel: int) -> tuple[int, ...]:
    w = max(2, int(wheel))
    return tuple(r for r in range(w) if math.gcd(r, w) == 1)


@dataclass(frozen=True)
class SkeletonResampleNull:
    wheel: int = 30
    allowed_residues: tuple[int, ...] | None = None
    residue_weights: tuple[float, ...] | None = None
    builder: WindowBuilder = field(default_factory=WindowBuilder)
    label: str = "skeleton_resample"

    @property
    def name(self) -> str:
        return f"{self.label}_wheel_{int(self.wheel)}"

    @property
    def residues(self) -> tuple[int, ...]:
        return tuple(int(r) for r in (self.allowed_residues or coprime_residues(self.wheel)))

    def _mark_positions(self, values: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
        residues = self.residues
        mask = np.isin(values % int(self.wheel), np.asarray(residues, dtype=np.int64))
        choices = values[mask]
        if choices.size == 0 or int(count) <= 0:
            return np.empty(0, dtype=np.int64)
        p = None
        if self.residue_weights is not None:
            weights = np.asarray(self.residue_weights, dtype=np.float64)
            by_residue = {r: float(weights[i]) for i, r in enumerate(residues) if i < weights.size}
            p = np.asarray([max(0.0, by_residue.get(int(v % int(self.wheel)), 0.0)) for v in choices], dtype=np.float64)
            p = None if float(p.sum()) <= 0.0 else p / float(p.sum())
        size = min(int(count), int(choices.size))
        return np.asarray(rng.choice(choices, size=size, replace=False, p=p), dtype=np.int64)

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        donors = real.sample(int(n_windows), rng)
        return [self.twin_for(w, rng) for w in donors]

    def twin_for(self, w: Window, rng: np.random.Generator) -> Window:
        count = int(np.asarray(w.values).sum())
        start = int(w.start)
        length = int(len(w.values))

        def marker(values: np.ndarray, local_rng: np.random.Generator | None) -> np.ndarray:
            return self._mark_positions(values, local_rng or rng, count)

        return self.builder.build(
            start,
            length,
            marker,
            rng,
            {
                "null_model": self.name,
                "wheel": int(self.wheel),
                "allowed_residues": list(self.residues),
                "donor_prime_count": count,
                "paired_skeleton": True,
            },
        )

    def sample_pairs(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> tuple[list[Window], list[Window]]:
        donors = real.sample(int(n_windows), rng)
        out: list[Window] = []
        for w in donors:
            out.append(self.twin_for(w, rng))
        return donors, out

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        from .absorb import GeneratorConstraint, TiltedNull

        if isinstance(stat, GeneratorConstraint):
            wheel = int(stat.wheel or self.wheel)
            residues = tuple(int(r) for r in (stat.allowed_residues if stat.allowed_residues is not None else coprime_residues(wheel)))
            weights = None if stat.residue_weights is None else tuple(float(x) for x in stat.residue_weights)
            return SkeletonResampleNull(
                wheel=wheel,
                allowed_residues=residues,
                residue_weights=weights,
                builder=self.builder,
                label=f"{self.label}_constrained",
            )
        return TiltedNull(base=self, stats=[stat])

    def as_dataset(self, real: SequenceDataset, n_cache: int, rng: np.random.Generator) -> SequenceDataset:
        return CachedWindowDataset(
            name=f"{self.name}_cached",
            domain=real.domain,
            windows=self.sample_like(real, n_cache, rng),
        )
