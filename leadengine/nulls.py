from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from primearena.null_worlds import make_real_window, wheel_iid_no_pair_bias_null

from .core import NullModel, SequenceDataset, Statistic, Window


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
    wheel: int = 30

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
