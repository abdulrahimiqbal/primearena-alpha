from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .core import NullModel, SequenceDataset, Statistic, Window


class AbsorptionError(RuntimeError):
    """Raised when exponential tilting would produce a degenerate null."""


def _feature_matrix(stats: Sequence[Statistic], windows: list[Window]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for w in windows:
        parts = [np.asarray(stat(w), dtype=np.float64).reshape(-1) for stat in stats]
        rows.append(np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64))
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    width = max(int(r.size) for r in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _augment_moments(phi: np.ndarray, match_second_moments: bool) -> np.ndarray:
    if not match_second_moments:
        return phi
    return np.concatenate([phi, phi * phi], axis=1)


@dataclass
class TiltedNull(NullModel):
    base: NullModel
    stats: list[Statistic]
    oversample_factor: int = 10
    match_second_moments: bool = True
    min_ess_ratio: float = 0.1
    last_ess: float | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        stat_names = "_".join(str(s.name) for s in self.stats)
        return f"tilted_{self.base.name}_{stat_names}"

    @property
    def ess(self) -> float | None:
        return self.last_ess

    def _fit_weights(self, real_windows: list[Window], candidates: list[Window]) -> np.ndarray:
        if not self.stats:
            return np.full(len(candidates), 1.0 / max(len(candidates), 1), dtype=np.float64)

        raw_real = _feature_matrix(self.stats, real_windows)
        raw_null = _feature_matrix(self.stats, candidates)
        if raw_real.size and raw_null.size:
            raw_target = raw_real.mean(axis=0)
            raw_scale = raw_null.std(axis=0)
            if float(np.max(np.abs(raw_target))) > 10_000.0 and float(np.max(raw_scale)) > 10_000.0:
                raise AbsorptionError("absorbed statistic appears to be a high-leverage window identifier")

        phi_real = _augment_moments(raw_real, self.match_second_moments)
        phi_null = _augment_moments(raw_null, self.match_second_moments)
        target = phi_real.mean(axis=0)

        center = phi_null.mean(axis=0)
        scale = phi_null.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        z = (phi_null - center) / scale
        target_z = (target - center.reshape(-1)) / scale.reshape(-1)

        # Refuse extremely high-leverage targets up front. These usually mean
        # the requested statistic is an identifier rather than structure.
        if float(np.max(np.abs(target_z))) > 6.0:
            raise AbsorptionError("absorbed statistic target is outside the candidate null support")

        def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
            logits = z @ alpha
            log_norm = logsumexp(logits)
            probs = np.exp(logits - log_norm)
            loss = float(log_norm - np.log(max(len(candidates), 1)) - alpha @ target_z)
            grad = (probs @ z) - target_z
            return loss, grad

        result = minimize(
            fun=lambda a: objective(a)[0],
            jac=lambda a: objective(a)[1],
            x0=np.zeros(z.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        if not result.success:
            raise AbsorptionError(f"tilting optimizer failed: {result.message}")

        probs = softmax(z @ np.asarray(result.x, dtype=np.float64))
        ess = float(1.0 / max(float(np.sum(probs * probs)), 1e-300))
        self.last_ess = ess
        return probs

    def sample_like(self, real: SequenceDataset, n_windows: int, rng: np.random.Generator) -> list[Window]:
        n_windows = int(n_windows)
        if n_windows <= 0:
            return []
        real_windows = real.sample(n_windows, rng)
        candidate_count = max(n_windows, n_windows * max(1, int(self.oversample_factor)))
        candidates = self.base.sample_like(real, candidate_count, rng)
        probs = self._fit_weights(real_windows, candidates)
        ess = float(self.last_ess or 0.0)
        if ess / max(float(n_windows), 1.0) < float(self.min_ess_ratio):
            raise AbsorptionError(f"tilted null ESS too low: {ess:.3f} for n={n_windows}")
        idx = rng.choice(len(candidates), size=n_windows, replace=True, p=probs)
        return [candidates[int(i)] for i in idx]

    def absorb(self, stat: Statistic, real: SequenceDataset) -> NullModel:
        return TiltedNull(
            base=self.base,
            stats=[*self.stats, stat],
            oversample_factor=self.oversample_factor,
            match_second_moments=self.match_second_moments,
            min_ess_ratio=self.min_ess_ratio,
        )

    def as_dataset(self, real: SequenceDataset, n_cache: int, rng: np.random.Generator) -> SequenceDataset:
        from .nulls import CachedWindowDataset

        return CachedWindowDataset(
            name=f"{self.name}_cached",
            domain=real.domain,
            windows=self.sample_like(real, n_cache, rng),
        )
