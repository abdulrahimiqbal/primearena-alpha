from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .core import NullModel, SequenceDataset, Statistic
from .scoring import _feature_matrix


def _dataset_for_decade(real_factory, decade: int) -> SequenceDataset:
    if callable(real_factory):
        return real_factory(int(decade))
    return real_factory


def _auc_scores(stat: Statistic, real: SequenceDataset, null: NullModel, n: int, rng: np.random.Generator) -> tuple[float, np.ndarray, np.ndarray]:
    real_windows = real.sample(n, rng)
    null_windows = null.sample_like(real, n, rng)
    x = np.vstack([_feature_matrix(stat, real_windows), _feature_matrix(stat, null_windows)]).astype(np.float32)
    y = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    order = rng.permutation(len(y))
    mid = len(order) // 2
    train_idx, test_idx = order[:mid], order[mid:]
    model = LogisticRegression(max_iter=1000)
    model.fit(x[train_idx], y[train_idx])
    scores = model.predict_proba(x[test_idx])[:, 1]
    labels = y[test_idx]
    return float(roc_auc_score(labels, scores)), scores.astype(np.float64), labels.astype(np.int8)


def effect_profile(stat, real_factory, null, decades, n_per_decade, rng):
    out = {}
    for decade in decades:
        real = _dataset_for_decade(real_factory, int(decade))
        auc, scores, labels = _auc_scores(stat, real, null, int(n_per_decade), rng)
        boot_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)) ^ int(decade))
        vals = []
        for _ in range(200):
            idx = boot_rng.integers(0, len(labels), size=len(labels))
            vals.append(float(roc_auc_score(labels[idx], scores[idx])) - 0.5)
        se = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[int(decade)] = (float(auc - 0.5), se)
    return out


def _template_values(name: str, decades: list[int]) -> np.ndarray:
    mids = np.asarray([10.0 ** (d + 0.5) for d in decades], dtype=np.float64)
    logs = np.log(mids)
    if name == "1":
        return np.ones_like(logs)
    if name == "1/log n":
        return 1.0 / logs
    if name == "1/log^2 n":
        return 1.0 / (logs * logs)
    if name == "loglog n / log n":
        return np.log(logs) / logs
    raise ValueError(f"unknown template {name}")


TEMPLATES = ("1", "1/log n", "1/log^2 n", "loglog n / log n")


def fit_templates(profile):
    decades = sorted(int(d) for d in profile)
    y = np.asarray([float(profile[d][0]) for d in decades], dtype=np.float64)
    mean = float(np.mean(y)) if y.size else 0.0
    ss_tot = float(np.sum((y - mean) ** 2))
    out = {}
    for name in TEMPLATES:
        x = _template_values(name, decades)
        denom = float(x @ x)
        coeff = float((x @ y) / denom) if denom > 0 else 0.0
        pred = coeff * x
        ss_res = float(np.sum((y - pred) ** 2))
        r2 = 1.0 if ss_tot <= 1e-15 and ss_res <= 1e-15 else 1.0 - ss_res / max(ss_tot, 1e-15)
        out[name] = {
            "r2": float(r2),
            "coefficients": [coeff],
            "predicted": {int(d): float(v) for d, v in zip(decades, pred)},
        }
    return out


def explained_by(profile) -> str | None:
    if not profile:
        return None
    fits = fit_templates(profile)
    decades = sorted(int(d) for d in profile)
    largest = decades[-1]
    observed, se = profile[largest]
    candidates = [name for name in TEMPLATES if name != "1"]
    best = max(candidates, key=lambda name: fits[name]["r2"])
    pred = float(fits[best]["predicted"][largest])
    if float(fits[best]["r2"]) >= 0.9 and abs(pred - float(observed)) <= 2.0 * max(float(se), 1e-12):
        return best
    return None
