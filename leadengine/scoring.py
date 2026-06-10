from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .core import NullModel, SequenceDataset, Statistic, Window


def _feature_matrix(stat: Statistic, windows: list[Window]) -> np.ndarray:
    rows = [np.asarray(stat(w), dtype=np.float32).reshape(-1) for w in windows]
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    width = max(int(r.size) for r in rows)
    out = np.zeros((len(rows), width), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def auc_real_vs_null(
    stat: Statistic,
    real: SequenceDataset,
    null: NullModel,
    n: int,
    rng: np.random.Generator,
    seed_split: int = 0,
) -> float:
    real_windows = real.sample(int(n), rng)
    null_windows = null.sample_like(real, int(n), rng)
    x_real = _feature_matrix(stat, real_windows)
    x_null = _feature_matrix(stat, null_windows)
    x = np.vstack([x_real, x_null]).astype(np.float32)
    y = np.concatenate([np.ones(len(x_real), dtype=np.int8), np.zeros(len(x_null), dtype=np.int8)])

    split_seed = int(rng.integers(0, 2**32 - 1)) ^ int(seed_split)
    split_rng = np.random.default_rng(split_seed)
    order = split_rng.permutation(len(y))
    midpoint = len(order) // 2
    train_idx = order[:midpoint]
    test_idx = order[midpoint:]

    model = LogisticRegression(max_iter=1000)
    model.fit(x[train_idx], y[train_idx])
    scores = model.predict_proba(x[test_idx])[:, 1]
    return float(roc_auc_score(y[test_idx], scores))
