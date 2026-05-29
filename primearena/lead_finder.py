from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .measurements import Measurement, evaluate_measurements, measurement_summary
from .null_worlds import PrimeWindow, build_calibration, generate_real_fake_pair


@dataclass
class PrimeLeadDataset:
    features: np.ndarray
    labels: np.ndarray
    windows: List[PrimeWindow]
    feature_names: List[str]


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / max(pos.size * neg.size, 1))


def _log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def _accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) >= 0.5) == np.asarray(labels).astype(bool)))


def _bootstrap_auc_ci(probs: np.ndarray, labels: np.ndarray, seed: int, rounds: int = 200) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    vals: List[float] = []
    for _ in range(max(1, int(rounds))):
        idx = rng.integers(0, len(labels), size=len(labels))
        vals.append(auc_score(probs[idx], labels[idx]))
    return {
        "auc_ci_low": float(np.quantile(vals, 0.025)),
        "auc_ci_high": float(np.quantile(vals, 0.975)),
        "bootstrap_rounds": int(rounds),
    }


def build_dataset(
    null_name: str,
    samples: int,
    n_min: int,
    n_max: int,
    window_size: int,
    measurements: Sequence[Measurement],
    seed: int,
    q_values: Iterable[int],
    sieve_bounds: Iterable[int],
    calibration_samples: int = 256,
) -> PrimeLeadDataset:
    rng = np.random.default_rng(seed)
    calibration = build_calibration(
        n_min=n_min,
        n_max=n_max,
        window_size=window_size,
        samples=calibration_samples,
        seed=seed + 17,
        q_values=q_values,
    )
    params: Dict[str, Any] = {
        **calibration,
        "q_values": [int(q) for q in q_values],
        "sieve_bound": max(int(x) for x in sieve_bounds),
        "q": 30 if 30 in [int(q) for q in q_values] else int(list(q_values)[0]),
        "density_blocks": max(1, int(window_size) // 64),
        "exact_window_pair_match": True,
    }
    feats: List[np.ndarray] = []
    labels: List[int] = []
    windows: List[PrimeWindow] = []
    for _ in range(max(1, int(samples))):
        n = int(rng.integers(int(n_min), int(n_max) + 1))
        real, fake = generate_real_fake_pair(null_name, n, window_size, rng, params=params)
        for label, window in [(1, real), (0, fake)]:
            feats.append(evaluate_measurements(window, measurements))
            labels.append(label)
            windows.append(window)
    feature_names: List[str] = []
    for m in measurements:
        dim = int(m.evaluate(windows[0]).shape[0]) if windows else 1
        feature_names.extend([f"{m.name}[{i}]" for i in range(dim)])
    return PrimeLeadDataset(
        features=np.stack(feats).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        windows=windows,
        feature_names=feature_names,
    )


def _standardize(train: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train - mean) / std).astype(np.float32), [((x - mean) / std).astype(np.float32) for x in others], mean, std


def train_logistic_discriminator(
    train: PrimeLeadDataset,
    eval_ds: PrimeLeadDataset,
    ood: PrimeLeadDataset,
    seed: int,
    steps: int = 250,
    lr: float = 0.05,
    l1: float = 1e-4,
    shuffled_labels: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    x_train, (x_eval, x_ood), mean, std = _standardize(train.features, eval_ds.features, ood.features)
    y_train_np = train.labels.copy()
    if shuffled_labels:
        np.random.default_rng(seed + 99).shuffle(y_train_np)
    x = torch.from_numpy(x_train).float()
    y = torch.from_numpy(y_train_np).float()
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=float(lr), weight_decay=1e-4)
    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + float(l1) * torch.mean(torch.abs(w))
        loss.backward()
        opt.step()

    with torch.no_grad():
        eval_logits = torch.from_numpy(x_eval).float() @ w + b
        ood_logits = torch.from_numpy(x_ood).float() @ w + b
        eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
        ood_probs = torch.sigmoid(ood_logits).cpu().numpy()
    eval_ce = _log_loss(eval_probs, eval_ds.labels)
    ood_ce = _log_loss(ood_probs, ood.labels)
    null_ce = _log_loss(np.full_like(eval_ds.labels, float(np.mean(train.labels))), eval_ds.labels)
    weights = w.detach().cpu().numpy().astype(np.float64)
    top_idx = np.argsort(np.abs(weights))[::-1][:10]
    return {
        "eval_probs": eval_probs,
        "ood_probs": ood_probs,
        "weights": weights,
        "bias": float(b.detach().cpu().item()),
        "standardize_mean": mean.reshape(-1).astype(float).tolist(),
        "standardize_std": std.reshape(-1).astype(float).tolist(),
        "top_features": [
            {
                "feature": eval_ds.feature_names[int(i)] if int(i) < len(eval_ds.feature_names) else f"feature_{int(i)}",
                "weight": float(weights[int(i)]),
            }
            for i in top_idx
        ],
        "metrics": {
            "real_vs_fake_accuracy": _accuracy(eval_probs, eval_ds.labels),
            "real_vs_fake_AUC": auc_score(eval_probs, eval_ds.labels),
            "cross_entropy": eval_ce,
            "heldout_bits_saved": float((null_ce - eval_ce) / np.log(2.0)),
            "OOD_AUC": auc_score(ood_probs, ood.labels),
            "OOD_accuracy": _accuracy(ood_probs, ood.labels),
            "OOD_cross_entropy": ood_ce,
        },
    }


def run_null_experiment(
    null_name: str,
    measurements: Sequence[Measurement],
    samples: int,
    n_min: int,
    n_max: int,
    window_size: int,
    q_values: Sequence[int],
    sieve_bounds: Sequence[int],
    seed: int,
    ood_multiplier: int = 10,
    train_steps: int = 250,
) -> Dict[str, Any]:
    train_samples = max(64, int(samples * 0.6))
    eval_samples = max(64, int(samples * 0.2))
    ood_samples = max(64, int(samples * 0.2))
    calibration_samples = max(64, min(512, int(samples // 4)))
    train = build_dataset(null_name, train_samples, n_min, n_max, window_size, measurements, seed, q_values, sieve_bounds, calibration_samples)
    eval_ds = build_dataset(null_name, eval_samples, n_min, n_max, window_size, measurements, seed + 1009, q_values, sieve_bounds, calibration_samples)
    ood_min = max(n_max + 1, int(n_min * ood_multiplier))
    ood_max = max(ood_min + 10, int(n_max * ood_multiplier))
    ood = build_dataset(null_name, ood_samples, ood_min, ood_max, window_size, measurements, seed + 2003, q_values, sieve_bounds, calibration_samples)
    trained = train_logistic_discriminator(train, eval_ds, ood, seed=seed + 3001, steps=train_steps)
    permuted = train_logistic_discriminator(train, eval_ds, ood, seed=seed + 4001, steps=max(50, train_steps // 2), shuffled_labels=True)
    metrics = dict(trained["metrics"])
    metrics["compression_bits_saved"] = float(metrics["heldout_bits_saved"])
    metrics["permutation_test_AUC"] = float(permuted["metrics"]["real_vs_fake_AUC"])
    metrics["permutation_auc_delta"] = float(metrics["real_vs_fake_AUC"] - metrics["permutation_test_AUC"])
    metrics.update(_bootstrap_auc_ci(trained["eval_probs"], eval_ds.labels, seed=seed + 5001, rounds=200))
    ms = measurement_summary(measurements)
    metrics["feature_complexity"] = float(ms["total_complexity"])
    metrics["measurement_cost"] = float(ms["total_cost"])
    return {
        "null_model": null_name,
        "seed": int(seed),
        "samples": int(samples),
        "train_samples": int(train_samples),
        "eval_samples": int(eval_samples),
        "ood_samples": int(ood_samples),
        "n_range": [int(n_min), int(n_max)],
        "ood_n_range": [int(ood_min), int(ood_max)],
        "window_size": int(window_size),
        "measurement_summary": ms,
        "metrics": metrics,
        "top_features": trained["top_features"],
    }
