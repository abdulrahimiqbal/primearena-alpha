from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .baselines import residual_survivors_after
from .config import EnvConfig, ModelConfig, RunConfig
from .counterfactuals import build_counterfactual_pairs
from .oracle import is_prime, next_prime


RANK_FEATURE_DIM = 8


@dataclass
class ResidualRankBatch:
    features: np.ndarray
    true_index: np.ndarray
    prime_labels: Optional[np.ndarray]
    n_values: np.ndarray
    true_next_prime: np.ndarray
    survivor_values: np.ndarray
    metadata: Optional[List[Dict[str, Any]]] = None


class ResidualRankNet(nn.Module):
    def __init__(self, feature_dim: int, window_size: int, cfg: ModelConfig):
        super().__init__()
        self.window_size = int(window_size)
        self.hidden_dim = int(cfg.hidden_dim)
        self.input = nn.Linear(feature_dim, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.window_size, cfg.hidden_dim)) if cfg.use_positional_embeddings else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=max(1, cfg.n_heads),
            dim_feedforward=cfg.hidden_dim * max(1, cfg.ff_mult),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=max(1, cfg.layers))
        self.rank_head = nn.Linear(cfg.hidden_dim, 1)
        self.prime_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input(features)
        if self.pos is not None:
            h = h + self.pos[:, : h.shape[1], :]
        h = self.backbone(h)
        rank_logits = self.rank_head(h).squeeze(-1)
        prime_logits = self.prime_head(h).squeeze(-1)
        return rank_logits, prime_logits


def make_rank_model(cfg: RunConfig, device: torch.device) -> ResidualRankNet:
    return ResidualRankNet(RANK_FEATURE_DIM, cfg.env.residual_rank_window_size, cfg.model).to(device)


def save_rank_checkpoint(path: str, model: ResidualRankNet, optimizer: Optional[torch.optim.Optimizer], step: int, extra: Optional[dict] = None) -> None:
    payload: Dict[str, Any] = {"model_state": model.state_dict(), "step": int(step), "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_rank_checkpoint(path: str, model: ResidualRankNet, optimizer: Optional[torch.optim.Optimizer] = None, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _rank_features(n: int, survivors: List[int], cfg: EnvConfig) -> np.ndarray:
    w = len(survivors)
    arr = np.asarray(survivors, dtype=np.float64)
    distances = arr - float(n)
    gaps = np.diff(np.concatenate([[float(n)], arr]))
    max_dist = max(float(distances[-1]), 1.0)
    log_n = math.log(max(n, 3))
    density = w / max(max_dist, 1.0)
    rows: List[List[float]] = []
    for i, (x, dist, gap) in enumerate(zip(arr, distances, gaps)):
        rows.append(
            [
                (i + 1) / max(w, 1),
                math.log10(max(float(dist), 1.0)) / math.log10(max(max_dist, 2.0)),
                float(dist) / max_dist,
                1.0 / max(float(gap), 1.0),
                float(gap) / max(max_dist, 1.0),
                log_n / 32.0,
                density,
                float(cfg.residual_rank_sieve_bound) / 1000.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def build_rank_example(
    cfg: EnvConfig,
    rng: np.random.Generator,
    sieve_bound: Optional[int] = None,
    include_prime_labels: bool = False,
) -> Dict[str, Any]:
    bound = int(sieve_bound or cfg.residual_rank_sieve_bound)
    old_bound = cfg.residual_sieve_bound
    old_rank_bound = cfg.residual_rank_sieve_bound
    cfg.residual_sieve_bound = bound
    cfg.residual_rank_sieve_bound = bound
    try:
        for _ in range(cfg.max_resample_attempts):
            n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
            true_p = next_prime(n)
            survivors = residual_survivors_after(n, cfg, cfg.residual_rank_window_size, bound=bound)
            if true_p in survivors:
                true_idx = survivors.index(true_p)
                labels = np.asarray([is_prime(x) for x in survivors], dtype=np.float32) if include_prime_labels else None
                return {
                    "features": _rank_features(n, survivors, cfg),
                    "true_index": int(true_idx),
                    "prime_labels": labels,
                    "n": int(n),
                    "true_next_prime": int(true_p),
                    "survivors": np.asarray(survivors, dtype=np.int64),
                    "survivor_distances": np.asarray([x - n for x in survivors], dtype=np.int64),
                }
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
        true_p = next_prime(n)
        survivors = residual_survivors_after(n, cfg, cfg.residual_rank_window_size, bound=bound)
        while true_p not in survivors:
            survivors.extend(residual_survivors_after(survivors[-1], cfg, cfg.residual_rank_window_size, bound=bound, start=survivors[-1] + 1))
        survivors = survivors[: cfg.residual_rank_window_size]
        true_idx = min(survivors.index(true_p), cfg.residual_rank_window_size - 1)
        labels = np.asarray([is_prime(x) for x in survivors], dtype=np.float32) if include_prime_labels else None
        return {
            "features": _rank_features(n, survivors, cfg),
            "true_index": int(true_idx),
            "prime_labels": labels,
            "n": int(n),
            "true_next_prime": int(true_p),
            "survivors": np.asarray(survivors, dtype=np.int64),
            "survivor_distances": np.asarray([x - n for x in survivors], dtype=np.int64),
        }
    finally:
        cfg.residual_sieve_bound = old_bound
        cfg.residual_rank_sieve_bound = old_rank_bound


def _examples_to_batch(examples: List[Dict[str, Any]], include_prime_labels: bool = False) -> ResidualRankBatch:
    prime_labels = None
    if include_prime_labels:
        prime_labels = np.stack([ex["prime_labels"] for ex in examples]).astype(np.float32)
    return ResidualRankBatch(
        features=np.stack([ex["features"] for ex in examples]).astype(np.float32),
        true_index=np.asarray([ex["true_index"] for ex in examples], dtype=np.int64),
        prime_labels=prime_labels,
        n_values=np.asarray([ex["n"] for ex in examples], dtype=np.int64),
        true_next_prime=np.asarray([ex["true_next_prime"] for ex in examples], dtype=np.int64),
        survivor_values=np.stack([ex["survivors"] for ex in examples]).astype(np.int64),
        metadata=[dict(ex.get("metadata", {})) for ex in examples],
    )


def build_rank_batch(
    cfg: EnvConfig,
    batch_size: int,
    seed: int,
    sieve_bound: Optional[int] = None,
    include_prime_labels: bool = False,
) -> ResidualRankBatch:
    rng = np.random.default_rng(seed)
    examples = [build_rank_example(cfg, rng, sieve_bound=sieve_bound, include_prime_labels=include_prime_labels) for _ in range(batch_size)]
    return _examples_to_batch(examples, include_prime_labels=include_prime_labels)


def hard_pattern_key(example: Dict[str, Any], cfg: EnvConfig) -> Tuple[Any, ...]:
    survivors = np.asarray(example["survivors"], dtype=np.int64)
    n = int(example["n"])
    sig_len = max(1, min(int(cfg.residual_rank_hard_signature_len), len(survivors), 16))
    gaps = np.diff(np.concatenate([[n], survivors[:sig_len]])).astype(np.int64)
    gap_bins = tuple(int(min(max(g, 0), 96) // 8) for g in gaps[:8])
    mod = max(2, int(cfg.residual_rank_hard_match_mod))
    residues = tuple(int(x % mod) for x in survivors[:8])
    scale_bin = int(math.log10(max(n, 3)) * 8)
    return (scale_bin, gap_bins, residues)


def rank_batch_diagnostics(batch: ResidualRankBatch) -> Dict[str, float]:
    out: Dict[str, float] = {
        "avg_true_index": float(np.mean(batch.true_index)) if len(batch.true_index) else 0.0,
        "nearest_fail_rate": float(np.mean(batch.true_index > 0)) if len(batch.true_index) else 0.0,
        "true_index_gt3_rate": float(np.mean(batch.true_index > 3)) if len(batch.true_index) else 0.0,
    }
    if batch.metadata:
        numeric_keys: List[str] = []
        for meta in batch.metadata:
            for key, value in meta.items():
                if isinstance(value, (bool, int, float)) and key not in numeric_keys:
                    numeric_keys.append(key)
        for key in numeric_keys:
            vals = [float(meta.get(key, 0.0)) for meta in batch.metadata]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


def _score_hard_pool(
    cfg: EnvConfig,
    examples: List[Dict[str, Any]],
    include_input_only: bool,
    solver_score_fn: Optional[Callable[[ResidualRankBatch], np.ndarray]],
    include_prime_labels: bool,
) -> List[Dict[str, Any]]:
    batch = _examples_to_batch(examples, include_prime_labels=include_prime_labels)
    nearest_logits = baseline_logits(batch, "nearest")
    nearest_pred = np.argmax(nearest_logits, axis=1)
    input_pred = nearest_pred.copy()
    input_margin = np.zeros(len(examples), dtype=np.float64)
    if include_input_only and len(examples) > 1:
        try:
            weights = fit_input_only_ranker(batch)
            input_logits_arr = input_only_logits(batch, weights)
            input_pred = np.argmax(input_logits_arr, axis=1)
            sorted_logits = np.sort(input_logits_arr, axis=1)
            input_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
        except np.linalg.LinAlgError:
            input_pred = nearest_pred.copy()

    solver_pred = np.full(len(examples), -1, dtype=np.int64)
    solver_uncertainty = np.zeros(len(examples), dtype=np.float64)
    if solver_score_fn is not None and len(examples):
        solver_logits = np.asarray(solver_score_fn(batch), dtype=np.float64)
        solver_pred = np.argmax(solver_logits, axis=1)
        shifted = solver_logits - solver_logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        top2 = np.sort(probs, axis=1)[:, -2:]
        solver_uncertainty = 1.0 - (top2[:, 1] - top2[:, 0])

    buckets: Dict[Tuple[Any, ...], set[int]] = {}
    for ex in examples:
        buckets.setdefault(hard_pattern_key(ex, cfg), set()).add(int(ex["true_index"]))

    for i, ex in enumerate(examples):
        true_idx = int(ex["true_index"])
        key = hard_pattern_key(ex, cfg)
        nearest_wrong = bool(nearest_pred[i] != true_idx)
        input_wrong = bool(input_pred[i] != true_idx)
        solver_wrong = bool(solver_pred[i] >= 0 and solver_pred[i] != true_idx)
        matched_pattern = bool(len(buckets.get(key, set())) > 1)
        min_idx = int(cfg.residual_rank_hard_min_true_index)
        score = 0.0
        score += 1.5 if nearest_wrong else 0.0
        score += 1.0 if input_wrong else 0.0
        score += 1.0 if solver_wrong else 0.0
        score += 0.75 if matched_pattern else 0.0
        score += min(true_idx, 16) / 16.0
        score += 0.25 * float(solver_uncertainty[i])
        score += 0.10 / max(float(input_margin[i]), 0.10)
        if true_idx < min_idx:
            score -= 2.0
        meta = {
            "hard_score": float(score),
            "nearest_wrong": nearest_wrong,
            "input_only_wrong": input_wrong,
            "solver_wrong": solver_wrong,
            "solver_uncertainty": float(solver_uncertainty[i]),
            "matched_pattern": matched_pattern,
            "true_index": true_idx,
        }
        ex["metadata"] = {**dict(ex.get("metadata", {})), **meta}
    return examples


def build_hard_rank_batch(
    cfg: EnvConfig,
    batch_size: int,
    seed: int,
    sieve_bound: Optional[int] = None,
    include_prime_labels: bool = False,
    solver_score_fn: Optional[Callable[[ResidualRankBatch], np.ndarray]] = None,
) -> ResidualRankBatch:
    """Build a hard/balanced residual-rank batch.

    The adversary over-samples examples where simple explanations fail:
    nearest-survivor is wrong, a fitted input-only ranker is wrong, true index
    is not the first survivor, or the coarse wheel/sieve signature is shared by
    examples with different labels.
    """
    rng = np.random.default_rng(seed)
    hard_fraction = min(max(float(cfg.residual_rank_hard_fraction), 0.0), 1.0)
    hard_count = int(round(batch_size * hard_fraction))
    random_count = max(0, int(batch_size) - hard_count)
    multiplier = max(2, int(cfg.residual_rank_hard_pool_multiplier))
    pool_size = max(hard_count * multiplier, hard_count + 32, batch_size)
    examples = [
        build_rank_example(cfg, rng, sieve_bound=sieve_bound, include_prime_labels=include_prime_labels)
        for _ in range(pool_size)
    ]
    scored = _score_hard_pool(
        cfg,
        examples,
        include_input_only=bool(cfg.residual_rank_hard_include_input_only),
        solver_score_fn=solver_score_fn,
        include_prime_labels=include_prime_labels,
    )
    order = sorted(range(len(scored)), key=lambda i: float(scored[i].get("metadata", {}).get("hard_score", 0.0)), reverse=True)

    selected: List[int] = []
    if cfg.residual_rank_hard_balance_indices and hard_count > 0:
        by_idx: Dict[int, List[int]] = {}
        for idx in order:
            bucket = min(int(scored[idx]["true_index"]), 8)
            by_idx.setdefault(bucket, []).append(idx)
        while len(selected) < hard_count and by_idx:
            for bucket in sorted(list(by_idx)):
                if by_idx[bucket]:
                    selected.append(by_idx[bucket].pop(0))
                    if len(selected) >= hard_count:
                        break
                if not by_idx.get(bucket):
                    by_idx.pop(bucket, None)
    for idx in order:
        if len(selected) >= hard_count:
            break
        if idx not in selected:
            selected.append(idx)

    remaining = [i for i in range(len(scored)) if i not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[:random_count])
    if len(selected) < batch_size:
        selected.extend(order[: batch_size - len(selected)])
    chosen = [scored[i] for i in selected[:batch_size]]
    return _examples_to_batch(chosen, include_prime_labels=include_prime_labels)


def build_residual_rank_batch(
    cfg: EnvConfig,
    batch_size: int,
    seed: int,
    sieve_bound: Optional[int] = None,
    include_prime_labels: bool = False,
    solver_score_fn: Optional[Callable[[ResidualRankBatch], np.ndarray]] = None,
) -> ResidualRankBatch:
    if cfg.residual_rank_hard_mode:
        return build_hard_rank_batch(
            cfg,
            batch_size,
            seed=seed,
            sieve_bound=sieve_bound,
            include_prime_labels=include_prime_labels,
            solver_score_fn=solver_score_fn,
        )
    return build_rank_batch(cfg, batch_size, seed=seed, sieve_bound=sieve_bound, include_prime_labels=include_prime_labels)


def rank_metrics(logits: np.ndarray, true_index: np.ndarray) -> Dict[str, float]:
    logits = logits.astype(np.float64)
    true_index = true_index.astype(np.int64)
    order = np.argsort(logits, axis=1)[:, ::-1]
    ranks = np.asarray([int(np.where(order[i] == true_index[i])[0][0]) + 1 for i in range(len(true_index))], dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    true_probs = probs[np.arange(len(true_index)), true_index]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(true_index)), true_index] = 1.0
    return {
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "top5_accuracy": float(np.mean(ranks <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "cross_entropy": float(np.mean(-np.log(np.maximum(true_probs, 1e-12)))),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "avg_true_rank": float(np.mean(ranks)),
    }


def calibration_bins(logits: np.ndarray, true_index: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = pred == true_index
    rows: List[Dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "confidence": float(np.mean(conf[mask])) if mask.any() else 0.0,
                "accuracy": float(np.mean(correct[mask])) if mask.any() else 0.0,
            }
        )
    return rows


def baseline_logits(batch: ResidualRankBatch, name: str, seed: int = 0) -> np.ndarray:
    b, w = batch.true_index.shape[0], batch.survivor_values.shape[1]
    distances = batch.survivor_values - batch.n_values[:, None]
    if name in {"nearest", "distance_weighted"}:
        return -distances.astype(np.float64)
    if name == "one_over_logn":
        return 1.0 / np.maximum(np.log(np.maximum(batch.survivor_values.astype(np.float64), 3.0)), 1e-9)
    if name == "survivor_density":
        gaps = np.diff(np.concatenate([batch.n_values[:, None], batch.survivor_values], axis=1), axis=1)
        return 1.0 / np.maximum(gaps.astype(np.float64), 1.0)
    if name == "random":
        return np.random.default_rng(seed).normal(size=(b, w))
    if name == "uniform":
        return np.zeros((b, w), dtype=np.float64)
    raise ValueError(f"Unknown residual rank baseline: {name}")


def fit_input_only_ranker(train: ResidualRankBatch, ridge: float = 1e-3) -> np.ndarray:
    b, w, f = train.features.shape
    dim = f + 1
    xtx = ridge * np.eye(dim, dtype=np.float64)
    xty = np.zeros((dim, 1), dtype=np.float64)
    chunk = max(1, min(b, 1024))
    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        x = train.features[start:end].reshape((end - start) * w, f).astype(np.float64)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        y = np.zeros((end - start, w), dtype=np.float64)
        y[np.arange(end - start), train.true_index[start:end]] = 1.0
        target = y.reshape((end - start) * w, 1)
        xtx += x_aug.T @ x_aug
        xty += x_aug.T @ target
    return np.linalg.solve(xtx, xty).reshape(-1)


def input_only_logits(batch: ResidualRankBatch, weights: np.ndarray) -> np.ndarray:
    b, w, f = batch.features.shape
    x = batch.features.reshape(b * w, f).astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return (x_aug @ weights.reshape(-1, 1)).reshape(b, w)


def evaluate_rank_model(
    cfg: RunConfig,
    model: ResidualRankNet,
    device: torch.device,
    samples: int,
    seed: int,
    sieve_bound: Optional[int] = None,
) -> Dict[str, Any]:
    batch = build_residual_rank_batch(cfg.env, samples, seed=seed, sieve_bound=sieve_bound)
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, batch.features.shape[0], 128):
            x = torch.from_numpy(batch.features[start : start + 128]).float().to(device)
            logits, prime_logits = model(x)
            outs.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(outs, axis=0) if outs else np.zeros((0, cfg.env.residual_rank_window_size), dtype=np.float32)
    out = rank_metrics(logits_np, batch.true_index)
    out.update(rank_batch_diagnostics(batch))
    out["calibration_bins"] = calibration_bins(logits_np, batch.true_index)
    return out


def evaluate_counterfactual_ranker(
    cfg: RunConfig,
    score_fn,
    samples: int,
    seed: int,
    sieve_bound: int,
) -> Dict[str, float]:
    pairs = build_counterfactual_pairs(cfg.env, samples=max(4, samples // 16), seed=seed, bound=sieve_bound)
    correct = 0
    total = 0
    margins: List[float] = []
    for pair in pairs:
        for n, true_idx in [(pair.n_a, pair.true_index_a), (pair.n_b, pair.true_index_b)]:
            local_cfg = cfg.env
            old_min, old_max = local_cfg.n_min, local_cfg.n_max
            local_cfg.n_min = n
            local_cfg.n_max = n
            batch = build_rank_batch(local_cfg, 1, seed=seed + total, sieve_bound=sieve_bound)
            local_cfg.n_min, local_cfg.n_max = old_min, old_max
            logits = score_fn(batch)
            pred = int(np.argmax(logits[0]))
            correct += int(pred == true_idx)
            scores = logits[0]
            if 0 <= true_idx < scores.shape[0]:
                margins.append(float(scores[true_idx] - np.max(np.delete(scores, true_idx))))
            total += 1
    accuracy = correct / max(total, 1)
    return {
        "counterfactual_pair_accuracy": float(accuracy),
        "counterfactual_margin": float(np.mean(margins)) if margins else 0.0,
        "matched_pattern_generalization_score": float(accuracy - 1.0 / max(cfg.env.residual_rank_window_size, 1)),
    }


def batch_to_torch(batch: ResidualRankBatch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    features = torch.from_numpy(batch.features).float().to(device)
    target = torch.from_numpy(batch.true_index).long().to(device)
    prime = torch.from_numpy(batch.prime_labels).float().to(device) if batch.prime_labels is not None else None
    return features, target, prime


def train_loss_for_batch(
    cfg: RunConfig,
    model: ResidualRankNet,
    batch: ResidualRankBatch,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    features, target, prime = batch_to_torch(batch, device)
    logits, prime_logits = model(features)
    ce = F.cross_entropy(logits, target)
    loss = ce
    bce_value = torch.tensor(0.0, device=device)
    if cfg.env.residual_rank_target == "survivor_primality" and prime is not None:
        bce_value = F.binary_cross_entropy_with_logits(prime_logits, prime)
        loss = ce + 0.25 * bce_value
    with torch.no_grad():
        metrics = rank_metrics(logits.detach().cpu().numpy(), batch.true_index)
    metrics.update(rank_batch_diagnostics(batch))
    metrics.update({"loss": float(loss.detach().item()), "cross_entropy_loss": float(ce.detach().item()), "binary_primality_loss": float(bce_value.detach().item())})
    return loss, metrics
