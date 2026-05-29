from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .config import load_config
from .counterfactuals import build_counterfactual_pairs
from .model import choose_device
from .oracle import small_prime_sieve
from .residual_rank import (
    ResidualRankBatch,
    baseline_logits,
    build_rank_batch,
    build_residual_rank_batch,
    fit_input_only_ranker,
    input_only_logits,
    rank_metrics,
)
from .residual_rank_controls import _model_logits_chunked, _nearest_fail_metrics, _train_input_only_mlp


DECISION_LEARNABLE = "Residual hard task contains learnable signal beyond nearest/input-only baselines."
DECISION_INFORMATION_LIMITED = "Residual hard task appears information-limited under current features."
DECISION_SIGN_BUG_PREFIX = "Signal audit found a likely metric/sign bug:"


def _clone_batch(
    batch: ResidualRankBatch,
    *,
    features: Optional[np.ndarray] = None,
    true_index: Optional[np.ndarray] = None,
) -> ResidualRankBatch:
    return ResidualRankBatch(
        features=(features if features is not None else batch.features).astype(np.float32),
        true_index=(true_index if true_index is not None else batch.true_index).astype(np.int64),
        prime_labels=batch.prime_labels,
        n_values=batch.n_values,
        true_next_prime=batch.true_next_prime,
        survivor_values=batch.survivor_values,
        metadata=batch.metadata,
    )


def _take_batch(batch: ResidualRankBatch, count: int) -> ResidualRankBatch:
    n = min(max(1, int(count)), int(batch.features.shape[0]))
    return ResidualRankBatch(
        features=batch.features[:n].astype(np.float32),
        true_index=batch.true_index[:n].astype(np.int64),
        prime_labels=batch.prime_labels[:n] if batch.prime_labels is not None else None,
        n_values=batch.n_values[:n],
        true_next_prime=batch.true_next_prime[:n],
        survivor_values=batch.survivor_values[:n],
        metadata=batch.metadata[:n] if batch.metadata is not None else None,
    )


def _select_columns(batch: ResidualRankBatch, columns: Iterable[int]) -> ResidualRankBatch:
    cols = [int(c) for c in columns]
    return _clone_batch(batch, features=batch.features[:, :, cols])


def _metrics_row(condition: str, logits: np.ndarray, batch: ResidualRankBatch) -> Dict[str, Any]:
    row: Dict[str, Any] = {"condition": condition}
    row.update(rank_metrics(logits, batch.true_index))
    row.update(_nearest_fail_metrics(logits, batch))
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for condition in sorted({str(r.get("condition", "")) for r in rows}):
        subset = [r for r in rows if r.get("condition") == condition]
        stats: Dict[str, float] = {}
        keys = sorted({k for r in subset for k, v in r.items() if isinstance(v, (int, float)) and k not in {"seed"}})
        for key in keys:
            vals = np.asarray([float(r[key]) for r in subset if isinstance(r.get(key), (int, float))], dtype=np.float64)
            if vals.size:
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[condition] = stats
    return out


def _mrr(row: Dict[str, float], condition: str) -> float:
    return float(row.get(condition, {}).get("mean_reciprocal_rank_mean", 0.0))


def _ranker_from_linear(train: ResidualRankBatch, eval_batch: ResidualRankBatch) -> Tuple[np.ndarray, np.ndarray]:
    weights = fit_input_only_ranker(train)
    return input_only_logits(eval_batch, weights), weights


def _empirical_position_prior_logits(train: ResidualRankBatch, eval_batch: ResidualRankBatch) -> np.ndarray:
    w = int(eval_batch.survivor_values.shape[1])
    counts = np.bincount(train.true_index, minlength=w).astype(np.float64) + 1.0
    logits = np.log(counts / counts.sum())
    return np.repeat(logits[None, :], eval_batch.true_index.shape[0], axis=0)


def _position_distribution(batch: ResidualRankBatch, train_batch: ResidualRankBatch) -> Dict[str, Any]:
    idx = batch.true_index.astype(np.int64)
    buckets = {
        "0": int(np.sum(idx == 0)),
        "1": int(np.sum(idx == 1)),
        "2": int(np.sum(idx == 2)),
        "3": int(np.sum(idx == 3)),
        "4": int(np.sum(idx == 4)),
        "5": int(np.sum(idx == 5)),
        "6": int(np.sum(idx == 6)),
        "7": int(np.sum(idx == 7)),
        "8": int(np.sum(idx == 8)),
        "9": int(np.sum(idx == 9)),
        "10": int(np.sum(idx == 10)),
        "11_15": int(np.sum((idx >= 11) & (idx <= 15))),
        "16_31": int(np.sum((idx >= 16) & (idx <= 31))),
        "32_63": int(np.sum((idx >= 32) & (idx <= 63))),
        "64_plus": int(np.sum(idx >= 64)),
    }
    prior_logits = _empirical_position_prior_logits(train_batch, batch)
    prior = rank_metrics(prior_logits, batch.true_index)
    return {
        "true_index_histogram": buckets,
        "nearest_success_rate": float(np.mean(idx == 0)),
        "nearest_fail_rate": float(np.mean(idx > 0)),
        "average_true_index": float(np.mean(idx)) if idx.size else 0.0,
        "always_index0_mrr": float(np.mean(1.0 / (idx + 1))) if idx.size else 0.0,
        "empirical_position_prior_mrr": float(prior["mean_reciprocal_rank"]),
        "empirical_position_prior_top1": float(prior["top1_accuracy"]),
    }


def _entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0


def _conditional_entropy_audit(batch: ResidualRankBatch, sieve_bound: int, min_group_size: int = 10) -> Dict[str, Any]:
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    distances = batch.survivor_values[:, :4] - batch.n_values[:, None]
    gaps = np.diff(np.concatenate([batch.n_values[:, None], batch.survivor_values[:, :4]], axis=1), axis=1)
    density = batch.features[:, 0, 6] if batch.features.shape[-1] > 6 else np.zeros(len(batch.true_index))
    for i in range(len(batch.true_index)):
        dist_bins = tuple(int(min(8, max(0, math.log2(max(int(d), 1)) // 2))) for d in distances[i, :4])
        gap_bins = tuple(int(min(8, max(0, int(g) // 24))) for g in gaps[i, :4])
        nearest_fail = bool(batch.true_index[i] > 0)
        density_bucket = int(min(10, max(0, round(float(density[i]) * 80.0))))
        scale_bucket = int(math.log10(max(int(batch.n_values[i]), 3)) * 2)
        key = (int(sieve_bound), dist_bins, gap_bins, nearest_fail, density_bucket, scale_bucket)
        groups.setdefault(key, []).append(int(batch.true_index[i]))

    eligible = [labels for labels in groups.values() if len(labels) >= int(min_group_size)]
    weighted_entropy = 0.0
    weighted_majority = 0.0
    weighted_distinct = 0.0
    total = sum(len(x) for x in eligible)
    for labels in eligible:
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=batch.survivor_values.shape[1])
        weight = len(labels) / max(total, 1)
        weighted_entropy += weight * _entropy_from_counts(counts)
        weighted_majority += weight * (float(counts.max()) / max(float(counts.sum()), 1.0))
        weighted_distinct += weight * float(np.count_nonzero(counts))

    return {
        "group_count": int(len(groups)),
        "eligible_group_count": int(len(eligible)),
        "min_group_size": int(min_group_size),
        "weighted_label_entropy": float(weighted_entropy),
        "weighted_majority_label_accuracy": float(weighted_majority),
        "weighted_distinct_true_indices": float(weighted_distinct),
    }


def _evenly_spaced(values: List[int], count: int) -> List[int]:
    if len(values) <= count:
        return list(values)
    idx = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(i)] for i in sorted(set(idx.tolist()))]


def _oracle_primes(sieve_bound: int) -> List[int]:
    near = [p for p in small_prime_sieve(1000) if p > int(sieve_bound)]
    far = [p for p in small_prime_sieve(5000) if p > 1000]
    return _evenly_spaced(near, 16) + _evenly_spaced(far, 8)


def _append_oracle_features(batch: ResidualRankBatch, primes: List[int]) -> ResidualRankBatch:
    survivors = batch.survivor_values.astype(np.int64)
    extras: List[np.ndarray] = []
    for p in primes:
        rem = survivors % int(p)
        extras.append((rem == 0).astype(np.float32)[..., None])
        extras.append((np.minimum(rem, int(p) - rem).astype(np.float32) / float(p))[..., None])
    if extras:
        features = np.concatenate([batch.features.astype(np.float32), *extras], axis=-1)
    else:
        features = batch.features.astype(np.float32)
    return _clone_batch(batch, features=features)


def _fit_sklearn_baseline(train: ResidualRankBatch, eval_batch: ResidualRankBatch, seed: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        return None, {"sklearn_available": False, "sklearn_error": f"{type(exc).__name__}: {exc}"}

    rng = np.random.default_rng(seed)
    b, w, f = train.features.shape
    x = train.features.reshape(b * w, f)
    y = np.zeros((b, w), dtype=np.int8)
    y[np.arange(b), train.true_index] = 1
    y_flat = y.reshape(b * w)
    pos = np.flatnonzero(y_flat == 1)
    neg = np.flatnonzero(y_flat == 0)
    neg_keep = rng.choice(neg, size=min(len(neg), max(10 * len(pos), 10_000)), replace=False)
    keep = np.concatenate([pos, neg_keep])
    rng.shuffle(keep)
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=seed)
    clf.fit(x[keep], y_flat[keep])
    probs = clf.predict_proba(eval_batch.features.reshape(eval_batch.features.shape[0] * eval_batch.features.shape[1], f))[:, 1]
    return probs.reshape(eval_batch.features.shape[0], eval_batch.features.shape[1]), {
        "sklearn_available": True,
        "sklearn_model": "HistGradientBoostingClassifier",
        "sklearn_train_rows": int(len(keep)),
    }


def _counterfactual_sign_sanity(
    cfg,
    score_fn: Callable[[ResidualRankBatch], np.ndarray],
    samples: int,
    seed: int,
    sieve_bound: int,
) -> Dict[str, Any]:
    pair_count = max(32, min(512, int(samples) // 16))
    pairs = build_counterfactual_pairs(cfg.env, samples=pair_count, seed=seed, bound=sieve_bound)
    correct = 0
    inverted = 0
    total = 0
    true_scores: List[float] = []
    best_other_scores: List[float] = []
    margins: List[float] = []
    for pair in pairs:
        for n, true_idx in [(pair.n_a, pair.true_index_a), (pair.n_b, pair.true_index_b)]:
            old_min, old_max = cfg.env.n_min, cfg.env.n_max
            cfg.env.n_min = int(n)
            cfg.env.n_max = int(n)
            try:
                batch = build_rank_batch(cfg.env, 1, seed=seed + total + 17, sieve_bound=sieve_bound)
            finally:
                cfg.env.n_min, cfg.env.n_max = old_min, old_max
            logits = np.asarray(score_fn(batch), dtype=np.float64)[0]
            if not (0 <= int(true_idx) < logits.shape[0]):
                continue
            pred = int(np.argmax(logits))
            anti_pred = int(np.argmin(logits))
            correct += int(pred == int(true_idx))
            inverted += int(anti_pred == int(true_idx))
            other = np.delete(logits, int(true_idx))
            true_scores.append(float(logits[int(true_idx)]))
            best_other = float(np.max(other)) if other.size else 0.0
            best_other_scores.append(best_other)
            margins.append(float(logits[int(true_idx)] - best_other))
            total += 1
    accuracy = correct / max(total, 1)
    inverted_accuracy = inverted / max(total, 1)
    return {
        "counterfactual_pair_accuracy": float(accuracy),
        "inverted_counterfactual_pair_accuracy": float(inverted_accuracy),
        "pair_score_mean_positive": float(np.mean(true_scores)) if true_scores else 0.0,
        "pair_score_mean_negative": float(np.mean(best_other_scores)) if best_other_scores else 0.0,
        "counterfactual_margin": float(np.mean(margins)) if margins else 0.0,
        "counterfactual_pairs": int(len(pairs)),
        "counterfactual_examples": int(total),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Residual Signal Audit",
        "",
        "This audit checks whether the residual hard dataset contains learnable signal in the visible features. It is not a discovery claim.",
        "",
        f"Config: `{summary['config']}`",
        f"Samples: `{summary['samples']}`",
        f"Sieve bound: `{summary['sieve_bound']}`",
        f"Seeds: `{summary['seeds']}`",
        "",
        "## Baselines",
        "",
        "| condition | top1 | top3 | MRR | nearest-fail MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, stats in summary["aggregates"].items():
        if "mean_reciprocal_rank_mean" not in stats:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(stats.get("top1_accuracy_mean")),
                    _fmt(stats.get("top3_accuracy_mean")),
                    _fmt(stats.get("mean_reciprocal_rank_mean")),
                    _fmt(stats.get("nearest_fail_mean_reciprocal_rank_mean")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Key Checks",
            "",
            f"- counterfactual_pair_accuracy: `{_fmt(summary['counterfactual_sign_sanity']['counterfactual_pair_accuracy_mean'])}`",
            f"- inverted_counterfactual_pair_accuracy: `{_fmt(summary['counterfactual_sign_sanity']['inverted_counterfactual_pair_accuracy_mean'])}`",
            f"- true_minus_shuffled_mrr: `{_fmt(summary['label_permutation']['true_minus_shuffled_mrr'])}`",
            f"- best_strong_baseline_mrr_minus_nearest_mrr: `{_fmt(summary['strong_baselines']['best_current_feature_mrr_minus_nearest_mrr'])}`",
            f"- oracle_feature_delta: `{_fmt(summary['oracle_feature_test']['oracle_feature_delta'])}`",
            f"- oracle_samples_used: `{summary['oracle_feature_test']['oracle_samples_used']}`",
            "",
            "## Position Distribution",
            "",
        ]
    )
    pos = summary["position_distribution"]
    lines.extend(
        [
            f"- nearest_success_rate: `{_fmt(pos['nearest_success_rate_mean'])}`",
            f"- nearest_fail_rate: `{_fmt(pos['nearest_fail_rate_mean'])}`",
            f"- average_true_index: `{_fmt(pos['average_true_index_mean'])}`",
            f"- always_index0_mrr: `{_fmt(pos['always_index0_mrr_mean'])}`",
            f"- empirical_position_prior_mrr: `{_fmt(pos['empirical_position_prior_mrr_mean'])}`",
            "",
            "## Conditional Entropy",
            "",
        ]
    )
    ent = summary["conditional_entropy"]
    lines.extend(
        [
            f"- eligible_group_count: `{_fmt(ent['eligible_group_count_mean'])}`",
            f"- weighted_label_entropy: `{_fmt(ent['weighted_label_entropy_mean'])}`",
            f"- weighted_majority_label_accuracy: `{_fmt(ent['weighted_majority_label_accuracy_mean'])}`",
            f"- weighted_distinct_true_indices: `{_fmt(ent['weighted_distinct_true_indices_mean'])}`",
            "",
            "## Decision Checks",
            "",
        ]
    )
    for key, value in summary["decision_checks"].items():
        lines.append(f"- `{key}`: {'pass' if value else 'fail'}")
    lines.extend(["", "No discovery claim is made.", "", summary["decision"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_dicts(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows if isinstance(row.get(key), (int, float))], dtype=np.float64)
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return out


def run_signal_audit(
    config: str,
    out_dir: str,
    samples: int,
    sieve_bound: int,
    seeds: int,
) -> Dict[str, Any]:
    cfg = load_config(config)
    cfg.env.residual_rank_mode = True
    cfg.env.residual_rank_hard_mode = True
    cfg.env.residual_rank_sieve_bound = int(sieve_bound)
    original_hard_pool_multiplier = int(cfg.env.residual_rank_hard_pool_multiplier)
    cfg.env.residual_rank_hard_pool_multiplier = min(original_hard_pool_multiplier, 2)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    device = choose_device(cfg.train.device)

    requested_samples = int(samples)
    effective_samples = min(requested_samples, 10_000)
    train_steps = 40 if effective_samples <= 3000 else 300
    batch_size = min(256, max(64, int(effective_samples)))
    oracle_samples = min(int(effective_samples), 1024)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    cf_rows: List[Dict[str, Any]] = []
    entropy_rows: List[Dict[str, Any]] = []
    position_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    sklearn_rows: List[Dict[str, Any]] = []

    for seed_index in range(1, int(seeds) + 1):
        seed = 710_000 + 1009 * seed_index + int(sieve_bound)
        print(
            f"[residual-signal-audit] seed {seed_index}/{seeds}: building hard batches "
            f"({effective_samples} effective samples, pool multiplier {cfg.env.residual_rank_hard_pool_multiplier})",
            flush=True,
        )
        cfg.env.residual_rank_sieve_bound = int(sieve_bound)
        train_batch = build_residual_rank_batch(cfg.env, effective_samples, seed=seed, sieve_bound=sieve_bound)
        eval_batch = build_residual_rank_batch(cfg.env, effective_samples, seed=seed + 17, sieve_bound=sieve_bound)

        print(f"[residual-signal-audit] seed {seed_index}/{seeds}: baseline and linear diagnostics", flush=True)
        nearest_logits = baseline_logits(eval_batch, "nearest", seed=seed)
        row = _metrics_row("nearest", nearest_logits, eval_batch)
        row.update({"seed": seed, "seed_index": seed_index})
        rows.append(row)

        random_logits = baseline_logits(eval_batch, "random", seed=seed)
        row = _metrics_row("random", random_logits, eval_batch)
        row.update({"seed": seed, "seed_index": seed_index})
        rows.append(row)

        prior_logits = _empirical_position_prior_logits(train_batch, eval_batch)
        row = _metrics_row("empirical_position_prior", prior_logits, eval_batch)
        row.update({"seed": seed, "seed_index": seed_index})
        rows.append(row)

        for condition, cols in [
            ("distance_only_linear", [0, 1, 2, 5]),
            ("density_only_linear", [3, 4, 6, 7]),
            ("input_only_linear", list(range(eval_batch.features.shape[-1]))),
        ]:
            train_part = _select_columns(train_batch, cols)
            eval_part = _select_columns(eval_batch, cols)
            logits, _weights = _ranker_from_linear(train_part, eval_part)
            row = _metrics_row(condition, logits, eval_part)
            row.update({"seed": seed, "seed_index": seed_index})
            rows.append(row)

        print(f"[residual-signal-audit] seed {seed_index}/{seeds}: trained input-only and permutation diagnostics", flush=True)
        true_logits, true_mlp = _train_input_only_mlp(
            train_batch,
            eval_batch,
            device=device,
            seed=seed + 101,
            steps=train_steps,
            batch_size=batch_size,
            hidden_dim=64,
            eval_chunk_size=2048,
        )
        row = _metrics_row("trained_input_only_mlp_true", true_logits, eval_batch)
        row.update({"seed": seed, "seed_index": seed_index, "train_steps": train_steps})
        rows.append(row)

        shuffled_idx = train_batch.true_index.copy()
        np.random.default_rng(seed + 202).shuffle(shuffled_idx)
        shuffled_train = _clone_batch(train_batch, true_index=shuffled_idx)
        shuffled_logits, _shuffled_mlp = _train_input_only_mlp(
            shuffled_train,
            eval_batch,
            device=device,
            seed=seed + 203,
            steps=train_steps,
            batch_size=batch_size,
            hidden_dim=64,
            eval_chunk_size=2048,
        )
        row = _metrics_row("trained_input_only_mlp_shuffled", shuffled_logits, eval_batch)
        row.update({"seed": seed, "seed_index": seed_index, "train_steps": train_steps})
        rows.append(row)

        print(f"[residual-signal-audit] seed {seed_index}/{seeds}: sklearn/counterfactual/entropy/oracle diagnostics", flush=True)
        sklearn_logits, sklearn_meta = _fit_sklearn_baseline(train_batch, eval_batch, seed=seed + 303)
        sklearn_meta.update({"seed": seed, "seed_index": seed_index, "condition": "sklearn_hist_gradient_boosting"})
        if sklearn_logits is not None:
            row = _metrics_row("sklearn_hist_gradient_boosting", sklearn_logits, eval_batch)
            row.update(sklearn_meta)
            rows.append(row)
        sklearn_rows.append(sklearn_meta)

        score_fn = lambda batch, model=true_mlp: _model_logits_chunked(model, batch.features, device, chunk_size=2048)
        cf = _counterfactual_sign_sanity(cfg, score_fn, samples=samples, seed=seed + 404, sieve_bound=sieve_bound)
        cf.update({"seed": seed, "seed_index": seed_index, "condition": "counterfactual_sign_sanity"})
        cf_rows.append(cf)
        rows.append(cf)

        entropy = _conditional_entropy_audit(eval_batch, sieve_bound=sieve_bound)
        entropy.update({"seed": seed, "seed_index": seed_index, "condition": "conditional_entropy"})
        entropy_rows.append(entropy)
        rows.append(entropy)

        pos = _position_distribution(eval_batch, train_batch)
        pos_flat = {k: v for k, v in pos.items() if isinstance(v, (int, float))}
        pos_flat.update({"seed": seed, "seed_index": seed_index, "condition": "position_distribution"})
        position_rows.append(pos_flat)
        rows.append(pos_flat)

        oracle_train = _take_batch(train_batch, oracle_samples)
        oracle_eval = _take_batch(eval_batch, oracle_samples)
        oracle_primes = _oracle_primes(sieve_bound)
        base_logits, _base_model = _train_input_only_mlp(
            oracle_train,
            oracle_eval,
            device=device,
            seed=seed + 505,
            steps=train_steps,
            batch_size=batch_size,
            hidden_dim=64,
            eval_chunk_size=2048,
        )
        oracle_train_plus = _append_oracle_features(oracle_train, oracle_primes)
        oracle_eval_plus = _append_oracle_features(oracle_eval, oracle_primes)
        oracle_logits, _oracle_model = _train_input_only_mlp(
            oracle_train_plus,
            oracle_eval_plus,
            device=device,
            seed=seed + 506,
            steps=train_steps,
            batch_size=batch_size,
            hidden_dim=64,
            eval_chunk_size=2048,
        )
        base_metrics = rank_metrics(base_logits, oracle_eval.true_index)
        oracle_metrics = rank_metrics(oracle_logits, oracle_eval_plus.true_index)
        oracle_row = {
            "condition": "oracle_feature_test",
            "seed": seed,
            "seed_index": seed_index,
            "oracle_samples_used": int(oracle_samples),
            "oracle_prime_count": int(len(oracle_primes)),
            "mrr_without_oracle_features": float(base_metrics["mean_reciprocal_rank"]),
            "mrr_with_extra_residue_features": float(oracle_metrics["mean_reciprocal_rank"]),
            "oracle_feature_delta": float(oracle_metrics["mean_reciprocal_rank"] - base_metrics["mean_reciprocal_rank"]),
        }
        oracle_rows.append(oracle_row)
        rows.append(oracle_row)

    aggregates = _aggregate([r for r in rows if "mean_reciprocal_rank" in r])
    cf_agg = _mean_dicts(cf_rows)
    entropy_agg = _mean_dicts(entropy_rows)
    position_agg = _mean_dicts(position_rows)
    oracle_agg = _mean_dicts(oracle_rows)
    sklearn_agg = {
        "available_any_seed": bool(any(r.get("sklearn_available") for r in sklearn_rows)),
        "rows": sklearn_rows,
    }

    nearest_mrr = _mrr(aggregates, "nearest")
    current_feature_conditions = [
        "distance_only_linear",
        "density_only_linear",
        "input_only_linear",
        "trained_input_only_mlp_true",
        "empirical_position_prior",
    ]
    if "sklearn_hist_gradient_boosting" in aggregates:
        current_feature_conditions.append("sklearn_hist_gradient_boosting")
    best_condition = max(current_feature_conditions, key=lambda name: _mrr(aggregates, name))
    best_current_mrr = _mrr(aggregates, best_condition)
    true_mrr = _mrr(aggregates, "trained_input_only_mlp_true")
    shuffled_mrr = _mrr(aggregates, "trained_input_only_mlp_shuffled")
    true_minus_shuffled = true_mrr - shuffled_mrr
    oracle_delta = float(oracle_agg.get("oracle_feature_delta_mean", 0.0))
    cf_acc = float(cf_agg.get("counterfactual_pair_accuracy_mean", 0.0))
    inverted_cf_acc = float(cf_agg.get("inverted_counterfactual_pair_accuracy_mean", 0.0))

    sign_bug = bool(inverted_cf_acc - cf_acc >= 0.10)
    learnable_current = bool(true_minus_shuffled >= 0.02 and best_current_mrr - nearest_mrr >= 0.02)
    if sign_bug:
        details = f"inverted_counterfactual_accuracy exceeds accuracy by {inverted_cf_acc - cf_acc:.4f}"
        decision = f"{DECISION_SIGN_BUG_PREFIX} {details}."
    elif learnable_current:
        decision = DECISION_LEARNABLE
    else:
        decision = DECISION_INFORMATION_LIMITED

    decision_checks = {
        "true_minus_shuffled_mrr >= 0.02": bool(true_minus_shuffled >= 0.02),
        "best_strong_baseline_mrr_minus_nearest_mrr >= 0.02": bool(best_current_mrr - nearest_mrr >= 0.02),
        "oracle_feature_delta >= 0.02": bool(oracle_delta >= 0.02),
        "inverted_counterfactual_accuracy_minus_accuracy >= 0.10": sign_bug,
    }
    summary = {
        "config": config,
        "requested_samples": int(requested_samples),
        "samples": int(effective_samples),
        "sieve_bound": int(sieve_bound),
        "seeds": int(seeds),
        "original_hard_pool_multiplier": int(original_hard_pool_multiplier),
        "effective_hard_pool_multiplier": int(cfg.env.residual_rank_hard_pool_multiplier),
        "train_steps_per_mlp": int(train_steps),
        "batch_size": int(batch_size),
        "rows": rows,
        "aggregates": aggregates,
        "counterfactual_sign_sanity": cf_agg,
        "label_permutation": {
            "true_label_mrr": float(true_mrr),
            "shuffled_label_mrr": float(shuffled_mrr),
            "true_minus_shuffled_mrr": float(true_minus_shuffled),
        },
        "strong_baselines": {
            "nearest_mrr": float(nearest_mrr),
            "best_current_feature_condition": best_condition,
            "best_current_feature_mrr": float(best_current_mrr),
            "best_current_feature_mrr_minus_nearest_mrr": float(best_current_mrr - nearest_mrr),
        },
        "oracle_feature_test": {
            **oracle_agg,
            "mrr_without_oracle_features": float(oracle_agg.get("mrr_without_oracle_features_mean", 0.0)),
            "mrr_with_extra_residue_features": float(oracle_agg.get("mrr_with_extra_residue_features_mean", 0.0)),
            "oracle_feature_delta": float(oracle_delta),
            "oracle_samples_used": int(oracle_samples),
            "oracle_primes": _oracle_primes(sieve_bound),
        },
        "sklearn": sklearn_agg,
        "conditional_entropy": entropy_agg,
        "position_distribution": position_agg,
        "decision_checks": decision_checks,
        "decision": decision,
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "summary.csv", rows)
    _write_report(out / "SIGNAL_AUDIT_REPORT.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/residual_rank_hard_probe.json")
    parser.add_argument("--out-dir", default="runs/residual_signal_audit_001")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--sieve-bound", type=int, default=211)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    summary = run_signal_audit(
        config=args.config,
        out_dir=args.out_dir,
        samples=args.samples,
        sieve_bound=args.sieve_bound,
        seeds=args.seeds,
    )
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
