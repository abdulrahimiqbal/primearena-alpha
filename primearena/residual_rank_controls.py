from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import load_config
from .model import choose_device
from .residual_rank import (
    baseline_logits,
    build_residual_rank_batch,
    evaluate_counterfactual_ranker,
    fit_input_only_ranker,
    input_only_logits,
    load_rank_checkpoint,
    make_rank_model,
    rank_batch_diagnostics,
    rank_metrics,
)


BASELINES = ["nearest", "random", "uniform", "distance_weighted", "one_over_logn", "survivor_density"]
DECISION_SUPPORTS = "Residual hard controls support signal beyond nearest/input-only/random baselines."
DECISION_REJECTS = "Residual hard controls do not support signal beyond nearest/input-only/random baselines."


def _latest_rank_checkpoint() -> Optional[str]:
    paths = sorted(glob.glob("runs/residual_rank*/checkpoints/best_by_mrr.pt"), key=lambda p: Path(p).stat().st_mtime)
    return paths[-1] if paths else None


class InputOnlyMLP(nn.Module):
    """Shared per-survivor input-only ranker.

    It sees only the residual-rank feature rows. There is no attention, no
    cross-candidate message passing, and no hidden state from the trained
    residual ranker.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _model_logits_chunked(model: nn.Module, features: np.ndarray, device: torch.device, chunk_size: int = 512) -> np.ndarray:
    outs: List[np.ndarray] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], max(1, int(chunk_size))):
            x = torch.from_numpy(features[start : start + chunk_size]).float().to(device)
            y = model(x)
            logits = y[0] if isinstance(y, tuple) else y
            outs.append(logits.detach().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, features.shape[1]), dtype=np.float32)


def _train_input_only_mlp(
    train_batch,
    eval_batch,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float = 1e-3,
    eval_chunk_size: int = 2048,
) -> Tuple[np.ndarray, InputOnlyMLP]:
    torch.manual_seed(int(seed))
    model = InputOnlyMLP(train_batch.features.shape[-1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    features = torch.from_numpy(train_batch.features).float()
    targets = torch.from_numpy(train_batch.true_index).long()
    n = int(features.shape[0])
    rng = np.random.default_rng(seed)
    for _ in range(max(1, int(steps))):
        idx = rng.integers(0, n, size=min(max(1, int(batch_size)), n))
        x = features[idx].to(device)
        y = targets[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    logits = _model_logits_chunked(model, eval_batch.features, device, chunk_size=eval_chunk_size)
    return logits, model


def _nearest_fail_metrics(logits: np.ndarray, batch) -> Dict[str, float]:
    mask = batch.true_index > 0
    if not np.any(mask):
        return {
            "nearest_fail_top1_accuracy": 0.0,
            "nearest_fail_top3_accuracy": 0.0,
            "nearest_fail_top5_accuracy": 0.0,
            "nearest_fail_mean_reciprocal_rank": 0.0,
            "nearest_fail_cross_entropy": 0.0,
            "nearest_fail_avg_true_rank": 0.0,
            "nearest_fail_count": 0,
        }
    metrics = rank_metrics(logits[mask], batch.true_index[mask])
    return {
        "nearest_fail_top1_accuracy": metrics["top1_accuracy"],
        "nearest_fail_top3_accuracy": metrics["top3_accuracy"],
        "nearest_fail_top5_accuracy": metrics["top5_accuracy"],
        "nearest_fail_mean_reciprocal_rank": metrics["mean_reciprocal_rank"],
        "nearest_fail_cross_entropy": metrics["cross_entropy"],
        "nearest_fail_avg_true_rank": metrics["avg_true_rank"],
        "nearest_fail_count": int(mask.sum()),
    }


def _eval_logits(name: str, logits: np.ndarray, batch, score_fn: Callable, cfg, seed: int, sieve_bound: int) -> Dict[str, Any]:
    metrics = rank_metrics(logits, batch.true_index)
    metrics.update(rank_batch_diagnostics(batch))
    metrics.update(_nearest_fail_metrics(logits, batch))
    cf = evaluate_counterfactual_ranker(cfg, score_fn, samples=max(64, len(batch.true_index)), seed=seed + 909, sieve_bound=sieve_bound)
    return {"condition": name, **metrics, **cf}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for condition in sorted({str(r["condition"]) for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        stats: Dict[str, float] = {}
        keys = [k for k, v in subset[0].items() if isinstance(v, (int, float)) and k not in {"seed", "seed_index", "sieve_bound"}]
        for key in keys:
            vals = np.asarray([float(r[key]) for r in subset if isinstance(r.get(key), (int, float))], dtype=np.float64)
            if vals.size:
                stats[f"{key}_mean"] = float(np.mean(vals))
                stats[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[condition] = stats
    return out


def _comparison(aggs: Dict[str, Dict[str, float]], condition: str, baseline: str) -> Dict[str, Optional[float]]:
    a = aggs.get(condition, {})
    b = aggs.get(baseline, {})
    keys = [
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "mean_reciprocal_rank",
        "cross_entropy",
        "avg_true_rank",
        "nearest_fail_top1_accuracy",
        "nearest_fail_mean_reciprocal_rank",
        "counterfactual_pair_accuracy",
    ]
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        av = a.get(f"{key}_mean")
        bv = b.get(f"{key}_mean")
        out[f"delta_{key}_vs_{baseline}"] = None if av is None or bv is None else float(av - bv)
    return out


def _paired_rows(rows: List[Dict[str, Any]], a: str, b: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_key: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition") not in {a, b} or "top1_accuracy" not in row:
            continue
        key = (int(row["sieve_bound"]), int(row["seed_index"]))
        by_key.setdefault(key, {})[str(row["condition"])] = row
    return [(pair[a], pair[b]) for pair in by_key.values() if a in pair and b in pair]


def _bootstrap_delta_ci(
    rows: List[Dict[str, Any]],
    condition: str,
    baseline: str,
    metric: str,
    seed: int = 12345,
    rounds: int = 2000,
) -> Dict[str, float]:
    pairs = _paired_rows(rows, condition, baseline)
    if not pairs:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "pairs": 0}
    deltas = np.asarray([float(a[metric]) - float(b[metric]) for a, b in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(1, int(rounds))):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        boot.append(float(np.mean(deltas[idx])))
    return {
        "mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "pairs": int(len(deltas)),
    }


def _seed_repeat_count(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"passing_seed_count": 0, "seed_results": []}
    seeds = sorted({int(r["seed_index"]) for r in rows if "seed_index" in r})
    for seed_index in seeds:
        subset = [r for r in rows if int(r.get("seed_index", -1)) == seed_index]
        aggs = _aggregate([r for r in subset if "top1_accuracy" in r])
        cmp = _comparison(aggs, "trained_rank", "trained_input_only")
        mrr_delta = cmp.get("delta_mean_reciprocal_rank_vs_trained_input_only") or 0.0
        top1_delta = cmp.get("delta_top1_accuracy_vs_trained_input_only") or 0.0
        nf_delta = cmp.get("delta_nearest_fail_mean_reciprocal_rank_vs_trained_input_only") or 0.0
        passed = bool(mrr_delta >= 0.02 and top1_delta >= 0.02 and nf_delta >= 0.03)
        out["passing_seed_count"] += int(passed)
        out["seed_results"].append(
            {
                "seed_index": seed_index,
                "mrr_delta_vs_trained_input_only": float(mrr_delta),
                "top1_delta_vs_trained_input_only": float(top1_delta),
                "nearest_fail_mrr_delta_vs_trained_input_only": float(nf_delta),
                "passes_effect_gates": passed,
            }
        )
    return out


def _decision(summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    cmp = summary["comparisons"].get("trained_vs_trained_input_only", {})
    aggs = summary["aggregates"]
    repeats = summary.get("seed_repeatability", {})
    gates = {
        "mrr_delta_vs_trained_input_only": float(cmp.get("delta_mean_reciprocal_rank_vs_trained_input_only") or 0.0),
        "top1_delta_vs_trained_input_only": float(cmp.get("delta_top1_accuracy_vs_trained_input_only") or 0.0),
        "nearest_fail_mrr_delta_vs_trained_input_only": float(cmp.get("delta_nearest_fail_mean_reciprocal_rank_vs_trained_input_only") or 0.0),
        "counterfactual_pair_accuracy": float(aggs.get("trained_rank", {}).get("counterfactual_pair_accuracy_mean") or 0.0),
        "passing_seed_count": int(repeats.get("passing_seed_count", 0)),
    }
    checks = {
        "mrr_delta_vs_trained_input_only >= 0.02": gates["mrr_delta_vs_trained_input_only"] >= 0.02,
        "top1_delta_vs_trained_input_only >= 0.02": gates["top1_delta_vs_trained_input_only"] >= 0.02,
        "nearest_fail_mrr_delta_vs_trained_input_only >= 0.03": gates["nearest_fail_mrr_delta_vs_trained_input_only"] >= 0.03,
        "counterfactual_pair_accuracy >= 0.55": gates["counterfactual_pair_accuracy"] >= 0.55,
        "effect_repeats_across_at_least_3_seeds": gates["passing_seed_count"] >= 3,
    }
    verdict = DECISION_SUPPORTS if all(checks.values()) else DECISION_REJECTS
    return verdict, {"values": gates, "checks": checks}


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Residual Prime Rank Controls",
        "",
        "This report evaluates prediction/ranking over pre-sieved survivors. It is not a discovery claim.",
        f"Hard sampling: **{bool(summary.get('hard_mode', False))}**",
        f"Solver-uncertainty adversary: **{bool(summary.get('hard_solver_uncertainty', False))}**",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "| condition | top1 | top5 | MRR | CE | avg rank | nearest fail | input wrong | matched | counterfactual acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, stats in summary["aggregates"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(stats.get("top1_accuracy_mean")),
                    _fmt(stats.get("top5_accuracy_mean")),
                    _fmt(stats.get("mean_reciprocal_rank_mean")),
                    _fmt(stats.get("cross_entropy_mean")),
                    _fmt(stats.get("avg_true_rank_mean")),
                    _fmt(stats.get("nearest_fail_rate_mean")),
                    _fmt(stats.get("input_only_wrong_mean_mean")),
                    _fmt(stats.get("matched_pattern_mean_mean")),
                    _fmt(stats.get("counterfactual_pair_accuracy_mean")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Comparisons", ""])
    for name, cmp in summary["comparisons"].items():
        lines.append(f"### {name}")
        for key, value in cmp.items():
            lines.append(f"- `{key}`: {_fmt(value)}")
    lines.extend(["", "No discovery claim is made.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision_report(path: Path, summary: Dict[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "# ResidualPrimeRank-Hard Decision Report",
        "",
        "This is a control report, not a discovery claim.",
        "",
        f"Rank checkpoint: `{summary.get('rank_checkpoint')}`",
        f"Samples per bound/seed: `{summary['samples']}`",
        f"Seeds: `{summary['seeds']}`",
        f"Sieve bounds: `{summary['sieve_bounds']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| condition | top1 | MRR | nearest-fail MRR | counterfactual acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ["trained_rank", "trained_input_only", "nearest", "random", "random_model"]:
        stats = summary["aggregates"].get(condition, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(stats.get("top1_accuracy_mean")),
                    _fmt(stats.get("mean_reciprocal_rank_mean")),
                    _fmt(stats.get("nearest_fail_mean_reciprocal_rank_mean")),
                    _fmt(stats.get("counterfactual_pair_accuracy_mean")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision Gates", ""])
    for name, value in decision["values"].items():
        lines.append(f"- `{name}`: {_fmt(value)}")
    lines.append("")
    for name, passed in decision["checks"].items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(["", "## Bootstrap Confidence Intervals", ""])
    for name, ci in summary.get("bootstrap_confidence_intervals", {}).items():
        lines.append(f"- `{name}`: mean={_fmt(ci.get('mean'))}, 95% CI [{_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}], pairs={ci.get('pairs')}")
    lines.extend(["", "## Seed Repeatability", ""])
    for row in summary.get("seed_repeatability", {}).get("seed_results", []):
        lines.append(
            f"- seed {row['seed_index']}: mrr_delta={_fmt(row['mrr_delta_vs_trained_input_only'])}, "
            f"top1_delta={_fmt(row['top1_delta_vs_trained_input_only'])}, "
            f"nearest_fail_mrr_delta={_fmt(row['nearest_fail_mrr_delta_vs_trained_input_only'])}, "
            f"passes={row['passes_effect_gates']}"
        )
    lines.extend(["", "No discovery claim is made.", "", summary["decision_verdict"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_controls(
    config: str,
    out_dir: str,
    sieve_bounds: List[int],
    samples: int,
    seeds: int,
    rank_checkpoint: Optional[str] = None,
    hard_mode: Optional[bool] = None,
    window_size: Optional[int] = None,
    model_chunk_size: int = 512,
    trained_input_steps: int = 1000,
    trained_input_batch_size: int = 512,
    trained_input_hidden_dim: int = 64,
    bootstrap_rounds: int = 2000,
) -> Dict[str, Any]:
    cfg = load_config(config)
    cfg.env.residual_rank_mode = True
    if hard_mode is not None:
        cfg.env.residual_rank_hard_mode = bool(hard_mode)
    if window_size is not None:
        cfg.env.residual_rank_window_size = int(window_size)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    device = choose_device(cfg.train.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if rank_checkpoint is None:
        rank_checkpoint = _latest_rank_checkpoint()

    rows: List[Dict[str, Any]] = []
    for bound in sieve_bounds:
        for seed_index in range(1, int(seeds) + 1):
            seed = 90_000 + int(bound) * 101 + seed_index
            cfg.env.residual_rank_sieve_bound = int(bound)
            train_batch = build_residual_rank_batch(cfg.env, samples, seed=seed, sieve_bound=bound)
            eval_batch = build_residual_rank_batch(cfg.env, samples, seed=seed + 17, sieve_bound=bound)

            for baseline in BASELINES:
                logits = baseline_logits(eval_batch, baseline, seed=seed)
                score_fn = lambda batch, b=baseline: baseline_logits(batch, b, seed=seed)
                row = _eval_logits(baseline, logits, eval_batch, score_fn, cfg, seed, bound)
                row.update({"sieve_bound": bound, "seed_index": seed_index, "seed": seed})
                rows.append(row)

            weights = fit_input_only_ranker(train_batch)
            input_logits = input_only_logits(eval_batch, weights)
            input_score = lambda batch, w=weights: input_only_logits(batch, w)
            row = _eval_logits("input_only", input_logits, eval_batch, input_score, cfg, seed, bound)
            row.update({"sieve_bound": bound, "seed_index": seed_index, "seed": seed})
            rows.append(row)

            trained_input_logits, trained_input_model = _train_input_only_mlp(
                train_batch,
                eval_batch,
                device=device,
                seed=seed + 303,
                steps=trained_input_steps,
                batch_size=trained_input_batch_size,
                hidden_dim=trained_input_hidden_dim,
                eval_chunk_size=max(1, int(model_chunk_size) * 4),
            )
            trained_input_score = lambda batch, m=trained_input_model: _model_logits_chunked(
                m, batch.features, device, chunk_size=max(1, int(model_chunk_size) * 4)
            )
            row = _eval_logits("trained_input_only", trained_input_logits, eval_batch, trained_input_score, cfg, seed, bound)
            row.update(
                {
                    "sieve_bound": bound,
                    "seed_index": seed_index,
                    "seed": seed,
                    "trained_input_steps": int(trained_input_steps),
                    "trained_input_batch_size": int(trained_input_batch_size),
                    "trained_input_hidden_dim": int(trained_input_hidden_dim),
                }
            )
            rows.append(row)

            torch.manual_seed(seed + 404)
            model = make_rank_model(cfg, device)
            model.eval()
            random_logits = _model_logits_chunked(model, eval_batch.features, device, chunk_size=model_chunk_size)
            random_score = lambda batch, m=model: _model_logits_chunked(m, batch.features, device, chunk_size=model_chunk_size)
            row = _eval_logits("random_model", random_logits, eval_batch, random_score, cfg, seed, bound)
            row.update({"sieve_bound": bound, "seed_index": seed_index, "seed": seed})
            rows.append(row)

            if rank_checkpoint:
                trained = make_rank_model(cfg, device)
                try:
                    load_rank_checkpoint(rank_checkpoint, trained, map_location=device)
                    trained.eval()
                    trained_logits = _model_logits_chunked(trained, eval_batch.features, device, chunk_size=model_chunk_size)
                    trained_score = lambda batch, m=trained: _model_logits_chunked(m, batch.features, device, chunk_size=model_chunk_size)
                    row = _eval_logits("trained_rank", trained_logits, eval_batch, trained_score, cfg, seed, bound)
                    row.update({"sieve_bound": bound, "seed_index": seed_index, "seed": seed, "checkpoint": rank_checkpoint})
                    rows.append(row)
                except Exception as exc:
                    rows.append({"condition": "trained_rank", "sieve_bound": bound, "seed_index": seed_index, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    aggs = _aggregate([r for r in rows if "top1_accuracy" in r])
    comparisons: Dict[str, Dict[str, Optional[float]]] = {}
    if "trained_rank" in aggs:
        for baseline in ["nearest", "random", "input_only", "trained_input_only", "random_model"]:
            comparisons[f"trained_vs_{baseline}"] = _comparison(aggs, "trained_rank", baseline)
    else:
        comparisons["input_only_vs_nearest"] = _comparison(aggs, "input_only", "nearest")
    seed_repeatability = _seed_repeat_count(rows)
    bootstrap_confidence_intervals = {
        "trained_vs_trained_input_only_mrr": _bootstrap_delta_ci(rows, "trained_rank", "trained_input_only", "mean_reciprocal_rank", rounds=bootstrap_rounds),
        "trained_vs_trained_input_only_top1": _bootstrap_delta_ci(rows, "trained_rank", "trained_input_only", "top1_accuracy", rounds=bootstrap_rounds),
        "trained_vs_trained_input_only_nearest_fail_mrr": _bootstrap_delta_ci(rows, "trained_rank", "trained_input_only", "nearest_fail_mean_reciprocal_rank", rounds=bootstrap_rounds),
        "trained_vs_nearest_mrr": _bootstrap_delta_ci(rows, "trained_rank", "nearest", "mean_reciprocal_rank", rounds=bootstrap_rounds),
        "trained_vs_random_model_mrr": _bootstrap_delta_ci(rows, "trained_rank", "random_model", "mean_reciprocal_rank", rounds=bootstrap_rounds),
    }
    summary = {
        "config": config,
        "rank_checkpoint": rank_checkpoint,
        "hard_mode": bool(cfg.env.residual_rank_hard_mode),
        "hard_solver_uncertainty": bool(cfg.env.residual_rank_hard_solver_uncertainty),
        "samples": int(samples),
        "window_size": int(cfg.env.residual_rank_window_size),
        "seeds": int(seeds),
        "sieve_bounds": [int(x) for x in sieve_bounds],
        "model_chunk_size": int(model_chunk_size),
        "trained_input_steps": int(trained_input_steps),
        "trained_input_batch_size": int(trained_input_batch_size),
        "trained_input_hidden_dim": int(trained_input_hidden_dim),
        "rows": rows,
        "aggregates": aggs,
        "comparisons": comparisons,
        "seed_repeatability": seed_repeatability,
        "bootstrap_confidence_intervals": bootstrap_confidence_intervals,
    }
    verdict, decision = _decision(summary)
    summary["decision_verdict"] = verdict
    summary["decision"] = decision
    summary["conclusion"] = verdict
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "summary.csv", rows)
    _write_report(out / "RESIDUAL_RANK_REPORT.md", summary)
    _write_decision_report(out / "RESIDUAL_RANK_HARD_DECISION_REPORT.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/residual_rank_probe.json")
    parser.add_argument("--out-dir", default="runs/residual_rank_controls_001")
    parser.add_argument("--sieve-bounds", nargs="+", type=int, default=[30, 210, 211, 1000])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rank-checkpoint", default=None)
    parser.add_argument("--imitation-checkpoint", default=None)
    parser.add_argument("--mcts-checkpoint", default=None)
    parser.add_argument("--hard", action="store_true", help="Evaluate on adversarial hard residual-rank batches.")
    parser.add_argument("--no-hard", action="store_true", help="Evaluate on neutral/random residual-rank batches.")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--model-chunk-size", type=int, default=512)
    parser.add_argument("--trained-input-steps", type=int, default=1000)
    parser.add_argument("--trained-input-batch-size", type=int, default=512)
    parser.add_argument("--trained-input-hidden-dim", type=int, default=64)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()
    hard_mode = True if args.hard else False if args.no_hard else None
    summary = run_controls(
        config=args.config,
        out_dir=args.out_dir,
        sieve_bounds=args.sieve_bounds,
        samples=args.samples,
        seeds=args.seeds,
        rank_checkpoint=args.rank_checkpoint,
        hard_mode=hard_mode,
        window_size=args.window_size,
        model_chunk_size=args.model_chunk_size,
        trained_input_steps=args.trained_input_steps,
        trained_input_batch_size=args.trained_input_batch_size,
        trained_input_hidden_dim=args.trained_input_hidden_dim,
        bootstrap_rounds=args.bootstrap_rounds,
    )
    print(json.dumps({"conclusion": summary["conclusion"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
