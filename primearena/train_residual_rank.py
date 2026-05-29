from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import RunConfig, load_config, save_config
from .logging_utils import JsonlLogger
from .model import choose_device
from .residual_rank import build_residual_rank_batch, evaluate_rank_model, load_rank_checkpoint, make_rank_model, rank_batch_diagnostics, rank_metrics, save_rank_checkpoint


def make_run_dir(cfg: RunConfig, run_name: Optional[str]) -> Path:
    root = Path(cfg.train.run_dir)
    name = run_name or cfg.train.run_name or time.strftime("residual_rank_%Y%m%d_%H%M%S")
    out = root / name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


def train_step_for_batch(cfg: RunConfig, model, batch, device: torch.device, micro_batch_size: Optional[int] = None) -> Dict[str, float]:
    n = int(batch.features.shape[0])
    micro = int(micro_batch_size or n)
    micro = max(1, min(micro, n))
    logits_out = []
    ce_total = 0.0
    bce_total = 0.0
    for start in range(0, n, micro):
        end = min(start + micro, n)
        features = torch.from_numpy(batch.features[start:end]).float().to(device)
        target = torch.from_numpy(batch.true_index[start:end]).long().to(device)
        logits, prime_logits = model(features)
        ce = F.cross_entropy(logits, target, reduction="sum")
        loss = ce / max(n, 1)
        ce_total += float(ce.detach().item())
        if cfg.env.residual_rank_target == "survivor_primality" and batch.prime_labels is not None:
            prime = torch.from_numpy(batch.prime_labels[start:end]).float().to(device)
            bce = F.binary_cross_entropy_with_logits(prime_logits, prime, reduction="sum")
            loss = loss + 0.25 * bce / max(n * batch.features.shape[1], 1)
            bce_total += float(bce.detach().item())
        loss.backward()
        logits_out.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(logits_out, axis=0)
    metrics = rank_metrics(logits_np, batch.true_index)
    metrics.update(rank_batch_diagnostics(batch))
    loss_value = ce_total / max(n, 1) + 0.25 * bce_total / max(n * batch.features.shape[1], 1)
    metrics.update(
        {
            "loss": float(loss_value),
            "cross_entropy_loss": float(ce_total / max(n, 1)),
            "binary_primality_loss": float(bce_total / max(n * batch.features.shape[1], 1)),
        }
    )
    return metrics


def train_residual_rank(cfg: RunConfig, run_name: Optional[str] = None, steps: Optional[int] = None, batch_size: Optional[int] = None) -> Path:
    cfg.env.residual_rank_mode = True
    if steps is not None:
        cfg.train.steps = int(steps)
    if batch_size is not None:
        cfg.train.batch_size = int(batch_size)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = choose_device(cfg.train.device)
    model = make_rank_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    run_dir = make_run_dir(cfg, run_name)
    save_config(cfg, run_dir / "config.json")
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    best_mrr = -1.0
    best_top1 = -1.0
    start_step = 1
    if cfg.train.resume_checkpoint:
        payload = load_rank_checkpoint(cfg.train.resume_checkpoint, model, optimizer=optimizer, map_location=device)
        start_step = int(payload.get("step", 0)) + 1
        extra = payload.get("extra", {}) or {}
        if extra.get("metric_name") == "mean_reciprocal_rank":
            best_mrr = float(extra.get("metric", best_mrr))
        if extra.get("metric_name") == "top1_accuracy":
            best_top1 = float(extra.get("metric", best_top1))
    micro_batch_size = 32 if device.type == "mps" and cfg.env.residual_rank_window_size >= 256 else cfg.train.batch_size

    print(f"Residual rank run dir: {run_dir}", flush=True)
    print(
        f"Device: {device}; window={cfg.env.residual_rank_window_size}; "
        f"sieve_bound={cfg.env.residual_rank_sieve_bound}; hard_mode={cfg.env.residual_rank_hard_mode}; "
        f"hard_solver_uncertainty={cfg.env.residual_rank_hard_solver_uncertainty}",
        flush=True,
    )

    include_prime_labels = cfg.env.residual_rank_target == "survivor_primality"
    for step in range(start_step, cfg.train.steps + 1):
        model.train()
        def solver_score_fn(rank_batch):
            was_training = model.training
            model.eval()
            outs = []
            with torch.no_grad():
                for start in range(0, rank_batch.features.shape[0], 64):
                    logits_t, _ = model(torch.from_numpy(rank_batch.features[start : start + 64]).float().to(device))
                    outs.append(logits_t.detach().cpu().numpy())
            if was_training:
                model.train()
            return np.concatenate(outs, axis=0)

        batch = build_residual_rank_batch(
            cfg.env,
            cfg.train.batch_size,
            seed=cfg.train.seed * 1_000_003 + step,
            sieve_bound=cfg.env.residual_rank_sieve_bound,
            include_prime_labels=include_prime_labels,
            solver_score_fn=solver_score_fn if cfg.env.residual_rank_hard_mode and cfg.env.residual_rank_hard_solver_uncertainty else None,
        )
        optimizer.zero_grad(set_to_none=True)
        train_metrics = train_step_for_batch(cfg, model, batch, device, micro_batch_size=micro_batch_size)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        row: Dict[str, object] = {"step": step, "mode": "residual_rank", **train_metrics}
        if step == 1 or step % max(1, cfg.train.eval_every) == 0 or step == cfg.train.steps:
            model.eval()
            eval_metrics = evaluate_rank_model(
                cfg,
                model,
                device,
                samples=min(max(16, cfg.eval.episodes), 1024),
                seed=cfg.train.seed + 17 * step,
                sieve_bound=cfg.env.residual_rank_sieve_bound,
            )
            row.update({f"eval_{k}": v for k, v in eval_metrics.items() if k != "calibration_bins"})
            mrr = float(eval_metrics["mean_reciprocal_rank"])
            top1 = float(eval_metrics["top1_accuracy"])
            if mrr > best_mrr:
                best_mrr = mrr
                save_rank_checkpoint(str(run_dir / "checkpoints" / "best_by_mrr.pt"), model, optimizer, step, {"metric": mrr, "metric_name": "mean_reciprocal_rank", "config": cfg.to_dict()})
                row["checkpoint_promoted_mrr"] = True
            if top1 > best_top1:
                best_top1 = top1
                save_rank_checkpoint(str(run_dir / "checkpoints" / "best_by_top1.pt"), model, optimizer, step, {"metric": top1, "metric_name": "top1_accuracy", "config": cfg.to_dict()})
                row["checkpoint_promoted_top1"] = True
            row["best_eval_mrr"] = best_mrr
            row["best_eval_top1"] = best_top1
            print(json.dumps({k: v for k, v in row.items() if k != "calibration_bins"}, sort_keys=True), flush=True)
        logger.log(row)

    save_rank_checkpoint(str(run_dir / "checkpoints" / f"step_{cfg.train.steps:07d}.pt"), model, optimizer, cfg.train.steps, {"config": cfg.to_dict()})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/residual_rank_train.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--sieve-bound", type=int, default=None)
    parser.add_argument("--hard", action="store_true", help="Enable adversarial hard residual-rank sampling.")
    parser.add_argument("--no-hard", action="store_true", help="Disable adversarial hard residual-rank sampling.")
    parser.add_argument("--hard-fraction", type=float, default=None)
    parser.add_argument("--hard-pool-multiplier", type=int, default=None)
    parser.add_argument("--hard-solver-uncertainty", action="store_true", help="Let the hard sampler target current-model uncertainty.")
    parser.add_argument("--no-hard-solver-uncertainty", action="store_true", help="Disable current-model scoring inside the hard sampler.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.hard:
        cfg.env.residual_rank_hard_mode = True
    if args.no_hard:
        cfg.env.residual_rank_hard_mode = False
    if args.hard_fraction is not None:
        cfg.env.residual_rank_hard_fraction = float(args.hard_fraction)
    if args.hard_pool_multiplier is not None:
        cfg.env.residual_rank_hard_pool_multiplier = int(args.hard_pool_multiplier)
    if args.eval_samples is not None:
        cfg.eval.episodes = int(args.eval_samples)
    if args.window_size is not None:
        cfg.env.residual_rank_window_size = int(args.window_size)
    if args.sieve_bound is not None:
        cfg.env.residual_rank_sieve_bound = int(args.sieve_bound)
    if args.hard_solver_uncertainty:
        cfg.env.residual_rank_hard_solver_uncertainty = True
    if args.no_hard_solver_uncertainty:
        cfg.env.residual_rank_hard_solver_uncertainty = False
    out = train_residual_rank(cfg, run_name=args.run_name, steps=args.steps, batch_size=args.batch_size)
    print(out)


if __name__ == "__main__":
    main()
