"""FROZEN — legacy RL stack, do not extend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .baselines import all_baselines
from .config import EnvConfig, RunConfig, load_config
from .env import PrimeArena
from .expert import rollout_expert


def evaluate_baselines(cfg: RunConfig, episodes: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        env = PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        results = all_baselines(env.n, cfg.env)
        row = {f"{name}_cost": res.cost for name, res in results.items()}
        row.update({f"{name}_tests": res.tests for name, res in results.items()})
        rows.append(row)
    keys = sorted(rows[0].keys()) if rows else []
    out = {f"avg_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    out["wheel_vs_odd_delta"] = out.get("avg_odd_scan_cost", 0.0) - out.get("avg_wheel_scan_cost", 0.0)
    out["sieve_vs_odd_delta"] = out.get("avg_odd_scan_cost", 0.0) - out.get("avg_segmented_sieve_cost", 0.0)
    return out


def evaluate_expert(cfg: RunConfig, episodes: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    success = []
    costs = []
    baseline = []
    lengths = []
    for _ in range(episodes):
        env = PrimeArena(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        rollout_expert(env)
        success.append(float(env.success))
        costs.append(float(env.total_cost))
        baseline.append(float(env.baseline_cost))
        lengths.append(float(env.steps))
    return {
        "expert_success_rate": float(np.mean(success)),
        "expert_avg_cost": float(np.mean(costs)),
        "expert_avg_baseline_cost": float(np.mean(baseline)),
        "expert_vs_baseline_delta": float(np.mean(np.array(baseline) - np.array(costs))),
        "expert_avg_len": float(np.mean(lengths)),
    }


def evaluate(cfg: RunConfig, episodes: int, seed: int = 0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(evaluate_baselines(cfg, episodes, seed))
    out.update(evaluate_expert(cfg, episodes, seed + 17))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.json")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = evaluate(cfg, args.episodes, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
