from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .config import RunConfig, load_config
from .env import PrimeArena
from .eval_safety import select_eval_action
from .model import PolicyValueNet, choose_device, load_checkpoint


@dataclass
class ArenaResult:
    checkpoint: str
    episodes: int
    success_rate: float
    wrong_guess_rate: float
    avg_cost: float
    avg_baseline_cost: float
    avg_score: float
    model_vs_baseline_delta: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def score_episode(success: bool, cost: float, baseline_cost: float, wrong_guess: bool) -> float:
    if not success:
        return -1.0 - (1.0 if wrong_guess else 0.25)
    efficiency = (baseline_cost - cost) / max(baseline_cost, 1e-6)
    return 1.0 + float(np.clip(efficiency, -1.0, 1.0))


def _load_model(cfg: RunConfig, checkpoint: str, device: torch.device) -> PolicyValueNet:
    probe = PrimeArena(cfg.env, seed=cfg.train.seed)
    model = PolicyValueNet(probe.observation_dim, probe.action_count, cfg.model).to(device)
    load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    model.eval()
    return model


def evaluate_checkpoint(
    cfg: RunConfig,
    checkpoint: str,
    seeds: Iterable[int],
    device: Optional[torch.device] = None,
) -> ArenaResult:
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    seeds = list(seeds)
    device = device or choose_device(cfg.train.device)
    model = _load_model(cfg, checkpoint, device)
    scores: List[float] = []
    successes: List[float] = []
    wrongs: List[float] = []
    costs: List[float] = []
    baselines: List[float] = []

    for seed in seeds:
        env = PrimeArena(cfg.env, seed=int(seed))
        final_info = None
        while not env.done:
            obs = torch.from_numpy(env.observation()).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask_float()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            decision = select_eval_action(
                env,
                logits,
                certified_guesses_only=bool(getattr(cfg.eval, "eval_certified_guesses_only", True)),
            )
            action_obj = decision.get("action")
            if action_obj is None:
                env.done = True
                env.success = False
                env.trace.append("guardrail: no certified-safe arena action")
                final_info = env.info(reason="certified_arena_no_safe_action")
                break
            action = int(action_obj)
            result = env.step(action)
            final_info = result.info
        wrong = bool((final_info or {}).get("reason") == "wrong_guess")
        scores.append(score_episode(env.success, env.total_cost, env.baseline_cost, wrong))
        successes.append(float(env.success))
        wrongs.append(float(wrong))
        costs.append(float(env.total_cost))
        baselines.append(float(env.baseline_cost))

    return ArenaResult(
        checkpoint=checkpoint,
        episodes=len(seeds),
        success_rate=float(np.mean(successes)) if successes else 0.0,
        wrong_guess_rate=float(np.mean(wrongs)) if wrongs else 0.0,
        avg_cost=float(np.mean(costs)) if costs else 0.0,
        avg_baseline_cost=float(np.mean(baselines)) if baselines else 0.0,
        avg_score=float(np.mean(scores)) if scores else 0.0,
        model_vs_baseline_delta=float(np.mean(np.asarray(baselines) - np.asarray(costs))) if costs else 0.0,
    )


def run_league(
    cfg: RunConfig,
    checkpoints: List[str],
    episodes: int = 100,
    seed: int = 0,
    out_dir: str | Path = "runs/league",
) -> Dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seeds = [int(x) for x in rng.integers(0, 1_000_000_000, size=episodes)]
    device = choose_device(cfg.train.device)
    results = [evaluate_checkpoint(cfg, ckpt, seeds, device) for ckpt in checkpoints]
    results.sort(key=lambda r: (r.avg_score, r.success_rate, r.model_vs_baseline_delta), reverse=True)

    (out / "league.json").write_text(json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True) + "\n")
    with (out / "leaderboard.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()) if results else ["checkpoint"])
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    return {"out_dir": str(out), "leader": results[0].to_dict() if results else None, "results": [r.to_dict() for r in results]}


def compare_and_promote(
    cfg: RunConfig,
    champion: str,
    challenger: str,
    episodes: int = 100,
    seed: int = 0,
    promote_to: Optional[str] = None,
) -> Dict[str, object]:
    league = run_league(cfg, [champion, challenger], episodes=episodes, seed=seed, out_dir=cfg.distributed.league_dir)
    results = league["results"]
    by_path = {r["checkpoint"]: r for r in results}
    champ = by_path[champion]
    chal = by_path[challenger]
    success_delta = float(chal["success_rate"] - champ["success_rate"])
    score_delta = float(chal["avg_score"] - champ["avg_score"])
    promoted = bool(
        success_delta >= cfg.distributed.promotion_min_success_delta
        and score_delta >= cfg.distributed.promotion_min_score_delta
    )
    if promoted and promote_to:
        Path(promote_to).parent.mkdir(parents=True, exist_ok=True)
        Path(promote_to).write_bytes(Path(challenger).read_bytes())
    return {"promoted": promoted, "success_delta": success_delta, "score_delta": score_delta, "league": league}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint_glob:
        checkpoints.extend(sorted(glob.glob(args.checkpoint_glob)))
    if not checkpoints:
        raise SystemExit("No checkpoints supplied. Use --checkpoints or --checkpoint-glob.")
    result = run_league(cfg, checkpoints, episodes=args.episodes, seed=args.seed, out_dir=args.out_dir or cfg.distributed.league_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
