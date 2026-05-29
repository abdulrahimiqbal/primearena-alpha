from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import RunConfig, load_config
from .curriculum import AdversarialCurriculum
from .env_factory import ArenaEnv, make_arena_env, make_probe_env
from .expert import rollout_expert
from .mcts import run_mcts, run_mcts_batch, sample_from_policy
from .model import PolicyValueNet, choose_device, load_checkpoint
from .replay import ReplayItem, load_replay_shards, save_replay_items
from .train import expert_episode_items, policy_one_hot, train, value_target_from_reward


def _make_model(cfg: RunConfig, device: torch.device) -> PolicyValueNet:
    probe_env = make_probe_env(cfg)
    return PolicyValueNet(probe_env.observation_dim, probe_env.action_count, cfg.model).to(device)


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _emit_worker_progress(
    progress_dir: Optional[str | Path],
    payload: Dict[str, Any],
    progress_callback: ProgressCallback = None,
) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", time.time())
    worker_id = int(payload.get("worker_id", -1))
    line = json.dumps(payload, sort_keys=True)
    print(f"[primearena-worker-progress] {line}", flush=True)

    if progress_dir is not None:
        p = Path(progress_dir)
        p.mkdir(parents=True, exist_ok=True)
        jsonl_path = p / f"worker_{worker_id:04d}.jsonl"
        latest_path = p / f"worker_{worker_id:04d}.latest.json"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if progress_callback is not None:
        progress_callback(payload)


def mcts_parallel_episode_items(
    envs: List[ArenaEnv],
    model: PolicyValueNet,
    cfg: RunConfig,
    device: torch.device,
    rng: np.random.Generator,
    deadline: Optional[float] = None,
    max_items: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
) -> Tuple[List[ReplayItem], Dict[str, int]]:
    """Generate episodes for multiple envs, batching MCTS neural inference."""
    per_env_obs: List[List[np.ndarray]] = [[] for _ in envs]
    per_env_masks: List[List[np.ndarray]] = [[] for _ in envs]
    per_env_pis: List[List[np.ndarray]] = [[] for _ in envs]
    final_rewards = [0.0 for _ in envs]
    meta = {
        "forced_timeouts": 0,
        "deadline_stops": 0,
        "item_cap_stops": 0,
        "positions": 0,
    }

    def force_timeout(idx: int, reason: str) -> None:
        env = envs[idx]
        if env.done:
            return
        env.done = True
        env.success = False
        env.trace.append(reason)
        final_rewards[idx] = float(cfg.env.timeout_reward - cfg.env.step_cost_scale * env.total_cost)
        meta["forced_timeouts"] += 1

    while any(not e.done for e in envs):
        if deadline is not None and time.time() >= deadline:
            meta["deadline_stops"] += 1
            for i, env in enumerate(envs):
                if not env.done:
                    force_timeout(i, "guardrail: worker deadline reached")
            break

        if max_items is not None and meta["positions"] >= max_items:
            meta["item_cap_stops"] += 1
            for i, env in enumerate(envs):
                if not env.done:
                    force_timeout(i, "guardrail: replay item cap reached")
            break

        if max_episode_steps is not None:
            for i, env in enumerate(envs):
                if not env.done and env.steps >= max_episode_steps:
                    force_timeout(i, "guardrail: self-play episode step cap reached")

        active_idx = [i for i, e in enumerate(envs) if not e.done]
        if not active_idx:
            break
        active_envs = [envs[i] for i in active_idx]
        policies = run_mcts_batch(
            active_envs,
            model,
            cfg.mcts.simulations,
            cfg.mcts.c_puct,
            cfg.mcts.gamma,
            device,
            rng=rng,
            add_root_noise_flag=cfg.mcts.add_root_noise,
            root_dirichlet_alpha=cfg.mcts.root_dirichlet_alpha,
            root_exploration_fraction=cfg.mcts.root_exploration_fraction,
        )
        for idx, pi in zip(active_idx, policies):
            env = envs[idx]
            per_env_obs[idx].append(env.observation().copy())
            per_env_masks[idx].append(env.action_mask_float().copy())
            per_env_pis[idx].append(pi.copy())
            meta["positions"] += 1
            action = sample_from_policy(pi, cfg.mcts.temperature, rng)
            result = env.step(action)
            final_rewards[idx] = float(result.reward)

    items: List[ReplayItem] = []
    for obs_list, mask_list, pi_list, reward in zip(per_env_obs, per_env_masks, per_env_pis, final_rewards):
        target = value_target_from_reward(reward)
        for obs, mask, pi in zip(obs_list, mask_list, pi_list):
            items.append(ReplayItem(obs, mask, pi, target))
    return items, meta


def generate_selfplay_shard(
    cfg: RunConfig,
    checkpoint: Optional[str],
    out_dir: str | Path,
    worker_id: int = 0,
    episodes: int = 32,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
    progress_dir: Optional[str | Path] = None,
    progress_every_episodes: int = 4,
    progress_callback: ProgressCallback = None,
) -> Path:
    """Generate one replay shard. This is the function Modal workers call."""
    started_at = time.time()
    seed = int(cfg.train.seed + 1009 * worker_id if seed is None else seed)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    rng = np.random.default_rng(seed)
    device = choose_device(cfg.train.device)
    model = _make_model(cfg, device)
    model.eval()
    if checkpoint:
        load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    mode = mode or cfg.distributed.worker_mode or cfg.train.mode

    # Distributed self-play guardrails. These make each Modal worker produce
    # bounded, learner-friendly shards instead of one giant straggler shard.
    max_worker_seconds = int(getattr(cfg.distributed, "max_worker_seconds", 0) or 0)
    deadline = started_at + max_worker_seconds if max_worker_seconds > 0 else None
    max_replay_items = int(getattr(cfg.distributed, "max_replay_items_per_shard", 0) or 0)
    max_selfplay_episode_steps = int(getattr(cfg.distributed, "max_selfplay_episode_steps", 0) or 0)
    if max_selfplay_episode_steps > 0:
        cfg.env.max_steps = min(int(cfg.env.max_steps), max_selfplay_episode_steps)

    curriculum = AdversarialCurriculum(cfg, rng)

    all_items: List[ReplayItem] = []
    stats = {
        "episodes": 0,
        "attempted_episodes": 0,
        "successes": 0,
        "avg_cost": 0.0,
        "avg_baseline_cost": 0.0,
        "forced_timeouts": 0,
        "deadline_stops": 0,
        "item_cap_stops": 0,
    }
    costs: List[float] = []
    baseline_costs: List[float] = []

    def progress(event: str, **extra: Any) -> None:
        elapsed = max(time.time() - started_at, 1e-9)
        avg_cost = float(np.mean(costs)) if costs else 0.0
        avg_baseline_cost = float(np.mean(baseline_costs)) if baseline_costs else 0.0
        payload: Dict[str, Any] = {
            "event": event,
            "worker_id": worker_id,
            "pid": os.getpid(),
            "mode": mode,
            "device": str(device),
            "checkpoint": checkpoint,
            "seed": seed,
            "episodes_done": int(stats["episodes"]),
            "attempted_episodes": int(stats.get("attempted_episodes", stats["episodes"])),
            "episodes_total": int(episodes),
            "successes": int(stats["successes"]),
            "success_rate": float(stats["successes"] / max(int(stats["episodes"]), 1)),
            "avg_cost": avg_cost,
            "avg_baseline_cost": avg_baseline_cost,
            "forced_timeouts": int(stats.get("forced_timeouts", 0)),
            "deadline_stops": int(stats.get("deadline_stops", 0)),
            "item_cap_stops": int(stats.get("item_cap_stops", 0)),
            "items": int(len(all_items)),
            "items_per_second": float(len(all_items) / elapsed),
            "elapsed_sec": float(elapsed),
        }
        payload.update(extra)
        _emit_worker_progress(progress_dir, payload, progress_callback)

    progress(
        "started",
        batched_mcts=bool(mode == "mcts" and cfg.mcts.batched_inference),
        max_worker_seconds=max_worker_seconds,
        max_replay_items_per_shard=max_replay_items,
        max_selfplay_episode_steps=max_selfplay_episode_steps,
    )

    try:
        if mode == "mcts" and cfg.mcts.batched_inference:
            batch_size = max(
                1,
                int(getattr(cfg.distributed, "selfplay_batch_size", 0) or cfg.train.episodes_per_step),
            )
            remaining = episodes
            batch_index = 0
            while remaining > 0:
                if deadline is not None and time.time() >= deadline:
                    stats["deadline_stops"] += 1
                    progress("worker_deadline_reached")
                    break
                if max_replay_items > 0 and len(all_items) >= max_replay_items:
                    stats["item_cap_stops"] += 1
                    progress("worker_item_cap_reached")
                    break

                batch_index += 1
                n_batch = min(batch_size, remaining)
                envs: List[ArenaEnv] = []
                for _ in range(n_batch):
                    env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
                    hard_n = curriculum.sample_n()
                    if hard_n is not None:
                        env.reset(hard_n)
                    envs.append(env)

                remaining_item_budget = None
                if max_replay_items > 0:
                    remaining_item_budget = max(1, max_replay_items - len(all_items))

                batch_items, batch_meta = mcts_parallel_episode_items(
                    envs,
                    model,
                    cfg,
                    device,
                    rng,
                    deadline=deadline,
                    max_items=remaining_item_budget,
                    max_episode_steps=max_selfplay_episode_steps if max_selfplay_episode_steps > 0 else None,
                )
                all_items.extend(batch_items)
                stats["forced_timeouts"] += int(batch_meta.get("forced_timeouts", 0))
                stats["deadline_stops"] += int(batch_meta.get("deadline_stops", 0))
                stats["item_cap_stops"] += int(batch_meta.get("item_cap_stops", 0))

                for env in envs:
                    stats["attempted_episodes"] += 1
                    stats["episodes"] += 1
                    stats["successes"] += int(env.success)
                    costs.append(float(env.total_cost))
                    baseline_costs.append(float(env.baseline_cost))
                remaining -= n_batch
                progress(
                    "batch_complete",
                    batch_index=batch_index,
                    batch_size=n_batch,
                    batch_positions=int(batch_meta.get("positions", 0)),
                    batch_forced_timeouts=int(batch_meta.get("forced_timeouts", 0)),
                    batch_deadline_stops=int(batch_meta.get("deadline_stops", 0)),
                    batch_item_cap_stops=int(batch_meta.get("item_cap_stops", 0)),
                )

                if batch_meta.get("deadline_stops", 0) or batch_meta.get("item_cap_stops", 0):
                    break
        else:
            from .train import mcts_episode_items

            emit_every = max(1, int(progress_every_episodes))
            for episode_idx in range(episodes):
                if deadline is not None and time.time() >= deadline:
                    stats["deadline_stops"] += 1
                    progress("worker_deadline_reached", episode_index=episode_idx)
                    break
                if max_replay_items > 0 and len(all_items) >= max_replay_items:
                    stats["item_cap_stops"] += 1
                    progress("worker_item_cap_reached", episode_index=episode_idx)
                    break

                env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
                hard_n = curriculum.sample_n()
                if hard_n is not None:
                    env.reset(hard_n)
                if mode == "mcts":
                    items = mcts_episode_items(env, model, cfg, device, rng)
                else:
                    items = expert_episode_items(env)
                all_items.extend(items)
                stats["attempted_episodes"] += 1
                stats["episodes"] += 1
                stats["successes"] += int(env.success)
                costs.append(float(env.total_cost))
                baseline_costs.append(float(env.baseline_cost))
                if (episode_idx + 1) % emit_every == 0 or (episode_idx + 1) == episodes:
                    progress("episode_complete", episode_index=episode_idx + 1)
    except Exception as exc:
        progress("failed", error_type=type(exc).__name__, error=str(exc))
        raise

    stats["success_rate"] = stats["successes"] / max(stats["episodes"], 1)
    stats["avg_cost"] = float(np.mean(costs)) if costs else 0.0
    stats["avg_baseline_cost"] = float(np.mean(baseline_costs)) if baseline_costs else 0.0
    stats["items"] = len(all_items)
    stats["worker_id"] = worker_id
    stats["mode"] = mode
    stats["checkpoint"] = checkpoint
    stats["created_at"] = time.time()
    stats["elapsed_sec"] = time.time() - started_at

    out = Path(out_dir) / f"selfplay_worker_{worker_id:04d}_{int(time.time())}.npz"
    save_replay_items(out, all_items, metadata=stats)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    progress("shard_saved", shard_path=str(out), sidecar_path=str(sidecar))
    return out


def train_from_replay_shards(
    config_path: str,
    shard_glob: str,
    run_name: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    max_shards: Optional[int] = None,
    run_dir: Optional[str] = None,
) -> Path:
    cfg = load_config(config_path)
    if run_dir is not None:
        cfg.train.run_dir = run_dir
    paths = sorted(glob.glob(shard_glob))
    if max_shards is not None:
        paths = paths[-max_shards:]
    if not paths:
        raise FileNotFoundError(f"No replay shards matched: {shard_glob}")
    cfg.train.prefill_replay_shards = paths
    if resume_checkpoint:
        cfg.train.resume_checkpoint = resume_checkpoint
    return train(cfg, run_name=run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-shard")
    p_gen.add_argument("--config", default="configs/smoke.json")
    p_gen.add_argument("--checkpoint", default=None)
    p_gen.add_argument("--out-dir", default="runs/replay_shards")
    p_gen.add_argument("--worker-id", type=int, default=0)
    p_gen.add_argument("--episodes", type=int, default=8)
    p_gen.add_argument("--mode", choices=["mcts", "imitation"], default=None)
    p_gen.add_argument("--progress-dir", default=None)
    p_gen.add_argument("--progress-every-episodes", type=int, default=4)

    p_train = sub.add_parser("train-from-shards")
    p_train.add_argument("--config", default="configs/smoke.json")
    p_train.add_argument("--shard-glob", default="runs/replay_shards/*.npz")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--resume-checkpoint", default=None)
    p_train.add_argument("--max-shards", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "generate-shard":
        cfg = load_config(args.config)
        out = generate_selfplay_shard(
            cfg,
            args.checkpoint,
            args.out_dir,
            args.worker_id,
            args.episodes,
            mode=args.mode,
            progress_dir=args.progress_dir,
            progress_every_episodes=args.progress_every_episodes,
        )
        print(out)
    elif args.cmd == "train-from-shards":
        out = train_from_replay_shards(args.config, args.shard_glob, args.run_name, args.resume_checkpoint, args.max_shards)
        print(out)


if __name__ == "__main__":
    main()
