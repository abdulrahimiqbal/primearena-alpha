from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import RunConfig, load_config, save_config
from .curriculum import AdversarialCurriculum
from .env_factory import ArenaEnv, make_arena_env, make_probe_env
from .eval_safety import guess_index_for_action, select_eval_action
from .expert import rollout_expert
from .logging_utils import JsonlLogger, make_summary_writer, tb_add
from .mcts import run_mcts, run_mcts_batch, sample_from_policy
from .model import PolicyValueNet, choose_device, save_checkpoint, load_checkpoint
from .replay import ReplayBuffer, ReplayItem, load_replay_shards, save_replay_items


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_run_dir(cfg: RunConfig, run_name: Optional[str]) -> Path:
    root = Path(cfg.train.run_dir)
    name = run_name or cfg.train.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def policy_one_hot(action_count: int, action: int) -> np.ndarray:
    p = np.zeros(action_count, dtype=np.float32)
    if 0 <= action < action_count:
        p[action] = 1.0
    return p


def value_target_from_reward(reward: float) -> float:
    return float(np.clip(reward, -1.0, 1.0))


def expert_episode_items(env: ArenaEnv) -> List[ReplayItem]:
    if hasattr(env, "expert_rollout"):
        rows = env.expert_rollout()
    else:
        rows = rollout_expert(env)
    if not rows:
        return []
    final = value_target_from_reward(float(rows[-1]["reward"]))
    items: List[ReplayItem] = []
    for row in rows:
        items.append(
            ReplayItem(
                observation=row["observation"],
                action_mask=row["mask"],
                policy_target=policy_one_hot(env.action_count, int(row["action"])),
                value_target=final,
            )
        )
    return items


def mcts_episode_items(env: ArenaEnv, model: PolicyValueNet, cfg: RunConfig, device: torch.device, rng: np.random.Generator) -> List[ReplayItem]:
    items: List[ReplayItem] = []
    policies: List[np.ndarray] = []
    observations: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    final_reward = 0.0
    while not env.done:
        pi = run_mcts(
            env,
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
        observations.append(env.observation().copy())
        masks.append(env.action_mask_float().copy())
        policies.append(pi.copy())
        action = sample_from_policy(pi, cfg.mcts.temperature, rng)
        result = env.step(action)
        final_reward = float(result.reward)
        if result.done:
            break
    target = value_target_from_reward(final_reward)
    for obs, mask, pi in zip(observations, masks, policies):
        items.append(ReplayItem(obs, mask, pi, target))
    return items


def mcts_batched_episode_items(
    envs: List[ArenaEnv],
    model: PolicyValueNet,
    cfg: RunConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> List[ReplayItem]:
    per_env_obs: List[List[np.ndarray]] = [[] for _ in envs]
    per_env_masks: List[List[np.ndarray]] = [[] for _ in envs]
    per_env_pis: List[List[np.ndarray]] = [[] for _ in envs]
    final_rewards = [0.0 for _ in envs]

    while any(not e.done for e in envs):
        active_idx = [i for i, e in enumerate(envs) if not e.done]
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
            action = sample_from_policy(pi, cfg.mcts.temperature, rng)
            result = env.step(action)
            final_rewards[idx] = float(result.reward)

    items: List[ReplayItem] = []
    for obs_list, mask_list, pi_list, reward in zip(per_env_obs, per_env_masks, per_env_pis, final_rewards):
        target = value_target_from_reward(reward)
        for obs, mask, pi in zip(obs_list, mask_list, pi_list):
            items.append(ReplayItem(obs, mask, pi, target))
    return items


def train_batch(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: List[ReplayItem],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    arrays = ReplayBuffer.to_arrays(batch)
    obs = torch.from_numpy(arrays["observations"]).float().to(device)
    mask = torch.from_numpy(arrays["action_masks"]).float().to(device)
    pi = torch.from_numpy(arrays["policy_targets"]).float().to(device)
    z = torch.from_numpy(arrays["value_targets"]).float().to(device)

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = bool(use_amp and device.type == "cuda")
    with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
        logits, value = model(obs, mask)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(pi * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, z)
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
        loss = policy_loss + value_loss - 0.005 * entropy

    if scaler is not None and autocast_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
    }


def make_env(cfg: RunConfig, rng: np.random.Generator, curriculum: Optional[AdversarialCurriculum] = None) -> ArenaEnv:
    env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
    if curriculum is not None:
        hard_n = curriculum.sample_n()
        if hard_n is not None:
            env.reset(hard_n)
    return env


def evaluate_greedy(model: PolicyValueNet, cfg: RunConfig, device: torch.device, episodes: int = 50, seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    failure_penalty_cost = float(getattr(cfg.eval, "failure_penalty_cost", 25.0))
    certified_guesses_only = bool(
        getattr(cfg.env, "residual_certified_guesses_only", True)
        if getattr(cfg.env, "residual_mode", False)
        else getattr(cfg.eval, "eval_certified_guesses_only", True)
    )
    successes = 0
    costs = []
    baseline_costs = []
    success_aware_baseline_costs = []
    success_aware_costs = []
    failed_traces: List[Dict[str, object]] = []
    wrong = 0
    invalid_actions = 0
    premature_guess_blocked_count = 0
    certified_guess_count = 0
    uncertified_guess_attempt_count = 0
    for _ in range(episodes):
        env = make_arena_env(cfg.env, seed=int(rng.integers(0, 1_000_000_000)))
        final_info = None
        selected_actions: List[Dict[str, object]] = []
        last_decision: Dict[str, object] = {}
        final_guess_status: Optional[Dict[str, object]] = None
        while not env.done:
            obs = torch.from_numpy(env.observation()).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.action_mask_float()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs, mask)
            decision = select_eval_action(env, logits, certified_guesses_only=certified_guesses_only)
            last_decision = decision
            if decision.get("uncertified_attempt"):
                uncertified_guess_attempt_count += 1
            if decision.get("blocked"):
                premature_guess_blocked_count += 1
            action_obj = decision.get("action")
            if action_obj is None:
                env.done = True
                env.success = False
                env.trace.append("guardrail: no certified-safe eval action")
                final_info = env.info(reason="certified_eval_no_safe_action")
                selected_actions.append(
                    {
                        "action": None,
                        "action_text": "no certified-safe eval action",
                        "blocked": bool(decision.get("blocked", False)),
                        "blocked_action": decision.get("blocked_action"),
                        "blocked_action_text": decision.get("blocked_action_text"),
                        "blocked_candidate_status": decision.get("blocked_candidate_status"),
                        "certification_action": decision.get("certification_action"),
                        "certification_action_text": decision.get("certification_action_text"),
                    }
                )
                break
            action = int(action_obj)
            legal = env.legal_actions()
            if action < 0 or action >= env.action_count or not legal[action]:
                invalid_actions += 1
            guess_idx = guess_index_for_action(env, action)
            if guess_idx is not None:
                final_guess_status = env.candidate_status(guess_idx)
                if env.is_certified_next_prime_guess(guess_idx):
                    certified_guess_count += 1
            selected_actions.append(
                {
                    "action": action,
                    "action_text": env.action_to_str(action),
                    "blocked": bool(decision.get("blocked", False)),
                    "blocked_action": decision.get("blocked_action"),
                    "blocked_action_text": decision.get("blocked_action_text"),
                    "blocked_candidate_status": decision.get("blocked_candidate_status"),
                    "certification_action": decision.get("certification_action"),
                    "certification_action_text": decision.get("certification_action_text"),
                    "selected_candidate_status": final_guess_status if guess_idx is not None else None,
                }
            )
            result = env.step(action)
            final_info = result.info
            if result.done:
                break
        successes += int(env.success)
        wrong += int((final_info or {}).get("reason") == "wrong_guess")
        model_cost = float(env.total_cost)
        baseline_cost = float(env.baseline_cost)
        success_aware_baseline_cost = baseline_cost + float(cfg.env.guess_cost)
        effective_model_cost = model_cost if env.success else baseline_cost + failure_penalty_cost
        costs.append(model_cost)
        baseline_costs.append(baseline_cost)
        success_aware_baseline_costs.append(success_aware_baseline_cost)
        success_aware_costs.append(float(effective_model_cost))
        if not env.success and len(failed_traces) < 5:
            failed_traces.append(
                {
                    "n": int(env.n),
                    "true_next_prime": int(env.true_next_prime),
                    "reason": (final_info or {}).get("reason"),
                    "total_cost": model_cost,
                    "baseline_cost": baseline_cost,
                    "trace": list(env.trace[-20:]),
                    "selected_actions": selected_actions[-30:],
                    "blocked_guess_count": int(sum(1 for row in selected_actions if row.get("blocked"))),
                    "candidate_status_before_final_guess": final_guess_status or last_decision.get("blocked_candidate_status"),
                    "top5_actions_at_failure": last_decision.get("top5_actions", []),
                    "safe_top5_actions_at_failure": last_decision.get("safe_top5_actions", []),
                }
            )

    baseline_arr = np.array(baseline_costs)
    success_aware_baseline_arr = np.array(success_aware_baseline_costs)
    cost_arr = np.array(costs)
    success_aware_cost_arr = np.array(success_aware_costs)
    out: Dict[str, object] = {
        "eval_success_rate": successes / max(episodes, 1),
        "eval_wrong_guess_rate": wrong / max(episodes, 1),
        "eval_wrong_guess_count": int(wrong),
        "eval_premature_guess_blocked_count": int(premature_guess_blocked_count),
        "eval_certified_guess_count": int(certified_guess_count),
        "eval_uncertified_guess_attempt_count": int(uncertified_guess_attempt_count),
        "eval_certified_guesses_only": certified_guesses_only,
        "eval_invalid_action_count": int(invalid_actions),
        "eval_invalid_action_rate": invalid_actions / max(episodes, 1),
        "eval_avg_cost": float(np.mean(costs)),
        "eval_avg_baseline_cost": float(np.mean(baseline_costs)),
        "eval_model_vs_baseline_delta": float(np.mean(baseline_arr - cost_arr)),
        "eval_failure_penalty_cost": failure_penalty_cost,
        "eval_success_aware_baseline_cost": float(np.mean(success_aware_baseline_costs)),
        "eval_success_aware_avg_cost": float(np.mean(success_aware_costs)),
        "eval_success_aware_model_vs_baseline_delta": float(np.mean(success_aware_baseline_arr - success_aware_cost_arr)),
        "eval_failed_traces": failed_traces,
    }
    if bool(getattr(cfg.env, "residual_mode", False)):
        out.update(
            {
                "residual_eval_success_rate": out["eval_success_rate"],
                "residual_wrong_guess_count": out["eval_wrong_guess_count"],
                "residual_success_aware_delta_vs_nearest": out["eval_success_aware_model_vs_baseline_delta"],
            }
        )
    return out


def _prefill_replay_from_shards(replay: ReplayBuffer, shard_patterns: List[str], limit: Optional[int] = None) -> int:
    if not shard_patterns:
        return 0
    paths: List[str] = []
    for pat in shard_patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        return 0
    items = load_replay_shards(paths, limit=limit)
    replay.extend(items)
    return len(items)


def train(cfg: RunConfig, run_name: Optional[str] = None) -> Path:
    set_seeds(cfg.train.seed)
    if cfg.train.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.train.torch_num_threads))
    device = choose_device(cfg.train.device)
    probe_env = make_probe_env(cfg)
    model = PolicyValueNet(probe_env.observation_dim, probe_env.action_count, cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    scheduler = None
    if cfg.train.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.train.steps),
            eta_min=cfg.train.lr * cfg.train.min_lr_ratio,
        )
    if cfg.train.resume_checkpoint:
        load_checkpoint(cfg.train.resume_checkpoint, model, optimizer, map_location=device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.use_amp and device.type == "cuda"))
    replay = ReplayBuffer(cfg.train.replay_capacity, seed=cfg.train.seed)
    prefilled = _prefill_replay_from_shards(replay, cfg.train.prefill_replay_shards, limit=cfg.train.replay_capacity)
    run_dir = make_run_dir(cfg, run_name)
    save_config(cfg, run_dir / "config.json")
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    writer = make_summary_writer(run_dir / "tensorboard")
    rng = np.random.default_rng(cfg.train.seed)
    curriculum = AdversarialCurriculum(cfg, rng)
    best_eval_delta = -1e18
    best_eval_success_aware_delta = -1e18
    best_promoted_eval_success_aware_delta = -1e18
    best_eval_success = -1e18
    best_promoted_eval_success = -1e18

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Model architecture: {cfg.model.architecture}")
    print(f"Observation dim: {probe_env.observation_dim}; actions: {probe_env.action_count}")
    print(f"AMP enabled: {bool(cfg.train.use_amp and device.type == 'cuda')}")
    print(f"Torch threads: {torch.get_num_threads()}")
    if cfg.train.resume_checkpoint:
        print(f"Resumed from checkpoint: {cfg.train.resume_checkpoint}")
    if prefilled:
        print(f"Prefilled replay with {prefilled} items from shards")

    for step in range(1, cfg.train.steps + 1):
        curriculum.maybe_mine(step)
        episode_rewards = []
        episode_success = []
        episode_costs = []
        episode_baseline_costs = []
        episode_lengths = []
        last_trace = []
        generated_items: List[ReplayItem] = []

        if cfg.train.mode == "mcts" and cfg.mcts.batched_inference and cfg.train.episodes_per_step > 1:
            envs = [make_env(cfg, rng, curriculum) for _ in range(cfg.train.episodes_per_step)]
            generated_items = mcts_batched_episode_items(envs, model, cfg, device, rng)
            replay.extend(generated_items)
            for env in envs:
                episode_rewards.append(float(value_target_from_reward(1.0 if env.success else -1.0)))
                episode_success.append(float(env.success))
                episode_costs.append(float(env.total_cost))
                episode_baseline_costs.append(float(env.baseline_cost))
                episode_lengths.append(float(env.steps))
                last_trace = list(env.trace[-20:])
        else:
            for _ in range(cfg.train.episodes_per_step):
                env = make_env(cfg, rng, curriculum)
                if cfg.train.mode == "mcts":
                    items = mcts_episode_items(env, model, cfg, device, rng)
                else:
                    items = expert_episode_items(env)
                generated_items.extend(items)
                replay.extend(items)
                episode_rewards.append(float(items[-1].value_target if items else 0.0))
                episode_success.append(float(env.success))
                episode_costs.append(float(env.total_cost))
                episode_baseline_costs.append(float(env.baseline_cost))
                episode_lengths.append(float(env.steps))
                last_trace = list(env.trace[-20:])

        train_stats = {}
        if len(replay) > 0:
            batch = replay.sample(cfg.train.batch_size)
            train_stats = train_batch(model, optimizer, batch, device, scaler=scaler, use_amp=cfg.train.use_amp)
            if scheduler is not None:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        row: Dict[str, object] = {
            "step": step,
            "mode": cfg.train.mode,
            "model_architecture": cfg.model.architecture,
            "episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "success_rate": float(np.mean(episode_success)) if episode_success else 0.0,
            "avg_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
            "avg_baseline_cost": float(np.mean(episode_baseline_costs)) if episode_baseline_costs else 0.0,
            "model_vs_baseline_delta": float(np.mean(np.array(episode_baseline_costs) - np.array(episode_costs))) if episode_costs else 0.0,
            "avg_episode_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "replay_size": len(replay),
            "generated_items": len(generated_items),
            "prefilled_replay_items": prefilled,
            "curriculum_hard_cases": len(curriculum.hard_cases),
            "lr": current_lr,
            "amp_enabled": bool(cfg.train.use_amp and device.type == "cuda"),
            "batched_mcts_inference": bool(cfg.train.mode == "mcts" and cfg.mcts.batched_inference),
            "last_trace": last_trace,
            **train_stats,
        }

        if step % cfg.train.eval_every == 0 or step == 1:
            row.update(evaluate_greedy(model, cfg, device, episodes=min(50, cfg.eval.episodes), seed=cfg.train.seed + step))
            promoted_delta = False
            promoted_success = False
            eval_success_rate = float(row["eval_success_rate"])
            eval_delta = float(row["eval_model_vs_baseline_delta"])
            eval_success_aware_delta = float(row["eval_success_aware_model_vs_baseline_delta"])
            eval_wrong_guess_rate = float(row["eval_wrong_guess_rate"])
            min_promotion_success_rate = float(getattr(cfg.eval, "min_promotion_success_rate", 0.80))
            min_success_aware_delta = float(getattr(cfg.eval, "min_success_aware_delta", 0.0))
            max_promotion_wrong_guess_rate = float(getattr(cfg.eval, "max_promotion_wrong_guess_rate", 0.20))
            promotion_success_gate = eval_success_rate >= min_promotion_success_rate
            promotion_delta_gate = eval_success_aware_delta >= min_success_aware_delta
            promotion_wrong_guess_gate = eval_wrong_guess_rate <= max_promotion_wrong_guess_rate
            eval_success_checkpoint_eligible = bool(promotion_success_gate and promotion_wrong_guess_gate)
            eval_checkpoint_eligible = bool(eval_success_checkpoint_eligible and promotion_delta_gate)

            if eval_delta > best_eval_delta:
                best_eval_delta = eval_delta
            if eval_success_aware_delta > best_eval_success_aware_delta:
                best_eval_success_aware_delta = eval_success_aware_delta
            if eval_success_rate > best_eval_success:
                best_eval_success = eval_success_rate
            if eval_checkpoint_eligible and eval_success_aware_delta > best_promoted_eval_success_aware_delta:
                best_promoted_eval_success_aware_delta = eval_success_aware_delta
                save_checkpoint(
                    str(run_dir / "checkpoints" / "best_by_delta.pt"),
                    model,
                    optimizer,
                    step,
                    {
                        "config": cfg.to_dict(),
                        "metric": best_promoted_eval_success_aware_delta,
                        "metric_name": "eval_success_aware_model_vs_baseline_delta",
                        "eval_success_rate": eval_success_rate,
                        "eval_model_vs_baseline_delta": eval_delta,
                    },
                )
                promoted_delta = True
            if eval_success_checkpoint_eligible and eval_success_rate > best_promoted_eval_success:
                best_promoted_eval_success = eval_success_rate
                save_checkpoint(
                    str(run_dir / "checkpoints" / "best_by_success.pt"),
                    model,
                    optimizer,
                    step,
                    {
                        "config": cfg.to_dict(),
                        "metric": best_promoted_eval_success,
                        "metric_name": "eval_success_rate",
                        "eval_success_aware_model_vs_baseline_delta": eval_success_aware_delta,
                    },
                )
                promoted_success = True
            row["best_eval_model_vs_baseline_delta"] = best_eval_delta
            row["best_eval_success_aware_model_vs_baseline_delta"] = best_eval_success_aware_delta
            row["best_eval_success_rate"] = best_eval_success
            row["best_promoted_eval_success_aware_model_vs_baseline_delta"] = best_promoted_eval_success_aware_delta
            row["best_promoted_eval_success_rate"] = best_promoted_eval_success
            row["promotion_min_success_rate"] = min_promotion_success_rate
            row["promotion_min_success_aware_delta"] = min_success_aware_delta
            row["promotion_max_wrong_guess_rate"] = max_promotion_wrong_guess_rate
            row["checkpoint_promotion_success_gate"] = promotion_success_gate
            row["checkpoint_promotion_delta_gate"] = promotion_delta_gate
            row["checkpoint_promotion_wrong_guess_gate"] = promotion_wrong_guess_gate
            row["eval_success_checkpoint_eligible"] = eval_success_checkpoint_eligible
            row["eval_checkpoint_eligible"] = eval_checkpoint_eligible
            row["checkpoint_promoted_delta"] = promoted_delta
            row["checkpoint_promoted_success"] = promoted_success

        if cfg.train.save_replay_shards and generated_items and (step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps):
            shard = run_dir / "replay_shards" / f"step_{step:07d}.npz"
            save_replay_items(shard, generated_items, metadata={"step": step, "run_dir": str(run_dir)})
            row["saved_replay_shard"] = str(shard)

        logger.log(row)
        for k, v in row.items():
            tb_add(writer, k, v, step)

        if step % max(1, cfg.train.eval_every) == 0 or step == 1:
            print(json.dumps({k: row[k] for k in row if k not in {"last_trace"}}, sort_keys=True))

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            ckpt = run_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint(str(ckpt), model, optimizer, step, {"config": cfg.to_dict()})

    if writer is not None:
        writer.flush()
        writer.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.json")
    parser.add_argument("--mode", type=str, choices=["imitation", "mcts"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode is not None:
        cfg.train.mode = args.mode
    if args.steps is not None:
        cfg.train.steps = args.steps
    train(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
