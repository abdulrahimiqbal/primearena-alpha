from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from primearena.config import load_config
from primearena.env import PrimeArena
from primearena.eval_safety import select_eval_action
from primearena.expert import rollout_expert
from primearena.oracle import is_prime, next_prime
from primearena.evaluate import evaluate
from primearena.baselines import nearest_survivor_first
from primearena.residual_env import PrimeArenaResidual
from primearena.residual_rank import build_hard_rank_batch, rank_batch_diagnostics


def _logits_for(env: PrimeArena, preferred_action: int) -> torch.Tensor:
    logits = torch.full((1, env.action_count), -100.0)
    logits[0, preferred_action] = 100.0
    return logits


def assert_certified_guess_safety(cfg) -> None:
    # Case A: an untested candidate guess is blocked even when the policy wants it.
    env = PrimeArena(cfg.env, seed=321)
    env.reset(10)
    guess0 = env.guess_action_offset
    decision = select_eval_action(env, _logits_for(env, guess0), certified_guesses_only=True)
    assert decision["blocked"], "untested guess was not blocked"
    assert decision["uncertified_attempt"], "untested guess attempt was not counted"
    assert decision["action"] != guess0, "untested guess survived certified eval mask"

    # Case B: a tested-prime candidate is still blocked if earlier candidates are unknown.
    env = PrimeArena(cfg.env, seed=322)
    env.reset(10)
    env.tested[2] = 1
    guess2 = env.guess_action_offset + 2
    decision = select_eval_action(env, _logits_for(env, guess2), certified_guesses_only=True)
    assert decision["blocked"], "prime guess with unknown prior candidates was not blocked"
    assert decision["action"] != guess2, "uncertified later prime survived certified eval mask"

    # Case C: a tested-prime candidate is allowed once every prior candidate is resolved.
    env = PrimeArena(cfg.env, seed=323)
    env.reset(10)
    env.tested[0] = -1
    env.eliminated[1] = True
    env.tested[2] = 1
    decision = select_eval_action(env, _logits_for(env, guess2), certified_guesses_only=True)
    assert not decision["blocked"], "certified guess was unexpectedly blocked"
    assert decision["action"] == guess2, "certified guess was not selected"


def assert_residual_env_smoke(cfg) -> None:
    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.residual_mode = True
    env_cfg.residual_sieve_bound = 30
    env_cfg.residual_window_size = 32
    env_cfg.max_steps = 64
    env_cfg.budget = 64.0
    env = PrimeArenaResidual(env_cfg, seed=444)
    assert env.observation().shape[0] == env.observation_dim
    assert env.legal_actions().sum() > 0
    baseline = nearest_survivor_first(env.n, env_cfg, bound=env_cfg.residual_sieve_bound)
    assert baseline.success
    guess0 = env.guess_action_offset
    decision = select_eval_action(env, _logits_for(env, guess0), certified_guesses_only=True)
    assert decision["blocked"], "residual untested commit was not blocked"
    rows = env.expert_rollout()
    assert rows, "residual expert produced no actions"
    assert env.success, f"residual expert failed: {env.info()}"


def assert_residual_rank_hard_smoke(cfg) -> None:
    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.residual_rank_mode = True
    env_cfg.residual_rank_sieve_bound = 30
    env_cfg.residual_rank_window_size = 64
    env_cfg.residual_rank_hard_mode = True
    env_cfg.residual_rank_hard_fraction = 1.0
    env_cfg.residual_rank_hard_pool_multiplier = 4
    env_cfg.residual_rank_hard_min_true_index = 1
    batch = build_hard_rank_batch(env_cfg, 8, seed=555, sieve_bound=30)
    assert batch.features.shape == (8, 64, batch.features.shape[-1])
    assert batch.survivor_values.shape == (8, 64)
    assert batch.metadata and "hard_score" in batch.metadata[0]
    diagnostics = rank_batch_diagnostics(batch)
    assert diagnostics["nearest_fail_rate"] > 0.0, "hard residual-rank batch did not include nearest-fail cases"


def main() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.json")
    assert next_prime(10) == 11
    assert next_prime(1000) == 1009
    assert is_prime(1009)

    env = PrimeArena(cfg.env, seed=123)
    obs = env.observation()
    assert obs.shape[0] == env.observation_dim
    assert env.legal_actions().sum() > 0
    assert_certified_guess_safety(cfg)
    assert_residual_env_smoke(cfg)
    assert_residual_rank_hard_smoke(cfg)

    rows = rollout_expert(env)
    assert rows, "expert produced no actions"
    assert env.success, f"expert failed: {env.info()}"

    metrics = evaluate(cfg, episodes=10, seed=123)
    assert metrics["expert_success_rate"] >= 0.99
    assert metrics["wheel_vs_odd_delta"] >= 0.0
    print("PrimeArena smoke test passed")
    print(metrics)


if __name__ == "__main__":
    main()
