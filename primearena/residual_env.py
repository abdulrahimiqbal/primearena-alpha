"""FROZEN — legacy RL stack, do not extend."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .baselines import nearest_survivor_first, residual_sieve_primes
from .config import EnvConfig
from .oracle import is_prime, next_prime


@dataclass
class ResidualStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class PrimeArenaResidual:
    """Residual survivor-search environment.

    Fixed action layout:
      [0, W)       primality-test survivor i
      [W, 2W)      commit survivor i as the next prime
      [2W]         expand to the next survivor window
    """

    def __init__(self, cfg: EnvConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.reset()

    @property
    def window_size(self) -> int:
        return int(self.cfg.residual_window_size)

    @property
    def num_filter_actions(self) -> int:
        return 0

    @property
    def guess_action_offset(self) -> int:
        return self.window_size

    @property
    def expand_action(self) -> int:
        return 2 * self.window_size

    @property
    def action_count(self) -> int:
        return 2 * self.window_size + 1

    @property
    def feature_dim_per_candidate(self) -> int:
        return 6

    @property
    def global_feature_dim(self) -> int:
        return 7

    @property
    def observation_dim(self) -> int:
        return self.window_size * self.feature_dim_per_candidate + self.global_feature_dim

    def clone(self) -> "PrimeArenaResidual":
        return copy.deepcopy(self)

    def reset(self, n: Optional[int] = None) -> np.ndarray:
        self.sieve_primes = residual_sieve_primes(self.cfg, self.cfg.residual_sieve_bound)
        if n is None:
            n = self._sample_n()
        self.n = int(n)
        self.true_next_prime = next_prime(self.n)
        self.window_start = self.n + 1
        self.steps = 0
        self.total_cost = 0.0
        self.done = False
        self.success = False
        self.trace: List[str] = []
        self.tested = np.zeros(self.window_size, dtype=np.int8)
        self.eliminated = np.zeros(self.window_size, dtype=np.bool_)
        self.survivors = self._make_survivor_window(self.window_start)
        self.window_end = int(self.survivors[-1]) if len(self.survivors) else self.window_start - 1
        self.baseline_cost = nearest_survivor_first(self.n, self.cfg, bound=self.cfg.residual_sieve_bound).cost
        self.total_tests = 0
        return self.observation()

    def _sample_n(self) -> int:
        for _ in range(self.cfg.max_resample_attempts):
            n = self.rng.randint(self.cfg.n_min, self.cfg.n_max)
            if not self.cfg.ensure_prime_in_window:
                return n
            p = next_prime(n)
            probe = self._survivors_from(n + 1, self.window_size)
            if p in probe:
                return n
        return self.rng.randint(self.cfg.n_min, self.cfg.n_max)

    def _is_survivor(self, x: int) -> bool:
        if x < 2:
            return False
        for q in self.sieve_primes:
            if x != q and x % q == 0:
                return False
        return True

    def _survivors_from(self, start: int, count: int) -> List[int]:
        out: List[int] = []
        x = int(start)
        while len(out) < count:
            if self._is_survivor(x):
                out.append(x)
            x += 1
        return out

    def _make_survivor_window(self, start: int) -> np.ndarray:
        return np.asarray(self._survivors_from(start, self.window_size), dtype=np.int64)

    def candidates(self) -> np.ndarray:
        return self.survivors.copy()

    def candidate_index(self, value: int) -> Optional[int]:
        hits = np.where(self.survivors == int(value))[0]
        return int(hits[0]) if len(hits) else None

    def is_candidate_tested_prime(self, index: int) -> bool:
        return 0 <= index < self.window_size and bool(self.tested[index] == 1)

    def are_all_prior_candidates_eliminated_or_composite(self, index: int) -> bool:
        if index < 0 or index >= self.window_size:
            return False
        for i in range(index):
            if bool(self.eliminated[i]) or self.tested[i] == -1:
                continue
            return False
        return True

    def is_certified_next_prime_guess(self, index: int) -> bool:
        return self.is_candidate_tested_prime(index) and self.are_all_prior_candidates_eliminated_or_composite(index)

    def certified_guess_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.bool_)
        legal = self.legal_actions()
        for i in range(self.window_size):
            action = self.guess_action_offset + i
            if legal[action] and self.is_certified_next_prime_guess(i):
                mask[action] = True
        return mask

    def candidate_status(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= self.window_size:
            return {"index": int(index), "valid": False}
        tested = int(self.tested[index])
        return {
            "index": int(index),
            "candidate": int(self.survivors[index]),
            "eliminated": bool(self.eliminated[index]),
            "tested": tested,
            "tested_prime": bool(tested == 1),
            "tested_composite": bool(tested == -1),
            "certified_next_prime_guess": bool(self.is_certified_next_prime_guess(index)),
        }

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.bool_)
        if self.done:
            return mask
        for i in range(self.window_size):
            if not self.eliminated[i]:
                if self.tested[i] == 0 and self.total_cost + self.cfg.test_cost <= self.cfg.budget:
                    mask[i] = True
                if self.total_cost + self.cfg.guess_cost <= self.cfg.budget:
                    mask[self.guess_action_offset + i] = True
        if self.total_cost + self.cfg.expand_cost <= self.cfg.budget:
            mask[self.expand_action] = True
        return mask

    def action_to_str(self, action: int) -> str:
        if 0 <= action < self.window_size:
            return f"test survivor {int(self.survivors[action])}"
        if self.guess_action_offset <= action < self.expand_action:
            return f"commit survivor {int(self.survivors[action - self.guess_action_offset])}"
        if action == self.expand_action:
            return "expand survivor window"
        return f"invalid {action}"

    def _finish(self, reward: float, success: bool, reason: str) -> ResidualStepResult:
        self.done = True
        self.success = success
        return ResidualStepResult(self.observation(), reward, True, self.info(reason=reason))

    def step(self, action: int) -> ResidualStepResult:
        if self.done:
            return ResidualStepResult(self.observation(), 0.0, True, self.info(reason="already_done"))
        legal = self.legal_actions()
        if action < 0 or action >= self.action_count or not legal[action]:
            self.steps += 1
            self.total_cost += self.cfg.invalid_action_penalty
            self.trace.append(f"invalid {action}")
            if self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget:
                return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "invalid_timeout")
            return ResidualStepResult(self.observation(), -self.cfg.invalid_action_penalty, False, self.info(reason="invalid"))

        self.steps += 1
        if action < self.window_size:
            c = int(self.survivors[action])
            prime = is_prime(c)
            self.tested[action] = 1 if prime else -1
            self.total_tests += 1
            self.total_cost += self.cfg.test_cost
            self.trace.append(f"test survivor {c} -> {'prime' if prime else 'composite'}")
            if self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget:
                return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "budget_or_steps")
            return ResidualStepResult(
                self.observation(),
                -self.cfg.step_cost_scale * self.cfg.test_cost,
                False,
                self.info(reason="test", tested_prime=prime),
            )

        if action < self.expand_action:
            idx = action - self.guess_action_offset
            c = int(self.survivors[idx])
            self.total_cost += self.cfg.guess_cost
            correct = c == self.true_next_prime
            self.trace.append(f"commit survivor {c} -> {'correct' if correct else 'wrong'}")
            if correct:
                delta = (self.baseline_cost - self.total_cost) / max(self.baseline_cost, 1e-6)
                reward = self.cfg.correct_reward + max(-1.0, min(1.0, delta))
                return self._finish(reward, True, "correct_guess")
            return self._finish(self.cfg.wrong_guess_reward - self.cfg.step_cost_scale * self.total_cost, False, "wrong_guess")

        self.window_start = self.window_end + 1
        self.survivors = self._make_survivor_window(self.window_start)
        self.window_end = int(self.survivors[-1])
        self.tested = np.zeros(self.window_size, dtype=np.int8)
        self.eliminated = np.zeros(self.window_size, dtype=np.bool_)
        self.total_cost += self.cfg.expand_cost
        self.trace.append(f"expand survivor window -> {int(self.survivors[0])}-{int(self.survivors[-1])}")
        if self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget:
            return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "budget_or_steps")
        return ResidualStepResult(
            self.observation(),
            -self.cfg.step_cost_scale * self.cfg.expand_cost,
            False,
            self.info(reason="expand"),
        )

    def observation(self) -> np.ndarray:
        feats: List[float] = []
        log_n = np.log10(max(self.n, 10))
        max_dist = max(int(self.survivors[-1]) - self.n, 1)
        gaps = np.diff(np.concatenate([[self.n], self.survivors])).astype(np.float64)
        density = self.window_size / max(int(self.survivors[-1]) - self.n, 1)
        for i, c in enumerate(self.survivors):
            dist = int(c) - self.n
            if self.cfg.residual_include_distance_features:
                log_dist = np.log10(max(dist, 1)) / np.log10(max(max_dist, 2))
                lin_dist = dist / max(max_dist, 1)
            else:
                log_dist = 0.0
                lin_dist = 0.0
            local_density = 1.0 / max(float(gaps[i]), 1.0) if self.cfg.residual_include_density_features else 0.0
            feats.extend(
                [
                    (i + 1) / self.window_size,
                    float(log_dist),
                    float(lin_dist),
                    float(self.tested[i] == -1),
                    float(self.tested[i] == 1),
                    float(local_density),
                ]
            )
        global_feats = [
            log_n / max(np.log10(max(self.cfg.n_max, self.n + 10)), 1.0),
            min(1.0, self.total_cost / max(self.cfg.budget, 1e-6)),
            1.0 - min(1.0, self.total_cost / max(self.cfg.budget, 1e-6)),
            self.steps / max(self.cfg.max_steps, 1),
            1.0 - self.steps / max(self.cfg.max_steps, 1),
            float(density),
            float(self.cfg.residual_sieve_bound) / 1000.0,
        ]
        return np.asarray(feats + global_feats, dtype=np.float32)

    def info(self, **extra: object) -> Dict[str, object]:
        d: Dict[str, object] = {
            "n": self.n,
            "true_next_prime": self.true_next_prime,
            "window_start": int(self.survivors[0]),
            "window_end": int(self.survivors[-1]),
            "steps": self.steps,
            "total_cost": round(float(self.total_cost), 6),
            "baseline_cost": round(float(self.baseline_cost), 6),
            "success": self.success,
            "total_tests": int(self.total_tests),
            "trace": list(self.trace[-20:]),
        }
        d.update(extra)
        return d

    def action_mask_float(self) -> np.ndarray:
        return self.legal_actions().astype(np.float32)

    def expert_rollout(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        while not self.done:
            target_idx: Optional[int] = None
            for i in range(self.window_size):
                if self.tested[i] == 0:
                    target_idx = i
                    break
                if self.tested[i] == 1 and self.are_all_prior_candidates_eliminated_or_composite(i):
                    target_idx = self.guess_action_offset + i
                    break
            action = self.expand_action if target_idx is None else int(target_idx)
            obs = self.observation().copy()
            mask = self.action_mask_float().copy()
            result = self.step(action)
            rows.append({"observation": obs, "mask": mask, "action": action, "reward": result.reward})
            if result.done:
                break
        return rows
