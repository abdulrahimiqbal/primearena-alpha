from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .baselines import wheel_scan
from .config import EnvConfig
from .oracle import is_prime, next_prime


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class PrimeArena:
    """Cost-limited search game for finding the next prime after n.

    Fixed action layout:
      [0, P)                         filter by small_prime[action]
      [P, P + W)                     primality-test candidate i
      [P + W, P + 2W)                guess candidate i as next prime
      [P + 2W]                       expand window
    """

    def __init__(self, cfg: EnvConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.reset()

    @property
    def window_size(self) -> int:
        return self.cfg.window_size

    @property
    def num_filter_actions(self) -> int:
        return len(self.cfg.small_primes)

    @property
    def action_count(self) -> int:
        return self.num_filter_actions + 2 * self.window_size + 1

    @property
    def expand_action(self) -> int:
        return self.num_filter_actions + 2 * self.window_size

    @property
    def feature_dim_per_candidate(self) -> int:
        # offset, log distance, parity, eliminated, tested composite, tested prime,
        # plus two residue features for each small prime: normalized residue, divisible flag.
        return 6 + 2 * len(self.cfg.small_primes)

    @property
    def global_feature_dim(self) -> int:
        return 7

    @property
    def observation_dim(self) -> int:
        return self.window_size * self.feature_dim_per_candidate + self.global_feature_dim

    def clone(self) -> "PrimeArena":
        return copy.deepcopy(self)

    def reset(self, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            n = self._sample_n()
        self.n = int(n)
        self.true_next_prime = next_prime(self.n)
        self.window_start = self.n + 1
        self.steps = 0
        self.total_cost = 0.0
        self.done = False
        self.success = False
        self.applied_filters = [False] * len(self.cfg.small_primes)
        self.eliminated = np.zeros(self.window_size, dtype=np.bool_)
        # -1 composite, 0 unknown, 1 prime
        self.tested = np.zeros(self.window_size, dtype=np.int8)
        self.trace: List[str] = []
        self.baseline_cost = wheel_scan(self.n, self.cfg).cost
        return self.observation()

    def _sample_n(self) -> int:
        for _ in range(self.cfg.max_resample_attempts):
            n = self.rng.randint(self.cfg.n_min, self.cfg.n_max)
            if not self.cfg.ensure_prime_in_window:
                return n
            p = next_prime(n)
            if p <= n + self.cfg.window_size:
                return n
        return self.rng.randint(self.cfg.n_min, self.cfg.n_max)

    def candidates(self) -> np.ndarray:
        return np.arange(self.window_start, self.window_start + self.window_size, dtype=np.int64)

    def candidate_index(self, value: int) -> Optional[int]:
        idx = value - self.window_start
        if 0 <= idx < self.window_size:
            return int(idx)
        return None

    @property
    def guess_action_offset(self) -> int:
        return self.num_filter_actions + self.window_size

    def is_candidate_tested_prime(self, index: int) -> bool:
        return 0 <= index < self.window_size and bool(self.tested[index] == 1)

    def are_all_prior_candidates_eliminated_or_composite(self, index: int) -> bool:
        if index < 0 or index >= self.window_size:
            return False
        candidates = self.candidates()
        for i in range(index):
            if candidates[i] < 2:
                continue
            if bool(self.eliminated[i]) or self.tested[i] == -1:
                continue
            return False
        return True

    def is_certified_next_prime_guess(self, index: int) -> bool:
        return self.is_candidate_tested_prime(index) and self.are_all_prior_candidates_eliminated_or_composite(index)

    def certified_guess_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.bool_)
        if self.done:
            return mask
        legal = self.legal_actions()
        offset_guess = self.guess_action_offset
        for i in range(self.window_size):
            action = offset_guess + i
            if legal[action] and self.is_certified_next_prime_guess(i):
                mask[action] = True
        return mask

    def candidate_status(self, index: int) -> Dict[str, object]:
        cands = self.candidates()
        if index < 0 or index >= self.window_size:
            return {"index": int(index), "valid": False}
        tested = int(self.tested[index])
        return {
            "index": int(index),
            "candidate": int(cands[index]),
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
        # filter actions
        for i, used in enumerate(self.applied_filters):
            if not used and self.total_cost + self.cfg.filter_cost <= self.cfg.budget:
                mask[i] = True
        # test / guess candidates
        offset_test = self.num_filter_actions
        offset_guess = self.guess_action_offset
        candidates = self.candidates()
        for i, c in enumerate(candidates):
            if c >= 2 and not self.eliminated[i]:
                if self.tested[i] == 0 and self.total_cost + self.cfg.test_cost <= self.cfg.budget:
                    mask[offset_test + i] = True
                if self.total_cost + self.cfg.guess_cost <= self.cfg.budget:
                    mask[offset_guess + i] = True
        if self.total_cost + self.cfg.expand_cost <= self.cfg.budget:
            mask[self.expand_action] = True
        return mask

    def action_to_str(self, action: int) -> str:
        p = self.num_filter_actions
        w = self.window_size
        if action < p:
            return f"filter mod {self.cfg.small_primes[action]}"
        if action < p + w:
            i = action - p
            return f"test {int(self.candidates()[i])}"
        if action < p + 2 * w:
            i = action - p - w
            return f"guess {int(self.candidates()[i])}"
        if action == self.expand_action:
            return "expand window"
        return f"invalid {action}"

    def _finish(self, reward: float, success: bool, reason: str) -> StepResult:
        self.done = True
        self.success = success
        obs = self.observation()
        return StepResult(obs, reward, True, self.info(reason=reason))

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.observation(), 0.0, True, self.info(reason="already_done"))
        legal = self.legal_actions()
        if action < 0 or action >= self.action_count or not legal[action]:
            self.steps += 1
            self.total_cost += self.cfg.invalid_action_penalty
            reward = -self.cfg.invalid_action_penalty
            self.trace.append(f"invalid {action}")
            if self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget:
                return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "invalid_timeout")
            return StepResult(self.observation(), reward, False, self.info(reason="invalid"))

        self.steps += 1
        p = self.num_filter_actions
        w = self.window_size
        candidates = self.candidates()

        if action < p:
            q = self.cfg.small_primes[action]
            self.applied_filters[action] = True
            before = int(self.eliminated.sum())
            # eliminate multiples of q but never eliminate q itself if q is in the tiny range
            divisible = (candidates % q == 0) & (candidates != q)
            self.eliminated |= divisible
            eliminated_now = int(self.eliminated.sum()) - before
            self.total_cost += self.cfg.filter_cost
            reward = -self.cfg.step_cost_scale * self.cfg.filter_cost
            self.trace.append(f"filter mod {q} -> eliminated {eliminated_now}")
            done = self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget
            if done:
                return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "budget_or_steps")
            return StepResult(self.observation(), reward, False, self.info(reason="filter"))

        if action < p + w:
            i = action - p
            c = int(candidates[i])
            prime = is_prime(c)
            self.tested[i] = 1 if prime else -1
            self.total_cost += self.cfg.test_cost
            reward = -self.cfg.step_cost_scale * self.cfg.test_cost
            self.trace.append(f"test {c} -> {'prime' if prime else 'composite'}")
            done = self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget
            if done:
                return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "budget_or_steps")
            return StepResult(self.observation(), reward, False, self.info(reason="test", tested_prime=prime))

        if action < p + 2 * w:
            i = action - p - w
            c = int(candidates[i])
            self.total_cost += self.cfg.guess_cost
            correct = c == self.true_next_prime
            self.trace.append(f"guess {c} -> {'correct' if correct else 'wrong'}")
            if correct:
                delta = (self.baseline_cost - self.total_cost) / max(self.baseline_cost, 1e-6)
                reward = self.cfg.correct_reward + max(-1.0, min(1.0, delta))
                return self._finish(reward, True, "correct_guess")
            return self._finish(self.cfg.wrong_guess_reward - self.cfg.step_cost_scale * self.total_cost, False, "wrong_guess")

        # expand window
        self.window_start += self.window_size
        self.eliminated = np.zeros(self.window_size, dtype=np.bool_)
        self.tested = np.zeros(self.window_size, dtype=np.int8)
        self.applied_filters = [False] * len(self.cfg.small_primes)
        self.total_cost += self.cfg.expand_cost
        reward = -self.cfg.step_cost_scale * self.cfg.expand_cost
        self.trace.append(f"expand -> {self.window_start}-{self.window_start + self.window_size - 1}")
        done = self.steps >= self.cfg.max_steps or self.total_cost >= self.cfg.budget
        if done:
            return self._finish(self.cfg.timeout_reward - self.cfg.step_cost_scale * self.total_cost, False, "budget_or_steps")
        return StepResult(self.observation(), reward, False, self.info(reason="expand"))

    def observation(self) -> np.ndarray:
        candidates = self.candidates().astype(np.float64)
        feats = []
        log_n = np.log10(max(self.n, 10))
        denom = max(np.log10(max(self.cfg.n_max, self.n + self.window_size)), 1.0)
        for i, c in enumerate(candidates):
            row = [
                (i + 1) / self.window_size,
                np.log10(max(c - self.n, 1)) / np.log10(self.window_size + 1),
                float(c % 2 == 1),
                float(self.eliminated[i]),
                float(self.tested[i] == -1),
                float(self.tested[i] == 1),
            ]
            for q in self.cfg.small_primes:
                residue = int(c % q)
                row.append(residue / max(q - 1, 1))
                row.append(float(residue == 0 and c != q))
            feats.extend(row)
        global_feats = [
            log_n / denom,
            min(1.0, self.total_cost / max(self.cfg.budget, 1e-6)),
            1.0 - min(1.0, self.total_cost / max(self.cfg.budget, 1e-6)),
            self.steps / max(self.cfg.max_steps, 1),
            1.0 - self.steps / max(self.cfg.max_steps, 1),
            float(np.log(max(self.n, 3))) / max(self.window_size, 1),
            (self.window_start - self.n) / max(self.window_size, 1),
        ]
        return np.asarray(feats + global_feats, dtype=np.float32)

    def info(self, **extra: object) -> Dict[str, object]:
        d: Dict[str, object] = {
            "n": self.n,
            "true_next_prime": self.true_next_prime,
            "window_start": self.window_start,
            "steps": self.steps,
            "total_cost": round(float(self.total_cost), 6),
            "baseline_cost": round(float(self.baseline_cost), 6),
            "success": self.success,
            "trace": list(self.trace[-20:]),
        }
        d.update(extra)
        return d

    def action_mask_float(self) -> np.ndarray:
        return self.legal_actions().astype(np.float32)
