from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional

from .config import EnvConfig
from .oracle import is_prime, next_prime, passes_wheel, small_prime_sieve


@dataclass
class BaselineResult:
    name: str
    n: int
    next_prime: int
    success: bool
    cost: float
    tests: int
    filters: int
    actions: List[str]


@dataclass
class ResidualBaselineResult:
    name: str
    n: int
    true_next_prime: int
    success: bool
    tests_used: int
    cost: float
    tested_order: List[int]


def odd_scan(n: int, cfg: EnvConfig) -> BaselineResult:
    p = next_prime(n)
    tests = 0
    actions: List[str] = []
    c = n + 1
    if c <= 2:
        tests += 1
        actions.append("test 2")
        return BaselineResult("odd_scan", n, 2, True, tests * cfg.test_cost, tests, 0, actions)
    if c % 2 == 0:
        c += 1
    while True:
        tests += 1
        actions.append(f"test {c}")
        if is_prime(c):
            break
        c += 2
    return BaselineResult("odd_scan", n, p, c == p, tests * cfg.test_cost, tests, 0, actions)


def wheel_scan(n: int, cfg: EnvConfig, wheel_primes: Optional[Iterable[int]] = None) -> BaselineResult:
    p = next_prime(n)
    wheel = list(wheel_primes or cfg.small_primes)
    tests = 0
    actions: List[str] = [f"filter mod {q}" for q in wheel]
    c = n + 1
    while True:
        if passes_wheel(c, wheel):
            tests += 1
            actions.append(f"test {c}")
            if is_prime(c):
                break
        c += 1
    cost = len(wheel) * cfg.filter_cost + tests * cfg.test_cost
    return BaselineResult("wheel_scan", n, p, c == p, cost, tests, len(wheel), actions)


def segmented_sieve_scan(n: int, cfg: EnvConfig) -> BaselineResult:
    """Segmented-sieve-style baseline with a simple cost proxy.

    It marks composites in the current window using primes up to sqrt(window_end), then
    tests survivors in order. The cost model is intentionally conservative and simple.
    """
    p = next_prime(n)
    start = n + 1
    end = start + cfg.window_size - 1
    actions: List[str] = []
    total_cost = 0.0
    filters = 0
    tests = 0
    while p > end:
        actions.append(f"expand {start}-{end}")
        total_cost += cfg.expand_cost
        start = end + 1
        end = start + cfg.window_size - 1
    primes = small_prime_sieve(int(end**0.5) + 1)
    survivors = [True] * (end - start + 1)
    for q in primes:
        filters += 1
        total_cost += cfg.filter_cost * 0.5
        actions.append(f"sieve mod {q}")
        first = max(q * q, ((start + q - 1) // q) * q)
        for x in range(first, end + 1, q):
            survivors[x - start] = False
    for i, ok in enumerate(survivors):
        c = start + i
        if c < 2:
            continue
        if ok:
            tests += 1
            total_cost += cfg.test_cost
            actions.append(f"test survivor {c}")
            if is_prime(c):
                return BaselineResult("segmented_sieve", n, p, c == p, total_cost, tests, filters, actions)
    return BaselineResult("segmented_sieve", n, p, False, total_cost, tests, filters, actions)


def all_baselines(n: int, cfg: EnvConfig) -> Dict[str, BaselineResult]:
    return {
        "odd_scan": odd_scan(n, cfg),
        "wheel_scan": wheel_scan(n, cfg),
        "segmented_sieve": segmented_sieve_scan(n, cfg),
    }


def residual_sieve_primes(cfg: EnvConfig, bound: Optional[int] = None) -> List[int]:
    return [p for p in small_prime_sieve(int(bound or cfg.residual_sieve_bound)) if p >= 2]


def is_residual_survivor(x: int, primes: Iterable[int]) -> bool:
    if x < 2:
        return False
    for q in primes:
        if x != q and x % q == 0:
            return False
    return True


def residual_survivors_after(
    n: int,
    cfg: EnvConfig,
    count: int,
    bound: Optional[int] = None,
    start: Optional[int] = None,
) -> List[int]:
    primes = residual_sieve_primes(cfg, bound)
    x = int(start if start is not None else n + 1)
    out: List[int] = []
    while len(out) < count:
        if is_residual_survivor(x, primes):
            out.append(x)
        x += 1
    return out


def residual_true_index(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> int:
    p = next_prime(n)
    survivors = residual_survivors_after(n, cfg, max(1, cfg.residual_window_size), bound=bound)
    while p not in survivors:
        survivors.extend(residual_survivors_after(survivors[-1], cfg, cfg.residual_window_size, bound=bound, start=survivors[-1] + 1))
    return survivors.index(p)


def _residual_order_result(name: str, n: int, cfg: EnvConfig, order: List[int]) -> ResidualBaselineResult:
    true_p = next_prime(n)
    tested: List[int] = []
    for x in order:
        tested.append(int(x))
        if is_prime(int(x)):
            break
    return ResidualBaselineResult(
        name=name,
        n=int(n),
        true_next_prime=int(true_p),
        success=bool(tested and tested[-1] == true_p),
        tests_used=len(tested),
        cost=float(len(tested) * cfg.test_cost + cfg.guess_cost),
        tested_order=tested,
    )


def nearest_survivor_first(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> ResidualBaselineResult:
    count = max(1, cfg.residual_window_size)
    survivors = residual_survivors_after(n, cfg, count, bound=bound)
    while next_prime(n) not in survivors:
        survivors.extend(residual_survivors_after(n, cfg, count, bound=bound, start=survivors[-1] + 1))
    return _residual_order_result("nearest_survivor_first", n, cfg, survivors)


def random_survivor_order(n: int, cfg: EnvConfig, bound: Optional[int] = None, seed: int = 0) -> ResidualBaselineResult:
    true_p = next_prime(n)
    survivors = residual_survivors_after(n, cfg, max(1, cfg.residual_window_size), bound=bound)
    while true_p not in survivors:
        survivors.extend(residual_survivors_after(n, cfg, cfg.residual_window_size, bound=bound, start=survivors[-1] + 1))
    rng = random.Random(seed)
    order = list(survivors)
    rng.shuffle(order)
    return _residual_order_result("random_survivor_order", n, cfg, order)


def distance_weighted_survivor_order(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> ResidualBaselineResult:
    survivors = residual_survivors_after(n, cfg, max(1, cfg.residual_window_size), bound=bound)
    while next_prime(n) not in survivors:
        survivors.extend(residual_survivors_after(n, cfg, cfg.residual_window_size, bound=bound, start=survivors[-1] + 1))
    order = sorted(survivors, key=lambda x: (x - n, x))
    return _residual_order_result("distance_weighted_survivor_order", n, cfg, order)


def one_over_logn_survivor_order(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> ResidualBaselineResult:
    survivors = residual_survivors_after(n, cfg, max(1, cfg.residual_window_size), bound=bound)
    while next_prime(n) not in survivors:
        survivors.extend(residual_survivors_after(n, cfg, cfg.residual_window_size, bound=bound, start=survivors[-1] + 1))
    order = sorted(survivors, key=lambda x: (-1.0 / max(math.log(max(x, 3)), 1e-9), x))
    return _residual_order_result("one_over_logn_survivor_order", n, cfg, order)


def sequential_miller_rabin_survivor_cost(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> ResidualBaselineResult:
    return nearest_survivor_first(n, cfg, bound=bound)


def segmented_sieve_oracle_cost(n: int, cfg: EnvConfig, bound: Optional[int] = None) -> ResidualBaselineResult:
    true_p = next_prime(n)
    return ResidualBaselineResult(
        name="segmented_sieve_oracle_cost",
        n=int(n),
        true_next_prime=int(true_p),
        success=True,
        tests_used=1,
        cost=float(cfg.test_cost + cfg.guess_cost),
        tested_order=[int(true_p)],
    )


def all_residual_baselines(n: int, cfg: EnvConfig, bound: Optional[int] = None, seed: int = 0) -> Dict[str, ResidualBaselineResult]:
    return {
        "nearest": nearest_survivor_first(n, cfg, bound=bound),
        "random": random_survivor_order(n, cfg, bound=bound, seed=seed),
        "distance_weighted": distance_weighted_survivor_order(n, cfg, bound=bound),
        "one_over_logn": one_over_logn_survivor_order(n, cfg, bound=bound),
        "sequential_miller_rabin": sequential_miller_rabin_survivor_cost(n, cfg, bound=bound),
        "segmented_sieve_oracle": segmented_sieve_oracle_cost(n, cfg, bound=bound),
    }
