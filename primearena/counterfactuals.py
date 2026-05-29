from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .config import EnvConfig
from .baselines import residual_true_index
from .residual_env import PrimeArenaResidual


@dataclass
class CounterfactualPair:
    n_a: int
    n_b: int
    true_index_a: int
    true_index_b: int
    scale_delta: float
    survivor_gap_l1: float
    same_mod210_pattern: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def survivor_signature(env: PrimeArenaResidual, mod: int = 210) -> np.ndarray:
    survivors = env.candidates()
    gaps = np.diff(np.concatenate([[env.n], survivors]))
    residues = survivors % mod
    # Keep the signature deliberately coarse so matched pairs exist locally.
    return np.concatenate([np.clip(gaps[:32], 0, 64), residues[:32]]).astype(np.float64)


def build_counterfactual_pairs(
    cfg: EnvConfig,
    samples: int = 128,
    seed: int = 0,
    bound: Optional[int] = None,
) -> List[CounterfactualPair]:
    rng = np.random.default_rng(seed)
    envs: List[PrimeArenaResidual] = []
    signatures: List[np.ndarray] = []
    true_indices: List[int] = []
    old_bound = cfg.residual_sieve_bound
    if bound is not None:
        cfg.residual_sieve_bound = int(bound)
    try:
        for _ in range(max(samples * 4, samples + 16)):
            env = PrimeArenaResidual(cfg, seed=int(rng.integers(0, 1_000_000_000)))
            idx = residual_true_index(env.n, cfg, bound=cfg.residual_sieve_bound)
            if idx >= cfg.residual_window_size:
                continue
            envs.append(env)
            signatures.append(survivor_signature(env))
            true_indices.append(idx)
            if len(envs) >= max(samples * 2, samples + 8):
                break
    finally:
        cfg.residual_sieve_bound = old_bound

    pairs: List[CounterfactualPair] = []
    used: set[int] = set()
    for i, env in enumerate(envs):
        if i in used:
            continue
        best_j = None
        best_score = float("inf")
        for j in range(i + 1, len(envs)):
            if j in used or true_indices[j] == true_indices[i]:
                continue
            scale_delta = abs(np.log(max(env.n, 3)) - np.log(max(envs[j].n, 3)))
            gap_l1 = float(np.mean(np.abs(signatures[i][:32] - signatures[j][:32])))
            score = scale_delta + 0.01 * gap_l1
            if score < best_score:
                best_score = score
                best_j = j
        if best_j is None:
            continue
        used.add(i)
        used.add(best_j)
        residues_a = env.candidates()[:32] % 210
        residues_b = envs[best_j].candidates()[:32] % 210
        pairs.append(
            CounterfactualPair(
                n_a=int(env.n),
                n_b=int(envs[best_j].n),
                true_index_a=int(true_indices[i]),
                true_index_b=int(true_indices[best_j]),
                scale_delta=float(abs(np.log(max(env.n, 3)) - np.log(max(envs[best_j].n, 3)))),
                survivor_gap_l1=float(np.mean(np.abs(signatures[i][:32] - signatures[best_j][:32]))),
                same_mod210_pattern=bool(np.array_equal(residues_a, residues_b)),
            )
        )
        if len(pairs) >= samples:
            break
    return pairs
