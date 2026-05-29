from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .baselines import wheel_scan
from .config import CurriculumConfig, EnvConfig, RunConfig, load_config
from .oracle import next_prime


@dataclass
class HardCase:
    n: int
    next_prime: int
    gap: int
    baseline_cost: float
    score: float
    reason: str = "gap_wheel"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class AdversarialCurriculum:
    """Samples a mixture of random n values and previously mined hard n values."""

    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.hard_cases: List[HardCase] = []
        if cfg.curriculum.hard_cases_path:
            self.hard_cases = load_hard_cases(cfg.curriculum.hard_cases_path)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.curriculum.enabled and self.hard_cases)

    def sample_n(self) -> Optional[int]:
        c = self.cfg.curriculum
        if self.enabled and self.rng.random() < c.hard_case_mix:
            weights = np.asarray([max(1e-6, h.score) for h in self.hard_cases], dtype=np.float64)
            weights /= weights.sum()
            idx = int(self.rng.choice(len(self.hard_cases), p=weights))
            return int(self.hard_cases[idx].n)
        return None

    def maybe_mine(self, step: int) -> None:
        c = self.cfg.curriculum
        if not c.enabled or c.mine_every <= 0 or step % c.mine_every != 0:
            return
        mined = mine_hard_cases(self.cfg.env, c, seed=int(self.rng.integers(0, 1_000_000_000)))
        merged: Dict[int, HardCase] = {h.n: h for h in self.hard_cases}
        for h in mined:
            old = merged.get(h.n)
            if old is None or h.score > old.score:
                merged[h.n] = h
        self.hard_cases = sorted(merged.values(), key=lambda x: x.score, reverse=True)[: max(c.top_k, 1)]
        if c.hard_cases_path:
            save_hard_cases(c.hard_cases_path, self.hard_cases)


def score_n(n: int, env_cfg: EnvConfig, c_cfg: CurriculumConfig) -> HardCase:
    p = next_prime(n)
    gap = int(p - n)
    baseline = wheel_scan(n, env_cfg)
    log_scale = max(float(np.log(max(n, 3))), 1.0)
    if c_cfg.score == "wheel_cost":
        score = float(baseline.cost)
        reason = "wheel_cost"
    else:
        score = float(gap / log_scale + 0.25 * baseline.cost)
        reason = "gap_wheel"
    if gap < c_cfg.min_gap:
        score *= 0.1
    return HardCase(n=int(n), next_prime=int(p), gap=gap, baseline_cost=float(baseline.cost), score=score, reason=reason)


def mine_hard_cases(env_cfg: EnvConfig, c_cfg: CurriculumConfig, seed: int = 0) -> List[HardCase]:
    rng = np.random.default_rng(seed)
    cases: List[HardCase] = []
    for _ in range(max(1, c_cfg.mine_candidates)):
        n = int(rng.integers(env_cfg.n_min, env_cfg.n_max + 1))
        h = score_n(n, env_cfg, c_cfg)
        if h.gap >= c_cfg.min_gap:
            cases.append(h)
    cases.sort(key=lambda x: x.score, reverse=True)
    return cases[: max(1, c_cfg.top_k)]


def load_hard_cases(path: str | Path) -> List[HardCase]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[HardCase] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(HardCase(**d))
    return out


def save_hard_cases(path: str | Path, cases: Iterable[HardCase]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(h.to_dict(), sort_keys=True) for h in cases) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.json")
    parser.add_argument("--out", type=str, default="runs/hard_cases.jsonl")
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.candidates is not None:
        cfg.curriculum.mine_candidates = args.candidates
    if args.top_k is not None:
        cfg.curriculum.top_k = args.top_k
    cases = mine_hard_cases(cfg.env, cfg.curriculum, seed=args.seed)
    save_hard_cases(args.out, cases)
    print(json.dumps({"out": args.out, "count": len(cases), "top": [h.to_dict() for h in cases[:10]]}, indent=2))


if __name__ == "__main__":
    main()
