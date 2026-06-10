from __future__ import annotations

from dataclasses import dataclass

from .core import NullModel, SequenceDataset, Statistic
from .scoring import auc_real_vs_null

import numpy as np


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    scores: dict[str, float]
    promoted: list[str]
    null_name: str


@dataclass(frozen=True)
class AbsorptionLedger:
    rounds: list[RoundResult]
    final_null: NullModel


def absorption_loop(real, base_null, candidate_stats, max_rounds, promote_fn):
    current = base_null
    rounds: list[RoundResult] = []
    for round_index in range(max(0, int(max_rounds))):
        scores: dict[str, float] = {}
        promoted_stats: list[Statistic] = []
        for stat_index, stat in enumerate(candidate_stats):
            rng = np.random.default_rng(1729 + 1009 * round_index + stat_index)
            auc = auc_real_vs_null(stat, real, current, n=2000, rng=rng, seed_split=round_index * 100 + stat_index)
            scores[str(stat.name)] = float(auc)
            if bool(promote_fn(float(auc))):
                promoted_stats.append(stat)
        rounds.append(
            RoundResult(
                round_index=round_index,
                scores=scores,
                promoted=[str(stat.name) for stat in promoted_stats],
                null_name=str(current.name),
            )
        )
        if not promoted_stats:
            break
        for stat in promoted_stats:
            current = current.absorb(stat, real)
    return AbsorptionLedger(rounds=rounds, final_null=current)
