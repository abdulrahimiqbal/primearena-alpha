from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LEAD_CARD_V2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "name",
        "formula",
        "complexity",
        "domain",
        "ood_auc",
        "bits_saved",
        "bits_per_complexity",
        "nulls_beaten",
        "absorption_round",
        "scaling",
        "explained_by",
        "extrapolation",
        "seeds",
        "failure_cases",
    ],
}


@dataclass(frozen=True)
class LeadCardV2:
    name: str
    formula: str
    complexity: float
    domain: str
    ood_auc: float
    bits_saved: float
    nulls_beaten: list[str]
    absorption_round: int
    scaling: dict[str, Any]
    explained_by: str | None
    extrapolation: dict[str, Any]
    seeds: list[int]
    failure_cases: list[str] = field(default_factory=list)

    @property
    def bits_per_complexity(self) -> float:
        return float(self.bits_saved) / max(float(self.complexity), 1e-12)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bits_per_complexity"] = self.bits_per_complexity
        return data


def ranking_key(card: LeadCardV2) -> float:
    return card.bits_per_complexity
