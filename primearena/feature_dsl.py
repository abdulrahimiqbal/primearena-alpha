from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .null_worlds import PrimeWindow


@dataclass(frozen=True)
class FeatureExpr:
    op: str
    args: Tuple[Any, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "args": list(self.args), "expression": str(self), "complexity": self.complexity()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FeatureExpr":
        return FeatureExpr(str(data["op"]), tuple(data.get("args", [])))

    def complexity(self) -> float:
        base = {
            "residue": 1.0,
            "gap": 1.0,
            "gap_mod": 1.5,
            "count_survivors": 1.0,
            "count_pattern": 2.0,
            "fourier_amp": 3.0,
            "character_sum": 3.0,
            "local_density": 1.0,
            "pair_residue": 2.0,
        }.get(self.op, 2.0)
        return float(base + 0.15 * len(self.args))

    def evaluate(self, window: PrimeWindow) -> np.ndarray:
        marked = window.values[window.indicator > 0].astype(np.int64)
        if self.op == "residue":
            d, q = int(self.args[0]), int(self.args[1])
            return np.asarray([(window.n + d) % q], dtype=np.float32)
        if self.op == "gap":
            i = int(self.args[0])
            gaps = np.diff(marked) if marked.size > 1 else np.asarray([], dtype=np.int64)
            return np.asarray([float(gaps[i]) if 0 <= i < gaps.size else 0.0], dtype=np.float32)
        if self.op == "gap_mod":
            i, q = int(self.args[0]), int(self.args[1])
            gaps = np.diff(marked) if marked.size > 1 else np.asarray([], dtype=np.int64)
            return np.asarray([float(gaps[i] % q) if 0 <= i < gaps.size else 0.0], dtype=np.float32)
        if self.op == "count_survivors":
            lo, hi = int(self.args[0]), int(self.args[1])
            mask = (window.offsets >= lo) & (window.offsets <= hi)
            return np.asarray([float(np.sum(window.indicator[mask] > 0))], dtype=np.float32)
        if self.op == "count_pattern":
            offsets = tuple(int(x) for x in self.args[0])
            values = set(int(v) for v in marked)
            hits = 0
            for v in window.values:
                hits += int(all(int(v) + d in values for d in offsets))
            return np.asarray([float(hits)], dtype=np.float32)
        if self.op == "fourier_amp":
            q, freq = int(self.args[0]), int(self.args[1])
            counts = np.bincount((marked % q).astype(np.int64), minlength=q).astype(np.float64) if marked.size else np.zeros(q)
            spec = np.fft.rfft(counts)
            idx = min(max(1, freq), spec.size - 1)
            return np.asarray([float(abs(spec[idx]) / max(marked.size, 1))], dtype=np.float32)
        if self.op == "character_sum":
            q, char_id = int(self.args[0]), int(self.args[1])
            if marked.size == 0:
                return np.zeros(2, dtype=np.float32)
            angles = 2.0 * np.pi * char_id * (marked % q) / float(q)
            return np.asarray([float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))], dtype=np.float32)
        if self.op == "local_density":
            h = int(self.args[0])
            mask = window.offsets <= h
            return np.asarray([float(np.mean(window.indicator[mask] > 0)) if np.any(mask) else 0.0], dtype=np.float32)
        if self.op == "pair_residue":
            q, lag = int(self.args[0]), int(self.args[1])
            residues = (marked % q).astype(np.int64)
            if residues.size <= lag:
                return np.zeros(q, dtype=np.float32)
            pair_delta = (residues[lag:] - residues[:-lag]) % q
            counts = np.bincount(pair_delta, minlength=q).astype(np.float32)
            return counts / max(float(counts.sum()), 1.0)
        raise ValueError(f"Unknown PrimeLead feature expression op: {self.op}")

    def __str__(self) -> str:
        if self.op == "residue":
            return f"residue(n + {self.args[0]}, {self.args[1]})"
        if self.op == "gap":
            return f"gap({self.args[0]})"
        if self.op == "gap_mod":
            return f"gap({self.args[0]}) mod {self.args[1]}"
        if self.op == "count_survivors":
            return f"count_survivors([{self.args[0]}, {self.args[1]}])"
        if self.op == "count_pattern":
            return f"count_pattern({tuple(self.args[0])})"
        if self.op == "fourier_amp":
            return f"fourier_amp({self.args[0]}, {self.args[1]})"
        if self.op == "character_sum":
            return f"character_sum({self.args[0]}, {self.args[1]})"
        if self.op == "local_density":
            return f"local_density({self.args[0]})"
        if self.op == "pair_residue":
            return f"pair_residue({self.args[0]}, lag={self.args[1]})"
        return f"{self.op}({', '.join(map(str, self.args))})"


def default_feature_expressions(q_values: Iterable[int]) -> list[FeatureExpr]:
    out = [
        FeatureExpr("local_density", (64,)),
        FeatureExpr("count_survivors", (1, 128)),
        FeatureExpr("gap", (0,)),
        FeatureExpr("gap_mod", (0, 6)),
        FeatureExpr("count_pattern", ((0, 2),)),
    ]
    for q in q_values:
        out.append(FeatureExpr("fourier_amp", (int(q), 1)))
        out.append(FeatureExpr("character_sum", (int(q), 1)))
        out.append(FeatureExpr("pair_residue", (int(q), 1)))
    return out
