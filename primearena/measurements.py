from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from .null_worlds import PrimeWindow
from .oracle import small_prime_sieve


@dataclass(frozen=True)
class Measurement:
    name: str
    cost: float
    complexity: float
    formula: str
    fn: Callable[[PrimeWindow], np.ndarray]

    def evaluate(self, window: PrimeWindow) -> np.ndarray:
        out = np.asarray(self.fn(window), dtype=np.float32).reshape(-1)
        if out.size == 0:
            return np.zeros(1, dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _selected_values(window: PrimeWindow) -> np.ndarray:
    return window.values[window.indicator > 0].astype(np.int64)


def _safe_density(window: PrimeWindow) -> float:
    return float(np.mean(window.indicator > 0)) if window.indicator.size else 0.0


def local_density_measurement(radius: int) -> Measurement:
    def fn(window: PrimeWindow) -> np.ndarray:
        block = max(1, int(radius))
        chunks = [window.indicator[i : i + block] for i in range(0, window.indicator.size, block)]
        dens = np.asarray([float(np.mean(c)) if c.size else 0.0 for c in chunks], dtype=np.float32)
        return np.asarray([_safe_density(window), float(np.std(dens)), float(np.max(dens) if dens.size else 0.0)], dtype=np.float32)

    return Measurement(
        name=f"local_density_H{int(radius)}",
        cost=1.0,
        complexity=1.0,
        formula=f"density over blocks of H={int(radius)}",
        fn=fn,
    )


def wheel_mask_measurement(bound: int) -> Measurement:
    primes = small_prime_sieve(int(bound))

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        if selected.size == 0:
            return np.zeros(3, dtype=np.float32)
        bad = np.zeros(selected.shape[0], dtype=bool)
        for p in primes:
            bad |= (selected != p) & ((selected % int(p)) == 0)
        return np.asarray([float(np.mean(bad)), float(np.sum(bad)), float(len(primes))], dtype=np.float32)

    return Measurement(
        name=f"wheel_coprime_B{int(bound)}",
        cost=1.0 + 0.002 * len(primes),
        complexity=1.0 + 0.01 * len(primes),
        formula=f"mean_{{x marked}} 1[gcd(x, primorial({int(bound)})) > 1]",
        fn=fn,
    )


def residue_hist_measurement(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        counts = np.bincount((selected % q).astype(np.int64), minlength=q).astype(np.float32) if selected.size else np.zeros(q, dtype=np.float32)
        counts /= max(float(counts.sum()), 1.0)
        return counts

    return Measurement(
        name=f"residue_hist_mod{q}",
        cost=1.0 + q / 1000.0,
        complexity=1.0 + q / 300.0,
        formula=f"histogram of marked residues x mod {q}",
        fn=fn,
    )


def residue_pair_measurement(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        residues = (selected % q).astype(np.int64)
        if residues.size <= 1:
            return np.zeros(min(q * q, 256), dtype=np.float32)
        bins = min(q * q, 256)
        pair_ids = (residues[:-1] * q + residues[1:]) % bins
        counts = np.bincount(pair_ids, minlength=bins).astype(np.float32)
        counts /= max(float(counts.sum()), 1.0)
        return counts

    return Measurement(
        name=f"pair_residue_mod{q}",
        cost=2.0 + q / 800.0,
        complexity=2.0 + q / 120.0,
        formula=f"histogram of consecutive marked residue pairs mod {q}",
        fn=fn,
    )


def _reduced_residue_map(q: int) -> Dict[int, int]:
    residues = [r for r in range(int(q)) if math.gcd(r, int(q)) == 1]
    return {int(r): i for i, r in enumerate(residues)}


def _consecutive_reduced_pair_ids(window: PrimeWindow, q: int) -> tuple[np.ndarray, int]:
    q = int(q)
    mapping = _reduced_residue_map(q)
    selected = _selected_values(window)
    residues = [mapping[int(x % q)] for x in selected if int(x % q) in mapping]
    m = max(1, len(mapping))
    if len(residues) <= 1:
        return np.asarray([], dtype=np.int64), m
    arr = np.asarray(residues, dtype=np.int64)
    return arr[:-1] * m + arr[1:], m


def consecutive_residue_pair_counts(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        pair_ids, m = _consecutive_reduced_pair_ids(window, q)
        counts = np.bincount(pair_ids, minlength=m * m).astype(np.float32) if pair_ids.size else np.zeros(m * m, dtype=np.float32)
        return counts / max(float(counts.sum()), 1.0)

    return Measurement(
        name=f"consecutive_residue_pair_counts_mod{q}",
        cost=1.8 + q / 2000.0,
        complexity=2.0 + len(_reduced_residue_map(q)) / 10.0,
        formula=f"normalized counts of consecutive marked prime residues (r_i, r_(i+1)) modulo {q}",
        fn=fn,
    )


def consecutive_residue_pair_transition_matrix(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        pair_ids, m = _consecutive_reduced_pair_ids(window, q)
        counts = np.bincount(pair_ids, minlength=m * m).astype(np.float32).reshape(m, m) if pair_ids.size else np.zeros((m, m), dtype=np.float32)
        rows = counts.sum(axis=1, keepdims=True)
        return (counts / np.maximum(rows, 1.0)).reshape(-1)

    return Measurement(
        name=f"consecutive_residue_pair_transition_mod{q}",
        cost=2.0 + q / 1800.0,
        complexity=2.5 + len(_reduced_residue_map(q)) / 8.0,
        formula=f"row-normalized transition matrix P(r_(i+1) | r_i) modulo {q}",
        fn=fn,
    )


def diagonal_vs_offdiagonal_pair_rate(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        pair_ids, m = _consecutive_reduced_pair_ids(window, q)
        if pair_ids.size == 0:
            return np.zeros(3, dtype=np.float32)
        a = pair_ids // m
        b = pair_ids % m
        same = float(np.mean(a == b))
        return np.asarray([same, 1.0 - same, same - (1.0 / max(m, 1))], dtype=np.float32)

    return Measurement(
        name=f"diagonal_vs_offdiagonal_pair_rate_mod{q}",
        cost=0.8 + q / 4000.0,
        complexity=1.0 + len(_reduced_residue_map(q)) / 50.0,
        formula=f"rate(r_i = r_(i+1)) vs off-diagonal consecutive residues modulo {q}",
        fn=fn,
    )


def same_residue_repeat_rate(q: int) -> Measurement:
    q = int(q)

    def fn(window: PrimeWindow) -> np.ndarray:
        pair_ids, m = _consecutive_reduced_pair_ids(window, q)
        if pair_ids.size == 0:
            return np.zeros(1, dtype=np.float32)
        return np.asarray([float(np.mean((pair_ids // m) == (pair_ids % m)))], dtype=np.float32)

    return Measurement(
        name=f"same_residue_repeat_rate_mod{q}",
        cost=0.6 + q / 5000.0,
        complexity=0.8 + len(_reduced_residue_map(q)) / 80.0,
        formula=f"mean 1[r_i = r_(i+1)] for consecutive marked residues modulo {q}",
        fn=fn,
    )


def pair_bias_spectrum(q: int, components: int = 8) -> Measurement:
    q = int(q)
    components = int(components)

    def fn(window: PrimeWindow) -> np.ndarray:
        pair_ids, m = _consecutive_reduced_pair_ids(window, q)
        mat = np.bincount(pair_ids, minlength=m * m).astype(np.float32).reshape(m, m) if pair_ids.size else np.zeros((m, m), dtype=np.float32)
        mat = mat / max(float(mat.sum()), 1.0)
        centered = mat - mat.mean()
        if centered.size == 1:
            vals = np.asarray([0.0], dtype=np.float32)
        else:
            vals = np.linalg.svd(centered, compute_uv=False).astype(np.float32)
        out = np.zeros(max(1, components), dtype=np.float32)
        out[: min(out.size, vals.size)] = vals[: min(out.size, vals.size)]
        return out

    return Measurement(
        name=f"pair_bias_spectrum_mod{q}",
        cost=1.5 + q / 2500.0,
        complexity=2.0 + components / 2.0 + len(_reduced_residue_map(q)) / 20.0,
        formula=f"top singular values of centered consecutive-residue pair matrix modulo {q}",
        fn=fn,
    )


def gap_constellation_measurement(k: int = 6) -> Measurement:
    k = int(k)

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        if selected.size <= 1:
            gaps = np.zeros(k, dtype=np.float32)
        else:
            raw = np.diff(selected).astype(np.float32)
            gaps = np.zeros(k, dtype=np.float32)
            gaps[: min(k, raw.size)] = raw[:k]
            gaps = np.log1p(gaps) / 8.0
        return np.asarray(
            [
                *gaps.tolist(),
                float(selected.size),
                float(window.gap_metadata.get("mean_gap", 0.0)) / max(float(window.indicator.size), 1.0),
                float(window.gap_metadata.get("max_gap", 0.0)) / max(float(window.indicator.size), 1.0),
            ],
            dtype=np.float32,
        )

    return Measurement(
        name=f"gap_constellation_k{k}",
        cost=1.5,
        complexity=1.5 + 0.1 * k,
        formula=f"first {k} consecutive marked gaps plus summary gap moments",
        fn=fn,
    )


def fourier_residue_measurement(q: int, freq_count: int = 4) -> Measurement:
    q = int(q)
    freq_count = int(freq_count)

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        counts = np.bincount((selected % q).astype(np.int64), minlength=q).astype(np.float32) if selected.size else np.zeros(q, dtype=np.float32)
        spectrum = np.fft.rfft(counts)
        amps = np.abs(spectrum[1 : freq_count + 1]).astype(np.float32)
        return amps / max(float(selected.size), 1.0)

    return Measurement(
        name=f"fourier_residue_mod{q}",
        cost=2.5 + q / 1000.0,
        complexity=2.5 + freq_count,
        formula=f"|FFT_k(count residues mod {q})| for k=1..{freq_count}",
        fn=fn,
    )


def character_sum_measurement(q: int, char_id: int = 1) -> Measurement:
    q = int(q)
    char_id = int(char_id)

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        if selected.size == 0:
            return np.zeros(2, dtype=np.float32)
        angles = 2.0 * math.pi * char_id * (selected % q).astype(np.float64) / float(q)
        return np.asarray([float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))], dtype=np.float32)

    return Measurement(
        name=f"character_sum_mod{q}_chi{char_id}",
        cost=2.0 + q / 1200.0,
        complexity=2.0 + q / 500.0,
        formula=f"mean exp(2*pi*i*{char_id}*x/{q}) over marked x",
        fn=fn,
    )


def extra_sieve_residue_measurement(start: int, end: int, max_primes: int = 16) -> Measurement:
    primes = [p for p in small_prime_sieve(int(end)) if p >= int(start)]
    if len(primes) > int(max_primes):
        idx = np.linspace(0, len(primes) - 1, int(max_primes)).round().astype(int)
        primes = [primes[int(i)] for i in sorted(set(idx.tolist()))]

    def fn(window: PrimeWindow) -> np.ndarray:
        selected = _selected_values(window)
        if selected.size == 0:
            return np.zeros(2 * len(primes), dtype=np.float32)
        vals: List[float] = []
        for p in primes:
            rem = selected % int(p)
            vals.append(float(np.mean(rem == 0)))
            vals.append(float(np.mean(np.minimum(rem, int(p) - rem) / float(p))))
        return np.asarray(vals, dtype=np.float32)

    return Measurement(
        name=f"extra_sieve_residue_{int(start)}_{int(end)}",
        cost=3.0 + 0.1 * len(primes),
        complexity=3.0 + 0.25 * len(primes),
        formula=f"divisibility and nearest residue distance for primes in [{int(start)}, {int(end)}]",
        fn=fn,
    )


def tuple_mask_measurement(offsets: Sequence[int]) -> Measurement:
    offs = tuple(int(x) for x in offsets)

    def fn(window: PrimeWindow) -> np.ndarray:
        marked = set(int(v) for v in _selected_values(window))
        hits = 0
        trials = 0
        for v in window.values:
            trials += 1
            hits += int(all(int(v) + d in marked for d in offs))
        return np.asarray([hits / max(trials, 1), float(hits)], dtype=np.float32)

    return Measurement(
        name="tuple_mask_" + "_".join(str(x) for x in offs),
        cost=2.0 + 0.2 * len(offs),
        complexity=2.0 + len(offs),
        formula="mean_x " + " ".join([f"1[x+{d} marked]" for d in offs]),
        fn=fn,
    )


def default_measurements(
    q_values: Iterable[int],
    sieve_bounds: Iterable[int],
    measurement_budget: float,
) -> List[Measurement]:
    candidates: List[Measurement] = [
        local_density_measurement(32),
        local_density_measurement(128),
        gap_constellation_measurement(6),
        tuple_mask_measurement((0, 2)),
    ]
    for bound in sieve_bounds:
        candidates.append(wheel_mask_measurement(int(bound)))
    for q in q_values:
        candidates.append(residue_hist_measurement(int(q)))
        candidates.append(residue_pair_measurement(int(q)))
        candidates.append(same_residue_repeat_rate(int(q)))
        candidates.append(diagonal_vs_offdiagonal_pair_rate(int(q)))
        candidates.append(consecutive_residue_pair_counts(int(q)))
        candidates.append(pair_bias_spectrum(int(q)))
        candidates.append(fourier_residue_measurement(int(q)))
        candidates.append(character_sum_measurement(int(q), 1))
    candidates.append(extra_sieve_residue_measurement(223, 1000, max_primes=12))
    candidates.append(extra_sieve_residue_measurement(1000, 5000, max_primes=8))
    selected: List[Measurement] = []
    spent = 0.0
    for m in sorted(candidates, key=lambda x: (x.cost, x.complexity, x.name)):
        if spent + m.cost <= float(measurement_budget) or not selected:
            selected.append(m)
            spent += m.cost
    return selected


def evaluate_measurements(window: PrimeWindow, measurements: Sequence[Measurement]) -> np.ndarray:
    parts = [m.evaluate(window) for m in measurements]
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def measurement_summary(measurements: Sequence[Measurement]) -> Dict[str, object]:
    return {
        "names": [m.name for m in measurements],
        "formulas": [m.formula for m in measurements],
        "total_cost": float(sum(m.cost for m in measurements)),
        "total_complexity": float(sum(m.complexity for m in measurements)),
    }
