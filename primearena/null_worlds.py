from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .oracle import small_prime_sieve


@dataclass
class PrimeWindow:
    n: int
    offsets: np.ndarray
    values: np.ndarray
    indicator: np.ndarray
    null_model_name: str
    null_model_params: Dict[str, Any]
    residue_metadata: Dict[str, Any]
    gap_metadata: Dict[str, Any]
    is_real: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["offsets"] = self.offsets.astype(int).tolist()
        out["values"] = self.values.astype(int).tolist()
        out["indicator"] = self.indicator.astype(int).tolist()
        return out


def _segment_prime_indicator(n: int, window_size: int) -> np.ndarray:
    start = int(n) + 1
    end = start + int(window_size) - 1
    values = np.arange(start, end + 1, dtype=np.int64)
    is_prime = np.ones(values.shape[0], dtype=bool)
    is_prime[values < 2] = False
    primes = small_prime_sieve(int(math.isqrt(max(end, 2))) + 1)
    for p in primes:
        first = max(p * p, ((start + p - 1) // p) * p)
        if first <= end:
            is_prime[first - start :: p] = False
    return is_prime.astype(np.int8)


def _gap_metadata(values: np.ndarray, indicator: np.ndarray) -> Dict[str, Any]:
    positions = values[indicator > 0]
    if positions.size <= 1:
        gaps = np.asarray([], dtype=np.int64)
    else:
        gaps = np.diff(positions).astype(np.int64)
    return {
        "count": int(positions.size),
        "gaps": gaps.astype(int).tolist(),
        "mean_gap": float(np.mean(gaps)) if gaps.size else 0.0,
        "max_gap": int(np.max(gaps)) if gaps.size else 0,
    }


def _residue_metadata(values: np.ndarray, indicator: np.ndarray, q_values: Iterable[int]) -> Dict[str, Any]:
    selected = values[indicator > 0]
    out: Dict[str, Any] = {}
    for q in q_values:
        q = int(q)
        counts = np.bincount((selected % q).astype(np.int64), minlength=q) if selected.size else np.zeros(q, dtype=np.int64)
        out[f"mod_{q}"] = counts.astype(int).tolist()
    return out


def make_real_window(n: int, window_size: int, q_values: Iterable[int] = (6, 30, 210)) -> PrimeWindow:
    offsets = np.arange(1, int(window_size) + 1, dtype=np.int64)
    values = int(n) + offsets
    indicator = _segment_prime_indicator(int(n), int(window_size))
    return PrimeWindow(
        n=int(n),
        offsets=offsets,
        values=values,
        indicator=indicator,
        null_model_name="real",
        null_model_params={},
        residue_metadata=_residue_metadata(values, indicator, q_values),
        gap_metadata=_gap_metadata(values, indicator),
        is_real=True,
    )


def _wheel_mask(values: np.ndarray, sieve_bound: int) -> np.ndarray:
    mask = np.ones(values.shape[0], dtype=bool)
    for p in small_prime_sieve(int(sieve_bound)):
        p = int(p)
        mask &= (values == p) | ((values % p) != 0)
    return mask


def _mod_coprime_mask(values: np.ndarray, q: int) -> np.ndarray:
    q = max(2, int(q))
    return np.asarray([math.gcd(int(x) % q, q) == 1 for x in values], dtype=bool)


def _calibration_allowed_mask(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    if params.get("wheel_mod_q", True):
        return _mod_coprime_mask(values, int(params.get("q", 30)))
    return _wheel_mask(values, int(params.get("sieve_bound", 30)))


def _ensure_nonempty(indicator: np.ndarray, rng: np.random.Generator, allowed: Optional[np.ndarray] = None) -> np.ndarray:
    if indicator.sum() > 0:
        return indicator.astype(np.int8)
    mask = np.ones(indicator.shape[0], dtype=bool) if allowed is None else allowed.astype(bool)
    choices = np.flatnonzero(mask)
    if choices.size:
        indicator[int(rng.choice(choices))] = 1
    return indicator.astype(np.int8)


def _indicator_from_positions(size: int, positions: List[int]) -> np.ndarray:
    out = np.zeros(int(size), dtype=np.int8)
    for pos in positions:
        if 0 <= int(pos) < size:
            out[int(pos)] = 1
    return out


def cramer_density_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    values = real.values
    probs = 1.0 / np.maximum(np.log(np.maximum(values.astype(np.float64), 3.0)), 1.0)
    indicator = (rng.random(values.shape[0]) < probs).astype(np.int8)
    indicator = _ensure_nonempty(indicator, rng)
    return _fake_window(real, indicator, "cramer", params)


def wheel_sieved_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    bound = int(params.get("sieve_bound", 30))
    mask = _wheel_mask(real.values, bound)
    base = 1.0 / np.maximum(np.log(np.maximum(real.values.astype(np.float64), 3.0)), 1.0)
    density = max(float(mask.mean()), 1e-6)
    probs = np.where(mask, np.minimum(base / density, 0.95), 0.0)
    indicator = (rng.random(real.values.shape[0]) < probs).astype(np.int8)
    indicator = _ensure_nonempty(indicator, rng, allowed=mask)
    return _fake_window(real, indicator, "wheel", params)


def wheel_iid_no_pair_bias_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    """Match local density and wheel admissibility, but sample positions iid.

    This is the known-positive calibration null: it preserves obvious modular
    obstruction and coarse density, while deliberately omitting consecutive
    prime residue-pair bias.
    """

    block_count = max(1, int(params.get("density_blocks", 8)))
    indicator = np.zeros_like(real.indicator, dtype=np.int8)
    allowed = _calibration_allowed_mask(real.values, params)
    chunks = np.array_split(np.arange(real.indicator.size), block_count)
    for chunk in chunks:
        target = int(real.indicator[chunk].sum())
        choices = [int(i) for i in chunk if allowed[int(i)]]
        if target > 0 and choices:
            picked = rng.choice(choices, size=min(target, len(choices)), replace=False)
            indicator[picked] = 1
    return _fake_window(real, _ensure_nonempty(indicator, rng, allowed=allowed), "wheel_iid_no_pair_bias", params)


def gap_histogram_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    gap_pool = [int(x) for x in params.get("gap_pool", []) if int(x) > 0]
    if not gap_pool:
        return shuffled_real_windows_null(real, rng, params)
    target_count = max(1, int(real.indicator.sum()))
    start_choices = np.flatnonzero(real.indicator > 0)
    pos = int(rng.choice(start_choices)) if start_choices.size else int(rng.integers(0, max(1, real.indicator.size)))
    positions = [pos]
    while len(positions) < target_count:
        pos += int(rng.choice(gap_pool))
        if pos >= real.indicator.size:
            break
        positions.append(pos)
    return _fake_window(real, _indicator_from_positions(real.indicator.size, positions), "gap_hist", params)


def residue_pair_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    q = int(params.get("q", 30))
    pair_counts = params.get("residue_pair_counts_by_q", {}).get(str(q), params.get("residue_pair_counts", {}))
    target_count = max(1, int(real.indicator.sum()))
    mask = _wheel_mask(real.values, int(params.get("sieve_bound", 30)))
    available_by_residue: Dict[int, List[int]] = {}
    for idx in np.flatnonzero(mask):
        available_by_residue.setdefault(int(real.values[idx] % q), []).append(int(idx))
    residues = sorted(available_by_residue)
    if not residues:
        return wheel_sieved_null(real, rng, params)
    current = int(rng.choice(residues))
    positions: List[int] = []
    for _ in range(target_count):
        choices = [idx for idx in available_by_residue.get(current, []) if not positions or idx > positions[-1]]
        if not choices:
            choices = [idx for idx in np.flatnonzero(mask) if not positions or idx > positions[-1]]
        if not choices:
            break
        positions.append(int(rng.choice(choices[: max(1, min(len(choices), 16))])))
        next_weights = np.asarray([float(pair_counts.get(f"{current},{r}", 1.0)) for r in residues], dtype=np.float64)
        next_weights /= max(float(next_weights.sum()), 1e-12)
        current = int(rng.choice(residues, p=next_weights))
    return _fake_window(real, _indicator_from_positions(real.indicator.size, positions), "residue_pair", params)


def residue_pair_matched_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    """Match local density, wheel admissibility, and empirical pair transitions."""

    q = int(params.get("q", 30))
    allowed = _calibration_allowed_mask(real.values, params)
    if params.get("exact_window_pair_match", True):
        block_count = max(1, int(params.get("density_blocks", 8)))
        chunks = np.array_split(np.arange(real.indicator.size), block_count)
        block_by_idx: Dict[int, set[int]] = {int(i): set(int(x) for x in chunk) for i, chunk in enumerate(chunks)}
        idx_to_block: Dict[int, int] = {}
        for b, chunk in block_by_idx.items():
            for idx in chunk:
                idx_to_block[int(idx)] = b
        positions: List[int] = []
        used: set[int] = set()
        for real_idx in np.flatnonzero(real.indicator > 0):
            real_idx = int(real_idx)
            residue = int(real.values[real_idx] % q)
            block = idx_to_block.get(real_idx, 0)
            block_set = block_by_idx.get(block, set())
            last = positions[-1] if positions else -1
            candidates = [
                int(i)
                for i in sorted(block_set)
                if allowed[int(i)] and int(i) not in used and int(i) > last and int(real.values[int(i)] % q) == residue
            ]
            nontrivial = [i for i in candidates if i != real_idx]
            if nontrivial:
                candidates = sorted(nontrivial, key=lambda i: abs(i - real_idx))
            if candidates:
                chosen = int(rng.choice(candidates[: max(1, min(4, len(candidates)))]))
            elif real_idx not in used and real_idx > last:
                chosen = real_idx
            else:
                fallback = [
                    int(i)
                    for i in np.flatnonzero(allowed)
                    if int(i) not in used and int(i) > last and int(real.values[int(i)] % q) == residue
                ]
                if not fallback:
                    break
                chosen = int(fallback[0])
            positions.append(chosen)
            used.add(chosen)
        return _fake_window(
            real,
            _ensure_nonempty(_indicator_from_positions(real.indicator.size, positions), rng, allowed=allowed),
            "residue_pair_matched_null",
            params,
        )

    pair_counts = params.get("residue_pair_counts_by_q", {}).get(str(q), params.get("residue_pair_counts", {}))
    residues = sorted({int(x % q) for x in real.values[allowed]})
    if not residues:
        return wheel_iid_no_pair_bias_null(real, rng, params)
    all_allowed = [int(i) for i in np.flatnonzero(allowed)]
    block_count = max(1, int(params.get("density_blocks", 8)))
    chunks = np.array_split(np.arange(real.indicator.size), block_count)
    positions: List[int] = []
    current = int(rng.choice(residues))
    for chunk in chunks:
        target = int(real.indicator[chunk].sum())
        if target <= 0:
            continue
        chunk_set = set(int(i) for i in chunk)
        block_positions: List[int] = []
        chunk_allowed = [i for i in all_allowed if i in chunk_set]
        for _ in range(target):
            next_weights = np.asarray([float(pair_counts.get(f"{current},{r}", 0.1)) for r in residues], dtype=np.float64)
            next_weights /= max(float(next_weights.sum()), 1e-12)
            proposed = int(rng.choice(residues, p=next_weights))
            remaining_after = target - len(block_positions) - 1

            def feasible(i: int) -> bool:
                later = [j for j in chunk_allowed if j not in block_positions and j > i]
                return len(later) >= remaining_after

            last = block_positions[-1] if block_positions else (positions[-1] if positions else -1)
            candidates = [
                int(i)
                for i in chunk_allowed
                if i not in block_positions and i > last and int(real.values[i] % q) == proposed and feasible(int(i))
            ]
            if not candidates:
                candidates = [
                    int(i)
                    for i in chunk_allowed
                    if i not in block_positions and i > last and feasible(int(i))
                ]
            if not candidates:
                break
            # Prefer an early feasible position so the transition chain can keep
            # placing later points in the same local-density block.
            chosen = int(rng.choice(candidates[: max(1, min(8, len(candidates)))]))
            block_positions.append(chosen)
            current = int(real.values[chosen] % q)
        if len(block_positions) < target:
            remaining = [i for i in chunk_allowed if i not in block_positions]
            for i in remaining[: max(0, target - len(block_positions))]:
                block_positions.append(int(i))
        block_positions = sorted(block_positions[:target])
        positions.extend(block_positions)
        if block_positions:
            current = int(real.values[block_positions[-1]] % q)
    return _fake_window(real, _ensure_nonempty(_indicator_from_positions(real.indicator.size, positions), rng, allowed=allowed), "residue_pair_matched_null", params)


def ktuple_local_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    block_count = int(params.get("density_blocks", 8))
    block_count = max(1, block_count)
    indicator = np.zeros_like(real.indicator, dtype=np.int8)
    chunks = np.array_split(np.arange(real.indicator.size), block_count)
    allowed = _wheel_mask(real.values, int(params.get("sieve_bound", 30)))
    for chunk in chunks:
        count = int(real.indicator[chunk].sum())
        choices = [int(i) for i in chunk if allowed[int(i)]]
        if choices and count > 0:
            picked = rng.choice(choices, size=min(count, len(choices)), replace=False)
            indicator[picked] = 1
    return _fake_window(real, _ensure_nonempty(indicator, rng, allowed=allowed), "ktuple_local", params)


def shuffled_real_windows_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    indicator = real.indicator.copy()
    rng.shuffle(indicator)
    return _fake_window(real, _ensure_nonempty(indicator, rng), "shuffled_real", params)


def block_bootstrap_null(real: PrimeWindow, rng: np.random.Generator, params: Dict[str, Any]) -> PrimeWindow:
    blocks = params.get("indicator_blocks", [])
    block_size = int(params.get("block_size", 32))
    out: List[int] = []
    while len(out) < real.indicator.size:
        if blocks:
            block = list(blocks[int(rng.integers(0, len(blocks)))])
        else:
            start = int(rng.integers(0, max(1, real.indicator.size - block_size + 1)))
            block = real.indicator[start : start + block_size].astype(int).tolist()
        out.extend(block)
    indicator = np.asarray(out[: real.indicator.size], dtype=np.int8)
    return _fake_window(real, _ensure_nonempty(indicator, rng), "block_bootstrap", params)


def _fake_window(real: PrimeWindow, indicator: np.ndarray, name: str, params: Dict[str, Any]) -> PrimeWindow:
    q_values = params.get("q_values", [6, 30, 210])
    return PrimeWindow(
        n=real.n,
        offsets=real.offsets.copy(),
        values=real.values.copy(),
        indicator=indicator.astype(np.int8),
        null_model_name=name,
        null_model_params=dict(params),
        residue_metadata=_residue_metadata(real.values, indicator, q_values),
        gap_metadata=_gap_metadata(real.values, indicator),
        is_real=False,
    )


NULL_GENERATORS = {
    "cramer": cramer_density_null,
    "cramer_density": cramer_density_null,
    "wheel": wheel_sieved_null,
    "wheel_sieved": wheel_sieved_null,
    "wheel_iid_no_pair_bias": wheel_iid_no_pair_bias_null,
    "gap_hist": gap_histogram_null,
    "gap_histogram": gap_histogram_null,
    "residue_pair": residue_pair_null,
    "residue_pair_matched": residue_pair_matched_null,
    "residue_pair_matched_null": residue_pair_matched_null,
    "ktuple_local": ktuple_local_null,
    "shuffled_real": shuffled_real_windows_null,
    "shuffled_real_windows": shuffled_real_windows_null,
    "block_bootstrap": block_bootstrap_null,
}


def build_calibration(
    n_min: int,
    n_max: int,
    window_size: int,
    samples: int,
    seed: int,
    q_values: Iterable[int],
    block_size: int = 32,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    windows = [make_real_window(int(rng.integers(n_min, n_max + 1)), window_size, q_values=q_values) for _ in range(max(1, int(samples)))]
    gap_pool: List[int] = []
    pair_counts: Dict[str, int] = {}
    pair_counts_by_q: Dict[str, Dict[str, int]] = {}
    indicator_blocks: List[List[int]] = []
    for w in windows:
        gap_pool.extend([int(x) for x in w.gap_metadata.get("gaps", []) if int(x) > 0])
        for q in q_values:
            selected = w.values[w.indicator > 0]
            residues = (selected % int(q)).astype(int)
            q_counts = pair_counts_by_q.setdefault(str(int(q)), {})
            for a, b in zip(residues[:-1], residues[1:]):
                key = f"{int(a)},{int(b)}"
                pair_counts[key] = pair_counts.get(key, 0) + 1
                q_counts[key] = q_counts.get(key, 0) + 1
        for start in range(0, w.indicator.size, block_size):
            block = w.indicator[start : start + block_size].astype(int).tolist()
            if len(block) == block_size:
                indicator_blocks.append(block)
    return {
        "gap_pool": gap_pool or [2, 4, 6, 10, 12],
        "residue_pair_counts": pair_counts,
        "residue_pair_counts_by_q": pair_counts_by_q,
        "indicator_blocks": indicator_blocks,
        "block_size": int(block_size),
    }


def generate_real_fake_pair(
    null_name: str,
    n: int,
    window_size: int,
    rng: np.random.Generator,
    params: Optional[Dict[str, Any]] = None,
) -> tuple[PrimeWindow, PrimeWindow]:
    params = dict(params or {})
    q_values = params.get("q_values", [6, 30, 210])
    real = make_real_window(int(n), int(window_size), q_values=q_values)
    generator = NULL_GENERATORS.get(null_name)
    if generator is None:
        raise ValueError(f"Unknown PrimeLead null model: {null_name}")
    fake = generator(real, rng, params)
    return real, fake
