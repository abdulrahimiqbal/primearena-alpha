from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from .core import NullModel, SequenceDataset
from .datasets import PrimeWindowDataset
from .dsl import Program, mutate_program, parse_program, program_shape, random_program
from .scoring import _feature_matrix


DEFAULT = "default"
DEFAULT_TRAIN_RANGE = (100_000, 1_000_000)
DEFAULT_VAL_RANGE = (1_000_000, 3_000_000)
DEFAULT_OOD_RANGE = (3_000_000, 10_000_000)
PAIR_Q10_LAG1 = "pair_hist(pairs(mod(positions(w),10),1),10)"


class ProgramSearch(Protocol):
    def search(self, real: SequenceDataset, null: NullModel, budget: int, seed: int) -> "SearchResult": ...


@dataclass(frozen=True)
class SearchLedgerEntry:
    program: Program
    train_auc: float
    val_auc: float
    ood_auc: float
    permutation_p: float | None
    fitness: float
    promoted: bool


@dataclass(frozen=True)
class SearchLog:
    budget: int
    seed: int
    distinct_programs_evaluated: int
    distinct_program_shapes: int
    sampled_fraction: str
    program_space: str
    generation_attempts: int
    programs_per_sec: float = 0.0
    fitness_subsample_windows: int = 0
    finalists_scored: int = 0


@dataclass(frozen=True)
class SearchResult:
    best: Program | None
    n_evaluated: int
    ledger: list[SearchLedgerEntry]
    log: SearchLog


def _range_dataset(real: SequenceDataset, n_min: int, n_max: int) -> SequenceDataset:
    if isinstance(real, PrimeWindowDataset):
        return PrimeWindowDataset(n_min=n_min, n_max=n_max - 1, window_len=real.window_len, name=real.name, domain=real.domain)
    if hasattr(real, "range_dataset"):
        return real.range_dataset(n_min, n_max)
    return real


def _fit_eval(
    program: Program,
    train_real: SequenceDataset,
    eval_real: SequenceDataset,
    null: NullModel,
    n_train: int,
    n_eval: int,
    seed: int,
    shuffle_labels: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_real_windows, train_null_windows = _sample_real_null(train_real, null, n_train, rng)
    x_train = np.vstack([_feature_matrix(program, train_real_windows), _feature_matrix(program, train_null_windows)]).astype(np.float32)
    y_train = np.concatenate([np.ones(n_train, dtype=np.int8), np.zeros(n_train, dtype=np.int8)])
    if shuffle_labels:
        rng.shuffle(y_train)

    eval_real_windows, eval_null_windows = _sample_real_null(eval_real, null, n_eval, rng)
    x_eval = np.vstack([_feature_matrix(program, eval_real_windows), _feature_matrix(program, eval_null_windows)]).astype(np.float32)
    y_eval = np.concatenate([np.ones(n_eval, dtype=np.int8), np.zeros(n_eval, dtype=np.int8)])

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1]
    return float(roc_auc_score(y_eval, scores)), scores.astype(np.float64), y_eval


def _fit_eval_windows(
    program: Program,
    train_real_windows,
    train_null_windows,
    eval_real_windows,
    eval_null_windows,
    seed: int,
    shuffle_labels: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train_real_windows), len(train_null_windows))
    n_eval = min(len(eval_real_windows), len(eval_null_windows))
    x_train = np.vstack([
        _feature_matrix(program, train_real_windows[:n_train]),
        _feature_matrix(program, train_null_windows[:n_train]),
    ]).astype(np.float32)
    y_train = np.concatenate([np.ones(n_train, dtype=np.int8), np.zeros(n_train, dtype=np.int8)])
    if shuffle_labels:
        rng.shuffle(y_train)
    x_eval = np.vstack([
        _feature_matrix(program, eval_real_windows[:n_eval]),
        _feature_matrix(program, eval_null_windows[:n_eval]),
    ]).astype(np.float32)
    y_eval = np.concatenate([np.ones(n_eval, dtype=np.int8), np.zeros(n_eval, dtype=np.int8)])
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1]
    return float(roc_auc_score(y_eval, scores)), scores.astype(np.float64), y_eval


def _holdout_auc(program: Program, real: SequenceDataset, null: NullModel, n: int, seed: int, shuffle_labels: bool = False) -> float:
    rng = np.random.default_rng(seed)
    real_windows, null_windows = _sample_real_null(real, null, n, rng)
    x = np.vstack([_feature_matrix(program, real_windows), _feature_matrix(program, null_windows)]).astype(np.float32)
    y = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    order = rng.permutation(len(y))
    mid = len(order) // 2
    train_idx, test_idx = order[:mid], order[mid:]
    y_train = y[train_idx].copy()
    if shuffle_labels:
        rng.shuffle(y_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(x[train_idx], y_train)
    scores = model.predict_proba(x[test_idx])[:, 1]
    return float(roc_auc_score(y[test_idx], scores))


def _sample_real_null(real: SequenceDataset, null: NullModel, n: int, rng: np.random.Generator):
    if hasattr(null, "sample_pairs"):
        return null.sample_pairs(real, int(n), rng)
    real_windows = real.sample(int(n), rng)
    null_windows = null.sample_like(real, int(n), rng)
    return real_windows, null_windows


def _holdout_auc_windows(program: Program, real_windows, null_windows, seed: int, shuffle_labels: bool = False) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(real_windows), len(null_windows))
    x = np.vstack([_feature_matrix(program, real_windows[:n]), _feature_matrix(program, null_windows[:n])]).astype(np.float32)
    y = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    order = rng.permutation(len(y))
    mid = len(order) // 2
    train_idx, test_idx = order[:mid], order[mid:]
    y_train = y[train_idx].copy()
    if shuffle_labels:
        rng.shuffle(y_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(x[train_idx], y_train)
    scores = model.predict_proba(x[test_idx])[:, 1]
    return float(roc_auc_score(y[test_idx], scores))


def _as_2d(rows: list[np.ndarray], dtype=np.float32) -> np.ndarray:
    width = max((int(np.asarray(r).size) for r in rows), default=0)
    out = np.zeros((len(rows), width), dtype=dtype)
    for i, row in enumerate(rows):
        arr = np.asarray(row, dtype=dtype).reshape(-1)
        out[i, : arr.size] = arr
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _batch_eval(expr, windows, cache: dict[str, object]):
    key = expr.describe()
    if key in cache:
        return cache[key]
    vals = [_batch_eval(arg, windows, cache) for arg in expr.args]
    if expr.op == "w":
        out = windows
    elif expr.op == "positions":
        out = [
            np.asarray(w.meta.get("integer_values"), dtype=np.int64)[np.asarray(w.values) > 0]
            if "integer_values" in w.meta
            else int(w.start) + np.flatnonzero(np.asarray(w.values) > 0).astype(np.int64)
            for w in windows
        ]
    elif expr.op == "gaps":
        out = [np.diff(np.asarray(x, dtype=np.int64)).astype(np.int64) if np.asarray(x).size > 1 else np.asarray([], dtype=np.int64) for x in vals[0]]
    elif expr.op == "mod":
        q = int(expr.params[0])
        out = [(np.asarray(x, dtype=np.int64) % q).astype(np.int64) for x in vals[0]]
    elif expr.op == "pairs":
        lag = int(expr.params[0])
        out = [
            np.stack([seq[:-lag], seq[lag:]], axis=1).astype(np.int64) if (seq := np.asarray(x, dtype=np.int64)).size > lag else np.empty((0, 2), dtype=np.int64)
            for x in vals[0]
        ]
    elif expr.op == "hist":
        q = int(expr.params[0])
        out = _as_2d([
            (lambda counts: counts / max(float(counts.sum()), 1.0))(np.bincount(np.asarray(x, dtype=np.int64) % q, minlength=q).astype(np.float32))
            if np.asarray(x).size else np.zeros(q, dtype=np.float32)
            for x in vals[0]
        ])
    elif expr.op == "pair_hist":
        q = int(expr.params[0])
        rows = []
        for x in vals[0]:
            pairs = np.asarray(x, dtype=np.int64).reshape(-1, 2)
            if pairs.size:
                ids = (pairs[:, 0] % q) * q + (pairs[:, 1] % q)
                counts = np.bincount(ids.astype(np.int64), minlength=q * q).astype(np.float32)
                rows.append(counts / max(float(counts.sum()), 1.0))
            else:
                rows.append(np.zeros(q * q, dtype=np.float32))
        out = _as_2d(rows)
    elif expr.op == "fourier_power":
        k = int(expr.params[0])
        rows = []
        for w in vals[0]:
            indicator = np.asarray(w.values, dtype=np.float64)
            spectrum = np.fft.rfft(indicator) if indicator.size else np.asarray([0.0])
            kk = min(k, spectrum.size - 1)
            rows.append(np.asarray([float(np.abs(spectrum[kk]) ** 2 / max(indicator.size, 1))], dtype=np.float32))
        out = _as_2d(rows)
    elif expr.op == "logweight":
        out = [(1.0 / np.maximum(np.log(np.maximum(np.asarray(x, dtype=np.float64), 3.0)), 1.0)).astype(np.float32) for x in vals[0]]
    elif expr.op == "ratios":
        source = [np.asarray(w.values, dtype=np.float64) for w in vals[0]] if expr.args[0].op == "w" else vals[0]
        out = [
            (np.minimum(seq[:-1], seq[1:]) / np.maximum(np.maximum(seq[:-1], seq[1:]), 1e-12)).astype(np.float32)
            if (seq := np.asarray(x, dtype=np.float64)).size > 1 else np.empty(0, dtype=np.float32)
            for x in source
        ]
    elif expr.op == "fhist":
        bins = int(expr.params[0])
        source = []
        if expr.args[0].op == "w":
            for w in vals[0]:
                seq = np.asarray(w.values, dtype=np.float64)
                if w.meta.get("domain") == "ff_angles":
                    seq = seq / np.pi
                source.append(seq)
        else:
            source = vals[0]
        out = _as_2d([np.histogram(np.asarray(x, dtype=np.float64), bins=bins, range=(0.0, 1.0))[0].astype(np.float32) for x in source])
        sums = np.maximum(out.sum(axis=1, keepdims=True), 1.0)
        out = out / sums
    elif expr.op == "mean":
        out = _as_2d([np.asarray([float(np.mean(np.asarray(x, dtype=np.float64))) if np.asarray(x).size else 0.0]) for x in vals[0]])
    elif expr.op == "var":
        out = _as_2d([np.asarray([float(np.var(np.asarray(x, dtype=np.float64))) if np.asarray(x).size else 0.0]) for x in vals[0]])
    elif expr.op == "normalize":
        vec = np.asarray(vals[0], dtype=np.float32)
        total = np.maximum(np.sum(np.abs(vec), axis=1, keepdims=True), 1e-12)
        out = vec / total
    elif expr.op == "concat":
        out = np.concatenate([np.asarray(v, dtype=np.float32) for v in vals], axis=1)
    elif expr.op == "scalar_vec":
        out = np.asarray(vals[0], dtype=np.float32).reshape(len(windows), -1)
    else:
        out = _feature_matrix(Program(expr), windows).astype(np.float32)
    cache[key] = out
    return out


def _feature_matrix_cached(program: Program, windows, cache: dict[str, object]) -> np.ndarray:
    key = program.describe()
    if key not in cache:
        out = _batch_eval(program.root, windows, cache)
        cache[key] = np.asarray(out, dtype=np.float32).reshape(len(windows), -1)
    return np.asarray(cache[key], dtype=np.float32)


def _screen_auc(
    program: Program,
    real_windows,
    null_windows,
    cache_real: dict[str, object],
    cache_null: dict[str, object],
    seed: int,
    shuffle_labels: bool = False,
) -> float:
    x_real = _feature_matrix_cached(program, real_windows, cache_real)
    x_null = _feature_matrix_cached(program, null_windows, cache_null)
    width = max(x_real.shape[1], x_null.shape[1])
    if width == 0:
        return 0.5
    if x_real.shape[1] != width:
        x_real = np.pad(x_real, ((0, 0), (0, width - x_real.shape[1])))
    if x_null.shape[1] != width:
        x_null = np.pad(x_null, ((0, 0), (0, width - x_null.shape[1])))
    x = np.vstack([x_real, x_null]).astype(np.float32)
    y = np.concatenate([np.ones(len(x_real), dtype=np.int8), np.zeros(len(x_null), dtype=np.int8)])
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y)
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    weights = (pos.mean(axis=0) - neg.mean(axis=0)) / scale
    scores = x @ weights
    if float(np.std(scores)) <= 1e-12:
        return 0.5
    return float(roc_auc_score(y, scores))


def _permutation_p(scores: np.ndarray, labels: np.ndarray, seed: int, rounds: int = 1000) -> float:
    rng = np.random.default_rng(seed)
    observed = float(roc_auc_score(labels, scores))
    ranks = rankdata(scores, method="average").astype(np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(labels.size - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    count = 0
    shuffled = labels.copy()
    for _ in range(max(1000, int(rounds))):
        rng.shuffle(shuffled)
        pos_sum = float(np.sum(ranks[shuffled == 1]))
        auc = (pos_sum - n_pos * (n_pos + 1) / 2.0) / max(float(n_pos * n_neg), 1.0)
        if auc >= observed:
            count += 1
    return float((count + 1) / (max(1000, int(rounds)) + 1))


def _seed_programs() -> list[Program]:
    texts = [
        "pair_hist(pairs(mod(positions(w),10),1),10)",
        "normalize(pair_hist(pairs(mod(positions(w),10),1),10))",
        "concat(hist(mod(positions(w),6),6),pair_hist(pairs(mod(positions(w),10),1),10))",
        "fhist(ratios(w),8)",
        "fhist(ratios(w),16)",
        "normalize(fhist(ratios(w),8))",
        "normalize(fhist(ratios(w),16))",
        "fhist(w,8)",
        "fhist(w,16)",
        "normalize(fhist(w,8))",
        "normalize(fhist(w,16))",
    ]
    for q in (3, 4, 5, 6, 7, 8, 9, 10, 12, 30):
        texts.append(f"hist(mod(positions(w),{q}),{q})")
    for q in (5, 6, 8, 10, 12, 30):
        for lag in (1, 2, 3):
            texts.append(f"pair_hist(pairs(mod(positions(w),{q}),{lag}),{q})")
    for k in (1, 2, 3, 4, 8, 16):
        texts.append(f"scalar_vec(fourier_power(w,{k}))")
    return [parse_program(t) for t in texts]


def _rank_ledger(entries: list[SearchLedgerEntry]) -> list[SearchLedgerEntry]:
    return sorted(entries, key=lambda e: (e.val_auc, -e.program.complexity(), e.train_auc, e.ood_auc), reverse=True)


def _empty_result(budget: int, seed: int, attempts: int = 0) -> SearchResult:
    return SearchResult(
        best=None,
        n_evaluated=0,
        ledger=[],
        log=SearchLog(
            budget=int(budget),
            seed=int(seed),
            distinct_programs_evaluated=0,
            distinct_program_shapes=0,
            sampled_fraction="open",
            program_space="full_dsl_depth<=8_complexity<=20",
            generation_attempts=int(attempts),
        ),
    )


def evolutionary_search(
    real,
    null,
    budget,
    seed,
    primitives=DEFAULT,
    lam=0.01,
    pop=256,
    tournament=4,
    shuffle_labels=False,
    train_range=DEFAULT_TRAIN_RANGE,
    val_range=DEFAULT_VAL_RANGE,
    ood_range=DEFAULT_OOD_RANGE,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    train_real = _range_dataset(real, int(train_range[0]), int(train_range[1]))
    val_real = _range_dataset(real, int(val_range[0]), int(val_range[1]))
    ood_real = _range_dataset(real, int(ood_range[0]), int(ood_range[1]))

    sample_train = 192 if int(budget) <= 2500 else 256
    sample_eval = 256 if int(budget) <= 2500 else 384
    fitness_side = 500
    train_rng = np.random.default_rng(30_001 + int(seed))
    val_rng = np.random.default_rng(30_101 + int(seed))
    train_real_windows, train_null_windows = _sample_real_null(train_real, null, fitness_side, train_rng)
    val_real_windows, val_null_windows = _sample_real_null(val_real, null, fitness_side, val_rng)
    train_cache_real: dict[str, np.ndarray] = {}
    train_cache_null: dict[str, np.ndarray] = {}
    val_cache_real: dict[str, np.ndarray] = {}
    val_cache_null: dict[str, np.ndarray] = {}

    queue: list[Program] = []
    seen: dict[str, Program] = {}
    attempts = 0

    def add_candidate(prog: Program) -> bool:
        desc = prog.describe()
        if desc in seen:
            return False
        seen[desc] = prog
        queue.append(prog)
        return True

    for prog in _seed_programs():
        add_candidate(prog)

    def parent_from_population(entries: list[SearchLedgerEntry]) -> Program | None:
        if not entries:
            return None
        k = min(max(1, int(tournament)), len(entries))
        picks = rng.choice(len(entries), size=k, replace=False)
        return max((entries[int(i)] for i in picks), key=lambda e: e.fitness).program

    def fill_queue(entries: list[SearchLedgerEntry]) -> None:
        nonlocal attempts
        target = max(int(pop), 64)
        while len(queue) < target and attempts < max(1000, int(budget) * 60):
            attempts += 1
            try:
                parent = parent_from_population(entries)
                if parent is not None and rng.random() < 0.65:
                    add_candidate(mutate_program(rng, parent))
                else:
                    add_candidate(random_program(rng))
            except Exception:
                continue

    fill_queue([])
    ledger: list[SearchLedgerEntry] = []
    t0 = time.perf_counter()
    while len(ledger) < int(budget):
        if not queue:
            fill_queue(ledger[-int(pop) :])
            if not queue:
                break
        prog = queue.pop(0)
        i = len(ledger)
        train_auc = _screen_auc(
            prog,
            train_real_windows,
            train_null_windows,
            seed=10_000 + 37 * i + int(seed),
            cache_real=train_cache_real,
            cache_null=train_cache_null,
            shuffle_labels=shuffle_labels,
        )
        val_auc = _screen_auc(
            prog,
            val_real_windows,
            val_null_windows,
            seed=20_000 + 37 * i + int(seed),
            cache_real=val_cache_real,
            cache_null=val_cache_null,
            shuffle_labels=shuffle_labels,
        )
        fitness = float(train_auc - float(lam) * prog.complexity())
        ledger.append(
            SearchLedgerEntry(
                program=prog,
                train_auc=float(train_auc),
                val_auc=float(val_auc),
                ood_auc=0.0,
                permutation_p=None,
                fitness=fitness,
                promoted=False,
            )
        )
        if len(ledger) % max(32, int(pop) // 4) == 0:
            fill_queue(_rank_ledger(ledger)[: int(pop)])
    elapsed = max(time.perf_counter() - t0, 1e-9)

    ranked = _rank_ledger(ledger)
    shapes = {program_shape(entry.program) for entry in ledger}
    if not ranked:
        return _empty_result(budget, seed, attempts)

    finalist_count = min(10, len(ranked))
    finalist_scores: dict[str, tuple[float, float, bool]] = {}
    matched_family = str(getattr(null, "name", "")) in {"sato_tate"}
    for j, entry in enumerate(ranked[:finalist_count]):
        auc, scores, labels = _fit_eval(
            entry.program,
            train_real,
            ood_real,
            null,
            n_train=max(sample_train, 512),
            n_eval=max(sample_eval, 768),
            seed=999_001 + 1009 * j + int(seed),
            shuffle_labels=shuffle_labels,
        )
        p_value = _permutation_p(scores, labels, seed=999_777 + 1009 * j + int(seed), rounds=1000)
        promoted = bool((not shuffle_labels) and not matched_family and auc >= 0.60 and p_value < 0.001)
        finalist_scores[entry.program.describe()] = (float(auc), float(p_value), promoted)

    best_entry = max(
        ranked[:finalist_count],
        key=lambda e: (finalist_scores[e.program.describe()][0], -finalist_scores[e.program.describe()][1], e.val_auc),
    )
    best = best_entry.program
    ood_auc, p_value, promoted = finalist_scores[best.describe()]
    best.ood_auc = float(ood_auc)
    best.promoted = bool(promoted)
    best.meta.update({"permutation_p": p_value, "val_auc": best_entry.val_auc, "train_auc": best_entry.train_auc})

    log = SearchLog(
        budget=int(budget),
        seed=int(seed),
        distinct_programs_evaluated=len(ledger),
        distinct_program_shapes=len(shapes),
        sampled_fraction="open",
        program_space="full_dsl_depth<=8_complexity<=20",
        generation_attempts=int(attempts),
        programs_per_sec=float(len(ledger) / elapsed),
        fitness_subsample_windows=int(2 * fitness_side),
        finalists_scored=int(finalist_count),
    )

    final_ledger: list[SearchLedgerEntry] = []
    for entry in ledger:
        desc = entry.program.describe()
        if desc in finalist_scores:
            auc, p, pro = finalist_scores[desc]
            prog = best if desc == best.describe() else entry.program
            prog.ood_auc = float(auc)
            prog.promoted = bool(pro)
            final_ledger.append(
                SearchLedgerEntry(
                    program=prog,
                    train_auc=entry.train_auc,
                    val_auc=entry.val_auc,
                    ood_auc=float(auc),
                    permutation_p=float(p),
                    fitness=entry.fitness,
                    promoted=bool(pro),
                )
            )
        else:
            final_ledger.append(entry)
    return SearchResult(best=best, n_evaluated=len(ledger), ledger=_rank_ledger(final_ledger), log=log)
