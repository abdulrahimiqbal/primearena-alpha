from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
    train_real_windows = train_real.sample(n_train, rng)
    train_null_windows = null.sample_like(train_real, n_train, rng)
    x_train = np.vstack([_feature_matrix(program, train_real_windows), _feature_matrix(program, train_null_windows)]).astype(np.float32)
    y_train = np.concatenate([np.ones(n_train, dtype=np.int8), np.zeros(n_train, dtype=np.int8)])
    if shuffle_labels:
        rng.shuffle(y_train)

    eval_real_windows = eval_real.sample(n_eval, rng)
    eval_null_windows = null.sample_like(eval_real, n_eval, rng)
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
    real_windows = real.sample(n, rng)
    null_windows = null.sample_like(real, n, rng)
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


def _permutation_p(scores: np.ndarray, labels: np.ndarray, seed: int, rounds: int = 1000) -> float:
    rng = np.random.default_rng(seed)
    observed = float(roc_auc_score(labels, scores))
    count = 0
    shuffled = labels.copy()
    for _ in range(max(1000, int(rounds))):
        rng.shuffle(shuffled)
        if float(roc_auc_score(shuffled, scores)) >= observed:
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
    train_rng = np.random.default_rng(30_001 + int(seed))
    val_rng = np.random.default_rng(30_101 + int(seed))
    train_real_windows = train_real.sample(sample_train, train_rng)
    train_null_windows = null.sample_like(train_real, sample_train, train_rng)
    val_real_windows = val_real.sample(sample_eval, val_rng)
    val_null_windows = null.sample_like(val_real, sample_eval, val_rng)

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
    while len(ledger) < int(budget):
        if not queue:
            fill_queue(ledger[-int(pop) :])
            if not queue:
                break
        prog = queue.pop(0)
        i = len(ledger)
        train_auc = _holdout_auc_windows(
            prog,
            train_real_windows,
            train_null_windows,
            seed=10_000 + 37 * i + int(seed),
            shuffle_labels=shuffle_labels,
        )
        val_auc, _, _ = _fit_eval_windows(
            prog,
            train_real_windows,
            train_null_windows,
            val_real_windows,
            val_null_windows,
            seed=20_000 + 37 * i + int(seed),
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

    ranked = _rank_ledger(ledger)
    shapes = {program_shape(entry.program) for entry in ledger}
    log = SearchLog(
        budget=int(budget),
        seed=int(seed),
        distinct_programs_evaluated=len(ledger),
        distinct_program_shapes=len(shapes),
        sampled_fraction="open",
        program_space="full_dsl_depth<=8_complexity<=20",
        generation_attempts=int(attempts),
    )
    if not ranked:
        return _empty_result(budget, seed, attempts)

    best_entry = ranked[0]
    best = best_entry.program
    ood_auc, scores, labels = _fit_eval(
        best,
        train_real,
        ood_real,
        null,
        n_train=max(sample_train, 512),
        n_eval=max(sample_eval, 768),
        seed=999_001 + int(seed),
        shuffle_labels=shuffle_labels,
    )
    p_value = _permutation_p(scores, labels, seed=999_777 + int(seed), rounds=1000)
    matched_family = str(getattr(null, "name", "")) in {"sato_tate"}
    promoted = bool((not shuffle_labels) and not matched_family and ood_auc >= 0.60 and p_value < 0.001)
    best.ood_auc = float(ood_auc)
    best.promoted = promoted
    best.meta.update({"permutation_p": p_value, "val_auc": best_entry.val_auc, "train_auc": best_entry.train_auc})

    final_ledger: list[SearchLedgerEntry] = []
    for entry in ledger:
        if entry.program.describe() == best.describe():
            final_ledger.append(
                SearchLedgerEntry(
                    program=best,
                    train_auc=entry.train_auc,
                    val_auc=entry.val_auc,
                    ood_auc=float(ood_auc),
                    permutation_p=float(p_value),
                    fitness=entry.fitness,
                    promoted=promoted,
                )
            )
        else:
            final_ledger.append(entry)
    return SearchResult(best=best, n_evaluated=len(ledger), ledger=_rank_ledger(final_ledger), log=log)
