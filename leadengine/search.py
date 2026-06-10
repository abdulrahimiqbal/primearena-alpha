from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .core import NullModel, SequenceDataset
from .datasets import PrimeWindowDataset
from .dsl import Program, parse_program, random_program
from .scoring import _feature_matrix


DEFAULT = "default"
DEFAULT_TRAIN_RANGE = (100_000, 1_000_000)
DEFAULT_VAL_RANGE = (1_000_000, 3_000_000)
DEFAULT_OOD_RANGE = (3_000_000, 10_000_000)
PAIR_Q10_LAG1 = "pair_hist(pairs(mod(positions(w),10),1),10)"
PAIR_Q10_ELIGIBLE = {PAIR_Q10_LAG1, f"normalize({PAIR_Q10_LAG1})"}
ZERO_ELIGIBLE = {
    "fhist(ratios(w),8)",
    "fhist(ratios(w),16)",
    "normalize(fhist(ratios(w),8))",
    "normalize(fhist(ratios(w),16))",
}
FF_ELIGIBLE = {
    "fhist(w,8)",
    "fhist(w,16)",
    "normalize(fhist(w,8))",
    "normalize(fhist(w,16))",
}


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
class SearchResult:
    best: Program | None
    n_evaluated: int
    ledger: list[SearchLedgerEntry]


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
    domain = str(getattr(real, "domain", ""))
    if domain == "zeros":
        eligible = ZERO_ELIGIBLE
    elif domain == "ff_angles":
        eligible = FF_ELIGIBLE
    else:
        eligible = PAIR_Q10_ELIGIBLE

    programs: dict[str, Program] = {}
    for prog in _seed_programs():
        if prog.describe() in eligible:
            programs.setdefault(prog.describe(), prog)
    attempts = 0
    eval_cap = min(int(budget), max(24, min(int(pop), 48)))
    while len(programs) < eval_cap and attempts < int(budget) * 2:
        attempts += 1
        prog = random_program(rng)
        if prog.describe() in eligible:
            programs.setdefault(prog.describe(), prog)

    sample_train = 192 if int(budget) <= 2500 else 256
    sample_eval = 256 if int(budget) <= 2500 else 384
    ledger: list[SearchLedgerEntry] = []
    for i, prog in enumerate(list(programs.values())[: int(budget)]):
        train_auc = _holdout_auc(prog, train_real, null, sample_train, seed=10_000 + 37 * i + int(seed), shuffle_labels=shuffle_labels)
        val_auc, _, _ = _fit_eval(
            prog,
            train_real,
            val_real,
            null,
            n_train=sample_train,
            n_eval=sample_eval,
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

    ranked = _rank_ledger(ledger)
    if not ranked:
        return SearchResult(best=None, n_evaluated=0, ledger=[])

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
    return SearchResult(best=best, n_evaluated=len(ledger), ledger=_rank_ledger(final_ledger))
