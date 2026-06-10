import numpy as np
import pytest
from leadengine.datasets import PrimeWindowDataset
from leadengine.stats import ResiduePairCount
from leadengine.nulls import WheelNull
from leadengine.scoring import auc_real_vs_null
from leadengine.dsl import parse_program
from leadengine.search import evolutionary_search

def _setup():
    rng = np.random.default_rng(0)
    real = PrimeWindowDataset(n_min=100_000, n_max=10_000_000, window_len=512)
    return rng, real, WheelNull(wheel=30)

def test_dsl_expresses_pair_statistic():
    # Expressivity check: separates DSL bugs from search failures.
    rng, real, null = _setup()
    prog = parse_program("pair_hist(pairs(mod(positions(w),10),1),10)")
    assert auc_real_vs_null(prog, real, null, n=4000, rng=rng) >= 0.62

def test_roundtrip_parse_describe():
    s = "concat(hist(mod(positions(w),6),6),pair_hist(pairs(mod(positions(w),10),1),10))"
    assert parse_program(s).describe() == s

@pytest.mark.slow
@pytest.mark.master_calibration
def test_search_rediscovers_pair_bias():
    # THE PHASE GATE. pair-counting is reachable only by composing primitives.
    rng, real, null = _setup()
    res = evolutionary_search(real, null, budget=5000, seed=0)
    best = res.best
    assert best.ood_auc >= 0.60
    absorbed = null.absorb(best, real)
    # The discovered program must capture the SAME structure: absorbing it must also
    # explain away the hand-built pair statistic.
    assert auc_real_vs_null(ResiduePairCount(q=10), real, absorbed,
                            n=4000, rng=rng) <= 0.64

@pytest.mark.slow
def test_shuffled_labels_promote_nothing():
    # False-positive control under the same search budget.
    rng, real, null = _setup()
    res = evolutionary_search(real, null, budget=2000, seed=1, shuffle_labels=True)
    assert res.best is None or res.best.promoted is False

def test_complexity_tiebreak():
    # Equal-AUC programs must rank by lower complexity. TODO(agent): construct two
    # semantically identical programs of different complexity (e.g., with/without a
    # redundant normalize) and assert ranking order in the search ledger.
    from leadengine.search import SearchLedgerEntry, _rank_ledger

    simple = parse_program("hist(mod(positions(w),6),6)")
    redundant = parse_program("normalize(hist(mod(positions(w),6),6))")
    assert simple.complexity() < redundant.complexity()
    ledger = _rank_ledger([
        SearchLedgerEntry(program=redundant, train_auc=0.7, val_auc=0.7, ood_auc=0.7, permutation_p=None, fitness=0.0, promoted=False),
        SearchLedgerEntry(program=simple, train_auc=0.7, val_auc=0.7, ood_auc=0.7, permutation_p=None, fitness=0.0, promoted=False),
    ])
    assert ledger[0].program == simple

def test_search_split_defaults_unchanged():
    from leadengine.search import DEFAULT_TRAIN_RANGE, DEFAULT_VAL_RANGE, DEFAULT_OOD_RANGE

    assert DEFAULT_TRAIN_RANGE == (100_000, 1_000_000)
    assert DEFAULT_VAL_RANGE == (1_000_000, 3_000_000)
    assert DEFAULT_OOD_RANGE == (3_000_000, 10_000_000)
