import numpy as np
import pytest
from leadengine.zeros import load_zeros, unfold, ZeroSpacingDataset, \
    PoissonSpacingNull, GUESpacingNull
from leadengine.search import evolutionary_search

F = "tests/fixtures/zeros_10k.txt"

def test_zeros_parse_known_values():
    z = load_zeros(F)
    assert abs(z[0] - 14.134725) < 1e-5
    assert abs(z[1] - 21.022040) < 1e-5
    assert abs(z[2] - 25.010858) < 1e-5

def test_unfolded_mean_spacing_is_one():
    z = load_zeros(F)
    s = unfold(z[1000:])           # skip low zeros where density approx is worst
    assert 0.95 <= s.mean() <= 1.05

@pytest.mark.slow
def test_zeros_vs_poisson_promoted():
    # Known positive: level repulsion. The engine must find it.
    real = ZeroSpacingDataset(F, window_len=64)
    res = evolutionary_search(real, PoissonSpacingNull(), budget=1500, seed=0)
    assert res.best is not None and res.best.ood_auc >= 0.65

@pytest.mark.slow
def test_gue_vs_gue_promotes_nothing():
    # False-positive control: two independent GUE samples are indistinguishable.
    a = GUESpacingNull(seed=1).as_dataset(window_len=64, n_cache=4000)
    res = evolutionary_search(a, GUESpacingNull(seed=2), budget=1500, seed=0)
    assert res.best is None or res.best.promoted is False
