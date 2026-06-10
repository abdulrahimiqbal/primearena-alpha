import numpy as np
import pytest
from sympy import isprime
from leadengine.datasets import PrimeWindowDataset
from leadengine.stats import ResiduePairCount
from leadengine.nulls import WheelNull
from leadengine.scoring import auc_real_vs_null

def test_prime_dataset_deterministic():
    ds = PrimeWindowDataset(n_min=100_000, n_max=1_000_000, window_len=512)
    a = ds.sample(50, np.random.default_rng(0))
    b = ds.sample(50, np.random.default_rng(0))
    assert all(x.start == y.start and (x.values == y.values).all()
               for x, y in zip(a, b))

def test_prime_dataset_labels_correct():
    ds = PrimeWindowDataset(n_min=100_000, n_max=200_000, window_len=64)
    for w in ds.sample(5, np.random.default_rng(1)):
        expected = np.fromiter(
            (isprime(n) for n in range(w.start, w.start + 64)), dtype=np.int8)
        assert (w.values.astype(np.int8) == expected).all()

def test_scale_of():
    ds = PrimeWindowDataset(n_min=100_000, n_max=10_000_000, window_len=64)
    w = ds.sample(1, np.random.default_rng(2))[0]
    assert ds.scale_of(w) == int(np.log10(w.start))

@pytest.mark.slow
@pytest.mark.master_calibration
def test_calibration_through_new_interfaces():
    rng = np.random.default_rng(0)
    real = PrimeWindowDataset(n_min=100_000, n_max=10_000_000, window_len=512)
    null = WheelNull(wheel=30)
    auc = auc_real_vs_null(ResiduePairCount(q=10), real, null, n=4000, rng=rng)
    assert auc >= 0.62
