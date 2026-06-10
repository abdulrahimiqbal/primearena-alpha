import math

import numpy as np
from scipy.stats import ks_2samp

from leadengine.absorb import GeneratorConstraint
from leadengine.datasets import PrimeWindowDataset, WindowBuilder
from leadengine.nulls import SkeletonResampleNull, WheelNull


def _positions(w):
    vals = np.asarray(w.meta["integer_values"], dtype=np.int64)
    return vals[np.asarray(w.values) > 0]


def test_shared_window_builder_for_real_and_rebuilt_null():
    real = PrimeWindowDataset(100_000, 200_000, 256)
    null = SkeletonResampleNull(30)
    assert isinstance(real.builder, WindowBuilder)
    assert isinstance(null.builder, WindowBuilder)
    assert WheelNull(30).superseded is True
    rw = real.sample(1, np.random.default_rng(1))[0]
    nw = null.sample_like(real, 1, np.random.default_rng(1))[0]
    assert rw.meta["window_builder"] == "WindowBuilder"
    assert nw.meta["window_builder"] == "WindowBuilder"


def test_skeleton_support_contract_wheel30():
    rng = np.random.default_rng(2)
    real = PrimeWindowDataset(100_000, 1_000_000, 512)
    null = SkeletonResampleNull(30)
    rw = real.sample(500, rng)
    nw = null.sample_like(real, 500, rng)
    counts_real = np.asarray([int(w.values.sum()) for w in rw])
    counts_null = np.asarray([int(w.values.sum()) for w in nw])
    assert ks_2samp(counts_real, counts_null).pvalue > 0.01
    for w in nw:
        pos = _positions(w)
        assert np.all(pos >= int(w.start))
        assert np.all(pos < int(w.start) + len(w.values))
        assert all(math.gcd(int(x), 30) == 1 for x in pos)


def test_generator_constraint_restricts_sampler_residues():
    real = PrimeWindowDataset(100_000, 200_000, 256)
    constrained = SkeletonResampleNull(30).absorb(GeneratorConstraint(wheel=30, allowed_residues=(1, 7)), real)
    windows = constrained.sample_like(real, 100, np.random.default_rng(3))
    for w in windows:
        residues = set((_positions(w) % 30).astype(int).tolist())
        assert residues <= {1, 7}
