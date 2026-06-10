import numpy as np
import pytest
from leadengine.datasets import PrimeWindowDataset
from leadengine.stats import ResiduePairCount
from leadengine.nulls import WheelNull
from leadengine.scoring import auc_real_vs_null
from leadengine.absorb import AbsorptionError
from leadengine.loop import absorption_loop

def _setup():
    rng = np.random.default_rng(0)
    real = PrimeWindowDataset(n_min=100_000, n_max=10_000_000, window_len=512)
    return rng, real, WheelNull(wheel=30)

@pytest.mark.slow
@pytest.mark.master_calibration
def test_absorb_kills_pair_lead():
    rng, real, null = _setup()
    pair = ResiduePairCount(q=10)
    auc0 = auc_real_vs_null(pair, real, null, n=4000, rng=rng)
    assert auc0 >= 0.62                      # lead exists pre-absorption
    null2 = null.absorb(pair, real)
    auc1 = auc_real_vs_null(pair, real, null2, n=4000, rng=rng)
    assert auc1 <= 0.58                      # absorption kills the lead

@pytest.mark.slow
def test_absorb_preserves_prior_structure():
    # Absorbing the pair statistic must not break the null's gap-histogram agreement.
    rng, real, null = _setup()
    def gap_js(nl):
        # TODO(agent): Jensen-Shannon divergence between pooled prime-gap histograms
        # (bins 2..60, even gaps) of 2000 real vs 2000 null windows. Deterministic rng.
        local_rng = np.random.default_rng(12345)
        real_windows = real.sample(2000, local_rng)
        null_windows = nl.sample_like(real, 2000, local_rng)

        def pooled_hist(windows):
            bins = np.arange(2, 62, 2)
            counts = np.zeros(len(bins), dtype=np.float64)
            for w in windows:
                positions = np.flatnonzero(w.values > 0) + int(w.start)
                if positions.size <= 1:
                    continue
                gaps = np.diff(positions)
                for gap in gaps:
                    if 2 <= int(gap) <= 60 and int(gap) % 2 == 0:
                        counts[(int(gap) - 2) // 2] += 1.0
            counts += 1e-12
            return counts / counts.sum()

        p = pooled_hist(real_windows)
        q = pooled_hist(null_windows)
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    before = gap_js(null)
    after = gap_js(null.absorb(ResiduePairCount(q=10), real))
    assert after <= before + 0.02

@pytest.mark.slow
def test_loop_no_false_positive_on_null_data():
    # When "real" is itself drawn from the null, the loop must promote nothing in round 1.
    rng, real, null = _setup()
    fake_real = null.as_dataset(real, n_cache=6000, rng=np.random.default_rng(7))
    # TODO(agent): implement NullModel.as_dataset() returning a SequenceDataset view of
    # cached null samples.
    ledger = absorption_loop(fake_real, null, [ResiduePairCount(q=10)],
                             max_rounds=3,
                             promote_fn=lambda auc: auc >= 0.60)
    assert ledger.rounds[0].promoted == []
    assert len(ledger.rounds) == 1

def test_ess_guard():
    # Absorbing a near-deterministic statistic must trip the ESS guard, not return garbage.
    rng, real, null = _setup()
    class WindowStart:
        name = "window_start"
        def __call__(self, w): return np.array([float(w.start)])
        def complexity(self): return 1.0
        def describe(self): return "start(w)"
    with pytest.raises(AbsorptionError):
        null.absorb(WindowStart(), real).sample_like(real, 500, rng)
