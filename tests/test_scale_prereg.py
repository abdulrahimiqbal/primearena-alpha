import numpy as np
import pytest
from sympy import primerange
from leadengine.sieve import primes_in
from leadengine.scale import fit_templates, explained_by
from leadengine.prereg import register, score_extrapolation, PreregError

def test_sieve_matches_sympy_at_1e9():
    lo, hi = 10**9, 10**9 + 10**5
    assert set(primes_in(lo, hi).tolist()) == set(primerange(lo, hi))

@pytest.mark.slow
def test_sieve_matches_sympy_at_1e11():
    lo, hi = 10**11, 10**11 + 10**5
    assert set(primes_in(lo, hi).tolist()) == set(primerange(lo, hi))

def test_explain_away_catches_log_decay():
    decades = [5, 6, 7, 8, 9]
    mids = [10**(d + 0.5) for d in decades]
    profile = {d: (0.8 / np.log(m), 0.002) for d, m in zip(decades, mids)}
    assert explained_by(profile) == "1/log n"

def test_explain_away_passes_stable_effect():
    profile = {d: (0.05, 0.002) for d in [5, 6, 7, 8, 9]}
    assert explained_by(profile) in (None, "1")  # constant is not a decay explanation
    # TODO(agent): if your template lib treats "1" as explanatory, the constant template
    # must be tagged "stable", never "explained". Encode that and assert
    # explained_by(profile) is None.
    assert explained_by(profile) is None

def test_prereg_required_and_seed_derived():
    with pytest.raises(PreregError):
        score_extrapolation("lead_x", target_decade=8)
    h = register("lead_x", fit_decades=[5, 6, 7], predicted_effect=0.05,
                 ci_low=0.04, ci_high=0.06, out_dir="prereg")
    seed = score_extrapolation.derive_seed("lead_x")
    assert seed == int(h[:8], 16)

def test_extrapolation_pass_and_fail():
    # TODO(agent): build two synthetic leads on synthetic datasets you control:
    # (a) true stable effect 0.05 across decades -> extrapolation passed == True
    # (b) finite-size artifact, effect = 0.4/log n fit only on decades 5-6
    #     -> extrapolation at decade 8 passed == False
    h1 = register("stable_effect", fit_decades=[5, 6, 7], predicted_effect=0.05,
                  ci_low=0.04, ci_high=0.06, out_dir="prereg")
    stable = score_extrapolation("stable_effect", target_decade=8, observed_effect=0.052)
    assert stable["seed"] == int(h1[:8], 16)
    assert stable["passed"] is True

    early_prediction = 0.4 / np.log(10**6.5)
    true_target = 0.4 / np.log(10**8.5)
    register("finite_artifact", fit_decades=[5, 6], predicted_effect=early_prediction,
             ci_low=early_prediction - 0.002, ci_high=early_prediction + 0.002,
             out_dir="prereg")
    artifact = score_extrapolation("finite_artifact", target_decade=8, observed_effect=true_target)
    assert artifact["passed"] is False
